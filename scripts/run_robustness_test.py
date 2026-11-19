"""LLM 변형 질문 견고성 시험: 말투·오타·사투리가 달라져도 질문 해석(QueryPlan)이 같은지 잰다.

기준 질문  data/meta/robustness_seeds.jsonl (364문항)
  템플릿으로 만든 질문과 기준 슬롯(reference). 템플릿이 정한 슬롯(intended_slots)은 의도값이고, 나머지 슬롯은
  2026-11-16 수정 전 해석기의 기준 문장 해석값이다(지명 속 '오일장' 버그로 틀렸던 1문항은 note 를 달고 바로잡음).
변형 질문  data/meta/robustness_variants_gemma4.jsonl (개발용), robustness_variants_qwen3.jsonl (평가용)
  데스크탑 GPU 의 Ollama 로컬 LLM 이 기준 질문마다 7가지 말투(casual·typo·jeju·elderly·short·reorder·mixed)로
  다시 쓴 질문(scripts/generate_question_variants.py). 1차로 개발용 세트의 오답만 보고 해석기를 고친 뒤 평가용 세트를
  처음 채점했다(공식 평가 점수). 그 뒤 평가용 오답에서 드러난 실제 한국어 표현을 2차로 고쳤으므로 현재 코드의 평가용
  점수는 사후 점수다. 수정 전·1차 점수는 data/meta/robustness_baseline.json 에 있다.
채점  12개 슬롯(의도·주제·지역·지역 수·시작/끝 시각·날씨·요일·이용자·대기시간·판정 보류·경계 변수)이 기준과
      모두 같으면 일치. 후속 질문은 이전 질문 해석과 병합한 뒤 비교한다.
--pipeline N  같은 질문 전체를 로컬 엔진 답변 경로(강제 오프라인, 캐시 미사용)로 끝까지 돌려 실행 오류,
              세 구획 누락, 수치 감사 불일치를 센다(프로세스 N개).
python scripts/run_robustness_test.py [--pipeline 8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.agent.intent_parser import merge_dialogue_context, parse_question  # noqa: E402
from src.config import META_DIR, PROCESSED_DIR  # noqa: E402
from src.tools.gazetteer_index import load_place_index  # noqa: E402

SEEDS_PATH = META_DIR / "robustness_seeds.jsonl"
BASELINE_PATH = META_DIR / "robustness_baseline.json"
VARIANT_SETS = [("gemma4:12b", "개발용", META_DIR / "robustness_variants_gemma4.jsonl"),
                ("qwen3:14b", "평가용", META_DIR / "robustness_variants_qwen3.jsonl")]
SLOTS = ["intent", "topic", "emds", "area_count", "hour_from", "hour_to", "weather", "day_type", "user_group",
         "wait_minutes", "abstain", "variable"]
STYLES = ["casual", "typo", "jeju", "elderly", "short", "reorder", "mixed"]
REPORT_JSON = PROCESSED_DIR / "robustness_report.json"
REPORT_MD = PROCESSED_DIR / "robustness_report.md"
ENGINES: dict = {}
SECTIONS: tuple = ()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_emds(labels) -> list[str]:
    """해석 결과의 지역 이름(별칭·읍면동)을 읍면동 집합으로 바꾼다."""
    index = load_place_index()
    found = set()
    for label in labels:
        if label in index.alias_rows:
            found.update(index.alias_rows[label]["targets"])
        elif label in index.regions:
            found.add(label)
        else:
            match = index.match_token(label)
            found.update(match.targets if match else [f"?{label}"])
    return sorted(found)


def read_plan(text: str, workspace: str, context: str | None) -> dict:
    plan = parse_question(text, workspace)
    if context:
        plan = merge_dialogue_context(plan, parse_question(context, workspace))
    return {"intent": plan.intent, "topic": plan.topic, "emds": collect_emds(plan.areas), "area_count": len(plan.areas),
            "hour_from": plan.hour_from, "hour_to": plan.hour_to, "weather": plan.weather, "day_type": plan.day_type,
            "user_group": plan.user_group, "wait_minutes": plan.wait_minutes,
            "abstain": plan.needs_abstain is not None, "variable": plan.variable}


def find_mismatches(got: dict, reference: dict) -> list[str]:
    return [slot for slot in SLOTS if got.get(slot) != reference[slot]]


def format_rate(good: int, total: int) -> str:
    return f"{good}/{total} ({good / total:.1%})" if total else "-"


def score_items(items: list[dict]) -> tuple[dict, list[dict]]:
    """(질문, 기준) 목록을 채점해 요약과 불일치 행을 돌려준다."""
    misses, by_style, by_family, slot_wrong = [], defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0]), Counter()
    errors = 0
    for item in items:
        try:
            got = read_plan(item["text"], item["workspace"], item.get("context"))
            wrong = find_mismatches(got, item["reference"])
        except Exception as exc:  # noqa: BLE001 - 해석기 예외도 집계 대상
            got, wrong = {"error": f"{type(exc).__name__}: {exc}"}, ["parser_error"]
            errors += 1
        for bucket in (by_style[item["style"]], by_family[item["family"]]):
            bucket[0] += not wrong
            bucket[1] += 1
        slot_wrong.update(wrong)
        if wrong:
            misses.append({**{k: item[k] for k in ("seed_id", "family", "style", "text", "context")},
                           "wrong": {slot: [got.get(slot), item["reference"].get(slot)] for slot in wrong}})
    total = len(items)
    summary = {"items": total, "full_match": format_rate(total - len(misses), total),
               "full_match_ratio": round((total - len(misses)) / total, 4) if total else None,
               "parser_errors": errors,
               "slot_accuracy": {slot: format_rate(total - slot_wrong.get(slot, 0), total) for slot in SLOTS},
               "by_style": {style: format_rate(*by_style[style]) for style in STYLES if style in by_style},
               "by_family": {family: format_rate(*pair) for family, pair in sorted(by_family.items())}}
    return summary, misses


def build_items(seeds: dict, records: list[dict], model: str) -> list[dict]:
    items = []
    for record in records:
        seed = seeds[record["seed_id"]]
        for style, text in record.get("variants", {}).items():
            items.append({"seed_id": seed["id"], "family": seed["family"], "workspace": seed["workspace"],
                          "context": seed.get("context"), "reference": seed["reference"], "style": style,
                          "text": text, "model": model})
    return items


def prepare_worker() -> None:
    global SECTIONS
    from src.agent.orchestrator import HybridOrchestrator
    from src.agent.prompts import SECTION_FACTS, SECTION_LIMIT, SECTION_REVERSAL
    from src.agent.response_cache import ResponseCache
    from src.tools.db_connector import ensure_database

    ensure_database()
    SECTIONS = (SECTION_FACTS, SECTION_REVERSAL, SECTION_LIMIT)
    for workspace in ("life", "lawyer"):
        ENGINES[workspace] = HybridOrchestrator(workspace, force_offline=True, cache=ResponseCache(enabled=False))


def answer_item(item: dict) -> dict:
    engine = ENGINES[item["workspace"]]
    engine.reset()
    started = time.perf_counter()
    try:
        if item.get("context"):
            engine.answer(item["context"])
        bundle = engine.answer(item["text"])
        narrative = bundle.get("narrative", "") or ""
        audit = bundle.get("number_audit") or {}
        row = {"ok": True, "verdict": bundle.get("verdict"), "audit_ratio": audit.get("ratio"),
               "missing_sections": [s for s in SECTIONS if s not in narrative],
               "ms": round((time.perf_counter() - started) * 1000, 1), "text": item["text"]}
        if audit.get("ratio") not in (None, 1.0):
            row.update(ungrounded=audit.get("ungrounded"), context=item.get("context"), workspace=item["workspace"])
        return row
    except Exception as exc:  # noqa: BLE001 - 실행 오류 자체가 측정 대상
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-800:],
                "ms": round((time.perf_counter() - started) * 1000, 1), "text": item["text"]}


def run_pipeline(items: list[dict], workers: int) -> dict:
    unique, seen = [], set()
    for item in items:
        key = (item["workspace"], item.get("context"), item["text"])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    started = time.perf_counter()
    with Pool(workers, initializer=prepare_worker) as pool:
        rows = list(pool.imap_unordered(answer_item, unique, chunksize=8))
    errors = [r for r in rows if not r["ok"]]
    missing = [r for r in rows if r["ok"] and r["missing_sections"]]
    audit_low = [r for r in rows if r["ok"] and r.get("audit_ratio") not in (None, 1.0)]
    times = sorted(r["ms"] for r in rows if r["ok"])
    return {"questions": len(rows), "runtime_errors": len(errors), "missing_sections": len(missing),
            "audit_below_full": len(audit_low), "seconds": round(time.perf_counter() - started, 1),
            "ms_median": times[len(times) // 2] if times else None,
            "ms_p95": times[max(0, int(len(times) * 0.95) - 1)] if times else None, "ms_max": times[-1] if times else None,
            "verdicts": dict(Counter(r.get("verdict") for r in rows if r["ok"])),
            "error_samples": [{"text": r["text"], "error": r["error"], "trace": r["trace"]} for r in errors[:5]],
            "missing_samples": [r["text"] for r in missing[:5]],
            "audit_samples": [{k: r.get(k) for k in ("workspace", "context", "text", "audit_ratio", "ungrounded")}
                              for r in audit_low[:5]]}


def describe_history(report: dict, name: str) -> tuple[str, str]:
    """기준값 파일에 기록된 (수정 전, 1차 수정) 일치 문자열."""
    baseline = report.get("baseline", {})
    before = baseline.get("sets", {}).get(name, {}).get("full_match", "-")
    first = baseline.get("round1", {}).get("sets", {}).get(name, {}).get("full_match", "-")
    return before, first


def render_markdown(report: dict) -> str:
    lines = ["# LLM 변형 질문 견고성 시험 결과", "", f"- 실행: {report['generated_at']}",
             f"- 기준 질문 일치(기준 문장 자체): {report['canonical']['full_match']}", ""]
    lines += ["| 세트 | 생성 모델 | 변형 질문 | 수정 전 | 1차 수정(평가용은 공식 점수) | 현재 코드 |",
              "|---|---|---|---|---|---|"]
    for name, role, _ in VARIANT_SETS:
        current = report["sets"].get(name)
        if current:
            before, first = describe_history(report, name)
            lines.append(f"| {role} | {name} | {current['items']} | {before} | {first} | {current['full_match']} |")
    lines += ["", "평가용 세트의 현재 점수는 1차 채점 뒤 그 오답을 보고 2차 수정을 한 사후 점수다(일반화 성능으로 주장하지 않음)."]
    for name, role, _ in VARIANT_SETS:
        current = report["sets"].get(name)
        if not current:
            continue
        lines += ["", f"## {role}({name}) 말투별·슬롯별", "",
                  "| 말투 | 일치 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in current["by_style"].items()]
        lines += ["", "| 슬롯 | 정확도 |", "|---|---|"] + [f"| {k} | {v} |" for k, v in current["slot_accuracy"].items()]
    if report.get("pipeline"):
        pipe = report["pipeline"]
        lines += ["", "## 로컬 엔진 전체 답변 경로", "",
                  f"- 질문 {pipe['questions']}개(중복 제거), 실행 오류 {pipe['runtime_errors']}건, 세 구획 누락 "
                  f"{pipe['missing_sections']}건, 수치 감사 불일치 {pipe['audit_below_full']}건",
                  f"- 응답 시간 중앙값 {pipe['ms_median']}ms, p95 {pipe['ms_p95']}ms, 최대 {pipe['ms_max']}ms"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM 변형 질문 견고성 시험")
    parser.add_argument("--pipeline", type=int, default=0, help="로컬 엔진 전체 답변 경로 실행 프로세스 수(0=생략)")
    args = parser.parse_args()
    seeds = {seed["id"]: seed for seed in load_jsonl(SEEDS_PATH)}
    canonical_items = [{"seed_id": s["id"], "family": s["family"], "workspace": s["workspace"],
                        "context": s.get("context"), "reference": s["reference"], "style": "canonical",
                        "text": s["text"]} for s in seeds.values()]
    canonical, canonical_misses = score_items(canonical_items)
    report = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "canonical": canonical,
              "canonical_misses": canonical_misses, "sets": {}, "misses": {}}
    all_items = list(canonical_items)
    for name, role, path in VARIANT_SETS:
        if not path.exists():
            print(f"[건너뜀] {path.name} 없음")
            continue
        items = build_items(seeds, load_jsonl(path), name)
        summary, misses = score_items(items)
        report["sets"][name] = {"role": role, **summary}
        report["misses"][name] = misses
        all_items += items
    if BASELINE_PATH.exists():
        report["baseline"] = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    if args.pipeline:
        report["pipeline"] = run_pipeline(all_items, args.pipeline)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    REPORT_MD.write_text(render_markdown(report), encoding="utf-8")
    print(f"기준 질문 일치: {canonical['full_match']}")
    for name, current in report["sets"].items():
        before, first = describe_history(report, name)
        print(f"{current['role']}({name}) 변형 {current['items']}개 일치: 수정 전 {before} -> 1차 수정 {first} "
              f"-> 현재 {current['full_match']}")
        print("  말투별:", ", ".join(f"{k} {v}" for k, v in current["by_style"].items()))
    if report.get("pipeline"):
        pipe = report["pipeline"]
        print(f"전체 답변 경로: 질문 {pipe['questions']}개, 실행 오류 {pipe['runtime_errors']}건, "
              f"세 구획 누락 {pipe['missing_sections']}건, 수치 감사 불일치 {pipe['audit_below_full']}건, "
              f"중앙값 {pipe['ms_median']}ms, p95 {pipe['ms_p95']}ms")
    print(f"저장: {REPORT_JSON.relative_to(ROOT)}, {REPORT_MD.relative_to(ROOT)}")
    failed = canonical["parser_errors"] or any(s["parser_errors"] for s in report["sets"].values())
    failed = failed or bool(report.get("pipeline", {}).get("runtime_errors"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
