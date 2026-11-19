"""실제 Claude 경로 40문항 시험(API Key 필요, 실제 요금 발생).

문항: 24문항 벤치마크 전부(정답 라벨로 채점) + 앱 시연 버튼 질문 중 벤치마크에 없는 6개 + 이전 실제 연결 시험 질문 4개
      + 5턴 연속 후속 질문 1벌 + 한영 혼용 질문 1개 = 40문항(앞 질문을 잇는 5턴은 한 대화로 차례로 묻는다).
같은 질문을 로컬 엔진(force_offline)으로도 답해 판정을 대조한다. 문항마다 아래 기준표 네 가지를 모두 지키면 통과:
  1) Claude 모드로 답함(폴백 아님)  2) 결론·세 구획·현장 확인 질문·해설을 모두 갖춤
  3) 해설 속 숫자가 모두 도구 결과에 있음(수치 감사 불일치 0)  4) 판정이 정답 라벨(벤치마크) 또는 로컬 엔진 판정(나머지)과 같음

python scripts/run_live_test.py              # 40문항(문항마다 수십 초, 실제 요금 발생)
python scripts/run_live_test.py --dry-run    # API 를 부르지 않고(키 없이) 같은 절차만 확인(모두 로컬 폴백)
결과: data/processed/live_report.json, live_report.md
"""
from __future__ import annotations

import argparse
import ast
import json
import statistics
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_benchmark import score_ours  # noqa: E402
from run_fuzz_stress_test import CHAT_CASES, MULTI_TURN, check_bundle  # noqa: E402
from src.agent.orchestrator import HybridOrchestrator  # noqa: E402
from src.agent.response_cache import ResponseCache  # noqa: E402
from src.config import BENCHMARK_ITEMS_PATH, PROCESSED_DIR, read_api_key  # noqa: E402
from src.tools.analysis_cubes import warm_cubes  # noqa: E402

REPORT = PROCESSED_DIR / "live_report.json"
# 이전 실제 연결 시험(run_fuzz_stress_test.py --live 5, 11-16~11-18) 질문. 벤치마크와 같은 질문(L04 = T13)은 뺀다.
PREVIOUS_LIVE = ["노형똥 퇴근 버스 어때?", "우도면 새벽 3시 비 오는 장날 어르신 버스 붐벼?", "'; DROP TABLE regions; --",
                 "전기차 충전 대기시간이 평균 20분이라는 보도, 믿어도 돼?", "작년 겨울 노형동 밤 버스 붐볐어?"]
MIXED_LANGUAGE = [text for kind, text in CHAT_CASES if kind == "english" and "노형동" in text]  # '노형동 vs 연동 which is better at night?'


def load_scenarios() -> dict:
    """앱 시연 버튼 질문(src/app.py 의 SCENARIOS)을 앱을 띄우지 않고 읽는다."""
    tree = ast.parse((ROOT / "src" / "app.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(target, "id", "") == "SCENARIOS" for target in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("src/app.py 에서 SCENARIOS 를 찾지 못함")


def build_questions() -> list[dict]:
    items = json.loads(BENCHMARK_ITEMS_PATH.read_text(encoding="utf-8"))["items"]
    known = {item["question"] for item in items}
    questions = [{"id": f"B-{item['id']}", "group": "benchmark", "workspace": item["workspace"], "question": item["question"],
                  "item": item} for item in items]
    extra = [(workspace, text) for workspace, rows in load_scenarios().items() for _, text in rows if text not in known]
    questions += [{"id": f"S{number:02d}", "group": "scenario", "workspace": workspace, "question": text}
                  for number, (workspace, text) in enumerate(extra, start=1)]
    previous = [text for text in PREVIOUS_LIVE if text not in known]
    questions += [{"id": f"P{number:02d}", "group": "previous_live", "workspace": "life", "question": text}
                  for number, text in enumerate(previous, start=1)]
    questions += [{"id": f"M{number:02d}", "group": "multi_turn", "workspace": "life", "question": text, "dialogue": "M"}
                  for number, text in enumerate(MULTI_TURN, start=1)]
    questions += [{"id": f"E{number:02d}", "group": "mixed_language", "workspace": "life", "question": text}
                  for number, text in enumerate(MIXED_LANGUAGE, start=1)]
    return questions


def count_usage(bundle: dict, before: dict) -> dict:
    """한 대화에서 쌓인 토큰 수에서 이번 질문 몫만 뺀다(대화마다 Claude 연결 하나를 쓰므로 누적값이 온다)."""
    usage = ((bundle.get("claude") or {}).get("usage")) or {}
    return {key: int(usage.get(key, 0)) - int(before.get(key, 0)) for key in ("input_tokens", "output_tokens", "requests")}


def score_row(entry: dict, bundle: dict, local: dict, seconds: float, usage: dict) -> dict:
    audit = bundle.get("number_audit") or {}
    item = entry.get("item")
    gold = None
    if item:
        passed, detail = score_ours(item, bundle)
        local_passed, local_detail = score_ours(item, local)
        gold = {"expected": item["expected"]["verdicts"], "pass": passed, "detail": detail,
                "local_pass": local_passed, "local_detail": local_detail}
    sections = check_bundle(bundle)
    checks = {
        "claude_mode": bundle.get("mode") == "claude",
        "sections": sections is None,
        "numbers": bool(audit) and not audit.get("ungrounded"),
        "verdict": gold["pass"] if gold else bundle.get("verdict") == local.get("verdict"),
    }
    claude = bundle.get("claude") or {}
    return {
        "id": entry["id"], "group": entry["group"], "workspace": entry["workspace"], "question": entry["question"],
        "passed": all(checks.values()), "checks": checks, "mode": bundle.get("mode"), "mode_reason": bundle.get("mode_reason"),
        "verdict": bundle.get("verdict"), "local_verdict": local.get("verdict"), "gold": gold, "sections_problem": sections,
        "number_audit": audit, "local_number_audit": local.get("number_audit"), "seconds": round(seconds, 1),
        "claude": {"rounds": claude.get("rounds"), "seconds": claude.get("seconds"), "model": claude.get("model"), **usage,
                   "tool_calls": claude.get("tool_calls")},
        "tool_sources": {run["name"]: run.get("source") for run in bundle.get("tool_runs", [])},
        "headline": bundle.get("headline"), "field_question": bundle.get("field_question"),
        "narrative": bundle.get("narrative"),
    }


def ask_question(entry: dict, live: HybridOrchestrator, local: HybridOrchestrator, before: dict) -> tuple[dict, dict]:
    local_bundle = local.answer(entry["question"])
    started = time.perf_counter()
    bundle = live.answer(entry["question"])
    seconds = time.perf_counter() - started
    usage = count_usage(bundle, before)
    after = ((bundle.get("claude") or {}).get("usage")) or before
    return score_row(entry, bundle, local_bundle, seconds, usage), after


def summarize_rows(rows: list[dict]) -> dict:
    answered = [row for row in rows if not row.get("error")]
    seconds = [row["seconds"] for row in answered]
    gold_rows = [row for row in answered if row.get("gold")]
    audits = [row["number_audit"] for row in answered if row.get("number_audit")]
    modes: dict = {}
    for row in answered:
        modes[row["mode"]] = modes.get(row["mode"], 0) + 1
    quantiles = statistics.quantiles(seconds, n=20) if len(seconds) >= 2 else seconds
    return {
        "questions": len(rows), "passed": sum(1 for row in rows if row.get("passed")),
        "runtime_errors": sum(1 for row in rows if row.get("error")), "modes": modes,
        "fallback_reasons": [row["mode_reason"] for row in answered if row["mode"] != "claude"],
        "checks": {name: sum(1 for row in answered if row["checks"][name]) for name in ("claude_mode", "sections", "numbers", "verdict")},
        "benchmark_claude": f"{sum(1 for row in gold_rows if row['gold']['pass'])} / {len(gold_rows)}",
        "benchmark_local": f"{sum(1 for row in gold_rows if row['gold']['local_pass'])} / {len(gold_rows)}",
        "verdict_same_as_local": f"{sum(1 for row in answered if row['verdict'] == row['local_verdict'])} / {len(answered)}",
        "numbers": f"{sum(a.get('grounded', 0) for a in audits)} / {sum(a.get('checked', 0) for a in audits)}",
        "seconds_median": round(statistics.median(seconds), 1) if seconds else None,
        "seconds_p95": round(quantiles[-1], 1) if seconds else None,
        "seconds_min": min(seconds) if seconds else None, "seconds_max": max(seconds) if seconds else None,
        "input_tokens": sum(row["claude"].get("input_tokens", 0) for row in answered),
        "output_tokens": sum(row["claude"].get("output_tokens", 0) for row in answered),
        "requests": sum(row["claude"].get("requests", 0) for row in answered),
    }


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# 실제 Claude 경로 40문항 시험 결과", "",
        f"- 실행: {report['generated_at']} · 모델 {report['model']} · 문항마다 새 연결(5턴 후속 질문은 한 대화), 캐시 미사용",
        f"- 기준표 통과: {summary['passed']} / {summary['questions']} (실행 오류 {summary['runtime_errors']}건)",
        f"- Claude 모드로 답함 {summary['checks']['claude_mode']}, 결론·세 구획·현장 확인 질문 {summary['checks']['sections']}, "
        f"해설 숫자 모두 도구 근거 {summary['checks']['numbers']}, 판정 일치 {summary['checks']['verdict']}",
        f"- 벤치마크 24문항 정답: Claude 경로 {summary['benchmark_claude']} · 같은 실행의 로컬 엔진 {summary['benchmark_local']}",
        f"- 판정이 로컬 엔진과 같음: {summary['verdict_same_as_local']} · 해설 숫자 도구 근거 일치 {summary['numbers']}",
        f"- 답변 시간: 중앙값 {summary['seconds_median']}초, p95 {summary['seconds_p95']}초, 범위 {summary['seconds_min']}~{summary['seconds_max']}초",
        f"- 토큰: 입력 {summary['input_tokens']:,} · 출력 {summary['output_tokens']:,} (요청 {summary['requests']}회)",
        f"- 폴백: {summary['fallback_reasons'] or '없음'}", "",
        "| 문항 | 묶음 | 질문 | 모드 | 판정 | 로컬 판정 | 정답·대조 | 숫자 | 초 | 통과 |", "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        if row.get("error"):
            lines.append(f"| {row['id']} | {row['group']} | {row['question'][:30]} | 오류 | - | - | {row['error'][:40]} | - | - | 아니오 |")
            continue
        audit = row.get("number_audit") or {}
        mark = (("정답" if row["gold"]["pass"] else "오답") if row["gold"] else ("같음" if row["verdict"] == row["local_verdict"] else "다름"))
        lines.append(f"| {row['id']} | {row['group']} | {row['question'][:30].replace('|', '/')} | {row['mode']} | {row['verdict']} | "
                     f"{row['local_verdict']} | {mark} | {audit.get('grounded', '-')}/{audit.get('checked', '-')} | {row['seconds']} | "
                     f"{'예' if row['passed'] else '아니오'} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="API 를 부르지 않고 같은 절차만 확인(모두 로컬 폴백)")
    args = parser.parse_args()
    key = "" if args.dry_run else read_api_key()
    if not key and not args.dry_run:
        print("API Key 가 없습니다(.env 의 ANTHROPIC_API_KEY). --dry-run 으로 절차만 확인할 수 있습니다.")
        return 2
    questions = build_questions()
    if len(questions) != 40:
        raise RuntimeError(f"문항 수가 40이 아님: {len(questions)}")
    warm_cubes()
    rows = []
    dialogues: dict = {}
    for number, entry in enumerate(questions, start=1):
        dialogue = entry.get("dialogue")
        if dialogue:
            live, local, before = dialogues.setdefault(dialogue, [
                HybridOrchestrator(entry["workspace"], api_key=key, cache=ResponseCache(enabled=False)),
                HybridOrchestrator(entry["workspace"], force_offline=True, cache=ResponseCache(enabled=False)), {}])
        else:
            live = HybridOrchestrator(entry["workspace"], api_key=key, cache=ResponseCache(enabled=False))
            local = HybridOrchestrator(entry["workspace"], force_offline=True, cache=ResponseCache(enabled=False))
            before = {}
        try:
            row, after = ask_question(entry, live, local, before)
            if dialogue:
                dialogues[dialogue][2] = after
        except Exception as exc:  # 한 문항의 예외가 시험 전체를 멈추지 않게 기록만 한다
            row = {"id": entry["id"], "group": entry["group"], "workspace": entry["workspace"], "question": entry["question"],
                   "passed": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=4)}
        rows.append(row)
        audit = row.get("number_audit") or {}
        print(f"[{number:02d}/40] {row['id']} {row.get('mode')} {row.get('verdict')} (로컬 {row.get('local_verdict')}) "
              f"숫자 {audit.get('grounded', '-')}/{audit.get('checked', '-')} {row.get('seconds')}s 통과 {row['passed']}", flush=True)
    report = {"generated_by": "scripts/run_live_test.py", "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "dry_run": args.dry_run, "model": next((row["claude"]["model"] for row in rows if (row.get("claude") or {}).get("model")),
                                                     None), "summary": summarize_rows(rows), "rows": rows}
    summary = report["summary"]
    print(f"기준표 통과 {summary['passed']} / {summary['questions']}, 벤치마크 Claude {summary['benchmark_claude']} · 로컬 "
          f"{summary['benchmark_local']}, 숫자 {summary['numbers']}, 모드 {summary['modes']}")
    if args.dry_run:
        render_markdown(report)  # 보고서 형식까지 확인만 하고 제출 보고서는 쓰지 않는다
        print("(dry-run: 보고서 파일을 쓰지 않음)")
        return 0
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    REPORT.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(f"저장: {REPORT.relative_to(ROOT)}, {REPORT.with_suffix('.md').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
