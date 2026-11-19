"""대규모 조합 행동 시험: 문장틀 x 끼울 값 x 입력 변형으로 만든 질문을 로컬 해석기로 채점한다.

--suite scale(기본, 1차 2,550만 문항): 정의 scripts/scale_suite.py, LLM 문장틀 data/meta/scale_frames_*.jsonl.
--suite diverse(2차 7천만 문항 이상): 정의 scripts/diverse_suite.py(질문 유형 57종·말투 36종·상황·확장 어휘·입력 변형 21종),
  LLM 문장틀 data/meta/diverse_frames_gemma4.jsonl(개발용)·diverse_frames_qwen3.jsonl(평가용), 보고서 diverse_report.*.
--suite mega(3차 1억 문항): 정의 scripts/mega_suite.py(다양성 시험 + 새 어휘 data/meta/mega_vocab.json·새 입력 변형 10종·
  새 앞 질문 표현 8가지), 보고서 mega_report.*. 평가용에만 넣은 새 요소가 든 문항(fresh)을 따로 집계한다.
--suite restyle(4차 6천만 문항): 정의 scripts/restyle_suite.py(데스크탑 GPU 로컬 LLM 이 새 말투 24종·새 상황으로 새로 쓴 문장틀
  data/meta/restyle_frames_gemma4.jsonl(개발용)·restyle_frames_qwen3.jsonl(평가용) x 1억 시험 값 x 새 입력 변형 10종),
  보고서 restyle_report.*. 평가용 전용 새 말투·상황·변형이 든 문항(new_any)을 따로 집계한다.
- 개발용(dev): 기본 틀 + gemma4 틀에 평가용 전용 요소를 뺀 조합만 쓴다. 오답을 보고 해석기를 고치는 세트.
- 평가용(heldout): qwen3 틀 + 모든 요소. 해석기 수정 전에 한 번, 개발용 오답만 보고 고친 뒤 한 번 채점한다
  (그 사이 평가용 오답 문장은 열어 보지 않는다: --hide-samples heldout).
문항 번호 -> 문장은 결정적이다(시드 고정). 결과는 작업 묶음마다 이어 쓰므로 끊겨도 같은 명령으로 이어서 돈다.
채점 기준은 LLM 변형 견고성 시험과 같은 12개 슬롯 완전 일치(scripts/run_robustness_test.py read_plan).
--pipeline N  표본 문항을 로컬 엔진 전체 답변 경로(강제 오프라인, 캐시 미사용)로 끝까지 돌린다(프로세스 N개).
여러 컴퓨터로 나눠 돌리기: 같은 명령에 --parts N --part K[,K2..] 를 붙여 컴퓨터마다 다른 몫을 돌리고(작업 묶음 번호 % N),
  결과 파일(*_pXXofNN.jsonl/.hashes, 전체 답변 경로는 --pipeline-part K --pipeline-parts M)을 한 폴더에 모은 뒤
  --merge-only 로 합친다. 합칠 때 계획의 작업 묶음이 하나라도 빠지면 멈춘다.
python scripts/run_scale_test.py --tag final --workers 15 [--report] [--pipeline 15 --pipeline-sample 300000]
  (1차 기본 세트당 12,750,000문항 = 두 세트 25,506,557문항. 16스레드 PC 에서 채점 약 10분, 표본 30만 답변 경로 약 40분)
python scripts/run_scale_test.py --suite diverse --per-set 35500000 --tag d1 --parts 20 --part 0,1,..,10 --workers 15
빠른 확인: python scripts/run_scale_test.py --tag smoke --per-set 20000 --workers 4
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
import traceback
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for folder in (ROOT, ROOT / "scripts"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
import scale_suite as suite  # noqa: E402  (--suite diverse 이면 load_suite 가 바꾼다)
from run_robustness_test import answer_item, find_mismatches, format_rate, prepare_worker, read_plan  # noqa: E402
from src.config import META_DIR, PROCESSED_DIR  # noqa: E402

SUITE_FILES = {
    "scale": {"module": "scale_suite", "report": "scale_report", "baseline": META_DIR / "scale_baseline.json",
              "title": "대규모 조합 행동 시험 결과"},
    "diverse": {"module": "diverse_suite", "report": "diverse_report", "baseline": META_DIR / "diverse_baseline.json",
                "title": "다양성 대규모 조합 행동 시험 결과"},
    "mega": {"module": "mega_suite", "report": "mega_report", "baseline": META_DIR / "mega_baseline.json",
             "title": "1억 문항 확장 시험 결과"},
    "restyle": {"module": "restyle_suite", "report": "restyle_report", "baseline": META_DIR / "restyle_baseline.json",
                "title": "6천만 문항 새 말투 시험 결과"},
}
SET_ROLES = {"dev": "개발용", "heldout": "평가용"}
NOVEL_PARTS = {"value": "새 값(어휘)", "form": "지명 감싸기", "context": "앞 질문 표현", "kind": "입력 변형",
               "style": "말투", "situation": "상황 요소", "fresh": "1억 시험 새 요소(전체)",
               "fresh_value": "1억 시험 새 값", "fresh_form": "1억 시험 새 지명 감싸기",
               "fresh_context": "1억 시험 새 앞 질문 표현", "fresh_kind": "1억 시험 새 입력 변형",
               "new_style": "6천만 시험 평가용 전용 새 말투", "new_situation": "6천만 시험 평가용 전용 새 상황",
               "new_kind": "6천만 시험 평가용 전용 새 입력 변형", "new_any": "6천만 시험 평가용 전용 새 요소(전체)"}
STATE: dict = {}


def load_suite(name: str) -> None:
    """시험 정의 모듈을 고른다(작업 프로세스도 초기화할 때 다시 부른다)."""
    global suite
    suite = importlib.import_module(SUITE_FILES[name]["module"])


def count_cases(dims: list, set_name: str) -> int:
    return suite.count_space(dims, set_name) if getattr(suite, "SET_AWARE", False) else suite.count_space(dims)


def make_case(template: dict, frame: dict, dims: list, index: int, set_name: str) -> dict:
    if getattr(suite, "SET_AWARE", False):
        return suite.build_case(template, frame, dims, index, set_name)
    return suite.build_case(template, frame, dims, index)


def prepare_scale(set_names: list[str], suite_name: str = "scale") -> None:
    """작업 프로세스 준비: 값 묶음·문장틀을 파일에서 다시 만든다(결정적이라 주 프로세스와 같다)."""
    load_suite(suite_name)
    STATE["pools"] = suite.build_pools()
    STATE["frames"] = {frame["id"]: frame for frame in suite.load_frames()}
    STATE["templates"] = {name: suite.load_templates(name)[0] for name in set_names}
    STATE["dims"] = {}


def load_dims(set_name: str, template: dict, frame: dict) -> list:
    flags = tuple(suite.check_attached(template["text"], name) for name in frame["slots"] if name not in ("P1", "P2"))
    key = (set_name, frame["id"], flags)
    if key not in STATE["dims"]:
        STATE["dims"][key] = suite.plan_dimensions(template, frame, STATE["pools"], set_name)
    return STATE["dims"][key]


def add_count(table: dict, key: str, failed: bool) -> None:
    pair = table.setdefault(key, [0, 0])
    pair[0] += failed
    pair[1] += 1


def run_task(task: tuple) -> tuple[dict, bytes]:
    """작업 묶음 하나(문장틀 하나의 연속 구간)를 채점한다."""
    set_name, number, start, stop, space, stride, offset, part = task
    template = STATE["templates"][set_name][number]
    frame = STATE["frames"][template["frame"]]
    dims = load_dims(set_name, template, frame)
    record = {"set": set_name, "part": part, "task": f"{template['key']}:{start}", "template": template["key"],
              "frame": frame["id"], "family": frame["family"], "style": template["style"], "source": template["source"],
              "situation": template.get("situation"), "n": 0, "fail": 0, "errors": 0, "kinds": {}, "slots": {},
              "values": {}, "novel": {}, "parts": {}, "clusters": {}, "ms": [], "error_samples": []}
    digests = bytearray()
    for step in range(start, stop):
        case = make_case(template, frame, dims, (offset + stride * step) % space, set_name)
        began = time.perf_counter()
        try:
            got = read_plan(case["text"], template["workspace"], case["context"])
            wrong = find_mismatches(got, case["expected"])
        except Exception as exc:  # noqa: BLE001 - 해석기 예외도 집계 대상
            got, wrong = {"error": f"{type(exc).__name__}: {exc}"}, ["parser_error"]
            record["errors"] += 1
            if len(record["error_samples"]) < 3:
                record["error_samples"].append({"text": case["text"], "trace": traceback.format_exc()[-600:]})
        elapsed = (time.perf_counter() - began) * 1000
        digests += hashlib.blake2b(f"{template['workspace']}\x1f{case['context']}\x1f{case['text']}".encode("utf-8"),
                                   digest_size=8).digest()
        failed = bool(wrong)
        record["n"] += 1
        record["fail"] += failed
        add_count(record["kinds"], case["kind"], failed)
        add_count(record["novel"], "unseen" if case["unseen"] else "seen", failed)
        for name in case.get("novel_parts", ()):
            add_count(record["parts"], name, failed)
        for key in case["values"]:
            add_count(record["values"], key, failed)
        for slot in wrong:
            record["slots"][slot] = record["slots"].get(slot, 0) + 1
        if failed:
            entry = record["clusters"].setdefault(f"{frame['family']}|{','.join(wrong)}", [0, []])
            entry[0] += 1
            if len(entry[1]) < 3:
                entry[1].append({"text": case["text"], "context": case["context"], "kind": case["kind"],
                                 "style": template["style"], "template": template["text"],
                                 "got": {slot: got.get(slot) for slot in wrong},
                                 "expected": {slot: case["expected"].get(slot) for slot in wrong}})
        if step % 25 == 0:
            record["ms"].append(round(elapsed, 3))
    return record, bytes(digests)


def compute_quota(spaces: list[int], target: int) -> int:
    """문장틀마다 min(조합 공간, 할당량)개를 뽑아 합이 target 이상이 되는 가장 작은 할당량."""
    if sum(spaces) <= target:
        return max(spaces)
    low, high = 1, max(spaces)
    while low < high:
        middle = (low + high) // 2
        if sum(min(space, middle) for space in spaces) >= target:
            high = middle
        else:
            low = middle + 1
    return low


def plan_tasks(set_name: str, per_set: int, chunk: int, limit: int, parts: int = 1) -> dict:
    """세트의 작업 묶음 계획. 작업 묶음마다 몫 번호(세트 안 순번 % parts)를 붙인다."""
    pools = suite.build_pools()
    frames = {frame["id"]: frame for frame in suite.load_frames()}
    templates, dropped = suite.load_templates(set_name)
    picked = range(len(templates))
    if limit and limit < len(templates):
        picked = sorted({round(i * (len(templates) - 1) / (limit - 1)) for i in range(limit)}) if limit > 1 else [0]
    spaces = {}
    for number in picked:
        template = templates[number]
        spaces[number] = count_cases(suite.plan_dimensions(template, frames[template["frame"]], pools, set_name), set_name)
    quota = compute_quota(list(spaces.values()), per_set)
    tasks, loads = [], [0] * parts
    for number, space in spaces.items():
        count = min(space, quota)
        stride, offset = suite.pick_stride(space), suite.compute_hash(templates[number]["key"]) % space
        for start in range(0, count, chunk):
            part = min(range(parts), key=loads.__getitem__)  # 문항 수가 가장 적은 몫에 넣어 몫 크기를 고르게 한다
            loads[part] += min(count, start + chunk) - start
            tasks.append((set_name, number, start, min(count, start + chunk), space, stride, offset, part))
    return {"templates": len(spaces), "templates_available": len(templates), "dropped": dropped, "quota": quota,
            "questions": sum(min(space, quota) for space in spaces.values()), "tasks": tasks,
            "by_source": {source: sum(1 for n in spaces if templates[n]["source"] == source)
                          for source in suite.SET_SOURCES[set_name]}}


def count_unique(paths: list[Path]) -> int:
    import numpy as np
    chunks = [np.fromfile(path, dtype=np.uint64) for path in paths if path.exists()]
    return int(np.unique(np.concatenate(chunks)).size) if chunks else 0


def summarize_rates(table: dict) -> dict:
    return {key: format_rate(n - fail, n) for key, (fail, n) in sorted(table.items())}


def merge_shard(paths: list[Path], hide: bool) -> dict:
    """작업 묶음 기록(여러 몫 파일 가능)을 합쳐 세트 요약을 만든다. 같은 작업 묶음은 한 번만 센다."""
    agg = {"n": 0, "fail": 0, "errors": 0, "kinds": {}, "slots": {}, "values": {}, "novel": {}, "parts": {},
           "clusters": {}, "family": {}, "style": {}, "source": {}, "family_kind": {}, "templates": {}, "situation": {},
           "ms": [], "error_samples": [], "tasks": set()}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["task"] in agg["tasks"]:
                continue
            agg["tasks"].add(record["task"])
            for key in ("n", "fail", "errors"):
                agg[key] += record[key]
            for table in ("kinds", "values", "novel", "parts"):
                for key, (fail, n) in record.get(table, {}).items():
                    pair = agg[table].setdefault(key, [0, 0])
                    pair[0] += fail
                    pair[1] += n
            for key, count in record["slots"].items():
                agg["slots"][key] = agg["slots"].get(key, 0) + count
            keys = [("family", record["family"]), ("style", record["style"]), ("source", record["source"]),
                    ("templates", record["template"])]
            keys += [("situation", f"{name}:{value}") for name, value in (record.get("situation") or {}).items()]
            for table, key in keys:
                pair = agg[table].setdefault(key, [0, 0])
                pair[0] += record["fail"]
                pair[1] += record["n"]
            for kind, (fail, n) in record["kinds"].items():
                pair = agg["family_kind"].setdefault(f"{record['family']}|{kind}", [0, 0])
                pair[0] += fail
                pair[1] += n
            for signature, (count, samples) in record["clusters"].items():
                entry = agg["clusters"].setdefault(signature, [0, []])
                entry[0] += count
                entry[1].extend(samples[: max(0, 8 - len(entry[1]))])
            agg["ms"] += record["ms"]
            agg["error_samples"] += record["error_samples"][: max(0, 5 - len(agg["error_samples"]))]
    n, fail = agg["n"], agg["fail"]
    strict = [agg["kinds"].get(k, [0, 0]) for k in suite.KINDS if k not in suite.STRESS_KINDS]
    stress = [agg["kinds"].get(k, [0, 0]) for k in suite.STRESS_KINDS]
    times = sorted(agg["ms"])
    clusters = sorted(agg["clusters"].items(), key=lambda item: -item[1][0])[:40]
    worst_values = sorted(((k, f, m) for k, (f, m) in agg["values"].items() if m >= 200), key=lambda x: -x[1] / x[2])[:40]
    worst_templates = sorted(((k, f, m) for k, (f, m) in agg["templates"].items() if m >= 50),
                             key=lambda x: -x[1] / x[2])[:40]
    return {
        "questions": n, "full_match": format_rate(n - fail, n), "full_match_ratio": round((n - fail) / n, 6) if n else None,
        "parser_errors": agg["errors"], "tasks_merged": len(agg["tasks"]),
        "strict": format_rate(sum(m - f for f, m in strict), sum(m for _, m in strict)),
        "stress": format_rate(sum(m - f for f, m in stress), sum(m for _, m in stress)),
        "by_kind": {k: format_rate(m - f, m) for k in suite.KINDS for f, m in [agg["kinds"].get(k, [0, 0])] if m},
        "by_family": summarize_rates(agg["family"]), "by_style": summarize_rates(agg["style"]),
        "by_source": summarize_rates(agg["source"]), "by_family_kind": summarize_rates(agg["family_kind"]),
        "by_situation": summarize_rates(agg["situation"]), "novelty": summarize_rates(agg["novel"]),
        "novel_parts": summarize_rates(agg["parts"]),
        "slot_accuracy": {slot: format_rate(n - agg["slots"].get(slot, 0), n) for slot in suite.SLOTS},
        "parse_ms_median": times[len(times) // 2] if times else None,
        "parse_ms_p95": times[max(0, int(len(times) * 0.95) - 1)] if times else None,
        "parse_ms_max": times[-1] if times else None,
        "top_clusters": [{"signature": sig, "count": count, **({} if hide else {"samples": samples})}
                         for sig, (count, samples) in clusters],
        "worst_values": [{"value": k, "fail": f, "n": m, "fail_ratio": round(f / m, 4)} for k, f, m in worst_values],
        "worst_templates": ([{"template": k, "fail": f, "n": m} for k, f, m in worst_templates] if not hide else []),
        "error_samples": agg["error_samples"],
    }


def render_markdown(report: dict) -> str:
    title = SUITE_FILES[report.get("suite", "scale")]["title"]
    lines = [f"# {title}", "", f"- 실행: {report['generated_at']} (tag: {report['tag']})",
             f"- 전체 문항 {report['total']['questions']:,}개, 서로 다른 문장 {report['total']['unique']:,}개", "",
             "| 세트 | 문장틀 | 문항 | 서로 다른 문장 | 완전 일치 | 엄격 변형 | 부하 변형 | 해석기 예외 |",
             "|---|---|---|---|---|---|---|---|"]
    for name, current in report["sets"].items():
        lines.append(f"| {SET_ROLES.get(name, name)} | {current['templates']:,} | {current['questions']:,} | "
                     f"{current['unique']:,} | {current['full_match']} | {current['strict']} | {current['stress']} | "
                     f"{current['parser_errors']} |")
    baseline = report.get("baseline")
    if baseline:
        lines += ["", "## 수정 전 대비", "", "| 세트 | 수정 전 | 현재 |", "|---|---|---|"]
        for name, current in report["sets"].items():
            before = baseline.get("sets", {}).get(name, {}).get("full_match", "-")
            lines.append(f"| {SET_ROLES.get(name, name)} | {before} | {current['full_match']} |")
    for name, current in report["sets"].items():
        role = SET_ROLES.get(name, name)
        lines += ["", f"## {role}: 변형별", "", "| 변형 | 일치 |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in current["by_kind"].items()]
        lines += ["", f"## {role}: 질문 유형별", "", "| 유형 | 일치 |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in current["by_family"].items()]
        lines += ["", f"## {role}: 슬롯별", "", "| 슬롯 | 정확도 |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in current["slot_accuracy"].items()]
        if current.get("novel_parts"):
            lines += ["", f"## {role}: 개발용에 없던 요소가 든 문항", "", "| 요소 | 일치 |", "|---|---|"]
            lines += [f"| {NOVEL_PARTS.get(k, k)} | {v} |" for k, v in current["novel_parts"].items()]
        lines += ["", f"- 개발용에 없던 요소가 든 문항: {current['novelty'].get('unseen', '-')}, "
                      f"나머지: {current['novelty'].get('seen', '-')}",
                  f"- 해석 시간(표본) 중앙값 {current['parse_ms_median']}ms, p95 {current['parse_ms_p95']}ms"]
    if report.get("pipeline"):
        pipe = report["pipeline"]
        lines += ["", "## 로컬 엔진 전체 답변 경로(표본)", "",
                  f"- 질문 {pipe['questions']:,}개, 실행 오류 {pipe['runtime_errors']}건, 세 구획 누락 "
                  f"{pipe['missing_sections']}건, 수치 감사 불일치 {pipe['audit_below_full']}건",
                  f"- 응답 시간 중앙값 {pipe['ms_median']}ms, p95 {pipe['ms_p95']}ms, 최대 {pipe['ms_max']}ms"]
    return "\n".join(lines) + "\n"


def pick_pipeline(plans: dict, size: int) -> list[dict]:
    """세트마다 같은 수의 문항을 작업 순서 전체에 고르게 뽑아 전체 답변 경로 표본을 만든다."""
    pools = suite.build_pools()
    frames = {frame["id"]: frame for frame in suite.load_frames()}
    items = []
    for set_name, plan in plans.items():
        templates = suite.load_templates(set_name)[0]
        total = sum(task[3] - task[2] for task in plan["tasks"])
        want = max(1, size // len(plans))
        gap = max(1, total // want)
        picked, before, wanted = [], 0, 0  # 전체 순서의 0, gap, 2*gap, ... 번째(목록을 만들지 않고 계산)
        for task in plan["tasks"]:
            size_here = task[3] - task[2]
            while wanted < want and wanted * gap < before + size_here:
                picked.append((task, task[2] + wanted * gap - before))
                wanted += 1
            before += size_here
        dims_cache: dict = {}
        for task, step in picked:
            number, space, stride, offset = task[1], task[4], task[5], task[6]
            template = templates[number]
            frame = frames[template["frame"]]
            if number not in dims_cache:
                dims_cache[number] = suite.plan_dimensions(template, frame, pools, set_name)
            case = make_case(template, frame, dims_cache[number], (offset + stride * step) % space, set_name)
            items.append({"workspace": template["workspace"], "context": case["context"], "text": case["text"]})
    return items


def run_answers(items: list[dict], workers: int) -> dict:
    """전체 답변 경로를 돌려 합칠 수 있는 형태(응답 시간 목록 포함)로 돌려준다(run_robustness_test.run_pipeline 과 같은 판정)."""
    from src.tools.db_connector import ensure_database

    ensure_database()  # 작업 프로세스들이 동시에 DB 를 만들다 파일 이름 바꾸기에서 부딪치지 않도록 먼저 한 번 만든다
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
    return {"questions": len(rows), "runtime_errors": len(errors), "missing_sections": len(missing),
            "audit_below_full": len(audit_low), "seconds": round(time.perf_counter() - started, 1),
            "ms": [r["ms"] for r in rows if r["ok"]],
            "verdicts": dict(Counter(r.get("verdict") for r in rows if r["ok"])),
            "error_samples": [{"text": r["text"], "error": r["error"], "trace": r["trace"]} for r in errors[:5]],
            "missing_samples": [r["text"] for r in missing[:5]],
            "audit_samples": [{k: r.get(k) for k in ("workspace", "context", "text", "audit_ratio", "ungrounded")}
                              for r in audit_low[:5]]}


def merge_answers(parts: list[dict]) -> dict:
    """전체 답변 경로 몫들을 합쳐 보고서 형태(중앙값·p95·최대)로 만든다."""
    times = sorted(ms for part in parts for ms in part["ms"])
    verdicts: Counter = Counter()
    for part in parts:
        verdicts.update(part["verdicts"])
    merged = {key: sum(part[key] for part in parts)
              for key in ("questions", "runtime_errors", "missing_sections", "audit_below_full")}
    merged.update(seconds=round(max(part["seconds"] for part in parts), 1), parts=len(parts),
                  ms_median=times[len(times) // 2] if times else None,
                  ms_p95=times[max(0, int(len(times) * 0.95) - 1)] if times else None,
                  ms_max=times[-1] if times else None, verdicts=dict(verdicts))
    for key in ("error_samples", "missing_samples", "audit_samples"):
        merged[key] = [sample for part in parts for sample in part[key]][:5]
    return merged


def name_shard(out_dir: Path, tag: str, set_name: str, part: int, parts: int, suffix: str) -> Path:
    if parts <= 1:
        return out_dir / f"{tag}_{set_name}.{suffix}"
    return out_dir / f"{tag}_{set_name}_p{part:02d}of{parts:02d}.{suffix}"


def request_awake() -> None:
    """Windows 에서 이 시험이 도는 동안만 시스템 절전을 막아 달라고 요청한다(프로세스가 끝나면 풀림, 설정은 그대로)."""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


def main() -> int:
    request_awake()
    parser = argparse.ArgumentParser(description="대규모 조합 행동 시험")
    parser.add_argument("--suite", choices=tuple(SUITE_FILES), default="scale")
    parser.add_argument("--tag", default="run")
    parser.add_argument("--sets", default="dev,heldout")
    parser.add_argument("--per-set", type=int, default=12_750_000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--chunk", type=int, default=1000)
    parser.add_argument("--limit-templates", type=int, default=0, help="세트당 문장틀 수 제한(빠른 확인용)")
    parser.add_argument("--out-dir", default=str(PROCESSED_DIR / "scale_runs"))
    parser.add_argument("--hide-samples", default="", help="오답 문장을 보고서에 남기지 않을 세트(쉼표 구분)")
    parser.add_argument("--report", action="store_true", help="data/processed/<시험>_report.json/.md 로 저장")
    parser.add_argument("--save-baseline", action="store_true", help="세트 요약을 data/meta/<시험>_baseline.json 에 저장")
    parser.add_argument("--pipeline", type=int, default=0, help="전체 답변 경로 표본 실행 프로세스 수(0=생략)")
    parser.add_argument("--pipeline-sample", type=int, default=200_000)
    parser.add_argument("--parts", type=int, default=1, help="여러 컴퓨터로 나눌 몫 수")
    parser.add_argument("--part", default="", help="이 컴퓨터가 돌릴 몫 번호(쉼표 구분, 비우면 전부)")
    parser.add_argument("--pipeline-parts", type=int, default=1)
    parser.add_argument("--pipeline-part", type=int, default=-1, help="전체 답변 경로 표본 중 이 컴퓨터 몫(-1=전부)")
    parser.add_argument("--merge-only", action="store_true", help="채점 없이 모아 둔 몫 파일만 합쳐 보고서를 만든다")
    args = parser.parse_args()
    load_suite(args.suite)
    files = SUITE_FILES[args.suite]
    sets = [name for name in args.sets.split(",") if name]
    hidden = {name for name in args.hide_samples.split(",") if name}
    mine = {int(k) for k in args.part.split(",") if k.strip()} if args.part else set(range(args.parts))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    plans = {name: plan_tasks(name, args.per_set, args.chunk, args.limit_templates, args.parts) for name in sets}
    signature = {"suite": args.suite, "parts": args.parts,
                 **{name: {k: plan[k] for k in ("templates", "quota", "questions")} for name, plan in plans.items()}}
    plan_path = out_dir / f"{args.tag}_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != signature:
        print(f"[중단] {plan_path.name} 의 계획과 지금 계획이 다릅니다. 다른 --tag 를 쓰거나 이전 결과를 지우세요.")
        return 2
    plan_path.write_text(json.dumps(signature, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, plan in plans.items():
        print(f"{SET_ROLES.get(name, name)}: 문장틀 {plan['templates']:,}개(사용 가능 {plan['templates_available']:,}, "
              f"출처 {plan['by_source']}), 탈락 {plan['dropped']}, 할당량 {plan['quota']:,}, 문항 {plan['questions']:,}개, "
              f"작업 {len(plan['tasks']):,}개", flush=True)
    templates = {name: suite.load_templates(name)[0] for name in sets}
    keys = {(name, part): name_shard(out_dir, args.tag, name, part, args.parts, "jsonl")
            for name in sets for part in range(args.parts)}
    digests = {(name, part): name_shard(out_dir, args.tag, name, part, args.parts, "hashes")
               for name in sets for part in range(args.parts)}
    done = set()
    for path in keys.values():
        if path.exists():
            done.update(json.loads(line)["task"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if args.merge_only:
        missing = [task for name in sets for task in plans[name]["tasks"]
                   if f"{templates[name][task[1]]['key']}:{task[2]}" not in done]
        if missing:
            print(f"[중단] 계획의 작업 묶음 {len(missing):,}개가 아직 없습니다(몫 {sorted({t[7] for t in missing})}).")
            return 2
    else:
        todo = [task for name in sets for task in plans[name]["tasks"]
                if task[7] in mine and f"{templates[name][task[1]]['key']}:{task[2]}" not in done]
        total_questions = sum(task[3] - task[2] for task in todo)
        print(f"이미 끝난 작업 {len(done):,}개, 이 컴퓨터 몫 {sorted(mine) if args.parts > 1 else '전부'}의 남은 작업 "
              f"{len(todo):,}개(문항 {total_questions:,}개), 프로세스 {args.workers}개", flush=True)
        sinks = {key: path.open("a", encoding="utf-8") for key, path in keys.items() if key[1] in mine}
        hash_sinks = {key: path.open("ab") for key, path in digests.items() if key[1] in mine}
        finished_questions, last_print = 0, time.perf_counter()
        try:
            with Pool(args.workers, initializer=prepare_scale, initargs=(sets, args.suite)) as pool:
                for finished, (record, digest) in enumerate(pool.imap_unordered(run_task, todo, chunksize=1), start=1):
                    key = (record["set"], record["part"])
                    hash_sinks[key].write(digest)
                    hash_sinks[key].flush()
                    sinks[key].write(json.dumps(record, ensure_ascii=False) + "\n")
                    sinks[key].flush()
                    finished_questions += record["n"]
                    if time.perf_counter() - last_print > 60 or finished == len(todo):
                        elapsed = time.perf_counter() - started
                        rate = finished_questions / max(elapsed, 1e-9)
                        remain = (total_questions - finished_questions) / max(rate, 1e-9)
                        print(f"{finished:,}/{len(todo):,} 작업, 문항 {finished_questions:,}/{total_questions:,}, "
                              f"{rate:,.0f}문항/초, 남은 시간 약 {remain / 60:,.0f}분", flush=True)
                        last_print = time.perf_counter()
        finally:
            for sink in list(sinks.values()) + list(hash_sinks.values()):
                sink.close()
    if args.pipeline and args.pipeline_part >= 0:
        items = pick_pipeline(plans, args.pipeline_sample)
        share = [item for number, item in enumerate(items) if number % args.pipeline_parts == args.pipeline_part]
        part_path = out_dir / f"{args.tag}_pipeline_p{args.pipeline_part:02d}of{args.pipeline_parts:02d}.json"
        part_path.write_text(json.dumps(run_answers(share, args.pipeline), ensure_ascii=False), encoding="utf-8")
        print(f"전체 답변 경로 몫 저장: {part_path.name} (질문 {len(share):,}개)")
    if args.parts > 1 and not args.merge_only:
        print("몫 채점 끝. 모든 몫을 한 폴더에 모은 뒤 같은 명령에 --merge-only 를 붙여 보고서를 만드세요.")
        return 0
    report = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "tag": args.tag, "suite": args.suite, "sets": {},
              "protocol": {"seed": suite.SCALE_SEED, "kinds": suite.KINDS, "stress_kinds": list(suite.STRESS_KINDS),
                           "per_set_target": args.per_set, "chunk": args.chunk, "parts": args.parts}}
    for name in sets:
        shard_paths = [keys[(name, part)] for part in range(args.parts)]
        merged = merge_shard(shard_paths, name in hidden)
        plan = plans[name]
        report["sets"][name] = {"role": SET_ROLES.get(name, name), "templates": plan["templates"],
                                "templates_by_source": plan["by_source"], "dropped_templates": plan["dropped"],
                                "quota": plan["quota"],
                                "unique": count_unique([digests[(name, part)] for part in range(args.parts)]), **merged}
    report["total"] = {"questions": sum(s["questions"] for s in report["sets"].values()),
                       "unique": count_unique(list(digests.values()))}
    if files["baseline"].exists() and not args.save_baseline:
        report["baseline"] = json.loads(files["baseline"].read_text(encoding="utf-8"))
    if args.pipeline:
        if args.pipeline_part >= 0 or args.merge_only:
            paths = sorted(out_dir.glob(f"{args.tag}_pipeline_p*of{args.pipeline_parts:02d}.json"))
            if len(paths) != args.pipeline_parts:
                print(f"[중단] 전체 답변 경로 몫 파일이 {len(paths)}/{args.pipeline_parts}개뿐입니다.")
                return 2
            report["pipeline"] = merge_answers([json.loads(p.read_text(encoding="utf-8")) for p in paths])
        else:
            report["pipeline"] = merge_answers([run_answers(pick_pipeline(plans, args.pipeline_sample), args.pipeline)])
    report["seconds"] = round(time.perf_counter() - started, 1)
    (out_dir / f"{args.tag}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.report:
        (PROCESSED_DIR / f"{files['report']}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                                               encoding="utf-8")
        (PROCESSED_DIR / f"{files['report']}.md").write_text(render_markdown(report), encoding="utf-8")
    if args.save_baseline:
        keep = ("templates", "questions", "unique", "full_match", "full_match_ratio", "strict", "stress", "by_kind",
                "by_family", "by_style", "slot_accuracy", "novelty", "novel_parts", "parser_errors")
        saved = json.loads(files["baseline"].read_text(encoding="utf-8")) if files["baseline"].exists() else {"sets": {}}
        for name, current in report["sets"].items():
            saved["sets"][name] = {"measured_at": report["generated_at"], "tag": args.tag,
                                   **{k: current[k] for k in keep if k in current}}
        files["baseline"].write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"전체 문항 {report['total']['questions']:,}개, 서로 다른 문장 {report['total']['unique']:,}개")
    for name, current in report["sets"].items():
        print(f"{current['role']}: 완전 일치 {current['full_match']}, 엄격 {current['strict']}, 부하 {current['stress']}, "
              f"해석기 예외 {current['parser_errors']}건, 해석 중앙값 {current['parse_ms_median']}ms")
    if report.get("pipeline"):
        pipe = report["pipeline"]
        print(f"전체 답변 경로 표본: 질문 {pipe['questions']:,}개, 실행 오류 {pipe['runtime_errors']}건, 세 구획 누락 "
              f"{pipe['missing_sections']}건, 수치 감사 불일치 {pipe['audit_below_full']}건")
    failed = any(s["parser_errors"] for s in report["sets"].values())
    failed = failed or bool(report.get("pipeline", {}).get("runtime_errors"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
