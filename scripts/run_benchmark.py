"""24문항 정량 벤치마크: Baseline 1(단순 평균/엑셀 필터) vs Baseline 2(단일 표 단순 조회) vs Ours(본 에이전트).

정직한 설계
  - 문항·정답: data/meta/benchmark_items.json (Dev 8 / Held-out Test 16). 정답은 도구 임계값이 아니라
    합성 생성기 설계값(심어 둔 은폐 구간, 지역 프로필의 무잡음 기대값 oracle, 배차 설계)과 보유 기간·컬럼 사전에서 온다.
  - oracle 은 synthesize_floating(noisy=False)로 '잡음 없는 기대값'을 계산해 비교 문항의 정답 라벨을 사전 검증한다.
  - 결과는 건수와 분모(성공/전체)를 함께 적고, 실패·보류 사례를 숨기지 않는다.
  - Ours 는 재현성을 위해 로컬 엔진(결정론적)으로 실행한다. Claude 경로는 같은 도구를 쓰므로 수치는 동일하다.
실행: python scripts/run_benchmark.py
"""
from __future__ import annotations

import json
import math
import re
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.agent.intent_parser import parse_question  # noqa: E402
from src.agent.orchestrator import HybridOrchestrator  # noqa: E402
from src.agent.response_cache import ResponseCache  # noqa: E402
from src.config import BENCHMARK_ITEMS_PATH, GAZETTEER_PATH, PROCESSED_DIR  # noqa: E402
from src.tools import synthetic_seed as seed  # noqa: E402
from src.tools.analysis_cubes import load_bus, load_ev, load_floating, load_service, load_weather, stamp_db, warm_cubes  # noqa: E402
from src.tools.gazetteer_index import load_place_index  # noqa: E402

REPORT_JSON = PROCESSED_DIR / "benchmark_report.json"
REPORT_MD = PROCESSED_DIR / "benchmark_report.md"


# ---------------------------------------------------------------- oracle(정답 라벨 사전 검증)

def compute_oracle(region: str, hours: list[int] | None, resident_only: bool) -> float:
    gazetteer = json.loads(GAZETTEER_PATH.read_text(encoding="utf-8"))
    row = next(r for r in gazetteer["regions"] if r["emd_nm"] == region)
    calendar = seed.build_calendar()
    weather = load_weather(stamp_db())
    code = seed.pick_observatory(row["lat"], row["lon"], gazetteer["weather_stations"])
    pivot = weather.pivot_table(index="observatory_code", columns="base_date", values="daily_rainfall", aggfunc="first")
    rain = np.vstack([pivot.reindex(columns=calendar["base_date"]).loc[code].fillna(0).to_numpy() >= 1.0])
    frame = seed.synthesize_floating(np.random.default_rng(0), calendar, [row], rain, noisy=False)
    if resident_only:
        frame = frame[frame["resd"] == "제주"]
    frame = frame.assign(presence=frame[["resd_pop_cnt", "work_pop_cnt", "visit_pop_cnt"]].sum(axis=1))
    per_hour = frame.groupby(["base_date", "time"])["presence"].sum().reset_index()
    if hours:
        per_hour = per_hour[per_hour["time"].isin(hours)]
    return float(per_hour["presence"].mean())


def check_oracle(item: dict) -> dict | None:
    expected = item["expected"]
    pair = expected.get("oracle_pair")
    if not pair:
        return None
    kind = expected["oracle_expect"]
    evening = list(range(18, 23)) if kind != "reversal_night" else [21, 22, 23]
    if kind == "first_higher_evening":
        evening = [18, 19, 20, 21]
    values = {
        region: {"naive_24h": compute_oracle(region, None, False),
                 "resident_window": compute_oracle(region, evening, True)}
        for region in pair
    }
    a, b = pair
    naive_first = values[a]["naive_24h"] > values[b]["naive_24h"]
    window_first = values[a]["resident_window"] > values[b]["resident_window"]
    if kind in ("reversal", "reversal_night"):
        consistent = naive_first and not window_first
    elif kind == "no_reversal":
        consistent = naive_first == window_first
    else:
        consistent = window_first
    return {"pair": pair, "window_hours": evening, "values": values, "label_consistent": consistent,
            "naive_margin": abs(values[a]["naive_24h"] - values[b]["naive_24h"]) / max(values[a]["naive_24h"], values[b]["naive_24h"]),
            "window_margin": abs(values[a]["resident_window"] - values[b]["resident_window"]) /
                             max(values[a]["resident_window"], values[b]["resident_window"])}


# ---------------------------------------------------------------- 기준선

def find_units(plan) -> list:
    units, _ = load_place_index().resolve_areas(plan.areas)
    if len(units) == 1:
        units.append(load_place_index().pick_partner(units[0]))
    return units


def read_direction(text: str) -> int:
    tail = text.split("보다", 1)[1] if "보다" in text else text
    return -1 if any(w in tail for w in ("적", "낮", "한산", "덜")) else 1


def run_baseline(item: dict, method: str) -> dict:
    """method: mean(Baseline 1, 단순 평균/엑셀 필터) | lookup(Baseline 2, 단일 표 단순 조회)."""
    started = time.perf_counter()
    plan = parse_question(item["question"], item["workspace"])
    text = item["question"]
    stamp = stamp_db()
    verdict, note = "answered", ""
    floating = load_floating(stamp)
    units = find_units(plan) if plan.areas else []
    if re.search(r"\d+(\.\d+)?\s*배|폭증|반토막|급감|때문|평균\s*\d+", text):
        verdict, note = "confirmed", "보도·주장의 수치를 표에서 그대로 확인(집계 변경·인과·분모 점검 없음)"
        if method == "mean" and "전기차" in text:
            ev = load_ev(stamp)
            note += f" / 대기시간 평균을 가능한 행({int(ev['request_time'].astype(str).str.len().gt(10).sum())}행)으로 계산"
    elif plan.intent in ("hidden_slice",) or "숨은" in text or "증편" in text or "평균적" in text:
        note = "월·연 평균 승하차만 비교" if method == "mean" else "시간대별 평균(요일·날씨 구분 없음)만 조회"
        verdict = "none"
    elif "평소보다" in text:
        bus = load_bus(stamp)
        hours = list(range(7, 12)) if "오전" in text else list(range(0, 24))
        overall = float(bus["volume"].mean())
        subset = bus[bus["hour"].isin(hours)]
        verdict = "confirmed" if float(subset["volume"].mean()) > overall else "refuted"
        note = "조건(시간대) 필터 평균을 전체 평균과 비교(같은 시각 기준선 없음)"
    elif len(units) >= 2:
        values = {}
        for unit in units[:2]:
            frame = floating[floating["emd"].isin(unit.emds)]
            if method == "lookup" and plan.hour_from is not None:
                frame = frame[frame["hour"].between(plan.hour_from, plan.hour_to or plan.hour_from)]
            values[unit.label] = float(frame["total"].mean()) if not frame.empty else 0.0
        leader = max(values, key=values.get)
        first = units[0].label
        if plan.intent == "fact_check" and "보다" in text:
            holds = (leader == first) if read_direction(text) > 0 else (leader != first)
            verdict = "confirmed" if holds else "refuted"
        else:
            verdict = "no_reversal"
        note = f"관광객 포함 체류 평균 비교: {values}"
    elif plan.needs_abstain:
        verdict, note = "answered", "해당 컬럼·기간이 없어도 판정 보류 없이 답변 시도"
    elapsed = round((time.perf_counter() - started) * 1000, 1)
    return {"verdict": verdict, "note": note, "elapsed_ms": elapsed}


# ---------------------------------------------------------------- 채점

def score_ours(item: dict, bundle: dict) -> tuple[bool, str]:
    expected = item["expected"]
    verdict = bundle["verdict"]
    outputs = bundle.get("outputs") or {}
    if verdict in expected.get("must_not", []):
        return False, f"금지 판정 {verdict}(오경보)"
    if verdict not in expected["verdicts"]:
        return False, f"판정 {verdict} (정답 {expected['verdicts']})"
    if expected.get("abstain"):
        ok = bool(bundle.get("field_question"))
        return ok, "판정 보류 + 현장 확인 질문" if ok else "현장 확인 질문 누락"
    if "slice" in expected:
        want = expected["slice"]
        slices = (outputs.get("excavate_masked_slice") or {}).get("slices") or []
        for found in slices:
            if (want["station"] in found["target"] and want["day_group"] in found["day_group"]
                    and set(found["hours"]) & set(want["hours"])):
                return True, f"{found['target']} · {found['day_group']} · {found['window']} ({found['lift']:.2f}배)"
        return False, "심어 둔 은폐 구간을 찾지 못함: " + "; ".join(
            f"{s['target']}·{s['day_group']}·{s['window']}" for s in slices[:3])
    if "boundary_hours" in expected:
        boundary = ((outputs.get("simulate_reversal_boundary") or {}).get("boundary") or {}).get("boundary")
        ok = boundary in expected["boundary_hours"]
        return ok, f"귀가 경계 {boundary}시 (정답 {expected['boundary_hours']})"
    if "break_month" in expected:
        breaks = (outputs.get("excavate_masked_slice") or {}).get("breaks") or []
        ok = any(b["month"] == expected["break_month"] for b in breaks)
        return ok, f"단절 월 {[b['month'] for b in breaks]} (정답 {expected['break_month']})"
    return True, f"판정 {verdict}"


def score_baseline(item: dict, verdict: str) -> bool:
    expected = item["expected"]
    if verdict in expected.get("must_not", []):
        return False
    if expected.get("abstain"):
        return verdict == "abstain"
    if any(k in expected for k in ("slice", "boundary_hours", "break_month")):
        return False
    return verdict in expected["verdicts"]


GROUPS = {
    "mean_illusion": ("평균 착시 오판 건수(낮을수록 좋음)", ["D1", "D2", "T1", "T2", "T4"], "fail"),
    "masked_slice": ("은폐 슬라이스 검출 건수", ["D3", "D4", "T7", "T8", "T9", "T10"], "pass"),
    "causal_block": ("집계 변경·미측정 인과·분모 함정 차단 건수", ["D5", "D6", "T13", "T14"], "pass"),
    "false_alarm": ("정상 주장·대조군 오경보 건수(낮을수록 좋음)", ["D7", "T3", "T11", "T12", "T15"], "fail"),
    "abstention": ("판정 보류(Abstention) 정확 건수", ["D8", "T5", "T6", "T16"], "pass"),
}


def summarize(rows: list[dict], key: str) -> dict:
    out = {}
    for name, (label, ids, mode) in GROUPS.items():
        picked = [r for r in rows if r["id"] in ids]
        count = sum(1 for r in picked if (r[key]["pass"] if mode == "pass" else not r[key]["pass"]))
        out[name] = {"label": label, "count": count, "total": len(picked), "text": f"{count} / {len(picked)}"}
    for split in ("dev", "test"):
        picked = [r for r in rows if r["split"] == split]
        passed = sum(1 for r in picked if r[key]["pass"])
        out[f"accuracy_{split}"] = {"count": passed, "total": len(picked), "text": f"{passed} / {len(picked)}"}
    passed = sum(1 for r in rows if r[key]["pass"])
    out["accuracy_all"] = {"count": passed, "total": len(rows), "text": f"{passed} / {len(rows)}"}
    times = [r[key]["elapsed_ms"] for r in rows]
    out["avg_ms"] = round(statistics.mean(times), 1)
    out["p95_ms"] = round(sorted(times)[max(0, math.ceil(len(times) * 0.95) - 1)], 1)
    return out


def main() -> int:
    items = json.loads(BENCHMARK_ITEMS_PATH.read_text(encoding="utf-8"))["items"]
    warm_cubes()
    _ = load_service(stamp_db())
    oracle_rows = {item["id"]: check_oracle(item) for item in items if item["expected"].get("oracle_pair")}
    rows = []
    tracemalloc.start()
    for item in items:
        orchestrator = HybridOrchestrator(item["workspace"], force_offline=True, cache=ResponseCache(enabled=False))
        started = time.perf_counter()
        bundle = orchestrator.answer(item["question"])
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        passed, detail = score_ours(item, bundle)
        base_mean = run_baseline(item, "mean")
        base_lookup = run_baseline(item, "lookup")
        rows.append({
            "id": item["id"], "split": item["split"], "category": item["category"], "question": item["question"],
            "expected": item["expected"]["verdicts"], "rationale": item["rationale"],
            "ours": {"verdict": bundle["verdict"], "pass": passed, "detail": detail, "elapsed_ms": elapsed,
                     "headline": bundle["headline"], "field_question": bundle["field_question"],
                     "number_audit": bundle.get("number_audit")},
            "baseline_mean": {**base_mean, "pass": score_baseline(item, base_mean["verdict"])},
            "baseline_lookup": {**base_lookup, "pass": score_baseline(item, base_lookup["verdict"])},
            "oracle": oracle_rows.get(item["id"]),
        })
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / 1024 / 1024
    except Exception:
        rss = None
    report = {
        "generated_by": "scripts/run_benchmark.py",
        "protocol": json.loads(BENCHMARK_ITEMS_PATH.read_text(encoding="utf-8"))["protocol"],
        "engine": "로컬 신경-기호 엔진(결정론, force_offline=True, 캐시 미사용)",
        "summary": {
            "ours": summarize(rows, "ours"),
            "baseline_mean": summarize(rows, "baseline_mean"),
            "baseline_lookup": summarize(rows, "baseline_lookup"),
        },
        "peak_python_alloc_mb": round(peak / 1024 / 1024, 1),
        "process_rss_mb": None if rss is None else round(rss, 1),
        "oracle_label_check": {k: v["label_consistent"] for k, v in oracle_rows.items() if v},
        "items": rows,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    lines = ["# 24문항 벤치마크 결과", "", f"- 엔진: {report['engine']}",
             f"- 피크 파이썬 메모리 할당: {report['peak_python_alloc_mb']} MB, 프로세스 RSS: {report['process_rss_mb']} MB",
             f"- 정답 라벨 oracle 사전 검증: {report['oracle_label_check']}", "",
             "| 지표 | Baseline 1 단순 평균 | Baseline 2 단일 표 조회 | Ours |", "|---|---|---|---|"]
    for name in list(GROUPS) + ["accuracy_dev", "accuracy_test", "accuracy_all"]:
        label = GROUPS[name][0] if name in GROUPS else {"accuracy_dev": "Dev 정답(8)", "accuracy_test": "Held-out Test 정답(16)",
                                                         "accuracy_all": "전체 정답(24)"}[name]
        s = report["summary"]
        lines.append(f"| {label} | {s['baseline_mean'][name]['text']} | {s['baseline_lookup'][name]['text']} | {s['ours'][name]['text']} |")
    s = report["summary"]
    lines.append(f"| 평균 응답 시간(ms) | {s['baseline_mean']['avg_ms']} | {s['baseline_lookup']['avg_ms']} | {s['ours']['avg_ms']} |")
    lines += ["", "## 문항별 결과(실패·보류 포함 전체 공개)", "",
              "| ID | 구분 | 범주 | Ours 판정 | 통과 | 근거 | B1 | B2 |", "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['id']} | {row['split']} | {row['category']} | {row['ours']['verdict']} | "
                     f"{'O' if row['ours']['pass'] else 'X'} | {row['ours']['detail']} | "
                     f"{row['baseline_mean']['verdict']}({'O' if row['baseline_mean']['pass'] else 'X'}) | "
                     f"{row['baseline_lookup']['verdict']}({'O' if row['baseline_lookup']['pass'] else 'X'}) |")
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n저장: {REPORT_JSON.relative_to(ROOT)}, {REPORT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
