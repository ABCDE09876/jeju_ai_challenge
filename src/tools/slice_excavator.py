"""Tool 3: excavate_masked_slice

  find_masked_overload    : 연평균·월평균으로는 정상인데 [요일 × 시간대 × 기상/장날] 조각에서 부하가 폭증하는 은폐 구간
  detect_structural_break : 월초 계단식 점프(집계·수집 방식 변경 의심)를 찾아 집계변경 이력과 대조
  audit_causal_claim      : 주장의 '원인' 개념을 원자료 컬럼 사전과 대조해 미측정 인과 과장을 중립 문장으로 교정
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from src.tools.analysis_cubes import (load_bus, load_ev, load_floating, load_food, load_meta, load_wifi, stamp_db)
from src.tools.catalog_resolver import describe_unit
from src.tools.gazetteer_index import load_place_index
from src.tools.resilient_loader import average_safely, divide_safely
from src.tools.tool_common import (WEEKDAY_CODES, attach_josa, describe_tables, format_count, format_pct,
                                   format_times, measure_time, normalize_topic, tidy_payload)


def quote_word(text: str) -> str:
    return f"'{text}'"

SLICE_LIFT = 1.3
SLICE_MIN_DAYS = 5
SLICE_MIN_VOLUME = 5.0
DAY_GROUPS = [
    ("장날(2·7일)", lambda f: f["is_market"]),
    ("비 오는 평일", lambda f: f["is_rain"] & ~f["is_weekend"]),
    ("비 오는 주말", lambda f: f["is_rain"] & f["is_weekend"]),
    ("7~8월 주말", lambda f: f["peak_season"] & f["is_weekend"]),
    ("비 오는 날", lambda f: f["is_rain"]),
    ("평일", lambda f: ~f["is_weekend"]),
    ("주말", lambda f: f["is_weekend"]),
] + [(f"{label}요일", (lambda code: (lambda f: f["weekday_no"] == code))(code)) for label, code in WEEKDAY_CODES.items()]

CAUSE_LEXICON = {
    "tourist": ["관광객", "관광", "외지인", "여행객", "중국인", "외국인", "방문객"],
    "ev": ["전기차", "전기자동차", "EV"],
    "rain": ["비 때문", "비가 와", "비가 많이", "비 와서", "강수", "장마", "태풍", "우천", "폭우", "날씨"],
    "temperature": ["더위", "폭염", "기온", "추위", "한파"],
    "income": ["소득", "경기", "물가", "불황", "고물가", "경제"],
    "migration": ["이주", "전입", "인구 증가", "이사 온", "이민"],
    "construction": ["공사"],
    "event": ["축제", "행사", "공연"],
    "pandemic": ["코로나", "팬데믹", "감염병"],
    "collection_method": ["RFID", "집계 방식", "수집 방식", "계량 방식"],
    "parking": ["주차"],
    "elderly": ["고령화", "노인 인구"],
    "market_day": ["장날", "오일장"],
}
CAUSE_WORDS = {"tourist": "관광객", "ev": "전기차 보급", "rain": "비(강수)", "temperature": "기온", "income": "소득·경기",
               "migration": "인구 이주", "construction": "공사", "event": "축제·행사", "pandemic": "감염병",
               "collection_method": "집계·수집 방식", "parking": "주차", "elderly": "고령화", "market_day": "장날"}
CAUSE_CONCEPTS = {"tourist": "tourist", "ev": "ev_charging", "rain": "rain", "temperature": "temperature",
                  "collection_method": "collection_method", "elderly": "elderly"}
EFFECT_LEXICON = {
    "food_waste": (["음식물", "쓰레기"], ["food_waste_monthly"], "음식물쓰레기 배출량"),
    "ev_wait": (["충전 대기", "대기시간", "대기 시간", "충전"], ["ev_charging_wait"], "전기차 충전 대기"),
    "wifi_usage": (["와이파이", "wifi", "WiFi"], ["wifi_monthly_users"], "와이파이 이용자 수"),
    "bus_usage": (["버스", "대중교통", "승객", "승차"], ["bus_stop_hourly"], "버스 이용자 수"),
    "sales": (["매출", "소비", "카드", "장사"], ["card_time_monthly", "card_week_monthly"], "카드 매출"),
    "visitor": (["유동인구", "사람", "붐", "북적", "활기"], ["floating_pop_hourly"], "유동인구"),
}
TABLE_GRAIN = {"food_waste_monthly": ("월", "201801", "202105"), "wifi_monthly_users": ("월", "201901", "202112"),
               "bus_stop_hourly": ("일", "20260501", "20261031"), "floating_pop_hourly": ("일", "20260501", "20261031"),
               "card_time_monthly": ("월", "202511", "202610"), "card_week_monthly": ("월", "202511", "202610"),
               "weather_daily": ("일", "20260501", "20261031"), "ev_charging_wait": ("건", "2021", "2022")}
CAUSE_TABLE = {"rain": "weather_daily", "temperature": "weather_daily", "tourist": "floating_pop_hourly",
               "market_day": "market_days", "collection_method": "aggregation_changes"}
CAUSAL_MARKERS = ["때문", "탓", "덕분", "원인", "영향으로", "영향을", "로 인해", "으로 인해", "바람에", "까닭", "해서 ",
                  "늘어서", "줄어서", "와서"]


# ---------------------------------------------------------------- 은폐 구간

def scan_slices(frame: pd.DataFrame, value: str, key: str, label: str) -> list[dict]:
    """frame: (key, base_date, hour) 단위. 같은 시각 전체 평균 대비 조각 평균의 배율(lift)을 본다."""
    if frame.empty:
        return []
    baseline = frame.groupby("hour")[value].mean()
    overall = average_safely(frame[value])
    total_days = frame["base_date"].nunique()
    found = []
    for group_label, mask_fn in DAY_GROUPS:
        subset = frame[mask_fn(frame)]
        days = subset["base_date"].nunique()
        if days < SLICE_MIN_DAYS or days > 0.45 * total_days:
            continue
        slice_mean = subset.groupby("hour")[value].mean()
        lift = (slice_mean / baseline.reindex(slice_mean.index)).replace([np.inf, -np.inf], np.nan)
        hot = [h for h in sorted(lift.index) if lift.get(h, 0) >= SLICE_LIFT and baseline.get(h, 0) >= SLICE_MIN_VOLUME]
        runs, current = [], []
        for hour in hot:
            if current and hour != current[-1] + 1:
                runs.append(current)
                current = []
            current.append(hour)
        if current:
            runs.append(current)
        for run in runs:
            slice_value = float(slice_mean.reindex(run).mean())
            base_value = float(baseline.reindex(run).mean())
            hours_all = sorted(baseline.index)
            window = f"{run[0]:02d}~{run[-1]:02d}시"
            if len(run) >= 12:
                window = f"하루 종일({window})"
            found.append({
                "target": label, "target_key": key, "day_group": group_label, "hours": run,
                "window": window, "slice_mean": slice_value, "baseline_mean": base_value,
                "lift": divide_safely(slice_value, base_value), "days": int(days), "share_of_days": divide_safely(days, total_days),
                "overall_hourly_mean": overall,
                "profile": {"hours": [int(h) for h in hours_all],
                            "baseline": [float(baseline.get(h, 0.0)) for h in hours_all],
                            "slice": [float(slice_mean.get(h, 0.0)) for h in hours_all]},
            })
    found.sort(key=lambda s: (s["lift"] or 0) * math.sqrt(len(s["hours"])), reverse=True)
    picked: list[dict] = []
    for item in found:
        overlap = any(set(item["hours"]) & set(p["hours"]) and (item["lift"] or 0) <= (p["lift"] or 0) * 1.02
                      for p in picked if p["target_key"] == item["target_key"])
        if not overlap:
            picked.append(item)
        if len(picked) >= 3:
            break
    return picked


def find_masked_overload(units, stamp: float, user_group: str | None = None) -> list[dict]:
    bus = load_bus(stamp)
    floating = load_floating(stamp)
    slices = []
    for unit in units:
        spatial = describe_unit(unit, stamp)
        stations = [s for s in spatial["stations"] if s["emd_nm"] in unit.emds] or spatial["stations"][:2]
        for station in stations[:4]:
            frame = bus[bus["bus_station_id"] == station["bus_station_id"]]
            value = {"어르신": "board_elder", "청소년": "board_youth"}.get(user_group or "", "volume")
            suffix = {"board_elder": " 어르신 승차", "board_youth": " 청소년 승차"}.get(value, " 승하차")
            label = f"{station['bus_station_name']} 정류소{suffix}"
            slices += [{**s, "unit": unit.label} for s in scan_slices(frame, value, station["bus_station_id"], label)]
        region = floating[floating["emd"].isin(unit.emds)]
        if not region.empty:
            per_hour = region.groupby(["base_date", "hour"], as_index=False).agg(
                total=("total", "mean"), is_market=("is_market", "first"), is_rain=("is_rain", "first"),
                is_weekend=("is_weekend", "first"), peak_season=("peak_season", "first"),
                weekday_no=("weekday_no", "first"))
            slices += [{**s, "unit": unit.label}
                       for s in scan_slices(per_hour, "total", f"area:{unit.label}", f"{unit.label} 체류 인구")]
    slices.sort(key=lambda s: s["lift"] or 0, reverse=True)
    return slices[:5]


# ---------------------------------------------------------------- 집계 단절

def detect_structural_break(series: pd.Series, table: str, emd: str = "") -> list[dict]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    values = values[values > 0]
    if len(values) < 6:
        return []
    logs = np.log(values.astype(float))
    diffs = logs.diff().dropna()
    center = float(diffs.median())
    mad = float((diffs - center).abs().median()) * 1.4826
    changes = load_meta(stamp_db())["changes"]
    breaks = []
    months = list(values.index)
    for month, delta in diffs.items():
        ratio = math.exp(delta)
        z = abs(delta - center) / max(mad, 0.02)
        big = ratio >= 2.0 or ratio <= 0.6
        if not (big or (z >= 4 and (ratio >= 1.5 or ratio <= 0.67))):
            continue
        position = months.index(month)
        before = values.iloc[max(0, position - 3):position]
        after = values.iloc[position:position + 3]
        level_ratio = divide_safely(after.mean(), before.mean())
        if level_ratio is None or not (level_ratio >= 1.8 or level_ratio <= 0.65):
            continue
        record = next((c for c in changes if c["table_name"] == table and str(c["change_month"]) == str(month)
                       and (not c["emd_nm"] or c["emd_nm"] == emd)), None)
        breaks.append({
            "table": table, "emd": emd, "month": str(month), "prev_month": months[position - 1],
            "prev_value": float(values.iloc[position - 1]), "value": float(values.iloc[position]),
            "month_ratio": ratio, "before_mean": float(before.mean()), "after_mean": float(after.mean()),
            "level_ratio": level_ratio, "robust_z": z,
            "registered_change": record,
            "explanation": (f"집계변경 이력 일치: {record['change_type']}({record['before_desc']} -> {record['after_desc']})"
                            if record else "집계변경 이력 없음: 수집·집계 방식 변경 여부 현장 확인 필요"),
        })
    return breaks


def collect_breaks(units, topic: str, claim: str, stamp: float) -> list[dict]:
    text = claim or ""
    breaks = []
    wants_waste = topic == "waste" or any(w in text for w in ("음식물", "쓰레기"))
    wants_wifi = topic == "wifi" or any(w in text.lower() for w in ("와이파이", "wifi"))
    if wants_waste:
        food = load_food(stamp)
        emds = [e for u in units for e in u.emds] or sorted(food["emd_nm"].unique())
        for emd in emds:
            series = food[food["emd_nm"] == emd].set_index("base_year_month")["discharge_daily_avg"].sort_index()
            breaks += detect_structural_break(series, "food_waste_monthly", emd)
    if wants_wifi:
        wifi = load_wifi(stamp)
        series = wifi.groupby("base_year_month")["user_count"].sum().sort_index()
        breaks += detect_structural_break(series, "wifi_monthly_users", "")
    return breaks


# ---------------------------------------------------------------- 인과 감사

def find_concepts(text: str, lexicon: dict) -> list[str]:
    hits = []
    for concept, words in lexicon.items():
        for word in words:
            if word.lower() in text.lower():
                hits.append(concept)
                break
    return hits


def audit_causal_claim(claim: str, units=None) -> dict:
    text = (claim or "").strip()
    if not text:
        return {"status": "no_claim"}
    has_marker = any(marker in text for marker in CAUSAL_MARKERS)
    causes = [c for c in find_concepts(text, CAUSE_LEXICON)]
    effects = []
    for key, (words, tables, label) in EFFECT_LEXICON.items():
        if any(w.lower() in text.lower() for w in words):
            effects.append((key, tables, label))
    if not has_marker or not causes or not effects:
        return {"status": "no_causal_claim", "causes": causes, "effects": [e[0] for e in effects],
                "has_marker": has_marker}
    effect_key, effect_tables, effect_label = effects[0]
    columns = load_meta(stamp_db())["columns"]
    assessments = []
    for cause in causes:
        concept = CAUSE_CONCEPTS.get(cause)
        in_effect = [c["column_name"] for c in columns
                     if c["table_name"] in effect_tables and concept and concept in str(c["concepts"]).split(",")]
        cause_table = CAUSE_TABLE.get(cause)
        joinable = False
        reason = ""
        if in_effect:
            joinable = True
            reason = f"{effect_tables[0]} 안에 '{CAUSE_WORDS[cause]}' 측정 컬럼({', '.join(in_effect)})이 있음"
        elif cause_table:
            grain_e = TABLE_GRAIN.get(effect_tables[0], ("?", "", ""))
            grain_c = TABLE_GRAIN.get(cause_table, grain_e)
            overlap = not (grain_e[2] < grain_c[1] or grain_c[2] < grain_e[1]) if grain_c[1] and grain_e[1] else True
            same_grain = grain_e[0] == grain_c[0] or cause_table in ("market_days", "aggregation_changes")
            if cause_table == "aggregation_changes":
                joinable, reason = True, "집계변경 이력(aggregation_changes)으로 시점 대조 가능"
            elif overlap and same_grain:
                joinable, reason = True, f"{cause_table}와 기간·단위가 겹쳐 연관성은 검정 가능"
            else:
                why = "기간이 겹치지 않음" if not overlap else f"시간 단위 불일치({grain_e[0]} vs {grain_c[0]})"
                reason = f"원인 후보 자료({cause_table})와 {why}"
        else:
            reason = f"'{CAUSE_WORDS[cause]}'을(를) 측정한 컬럼이 원자료 어디에도 없음"
        assessments.append({"cause": cause, "cause_word": CAUSE_WORDS[cause], "measured_in_effect": in_effect,
                            "joinable": joinable, "reason": reason})
    unmeasured = [a for a in assessments if not a["joinable"]]
    subject = "해당 지역"
    if units:
        subject = units[0].label
    if unmeasured:
        cause_text = ", ".join(a["cause_word"] for a in unmeasured)
        status = "unmeasured_cause"
        neutral = (f"{subject}의 {effect_label} 변화는 원자료로 확인할 수 있지만, {attach_josa(quote_word(cause_text), '은/는')} "
                   f"이 자료로 측정되지 않아 원인으로 단정할 수 없습니다. 교정문: \"{subject}의 "
                   f"{attach_josa(effect_label, '이/가')} 변했다(원인 미확인, {cause_text} 영향은 추가 자료로 검증 필요).\"")
    else:
        status = "correlation_only"
        cause_text = ", ".join(a["cause_word"] for a in assessments)
        neutral = (f"{attach_josa(quote_word(cause_text), '과/와')} {effect_label}의 동시 변화(연관)는 검토할 수 있으나 "
                   f"인과는 별도 설계 없이 단정할 수 없습니다. 교정문: \"{attach_josa(cause_text, '과/와')} "
                   f"{attach_josa(effect_label, '이/가')} 함께 변했다(인과 미검증).\"")
    return {"status": status, "effect": effect_key, "effect_label": effect_label, "effect_tables": effect_tables,
            "assessments": assessments, "neutral_rewrite": neutral}


def audit_denominator(stamp: float) -> dict:
    """전기차 충전 대기처럼 '계산 가능한 행'이 전체의 일부뿐인 지표의 분모를 드러낸다."""
    ev = load_ev(stamp)
    rows = int(len(ev))
    if rows == 0:
        return {}
    kwh = pd.to_numeric(ev.get("charge_kwh"), errors="coerce").fillna(0)
    request = pd.to_datetime(ev.get("request_time"), errors="coerce")
    start = pd.to_datetime(ev.get("start_time"), errors="coerce")
    valid = request.notna() & start.notna() & (start >= request) & (kwh > 0)
    waits = ((start - request).dt.total_seconds() / 60)[valid]
    return {"rows": rows, "not_charged": int((kwh <= 0).sum()), "valid_time_rows": int(valid.sum()),
            "coverage": divide_safely(int(valid.sum()), rows), "mean_wait_min": average_safely(waits),
            "median_wait_min": float(waits.median()) if len(waits) else None}


def check_claim_numbers(claim: str, breaks: list[dict], slices: list[dict], extra: list[float] | None = None) -> list[dict]:
    checks = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*배", claim or ""):
        stated = float(match.group(1))
        candidates = [b["month_ratio"] for b in breaks] + [b["level_ratio"] for b in breaks]
        candidates += [s["lift"] for s in slices if s.get("lift")]
        candidates += [v for v in (extra or []) if v]
        if not candidates:
            checks.append({"stated": stated, "computed": None, "status": "검증 불가(대응 수치 없음)"})
            continue
        best = min(candidates, key=lambda v: abs(v - stated))
        agree = abs(best - stated) / max(stated, 1e-9) <= 0.05
        checks.append({"stated": stated, "computed": round(best, 2), "status": "수치 일치" if agree else "수치 불일치"})
    return checks


def compute_hallim_ratio(stamp: float) -> dict | None:
    food = load_food(stamp)
    rows = food[food["emd_nm"] == "한림읍"].set_index("base_year_month")["discharge_daily_avg"]
    if "201910" in rows.index and "201912" in rows.index:
        return {"oct": float(rows["201910"]), "dec": float(rows["201912"]),
                "ratio": divide_safely(rows["201912"], rows["201910"])}
    return None


def excavate_masked_slice(areas=None, topic: str = "crowd", claim: str = "", mode: str = "all",
                          user_group: str | None = None, question: str = "") -> dict:
    output = {"tool": "excavate_masked_slice", "ok": True}
    with measure_time(output):
        index = load_place_index()
        units, unknown = index.resolve_areas(areas or [])
        text = claim or question or ""
        if not units and text:
            units, _ = index.resolve_areas([m.surface for m in index.find_places(text)])
        topic = normalize_topic(topic)
        mode = mode if mode in ("slice", "break", "causal", "all") else "all"
        stamp = stamp_db()
        wants_slices = mode in ("slice", "all") and units and topic not in ("waste", "wifi", "ev")
        slices = find_masked_overload(units[:3], stamp, user_group) if wants_slices else []
        breaks = collect_breaks(units, topic, text, stamp) if mode in ("break", "all") else []
        causal = audit_causal_claim(text, units) if mode in ("causal", "all") else {"status": "skipped"}
        hallim = compute_hallim_ratio(stamp) if any(b["emd"] == "한림읍" for b in breaks) else None
        number_checks = check_claim_numbers(text, breaks, slices, [hallim["ratio"]] if hallim else None)
        wants_ev = topic == "ev" or any(w in text for w in ("전기차", "충전"))
        denominator = audit_denominator(stamp) if wants_ev else {}
        thin = bool(denominator) and (denominator.get("coverage") or 0) < 0.3
        if causal.get("status") == "unmeasured_cause":
            verdict = "unmeasured_cause"
        elif breaks:
            verdict = "structural_break"
        elif thin:
            verdict = "thin_denominator"
        elif slices:
            verdict = "masked_overload"
        else:
            verdict = "none"

        facts, reversals = [], []
        for item in slices[:3]:
            facts.append(f"{item['target']}: 전체 평균 시간당 {format_count(item['overall_hourly_mean'])}명, "
                         f"{item['day_group']} {item['window']}에는 {format_count(item['slice_mean'])}명"
                         f"(같은 시각 평소 {format_count(item['baseline_mean'])}명의 {format_times(item['lift'])}, "
                         f"해당 {item['days']}일)")
            reversals.append(f"평균에 가려진 과부하: {item['target']} · {item['day_group']} · {item['window']} "
                             f"({format_times(item['lift'])})")
        for item in breaks:
            facts.append(f"{item['emd'] or '제주 전역'} {item['table']}: {item['prev_month']} {format_count(item['prev_value'])} -> "
                         f"{item['month']} {format_count(item['value'])} ({format_times(item['month_ratio'])}, "
                         f"전후 3개월 평균 {format_times(item['level_ratio'])})")
            reversals.append(f"{item['month'][:4]}-{item['month'][4:]} 월초 계단식 단절 감지 -> {item['explanation']}")
        if hallim:
            facts.append(f"한림읍 2019-10 일평균 {hallim['oct']:,.2f} -> 2019-12 일평균 {hallim['dec']:,.2f} "
                         f"({hallim['ratio']:.2f}배)")
        if denominator:
            facts.append(f"전기차 충전 기록 {denominator['rows']:,}행 중 미충전 {denominator['not_charged']:,}행, "
                         f"대기시간 계산 가능(요청·시작 시각 유효) {denominator['valid_time_rows']:,}행"
                         f"({format_pct(denominator['coverage'])})")
            if denominator.get("mean_wait_min") is not None:
                facts.append(f"유효 {denominator['valid_time_rows']:,}행 기준 평균 대기 {denominator['mean_wait_min']:.1f}분, "
                             f"중앙값 {denominator['median_wait_min']:.1f}분")
            if thin:
                reversals.append(f"분모 함정: 대기시간 평균은 전체의 {format_pct(denominator['coverage'])}인 "
                                 f"{denominator['valid_time_rows']:,}행으로만 계산되어 제주 전체 대표값으로 쓸 수 없습니다.")
        for check in number_checks:
            facts.append(f"주장 속 수치 {check['stated']}배 재계산: {check['computed']} -> {check['status']}")
        if causal.get("status") in ("unmeasured_cause", "correlation_only"):
            for assessment in causal["assessments"]:
                reversals.append(f"인과 감사: '{assessment['cause_word']}' -> {assessment['reason']}")
        output.update({
            "topic": topic, "mode": mode, "areas": [u.export_unit() for u in units[:3]], "unknown_areas": unknown,
            "slices": slices, "breaks": breaks, "causal_audit": causal, "number_checks": number_checks,
            "hallim_reference": hallim, "denominator_audit": denominator, "verdict": verdict, "facts": facts,
            "reversals": reversals,
            "neutral_rewrite": causal.get("neutral_rewrite"),
            "thresholds": {"slice_lift": SLICE_LIFT, "slice_min_days": SLICE_MIN_DAYS,
                           "break_rule": "월간 배율 2배 이상/0.6배 이하 또는 로버스트 z>=4, 전후 3개월 수준비 1.8배 이상/0.65배 이하"},
            "provenance": describe_tables(["bus_stop_hourly", "floating_pop_hourly", "food_waste_monthly",
                                           "wifi_monthly_users", "aggregation_changes"]),
        })
    return tidy_payload(output)
