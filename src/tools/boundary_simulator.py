"""Tool 4: simulate_reversal_boundary

사용자 조건(귀가 시각·방문 시각·허용 대기시간·우천 여부)을 움직이며 결론이 뒤집히는 경계선(Reversal Boundary)을
계산하고, 원자료 한계로 남는 현장 확인 질문 1개, 1장짜리 증거 카드, 다음 대화용 추천 후속 질문 3개를 만든다.

귀가 불편 지수(분) = 기대 대기시간(60 / (2 x 시간당 정차 버스 수)) + 혼잡 가산(버스 1대당 승하차 10명 초과분 x 0.25분)
"""
from __future__ import annotations

import pandas as pd

from src.tools.analysis_cubes import load_bus, load_meta, load_service, stamp_db
from src.tools.catalog_resolver import describe_unit
from src.tools.gazetteer_index import load_place_index
from src.tools.resilient_loader import average_safely, divide_safely
from src.tools.tool_common import (WEATHER_LABELS, attach_josa, describe_tables, format_count, measure_time,
                                   normalize_day, normalize_topic, normalize_weather, tidy_payload)

VARIABLES = ("return_hour", "visit_hour", "wait_minutes", "rain")
CROWD_FREE = 10.0
CROWD_WEIGHT = 0.25


def pick_variable(variable, topic: str, question: str) -> str:
    if variable in VARIABLES:
        return variable
    text = question or ""
    if any(w in text for w in ("기다", "대기", "몇 분")):
        return "wait_minutes"
    if any(w in text for w in ("방문", "가면", "장보", "장날", "언제 가")):
        return "visit_hour"
    if any(w in text for w in ("비 오", "비오", "우천", "비가")) and not any(w in text for w in ("퇴근", "귀가")):
        return "rain"
    return "return_hour" if topic in ("commute", "night", "living", "general") else "visit_hour"


def pick_stations(unit, stamp: float) -> list[str]:
    spatial = describe_unit(unit, stamp)
    own = [s["bus_station_id"] for s in spatial["stations"] if s["emd_nm"] in unit.emds]
    return own or [s["bus_station_id"] for s in spatial["stations"][:2]]


def build_hour_curve(station_ids: list[str], hours: list[int], day_type: str, weather: str, stamp: float) -> list[dict]:
    service = load_service(stamp)
    bus = load_bus(stamp)
    service_day = "주말" if day_type in ("weekend", "토", "일") else "평일"
    rows = bus[bus["bus_station_id"].isin(station_ids) & bus["hour"].isin(hours)]
    if day_type in ("weekend", "토", "일"):
        rows = rows[rows["is_weekend"]]
    elif day_type == "market":
        rows = rows[rows["is_market"]]
    elif day_type != "any":
        rows = rows[~rows["is_weekend"]]
    if weather == "rain":
        rows = rows[rows["is_rain"]]
    elif weather == "dry":
        rows = rows[~rows["is_rain"]]
    curve = []
    for hour in hours:
        supply = service[(service["bus_station_id"].isin(station_ids)) & (service["day_type"] == service_day)
                         & (service["hour"] == hour)]["buses_per_hour"]
        buses = average_safely(supply)
        per_station = rows[rows["hour"] == hour].groupby("base_date")["volume"].sum() / max(len(station_ids), 1)
        volume = average_safely(per_station)
        wait = 30.0 / buses if buses else None
        load = divide_safely(volume, buses) if buses else None
        surcharge = max(0.0, (load or 0) - CROWD_FREE) * CROWD_WEIGHT if load is not None else 0.0
        score = (wait + surcharge) if wait is not None else None
        curve.append({"hour": hour, "buses_per_hour": buses, "expected_wait_min": wait, "volume_per_stop": volume,
                      "load_per_bus": load, "discomfort_min": score, "days": int(per_station.shape[0])})
    return curve


def locate_boundary(curves: dict[str, list[dict]], key: str, axis: str = "hour") -> dict:
    labels = list(curves)
    if len(labels) < 2:
        return {"boundary": None, "winners": {}}
    first, second = labels[0], labels[1]
    winners, boundary, previous = {}, None, None
    for a, b in zip(curves[first], curves[second]):
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            continue
        winner = first if va < vb else second
        winners[a[axis]] = winner
        if previous and winner != previous and boundary is None:
            boundary = a[axis]
        previous = winner
    return {"boundary": boundary, "winners": winners, "first": first, "second": second}


def simulate_wait_tolerance(curves: dict[str, list[dict]]) -> dict:
    """허용 대기시간 W(분)를 3~30분으로 늘리며 '대기 W분 이내로 귀가 가능한 저녁 시간대 비율'을 비교."""
    table = []
    labels = list(curves)
    flip_at, previous = None, None
    for limit in range(3, 31):
        row = {"wait_limit_min": limit}
        for label in labels:
            waits = [c["expected_wait_min"] for c in curves[label] if c["expected_wait_min"] is not None]
            row[label] = divide_safely(sum(1 for w in waits if w <= limit), len(waits))
        table.append(row)
        if len(labels) >= 2 and row[labels[0]] is not None and row[labels[1]] is not None:
            leader = labels[0] if row[labels[0]] > row[labels[1]] else (labels[1] if row[labels[1]] > row[labels[0]] else "동률")
            if previous and leader != previous and flip_at is None:
                flip_at = limit
            previous = leader
    return {"table": table, "flip_at_min": flip_at}


def compose_field_question(tables: list[str], variable: str) -> str:
    notes = load_meta(stamp_db())["limitations"]
    priority = {"return_hour": "bus_service_hourly", "visit_hour": "bus_stop_hourly", "wait_minutes": "bus_service_hourly",
                "rain": "weather_daily"}
    wanted = priority.get(variable)
    for note in notes:
        if note["table_name"] == wanted:
            return note["field_question"]
    for note in notes:
        if note["table_name"] in tables:
            return note["field_question"]
    return "이 결론이 실제 생활과 맞는지 해당 지역 주민께 직접 확인할 수 있을까요?"


def compose_followups(units, variable: str, weather: str) -> list[str]:
    main = units[0].label if units else "노형동"
    labels = {u.label for u in units}
    other = load_place_index().pick_partner(units[-1], exclude=labels).label if units else "아라동"
    options = []
    if variable == "return_hour":
        options.append("그럼 퇴근 시간을 밤 9시 이후로 바꾸면?")
    elif variable == "visit_hour":
        options.append(f"{main} 비 오는 장날에는 몇 시가 덜 붐벼?")
    else:
        options.append(f"{main} 퇴근 시간(17~19시) 기준으로 다시 봐줘")
    options.append("비 안 올 때는?" if weather == "rain" else "비 오는 날에는 어떻게 달라져?")
    options.append(f"거기 말고 {attach_josa(other, '이랑/랑')} 비교해 줘")
    return options[:3]


def simulate_reversal_boundary(areas=None, variable: str | None = None, topic: str = "commute", weather: str = "any",
                               day_type: str = "weekday", user_group: str | None = None, wait_minutes=None,
                               question: str = "") -> dict:
    output = {"tool": "simulate_reversal_boundary", "ok": True}
    with measure_time(output):
        index = load_place_index()
        units, unknown = index.resolve_areas(areas or [])
        if not units and question:
            units, _ = index.resolve_areas([m.surface for m in index.find_places(question)])
        topic = normalize_topic(topic)
        variable = pick_variable(variable, topic, question)
        partner = None
        if len(units) == 1 and variable != "visit_hour":
            partner = index.pick_partner(units[0])
            units.append(partner)
        weather = normalize_weather(weather)
        day = normalize_day(day_type)
        if day == "any":
            day = "weekday"
        stamp = stamp_db()
        if variable == "visit_hour":
            hours = list(range(8, 21))
        else:
            hours = list(range(17, 24))
        stations = {u.label: pick_stations(u, stamp) for u in units[:2]}
        scenario_weather = "rain" if variable == "rain" else weather
        curves = {label: build_hour_curve(ids, hours, day, scenario_weather, stamp)
                  for label, ids in stations.items() if ids}
        boundary = locate_boundary(curves, "discomfort_min") if variable != "visit_hour" else {"boundary": None,
                                                                                               "winners": {}}
        facts, reversals = [], []
        extra: dict = {}
        index_label = "방문 혼잡 지수(대기+혼잡, 분)" if variable == "visit_hour" else "귀가 불편 지수(대기+혼잡, 분)"
        if variable == "rain":
            dry_curves = {label: build_hour_curve(ids, hours, day, "dry", stamp) for label, ids in stations.items() if ids}
            dry_boundary = locate_boundary(dry_curves, "discomfort_min")
            extra = {"dry_curves": dry_curves, "dry_boundary": dry_boundary}
            reversals.append(f"비 안 오는 날 경계: {dry_boundary.get('boundary')}시 / 비 오는 날 경계: {boundary.get('boundary')}시")
        if variable == "wait_minutes":
            extra["wait_tolerance"] = simulate_wait_tolerance(curves)
            flip = extra["wait_tolerance"]["flip_at_min"]
            if flip:
                reversals.append(f"허용 대기시간이 {flip}분을 넘으면 저녁 귀가 가능 시간대 비율의 순위가 바뀝니다.")
        if variable == "visit_hour":
            peak_label = list(curves)[0] if curves else None
            if peak_label:
                vols = [(c["hour"], c["volume_per_stop"]) for c in curves[peak_label] if c["volume_per_stop"] is not None]
                if vols:
                    peak_hour, peak_vol = max(vols, key=lambda v: v[1])
                    calm = [h for h, v in vols if h > peak_hour and v <= 0.6 * peak_vol]
                    extra["visit"] = {"peak_hour": peak_hour, "peak_volume": peak_vol,
                                      "calm_after": calm[0] if calm else None}
                    facts.append(f"{peak_label} 정류소당 승하차 최대 {peak_hour}시 {format_count(peak_vol)}명")
                    if calm:
                        boundary = {"boundary": calm[0], "winners": {}, "kind": "visit_calm"}
                        reversals.append(f"방문 시각 경계: {calm[0]}시 이후 방문하면 정류소 승하차가 최대 혼잡({peak_hour}시)의 "
                                         f"60% 이하로 떨어집니다.")
        for label, curve in curves.items():
            sample = [c for c in curve if c["hour"] in (hours[0], hours[min(3, len(hours) - 1)], hours[-1])]
            parts = [f"{c['hour']}시 {c['discomfort_min']:.1f}분" for c in sample if c["discomfort_min"] is not None]
            if parts:
                facts.append(f"{label} {index_label}: " + ", ".join(parts))
        if boundary.get("boundary") is not None and boundary.get("winners"):
            winners = boundary["winners"]
            before = winners.get(hours[0])
            after = winners.get(boundary["boundary"])
            after_quoted = f"'{after}'"
            reversals.insert(0, f"결론 역전 경계: {boundary['boundary']}시. 그 전에는 '{before}', 그 이후에는 "
                                f"{attach_josa(after_quoted, '이/가')} 귀가에 유리합니다.")
        elif boundary.get("winners"):
            leader_quoted = f"'{next(iter(boundary['winners'].values()))}'"
            reversals.insert(0, f"검토한 {hours[0]}~{hours[-1]}시 전 구간에서 "
                                f"{attach_josa(leader_quoted, '이/가')} 계속 유리합니다(경계 없음).")
        tables = ["bus_stop_hourly", "bus_service_hourly", "weather_daily"]
        field_question = compose_field_question(tables, variable)
        if variable == "visit_hour" and extra.get("visit") and curves:
            peak = extra["visit"]["peak_hour"]
            when = "장날 " if day == "market" else ""
            field_question = (f"{list(curves)[0]} 주변 정류소에서 {when}{peak}시 전후에 버스가 만차로 그냥 지나간 적이 있는지 "
                              f"기사님이나 이용자께 확인할 수 있을까요?")
        followups = compose_followups(units, variable, weather)
        card = {
            "title": f"{' vs '.join(curves)} · {WEATHER_LABELS.get(scenario_weather, '')} "
                     f"{'방문 시각' if variable == 'visit_hour' else '귀가'} 경계 증거 카드",
            "headline": reversals[0] if reversals else "경계 분석 결과",
            "facts": facts[:3],
            "limitation": "배차표(bus_service_hourly)는 합성값이며 차내 혼잡·실제 지연은 원자료에 없음",
            "field_question": field_question,
            "chart": {"type": "line", "x": hours,
                      "series": {label: [c["discomfort_min"] for c in curve] for label, curve in curves.items()}},
        }
        output.update({
            "variable": variable, "topic": topic, "day_type": day, "weather": scenario_weather,
            "areas": [u.export_unit() for u in units[:2]], "unknown_areas": unknown,
            "auto_partner": partner.label if partner else None, "stations": stations,
            "curves": curves, "boundary": boundary, **extra,
            "facts": facts, "reversals": reversals, "field_question": field_question, "followups": followups,
            "evidence_card": card, "verdict": "boundary" if boundary.get("boundary") is not None else "no_boundary",
            "formula": "귀가 불편 지수 = 60/(2x시간당 버스) + max(0, 버스당 승하차-10)x0.25 (분)",
            "provenance": describe_tables(tables),
        })
        if not curves:
            output["ok"] = False
            output["message"] = "선택한 지역 주변에서 분석 가능한 정류소를 찾지 못했습니다."
    return tidy_payload(output)
