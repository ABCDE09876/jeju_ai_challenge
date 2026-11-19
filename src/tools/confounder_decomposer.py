"""Tool 2: decompose_confounders

단순 전체 평균(Naive Mean)에서 3대 허수를 차례로 벗겨 순위 역전(Rank Reversal)을 찾는다.
  ① strip_tourist_bias        : 도외 거주자(관광객) 체류를 걷어낸 도민 체류
  ② detect_night_cliff        : 요청 시간대(기본: 주제별 생활 시간대)만 보기 + 야간 잔류율(21~23시 / 12~17시)
  ③ evaluate_weather_spillover: 비 오는 날 조건(요청이 없으면 스트레스 테스트로 우천일 적용)
"""
from __future__ import annotations

import pandas as pd

from src.tools.analysis_cubes import load_bus, load_card, load_floating, stamp_db
from src.tools.catalog_resolver import describe_unit
from src.tools.gazetteer_index import load_place_index
from src.tools.resilient_loader import average_safely, divide_safely
from src.tools.tool_common import (DAY_LABELS, WEATHER_LABELS, attach_josa, build_hours, describe_tables,
                                   format_change, format_count, format_pct, measure_time, normalize_day,
                                   normalize_topic, normalize_weather, relax_by_ladder, tidy_payload)


def quote_name(label) -> str:
    return f"'{label}'"


def strip_tourist_bias(frame: pd.DataFrame) -> dict:
    total = average_safely(frame["total"])
    resident = average_safely(frame["resident"])
    share = divide_safely(frame["outsider"].sum(), frame["total"].sum())
    return {"naive_total": total, "resident": resident, "tourist_share": share}


def detect_night_cliff(frame: pd.DataFrame) -> dict:
    day = average_safely(frame.loc[frame["hour"].between(12, 17), "resident"])
    night = average_safely(frame.loc[frame["hour"].between(21, 23), "resident"])
    return {"day_resident": day, "night_resident": night, "night_retention": divide_safely(night, day)}


def evaluate_weather_spillover(frame: pd.DataFrame, hours: list[int], day_type: str) -> dict:
    window = frame[frame["hour"].isin(hours)]
    if day_type == "weekday":
        window = window[~window["is_weekend"]]
    elif day_type == "weekend":
        window = window[window["is_weekend"]]
    elif day_type == "market":
        window = window[window["is_market"]]
    rain = window[window["is_rain"]]
    dry = window[~window["is_rain"]]
    rain_mean, dry_mean = average_safely(rain["resident"]), average_safely(dry["resident"])
    delta = divide_safely((rain_mean or 0) - (dry_mean or 0), dry_mean) if rain_mean is not None else None
    return {"rain_resident": rain_mean, "dry_resident": dry_mean, "rain_delta": delta,
            "rain_days": int(rain["base_date"].nunique()), "dry_days": int(dry["base_date"].nunique())}


def measure_bus_rain(station_ids: list[str], hours: list[int]) -> dict:
    if not station_ids:
        return {}
    bus = load_bus(stamp_db())
    window = bus[bus["bus_station_id"].isin(station_ids) & bus["hour"].isin(hours) & ~bus["is_weekend"]]
    if window.empty:
        return {}
    per_day = window.groupby(["base_date", "is_rain"], as_index=False)["board"].sum()
    rain = average_safely(per_day.loc[per_day["is_rain"], "board"])
    dry = average_safely(per_day.loc[~per_day["is_rain"], "board"])
    return {"bus_board_window": average_safely(per_day["board"]), "bus_board_rain": rain, "bus_board_dry": dry,
            "bus_rain_delta": divide_safely((rain or 0) - (dry or 0), dry) if rain is not None else None,
            "bus_rain_days": int(per_day["is_rain"].sum()), "bus_dry_days": int((~per_day["is_rain"]).sum())}


def measure_card_split(emds: list[str]) -> dict:
    card = load_card(stamp_db())
    rows = card[card["emd"].isin(emds)]
    if rows.empty:
        return {}
    total = rows["use_amt"].sum()
    resident = rows.loc[rows["is_resident"], "use_amt"].sum()
    resident_night = rows.loc[rows["is_resident"] & rows["time"].isin(["21~24", "00~06"]), "use_amt"].sum()
    return {"card_resident_share": divide_safely(resident, total), "card_night_share": divide_safely(resident_night, resident),
            "card_split_share": divide_safely(rows["split_amt"].sum(), total)}


def rank_units(values: dict[str, float | None]) -> list[str]:
    usable = {k: v for k, v in values.items() if v is not None}
    return sorted(usable, key=lambda k: usable[k], reverse=True)


def decompose_confounders(areas=None, topic: str = "living", hour_from=None, hour_to=None, weather: str = "any",
                          day_type: str = "any", question: str = "") -> dict:
    output = {"tool": "decompose_confounders", "ok": True}
    with measure_time(output):
        index = load_place_index()
        units, unknown = index.resolve_areas(areas or [])
        if not units and question:
            units, _ = index.resolve_areas([m.surface for m in index.find_places(question)])
        auto_partner = None
        if len(units) == 1:
            auto_partner = index.pick_partner(units[0])
            units.append(auto_partner)
        if not units:
            output.update({"ok": False, "verdict": "insufficient_area", "unknown_areas": unknown,
                           "message": "비교할 제주 지역을 찾지 못했습니다.",
                           "suggestions": {name: index.suggest_names(name) for name in unknown}})
            return tidy_payload(output)

        topic = normalize_topic(topic)
        hours, explicit_hours = build_hours(hour_from, hour_to, topic)
        weather = normalize_weather(weather)
        day = normalize_day(day_type)
        stress_weather = weather if weather != "any" else "rain"
        window_label = f"{hours[0]}시" if len(hours) == 1 else f"{hours[0]}~{hours[-1]}시"
        stamp = stamp_db()
        cube = load_floating(stamp)

        metrics, notes = [], []
        stage_values = {"S0": {}, "S1": {}, "S2": {}, "S3": {}}
        for unit in units[:4]:
            frame = cube[cube["emd"].isin(unit.emds)]
            if frame.empty:
                notes.append(f"{unit.label}: 유동인구 원자료가 없어 비교에서 제외했습니다.")
                continue
            tourist = strip_tourist_bias(frame)
            cliff = detect_night_cliff(frame)
            spill = evaluate_weather_spillover(frame, hours, day)
            window_pick = relax_by_ladder(frame, hours, day, "any", "resident")
            stress_pick = relax_by_ladder(frame, hours, day, stress_weather, "resident")
            for pick in (window_pick, stress_pick):
                if pick.note and pick.note not in notes:
                    notes.append(f"{unit.label}: {pick.note}")
            window_value = average_safely(window_pick.frame["resident"]) if not window_pick.frame.empty else None
            stress_value = average_safely(stress_pick.frame["resident"]) if not stress_pick.frame.empty else None
            spatial = describe_unit(unit, stamp)
            stations = [s["bus_station_id"] for s in spatial["stations"] if s["emd_nm"] in unit.emds] or \
                       [s["bus_station_id"] for s in spatial["stations"][:2]]
            bus = measure_bus_rain(stations, hours)
            card = measure_card_split(unit.emds)
            stage_values["S0"][unit.label] = tourist["naive_total"]
            stage_values["S1"][unit.label] = tourist["resident"]
            stage_values["S2"][unit.label] = window_value
            stage_values["S3"][unit.label] = stress_value
            metrics.append({
                "label": unit.label, "emds": unit.emds, "auto_partner": unit is auto_partner,
                **tourist, **cliff, **spill, **bus, **card,
                "window_resident": window_value, "stress_resident": stress_value,
                "window_days": int(window_pick.frame["base_date"].nunique()) if not window_pick.frame.empty else 0,
                "stress_days": int(stress_pick.frame["base_date"].nunique()) if not stress_pick.frame.empty else 0,
                "stations_used": stations,
            })

        stage_labels = {
            "S0": "단순 전체 평균(관광객 포함·24시간·날씨 무관)",
            "S1": "① 관광객(도외 거주자) 거품 제거",
            "S2": f"② {DAY_LABELS.get(day, '전체 요일')} {window_label} 생활 시간대만",
            "S3": f"③ {WEATHER_LABELS[stress_weather]} 조건 추가" + ("" if weather != "any" else "(스트레스 테스트)"),
        }
        stages, previous_order, flips = [], None, []
        for key in ("S0", "S1", "S2", "S3"):
            order = rank_units(stage_values[key])
            stages.append({"stage": key, "label": stage_labels[key], "values": stage_values[key], "order": order,
                           "leader": order[0] if order else None})
            if previous_order and order and previous_order[0] != order[0]:
                flips.append({"stage": key, "label": stage_labels[key], "from": previous_order[0], "to": order[0]})
            previous_order = order or previous_order
        final_key = "S3" if weather != "any" else "S2"
        naive_leader = stages[0]["leader"]
        final_leader = next(s["leader"] for s in stages if s["stage"] == final_key)
        stress_leader = stages[3]["leader"]
        if len(metrics) < 2:
            verdict = "insufficient_area"
        elif naive_leader != final_leader:
            verdict = "reversal"
        elif stress_leader != final_leader:
            verdict = "conditional_reversal"
        else:
            verdict = "no_reversal"

        facts = []
        for item in metrics:
            facts.append(f"{item['label']} 시간당 평균 체류 인구(관광객 포함·24시간): {format_count(item['naive_total'])}명")
            facts.append(f"{item['label']} 도외 거주자(관광객) 체류 비중: {format_pct(item['tourist_share'])}")
            facts.append(f"{item['label']} 도민 체류({DAY_LABELS.get(day, '전체 요일')} {window_label}): "
                         f"{format_count(item['window_resident'])}명")
            facts.append(f"{item['label']} 야간 잔류율(도민 21~23시 ÷ 12~17시): {format_pct(item['night_retention'])}")
            if item.get("rain_delta") is not None:
                facts.append(f"{item['label']} 비 오는 날 {window_label} 도민 체류 변화: {format_change(item['rain_delta'])}"
                             f"(우천 {item['rain_days']}일 / 비우천 {item['dry_days']}일)")
            if item.get("bus_rain_delta") is not None:
                facts.append(f"{item['label']} 인근 정류소 평일 {window_label} 승차, 비 오는 날 변화: "
                             f"{format_change(item['bus_rain_delta'])}")
            if item.get("card_resident_share") is not None:
                line = f"{item['label']} 카드 이용액 중 제주도민 비중: {format_pct(item['card_resident_share'])}"
                if (item.get("card_split_share") or 0) >= 0.5:
                    line += " (이름이 겹치는 법정동 금액을 행정동에 균등 배분한 값이 대부분이라 이웃 동과 같게 나올 수 있음)"
                facts.append(line)

        reversals = []
        for flip in flips:
            reversals.append(f"{flip['label']} 단계에서 1위가 '{flip['from']}' -> "
                             f"{attach_josa(quote_name(flip['to']), '으로/로')} 바뀝니다.")
        if verdict == "no_reversal" and naive_leader:
            reversals.append(f"세 가지 허수를 차례로 걷어내도 {attach_josa(quote_name(naive_leader), '이/가')} "
                             f"계속 앞섭니다(순위 유지).")
        if verdict == "conditional_reversal":
            reversals.append(f"평상시 결론('{final_leader}' 우위)은 유지되지만 {WEATHER_LABELS[stress_weather]}에는 "
                             f"{attach_josa(quote_name(stress_leader), '으로/로')} 뒤집힙니다(조건부 역전).")
        output.update({
            "topic": topic, "hours": hours, "hours_explicit": explicit_hours, "day_type": day, "weather": weather,
            "stress_weather": stress_weather, "final_stage": final_key, "areas": [u.export_unit() for u in units[:4]],
            "unknown_areas": unknown, "auto_partner": auto_partner.label if auto_partner else None,
            "metrics": metrics, "stages": stages, "flips": flips, "verdict": verdict,
            "naive_leader": naive_leader, "final_leader": final_leader, "stress_leader": stress_leader,
            "facts": facts, "reversals": reversals, "relaxation": notes,
            "provenance": describe_tables(["floating_pop_hourly", "card_time_monthly", "bus_stop_hourly",
                                           "weather_daily"]),
            "method": "권역은 소속 읍면동의 시간당 평균(읍면동 수 차이 보정). 우천일=일 강수량 1.0mm 이상.",
        })
    return tidy_payload(output)
