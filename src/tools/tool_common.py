"""4대 도구가 함께 쓰는 조건 마스크, 점진적 완화 사다리, 수치 표기, 출처 태그."""
from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np
import pandas as pd

from src.tools.analysis_cubes import load_registry, stamp_db
from src.tools.resilient_loader import relax_sparse_filter, tidy_number

WEEKDAY_CODES = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}
DAY_LABELS = {"weekday": "평일", "weekend": "주말", "market": "장날(2·7일)", "any": "전체 요일",
              "월": "월요일", "화": "화요일", "수": "수요일", "목": "목요일", "금": "금요일", "토": "토요일", "일": "일요일"}
WEATHER_LABELS = {"rain": "비 오는 날", "dry": "비 안 오는 날", "any": "날씨 무관"}
TOPICS = ("commute", "night", "living", "spending", "crowd", "waste", "wifi", "ev", "general")
TOPIC_TABLES = {
    "commute": ["bus_stop_hourly", "bus_service_hourly", "floating_pop_hourly", "weather_daily"],
    "night": ["floating_pop_hourly", "card_time_monthly", "bus_service_hourly", "weather_daily"],
    "living": ["floating_pop_hourly", "card_time_monthly", "bus_stop_hourly", "weather_daily"],
    "spending": ["card_time_monthly", "card_week_monthly", "floating_pop_hourly"],
    "crowd": ["bus_stop_hourly", "floating_pop_hourly", "market_days", "weather_daily"],
    "waste": ["food_waste_monthly", "aggregation_changes", "floating_pop_hourly"],
    "wifi": ["wifi_monthly_users", "aggregation_changes"],
    "ev": ["ev_charging_wait"],
    "general": ["floating_pop_hourly", "bus_stop_hourly", "card_time_monthly", "weather_daily"],
}
DEFAULT_HOURS = {"commute": (17, 19), "night": (21, 23), "living": (18, 22), "crowd": (8, 20),
                 "spending": (18, 21), "general": (18, 22)}


DIGIT_FINALS = {"0": True, "1": True, "2": False, "3": True, "4": False, "5": False, "6": True, "7": True, "8": True,
                "9": False}


def has_final_consonant(word: str) -> tuple[bool, bool]:
    """(받침 있음, 받침이 ㄹ). 따옴표·괄호 등은 건너뛰고 마지막 한글/숫자로 판단한다."""
    for char in reversed(str(word)):
        code = ord(char) - 0xAC00
        if 0 <= code < 11172:
            tail = code % 28
            return tail != 0, tail == 8
        if char.isdigit():
            return DIGIT_FINALS[char], char in "178"
        if char.isalpha():
            return False, False
    return False, False


def attach_josa(word: str, pair: str) -> str:
    """attach_josa('아라동', '이랑/랑') -> '아라동이랑'. pair: 이/가, 은/는, 을/를, 과/와, 이랑/랑, 으로/로."""
    with_final, without = pair.split("/")
    final, rieul = has_final_consonant(word)
    if pair == "으로/로":
        return word + ("로" if (not final or rieul) else "으로")
    return word + (with_final if final else without)


@contextmanager
def measure_time(bucket: dict):
    started = time.perf_counter()
    try:
        yield
    finally:
        bucket["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)


def normalize_topic(topic) -> str:
    value = str(topic or "general").strip().lower()
    return value if value in TOPICS else "general"


def normalize_day(day_type) -> str:
    value = str(day_type or "any").strip()
    aliases = {"평일": "weekday", "주중": "weekday", "주말": "weekend", "장날": "market", "오일장": "market",
               "전체": "any", "": "any", "none": "any"}
    value = aliases.get(value, value)
    if value.endswith("요일"):
        value = value[0]
    return value if value in DAY_LABELS else "any"


def normalize_weather(weather) -> str:
    value = str(weather or "any").strip().lower()
    aliases = {"비": "rain", "우천": "rain", "rainy": "rain", "맑음": "dry", "sunny": "dry", "none": "any", "": "any"}
    value = aliases.get(value, value)
    return value if value in WEATHER_LABELS else "any"


def build_hours(hour_from, hour_to, topic: str) -> tuple[list[int], bool]:
    """요청 시간대를 시간 목록으로. 지정이 없으면 주제별 기본 시간대를 쓰고 False 를 함께 돌려준다."""
    def clean(value):
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            return None
        return number if 0 <= number <= 24 else None

    start, end = clean(hour_from), clean(hour_to)
    explicit = start is not None or end is not None
    if start is None and end is None:
        start, end = DEFAULT_HOURS.get(topic, (18, 22))
    elif start is None:
        start = max(0, end - 2)
    elif end is None:
        end = min(23, start + 2)
    start, end = min(start, 23), min(end, 23)
    if end < start:
        return list(range(start, 24)) + list(range(0, end + 1)), explicit
    return list(range(start, end + 1)), explicit


def mask_day(frame: pd.DataFrame, day_type: str) -> pd.Series:
    if day_type == "weekday":
        return ~frame["is_weekend"]
    if day_type == "weekend":
        return frame["is_weekend"]
    if day_type == "market":
        return frame["is_market"]
    if day_type in WEEKDAY_CODES:
        return frame["weekday_no"] == WEEKDAY_CODES[day_type]
    return pd.Series(True, index=frame.index)


def mask_weather(frame: pd.DataFrame, weather: str) -> pd.Series:
    if weather == "rain":
        return frame["is_rain"]
    if weather == "dry":
        return ~frame["is_rain"]
    return pd.Series(True, index=frame.index)


def relax_by_ladder(frame: pd.DataFrame, hours: list[int], day_type: str, weather: str, value_column: str,
                    key_column: str = "base_date"):
    """[특정 요일·시각·날씨] -> [평일/주말 시간대 구간] -> [해당 지역 전체] 순으로 조건을 넓힌다.

    표본 수는 '서로 다른 일자 수'로 센다(시간대 행 수가 아님).
    """
    window = set(range(max(0, min(hours) - 1), min(23, max(hours) + 1) + 1))
    broad_day = "weekend" if day_type in ("weekend", "토", "일") else ("weekday" if day_type != "any" else "any")
    stages = [
        (f"요청 조건({DAY_LABELS.get(day_type, '전체')}·{min(hours)}~{max(hours)}시·{WEATHER_LABELS[weather]})",
         lambda f: f["hour"].isin(hours) & mask_day(f, day_type) & mask_weather(f, weather)),
        (f"인접 시간대·{DAY_LABELS.get(broad_day, '전체')} 구간({min(window)}~{max(window)}시)",
         lambda f: f["hour"].isin(window) & mask_day(f, broad_day)),
        ("해당 지역 전체 기간·전체 시간", lambda f: f["hour"] >= 0),
    ]
    day_counts = []
    for label, mask_fn in stages:
        subset = frame[mask_fn(frame)]
        day_counts.append(subset[key_column].nunique() if not subset.empty else 0)
    picked = relax_sparse_filter(frame, stages, value_column=value_column)
    if picked.level >= 0:
        days = picked.frame[key_column].nunique() if not picked.frame.empty else 0
        if days < 5 and picked.level < len(stages) - 1:
            for level in range(picked.level + 1, len(stages)):
                if day_counts[level] >= 5:
                    label, mask_fn = stages[level]
                    subset = frame[mask_fn(frame)]
                    picked.frame, picked.level, picked.label, picked.rows = subset, level, label, len(subset)
                    picked.relaxed = True
                    picked.note = (f"요청하신 세부 조건(N={day_counts[0]}일<5)의 표본이 적어 "
                                   f"{label}(N={day_counts[level]}일) 기준으로 자동 확장해 분석했습니다.")
                    break
    picked.requested_rows = day_counts[0]
    return picked


def format_count(value) -> str:
    number = tidy_number(value, 1)
    if number is None:
        return "산출 불가"
    return f"{number:,.0f}" if abs(number) >= 100 else f"{number:,.1f}"


def format_pct(ratio) -> str:
    number = tidy_number(ratio, 4)
    return "산출 불가" if number is None else f"{number * 100:.1f}%"


def format_times(ratio) -> str:
    number = tidy_number(ratio, 4)
    return "산출 불가" if number is None else f"{number:.2f}배"


def format_change(ratio) -> str:
    number = tidy_number(ratio, 4)
    if number is None:
        return "산출 불가"
    sign = "+" if number >= 0 else ""
    return f"{sign}{number * 100:.1f}%"


def describe_tables(tables: list[str]) -> list[dict]:
    registry = load_registry(stamp_db())
    rows = []
    for table in tables:
        meta = registry.get(table)
        if not meta:
            continue
        rows.append({
            "table": table, "source_type": meta.get("source_type"), "dataset_id": meta.get("dataset_id"),
            "reference_date": meta.get("reference_date"), "official_sheet": meta.get("official_sheet"),
            "public_source": meta.get("public_source"), "period": meta.get("period"),
        })
    return rows


def tidy_payload(value):
    """도구 출력 JSON 에서 NaN/inf/numpy 형식을 제거한다."""
    if isinstance(value, dict):
        return {str(k): tidy_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [tidy_payload(v) for v in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    number = tidy_number(value, 4)
    if number is None or isinstance(number, (int, float)):
        return number
    return str(number)
