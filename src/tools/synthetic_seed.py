"""공식 예시데이터 스키마를 보존한 시드 고정(SEED=20261120) 확장 테이블 생성기.

원칙
  - 컬럼명·코드값·단위는 「제주빅데이터센터_예시데이터 및 컬럼확인용.xlsx」의 해당 시트를 그대로 따른다.
  - 예시 시트는 시트당 1행뿐이라 표준편차는 원자료로 추정할 수 없다. 예시값은 규모(앵커)로만 쓰고,
    생성된 분포의 평균·표준편차는 column_profile 테이블에 그대로 공개한다.
  - 검증용 시나리오(PLANTED_SLICES 등)는 문서에 공개하며 원자료(official_raw)를 적재하면 해당 테이블이 대체된다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import ANALYSIS_END, ANALYSIS_START, RAIN_DAY_MM

WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]
WEEKDAY_LONG = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
MARKET_DAYS = (2, 7, 12, 17, 22, 27)  # 제주민속오일장: 매월 2·7일 장
BUS_HOURS = list(range(5, 24))
CARD_BANDS = ["00~06", "06~09", "09~12", "12~15", "15~18", "18~21", "21~24"]
CARD_MONTHS = [f"{y}{m:02d}" for y, m in [(2025, 11), (2025, 12)] + [(2026, k) for k in range(1, 11)]]
TOURIST_SEASON = {1: 0.85, 2: 0.9, 3: 0.95, 4: 1.05, 5: 1.0, 6: 0.95, 7: 1.25, 8: 1.35, 9: 1.0, 10: 1.1, 11: 0.9, 12: 0.85}


@dataclass(frozen=True)
class RegionProfile:
    resident: int      # 거주 규모
    worker: int        # 직장 유입 규모
    leisure: int       # 도민 여가·쇼핑 방문 규모
    tourist: int       # 외지인(관광) 체류 규모
    night: float       # 21~24시 도민 여가 체류 계수(낮을수록 야간 공동화)
    hotel: float       # 관광객 야간 숙박 체류 계수
    rain_tour: float   # 우천일 관광 체류 배율


PROFILES = {
    "한림읍": RegionProfile(25000, 6000, 3000, 9000, 0.50, 0.9, 0.70),
    "애월읍": RegionProfile(30000, 7000, 5000, 26000, 0.45, 0.8, 0.70),
    "구좌읍": RegionProfile(16000, 3000, 1500, 9000, 0.40, 0.7, 0.65),
    "조천읍": RegionProfile(26000, 5000, 2500, 14000, 0.50, 0.9, 0.65),
    "한경면": RegionProfile(9500, 1500, 800, 2500, 0.35, 0.5, 0.75),
    "추자면": RegionProfile(1600, 300, 100, 300, 0.40, 0.6, 0.80),
    "우도면": RegionProfile(1700, 400, 200, 6000, 0.30, 0.3, 0.50),
    "일도1동": RegionProfile(2500, 11000, 7000, 6000, 0.35, 0.5, 1.00),
    "일도2동": RegionProfile(33000, 7000, 4000, 1500, 0.90, 0.6, 0.90),
    "이도1동": RegionProfile(8000, 6000, 3000, 1200, 0.50, 0.6, 0.90),
    "이도2동": RegionProfile(50000, 16000, 9000, 2500, 1.10, 0.7, 0.90),
    "삼도1동": RegionProfile(11000, 4000, 2500, 1500, 0.60, 0.6, 0.90),
    "삼도2동": RegionProfile(8000, 7000, 5000, 3500, 0.30, 0.5, 0.85),
    "용담1동": RegionProfile(7500, 3000, 2000, 2000, 0.50, 0.6, 0.90),
    "용담2동": RegionProfile(14000, 9000, 3000, 12000, 0.70, 1.6, 1.00),
    "건입동": RegionProfile(9500, 6000, 2500, 3000, 0.35, 0.8, 0.90),
    "화북동": RegionProfile(25000, 6000, 2000, 600, 0.85, 0.4, 0.90),
    "삼양동": RegionProfile(27000, 2500, 2000, 800, 0.85, 0.4, 0.90),
    "봉개동": RegionProfile(6500, 2000, 800, 1500, 0.60, 0.4, 0.80),
    "아라동": RegionProfile(38000, 11000, 3500, 1200, 0.90, 0.4, 0.90),
    "오라동": RegionProfile(15000, 4000, 2500, 1000, 0.70, 0.6, 0.90),
    "연동": RegionProfile(43000, 18000, 12000, 11000, 1.15, 1.6, 1.08),
    "노형동": RegionProfile(55000, 12000, 13000, 9000, 1.20, 1.5, 1.08),
    "외도동": RegionProfile(22000, 2500, 2500, 1000, 0.85, 0.4, 0.90),
    "이호동": RegionProfile(5000, 1000, 1500, 3500, 0.60, 0.6, 0.60),
    "도두동": RegionProfile(3000, 1500, 2000, 3500, 0.50, 0.9, 0.85),
    "대정읍": RegionProfile(22000, 4500, 2000, 6000, 0.45, 0.6, 0.80),
    "남원읍": RegionProfile(18500, 3000, 1500, 5000, 0.40, 0.7, 0.80),
    "성산읍": RegionProfile(15000, 3500, 1500, 16000, 0.40, 0.8, 0.70),
    "안덕면": RegionProfile(12000, 4000, 1200, 10000, 0.50, 1.0, 0.80),
    "표선면": RegionProfile(12000, 2500, 1200, 6000, 0.40, 0.7, 0.75),
    "송산동": RegionProfile(4200, 1500, 1000, 2500, 0.40, 0.6, 0.85),
    "정방동": RegionProfile(2300, 1500, 1200, 3000, 0.35, 0.7, 0.85),
    "중앙동": RegionProfile(3300, 4000, 4500, 5000, 0.40, 0.8, 1.00),
    "천지동": RegionProfile(3800, 2000, 1500, 4000, 0.50, 0.8, 0.80),
    "효돈동": RegionProfile(5500, 800, 600, 1500, 0.50, 0.4, 0.75),
    "영천동": RegionProfile(4800, 800, 500, 800, 0.50, 0.4, 0.85),
    "동홍동": RegionProfile(24000, 5000, 4000, 1500, 0.95, 0.5, 0.90),
    "서홍동": RegionProfile(11000, 2000, 1200, 800, 0.80, 0.4, 0.90),
    "대륜동": RegionProfile(15000, 5000, 2000, 1500, 0.85, 0.5, 0.90),
    "대천동": RegionProfile(14000, 2000, 1200, 2000, 0.70, 0.6, 0.85),
    "중문동": RegionProfile(12000, 6000, 1500, 18000, 0.60, 1.6, 0.85),
    "예래동": RegionProfile(4000, 1000, 500, 3000, 0.45, 1.0, 0.80),
}


# 우천일 도민 여가 방문 배율: 실내 상권은 늘고(1.1) 해안·야외 상권은 줄어든다(0.7). 나머지 0.9.
RESIDENT_RAIN = {name: 1.1 for name in ("노형동", "연동", "이도2동", "일도2동")}
RESIDENT_RAIN.update({name: 0.7 for name in ("애월읍", "조천읍", "이호동", "삼도2동", "구좌읍", "한림읍", "성산읍", "중문동",
                                             "안덕면", "표선면", "우도면", "한경면", "남원읍", "대정읍")})


@dataclass(frozen=True)
class StationProfile:
    shape: str           # residential | work | tourist | hub | market
    peak_boarding: float
    peak_buses: float
    rain: float          # 우천일 이용 배율
    night: tuple         # 20,21,22,23시 배차 계수
    elder: float
    youth: float


DEFAULT_NIGHT = (0.70, 0.45, 0.35, 0.25)
STATIONS = {
    "1": StationProfile("hub", 50, 5, 0.90, DEFAULT_NIGHT, 0.10, 0.06),
    "96": StationProfile("market", 35, 5, 0.85, DEFAULT_NIGHT, 0.22, 0.05),
    "9001": StationProfile("hub", 260, 26, 1.00, (0.80, 0.60, 0.50, 0.40), 0.06, 0.05),
    "9002": StationProfile("hub", 240, 28, 1.05, (0.80, 0.60, 0.50, 0.40), 0.10, 0.10),
    "9003": StationProfile("work", 200, 22, 1.10, (0.75, 0.55, 0.45, 0.35), 0.10, 0.10),
    "9004": StationProfile("work", 220, 24, 1.05, (0.75, 0.60, 0.50, 0.40), 0.08, 0.14),
    "9005": StationProfile("work", 150, 18, 1.10, DEFAULT_NIGHT, 0.16, 0.08),
    "9006": StationProfile("work", 180, 20, 1.00, DEFAULT_NIGHT, 0.20, 0.08),
    "9007": StationProfile("work", 160, 12, 1.30, DEFAULT_NIGHT, 0.04, 0.12),
    "9008": StationProfile("work", 70, 6, 1.25, DEFAULT_NIGHT, 0.04, 0.04),
    "9009": StationProfile("residential", 120, 12, 1.15, (0.50, 0.15, 0.12, 0.10), 0.10, 0.16),
    "9010": StationProfile("residential", 90, 9, 1.30, (0.50, 0.15, 0.12, 0.10), 0.12, 0.14),
    "9011": StationProfile("tourist", 40, 4, 0.60, DEFAULT_NIGHT, 0.05, 0.04),
    "9012": StationProfile("residential", 80, 7, 1.20, DEFAULT_NIGHT, 0.10, 0.16),
    "9013": StationProfile("tourist", 60, 6, 0.60, DEFAULT_NIGHT, 0.05, 0.05),
    "9014": StationProfile("hub", 70, 8, 1.00, DEFAULT_NIGHT, 0.18, 0.10),
    "9015": StationProfile("tourist", 35, 4, 0.55, DEFAULT_NIGHT, 0.05, 0.04),
    "9016": StationProfile("hub", 60, 7, 0.90, DEFAULT_NIGHT, 0.16, 0.10),
    "9017": StationProfile("hub", 130, 18, 1.10, DEFAULT_NIGHT, 0.18, 0.10),
    "9018": StationProfile("residential", 70, 7, 1.20, DEFAULT_NIGHT, 0.06, 0.12),
    "9019": StationProfile("tourist", 60, 6, 0.80, DEFAULT_NIGHT, 0.05, 0.04),
    "9020": StationProfile("tourist", 50, 5, 0.70, DEFAULT_NIGHT, 0.06, 0.04),
    "9021": StationProfile("hub", 40, 5, 0.90, DEFAULT_NIGHT, 0.20, 0.08),
    "9022": StationProfile("residential", 45, 4, 1.15, DEFAULT_NIGHT, 0.05, 0.25),
}

# 검증용으로 심어 둔 '평균에 가려진 과부하' 시나리오(문서 공개). 벤치마크 정답은 이 설정에서 온다.
PLANTED_SLICES = [
    {"key": "H1", "station": "96", "days": "market", "hours": [9, 10, 11, 12, 13], "factor": 3.2,
     "label": "제주민속오일장 정류소 · 장날(2·7일) · 09~13시"},
    {"key": "H2", "station": "9009", "days": "rain_weekday", "hours": [17, 18, 19], "factor": 1.6,
     "label": "삼화지구 정류소 · 비 오는 평일 · 17~19시"},
    {"key": "H3", "station": "9013", "days": "peak_weekend", "hours": [14, 15, 16, 17, 18], "factor": 2.2,
     "label": "함덕해수욕장 정류소 · 7~8월 주말 · 14~18시"},
    {"key": "H4", "station": "9004", "days": "fri", "hours": [21, 22, 23], "factor": 1.8,
     "label": "노형오거리 정류소 · 금요일 · 21~23시"},
    {"key": "H5", "station": "9001", "days": "sun", "hours": [17, 18, 19, 20], "factor": 1.7,
     "label": "제주국제공항 정류소 · 일요일 · 17~20시"},
    {"key": "H6", "station": "9006", "days": "sat", "hours": [15, 16, 17, 18], "factor": 1.6,
     "label": "동문로터리 정류소 · 토요일 · 15~18시"},
]

BOARD_SHAPES = {
    "residential": [0.15, 0.45, 1.0, 0.95, 0.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.35, 0.45, 0.5, 0.4, 0.3, 0.22, 0.15, 0.08],
    "work": [0.05, 0.12, 0.3, 0.45, 0.45, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.7, 0.95, 1.0, 0.8, 0.6, 0.45, 0.35, 0.2],
    "tourist": [0.02, 0.05, 0.1, 0.2, 0.35, 0.55, 0.7, 0.8, 0.85, 0.9, 1.0, 1.0, 0.9, 0.7, 0.45, 0.3, 0.2, 0.12, 0.06],
    "hub": [0.2, 0.45, 0.8, 0.95, 0.9, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 1.0, 1.0, 0.85, 0.7, 0.55, 0.4, 0.25],
    "market": [0.1, 0.2, 0.35, 0.5, 0.6, 0.6, 0.6, 0.6, 0.55, 0.5, 0.5, 0.5, 0.45, 0.4, 0.3, 0.2, 0.12, 0.08, 0.05],
}
ALIGHT_SHAPES = {
    "residential": [0.05, 0.1, 0.2, 0.25, 0.25, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.45, 0.8, 1.0, 0.85, 0.6, 0.45, 0.35, 0.2],
    "work": [0.05, 0.3, 0.8, 1.0, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1],
    "tourist": [0.02, 0.05, 0.15, 0.35, 0.65, 0.9, 1.0, 0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.18, 0.12, 0.08, 0.04],
    "hub": [0.2, 0.45, 0.8, 0.95, 0.9, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 1.0, 1.0, 0.85, 0.7, 0.55, 0.4, 0.25],
    "market": [0.1, 0.25, 0.4, 0.6, 0.65, 0.6, 0.55, 0.5, 0.45, 0.45, 0.45, 0.45, 0.4, 0.35, 0.25, 0.18, 0.1, 0.06, 0.04],
}
SERVICE_SHAPE = {5: 0.3, 6: 0.6, 7: 1.0, 8: 1.0, 9: 1.0, 10: 0.75, 11: 0.75, 12: 0.75, 13: 0.75, 14: 0.75,
                 15: 0.75, 16: 0.75, 17: 1.0, 18: 1.0, 19: 1.0}

HOME_URBAN = [0.92] * 6 + [0.88, 0.75, 0.6] + [0.5] * 8 + [0.58, 0.68, 0.76, 0.82, 0.86, 0.89, 0.91]
HOME_RURAL = [0.94] * 6 + [0.9, 0.82, 0.72] + [0.65] * 8 + [0.7, 0.78, 0.84, 0.88, 0.9, 0.92, 0.93]
WORK_CURVE = [0.03] * 6 + [0.1, 0.35, 0.75, 1.0, 1.0, 1.0, 0.9, 1.0, 1.0, 1.0, 1.0, 0.8, 0.45, 0.25, 0.15, 0.08, 0.05, 0.05]
LEISURE_CURVE = [0.03, 0.02, 0.02, 0.02, 0.02, 0.03, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 0.85, 0.75, 0.6, 0.6, 0.65,
                 0.75, 0.95, 1.0, 0.9, 0.7, 0.5, 0.3]
TOURIST_CURVE = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.3, 0.4, 0.55, 0.75, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9,
                 0.75, 0.6, 0.5, 0.4, 0.33, 0.28]


# ---------------------------------------------------------------- 공통

def build_calendar(start: str = ANALYSIS_START, end: str = ANALYSIS_END) -> pd.DataFrame:
    dates = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="D")
    calendar = pd.DataFrame({"date": dates})
    calendar["base_date"] = calendar["date"].dt.strftime("%Y%m%d")
    calendar["weekday_no"] = calendar["date"].dt.weekday
    calendar["day_of_week"] = [WEEKDAYS[i] for i in calendar["weekday_no"]]
    calendar["is_weekend"] = calendar["weekday_no"] >= 5
    calendar["month"] = calendar["date"].dt.month
    calendar["is_market"] = calendar["date"].dt.day.isin(MARKET_DAYS)
    calendar["peak_season"] = calendar["month"].isin([7, 8])
    return calendar


def measure_km(lat1, lon1, lat2, lon2) -> float:
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    arc = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(arc))


def pick_observatory(lat: float, lon: float, observatories: list[dict]) -> str:
    best = min(observatories, key=lambda o: measure_km(lat, lon, o["lat"], o["lon"]))
    return best["observatory_code"]


def format_hhmm(rng: np.random.Generator, low_hour: int, high_hour: int, size: int) -> np.ndarray:
    hours = rng.integers(low_hour, high_hour + 1, size)
    minutes = rng.integers(0, 60, size)
    return hours * 100 + minutes


# ---------------------------------------------------------------- ③ 기상

def synthesize_weather(rng: np.random.Generator, calendar: pd.DataFrame, observatories: list[dict]) -> pd.DataFrame:
    rain_chance = {5: 0.30, 6: 0.42, 7: 0.40, 8: 0.33, 9: 0.33, 10: 0.20}
    rain_scale = {5: 11.0, 6: 17.0, 7: 19.0, 8: 24.0, 9: 18.0, 10: 10.0}
    month_celsius = {5: 18.6, 6: 22.3, 7: 26.6, 8: 27.6, 9: 24.4, 10: 19.9}
    offsets = {"184": 0.0, "185": -0.4, "188": -0.3, "189": 0.8, "328": 0.5}
    wind_base = {"184": 3.2, "185": 5.6, "188": 3.9, "189": 2.8, "328": 2.6}
    typhoon_days = {"20260812", "20260913"}

    months = calendar["month"].to_numpy()
    days = len(calendar)
    island_rain = rng.random(days) < np.vectorize(rain_chance.get)(months)
    typhoon = calendar["base_date"].isin(typhoon_days).to_numpy()
    island_rain |= typhoon
    frames = []
    for obs in observatories:
        code = obs["observatory_code"]
        local = np.where(island_rain, rng.random(days) < 0.85, rng.random(days) < 0.07)
        local |= typhoon
        scale = np.vectorize(rain_scale.get)(months)
        amount = np.where(local, rng.gamma(0.9, scale) + 0.5, 0.0)
        amount = np.where(typhoon, rng.uniform(140, 220, days), amount)
        celsius = np.vectorize(month_celsius.get)(months) + offsets.get(code, 0.0) + rng.normal(0, 1.3, days)
        celsius -= np.where(local, 1.2, 0.0)
        lowest = celsius - rng.uniform(2.5, 4.5, days)
        highest = celsius + rng.uniform(2.5, 5.5, days)
        avg_wind = wind_base.get(code, 3.0) * (1 + 0.45 * local) + rng.normal(0, 0.5, days)
        avg_wind = np.where(typhoon, rng.uniform(11, 15, days), np.clip(avg_wind, 0.5, None))
        max_wind = avg_wind * rng.uniform(2.2, 3.0, days)
        frames.append(pd.DataFrame({
            "base_date": calendar["base_date"],
            "observatory_code": code,
            "average_temperature": np.round(celsius, 1),
            "lowest_temperature": np.round(lowest, 1),
            "lowest_temperature_time": format_hhmm(rng, 4, 6, days),
            "highest_temperature": np.round(highest, 1),
            "highest_temperature_time": format_hhmm(rng, 12, 15, days),
            "daily_rainfall": np.round(amount, 1),
            "maximum_wind_speed": np.round(max_wind, 1),
            "maximum_wind_speed_time": format_hhmm(rng, 0, 23, days),
            "average_wind_speed": np.round(avg_wind, 1),
            "maximum_wind_speed_direction": np.round(rng.uniform(0, 360, days), 1),
            "pt_year": calendar["date"].dt.year,
            "pt_month": months,
        }))
    return pd.concat(frames, ignore_index=True)


def map_rain_days(weather: pd.DataFrame, calendar: pd.DataFrame, codes: list[str]) -> np.ndarray:
    """codes 순서대로 (지점 수 x 일수) 우천 여부 배열을 만든다."""
    pivot = weather.pivot_table(index="observatory_code", columns="base_date", values="daily_rainfall", aggfunc="first")
    pivot = pivot.reindex(columns=calendar["base_date"]).fillna(0.0)
    return np.vstack([pivot.loc[code].to_numpy() >= RAIN_DAY_MM for code in codes])


# ---------------------------------------------------------------- ① 유동인구

def synthesize_floating(rng: np.random.Generator, calendar: pd.DataFrame, regions: list[dict],
                        rain_days: np.ndarray, noisy: bool = True) -> pd.DataFrame:
    """4-11 내국인 유동인구(읍면동)_v02 스키마. resd: '제주'(도민) / '그외'(도외 거주자).

    noisy=False 이면 잡음 없는 기대값을 돌려준다(벤치마크 정답 oracle 용, 도구 계산과 독립).
    """
    days = len(calendar)
    hours = np.arange(24)
    month_tour = calendar["month"].map(TOURIST_SEASON).to_numpy()
    weekend = calendar["is_weekend"].to_numpy()
    market = calendar["is_market"].to_numpy()

    home_rows, work_rows, visit_rows = [], [], []
    out_home, out_work, out_visit = [], [], []
    for index, row in enumerate(regions):
        profile = PROFILES[row["emd_nm"]]
        rural = row["kind"] in ("읍", "면")
        home_curve = np.array(HOME_RURAL if rural else HOME_URBAN)
        leisure = np.array(LEISURE_CURVE, dtype=float)
        leisure[21:] *= profile.night
        leisure[:2] *= profile.night
        tour = np.array(TOURIST_CURVE, dtype=float)
        tour[:7] *= profile.hotel
        tour[20:] *= profile.hotel
        rain = rain_days[index][:, None]
        day_weekend = weekend[:, None]

        home = profile.resident * home_curve[None, :] * np.ones((days, 1))
        work = profile.worker * np.array(WORK_CURVE)[None, :] * np.where(day_weekend, 0.35, 1.0)
        visit = profile.leisure * leisure[None, :] * np.where(day_weekend, 1.25, 1.0)
        visit = visit * np.where(rain, RESIDENT_RAIN.get(row["emd_nm"], 0.9), 1.0)
        if row["emd_nm"] == "도두동":  # 장날 오일장 방문
            visit = visit * np.where(market[:, None] & (hours[None, :] >= 8) & (hours[None, :] <= 14), 2.5, 1.0)
        tourists = (profile.tourist * tour[None, :] * month_tour[:, None]
                    * np.where(day_weekend, 1.15, 1.0) * np.where(rain, profile.rain_tour, 1.0))
        if row["emd_nm"] == "도두동":
            tourists = tourists * np.where(market[:, None] & (hours[None, :] >= 8) & (hours[None, :] <= 14), 1.5, 1.0)

        def jitter():
            return rng.lognormal(0.0, 0.06, (days, 24)) if noisy else np.ones((days, 24))

        home_rows.append(home * jitter())
        work_rows.append(work * jitter())
        visit_rows.append(visit * jitter())
        out_home.append(profile.resident * 0.005 * np.ones((days, 24)) * jitter())
        out_work.append(work * 0.04 * jitter())
        out_visit.append(tourists * jitter())

    def stack(parts):
        return np.rint(np.stack(parts)).astype(np.int64).reshape(-1)

    region_count = len(regions)
    key_region = np.repeat(np.arange(region_count), days * 24)
    key_day = np.tile(np.repeat(np.arange(days), 24), region_count)
    key_hour = np.tile(hours, region_count * days)
    base = pd.DataFrame({
        "base_date": calendar["base_date"].to_numpy()[key_day],
        "time": key_hour,
        "dstrct_se": np.array([r["dstrct_se"] for r in regions])[key_region],
        "emd_cd": np.array([r["emd_cd"] for r in regions])[key_region],
        "emd_nm": np.array([f"{r['city']} {r['emd_nm']}" for r in regions])[key_region],
        "sex": "전체",
        "age": "전체",
    })
    local = base.assign(resd_pop_cnt=stack(home_rows), work_pop_cnt=stack(work_rows),
                        visit_pop_cnt=stack(visit_rows), resd="제주")
    outside = base.assign(resd_pop_cnt=stack(out_home), work_pop_cnt=stack(out_work),
                          visit_pop_cnt=stack(out_visit), resd="그외")
    return pd.concat([local, outside], ignore_index=True)


# ---------------------------------------------------------------- ② 버스

def match_planted(slice_rule: dict, calendar: pd.DataFrame, rain: np.ndarray) -> np.ndarray:
    kind = slice_rule["days"]
    weekday = calendar["weekday_no"].to_numpy()
    if kind == "market":
        return calendar["is_market"].to_numpy()
    if kind == "rain_weekday":
        return rain & (weekday < 5)
    if kind == "peak_weekend":
        return calendar["peak_season"].to_numpy() & (weekday >= 5)
    if kind == "fri":
        return weekday == 4
    if kind == "sun":
        return weekday == 6
    if kind == "sat":
        return weekday == 5
    raise ValueError(kind)


def synthesize_bus(rng: np.random.Generator, calendar: pd.DataFrame, stations: list[dict],
                   rain_days: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """3-04 버스 정류소별 이용 현황 스키마 + 합성 배차표(bus_service_hourly)."""
    days = len(calendar)
    hours = np.array(BUS_HOURS)
    weekday = calendar["weekday_no"].to_numpy()
    weekend = weekday >= 5
    month_tour = calendar["month"].map(lambda m: {7: 1.3, 8: 1.4}.get(m, 1.0)).to_numpy()
    market = calendar["is_market"].to_numpy()

    usage_parts = []
    service_rows = []
    for index, station in enumerate(stations):
        sid = station["bus_station_id"]
        profile = STATIONS[sid]
        rain = rain_days[index]
        board = profile.peak_boarding * np.array(BOARD_SHAPES[profile.shape])[None, :] * np.ones((days, 1))
        alight = profile.peak_boarding * np.array(ALIGHT_SHAPES[profile.shape])[None, :] * np.ones((days, 1))
        if profile.shape in ("residential", "work"):
            day_factor = np.where(weekend, 0.6, 1.0)
        elif profile.shape == "tourist":
            day_factor = np.where(weekend, 1.3, 1.0) * month_tour
        elif profile.shape == "market":
            day_factor = np.where(weekend, 0.9, 1.0)
        else:
            day_factor = np.ones(days)
        rain_factor = np.where(rain, profile.rain, 1.0)
        board = board * (day_factor * rain_factor)[:, None]
        alight = alight * (day_factor * rain_factor)[:, None]
        elder_share = np.full((days, len(hours)), profile.elder)
        for rule in PLANTED_SLICES:
            if rule["station"] != sid:
                continue
            day_mask = match_planted(rule, calendar, rain)
            hour_mask = np.isin(hours, rule["hours"])
            lift = np.where(day_mask[:, None] & hour_mask[None, :], rule["factor"], 1.0)
            board = board * lift
            alight = alight * lift
        if sid == "96":
            market_hours = (hours >= 8) & (hours <= 14)
            elder_share = np.where(market[:, None] & market_hours[None, :], 0.38, elder_share)
        board = board * rng.lognormal(0.0, 0.12, board.shape)
        alight = alight * rng.lognormal(0.0, 0.12, alight.shape)
        youth_share = np.full_like(elder_share, profile.youth)
        general_share = 1.0 - elder_share - youth_share
        for label, totals in (("승차", board), ("하차", alight)):
            for category, share in (("일반", general_share), ("청소년", youth_share), ("어르신", elder_share)):
                counts = np.rint(totals * share).astype(np.int64)
                usage_parts.append(pd.DataFrame({
                    "base_date": np.repeat(calendar["base_date"].to_numpy(), len(hours)),
                    "day_of_week": np.repeat(calendar["day_of_week"].to_numpy(), len(hours)),
                    "base_hour": np.tile(hours, days),
                    "bus_station_id": sid,
                    "bus_station_name": station["bus_station_name"],
                    "get_on_off_type": label,
                    "user_category": category,
                    "pay_type": "카드",
                    "user_count": counts.reshape(-1),
                }))
        for day_type, factor in (("평일", 1.0), ("주말", 0.8)):
            for hour in hours:
                if hour <= 19:
                    shape = SERVICE_SHAPE[hour]
                else:
                    shape = profile.night[hour - 20]
                buses = round(profile.peak_buses * shape * factor, 2)
                service_rows.append({"bus_station_id": sid, "bus_station_name": station["bus_station_name"],
                                     "day_type": day_type, "base_hour": int(hour), "buses_per_hour": buses,
                                     "route_count": max(1, int(round(profile.peak_buses / 3)))})
    return pd.concat(usage_parts, ignore_index=True), pd.DataFrame(service_rows)


# ---------------------------------------------------------------- ① 카드 소비

def synthesize_card(rng: np.random.Generator, regions: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1-14 카드 이용 현황(시간대)_v02 / 1-12 카드 이용 현황(요일별)_v02 스키마."""
    biz_rows = [("I56111", "한식 음식점업"), ("G47122", "체인화 편의점"), ("G47711", "차량용 주유소 운영업"),
                ("G47215", "빵 및 과자류 소매업"), ("I56112", "중식 음식점업"), ("G47130", "면세점")]
    resident_weight = {"I56111": 0.35, "G47122": 0.25, "G47711": 0.22, "G47215": 0.08, "I56112": 0.10, "G47130": 0.0}
    tourist_weight = {"I56111": 0.45, "G47122": 0.20, "G47711": 0.13, "G47215": 0.12, "I56112": 0.10, "G47130": 0.0}
    duty_free = {"연동": 0.9, "중문동": 0.6, "용담2동": 0.5}
    user_kinds = [("제주도민", "도민"), ("내국인관광객", "개별"), ("중국인관광객", "개별")]
    resident_band = [0.03, 0.08, 0.14, 0.20, 0.17, 0.26, 0.12]
    tourist_band = [0.02, 0.06, 0.17, 0.27, 0.24, 0.18, 0.06]
    resident_week = [0.145, 0.145, 0.145, 0.145, 0.145, 0.14, 0.135]
    tourist_week = [0.13, 0.13, 0.13, 0.13, 0.15, 0.17, 0.16]

    time_rows, week_rows = [], []
    for row in regions:
        name = row["emd_nm"]
        profile = PROFILES[name]
        legal = row["legal_dongs"]
        share = 1.0 / len(legal)
        bands_res = np.array(resident_band, dtype=float)
        bands_res[6] *= profile.night
        bands_res = bands_res / bands_res.sum()
        bands_tour = np.array(tourist_band) / sum(tourist_band)
        for month in CARD_MONTHS:
            season = TOURIST_SEASON[int(month[4:])]
            resident_base = (0.25 * profile.resident + 0.6 * profile.worker + 1.2 * profile.leisure) * 3000.0
            tourist_base = profile.tourist * 3500.0 * season
            for bjd in legal:
                for biz_cd, biz_nm in biz_rows:
                    if biz_cd == "G47130" and name not in duty_free:
                        continue
                    for user_type, tour_type in user_kinds:
                        if user_type == "제주도민":
                            amount = resident_base * resident_weight[biz_cd]
                            bands, weeks, ticket = bands_res, resident_week, 18000.0
                        elif user_type == "내국인관광객":
                            weight = duty_free.get(name, 0.0) * 0.5 if biz_cd == "G47130" else tourist_weight[biz_cd]
                            amount = tourist_base * weight
                            bands, weeks, ticket = bands_tour, tourist_week, 32000.0
                        else:
                            weight = duty_free.get(name, 0.0) * 2.0 if biz_cd == "G47130" else tourist_weight[biz_cd] * 0.25
                            amount = tourist_base * 0.12 * weight
                            bands, weeks, ticket = bands_tour, tourist_week, 45000.0
                        amount *= share
                        if amount <= 0:
                            continue
                        for band, portion in zip(CARD_BANDS, bands):
                            value = amount * portion * rng.lognormal(0, 0.07)
                            uses = max(1, int(round(value / ticket)))
                            time_rows.append((month, row["dstrct_se"], bjd, biz_cd, biz_nm, user_type, tour_type, band,
                                              "전체", "전체", max(1, int(round(uses / 2.4))), uses, int(round(value))))
                        for weekday_label, portion in zip(WEEKDAY_LONG, weeks):
                            value = amount * portion * rng.lognormal(0, 0.07)
                            uses = max(1, int(round(value / ticket)))
                            week_rows.append((month, row["dstrct_se"], bjd, biz_cd, biz_nm, user_type, tour_type,
                                              weekday_label, "전체", "전체", max(1, int(round(uses / 2.4))), uses,
                                              int(round(value))))
    time_columns = ["base_year_month", "dstrct_se", "bjd", "biz_cd", "biz_nm", "user_type", "tour_type", "time",
                    "age", "sex", "user_cnt", "use_cnt", "use_amt"]
    week_columns = time_columns[:7] + ["week"] + time_columns[8:]
    return pd.DataFrame(time_rows, columns=time_columns), pd.DataFrame(week_rows, columns=week_columns)


# ---------------------------------------------------------------- ③ 생활환경: 음식물쓰레기, ④ 와이파이

HALLIM_OCT_2019 = 1045.23
HALLIM_DEC_2019 = 61346.42


def build_hallim_daily(rng: np.random.Generator) -> pd.DataFrame:
    """한림읍 RFID 배출량 일별 표본(2019-09~2020-01). 10월·12월 일평균이 공개 사례 수치와 정확히 일치하도록 보정."""
    dates = pd.date_range("2019-09-01", "2020-01-31", freq="D")
    values = []
    for date in dates:
        if date < pd.Timestamp("2019-11-01"):
            values.append(rng.normal(1040, 70))
        else:
            values.append(rng.normal(61000, 2600))
    frame = pd.DataFrame({"date": dates, "amount": np.round(values, 2)})
    for month, target in (("2019-10", HALLIM_OCT_2019), ("2019-12", HALLIM_DEC_2019)):
        mask = frame["date"].dt.strftime("%Y-%m") == month
        idx = frame.index[mask]
        days = len(idx)
        partial = round(float(frame.loc[idx[:-1], "amount"].sum()), 2)
        frame.loc[idx[-1], "amount"] = round(target * days - partial, 2)
    frame["count"] = np.where(frame["date"] < pd.Timestamp("2019-11-01"),
                              rng.integers(18, 31, len(frame)), rng.integers(1650, 1900, len(frame)))
    return frame


def synthesize_food_waste(rng: np.random.Generator, regions: list[dict], hallim_daily: pd.DataFrame) -> pd.DataFrame:
    months = pd.period_range("2018-01", "2021-05", freq="M")
    hallim_monthly = hallim_daily.groupby(hallim_daily["date"].dt.strftime("%Y%m"))["amount"].mean()
    rows = []
    for row in regions:
        profile = PROFILES[row["emd_nm"]]
        base = (profile.resident + 0.6 * profile.tourist + 0.3 * profile.worker) * 2.0
        for period in months:
            key = period.strftime("%Y%m")
            season = 1.08 if period.month in (7, 8) else (0.97 if period.month in (12, 1, 2) else 1.0)
            covid = 0.9 if (period.year == 2020 and 2 <= period.month <= 5) else 1.0
            value = base * season * covid * rng.normal(1.0, 0.03)
            scope = "RFID 전면 계량"
            if row["emd_nm"] == "한림읍":
                if key in hallim_monthly.index:
                    value = float(hallim_monthly[key])
                elif key < "201911":
                    value = rng.normal(1040, 45)
                if key < "201911":
                    scope = "부분 RFID 계량(일부 배출함)"
            rows.append({"base_year_month": key, "emd_nm": row["emd_nm"], "discharge_daily_avg": round(value, 2),
                         "unit_note": "RFID 계량 배출량 지표(사례 원문 단위 미표기)", "collection_scope": scope})
    return pd.DataFrame(rows)


def synthesize_wifi(rng: np.random.Generator) -> pd.DataFrame:
    months = pd.period_range("2019-01", "2021-12", freq="M")
    rows = []
    for period in months:
        key = period.strftime("%Y%m")
        season = 1.3 if period.month in (7, 8) else (0.9 if period.month in (1, 2, 12) else 1.0)
        covid = 1.0
        if period.year == 2020 and 2 <= period.month <= 5:
            covid = {2: 0.75, 3: 0.55, 4: 0.6, 5: 0.7}[period.month]
        elif period.year == 2020 and period.month >= 6:
            covid = 0.85
        elif period.year == 2021:
            covid = 0.9
        unit = "장소" if key >= "202010" else "AP"
        dedup = 0.56 if unit == "장소" else 1.0
        for visit_type, base in (("관광", 420000), ("거주", 260000)):
            factor = season * covid if visit_type == "관광" else 1.0 + 0.004 * (period.year - 2019) * 12
            count = base * factor * dedup * rng.normal(1.0, 0.025)
            rows.append({"base_year_month": key, "visit_type": visit_type, "user_count": int(round(count)),
                         "agg_unit": unit})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- ③ 전기차 충전 대기 표본

EV_ROWS = 2447
EV_NOT_CHARGED = 154
EV_VALID_TIME = 113


def synthesize_ev_charging(rng: np.random.Generator, regions: list[dict]) -> pd.DataFrame:
    """공개 사례 구조(2,447행 중 미충전 154행, 대기시각 유효 113행)를 재현한 합성 표본."""
    names = [r["emd_nm"] for r in regions if r["city"] == "제주시"][:12]
    start = pd.Timestamp("2021-01-01")
    span_minutes = int((pd.Timestamp("2022-06-30") - start).total_seconds() // 60)
    begins = start + pd.to_timedelta(np.sort(rng.integers(0, span_minutes, EV_ROWS)), unit="min")
    durations = rng.integers(25, 95, EV_ROWS)
    kwh = np.round(rng.uniform(8, 48, EV_ROWS), 2)
    order = rng.permutation(EV_ROWS)
    not_charged = order[:EV_NOT_CHARGED]
    valid_time = order[EV_NOT_CHARGED:EV_NOT_CHARGED + EV_VALID_TIME]
    kwh[not_charged] = 0.0
    waits = np.clip(rng.gamma(2.2, 8.0, EV_VALID_TIME), 1, 95).round().astype(int)
    request = np.array([""] * EV_ROWS, dtype=object)
    request[valid_time] = (begins[valid_time] - pd.to_timedelta(waits, unit="min")).strftime("%Y-%m-%d %H:%M:%S")
    broken = order[EV_NOT_CHARGED + EV_VALID_TIME:EV_NOT_CHARGED + EV_VALID_TIME + 60]
    request[broken] = "0000-00-00 00:00:00"
    ends = begins + pd.to_timedelta(durations, unit="min")
    status = np.where(kwh > 0, "충전완료", "미충전")
    return pd.DataFrame({
        "charger_id": [f"EV-{i % 64 + 1:03d}" for i in range(EV_ROWS)],
        "emd_nm": [names[i % len(names)] for i in range(EV_ROWS)],
        "request_time": request,
        "start_time": begins.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": ends.strftime("%Y-%m-%d %H:%M:%S"),
        "charge_kwh": kwh,
        "status": status,
    })


def build_market_days(calendar: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "base_date": calendar["base_date"],
        "market_name": "제주민속오일장",
        "is_market_day": calendar["is_market"].astype(int),
        "rule": "매월 2·7·12·17·22·27일",
    })


def summarize_planted() -> list[dict]:
    return [{k: v for k, v in rule.items()} for rule in PLANTED_SLICES]
