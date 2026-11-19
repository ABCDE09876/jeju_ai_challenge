"""분석용 메모리 인덱스(인메모리 큐브).

센터 PC(i7-8700, RAM 32GB, GTX 1060 3GB)에서 GPU 없이 빠르게 돌도록 자주 쓰는 테이블을 한 번만 읽어
pandas 로 집계해 둔다. DB 파일 수정시각(stamp)이 바뀌면 자동으로 다시 읽는다.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import DB_PATH, RAIN_DAY_MM
from src.tools.db_connector import ensure_database, fetch_frame


def stamp_db() -> float:
    ensure_database()
    return DB_PATH.stat().st_mtime


def shorten_name(value) -> str:
    text = str(value or "").replace("\xa0", " ").strip()
    return text.split(" ")[-1] if text else ""


@lru_cache(maxsize=2)
def load_registry(stamp: float) -> dict[str, dict]:
    frame = fetch_frame("SELECT * FROM table_registry")
    return {row["table_name"]: row for row in frame.to_dict("records")}


@lru_cache(maxsize=2)
def load_regions(stamp: float) -> pd.DataFrame:
    return fetch_frame("SELECT * FROM regions")


@lru_cache(maxsize=2)
def load_stations(stamp: float) -> pd.DataFrame:
    return fetch_frame("SELECT * FROM bus_stations")


@lru_cache(maxsize=2)
def load_legal_map(stamp: float) -> pd.DataFrame:
    return fetch_frame("SELECT * FROM legal_dong_map")


@lru_cache(maxsize=2)
def load_weather(stamp: float) -> pd.DataFrame:
    frame = fetch_frame("SELECT base_date, observatory_code, daily_rainfall, average_temperature, maximum_wind_speed "
                        "FROM weather_daily")
    frame["base_date"] = frame["base_date"].astype(str)
    frame["observatory_code"] = frame["observatory_code"].astype(str)
    frame["daily_rainfall"] = pd.to_numeric(frame["daily_rainfall"], errors="coerce").fillna(0.0)
    frame["is_rain"] = frame["daily_rainfall"] >= RAIN_DAY_MM
    return frame


@lru_cache(maxsize=2)
def load_calendar(stamp: float) -> pd.DataFrame:
    market = fetch_frame("SELECT base_date, is_market_day FROM market_days")
    market["base_date"] = market["base_date"].astype(str)
    dates = pd.to_datetime(market["base_date"], format="%Y%m%d", errors="coerce")
    calendar = pd.DataFrame({
        "base_date": market["base_date"],
        "weekday_no": dates.dt.weekday,
        "month": dates.dt.month,
        "is_market": market["is_market_day"].astype(int) == 1,
    })
    calendar["is_weekend"] = calendar["weekday_no"] >= 5
    calendar["peak_season"] = calendar["month"].isin([7, 8])
    return calendar


def attach_flags(frame: pd.DataFrame, calendar: pd.DataFrame, weather: pd.DataFrame,
                 observatory_by_key: dict, key: str) -> pd.DataFrame:
    """일자 달력(요일·장날·성수기)과 해당 지점 관측소의 우천 여부를 붙인다."""
    frame = frame.merge(calendar, on="base_date", how="left")
    frame["observatory_code"] = frame[key].map(observatory_by_key).fillna("184")
    rain = weather[["base_date", "observatory_code", "is_rain", "daily_rainfall"]]
    frame = frame.merge(rain, on=["base_date", "observatory_code"], how="left")
    frame["is_rain"] = frame["is_rain"].fillna(False).astype(bool)
    frame["is_market"] = frame["is_market"].fillna(False).astype(bool)
    frame["is_weekend"] = frame["is_weekend"].fillna(False).astype(bool)
    frame["peak_season"] = frame["peak_season"].fillna(False).astype(bool)
    return frame


@lru_cache(maxsize=2)
def load_floating(stamp: float) -> pd.DataFrame:
    """읍면동×일×시간: 도민(resd=제주) 체류, 외지인(그외) 체류."""
    raw = fetch_frame("SELECT * FROM floating_pop_hourly")
    for column in ("resd_pop_cnt", "work_pop_cnt", "visit_pop_cnt"):
        raw[column] = pd.to_numeric(raw.get(column), errors="coerce")
    raw["masked"] = raw[["resd_pop_cnt", "work_pop_cnt", "visit_pop_cnt"]].isna().sum(axis=1)
    raw["presence"] = raw[["resd_pop_cnt", "work_pop_cnt", "visit_pop_cnt"]].fillna(0).sum(axis=1)
    raw["emd"] = raw["emd_nm"].map(shorten_name)
    raw["base_date"] = raw["base_date"].astype(str)
    raw["hour"] = pd.to_numeric(raw["time"], errors="coerce").fillna(-1).astype(int)
    has_resd = "resd" in raw.columns
    raw["is_local"] = raw["resd"].astype(str).str.strip().eq("제주") if has_resd else True
    local = raw[raw["is_local"]]
    outside = raw[~raw["is_local"]]
    keys = ["emd", "base_date", "hour"]
    grouped = local.groupby(keys, as_index=False).agg(
        resident=("presence", "sum"), resd_pop=("resd_pop_cnt", "sum"), work_pop=("work_pop_cnt", "sum"),
        visit_pop=("visit_pop_cnt", "sum"), masked=("masked", "sum"))
    outsiders = outside.groupby(keys, as_index=False).agg(outsider=("presence", "sum"))
    cube = grouped.merge(outsiders, on=keys, how="left")
    cube["outsider"] = cube["outsider"].fillna(0.0)
    cube["total"] = cube["resident"] + cube["outsider"]
    regions = load_regions(stamp)
    from src.tools.synthetic_seed import pick_observatory  # 지점-관측소 최근접 매칭
    observatories = fetch_frame("SELECT observatory_code, latitude AS lat, longitude AS lon FROM weather_stations")
    obs_rows = observatories.to_dict("records")
    region_obs = {row.emd_nm: pick_observatory(row.lat, row.lon, obs_rows) for row in regions.itertuples()}
    cube = attach_flags(cube, load_calendar(stamp), load_weather(stamp), region_obs, "emd")
    cube.attrs["has_resd"] = has_resd
    return cube


@lru_cache(maxsize=2)
def load_bus(stamp: float) -> pd.DataFrame:
    """정류소×일×시간: 승차·하차, 어르신·청소년 승차."""
    raw = fetch_frame("SELECT base_date, base_hour, bus_station_id, get_on_off_type, user_category, user_count "
                      "FROM bus_stop_hourly")
    raw["user_count"] = pd.to_numeric(raw["user_count"], errors="coerce")
    raw["masked"] = raw["user_count"].isna().astype(int)
    raw["user_count"] = raw["user_count"].fillna(0)
    raw["base_date"] = raw["base_date"].astype(str)
    raw["bus_station_id"] = raw["bus_station_id"].astype(str)
    raw["hour"] = pd.to_numeric(raw["base_hour"], errors="coerce").fillna(-1).astype(int)
    raw["is_board"] = raw["get_on_off_type"].astype(str).str.contains("승")
    category = raw["user_category"].astype(str)
    raw["board"] = np.where(raw["is_board"], raw["user_count"], 0)
    raw["alight"] = np.where(~raw["is_board"], raw["user_count"], 0)
    raw["board_elder"] = np.where(raw["is_board"] & category.str.contains("어르신|경로|노인"), raw["user_count"], 0)
    raw["board_youth"] = np.where(raw["is_board"] & category.str.contains("청소년|학생"), raw["user_count"], 0)
    cube = raw.groupby(["bus_station_id", "base_date", "hour"], as_index=False).agg(
        board=("board", "sum"), alight=("alight", "sum"), board_elder=("board_elder", "sum"),
        board_youth=("board_youth", "sum"), masked=("masked", "sum"))
    cube["volume"] = cube["board"] + cube["alight"]
    stations = load_stations(stamp)
    obs = dict(zip(stations["bus_station_id"].astype(str), stations["observatory_code"].astype(str)))
    return attach_flags(cube, load_calendar(stamp), load_weather(stamp), obs, "bus_station_id")


@lru_cache(maxsize=2)
def load_service(stamp: float) -> pd.DataFrame:
    frame = fetch_frame("SELECT bus_station_id, day_type, base_hour, buses_per_hour FROM bus_service_hourly")
    frame["bus_station_id"] = frame["bus_station_id"].astype(str)
    frame["hour"] = pd.to_numeric(frame["base_hour"], errors="coerce").astype(int)
    frame["buses_per_hour"] = pd.to_numeric(frame["buses_per_hour"], errors="coerce").fillna(0.0)
    return frame


def map_legal_dongs(pairs: pd.DataFrame, stamp: float) -> pd.DataFrame:
    """(법정동, 권역) 쌍마다 행정동과 배분 가중치를 정한다.

    한 법정동이 여러 행정동에 걸치면(예: 서귀동 -> 송산·정방·중앙·천지동, 월평동 -> 제주시 아라동·서귀포시 대천동)
    카드 행의 권역(dstrct_se) -> 시(제주시/서귀포시) 순으로 후보를 좁힌다. 남은 후보 중 주 관할(high)이 있으면 그쪽에,
    없으면 후보 행정동에 균등 배분한다. 면적·인구 가중치는 원자료에 없어 쓰지 않는다.
    """
    regions = load_regions(stamp)[["emd_nm", "city", "dstrct_se"]]
    candidates = load_legal_map(stamp).merge(regions, on="emd_nm", how="left")
    candidates["bjd"] = candidates["bjd"].astype(str).str.strip()
    links = []
    for bjd, district in pairs.itertuples(index=False):
        pool = candidates[candidates["bjd"] == bjd]
        if pool.empty:
            links.append((bjd, district, bjd, 1.0))
            continue
        narrowed = pool[pool["dstrct_se"] == district]
        if narrowed.empty:
            narrowed = pool[pool["city"] == district.split(" ")[0]]
        if narrowed.empty:
            narrowed = pool
        primary = narrowed[narrowed["mapping_confidence"] == "high"]
        names = (primary if not primary.empty else narrowed)["emd_nm"].drop_duplicates().tolist()
        links.extend((bjd, district, name, 1.0 / len(names)) for name in names)
    return pd.DataFrame(links, columns=["bjd", "dstrct_se", "emd", "weight"])


@lru_cache(maxsize=2)
def load_card(stamp: float) -> pd.DataFrame:
    """법정동(bjd)을 행정동으로 묶은 월×이용자구분×시간대 카드 이용금액(배분 규칙은 map_legal_dongs)."""
    raw = fetch_frame("SELECT base_year_month, dstrct_se, bjd, user_type, time, use_amt FROM card_time_monthly")
    raw["use_amt"] = pd.to_numeric(raw["use_amt"], errors="coerce").fillna(0)
    raw["bjd"] = raw["bjd"].astype(str).str.strip()
    raw["dstrct_se"] = raw["dstrct_se"].fillna("").astype(str).str.strip()
    links = map_legal_dongs(raw[["bjd", "dstrct_se"]].drop_duplicates(), stamp)
    card = raw.merge(links, on=["bjd", "dstrct_se"], how="left")
    card["use_amt"] = card["use_amt"] * card["weight"]
    card["split_amt"] = card["use_amt"].where(card["weight"] < 1, 0.0)  # 균등 배분으로 받은 금액
    card["is_resident"] = card["user_type"].astype(str).str.contains("도민")
    keys = ["emd", "base_year_month", "is_resident", "time"]
    return card.groupby(keys, as_index=False)[["use_amt", "split_amt"]].sum()


@lru_cache(maxsize=2)
def load_food(stamp: float) -> pd.DataFrame:
    frame = fetch_frame("SELECT base_year_month, emd_nm, discharge_daily_avg, collection_scope, source_type "
                        "FROM food_waste_monthly")
    frame["discharge_daily_avg"] = pd.to_numeric(frame["discharge_daily_avg"], errors="coerce")
    frame["base_year_month"] = frame["base_year_month"].astype(str)
    return frame


@lru_cache(maxsize=2)
def load_wifi(stamp: float) -> pd.DataFrame:
    frame = fetch_frame("SELECT base_year_month, visit_type, user_count FROM wifi_monthly_users")
    frame["user_count"] = pd.to_numeric(frame["user_count"], errors="coerce")
    frame["base_year_month"] = frame["base_year_month"].astype(str)
    return frame


@lru_cache(maxsize=2)
def load_ev(stamp: float) -> pd.DataFrame:
    return fetch_frame("SELECT * FROM ev_charging_wait")


@lru_cache(maxsize=2)
def load_meta(stamp: float) -> dict:
    return {
        "changes": fetch_frame("SELECT * FROM aggregation_changes").to_dict("records"),
        "limitations": fetch_frame("SELECT * FROM limitation_notes").to_dict("records"),
        "columns": fetch_frame("SELECT * FROM column_registry").to_dict("records"),
        "cases": fetch_frame("SELECT * FROM reference_cases").to_dict("records"),
    }


def warm_cubes() -> dict:
    """앱 시작 시 한 번 호출해 첫 질문 지연을 없앤다."""
    stamp = stamp_db()
    sizes = {
        "floating": len(load_floating(stamp)), "bus": len(load_bus(stamp)), "service": len(load_service(stamp)),
        "card": len(load_card(stamp)), "food": len(load_food(stamp)), "wifi": len(load_wifi(stamp)),
    }
    load_meta(stamp)
    load_registry(stamp)
    return sizes
