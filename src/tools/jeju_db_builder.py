"""4대 축 결합 로컬 DB(data/processed/jeju_truelife.db) 구축기.

적재 순서
  1) 공식 예시데이터 70개 시트 전부 -> bdc_<시트코드> 기준 테이블(원본 컬럼명·표기 그대로, TEXT 보존)
  2) 제주데이터허브 데이터목록 1,259건 -> datahub_catalog
  3) 지명 사전 -> regions / legal_dong_map / region_aliases / bus_stations / landmarks / weather_stations
  4) 분석 테이블: data/raw/*.csv 원자료가 있으면 최우선 적재(official_raw), 없으면 공식 스키마를 보존한
     시드 고정 확장 테이블(synthetic_seed). 사용자 제공 공개 통계 사례는 calibrated_reference 로 표기.
  5) 메타·집계변경 이력 축: table_registry, column_profile, column_registry, aggregation_changes,
     limitation_notes, reference_cases, build_log
  6) 조회 인덱스 생성
모든 분석 테이블 행에 source_type, dataset_id, reference_date 출처 태그를 남긴다.
"""
from __future__ import annotations

import json
import os
import platform
import re
import runpy
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (CATALOG_SCHEMA_PATH, DB_PATH, GAZETTEER_PATH, RAIN_DAY_MM, RAW_DIR, ROOT, SAMPLES_DIR,
                        SEED)
from src.tools import synthetic_seed as seed
from src.tools.db_connector import open_db
from src.tools.resilient_loader import load_any_csv, parse_number

CATALOG_DATE = "2026-11-05"
EXAMPLE_DATE = "2026-11-15"

TABLE_SPECS = {
    "floating_pop_hourly": {
        "axis": "① 유동·생활인구/소비", "official_sheet": "4-11 내국인 유동인구(읍면동)_v02", "dataset_id": "DH-753",
        "public_source": "제주데이터허브 #753 읍면동 단위 내국인 유동인구(API, 2017-01-01~ 일 단위 갱신)",
        "spatial_key": "행정동(emd_cd·emd_nm)", "time_grain": "일×시간",
        "required": ["base_date", "time", "emd_nm", "resd_pop_cnt", "work_pop_cnt", "visit_pop_cnt"],
    },
    "bus_stop_hourly": {
        "axis": "② 대중교통·이동", "official_sheet": "3-04 버스 정류소별 이용 현황", "dataset_id": "DH-725",
        "public_source": "제주데이터허브 #725 일일 정류소별 버스 이용자 정보(API) · #1249 이용자 유형별 버스정류소 이용 인원 현황(File)",
        "spatial_key": "정류소(bus_station_id)", "time_grain": "일×시간",
        "required": ["base_date", "base_hour", "bus_station_id", "get_on_off_type", "user_count"],
    },
    "bus_service_hourly": {
        "axis": "② 대중교통·이동", "official_sheet": "(직접 대응 시트 없음: 3-01 버스 노선 정보 구조 참고)",
        "dataset_id": "SYN-SERVICE",
        "public_source": "제주데이터허브 #727 버스 노선 기본 정보(API). 시간대별 배차 원자료는 데이터목록에 없어 합성",
        "spatial_key": "정류소(bus_station_id)", "time_grain": "평일/주말×시간", "required": [],
    },
    "card_time_monthly": {
        "axis": "① 유동·생활인구/소비", "official_sheet": "1-14 카드 이용 현황(시간대)_v02", "dataset_id": "BDC-1-14",
        "public_source": "제주빅데이터센터 방문 제공 데이터(카드 이용 현황 시간대별)",
        "spatial_key": "법정동(bjd)", "time_grain": "월×시간대",
        "required": ["base_year_month", "bjd", "user_type", "time", "use_amt"],
    },
    "card_week_monthly": {
        "axis": "① 유동·생활인구/소비", "official_sheet": "1-12 카드 이용 현황(요일별)_v02", "dataset_id": "BDC-1-12",
        "public_source": "제주빅데이터센터 방문 제공 데이터(카드 이용 현황 요일별)",
        "spatial_key": "법정동(bjd)", "time_grain": "월×요일",
        "required": ["base_year_month", "bjd", "user_type", "week", "use_amt"],
    },
    "weather_daily": {
        "axis": "③ 기상·환경", "official_sheet": "5-02 일일 날씨 정보", "dataset_id": "DH-756",
        "public_source": "제주데이터허브 #756 일별 기상 정보(API, 2014-01-01~)",
        "spatial_key": "관측소(observatory_code)", "time_grain": "일",
        "required": ["base_date", "observatory_code", "daily_rainfall"],
    },
    "food_waste_monthly": {
        "axis": "③ 기상·환경", "official_sheet": "5-05 음식물쓰레기 배출현황_제주시 / 5-06 _서귀포(월·읍면동 집계)",
        "dataset_id": "DH-515",
        "public_source": "제주데이터허브 #515 유동인구에 따른 음식물 쓰레기 배출량 예측 매쉬업결과(2018-01~2021-05) + 사용자 제공 한림읍 공개 통계 사례",
        "spatial_key": "행정동(emd_nm)", "time_grain": "월",
        "required": ["base_year_month", "emd_nm", "discharge_daily_avg"],
    },
    "wifi_monthly_users": {
        "axis": "① 유동·생활인구/소비", "official_sheet": "4-06 와이파이 일별 이용자 현황(월 집계)", "dataset_id": "DH-749",
        "public_source": "제주데이터허브 #749 와이파이 월간 사용자수(API) · #742 와이파이 사용자 이동 경로(수집 2017-07-01~2020-10-28)",
        "spatial_key": "제주 전역", "time_grain": "월",
        "required": ["base_year_month", "visit_type", "user_count"],
    },
    "ev_charging_wait": {
        "axis": "③ 기상·환경", "official_sheet": "(예시 시트 없음)", "dataset_id": "DH-136",
        "public_source": "제주데이터허브 #136 일별 전기자동차 충전 이력(API, 2021-01-01~2022-06-30) + 사용자 제공 사례 구조(2,447행·미충전 154행·시각 유효 113행)",
        "spatial_key": "행정동(emd_nm)", "time_grain": "건(세션)",
        "required": ["charger_id", "start_time", "charge_kwh"],
    },
    "market_days": {
        "axis": "② 대중교통·이동", "official_sheet": "(해당 없음)", "dataset_id": "RULE-MARKET",
        "public_source": "제주민속오일시장 개장 규칙(매월 2·7일)", "spatial_key": "도두동", "time_grain": "일",
        "required": [],
    },
}

COLUMN_LABELS = {
    "floating_pop_hourly": {
        "resd_pop_cnt": ("거주 인구 수", "resident", "명"), "work_pop_cnt": ("직장 인구 수", "worker,commute", "명"),
        "visit_pop_cnt": ("방문 인구 수", "visitor", "명"), "resd": ("거주지 구분(제주/그외)", "tourist,resident", ""),
        "time": ("시간(0~23시)", "hour", "시"), "base_date": ("기준일자", "date", ""),
        "emd_nm": ("읍면동명", "region", ""), "sex": ("성별", "sex", ""), "age": ("연령대", "age", ""),
    },
    "bus_stop_hourly": {
        "user_count": ("이용자 수", "bus_usage", "명"), "user_category": ("이용자 유형(일반/청소년/어르신)", "age,elderly,student", ""),
        "get_on_off_type": ("승하차 구분", "bus_direction", ""), "base_hour": ("시간", "hour", "시"),
        "base_date": ("기준일자", "date", ""), "day_of_week": ("요일", "weekday", ""), "pay_type": ("결제 유형", "payment", ""),
    },
    "bus_service_hourly": {
        "buses_per_hour": ("시간당 정차 버스 수(합성)", "bus_supply", "대/시"), "day_type": ("평일/주말", "weekday", ""),
    },
    "card_time_monthly": {
        "use_amt": ("이용 금액", "sales", "원"), "use_cnt": ("이용 건수", "sales", "건"), "user_cnt": ("이용자 수", "sales", "명"),
        "user_type": ("이용자 구분(제주도민/관광객)", "tourist,resident", ""), "time": ("시간대", "hour", ""),
        "biz_nm": ("업종명", "industry", ""), "bjd": ("법정동", "region", ""),
    },
    "card_week_monthly": {
        "use_amt": ("이용 금액", "sales", "원"), "use_cnt": ("이용 건수", "sales", "건"),
        "user_type": ("이용자 구분(제주도민/관광객)", "tourist,resident", ""), "week": ("요일", "weekday", ""),
    },
    "weather_daily": {
        "daily_rainfall": ("일 강수량", "rain", "mm"), "average_temperature": ("일 평균기온", "temperature", "℃"),
        "average_wind_speed": ("일 평균풍속", "wind", "m/s"), "maximum_wind_speed": ("일 최대풍속", "wind,typhoon", "m/s"),
    },
    "food_waste_monthly": {
        "discharge_daily_avg": ("RFID 계량 배출량 일평균", "food_waste", "원문 단위"),
        "collection_scope": ("계량 범위", "collection_method", ""),
    },
    "wifi_monthly_users": {
        "user_count": ("와이파이 이용자 수", "wifi_usage", "명"), "visit_type": ("방문 유형(관광/거주)", "tourist,resident", ""),
        "agg_unit": ("집계 단위(AP/장소)", "collection_method", ""),
    },
    "ev_charging_wait": {
        "request_time": ("충전 요청 시각", "ev_wait", ""), "start_time": ("충전 시작 시각", "ev_charging", ""),
        "charge_kwh": ("충전량", "ev_charging", "kWh"), "status": ("충전 상태", "ev_charging", ""),
    },
}

LIMITATIONS = [
    ("weather_daily", "강수량이 일 단위라 특정 시각(예: 18시 퇴근길)의 비 여부를 알 수 없음",
     "퇴근 시간대(18~19시)에 실제로 비가 내렸는지 해당 일자의 시간별 관측값으로 확인할 수 있을까요?"),
    ("floating_pop_hourly", "통신 기반 추정 인구로 거주 만족도·치안·소음 같은 생활 품질은 측정하지 않음",
     "이 동네에 실제 사는 주민께 밤 9시 이후 귀갓길이 불편한지 직접 여쭤볼 수 있을까요?"),
    ("floating_pop_hourly", "합성 확장 테이블은 성·연령을 '전체'로 집계해 세부 연령 분석 불가(원자료 적재 시 해제)",
     "어르신·청소년 등 특정 연령대가 실제로 붐비는 시간대를 현장 관찰로 확인할 수 있을까요?"),
    ("bus_stop_hourly", "정류소 승하차만 있고 차내 혼잡(재차 인원)·실제 배차 지연은 없음",
     "해당 정류소에서 퇴근 시간대 버스가 만차로 통과한 적이 있는지 기사님이나 이용자께 확인할 수 있을까요?"),
    ("bus_stop_hourly", "원자료는 소수 인원을 '*'로 마스킹하므로 적재 시 하한 추정이 필요",
     "마스킹된 시간대의 실제 이용 인원을 교통카드 원천 집계로 확인할 수 있을까요?"),
    ("bus_service_hourly", "시간대별 배차는 데이터목록에 원자료가 없어 합성값이며 실제 시간표와 다를 수 있음",
     "해당 노선의 실제 막차 시각과 배차 간격을 버스정보시스템 시간표로 확인할 수 있을까요?"),
    ("card_time_monthly", "카드 결제만 포함(현금·일부 간편결제 누락)이며 월 단위라 일별 날씨와 직접 연결 불가",
     "비 오는 날 매출이 실제로 줄었는지 매장 일별 POS 매출로 확인할 수 있을까요?"),
    ("food_waste_monthly", "RFID 계량 범위(설치 배출함 수) 변화가 배출량 변화처럼 보일 수 있고 관광객 수 컬럼이 없음",
     "2019년 11월 전후로 한림읍에 RFID 배출함이 몇 대 새로 설치됐는지 읍사무소 기록으로 확인할 수 있을까요?"),
    ("wifi_monthly_users", "2020년 10월 집계 단위(AP->장소) 전환 전후 수치를 직접 비교할 수 없음",
     "2020년 10월 전환 시점에 같은 기간을 두 방식으로 함께 집계한 병행 자료가 있는지 운영기관에 확인할 수 있을까요?"),
    ("ev_charging_wait", "대기시간은 요청·시작 시각이 모두 유효한 113행에서만 계산 가능(전체 2,447행의 4.6%)",
     "충전 요청 시각이 비어 있는 행이 앱 미사용 현장 충전인지 운영사에 확인할 수 있을까요?"),
]

AGGREGATION_CHANGES = [
    ("food_waste_monthly", "한림읍", "201911", "수집방식 전환", "부분 RFID 계량(일부 배출함)", "RFID 전면 계량",
     "사용자 제공 공개 통계 검증 사례: 2019-10 일평균 1,045.23 -> 2019-12 일평균 61,346.42(58.69배), 2019년 11월 RFID 전환",
     "user_provided_public_stat"),
    ("wifi_monthly_users", "", "202010", "집계단위 전환", "AP 단위 접속 집계", "장소 단위 집계",
     "사용자 제공 공개 통계 사례 + 데이터허브 목록 #742 와이파이 이동 경로 수집기간이 2020-10-28에 종료",
     "user_provided_public_stat"),
    ("card_time_monthly", "", "v02", "스키마 개정", "v01: 권역 컬럼 없음", "v02: dstrct_se(권역) 컬럼 추가",
     "공식 예시데이터 시트 1-07(v01)과 1-08(v02), 1-13(v01)과 1-14(v02) 컬럼 비교", "official_example"),
    ("floating_pop_hourly", "", "v02", "스키마 개정", "v01: 권역 컬럼 없음", "v02: dstrct_se(권역) 컬럼 추가",
     "공식 예시데이터 시트 4-10(v01)과 4-11(v02) 컬럼 비교", "official_example"),
]

REFERENCE_CASES = [
    ("HALLIM_RFID_2019", "한림읍 음식물쓰레기 RFID 전환 집계 단절",
     {"oct_2019_daily_avg": 1045.23, "dec_2019_daily_avg": 61346.42, "ratio": 58.69, "change_month": "2019-11"},
     "food_waste_monthly", "사용자 제공 공개 통계 수치. 로컬 표본(hallim_food_waste_daily_cp949.csv)은 이 수치에 맞춰 보정"),
    ("WIFI_UNIT_2020_10", "공공 와이파이 AP->장소 단위 집계 전환",
     {"change_month": "2020-10", "before": "AP", "after": "장소"},
     "wifi_monthly_users", "사용자 제공 사례 + 데이터허브 목록 #742 수집기간 종료일(2020-10-28)로 정황 확인. 전환 폭은 합성"),
    ("MARKET_455", "오일장 455번 버스 시간표 및 정류소 이용객 검증",
     {"route": "455", "stop": "제주민속오일장", "stop_id_official_example": "96"},
     "bus_stop_hourly", "사용자 제공 사례. 실제 시간표 수치는 미확보로 배차표는 합성(bus_service_hourly)"),
    ("EV_WAIT_2447", "전기차 충전 대기 산출 가능 행 부족",
     {"rows": 2447, "not_charged": 154, "valid_time_rows": 113},
     "ev_charging_wait", "사용자 제공 사례 구조를 재현한 합성 표본(ev_charging_wait_2447.csv)"),
]


# ---------------------------------------------------------------- 입력 읽기

def load_gazetteer() -> dict:
    if not GAZETTEER_PATH.exists():
        builder = runpy.run_path(str(ROOT / "scripts" / "build_gazetteer.py"))
        gazetteer = builder["build_gazetteer"]()
        GAZETTEER_PATH.parent.mkdir(parents=True, exist_ok=True)
        GAZETTEER_PATH.write_text(json.dumps(gazetteer, ensure_ascii=False, indent=1), encoding="utf-8")
    return json.loads(GAZETTEER_PATH.read_text(encoding="utf-8"))


def load_catalog_schema() -> dict:
    if CATALOG_SCHEMA_PATH.exists():
        return json.loads(CATALOG_SCHEMA_PATH.read_text(encoding="utf-8"))
    return {"datahub_catalog": {"datasets": []}, "center_examples": []}


def name_example_table(code: str) -> str:
    return "bdc_" + re.sub(r"[^0-9A-Za-z]+", "_", code).strip("_")


# ---------------------------------------------------------------- 적재 함수

def tag_rows(frame: pd.DataFrame, source_type: str, dataset_id: str, reference_date: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["source_type"] = source_type
    frame["dataset_id"] = dataset_id
    frame["reference_date"] = reference_date
    return frame


def write_examples(conn, schema: dict) -> list[dict]:
    registry = []
    for sheet in schema.get("center_examples", []):
        table = name_example_table(sheet["code"] or sheet["sheet"])
        columns = sheet["columns"]
        rows = sheet.get("example_rows", [])
        frame = pd.DataFrame(rows, columns=columns).astype(str) if rows else pd.DataFrame(columns=columns)
        frame.insert(0, "official_sheet", sheet["sheet"])
        frame = tag_rows(frame, "official_example", f"BDC-{sheet['code']}", EXAMPLE_DATE)
        frame.to_sql(table, conn, if_exists="replace", index=False)
        registry.append({
            "table_name": table, "axis": "공식 원본(예시데이터)", "official_sheet": sheet["sheet"],
            "public_source": "제주빅데이터센터_예시데이터 및 컬럼확인용.xlsx(DAKER 대회 제공, 2026-11-15)",
            "source_type": "official_example", "dataset_id": f"BDC-{sheet['code']}", "reference_date": EXAMPLE_DATE,
            "expansion_method": "확장 없음(원본 행·컬럼·표기 그대로 TEXT 보존)", "row_count": len(frame),
            "spatial_key": "", "time_grain": "", "period": "",
        })
    return registry


def write_catalog(conn, schema: dict) -> dict:
    datasets = schema.get("datahub_catalog", {}).get("datasets", [])
    frame = pd.DataFrame(datasets)
    if frame.empty:
        frame = pd.DataFrame(columns=["no", "category", "title", "kind", "registered", "updated", "period"])
    frame = frame.rename(columns={"등록자": "registrant"})
    frame = tag_rows(frame, "official_catalog", "DH-CATALOG-20261105", CATALOG_DATE)
    frame.to_sql("datahub_catalog", conn, if_exists="replace", index=False)
    return {
        "table_name": "datahub_catalog", "axis": "④ 메타데이터·집계변경 이력", "official_sheet": "반출내역(데이터목록)",
        "public_source": "제주데이터허브 데이터목록_20261105.xlsx(DAKER 대회 제공)", "source_type": "official_catalog",
        "dataset_id": "DH-CATALOG-20261105", "reference_date": CATALOG_DATE,
        "expansion_method": "확장 없음(1,259건 전량 적재)", "row_count": len(frame),
        "spatial_key": "", "time_grain": "", "period": "",
    }


def write_spatial(conn, gazetteer: dict, schema: dict) -> list[dict]:
    regions = pd.DataFrame(gazetteer["regions"])
    regions_out = regions.drop(columns=["legal_dongs"]).copy()
    regions_out.to_sql("regions", conn, if_exists="replace", index=False)

    legal = [{"bjd": bjd, "emd_nm": row["emd_nm"], "mapping_confidence": row["mapping_confidence"]}
             for row in gazetteer["regions"] for bjd in row["legal_dongs"]]
    pd.DataFrame(legal).to_sql("legal_dong_map", conn, if_exists="replace", index=False)

    aliases = [{"alias": a["alias"], "emd_nm": target, "kind": a["kind"], "partner_hint": a["partner_hint"],
                "note": a["note"]} for a in gazetteer["aliases"] for target in a["targets"]]
    pd.DataFrame(aliases).to_sql("region_aliases", conn, if_exists="replace", index=False)
    pd.DataFrame(gazetteer["landmarks"]).to_sql("landmarks", conn, if_exists="replace", index=False)

    observatories = gazetteer["weather_stations"]
    stations = pd.DataFrame(gazetteer["bus_stations"])
    stations["observatory_code"] = [seed.pick_observatory(r.lat, r.lon, observatories) for r in stations.itertuples()]
    stations_out = pd.DataFrame({
        "serial_number": range(1, len(stations) + 1),
        "base_date": "20261101",
        "bus_station_id": stations["bus_station_id"],
        "bus_station_name": stations["bus_station_name"],
        "bus_city_type": stations["bus_city_type"],
        "longitude": stations["lon"],
        "latitude": stations["lat"],
        "emd_nm": stations["emd_nm"],
        "observatory_code": stations["observatory_code"],
        "id_status": stations["id_status"],
        "coord_status": stations["coord_status"],
    })
    stations_out = tag_rows(stations_out, "gazetteer", "BDC-3-03", "2026-11-16")
    stations_out.to_sql("bus_stations", conn, if_exists="replace", index=False)

    example_obs = {}
    for sheet in schema.get("center_examples", []):
        if sheet["code"] == "5-01" and sheet.get("example_rows"):
            example_obs = sheet["example_rows"][0]
    obs_rows = []
    for obs in observatories:
        official = example_obs if example_obs.get("observatory_code") == obs["observatory_code"] else {}
        obs_rows.append({
            "observatory_code": obs["observatory_code"],
            "observatory_name": obs["observatory_name"],
            "start_date": official.get("start_date", ""),
            "end_date": official.get("end_date", ""),
            "management_office": official.get("management_office", "제주지방기상청(184)"),
            "longitude": float(official.get("longitude", obs["lon"])),
            "latitude": float(official.get("latitude", obs["lat"])),
            "legal_dongri_code": official.get("legal_dongri_code", ""),
            "sea_level_altitude": parse_number(official.get("sea_level_altitude", "")),
            "collect_date": official.get("collect_date", ""),
            "coord_status": obs["coord_status"],
        })
    weather_stations = tag_rows(pd.DataFrame(obs_rows), "gazetteer", "DH-757", "2026-11-16")
    weather_stations.to_sql("weather_stations", conn, if_exists="replace", index=False)

    return [
        {"table_name": "regions", "axis": "공간 키", "official_sheet": "4-11 emd_cd/emd_nm 형식 준용",
         "public_source": "행정기관코드 체계(공식 예시 확인 2건) + 근사 좌표", "source_type": "gazetteer",
         "dataset_id": "GAZ-REGIONS", "reference_date": "2026-11-16", "expansion_method": "큐레이션 사전",
         "row_count": len(regions_out), "spatial_key": "행정동", "time_grain": "", "period": ""},
        {"table_name": "legal_dong_map", "axis": "공간 키", "official_sheet": "1-08 bjd ↔ 4-11 emd_nm 연결",
         "public_source": "법정동-행정동 관할(일부 review 표시)", "source_type": "gazetteer", "dataset_id": "GAZ-LEGAL",
         "reference_date": "2026-11-16", "expansion_method": "큐레이션 사전", "row_count": len(legal),
         "spatial_key": "법정동→행정동", "time_grain": "", "period": ""},
        {"table_name": "region_aliases", "axis": "공간 키", "official_sheet": "(해당 없음)",
         "public_source": "생활권·지명 별칭 사전", "source_type": "gazetteer", "dataset_id": "GAZ-ALIASES",
         "reference_date": "2026-11-16", "expansion_method": "큐레이션 사전", "row_count": len(aliases),
         "spatial_key": "별칭→행정동", "time_grain": "", "period": ""},
        {"table_name": "bus_stations", "axis": "② 대중교통·이동", "official_sheet": "3-03 버스 정류소 정보",
         "public_source": "제주데이터허브 #726 버스 정류소 기본 정보(API). ID 1·96은 공식 예시값", "source_type": "gazetteer",
         "dataset_id": "BDC-3-03", "reference_date": "2026-11-16", "expansion_method": "주요 정류소 24곳 큐레이션",
         "row_count": len(stations_out), "spatial_key": "정류소", "time_grain": "", "period": ""},
        {"table_name": "weather_stations", "axis": "③ 기상·환경", "official_sheet": "5-01 날씨 관측소 정보",
         "public_source": "제주데이터허브 #757 기상관측소 정보(API). 중문(328)은 공식 예시값", "source_type": "gazetteer",
         "dataset_id": "DH-757", "reference_date": "2026-11-16", "expansion_method": "관측소 5곳",
         "row_count": len(weather_stations), "spatial_key": "관측소", "time_grain": "", "period": ""},
    ]


def read_raw_tables(raw_dir: Path, log) -> dict[str, tuple[pd.DataFrame, str]]:
    """data/raw/*.csv 중 공식 컬럼 서명이 맞는 파일을 분석 테이블 원자료로 채택한다."""
    adopted: dict[str, tuple[pd.DataFrame, str]] = {}
    if not raw_dir.exists():
        return adopted
    for path in sorted(raw_dir.glob("*.csv")):
        table = load_any_csv(path)
        if not table.ok or table.frame.empty:
            log(f"[원자료] {path.name}: 읽기 실패 또는 빈 파일 -> 건너뜀 ({'; '.join(table.notes)})")
            continue
        columns = set(table.frame.columns)
        for name, spec in TABLE_SPECS.items():
            required = spec["required"]
            if required and set(required) <= columns and name not in adopted:
                adopted[name] = (table.frame, path.name)
                log(f"[원자료] {path.name} -> {name} 최우선 적재({len(table.frame):,}행, 인코딩 {table.encoding})")
                break
    return adopted


def profile_columns(table: str, frame: pd.DataFrame, anchors: dict[str, str], anchor_sheet: str) -> list[dict]:
    rows = []
    for column in frame.columns:
        if column in ("source_type", "dataset_id", "reference_date"):
            continue
        series = pd.to_numeric(frame[column], errors="coerce") if frame[column].dtype != object else None
        if series is None or series.notna().sum() == 0:
            continue
        anchor_raw = anchors.get(column, "")
        anchor_value = parse_number(anchor_raw) if anchor_raw else None
        rows.append({
            "table_name": table, "column_name": column,
            "mean": round(float(series.mean()), 4), "std": round(float(series.std(ddof=0)), 4),
            "min": round(float(series.min()), 4), "max": round(float(series.max()), 4),
            "anchor_sheet": anchor_sheet,
            "anchor_value": anchor_value,
            "anchor_note": ("예시 1행 값(표준편차 추정 불가)" if anchor_value is not None
                            else ("예시값 마스킹('*')" if anchor_raw.startswith("*") else "예시값 없음")),
        })
    return rows


def write_samples(hallim_daily: pd.DataFrame, ev: pd.DataFrame, wifi: pd.DataFrame) -> list[str]:
    """탭 3(CSV 감사기) 시연·퍼징용 표본을 일부러 '현장에서 흔한 형식 문제'를 섞어 저장한다."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    hallim = hallim_daily.copy()
    lines = ["기준일자,읍면동,배출량,배출건수"]
    september = hallim.index[hallim["date"].dt.month == 9].tolist()
    masked_rows = set(september[3:5])
    null_row = september[10] if len(september) > 10 else None
    for idx, row in hallim.iterrows():
        amount = "NULL" if idx == null_row else f"\"{row['amount']:,.2f}\""
        count = "***" if idx in masked_rows else str(int(row["count"]))
        lines.append(f"{row['date'].strftime('%Y-%m-%d')},한림읍,{amount},{count}")
    path = SAMPLES_DIR / "hallim_food_waste_daily_cp949.csv"
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("cp949"))
    written.append(path.name)

    ev_out = ev.rename(columns={"charger_id": "충전기ID", "emd_nm": "읍면동", "request_time": "충전요청시각",
                                "start_time": "충전시작시각", "end_time": "충전종료시각", "charge_kwh": "충전량_kWh",
                                "status": "상태"})
    path = SAMPLES_DIR / "ev_charging_wait_2447.csv"
    ev_out.to_csv(path, index=False, encoding="utf-8-sig")
    written.append(path.name)

    wifi_total = wifi.groupby("base_year_month", as_index=False)["user_count"].sum()
    wifi_lines = ["기준년월\t이용자수"] + [f"{r.base_year_month}\t{r.user_count:,}" for r in wifi_total.itertuples()]
    path = SAMPLES_DIR / "wifi_monthly_users_tab_euckr.csv"
    path.write_bytes(("\n".join(wifi_lines) + "\n").encode("euc-kr"))
    written.append(path.name)

    routes = [("10000", "2-1", "2-1(월평마을~월평마을)", 55), ("10001", "3-2", "3-2(제주터미널~삼양)", 41),
              ("10002", "4-11", "4-11(노형~화북)", 38), ("10003", "455", "455(제주공항~오일장~노형)", 33),
              ("10004", "1-3", "1-3(시청~아라)", 27)]
    route_lines = ["serial_number,base_date,bus_route_id,bus_route_number,bus_route_name,start_date,bus_station_count,"
                   "bus_route_length,bus_transferability"]
    for i, (rid, number, label, count) in enumerate(routes, start=1):
        shown = number
        if re.fullmatch(r"\d{1,2}-\d{1,2}", number):
            month, day = number.split("-")
            shown = f"{int(month):02d}월 {int(day):02d}일"
        route_lines.append(f"{i},20261101,{rid},{shown},{label},20150729,{count},0,Y")
    path = SAMPLES_DIR / "bus_route_info_mangled.csv"
    path.write_text("\n".join(route_lines) + "\n", encoding="utf-8")
    written.append(path.name)
    return written


def create_indexes(conn) -> None:
    statements = [
        "CREATE INDEX IF NOT EXISTS ix_floating_emd ON floating_pop_hourly(emd_nm, base_date, time)",
        "CREATE INDEX IF NOT EXISTS ix_floating_resd ON floating_pop_hourly(resd)",
        "CREATE INDEX IF NOT EXISTS ix_bus_station ON bus_stop_hourly(bus_station_id, base_date, base_hour)",
        "CREATE INDEX IF NOT EXISTS ix_bus_type ON bus_stop_hourly(get_on_off_type, user_category)",
        "CREATE INDEX IF NOT EXISTS ix_service_station ON bus_service_hourly(bus_station_id, day_type, base_hour)",
        "CREATE INDEX IF NOT EXISTS ix_card_time ON card_time_monthly(bjd, base_year_month, user_type)",
        "CREATE INDEX IF NOT EXISTS ix_card_week ON card_week_monthly(bjd, base_year_month, user_type)",
        "CREATE INDEX IF NOT EXISTS ix_weather ON weather_daily(observatory_code, base_date)",
        "CREATE INDEX IF NOT EXISTS ix_food ON food_waste_monthly(emd_nm, base_year_month)",
        "CREATE INDEX IF NOT EXISTS ix_wifi ON wifi_monthly_users(base_year_month)",
        "CREATE INDEX IF NOT EXISTS ix_market ON market_days(base_date)",
        "CREATE INDEX IF NOT EXISTS ix_catalog ON datahub_catalog(category)",
        "CREATE INDEX IF NOT EXISTS ix_legal ON legal_dong_map(bjd)",
        "CREATE INDEX IF NOT EXISTS ix_alias ON region_aliases(alias)",
    ]
    for statement in statements:
        try:
            conn.execute(statement)
        except sqlite3.OperationalError:
            continue  # 원자료로 대체된 테이블에 해당 컬럼이 없으면 인덱스만 생략


# ---------------------------------------------------------------- 메인

def build_database(db_path: Path = DB_PATH, raw_dir: Path = RAW_DIR, log=print) -> dict:
    started = time.perf_counter()
    rng = np.random.default_rng(SEED)
    gazetteer = load_gazetteer()
    schema = load_catalog_schema()
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    building = db_path.with_name(db_path.name + ".building")
    if building.exists():
        building.unlink()

    calendar = seed.build_calendar()
    regions = gazetteer["regions"]
    stations = gazetteer["bus_stations"]
    observatories = gazetteer["weather_stations"]
    raw_tables = read_raw_tables(Path(raw_dir), log)
    example_lookup = {s["code"]: s for s in schema.get("center_examples", [])}

    log("기상 테이블 생성")
    weather = seed.synthesize_weather(rng, calendar, observatories)
    region_obs = [seed.pick_observatory(r["lat"], r["lon"], observatories) for r in regions]
    station_obs = [seed.pick_observatory(s["lat"], s["lon"], observatories) for s in stations]
    region_rain = seed.map_rain_days(weather, calendar, region_obs)
    station_rain = seed.map_rain_days(weather, calendar, station_obs)
    log("유동인구 테이블 생성")
    floating = seed.synthesize_floating(rng, calendar, regions, region_rain)
    log("버스 이용·배차 테이블 생성")
    bus, service = seed.synthesize_bus(rng, calendar, stations, station_rain)
    log("카드 소비 테이블 생성")
    card_time, card_week = seed.synthesize_card(rng, regions)
    log("생활환경·집계변경 사례 테이블 생성")
    hallim_daily = seed.build_hallim_daily(rng)
    food = seed.synthesize_food_waste(rng, regions, hallim_daily)
    wifi = seed.synthesize_wifi(rng)
    ev = seed.synthesize_ev_charging(rng, regions)
    market = seed.build_market_days(calendar)

    analysis = {
        "floating_pop_hourly": (floating, "synthetic_seed"),
        "bus_stop_hourly": (bus, "synthetic_seed"),
        "bus_service_hourly": (service, "synthetic_seed"),
        "card_time_monthly": (card_time, "synthetic_seed"),
        "card_week_monthly": (card_week, "synthetic_seed"),
        "weather_daily": (weather, "synthetic_seed"),
        "food_waste_monthly": (food, "calibrated_reference"),
        "wifi_monthly_users": (wifi, "synthetic_seed"),
        "ev_charging_wait": (ev, "calibrated_reference"),
        "market_days": (market, "public_rule"),
    }
    raw_files = {}
    for name, (frame, filename) in raw_tables.items():
        analysis[name] = (frame, "official_raw")
        raw_files[name] = filename

    registry, profiles, column_rows = [], [], []
    with open_db(building) as conn:
        registry += write_examples(conn, schema)
        registry.append(write_catalog(conn, schema))
        registry += write_spatial(conn, gazetteer, schema)
        for name, (frame, source_type) in analysis.items():
            spec = TABLE_SPECS[name]
            reference = datetime.now().strftime("%Y-%m-%d") if source_type == "official_raw" else CATALOG_DATE
            tagged = tag_rows(frame, source_type, spec["dataset_id"], reference)
            if name == "food_waste_monthly" and source_type == "calibrated_reference":
                tagged.loc[tagged["emd_nm"] != "한림읍", "source_type"] = "synthetic_seed"
                source_type = "synthetic_seed+calibrated_reference"
            tagged.to_sql(name, conn, if_exists="replace", index=False, chunksize=50000)
            sheet_code = spec["official_sheet"].split()[0]
            anchors = example_lookup.get(sheet_code, {}).get("example_rows", [{}])
            anchors = anchors[0] if anchors else {}
            profiles += profile_columns(name, frame, anchors, spec["official_sheet"])
            period = ""
            for key in ("base_date", "base_year_month"):
                if key in frame.columns and len(frame):
                    period = f"{frame[key].min()}~{frame[key].max()}"
                    break
            method = {
                "synthetic_seed": f"공식 예시 시트 컬럼·코드값 보존 + 시드({SEED}) 고정 확장. 수치 분포는 column_profile 공개",
                "calibrated_reference": "사용자 제공 공개 통계 사례 수치에 맞춰 보정한 시드 고정 표본",
                "synthetic_seed+calibrated_reference": "한림읍 행은 공개 통계 사례 수치로 보정, 나머지 읍면동은 시드 고정 확장",
                "public_rule": "공개 운영 규칙으로 계산",
                "official_raw": f"원자료 파일 {raw_files.get(name, '')} 적재(자가 치유 로더 정제)",
            }[source_type]
            registry.append({
                "table_name": name, "axis": spec["axis"], "official_sheet": spec["official_sheet"],
                "public_source": spec["public_source"], "source_type": source_type, "dataset_id": spec["dataset_id"],
                "reference_date": reference, "expansion_method": method, "row_count": int(len(frame)),
                "spatial_key": spec["spatial_key"], "time_grain": spec["time_grain"], "period": period,
            })
            for column, (label, concepts, unit) in COLUMN_LABELS.get(name, {}).items():
                column_rows.append({"table_name": name, "column_name": column, "korean_label": label,
                                    "concepts": concepts, "unit": unit, "present": int(column in frame.columns)})

        meta_frames = {
            "column_profile": pd.DataFrame(profiles),
            "column_registry": pd.DataFrame(column_rows),
            "aggregation_changes": pd.DataFrame(AGGREGATION_CHANGES, columns=[
                "table_name", "emd_nm", "change_month", "change_type", "before_desc", "after_desc", "evidence",
                "source_type"]),
            "limitation_notes": pd.DataFrame(LIMITATIONS, columns=["table_name", "limitation", "field_question"]),
            "reference_cases": pd.DataFrame([
                {"case_id": cid, "title": title, "numbers_json": json.dumps(nums, ensure_ascii=False),
                 "linked_table": table, "source_type": "user_provided_public_stat", "verification_status": status}
                for cid, title, nums, table, status in REFERENCE_CASES]),
        }
        for name, frame in meta_frames.items():
            frame.to_sql(name, conn, if_exists="replace", index=False)
            registry.append({
                "table_name": name, "axis": "④ 메타데이터·집계변경 이력", "official_sheet": "(메타)",
                "public_source": "본 프로젝트 정리", "source_type": "meta", "dataset_id": f"META-{name}",
                "reference_date": "2026-11-16", "expansion_method": "", "row_count": len(frame),
                "spatial_key": "", "time_grain": "", "period": "",
            })
        registry_frame = pd.DataFrame(registry)
        registry_frame["seed"] = SEED
        registry_frame.to_sql("table_registry", conn, if_exists="replace", index=False)
        create_indexes(conn)
        seconds = round(time.perf_counter() - started, 1)
        build_rows = [
            ("seed", str(SEED)), ("built_at", datetime.now().isoformat(timespec="seconds")),
            ("build_seconds", str(seconds)), ("python", platform.python_version()), ("pandas", pd.__version__),
            ("rain_day_mm", str(RAIN_DAY_MM)), ("raw_tables", json.dumps(raw_files, ensure_ascii=False)),
        ]
        pd.DataFrame(build_rows, columns=["key", "value"]).to_sql("build_log", conn, if_exists="replace", index=False)

    samples = write_samples(hallim_daily, ev, wifi)
    os.replace(building, db_path)
    total_rows = int(sum(r["row_count"] for r in registry))
    log(f"DB 구축 완료: {db_path.name} 테이블 {len(registry)}개, 총 {total_rows:,}행, {seconds}초")
    return {"tables": len(registry), "rows": total_rows, "seconds": seconds, "samples": samples,
            "raw_tables": raw_files}
