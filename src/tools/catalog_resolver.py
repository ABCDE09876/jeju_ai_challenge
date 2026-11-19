"""Tool 1: resolve_spatial_catalog

질문 의도(주제)에 맞춰 2~4종의 이종 데이터셋을 고르고, 공간 키를 자동 조인한다.
  행정동(유동인구) <-> 법정동(카드 소비) <-> 정류소 반경(버스) <-> 최근접 관측소(기상)
분석 테이블의 컬럼이 공식 예시 시트(bdc_*)의 실제 컬럼과 일치하는지도 함께 확인한다.
"""
from __future__ import annotations

from functools import lru_cache

from src.tools.analysis_cubes import load_legal_map, load_regions, load_stations, stamp_db
from src.tools.db_connector import fetch_frame, fetch_rows, open_db
from src.tools.gazetteer_index import load_place_index
from src.tools.synthetic_seed import measure_km
from src.tools.tool_common import (TOPIC_TABLES, describe_tables, measure_time, normalize_topic, tidy_payload)

CATALOG_KEYWORDS = {
    "commute": ["버스"], "night": ["유동인구"], "living": ["유동인구", "버스"], "spending": ["카드", "매출", "소비"],
    "crowd": ["버스", "유동인구"], "waste": ["음식물", "쓰레기"], "wifi": ["와이파이"], "ev": ["전기"],
    "general": ["유동인구", "버스", "기상"],
}
OBSERVATORY_NAMES = {"184": "제주", "185": "고산", "188": "성산", "189": "서귀포", "328": "중문"}


@lru_cache(maxsize=4)
def check_schema(stamp: float, table: str) -> dict:
    """분석 테이블 컬럼이 대응 공식 예시 시트 컬럼을 몇 개나 그대로 쓰는지 센다."""
    meta = fetch_rows("SELECT official_sheet FROM table_registry WHERE table_name = ?", (table,))
    if not meta:
        return {}
    sheet = str(meta[0]["official_sheet"] or "")
    code = sheet.split()[0] if sheet and sheet[0].isdigit() else ""
    if not code:
        return {"official_sheet": sheet, "matched": None, "total": None}
    example_table = "bdc_" + code.replace("-", "_")
    with open_db() as conn:
        try:
            example_cols = [r[1] for r in conn.execute(f'PRAGMA table_info("{example_table}")')]
            table_cols = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
        except Exception:
            return {"official_sheet": sheet, "matched": None, "total": None}
    official = [c for c in example_cols if c not in ("official_sheet", "source_type", "dataset_id", "reference_date")]
    matched = [c for c in official if c in table_cols]
    return {"official_sheet": sheet, "example_table": example_table, "matched": len(matched), "total": len(official),
            "missing": [c for c in official if c not in table_cols]}


def search_catalog(topic: str, limit: int = 3) -> list[dict]:
    hits: list[dict] = []
    for word in CATALOG_KEYWORDS.get(topic, ["유동인구"]):
        frame = fetch_frame("SELECT no, category, title, kind, period FROM datahub_catalog WHERE title LIKE ? "
                            "ORDER BY CAST(views AS INTEGER) DESC LIMIT ?", (f"%{word}%", limit))
        for row in frame.to_dict("records"):
            if row["no"] not in {h["no"] for h in hits}:
                hits.append(row)
    return hits[:limit]


def describe_unit(unit, stamp: float) -> dict:
    regions = load_regions(stamp)
    stations = load_stations(stamp)
    legal = load_legal_map(stamp)
    rows = regions[regions["emd_nm"].isin(unit.emds)]
    if rows.empty:
        return {"label": unit.label, "emds": unit.emds, "stations": [], "legal_dongs": [], "observatory": None}
    lat, lon = float(rows["lat"].mean()), float(rows["lon"].mean())
    rural = rows["kind"].isin(["읍", "면"]).all()
    radius = 4.0 if rural else 1.5
    nearby = []
    for row in stations.itertuples():
        distance = measure_km(lat, lon, row.latitude, row.longitude)
        if row.emd_nm in unit.emds or distance <= radius:
            nearby.append({"bus_station_id": str(row.bus_station_id), "bus_station_name": row.bus_station_name,
                           "km": round(distance, 2), "emd_nm": row.emd_nm, "id_status": row.id_status})
    nearby.sort(key=lambda s: s["km"])
    obs = fetch_frame("SELECT observatory_code, latitude, longitude FROM weather_stations")
    best = min(obs.itertuples(), key=lambda o: measure_km(lat, lon, o.latitude, o.longitude))
    return {
        "label": unit.label, "emds": unit.emds, "kind": unit.kind, "matched_from": unit.matched_from,
        "corrected": unit.corrected, "centroid": [round(lat, 4), round(lon, 4)], "radius_km": radius,
        "stations": nearby[:6],
        "legal_dongs": legal[legal["emd_nm"].isin(unit.emds)]["bjd"].tolist(),
        "dstrct_se": sorted(set(rows["dstrct_se"])),
        "observatory": {"code": str(best.observatory_code), "name": OBSERVATORY_NAMES.get(str(best.observatory_code), ""),
                        "km": round(measure_km(lat, lon, best.latitude, best.longitude), 1)},
    }


def resolve_spatial_catalog(areas=None, topic: str = "general", question: str = "") -> dict:
    output = {"tool": "resolve_spatial_catalog", "ok": True}
    with measure_time(output):
        index = load_place_index()
        units, unknown = index.resolve_areas(areas or [])
        if not units and question:
            for match in index.find_places(question):
                more, _ = index.resolve_areas([match.surface])
                units += [u for u in more if u.label not in {x.label for x in units}]
        topic = normalize_topic(topic)
        stamp = stamp_db()
        unit_rows = [describe_unit(unit, stamp) for unit in units[:4]]
        tables = TOPIC_TABLES[topic]
        datasets = []
        for row in describe_tables(tables):
            row["schema_check"] = check_schema(stamp, row["table"])
            datasets.append(row)
        facts = []
        for unit in unit_rows:
            stops = ", ".join(s["bus_station_name"] for s in unit["stations"][:3]) or "반경 내 주요 정류소 없음"
            obs = unit.get("observatory") or {}
            facts.append(f"{unit['label']}: 행정동 {len(unit['emds'])}곳({', '.join(unit['emds'][:4])}), "
                         f"법정동 {len(unit['legal_dongs'])}곳, 정류소 {len(unit['stations'])}곳({stops}), "
                         f"최근접 관측소 {obs.get('name', '')}({obs.get('code', '')})")
        for row in datasets:
            check = row.get("schema_check") or {}
            if check.get("total"):
                facts.append(f"{row['table']}: 공식 예시 시트 '{check['official_sheet']}' 컬럼 "
                             f"{check['matched']}/{check['total']}개 그대로 사용, 출처 {row['source_type']}({row['dataset_id']})")
        output.update({
            "topic": topic,
            "areas": unit_rows,
            "unknown_areas": unknown,
            "suggestions": {name: index.suggest_names(name) for name in unknown},
            "datasets": datasets,
            "join_keys": "행정동(emd) <-> 법정동(bjd, legal_dong_map) <-> 정류소 반경(bus_stations) <-> 관측소(weather_stations)",
            "catalog_matches": search_catalog(topic),
            "facts": facts,
            "provenance": [{k: d[k] for k in ("table", "source_type", "dataset_id", "reference_date")} for d in datasets],
        })
        if not units:
            output["ok"] = False
            output["message"] = "질문에서 제주 지역명을 찾지 못했습니다. 예: 노형동, 신제주, 삼화지구, 애월"
    return tidy_payload(output)
