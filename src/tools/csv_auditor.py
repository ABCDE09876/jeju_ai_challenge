"""탭 3: 새 제주 CSV 업로드 실시간 감사기.

아무 CSV 와 주장 문장을 넣으면
  ① 인코딩·구분자·천단위 콤마·마스킹을 자동 복구한 뒤 주장 속 수치를 재계산하고
  ② 월 경계(월말 7일 vs 다음 달 월초 7일)에서 계단식 집계 단절을 찾고
  ③ 주장의 '원인' 개념이 CSV 컬럼 스키마에 있는지 대조해 인과 과장을 중립 문장으로 교정한다.
공식 예시데이터 70개 시트와 컬럼을 대조해 어떤 공식 데이터와 닮았는지도 알려 준다.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np
import pandas as pd

from src.config import CATALOG_SCHEMA_PATH
from src.tools.analysis_cubes import load_meta, stamp_db
from src.tools.gazetteer_index import load_place_index
from src.tools.resilient_loader import load_any_csv, average_safely, divide_safely
from src.tools.slice_excavator import CAUSE_LEXICON, CAUSE_WORDS, CAUSAL_MARKERS, find_concepts
from src.tools.tool_common import attach_josa, format_pct, tidy_payload

DATE_HINTS = ("date", "일자", "날짜", "기준일", "ymd", "년월", "일시", "time", "시각")
COLUMN_CONCEPTS = {
    "tourist": ("관광", "visit_type", "tour", "외지", "방문유형", "user_type"),
    "rain": ("강수", "rain", "날씨", "weather", "우천"),
    "temperature": ("기온", "temp", "온도"),
    "ev": ("전기차", "ev", "충전"),
    "income": ("소득", "income", "급여"),
    "collection_method": ("수집방식", "집계방식", "collection", "rfid", "agg_unit", "집계단위"),
    "market_day": ("장날", "market"),
}


def match_official_schema(columns: list[str]) -> dict:
    if not CATALOG_SCHEMA_PATH.exists():
        return {}
    schema = json.loads(CATALOG_SCHEMA_PATH.read_text(encoding="utf-8"))
    wanted = {str(c).strip().lower() for c in columns}
    best, best_score = None, 0.0
    for sheet in schema.get("center_examples", []):
        official = {c.lower() for c in sheet["columns"]}
        if not official:
            continue
        overlap = len(wanted & official)
        score = overlap / len(official)
        if overlap >= 2 and score > best_score:
            best, best_score = sheet, score
    if not best:
        return {"matched": False}
    return {"matched": True, "sheet": best["sheet"], "coverage": round(best_score, 3),
            "shared_columns": sorted(wanted & {c.lower() for c in best["columns"]})}


def find_datetime_columns(frame: pd.DataFrame) -> list[str]:
    found = []
    for column in frame.columns:
        name = str(column).lower()
        series = frame[column]
        if series.dtype != object and not any(h in name for h in DATE_HINTS):
            continue
        sample = series.dropna().astype(str).head(200)
        if sample.empty:
            continue
        parsed = parse_dates(sample)  # YYYYMMDD·YYYYMM 같은 공공데이터 표기까지 인식
        if parsed.notna().mean() >= 0.6 or (any(h in name for h in DATE_HINTS) and parsed.notna().mean() >= 0.3):
            found.append(column)
    return found


def parse_dates(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    compact = text.str.fullmatch(r"\d{8}")
    parsed = pd.to_datetime(text.where(~compact), errors="coerce", format="mixed")
    if compact.any():
        parsed = parsed.fillna(pd.to_datetime(text.where(compact), errors="coerce", format="%Y%m%d"))
    months = text.str.fullmatch(r"\d{6}")
    if months.any():
        parsed = parsed.fillna(pd.to_datetime(text.where(months), errors="coerce", format="%Y%m"))
    return parsed


def pick_value_column(frame: pd.DataFrame, numeric: list[str], claim: str) -> str | None:
    if not numeric:
        return None
    for column in numeric:
        if any(token and token in claim for token in re.split(r"[_\s]+", str(column))):
            return column
    return max(numeric, key=lambda c: frame[c].notna().sum())


def find_region(frame: pd.DataFrame) -> tuple[str | None, list[str]]:
    index = load_place_index()
    for column in frame.columns:
        if frame[column].dtype != object:
            continue
        values = frame[column].dropna().astype(str).unique()[:30]
        hits = [v for v in values if index.match_token(v.replace(" ", ""))]
        if values.size and len(hits) >= max(1, 0.6 * len(values)):
            return column, [str(v) for v in values]
    return None, []


def scan_month_edges(daily: pd.Series) -> list[dict]:
    """월말 7일 평균 대비 다음 달 월초 7일 평균이 2배 이상/0.5배 이하로 바뀌면 월 경계 단절."""
    edges = []
    if daily.empty:
        return edges
    months = sorted(set(daily.index.to_period("M")))
    for left, right in zip(months[:-1], months[1:]):
        tail = daily[daily.index.to_period("M") == left].tail(7)
        head = daily[daily.index.to_period("M") == right].head(7)
        if len(tail) < 3 or len(head) < 3:
            continue
        ratio = divide_safely(head.mean(), tail.mean())
        if ratio is not None and (ratio >= 2.0 or ratio <= 0.5):
            edges.append({"boundary": f"{right.year}-{right.month:02d}-01", "before_7d": float(tail.mean()),
                          "after_7d": float(head.mean()), "ratio": ratio})
    return edges


def scan_monthly_jumps(monthly: pd.Series) -> list[dict]:
    jumps = []
    values = monthly.dropna()
    for (left, a), (right, b) in zip(values.items(), list(values.items())[1:]):
        ratio = divide_safely(b, a)
        if ratio is not None and (ratio >= 2.0 or ratio <= 0.6):
            jumps.append({"from": str(left), "to": str(right), "from_value": float(a), "to_value": float(b),
                          "ratio": ratio})
    return jumps


def has_clock_time(series: pd.Series) -> bool:
    """'2021-03-01 10:15:00' 처럼 시각이 들어 있는 컬럼인지(날짜만 있는 기준일·개통일 컬럼 제외)."""
    sample = series.dropna().astype(str).head(200)
    return bool(len(sample)) and sample.str.contains(r"\d{1,2}:\d{2}").mean() >= 0.3


def measure_durations(frame: pd.DataFrame, datetime_columns: list[str]) -> dict:
    """시각 컬럼 두 개(요청->시작 등) 사이 소요시간. 날짜만 있는 컬럼끼리는 소요시간으로 보지 않는다."""
    timed = [c for c in datetime_columns if has_clock_time(frame[c])]
    if len(timed) < 2:
        return {}
    first, second = timed[0], timed[1]
    start = parse_dates(frame[first])
    end = parse_dates(frame[second])
    minutes = (end - start).dt.total_seconds() / 60
    valid = start.notna() & end.notna() & (minutes >= 0)
    values = minutes[valid]
    return {"from_column": first, "to_column": second, "rows": int(len(frame)), "valid_rows": int(valid.sum()),
            "coverage": divide_safely(int(valid.sum()), len(frame)), "mean_minutes": average_safely(values),
            "median_minutes": float(values.median()) if len(values) else None}


def check_claim_figures(claim: str, monthly: pd.Series, mean_value, durations: dict) -> list[dict]:
    checks = []
    months_in_claim = [int(m) for m in re.findall(r"(\d{1,2})\s*월", claim)]
    years_in_claim = [int(y) for y in re.findall(r"(20\d{2})\s*년", claim)]
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*배", claim):
        stated = float(match.group(1))
        computed = None
        if len(months_in_claim) >= 2 and not monthly.empty:
            # 'X월에 Y월보다 N배' = X(먼저 언급된 달) / Y(기준 달). 감소(0.7배)도 그대로 0.7로 계산한다.
            if len(years_in_claim) >= 2:
                years = years_in_claim[:2]
            else:
                years = [years_in_claim[0] if years_in_claim else None] * 2
            keyed = {(p.year, p.month): v for p, v in monthly.items()}

            def find_month(month, year):
                if year and (year, month) in keyed:
                    return keyed[(year, month)]
                options = [v for (y, m), v in keyed.items() if m == month]
                return options[-1] if options else None

            target = find_month(months_in_claim[0], years[0])
            base = find_month(months_in_claim[1], years[1])
            if target is not None and base is not None:
                computed = divide_safely(target, base)
        if computed is None and len(monthly.dropna()) >= 2:
            ratios = [divide_safely(b, a) for a, b in zip(monthly.dropna().values[:-1], monthly.dropna().values[1:])]
            ratios = [r for r in ratios if r]
            computed = min(ratios, key=lambda r: abs(r - stated)) if ratios else None
        scale = max(abs(stated), 1e-9)
        status = "검증 불가" if computed is None else ("수치 일치" if abs(computed - stated) / scale <= 0.01 else
                                                    ("근사 일치(오차 5% 이내)" if abs(computed - stated) / scale <= 0.05
                                                     else "수치 불일치"))
        checks.append({"kind": "배율", "stated": stated, "computed": None if computed is None else round(computed, 2),
                       "status": status})
    for match in re.finditer(r"평균\s*(?:대기\s*)?(?:시간\s*)?(?:은|는|이|가)?\s*(\d+(?:\.\d+)?)\s*(분|명|건|kWh)?", claim):
        stated = float(match.group(1))
        unit = match.group(2) or ""
        computed = durations.get("mean_minutes") if unit == "분" and durations else mean_value
        status = "검증 불가" if computed is None else (
            "수치 일치" if abs(computed - stated) / max(stated, 1e-9) <= 0.05 else "수치 불일치")
        checks.append({"kind": f"평균({unit or '값'})", "stated": stated,
                       "computed": None if computed is None else round(computed, 2), "status": status})
    return checks


def audit_csv_claim(claim: str, columns: list[str]) -> dict:
    if not claim or not any(marker in claim for marker in CAUSAL_MARKERS):
        return {"status": "no_causal_claim"}
    causes = find_concepts(claim, CAUSE_LEXICON)
    if not causes:
        return {"status": "no_causal_claim"}
    lowered = [str(c).lower() for c in columns]
    rows = []
    for cause in causes:
        hints = COLUMN_CONCEPTS.get(cause, ())
        measured = [c for c in lowered if any(h.lower() in c for h in hints)]
        rows.append({"cause": cause, "cause_word": CAUSE_WORDS.get(cause, cause), "columns": measured,
                     "measured": bool(measured)})
    missing = [r for r in rows if not r["measured"]]
    if missing:
        words = ", ".join(r["cause_word"] for r in missing)
        rewrite = (f"업로드한 자료로는 수치 변화만 확인되며, {attach_josa(quote_text(words), '은/는')} 측정한 컬럼이 없어 "
                   f"원인으로 단정할 수 없습니다. 교정문: \"해당 지표가 변했다(원인 미확인, {words} 영향은 별도 자료로 검증 필요).\"")
        return {"status": "unmeasured_cause", "assessments": rows, "neutral_rewrite": rewrite}
    return {"status": "correlation_only", "assessments": rows,
            "neutral_rewrite": "원인 후보 컬럼이 있어 연관성은 검토할 수 있으나, 인과는 별도 설계 없이 단정할 수 없습니다."}


def quote_text(text: str) -> str:
    return f"'{text}'"


def audit_uploaded_csv(source, claim: str = "", filename: str = "") -> dict:
    claim = (claim or "").strip()
    loaded = load_any_csv(source)
    report = {"tool": "audit_uploaded_csv", "file": filename, "encoding": loaded.encoding,
              "delimiter": {"\t": "탭", ",": "콤마", ";": "세미콜론", "|": "파이프"}.get(loaded.delimiter, loaded.delimiter),
              "rows": loaded.count_rows(), "columns": [str(c) for c in loaded.frame.columns], "repair_notes": loaded.notes,
              "masked_cells": loaded.masked_cells, "restored_codes": loaded.restored_codes}
    if not loaded.ok or loaded.frame.empty:
        report.update({"ok": False, "verdict": "unreadable",
                       "facts": [f"파일을 표로 해석할 수 없습니다: {'; '.join(loaded.notes) or '내용 없음'}"],
                       "reversals": ["판정 보류: 읽을 수 있는 데이터 행이 없습니다."],
                       "limitations": ["열 이름이 있는 CSV(첫 줄 헤더) 형식인지 확인이 필요합니다."],
                       "field_question": "원본 파일을 내려받은 사이트에서 CSV(UTF-8 또는 CP949)로 다시 저장해 올려 주실 수 있을까요?"})
        return tidy_payload(report)

    frame = loaded.frame
    report["ok"] = True
    report["schema_match"] = match_official_schema(report["columns"])
    datetime_columns = find_datetime_columns(frame)
    numeric = [c for c in loaded.numeric_columns if c not in datetime_columns]
    value_column = pick_value_column(frame, numeric, claim)
    region_column, regions = find_region(frame)
    durations = measure_durations(frame, datetime_columns)
    monthly = pd.Series(dtype=float)
    daily = pd.Series(dtype=float)
    mean_value = None
    if value_column is not None:
        values = pd.to_numeric(frame[value_column], errors="coerce")
        mean_value = average_safely(values)
        if datetime_columns:
            # 시간축은 날짜가 가장 많이 채워진 컬럼을 쓴다(빈 시각이 많은 컬럼으로 추세를 만들지 않도록).
            axis = max(datetime_columns, key=lambda c: int(parse_dates(frame[c]).notna().sum()))
            dates = parse_dates(frame[axis])
            usable = dates.notna() & values.notna()
            if usable.any():
                by_day = pd.Series(values[usable].to_numpy(), index=dates[usable]).groupby(level=0).mean().sort_index()
                daily = by_day
                grouped = by_day.groupby(by_day.index.to_period("M"))
                counts = grouped.size()
                monthly = grouped.mean()
                if counts.median() > 1:
                    # 일 단위 자료는 관측일 5일 미만인 달을 추세 판단에서 뺀다(소표본 착시 방지).
                    monthly = monthly[counts >= 5]
    edges = scan_month_edges(daily) if len(daily) >= 20 else []
    jumps = scan_monthly_jumps(monthly)
    checks = check_claim_figures(claim, monthly, mean_value, durations)
    causal = audit_csv_claim(claim, report["columns"])
    changes = []
    try:
        registry = load_meta(stamp_db())["changes"]
        region_names = {r.replace("제주시 ", "").replace("서귀포시 ", "") for r in regions}
        for jump in jumps:
            month_key = jump["to"].replace("-", "")
            for change in registry:
                if str(change["change_month"]) == month_key and (not change["emd_nm"] or change["emd_nm"] in region_names):
                    changes.append({**change, "matched_month": jump["to"]})
    except Exception:
        changes = []

    thin = bool(durations) and (durations.get("coverage") or 0) < 0.3
    if causal["status"] == "unmeasured_cause":
        verdict = "unmeasured_cause"
    elif edges or jumps:
        verdict = "structural_break"
    elif thin:
        verdict = "thin_denominator"
    elif any(c["status"] == "수치 불일치" for c in checks):
        verdict = "figure_mismatch"
    else:
        verdict = "no_issue"

    facts = [f"파일 복구: 인코딩 {loaded.encoding}, 구분자 {report['delimiter']}, {loaded.count_rows():,}행 x "
             f"{len(report['columns'])}열, 마스킹 {loaded.masked_cells}셀, 코드 복원 {loaded.restored_codes}건"]
    if report["schema_match"].get("matched"):
        facts.append(f"공식 예시 시트 '{report['schema_match']['sheet']}'와 컬럼 "
                     f"{format_pct(report['schema_match']['coverage'])} 일치")
    if value_column is not None and mean_value is not None:
        facts.append(f"'{value_column}' 전체 평균 {mean_value:,.2f}")
    for period, value in list(monthly.items())[-6:]:
        facts.append(f"{period.year}-{period.month:02d} 일평균 {value:,.2f}")
    if durations:
        facts.append(f"'{durations['from_column']}'->'{durations['to_column']}' 소요시간 계산 가능 "
                     f"{durations['valid_rows']:,}/{durations['rows']:,}행({format_pct(durations['coverage'])}), "
                     f"평균 {durations['mean_minutes'] or 0:.1f}분")
    for check in checks:
        facts.append(f"주장 수치 {check['stated']}({check['kind']}) 재계산 {check['computed']} -> {check['status']}")
    reversals = []
    for edge in edges:
        reversals.append(f"월 경계 단절: {edge['boundary']} 전후 7일 평균 {edge['before_7d']:,.2f} -> "
                         f"{edge['after_7d']:,.2f} ({edge['ratio']:.2f}배)")
    for jump in jumps:
        reversals.append(f"월간 계단 점프: {jump['from']} -> {jump['to']} ({jump['ratio']:.2f}배)")
    for change in changes:
        reversals.append(f"등록된 집계변경 이력과 시점 일치: {change['emd_nm'] or '전역'} {change['change_type']}"
                         f"({change['before_desc']} -> {change['after_desc']})")
    if thin:
        reversals.append(f"분모 함정: 소요시간은 전체의 {format_pct(durations['coverage'])}만 계산 가능해 대표값으로 쓸 수 없습니다.")
    if causal.get("status") == "unmeasured_cause":
        for row in causal["assessments"]:
            if not row["measured"]:
                reversals.append(f"인과 감사: '{row['cause_word']}'을(를) 측정한 컬럼 없음")
    if not reversals:
        reversals.append("월 경계 단절·분모 함정·미측정 인과가 발견되지 않았습니다(정상).")
    limitations = ["업로드 자료 밖의 수집 방식 변경(기기 교체·집계 기준 변경)은 파일만으로 확인할 수 없습니다."]
    if loaded.masked_cells:
        limitations.append(f"마스킹 {loaded.masked_cells}셀은 0이 아니라 결측으로 처리했습니다.")
    if edges or jumps:
        field_question = "단절 시점 전후로 수집 기기·집계 기준이 바뀌었는지 자료 제공 기관에 확인할 수 있을까요?"
    elif thin:
        field_question = "시각이 비어 있는 행이 어떤 경로(현장 결제 등)로 생긴 기록인지 운영사에 확인할 수 있을까요?"
    else:
        field_question = "이 수치가 현장 체감과 맞는지 실제 이용자께 확인할 수 있을까요?"
    chart = {"monthly": [{"month": f"{p.year}-{p.month:02d}", "value": float(v)} for p, v in monthly.items()]}
    report.update({
        "verdict": verdict, "value_column": value_column, "datetime_columns": datetime_columns,
        "region_column": region_column, "regions": regions[:5], "durations": durations, "month_edges": edges,
        "monthly_jumps": jumps, "claim_checks": checks, "causal_audit": causal, "registered_changes": changes,
        "facts": facts, "reversals": reversals, "limitations": limitations, "field_question": field_question,
        "neutral_rewrite": causal.get("neutral_rewrite"), "chart": chart,
    })
    return tidy_payload(report)
