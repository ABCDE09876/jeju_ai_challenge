"""공식 제공 엑셀 2종(제주데이터허브 데이터목록, 제주빅데이터센터 예시데이터)의 구조를 해독한다.

출력: data/meta/catalog_schema.json
  - datahub_catalog : 데이터허브 공개 데이터셋 목록(카테고리, 데이터명, 타입, 수집기간, 제공기관 등)
  - center_examples : 빅데이터센터 예시 시트별 실제 컬럼명, 예시 행, 추정 자료형

원본 엑셀 전체를 컨텍스트에 읽지 않도록 시트명, 컬럼, 자료형, 예시 3행만 요약 출력한다.
실행: python scripts/inspect_official_meta.py [--quiet]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import date, datetime
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_DIR = ROOT / "data" / "official"
META_DIR = ROOT / "data" / "meta"
CATALOG_FILE = "제주데이터허브 데이터목록_20261105.xlsx"
EXAMPLE_FILE = "제주빅데이터센터_예시데이터 및 컬럼확인용.xlsx"
COLUMN_TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
MAX_EXAMPLE_ROWS = 1000  # 예시 시트의 모든 행을 보존(현재 공식 파일은 시트당 1행)
CATALOG_KEYS = {
    "No.": "no",
    "카테고리": "category",
    "데이터명": "title",
    "데이터 타입": "kind",
    "등록일": "registered",
    "수정일": "updated",
    "수집 기간": "period",
    "제공기관": "provider",
    "조회수": "views",
    "즐겨찾기수": "favorites",
}


def stringify_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (datetime, date)):
        return value.isoformat()[:10]
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def guess_kind(value) -> str:
    if value is None or value == "":
        return "empty"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "int" if value.is_integer() else "float"
    if isinstance(value, (datetime, date)):
        return "date"
    text = str(value).strip()
    if re.fullmatch(r"-?\d+", text):
        return "int-like text"
    if re.fullmatch(r"-?[\d,]+\.?\d*", text):
        return "numeric text"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", text):
        return "date text"
    return "text"


def read_sheet_rows(sheet) -> list[tuple]:
    return [tuple(row) for row in sheet.iter_rows(values_only=True)]


def locate_header(rows: list[tuple]) -> int:
    """영문 컬럼 토큰이 2개 이상 연속으로 나오는 첫 행을 헤더로 본다."""
    for row_no, row in enumerate(rows[:10]):
        tokens = [stringify_cell(v) for v in row if stringify_cell(v)]
        if len(tokens) >= 2 and sum(bool(COLUMN_TOKEN.match(t)) for t in tokens) >= max(2, len(tokens) - 1):
            return row_no
    return 0


def parse_catalog(path: Path) -> dict:
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = book.worksheets[0]
    rows = read_sheet_rows(sheet)
    header_no = next(
        (i for i, row in enumerate(rows[:10]) if any(stringify_cell(v) in ("No.", "데이터명") for v in row)),
        1,
    )
    headers = [stringify_cell(v) for v in rows[header_no]]
    datasets = []
    for row in rows[header_no + 1:]:
        cells = [stringify_cell(v) for v in row]
        if not any(cells):
            continue
        entry = {}
        for col_no, label in enumerate(headers):
            key = CATALOG_KEYS.get(label, label or f"col{col_no}")
            entry[key] = cells[col_no] if col_no < len(cells) else ""
        datasets.append(entry)
    book.close()
    categories = Counter(item.get("category", "") for item in datasets)
    kinds = Counter(item.get("kind", "") for item in datasets)
    providers = Counter(item.get("provider", "") for item in datasets)
    return {
        "file": path.name,
        "sheet": sheet.title,
        "title_row": stringify_cell(next((v for v in rows[0] if v), "")) if header_no > 0 else "",
        "header_row": header_no,
        "columns": headers,
        "row_count": len(datasets),
        "categories": dict(categories.most_common()),
        "kinds": dict(kinds.most_common()),
        "top_providers": dict(providers.most_common(25)),
        "datasets": datasets,
    }


def parse_examples(path: Path) -> list[dict]:
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheets = []
    for sheet in book.worksheets:
        rows = read_sheet_rows(sheet)
        if not rows:
            continue
        header_no = locate_header(rows)
        header_cells = rows[header_no]
        positions = [i for i, v in enumerate(header_cells) if stringify_cell(v)]
        columns = [stringify_cell(header_cells[i]) for i in positions]
        samples = []
        for row in rows[header_no + 1: header_no + 1 + MAX_EXAMPLE_ROWS]:
            if not any(stringify_cell(v) for v in row):
                continue
            samples.append({col: stringify_cell(row[pos]) if pos < len(row) else "" for col, pos in zip(columns, positions)})
        first_row = rows[header_no + 1] if len(rows) > header_no + 1 else ()
        kinds = {col: guess_kind(first_row[pos] if pos < len(first_row) else None) for col, pos in zip(columns, positions)}
        code_match = re.match(r"\s*(\d+-\d+)\s*(.*)", sheet.title)
        sheets.append({
            "sheet": sheet.title.strip(),
            "code": code_match.group(1) if code_match else "",
            "title": (code_match.group(2) if code_match else sheet.title).strip(),
            "header_row": header_no,
            "leading_blank_columns": positions[0] if positions else 0,
            "columns": columns,
            "inferred_types": kinds,
            "example_rows": samples,
        })
    book.close()
    return sheets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="요약 출력 생략")
    args = parser.parse_args()

    catalog_path = OFFICIAL_DIR / CATALOG_FILE
    example_path = OFFICIAL_DIR / EXAMPLE_FILE
    missing = [p.name for p in (catalog_path, example_path) if not p.exists()]
    if missing:
        print(f"[경고] 공식 엑셀이 없습니다: {missing}. data/official/ 에 넣은 뒤 다시 실행하세요.")
        return 1

    catalog = parse_catalog(catalog_path)
    examples = parse_examples(example_path)
    schema = {
        "generated_by": "scripts/inspect_official_meta.py",
        "sources": [CATALOG_FILE, EXAMPLE_FILE],
        "datahub_catalog": catalog,
        "center_examples": examples,
    }
    META_DIR.mkdir(parents=True, exist_ok=True)
    out_path = META_DIR / "catalog_schema.json"
    out_path.write_text(json.dumps(schema, ensure_ascii=False, indent=1), encoding="utf-8")

    if not args.quiet:
        print(f"[데이터허브 목록] 시트={catalog['sheet']} 헤더행={catalog['header_row']} 데이터셋={catalog['row_count']}건")
        print("  컬럼:", " | ".join(catalog["columns"]))
        print("  카테고리:", catalog["categories"])
        print("  타입:", catalog["kinds"])
        for item in catalog["datasets"][:3]:
            print("  예시:", item)
        print(f"[빅데이터센터 예시] 시트 {len(examples)}개")
        for sheet in examples:
            print(f"  {sheet['code']:>5} {sheet['title'][:30]:<30} ({len(sheet['columns'])}) {', '.join(sheet['columns'])}")
    print(f"저장 완료: {out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
