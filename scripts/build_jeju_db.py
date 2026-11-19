"""4대 축 결합 로컬 DB 구축 CLI.

python scripts/build_jeju_db.py            # data/raw/*.csv 원자료가 있으면 최우선 적재, 없으면 공식 스키마 보존 시드 확장
python scripts/build_jeju_db.py --quiet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CATALOG_SCHEMA_PATH, OFFICIAL_DIR  # noqa: E402
from src.tools.db_connector import fetch_frame  # noqa: E402
from src.tools.jeju_db_builder import build_database  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="제주의 두 얼굴 로컬 DB 구축")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    log = (lambda message: None) if args.quiet else print

    if not CATALOG_SCHEMA_PATH.exists() and any(OFFICIAL_DIR.glob("*.xlsx")):
        import subprocess
        log("공식 엑셀 메타(catalog_schema.json)가 없어 먼저 해독합니다.")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "inspect_official_meta.py"), "--quiet"], check=False)
    summary = build_database(log=log)
    registry = fetch_frame("SELECT table_name, axis, source_type, row_count FROM table_registry "
                           "WHERE source_type NOT IN ('official_example') ORDER BY axis, table_name")
    if not args.quiet:
        print(registry.to_string(index=False))
        print(f"공식 예시 시트 기준 테이블: "
              f"{int(fetch_frame('SELECT COUNT(*) AS n FROM table_registry WHERE source_type = ?', ('official_example',)).n[0])}개")
        print(f"표본 CSV: {', '.join(summary['samples'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
