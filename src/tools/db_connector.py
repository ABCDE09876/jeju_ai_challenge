"""SQLite 연결 계층.

Streamlit 은 위젯을 누를 때마다 스크립트를 다른 스레드에서 다시 실행한다. 연결 객체를 스레드 사이에서
재사용하면 sqlite3.ProgrammingError 가 나므로 모든 연결은 check_same_thread=False 로 열고,
open_db() 컨텍스트 매니저 안에서만 쓰고 즉시 닫는다.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from src.config import DB_PATH

BUILD_LOCK = threading.Lock()


@contextmanager
def open_db(path: Path | str = DB_PATH, keyed_rows: bool = False):
    """with open_db() as conn: ... 블록이 끝나면 커밋(예외 시 롤백) 후 연결을 닫는다."""
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
    if keyed_rows:
        conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def fetch_frame(sql: str, params=(), path: Path | str = DB_PATH) -> pd.DataFrame:
    with open_db(path) as conn:
        return pd.read_sql_query(sql, conn, params=list(params))


def fetch_rows(sql: str, params=(), path: Path | str = DB_PATH) -> list[dict]:
    with open_db(path, keyed_rows=True) as conn:
        return [dict(row) for row in conn.execute(sql, tuple(params)).fetchall()]


def fetch_scalar(sql: str, params=(), path: Path | str = DB_PATH, default=None):
    with open_db(path) as conn:
        row = conn.execute(sql, tuple(params)).fetchone()
    return default if row is None or row[0] is None else row[0]


def list_tables(path: Path | str = DB_PATH) -> list[str]:
    with open_db(path) as conn:
        return [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]


def check_database(path: Path | str = DB_PATH) -> bool:
    if not Path(path).exists():
        return False
    try:
        return int(fetch_scalar("SELECT COUNT(*) FROM table_registry", path=path, default=0)) > 0
    except sqlite3.Error:
        return False


def ensure_database(log=None) -> bool:
    """DB 가 없으면 한 번만 구축한다(동시 실행 시에도 잠금으로 1회 보장)."""
    if check_database():
        return True
    with BUILD_LOCK:
        if check_database():
            return True
        from src.tools.jeju_db_builder import build_database
        build_database(log=log or (lambda message: None))
    return check_database()
