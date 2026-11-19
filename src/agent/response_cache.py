"""스마트 응답 캐시(agent_response_cache).

- 로컬 모드: 같은 질문·같은 조건이면 저장된 결과를 0.1초 안에 재사용한다.
- 하이브리드 모드: Claude 호출을 건너뛰지 않는다(Opus 5.5 xhigh 고정 정책). Claude 연결이 끊긴 순간에만
  같은 질문의 이전 Claude 해설을 대체 표시하는 용도로 쓴다.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

from src.config import CACHE_DB_PATH, DB_PATH
from src.tools.db_connector import open_db

SLOT_KEYS = ("intent", "areas", "topic", "hour_from", "hour_to", "weather", "day_type", "user_group", "wait_minutes",
             "variable", "claim", "needs_abstain", "workspace")


class ResponseCache:
    def __init__(self, path: Path | str = CACHE_DB_PATH, enabled: bool = True):
        self.path = Path(path)
        self.ready = False
        if not enabled:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open_db(self.path) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS agent_response_cache ("
                    "cache_key TEXT PRIMARY KEY, question TEXT, mode TEXT, slots_json TEXT, payload_json TEXT, "
                    "created_at TEXT, hit_count INTEGER DEFAULT 0)")
            self.ready = True
        except sqlite3.Error:
            self.ready = False

    @staticmethod
    def make_key(plan: dict, mode: str) -> str:
        question = re.sub(r"[\s\W_]+", "", str(plan.get("question", ""))).lower()
        slots = {k: plan.get(k) for k in SLOT_KEYS}
        stamp = DB_PATH.stat().st_mtime if DB_PATH.exists() else 0
        raw = json.dumps({"q": question, "s": slots, "m": mode, "db": stamp}, ensure_ascii=False, sort_keys=True,
                         default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def load(self, key: str) -> dict | None:
        if not self.ready:
            return None
        try:
            with open_db(self.path) as conn:
                row = conn.execute("SELECT payload_json FROM agent_response_cache WHERE cache_key = ?", (key,)).fetchone()
                if row:
                    conn.execute("UPDATE agent_response_cache SET hit_count = hit_count + 1 WHERE cache_key = ?", (key,))
            return json.loads(row[0]) if row else None
        except (sqlite3.Error, json.JSONDecodeError):
            return None

    def save(self, key: str, plan: dict, mode: str, bundle: dict) -> None:
        if not self.ready:
            return
        try:
            payload = json.dumps(bundle, ensure_ascii=False, default=str)
            with open_db(self.path) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO agent_response_cache (cache_key, question, mode, slots_json, payload_json, "
                    "created_at, hit_count) VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT hit_count FROM "
                    "agent_response_cache WHERE cache_key = ?), 0))",
                    (key, plan.get("question", ""), mode, json.dumps({k: plan.get(k) for k in SLOT_KEYS},
                                                                    ensure_ascii=False, default=str),
                     payload, datetime.now().isoformat(timespec="seconds"), key))
        except sqlite3.Error:
            return

    def report_stats(self) -> dict:
        if not self.ready:
            return {"entries": 0, "hits": 0}
        try:
            with open_db(self.path) as conn:
                entries, hits = conn.execute("SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM agent_response_cache").fetchone()
            return {"entries": int(entries), "hits": int(hits)}
        except sqlite3.Error:
            return {"entries": 0, "hits": 0}

    def clear(self) -> None:
        if not self.ready:
            return
        try:
            with open_db(self.path) as conn:
                conn.execute("DELETE FROM agent_response_cache")
        except sqlite3.Error:
            return
