"""경로, 시드, Claude 모델 고정 정책 등 전역 설정.

Claude API 경로는 항상 claude-opus-5-5 + effort "xhigh"(엑스트라 하이)로 실행된다.
환경변수 ANTHROPIC_MODEL 에 다른 값을 넣어도 이 고정값이 우선한다.
비용 상한(호출 횟수 제한, 토큰 축소 등)은 두지 않는다.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META_DIR = ROOT / "data" / "meta"
PROCESSED_DIR = ROOT / "data" / "processed"
OFFICIAL_DIR = ROOT / "data" / "official"
RAW_DIR = ROOT / "data" / "raw"
SAMPLES_DIR = ROOT / "data" / "samples"
DOCS_DIR = ROOT / "docs"

DB_PATH = PROCESSED_DIR / "jeju_truelife.db"
CACHE_DB_PATH = PROCESSED_DIR / "agent_cache.db"
GAZETTEER_PATH = META_DIR / "jeju_gazetteer.json"
CATALOG_SCHEMA_PATH = META_DIR / "catalog_schema.json"
BENCHMARK_ITEMS_PATH = META_DIR / "benchmark_items.json"

SEED = 20261120
ANALYSIS_START = "20260501"
ANALYSIS_END = "20261031"

# ---- Claude 고정 정책 (사용자 지시: Claude API 사용 시 무조건 Opus 5.5 엑스트라 하이) ----
CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_EFFORT = "xhigh"
CLAUDE_THINKING = {"type": "adaptive", "display": "summarized"}
# max_tokens 는 Messages API 의 필수 항목이라 생략할 수 없다. 응답을 자르지 않도록 모델 최대 출력값을 넣는다.
CLAUDE_MAX_TOKENS = 128000

# ---- 가용성(연결) 판정용 시간 기준: 비용과 무관 ----
FIRST_EVENT_TIMEOUT_S = 3.5      # 첫 스트림 이벤트가 3.5초 안에 오지 않으면 연결 불량으로 보고 로컬 엔진으로 전환
STREAM_IDLE_TIMEOUT_S = 600.0    # 10분 동안 어떤 이벤트도 없으면 끊긴 연결로 판단

RAIN_DAY_MM = 1.0                # 우천일 판정 기준(일 강수량 1.0mm 이상)
MIN_SLICE_ROWS = 5               # 표본 하한: 이보다 적으면 조건을 한 단계 넓힌다


def load_env() -> None:
    """프로젝트 루트의 .env 를 읽어 환경변수로 올린다(이미 설정된 값은 유지)."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv 미설치 환경에서도 앱은 동작해야 한다
        return
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for encoding in ("utf-8", "utf-8-sig", "cp949"):  # 메모장 ANSI 저장(.env 한글 주석) 대비
        try:
            load_dotenv(env_path, override=False, encoding=encoding)
            return
        except (UnicodeDecodeError, ValueError):
            continue


def read_api_key(typed_key: str | None = None) -> str:
    """UI 입력값 > 환경변수 순으로 Anthropic 키를 고른다. 없으면 빈 문자열."""
    if typed_key and typed_key.strip():
        return typed_key.strip()
    load_env()
    return os.environ.get("ANTHROPIC_API_KEY", "").strip()


def describe_model_lock() -> str:
    requested = os.environ.get("ANTHROPIC_MODEL", "").strip()
    note = ""
    if requested and requested != CLAUDE_MODEL:
        note = f" (ANTHROPIC_MODEL={requested} 요청은 고정 정책에 따라 무시됨)"
    return f"{CLAUDE_MODEL} · effort={CLAUDE_EFFORT} · adaptive thinking 고정{note}"
