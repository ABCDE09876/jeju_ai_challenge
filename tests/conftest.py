"""공통 설정: 프로젝트 루트를 import 경로에 넣고, 로컬 DB 를 한 번만 준비한다."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def prepare_database():
    from src.tools.analysis_cubes import warm_cubes
    from src.tools.db_connector import ensure_database

    assert ensure_database()
    warm_cubes()
    yield


@pytest.fixture()
def build_orchestrator():
    from src.agent.orchestrator import HybridOrchestrator
    from src.agent.response_cache import ResponseCache

    return HybridOrchestrator("life", force_offline=True, cache=ResponseCache(enabled=False))
