"""제출 전 자체 감사: 하드코딩 키·비용 상한 흔적·금지 이름 규칙 위반이 0건이어야 한다."""
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_audit_clean():
    module = runpy.run_path(str(ROOT / "scripts" / "audit_codebase.py"))
    report = module["run_audit"]()
    assert report["naming"] == [], report["naming"][:10]
    assert report["secrets_and_billing"] == [], report["secrets_and_billing"][:10]


def test_env_excluded():
    ignore = (ROOT / ".claudeignore").read_text(encoding="utf-8")
    assert ".env" in ignore.split()
