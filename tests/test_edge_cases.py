"""엣지케이스 방어: 이상 입력, SQL 주입, 초장문, CSV 감사, 수치 근거 감사, 3구획 형식."""
from src.agent.number_auditor import audit_numbers
from src.agent.prompts import SECTION_FACTS, SECTION_LIMIT, SECTION_REVERSAL
from src.config import SAMPLES_DIR
from src.tools.csv_auditor import audit_uploaded_csv
from src.tools.db_connector import fetch_scalar


def test_empty_question(build_orchestrator):
    bundle = build_orchestrator.answer("")
    assert bundle["verdict"] == "info" and len(bundle["followups"]) == 3


def test_sql_injection(build_orchestrator):
    bundle = build_orchestrator.answer("'; DROP TABLE regions; --")
    assert bundle["verdict"]
    assert fetch_scalar("SELECT COUNT(*) FROM regions") == 43


def test_huge_input(build_orchestrator):
    bundle = build_orchestrator.answer("노형동 " + "가" * 10000)
    assert bundle["narrative"] and bundle["field_question"]


def test_three_sections(build_orchestrator):
    bundle = build_orchestrator.answer("일도1동이랑 이도1동 밤 9시 이후 비교해줘")
    narrative = bundle["narrative"]
    for header in (SECTION_FACTS, SECTION_REVERSAL, SECTION_LIMIT, "현장 확인 질문"):
        assert header in narrative
    assert narrative.count("현장 확인 질문:") == 1
    assert bundle["number_audit"]["ungrounded"] == []


def test_csv_hallim():
    report = audit_uploaded_csv((SAMPLES_DIR / "hallim_food_waste_daily_cp949.csv").read_bytes(),
                                "한림읍 음식물쓰레기 배출량이 2019년 12월에 10월보다 58.69배 폭증했다. 관광객 때문이다.")
    assert report["encoding"] == "cp949" and report["masked_cells"] == 2
    assert any(c["status"] == "수치 일치" and c["computed"] == 58.69 for c in report["claim_checks"])
    assert any(edge["boundary"] == "2019-11-01" for edge in report["month_edges"])
    assert report["registered_changes"] and report["verdict"] == "unmeasured_cause"


def test_csv_denominator():
    report = audit_uploaded_csv((SAMPLES_DIR / "ev_charging_wait_2447.csv").read_bytes(), "충전 대기시간은 평균 17분이다.")
    assert report["rows"] == 2447
    assert report["durations"]["valid_rows"] == 113
    assert report["verdict"] == "thin_denominator"


def test_number_audit():
    audit = audit_numbers("노형동 12,345명, 비중 58.2%, 1.78배", {"a": 12345, "b": 0.582, "c": 1.78})
    assert audit["checked"] == 3 and audit["grounded"] == 3
    audit = audit_numbers("근거 없는 수치 77,777명", {"a": 12345})
    assert audit["ungrounded"] == ["77,777"]
