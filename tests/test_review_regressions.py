"""독립 코드 리뷰에서 확인된 결함의 회귀 테스트."""
import io
import time
from pathlib import Path
from types import SimpleNamespace

from src.agent.claude_bridge import ClaudeBridge
from src.agent.grounded_synthesizer import ensure_sections, read_claim_direction
from src.agent.intent_parser import extract_hours
from src.agent.number_auditor import audit_numbers
from src.config import SAMPLES_DIR
from src.tools.csv_auditor import audit_uploaded_csv

ROOT = Path(__file__).resolve().parents[1]


def test_claim_order(build_orchestrator):
    from src.agent.orchestrator import HybridOrchestrator
    from src.agent.response_cache import ResponseCache

    for question in ("애월이 아라동보다 활기차다던데 맞아?", "아라동보다 애월이 활기차다던데 맞아?",
                     "애월이 아라동보다 비교적 활기차다던데 맞아?"):
        orchestrator = HybridOrchestrator("lawyer", force_offline=True, cache=ResponseCache(enabled=False))
        bundle = orchestrator.answer(question)
        assert bundle["verdict"] in ("refuted", "reversal"), (question, bundle["verdict"], bundle["headline"])
        assert "일치합니다" not in bundle["headline"]


def test_direction_words():
    assert read_claim_direction(" 비교적 붐빈다던데") == 1
    assert read_claim_direction(" 사람이 적지 않다") == 1
    assert read_claim_direction(" 덜 붐빈다") == -1
    assert read_claim_direction(" 한산하다") == -1
    assert read_claim_direction(" 많지 않다") == -1


def test_slice_comparative(build_orchestrator):
    bundle = build_orchestrator.answer("도두동 오일장 정류소는 장날에 평소보다 비교적 붐빈다던데 맞아?")
    assert bundle["verdict"] == "confirmed", bundle["headline"]


def test_section_normalizer():
    bundle = {"facts": ["a 1"], "reversals": ["b"], "limitations": ["c"], "field_question": "현장 질문?"}
    text = ("**결론**: x\n[원자료로 확인된 수치 사실]\n- 1\n[교란 제거 후 역전 구간]\n- y\n"
            "[원자료 컬럼 부재로 인해 단정할 수 없는 한계 및 현장 확인 질문 1개]\n- 현장 확인 질문: 하나?\n- 현장 확인 질문: 둘?")
    fixed, appended = ensure_sections(text, bundle)
    assert fixed.count("현장 확인 질문:") == 1 and appended == ""
    partial, appended = ensure_sections("**결론**: x\n[원자료로 확인된 수치 사실]\n- 1", bundle)
    assert partial.count("[원자료로 확인된 수치 사실]") == 1
    assert partial.count("현장 확인 질문:") == 1 and appended


def test_minute_audit():
    audit = audit_numbers("대기 42.7분, 불편 지수 63.2분, 18시 기준", {"wait": 7.5, "score": 9.1})
    assert audit["checked"] == 2 and audit["ungrounded"] == ["42.7", "63.2"]


def test_csv_decrease():
    raw = "기준년월,이용자수\n202008,120\n202009,100\n202010,70\n202011,71\n".encode("utf-8")
    report = audit_uploaded_csv(io.BytesIO(raw), "2020년 10월에 9월보다 0.7배로 줄었다")
    assert any(c["computed"] == 0.7 and c["status"] == "수치 일치" for c in report["claim_checks"])
    wifi = audit_uploaded_csv((SAMPLES_DIR / "wifi_monthly_users_tab_euckr.csv").read_bytes(),
                              "와이파이 이용자가 2020년 10월에 0.55배로 줄었다. 관광객이 줄었기 때문이다.")
    assert wifi["datetime_columns"] and any(j["to"] == "2020-10" for j in wifi["monthly_jumps"])


def test_packager_syntax():
    import runpy

    module = runpy.run_path(str(ROOT / "scripts" / "package_submission.py"))
    for path in (ROOT / "scripts" / "package_submission.py", ROOT / ".env.example"):
        cleaned, _ = module["scan_text"](path, path.read_text(encoding="utf-8"))
        if path.suffix == ".py":
            compile(cleaned, str(path), "exec")
        else:
            assert "ANTHROPIC_API_KEY=" in cleaned


def test_midnight_hours():
    assert extract_hours("밤 12시 이후에 버스 있어?", "commute") == (0, 5)
    assert extract_hours("퇴근 시간을 8시 이후로 바꾸면?", "commute") == (20, 23)


def test_followup_clock():
    from src.agent.intent_parser import merge_dialogue_context, parse_question

    first = parse_question("삼화지구 퇴근 버스 어때?")
    second = merge_dialogue_context(parse_question("그럼 8시 이후는?"), first)
    assert (second.hour_from, second.hour_to) == (20, 23)


def test_date_columns():
    report = audit_uploaded_csv((SAMPLES_DIR / "bus_route_info_mangled.csv").read_bytes(), "")
    assert report["verdict"] != "thin_denominator" and not report["durations"]
    assert report["restored_codes"] >= 1


def test_stream_cancel():
    pulled = {"count": 0}

    class SlowStream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def __iter__(self):
            for index in range(40):
                pulled["count"] += 1
                time.sleep(0.02)
                yield SimpleNamespace(type="content_block_delta",
                                      delta=SimpleNamespace(type="text_delta", text=f"{index}"))

        def get_final_message(self):
            return SimpleNamespace(stop_reason="end_turn", content=[], usage=None)

    client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kwargs: SlowStream()))
    generator = ClaudeBridge("sk-ant-test", client=client).stream_once([{"role": "user", "content": "x"}])
    next(generator)
    generator.close()
    time.sleep(0.3)
    assert pulled["count"] < 40


def test_card_coverage():
    from src.tools.analysis_cubes import load_card, load_regions, stamp_db
    from src.tools.db_connector import fetch_scalar

    stamp = stamp_db()
    card = load_card(stamp)
    regions = set(load_regions(stamp)["emd_nm"])
    assert regions <= set(card["emd"]), sorted(regions - set(card["emd"]))
    assert abs(card["use_amt"].sum() - fetch_scalar("SELECT SUM(use_amt) FROM card_time_monthly")) < 1
    jeju_wolpyeong = fetch_scalar("SELECT SUM(use_amt) FROM card_time_monthly WHERE bjd = '월평동' "
                                  "AND dstrct_se LIKE '제주시%'")
    ara_only = fetch_scalar("SELECT SUM(use_amt) FROM card_time_monthly WHERE bjd IN "
                            "(SELECT bjd FROM legal_dong_map WHERE emd_nm = '아라동' AND bjd != '월평동')")
    ara = card.loc[card["emd"] == "아라동", "use_amt"].sum()
    assert abs(ara - (ara_only + jeju_wolpyeong)) < 1
    jeongbang = card[card["emd"] == "정방동"]
    assert abs(jeongbang["split_amt"].sum() - jeongbang["use_amt"].sum()) < 1  # 서귀동 균등 배분분만 있음


def test_failed_verdict(build_orchestrator, monkeypatch):
    import src.tools.tool_registry as registry

    def raise_failure(**kwargs):
        raise ValueError("forced failure")

    monkeypatch.setitem(registry.TOOL_FUNCTIONS, "decompose_confounders", raise_failure)
    bundle = build_orchestrator.answer("애월이랑 아라동 저녁 생활 인구 비교해 줘")
    assert bundle["verdict"] == "inconclusive", bundle["headline"]
    assert "판정을 유보" in bundle["headline"] and "역전 없음" not in bundle["headline"]
    assert any("결과를 내지 못했습니다" in line for line in bundle["limitations"])


def test_label_normalizer():
    from src.tools.resilient_loader import load_any_csv

    raw = ("기준일자,요일,성별,gender,이용자수\n20260501,금요일,M,female,10\n20260502,토,남,여,**\n"
           "20260503,일요일,여성,F,7\n").encode("cp949")
    table = load_any_csv(io.BytesIO(raw))
    assert table.frame["요일"].tolist() == ["금", "토", "일"]
    assert table.frame["성별"].tolist() == ["남성", "남성", "여성"]
    assert table.frame["gender"].tolist() == ["여성", "여성", "여성"]
    assert any("통일" in note for note in table.notes)
