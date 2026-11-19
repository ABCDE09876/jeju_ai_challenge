"""LLM 변형 질문 견고성 시험(scripts/run_robustness_test.py)에서 찾은 해석기 오류의 회귀 테스트."""
import runpy
from pathlib import Path

from src.agent.intent_parser import extract_hours, merge_dialogue_context, parse_question

ROOT = Path(__file__).resolve().parents[1]


def ask(question, previous=None, workspace="life"):
    return merge_dialogue_context(parse_question(question, workspace), previous)


def test_canonical_seeds():
    module = runpy.run_path(str(ROOT / "scripts" / "run_robustness_test.py"))
    seeds = module["load_jsonl"](ROOT / "data" / "meta" / "robustness_seeds.jsonl")
    items = [{"seed_id": s["id"], "family": s["family"], "workspace": s["workspace"], "context": s.get("context"),
              "reference": s["reference"], "style": "canonical", "text": s["text"]} for s in seeds]
    summary, misses = module["score_items"](items)
    assert len(seeds) == 364 and not misses, misses[:5]
    assert summary["parser_errors"] == 0


def test_topic_mask():
    assert ask("제주민속오일장 상권 저녁 매출은 도민이 많이 써?").topic == "spending"
    plan = ask("제주민속오일장 집값 어때?")
    assert plan.topic == "general" and plan.needs_abstain


def test_glued_places():
    assert ask("동홍동이랑 남원읍중 비오는날 퇴근버스타기 어디가나아").areas == ["동홍동", "남원읍"]
    plan = ask("생활하기 더 나은 곳이 애월읍이야 아니면 이도2동이야?")
    assert plan.areas == ["애월읍", "이도2동"] and plan.intent == "compare"
    assert ask("비 마니와서 연동카드 매출 줄었다는거 맞어?", workspace="lawyer").areas == ["연동"]
    plan = ask("서귀포 신시가기 밤 10시 넘어도 사람 많아?")
    assert plan.areas == ["서귀포 신시가지"] and plan.corrections
    assert ask("운행중인 버스 몇 대야?").areas == []


def test_after_clock():
    plan = ask("외도동에서 밤 9시가 넘은 시간에도 버스가 다니나요?")
    assert (plan.intent, plan.hour_from, plan.hour_to, plan.variable) == ("boundary", 21, 23, "return_hour")
    plan = ask("외도동 밤 9시 넘어서도 사람들이 많이 있어?")
    assert (plan.intent, plan.topic, plan.hour_from, plan.hour_to) == ("profile", "crowd", 21, 23)
    plan = ask("조천읍이랑 추자면 중 퇴근후 생활하기 어디가 나아?")
    assert (plan.topic, plan.hour_from, plan.hour_to) == ("living", 18, 21)
    assert extract_hours("노형동 0시부터 1시 사이 생활 인구는?", "living") == (0, 1)
    assert extract_hours("밤 10시부터 12시까지 사람 많아?", "crowd") == (22, 23)


def test_slice_when():
    question = "제주국제공항 정류소가 평소에는 한산하다는데 언제쯤이면 붐비는지 알려줘"
    assert ask(question).intent == "hidden_slice"
    assert ask(question, workspace="lawyer").intent == "fact_check"
    assert ask("노형오거리 정류소 붐비는 시간").intent == "hidden_slice"
    assert ask("함덕해수욕장 정류소 보통은 한산한데 언제 피크타임이야?").intent == "hidden_slice"


def test_visit_boundary():
    for question in ("오일장 장날 몇 시쯤 가야 덜 붐빌까?", "오일장 장날 덜 붐비는 시간", "이호테우해변 장 보러 몇시가 덜붐벼?"):
        plan = ask(question)
        assert (plan.intent, plan.variable) == ("boundary", "visit_hour"), question
    plan = ask("노형동 정류소에서 10분까지 웨이팅 가능하면 몇 시까지 버스 타면 돼?")
    assert (plan.wait_minutes, plan.variable) == (10, "wait_minutes")


def test_claim_variants():
    for question in ("노형동은 저녁에 도민이 연동보다 많다는 게 맞수꽈?", "노형동 저녁 도민 유동인구 연동보다 많음",
                     "첨단과학기술단지 정류소 평균 이용객 적어서 증편 필요 없다는 대답 맞어?"):
        assert ask(question, workspace="lawyer").intent == "fact_check", question
    plan = ask("전기차 보급 때문에 버스 이용이 줄어들었다는 기사가 맞는 거야?", workspace="lawyer")
    assert (plan.intent, plan.topic) == ("fact_check", "commute")
    assert ask("봉개동 음쓰 2019년에 급증했다는 주장 검증해 줘", workspace="lawyer").topic == "waste"


def test_slot_slang():
    assert ask("외도부영 정류소 화욜 오전에 붐벼?").day_type == "화"
    assert ask("신제주랑 구제주중 퇴근 후 저녁 생활 어디가 나아?").day_type == "any"
    assert ask("삼양동이랑 화북동 중에 weekday 퇴근 bus 타기 어디가 나아?").day_type == "weekday"
    assert ask("첨단과학단지 하우스 프라이스 어때?").needs_abstain
    assert ask("제주공항이랑 화북동 중에 어디가 living하기 더 나아?").topic == "living"
    assert ask("외도동 밤 9시 넘어서도 사람 하영 있수꽈?").topic == "crowd"


def test_short_followup():
    first = ask("천지동 퇴근 버스 어때?")
    second = ask("예래동 비교", first)
    assert second.areas == ["천지동", "예래동"] and second.topic == "commute" and second.intent == "compare"
    assert ask("그럼 rainy day는?", first).weather == "rain"
    third = ask("그럼 8시가 지난 뒤에는 어떻게 돼?", first)
    assert (third.intent, third.hour_from, third.hour_to) == ("boundary", 20, 23)
