"""자모 퍼지 파서·별칭·멀티턴 문맥 계승·판정 보류 감지."""
from src.agent.intent_parser import merge_dialogue_context, parse_question


def ask(question, previous=None, workspace="life"):
    return merge_dialogue_context(parse_question(question, workspace), previous)


def test_alias_resolution():
    plan = ask("신제주랑 구제주 중 퇴근 후 저녁 생활 어디가 나아?")
    assert plan.areas == ["신제주", "구제주"]
    assert (plan.hour_from, plan.hour_to) == (18, 21)
    assert plan.intent == "compare"


def test_jamo_typo():
    plan = ask("노형똥이랑 아라동 비교해줘")
    assert plan.areas == ["노형동", "아라동"]
    assert any("노형동" in note for note in plan.corrections)


def test_choseong_input():
    plan = ask("ㅅㅈㅈ 저녁 생활 어때?")
    assert plan.areas == ["신제주"]


def test_followup_merge():
    first = ask("삼화지구로 이사 가면 퇴근 버스 괜찮을까? 아라동이랑 비교해 줘")
    second = ask("그럼 퇴근 시간을 8시 이후로 바꾸면?", first)
    assert second.areas == first.areas
    assert (second.hour_from, second.hour_to) == (20, 23)
    assert second.intent == "boundary" and "지역" in second.inherited
    third = ask("비 안 올 때는?", second)
    assert third.weather == "dry" and third.areas == first.areas


def test_replace_partner():
    first = ask("삼화지구랑 아라동 퇴근 버스 비교해 줘")
    second = ask("거기 말고 노형동이랑 비교해 줘", first)
    assert second.areas == ["삼화지구", "노형동"]


def test_abstain_detection():
    assert ask("노형동 집값 비싸?").needs_abstain
    assert ask("작년 겨울 노형동 밤 버스 붐볐어?").needs_abstain
    assert ask("노형동 저녁 버스 붐벼?").needs_abstain is None


def test_market_condition():
    plan = ask("첨단과학기술단지 정류소는 오일장 장날에 평소보다 붐빈다던데 사실이야?", workspace="lawyer")
    assert plan.areas == ["아라동"] and plan.day_type == "market"


def test_independent_claims():
    first = ask("한림읍 음식물쓰레기가 58.69배 폭증했다. 관광객 때문이다.", workspace="lawyer")
    second = ask("버스 이용이 줄어든 건 전기차 보급 때문이다", first, workspace="lawyer")
    assert second.areas == [] and second.topic == "commute"
