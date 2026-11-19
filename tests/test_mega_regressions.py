"""1억 문항 확장 시험(scripts/mega_suite.py) 개발용 오답에서 고친 해석기 규칙의 회귀 시험.

각 문항은 고친 슬롯이나, 새 규칙이 잘못 걸리지 않아야 하는 대조 슬롯만 확인한다.
마지막 묶음은 평가용 공식 채점 뒤 사후 수정(개발용 수정이 새로 틀리게 만든 문장을 되돌림)의 확인이다.
"""
import pytest

from src.agent.intent_parser import merge_dialogue_context, parse_question
from src.tools.gazetteer_index import load_place_index

COMPARE = "노형동이랑 연동 중에 {} 퇴근 버스 타기 어디가 나아?"
# (앞 질문, 질문, 기대 슬롯)
CASES = [
    # 시간대: 제주어 '아척', 늦은 저녁, 동틀 무렵(오전)
    ("", "노형동 아척에 버스 어때?", {"hour_from": 7, "hour_to": 9}),
    ("", "노형동 저녁 느지막이 버스 어때?", {"hour_from": 21, "hour_to": 23}),
    ("", "노형동 저녁 아주 늦게 사람 많아?", {"hour_from": 21, "hour_to": 23, "topic": "crowd"}),
    ("", "노형동 저녁에 늦게 출발해도 버스 있어?", {"hour_from": 18, "hour_to": 21}),
    ("", "노형동 동틀 무렵 5시부터 6시 사이에 버스 자주 와?", {"hour_from": 5, "hour_to": 6}),
    # 이용자
    ("", "고희를 넘기신 분이 노형동에서 버스 타기 괜찮을까?", {"user_group": "어르신"}),
    ("", "애들 학교 통학하는데 노형동 아침 8시 버스 괜찮음?", {"user_group": "청소년"}),
    ("", "노형동 애들 공부시키기 괜찮은 동네야?", {"user_group": None, "abstain": True}),
    # 날씨: 비 안 오는 날 / 비 / 그 밖
    ("", COMPARE.format("파란 하늘 보이는 날"), {"weather": "dry"}),
    ("", COMPARE.format("날씨 무진장 좋은 날"), {"weather": "dry"}),
    ("", COMPARE.format("우산 없이 다녀도 되는 날"), {"weather": "dry"}),
    ("", COMPARE.format("빗줄기 쏟아지는 날"), {"weather": "rain"}),
    ("", COMPARE.format("우산 없이는 못 나가는 날"), {"weather": "rain"}),
    ("", COMPARE.format("날씨 안 좋은 날"), {"weather": "any"}),
    ("", COMPARE.format("하늘이 흐린 날"), {"weather": "any"}),
    ("", COMPARE.format("빗줄기 하나 없는 날"), {"weather": "any"}),
    ("", "노형동 빨래방 있는 동네 살기 어때?", {"weather": "any", "topic": "living"}),
    ("애월읍 퇴근 버스 어때?", "궁금한 게 있는데 새로 이사 오게 되어 정말 기대되는데, 빨래 잘 마르는 날은 어때요?",
     {"weather": "dry", "topic": "commute", "emds": ["애월읍"]}),
    # 판정 보류 개념
    ("", "노형동 내 집 마련 비용 어때?", {"abstain": True}),
    ("", "내 집 마련하고 이사 왔는데 노형동 퇴근 버스 어때?", {"abstain": False, "topic": "commute"}),
    ("", "노형동 경적 소리 심해?", {"abstain": True}),
    ("", "노형동 환경적으로 살기 좋아?", {"abstain": False, "emds": ["노형동"]}),
    ("", "제주시일도2동이랑용담1동중오후8시이후거주환경적합성문의", {"abstain": False}),
    ("", "노형동 112 신고 건수 어때?", {"abstain": True}),
    ("", "노형동 주차장 회전율 어때?", {"abstain": True}),
    ("", "노형동 공기 퀄리티 어때?", {"abstain": True}),
    ("", "노형동 모의고사 성적 어때?", {"abstain": True}),
    ("", "노형동 만족 지수 어때?", {"abstain": True}),
    # 요일
    ("", "노형동 주말 아닌 평일 저녁 퇴근 버스 어때?", {"day_type": "weekday"}),
    ("", "노형동 평일 아닌 주말 저녁 버스 어때?", {"day_type": "weekend"}),
    ("", "노형동 월·화·수·목·금 버스 어때?", {"day_type": "weekday"}),
    ("", "노형동 월·화·수·목·금·토 버스 어때?", {"day_type": "any"}),
    # 모음 혼동 오타(단서어만 고치고 지명·다른 낱말 속 글자는 그대로)
    ("", "노형동 퇘근 버스 어때?", {"hour_from": 17, "hour_to": 19}),
    ("", "제주대랑 우도 중 저녁 셍활하기 어디가 나아?", {"topic": "living"}),
    ("", "실례가안된다면제주도제주시구좌읍일대,제주도제주시아라동일대,제주도협재일대중에서퇴근버스타기제일나은곳이어디인지여쭤봐도될까요?",
     {"emds": ["구좌읍", "아라동", "한림읍"]}),
    ("", "전기차충전기다리는시간데이터존재여부문의", {"intent": "catalog"}),
    # 주제어
    ("", "와파 관련해서 어떤 데이터가 있어?", {"intent": "catalog", "topic": "wifi"}),
    ("", "DRT 이용 관련해서 어떤 데이터들 있는지 궁금해!", {"topic": "commute"}),
    ("", "노형동 역세권 살기 어때?", {"topic": "living"}),
    ("", "노형동 사람 개많음?", {"topic": "crowd"}),
    ("", "진짜 제주시 화북동 동네 일대에서 충전 15분이나 기다려야대요? 가짜아니에요~", {"topic": "ev"}),
    # 주장 확인
    ("", "노형동 버스 증편 필요 없다는 거 찐임?", {"intent": "fact_check"}),
    ("", "노형동 버스 어때? 찐임?", {"intent": "profile"}),
    ("", "공무원 답변대로 제주시청 정류소 이용객 적어 증편 필요 없는 거 맞죠?", {"intent": "fact_check"}),
    ("", "노형동 버스 괜찮은 거 맞죠?", {"intent": "profile"}),
    # 숨은 혼잡
    ("", "아 진짜 국제여객선터미널 정류소 평소엔 한산한데 언제 사람 개많음?", {"intent": "hidden_slice"}),
    # 제외('아닌')와 대조
    ("", "현재 제주시 삼양동 일원이 아닌 오라동 일원은 오전 11시경 버스 상태를 확인해 주십시오.", {"emds": ["오라동"]}),
    ("", "오라동 지역이 아닌 삼도1동 부근 쪽은 오전 여덟 시쯤 버스 이용이 가능한지 알려주십시오.", {"emds": ["삼도1동"]}),
    ("", "연동은 아닌 것 같은데 노형동 퇴근 버스 어때?", {"emds": ["노형동"]}),
    ("", "노형동이 아닌가요? 연동 버스 어때?", {"emds": ["노형동", "연동"]}),
    ("", "퇴근 버스 타기 제일 나은 곳이 도두, 탑동, 화북공업단지 중에 여기가 아닌 거 아니지?",
     {"emds": ["도두동", "삼도2동", "화북동"]}),
    # 대기
    ("", "산방산 정류소에서 15분까지 킵할 수 있다면 몇 시까지 버스 타야 하는지 알려줘.", {"wait_minutes": 15}),
    ("", "노형동 정류소에서 10분 스킵하고 몇 시까지 타야 돼?", {"wait_minutes": None}),
    ("", "화북 그 언저리 버스 25분 이내 승차 가능한 마지막 시각", {"wait_minutes": 25}),
    # 지명: 근사 비교에 걸리던 낱말, 붙여 쓴 '거기'
    ("", "노형동 환경에서 퇴근 버스 어때?", {"emds": ["노형동"]}),
    ("", "노형동 변동 사항 있어?", {"emds": ["노형동"]}),
    ("", "한림 카드 매출 변동 관광객 영향 확인", {"emds": ["한림읍"]}),
    ("", "비 오는 계절인 환경에서 협재 주변 쪽이랑 제주시 일도1동 주변 쪽 중 어느 곳이 출근 버스 타기에 더 적합한지 문의하고자 합니다.",
     {"emds": ["일도1동", "한림읍"]}),
    ("", "거기우도랑거기삼도1동중에퇴근버스타기어디가나아?", {"emds": ["삼도1동", "우도면"]}),
    ("", "그건입니다. 제주시 화북동 정류소에서 20분까지 대기 가능할 때 몇 시까지 버스 타면 되나요?", {"emds": ["화북동"]}),
    # 평가용 공식 채점 뒤 사후 수정: 위 수정이 새로 틀리게 만든 두 가지를 되돌린 규칙과 대조
    ("", "작년 여름 노형동 버스에 사람들이 붐비는 시기였나요?", {"intent": "profile", "abstain": True}),
    ("", "혹시 재작년 겨울 연동 버스가 붐비는 시기인가요?", {"intent": "profile", "abstain": True}),
    ("", "2018년 가을 노형동 버스가 특히 붐비는 시기여던 것으로 알고 있나요?", {"intent": "profile", "abstain": True}),
    ("", "노형동 정류소 평소엔 한산한데 붐비는 시기 있어?", {"intent": "hidden_slice"}),
    ("", "집에 가려면 한결에서 아홉 시 넘어서 버스가 있어?", {"emds": ["한경면"]}),
    ("", "한결 편할 것 같습니다 노형동 버스 어때?", {"emds": ["노형동"]}),
    ("천지동 퇴근 버스 사정 어때요?", "이건 팔십 대 할아버지 기준으로 보면 정말 충경적이야!", {"abstain": False}),
    ("", "노형동 차량 경적 많이 울려?", {"abstain": True}),
    ("", "천지동에 숙소 잡으려는데 오전에 버스가 편하다는 거 찐짜?", {"intent": "profile"}),
    ("", "노형동 버스 증편 필요 없다는 거 찐짜야?", {"intent": "fact_check"}),
]


def collect_emds(labels) -> list[str]:
    index = load_place_index()
    found = set()
    for label in labels:
        if label in index.alias_rows:
            found.update(index.alias_rows[label]["targets"])
        elif label in index.regions:
            found.add(label)
        else:
            match = index.match_token(label)
            found.update(match.targets if match else [f"?{label}"])
    return sorted(found)


@pytest.mark.parametrize("context, text, expected", CASES)
def test_mega_case(context, text, expected):
    plan = parse_question(text, "life")
    if context:
        plan = merge_dialogue_context(plan, parse_question(context, "life"))
    got = {"intent": plan.intent, "topic": plan.topic, "emds": collect_emds(plan.areas), "hour_from": plan.hour_from,
           "hour_to": plan.hour_to, "weather": plan.weather, "day_type": plan.day_type, "user_group": plan.user_group,
           "wait_minutes": plan.wait_minutes, "abstain": plan.needs_abstain is not None, "variable": plan.variable}
    assert {key: got[key] for key in expected} == expected, text
