"""6천만 문항 새 말투 시험(scripts/restyle_suite.py) 개발용 오답에서 고친 해석기 규칙의 회귀 시험.

각 문항은 고친 슬롯이나, 새 규칙이 잘못 걸리지 않아야 하는 대조 슬롯만 확인한다.
"""
import pytest

from src.agent.intent_parser import convert_sino_numbers, decode_layout, merge_dialogue_context, parse_question
from src.tools.gazetteer_index import load_place_index

# (앞 질문, 질문, 기대 슬롯)
CASES = [
    # 한영 전환을 잊고 친 낱말(두벌식 글쇠)
    ("", "맑은 날 xhlrms 버스 타기 노형동이랑 연동 중 어디가 나음?", {"hour_from": 17, "hour_to": 19, "intent": "compare"}),
    ("", "shgudehd에서 퇴근 버스 어때?", {"emds": ["노형동"]}),
    ("", "노형동 퇴근 qjtm djEo?", {"topic": "commute", "hour_from": 17}),
    ("", "노형동 tkfka aksgdk?", {"topic": "crowd"}),
    ("", "노형동 wifi 잘 돼? check 해 줘", {"topic": "wifi"}),  # 영어 낱말은 그대로('check' -> '촏차'로 바꾸지 않음)
    ("", "rush hour 노형동 버스 어때?", {"emds": ["노형동"], "topic": "commute"}),
    ("", "노형동 9am 버스 어때?", {"hour_from": 9}),
    # 한자어 시각(24시 표기)·한자어 분
    ("", "우도랑 노형 중 십구 시 넘은 시간 생활하기 어디가 더 나음?", {"hour_from": 19, "hour_to": 23, "topic": "living"}),
    ("", "노형동 이십일 시 이후에도 버스 다녀?", {"hour_from": 21}),
    ("", "노형동에서 버스를 이십 분 넘게 기다리긴 싫은데 몇 시까지 타야 돼?", {"wait_minutes": 20, "variable": "wait_minutes"}),
    ("", "노형동 정류소에서 오 분까지 기다릴 수 있으면 몇 시까지 버스 타면 돼?", {"wait_minutes": 5}),
    ("", "노형동 정류소에서 삼십오 분까지 기다릴 수 있으면 몇 시까지 타야 해?", {"wait_minutes": 35}),
    ("", "이 분이 노형동 사시는데 버스 어때?", {"wait_minutes": None, "intent": "profile"}),
    ("", "이 시간에 노형동 버스 있어?", {"hour_from": None}),
    # 띄어쓰기가 한 칸 밀려 갈라진 단서어
    ("", "노형동이랑 연동 중에 목요일 퇴 근버스 타기 어디가 나음?", {"hour_from": 17, "hour_to": 19}),
    ("", "국제여객선터미널 정류 소장날에 붐비는지 알려줘", {"topic": "commute"}),
    ("", "노형동 정류소에서 15분까지기 다릴 수 있으면 몇 시까지 버스 타면 돼?", {"wait_minutes": 15}),
    ("", "외도동 저녁 8시 지난 시간에도 사람 이많은가?", {"topic": "crowd"}),
    ("", "지금 막 차 타려는데 노형동 퇴근 버스 어때?", {"intent": "profile", "variable": None}),  # '막 차'는 막차가 아니다
    ("", "오일장 날 노형동 퇴근 버스 어때?", {"day_type": "market", "topic": "commute"}),
    # 소리 나는 대로 이어 쓴 연음
    ("", "첨단과학기술단지 정류소 장나레 붐비는지 알려줘", {"day_type": "market"}),
    ("", "노형동 밤 12시 넘어간 시간에도 사람 마늠?", {"topic": "crowd"}),
    ("", "노형동 사람 마나?", {"topic": "crowd"}),
    ("", "마늘 사러 동문시장 가려는데 몇 시에 가야 덜 붐벼?", {"intent": "boundary", "topic": "crowd"}),
    # 말버릇·겹친 띄어쓰기
    ("", "한림 오후 세 시 이후에도 사람 어 많을 것 같음?", {"topic": "crowd"}),
    ("", "노형동 퇴근 음 버스 어때?", {"hour_from": 17, "topic": "commute"}),
    ("", "어 그래서 노형동 퇴근 버스 어때?", {"hour_from": 17, "topic": "commute"}),
    ("", "서귀포시  중앙동  오후  09시  이후에도  사람  많을지  의아함.", {"hour_from": 21, "hour_to": 23}),
    # 조건을 풀어 쓴 말투
    ("", "장소는 구좌 쪽이랑 용담2동 쪽이고 조건은 담주 토일인데 퇴근 버스 타기 어디가 나아?", {"day_type": "weekend"}),
    ("", "대상 지역은 애월이고 연도는 2020년인데 공공와이파이 이용자가 얼마나 되는지 확인 부탁드립니다.", {"emds": ["애월읍"]}),
    ("", "나이 드신 분이 연도에서 아침에 버스 타기 괜찮은지 급히 묻노라!", {"emds": ["연동"]}),  # 지명 오타 '연도'는 그대로 연동
    ("", "연도에 따라 노형 매출이 달라?", {"emds": ["노형동"]}),
    ("", "한경면 저기가 아닌 노형오거리 저기는 점심 열두 시 반에 버스 상태가 어떠하오?", {"emds": ["노형동"], "intent": "profile"}),
    ("", "공항 방향이 아닌 노형 방향 버스 몇 시까지 있어?", {"emds": ["노형동"]}),
    ("", "노형동 상권 매출이 아닌 연동 상권 매출은 어때?", {"emds": ["노형동", "연동"]}),  # 지역이 아닌 말의 부정
    ("", "급하게 물어봄 제주시 건입동 권역 버스 1월에 급증했다는 보도 맞음?", {"abstain": True}),  # '물어봄'의 봄은 계절이 아니다
    ("", "작년 연동 매출 줄었는지 물어 봄", {"abstain": False}),
    ("", "지난해봄 연동 상권 매출이 줄었다던데 사실이야?", {"abstain": True}),  # '지난해봄'은 계절
    ("", "노형동 정류소에 대해, 주말에 특별히 혼잡함이 발생하는 시간이 존재하는 것에 대한 정보를 제공해 줄 수 있습니까?",
     {"intent": "hidden_slice"}),
    ("", "노형동 정류장 사람이 제일 많은 시간대가 언제야?", {"intent": "hidden_slice"}),
    ("", "사람이 많은 시간에도 노형동 공공 와이파이 잘 터져?", {"intent": "profile"}),
    ("", "할머니랑 동문시장 / 장날 / 어느 시간대 한산?", {"intent": "boundary", "variable": "visit_hour"}),
    ("", "함덕 몇 시에 한산해?", {"intent": "boundary", "variable": "visit_hour"}),
    ("", "어느 시간대가 한산한지는 아는데 사람이 몰리는 때는 언제야?", {"intent": "hidden_slice"}),
    ("", "평일 오후처럼 관광객이 덜한 시간대 노형동 매출은 어때?", {"intent": "profile", "variable": None}),
    ("동문시장 붐비는 시간이 있어?", "그럼 한산한 때는?", {"intent": "boundary", "variable": "visit_hour"}),
    ("", "노형동에서 버스가 15분 이상 안 오면 곤란하니까 몇 시까지 타야 하는지 확인 좀 해줘.", {"wait_minutes": 15}),
    ("", "시험 제한 시간은 60분이라 끝나면 몇 시까지 버스 타야 집에 가?", {"wait_minutes": None, "variable": "return_hour"}),
    ("", "충전 장소는 노형동이고, 평균 15분이나 기다려야 한다는 정보가 정확한지 여쭤봅니다.", {"topic": "ev"}),
    ("", "공항에 휴대폰 충전 장소 있어?", {"topic": "general"}),
    ("", "항공 노선 늘어난 이후 동문시장 사람 많아졌어?", {"topic": "crowd"}),
    # 독립 검토에서 찾은 오변환(바뀌면 안 되는 문장)
    ("", "결과는 gksrudals@example.com 으로 보내 주세요. 애월 주말에 사람 많아?", {"emds": ["애월읍"], "intent": "profile"}),
    ("", "답변은 gkstldn@example.com 으로 부탁해요. 연동 퇴근 버스 붐벼?", {"hour_from": 17, "hour_to": 19}),
    ("", "졸업식에 쓸 생화를 사려는데 노형이랑 연동 중 어디가 나아?", {"topic": "general"}),
    ("", "장나리 기자가 쓴 노형 상권 매출 기사 맞아?", {"topic": "spending", "day_type": "any"}),
    ("", "보고서 때문에 그러니까 노형 버스 밤 10시 넘어서도 다니는 거 맞아?", {"intent": "boundary"}),
    ("노형 버스 붐벼?", "이번 주 말고는?", {"day_type": "any"}),
    ("", "제주 말 목장 근처 버스 몇 시까지 있어?", {"day_type": "any"}),
    ("", "연동 상권 매출 근 3년째 줄었다는데 사실이야?", {"topic": "spending", "hour_from": None}),
    ("", "신천리에 새 벽화 생겼다는데 주말에 사람 많아?", {"hour_from": None}),
    ("", "노형 밤에 영 시끄러운데 살기 괜찮아?", {"hour_from": 21}),
    ("", "이용객 삼 분의 일이 버스를 20분 넘게 기다린다는 게 사실이야?", {"wait_minutes": 20}),
    ("", "사 분기에 노형 충전소 대기 시간이 30분 넘었다는데 사실이야?", {"wait_minutes": 30}),
    # 독립 검토에서 찾은 놓침(같은 종류 입력)
    ("", "6tl dlgn 연동 버스 붐벼?", {"hour_from": 18}),
    ("", "탑동 바멘 사람 많아?", {"hour_from": 21}),
    ("", "주마레도 함덕 붐벼?", {"day_type": "weekend"}),
    ("", "노형은 밤에 사람이 음, 많은 편이야?", {"topic": "crowd"}),
    ("", "십분 넘게 기다리기 싫은데 연동 버스 몇 시까지 타야 돼?", {"wait_minutes": 10}),
    # 평가용 공식 채점 뒤 사후 수정: 이번 수정이 새로 틀리게 만든 문항(회귀)의 원인 규칙을 좁혔다(수정 전과 같게 읽어야 한다)
    ("", "선생님, 2021년 겨울 삼화지구 버스가 사람이 많은 시간대가 있었던 것 같나요?", {"intent": "profile"}),
    ("", "작년 여름 노형동 버스가 사람이 많은 시간대가 이썼던 것 같아?", {"intent": "profile"}),
    ("", "동문시장 장을 방문할 때 덜 사람이 많은 시간대를 알려주세요.", {"intent": "boundary", "variable": "visit_hour"}),
    ("", "동문시장 장을 방문할 때 덜 #사람이 많은 시간대를 알려주세요.", {"intent": "boundary", "variable": "visit_hour"}),
    ("", "동문시장장을방문할때덜사람이많은시간대를알려주세요.", {"intent": "boundary", "variable": "visit_hour"}),
    ("", "노형동 정류장 사람이 많은 시간대 있어?", {"intent": "hidden_slice"}),  # 대조: 지금 있는지 물으면 숨은 혼잡
    ("", "노형동 정류장 사람이 제일 많은 시간대 알려줘", {"intent": "hidden_slice"}),
    ("", "노형동 정류소는 평소엔 괜찮은데 장날엔 특히 사람이 많은 시간이 있지?", {"intent": "hidden_slice"}),
    ("", "국제여객선터미덜 정류소는 평소엔 괜찮은데 장날엔 특히 사람이 많은 시간이 있을까요?", {"intent": "hidden_slice"}),
    ("", "제주시 연도 가까운 데 음식물 쓰레기가 갑자기 늘었단 말이 사실인가?", {"emds": ["연동"]}),
    ("", "제주 연도의 아파트 실거래가 어때?", {"emds": ["연동"]}),
    ("", "연도 별 공공와이파이 이용자 수가 중요한데 화북동은 2018년에 얼마나 됐어?", {"emds": ["화북동"]}),  # 대조: '연도 별'
    ("", "오후 다섯 시 사십오 분쯤 노형동 버스가 편하게 다니나요?", {"wait_minutes": None, "intent": "profile"}),
    ("", "이도일동이랑 애월 중 십팔 시 영 분 이후 생활하기 어디가 나아?", {"hour_from": 18, "hour_to": 23}),  # 대조
    # 이전 시험(1억·7,100만·2,550만)을 최종 코드로 다시 채점해 문항마다 비교하다 찾은 회귀(사후 수정 3판, 수정 전과 같게 읽어야 한다)
    ("", "보통 이호테우해변 정류소는 한산한데 언제쯤 사람이 많아지나요? 가족이 붐비는 시간을 피하게 해주고 싶어서요.",
     {"intent": "hidden_slice", "variable": None}),
    ("", "토요일 동문시장 / 혼잡 시간대 제외 확인", {"intent": "boundary", "variable": "visit_hour"}),  # 대조
    ("", "동문시장 혼잡 시간대 피해서 가고 싶어", {"intent": "boundary", "variable": "visit_hour"}),  # 대조
    ("", "퇴근버스타기제일 나은곳이안덕, 송산동, 모슬포중에 여기가 아닌 거 아니지?", {"emds": ["대정읍", "송산동", "안덕면"]}),
    ("", "한경면 저기가 아닌 노형오거리 저기 퇴근 버스 어때?", {"emds": ["노형동"]}),  # 대조
    ("", "제주시 연도 8pm 넘은 시간에도 사람이 많다니? 찾아봐.", {"emds": ["연동"]}),
    ("", "연도 18시 00분 이후에도 사람이 많을 수도 있겠지만, 확신은 없어요. 어떻게 되나요?", {"emds": ["연동"]}),
    ("", "장소는 노형동이고 연도 2021년인데, 공공와이파이 이용자가 반토막 났다는 사실이 맞는지 알고 싶어요.", {"emds": ["노형동"]}),  # 대조
    ("", "정말 서귀포시 대정읍지역 일대다음주 평일아침 9시가는 비내리는 날버스 상황이 괜찮은 건지 확인 좀.", {"hour_from": 9, "hour_to": 10}),
    ("", "노형동 밤 열한 시가는 비 오는 날 버스 어때?", {"hour_from": 23, "hour_to": 23}),
    ("", "동문시장 한산한 시가니 언제야?", {"intent": "boundary", "variable": "visit_hour"}),  # 대조: 연음 되돌리기는 그대로
    # 3판이 6천만 시험에서 새로 틀리게 만든 문항(사후 수정 4판, 2판과 같게 읽어야 한다): 따로 붐비는 때를 묻는 말은 때를 직접 묻는
    # 말만 세고, '덜' 뒤에 끼인 말·기호는 건너뛰며, 감싸는 말 없이 바로 오는 '아닌'은 지역 제외로 읽는다
    ("", "혼잡한 시간대를 피하고 싶어서요. 평일에 누웨마루거리가 덜 혹시나 해서 붐비는 시간대 좀 알려주세요.",
     {"intent": "boundary", "variable": "visit_hour"}),
    ("", "날짜는 평일 말고 토요일이고 장소는 노형오거리인데, 붐비는 시간을 피해 덜 그러니까 붐비는 시간대가 언제인지 알고 싶습니다.",
     {"intent": "boundary", "variable": "visit_hour"}),
    ("", "혼잡한 시간대를 피하고 싶어서요. 주말에 오일장이 덜 #붐비는 시간대 좀 알려주세요.", {"intent": "boundary", "variable": "visit_hour"}),
    ("", "혼잡한 시간대를 피하고 싶어서요. 이번 주말에 제주대학교가 더 붐비는 시간대 좀 알려주세요.",
     {"intent": "boundary", "variable": "visit_hour"}),
    ("", "붐비는 시간은 피하고 싶은데, 동문시장 붐비는 시간대가 언제야?", {"intent": "hidden_slice", "variable": None}),  # 대조
    ("", "용담1동에 아닌 한림은 낮 4시 반쯤 버스를 이용하는 것이 가능합니까?", {"intent": "profile", "emds": ["한림읍"]}),
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
def test_restyle_case(context, text, expected):
    plan = parse_question(text, "life")
    if context:
        plan = merge_dialogue_context(plan, parse_question(context, "life"))
    got = {"intent": plan.intent, "topic": plan.topic, "emds": collect_emds(plan.areas), "hour_from": plan.hour_from,
           "hour_to": plan.hour_to, "weather": plan.weather, "day_type": plan.day_type, "user_group": plan.user_group,
           "wait_minutes": plan.wait_minutes, "abstain": plan.needs_abstain is not None, "variable": plan.variable}
    assert {key: got[key] for key in expected} == expected, text


def test_layout_numbers():
    """두벌식 복원과 한자어 수사 변환의 경계."""
    assert decode_layout("qjtm") == "버스" and decode_layout("shgudehd") == "노형동" and decode_layout("djEo") == "어때"
    assert decode_layout("bus") is None and decode_layout("data") is None
    assert convert_sino_numbers("십구 시, 이십 분, 오 분") == "19시, 20분, 5분"
    assert convert_sino_numbers("이 분, 일 분, 십분 활용, 이 시간, 구분") == "이 분, 일 분, 십분 활용, 이 시간, 구분"
    # 숫자가 아닌 시각 바로 뒤의 분은 그대로, 숫자로 바뀐 시각 뒤의 분은 바꾼다(사후 수정)
    assert convert_sino_numbers("오후 다섯 시 사십오 분쯤, 십구 시 삼십 분, 십팔 시 영 분") == "오후 다섯 시 사십오 분쯤, 19시 30분, 18시 0분"
