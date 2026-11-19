"""다양성 대규모 시험(7천만 문항 이상)의 정의: 질문 유형 57종 x 말투 36종 x 상황 조합 x 확장 어휘 x 입력 변형 21종.

scripts/scale_suite.py(2,550만 문항 시험)를 넓힌 2차 시험이다. 값 끼우기·조사 맞춤·자판 오타 같은 공통 부품은
scale_suite 의 함수를 그대로 쓰고, 여기서는 무엇을 더 다양하게 만드는지만 정의한다.
- 질문 유형: 1차 27종 + 새 30종(막차, 세 곳 비교, '노형 말고 연동은', 출처를 밝힌 주장, 월·퍼센트 주장, 자료 목록 질문,
  밤 분위기, 매출 비교, 시간 범위, 후속 질문 장소·시각 바꾸기 등).
- 말투 36종(개발용 24 + 평가용 전용 12)과 상황(화자·경로·목적·감정 조합). LLM 이 문장틀을 쓸 때 반영한다.
- 끼울 값: 1차 값 + data/meta/diverse_vocab.json(여러 에이전트가 넓히고 교차 검증한 어휘). 값마다 정답 슬롯은 코드가 정한다
  (시간대 낱말은 해석기 설계서의 시간대 규약 PERIOD_WINDOWS, 시각은 그 시각, '몇 시 이후'는 그 시각부터 23시).
- 지명 감싸기('노형 쪽', '제주 노형', '노형 근처' 등)·후속 질문의 앞 질문 표현 12가지·세 곳 조합도 조합 차원으로 넣는다.
- 입력 변형 21종(1차 11종 + 부분 띄어쓰기 생략·문장 안 쉼표·긴 인사말·긴 맺음말·문장 중간 이모지·글자 늘이기·
  여러 군데 오타 + 평가용 전용 3종: 따옴표 감싸기·영어 덧말·지명 가운데 띄어쓰기).
- 평가용 전용: 말투 12종·상황 요소 9개·새 어휘와 지명 감싸기·앞 질문 표현의 약 1/4·입력 변형 3종은 개발용에 나오지 않는다.
  1차 값은 1차 시험과 같은 개발용/평가용 구분을 그대로 쓴다.
- 라벨 검증(check_meaning): LLM 이 쓴 틀이 원래 틀에 없던 시간·기간·날씨·요일·이용자·주장·판정 보류 개념·비교·경계·
  숨은 혼잡·주제·지명을 덧붙이거나 원래 있던 시간·요일·주제 조건을 빼면 해석기와 무관한 독립 어휘로 걸러 낸다.
  어휘 값도 같은 어휘로 교차 확인한다(예: '비 오는 아침'은 날씨 값이 될 수 없다).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

import scale_suite as base

META = base.META
PLACEHOLDER = base.PLACEHOLDER
SLOTS = base.SLOTS
SCALE_SEED = base.SCALE_SEED
compute_hash = base.compute_hash
pick_stride = base.pick_stride
check_attached = base.check_attached
SET_AWARE = True  # count_space·build_case 가 세트 이름을 받는다(세트마다 입력 변형 목록이 다름)

# ---------------------------------------------------------------- 말투 36종(개발용 24 + 평가용 전용 12)

STYLES = {
    "casual": "친구에게 묻는 반말 구어체",
    "polite": "해요체로 묻는 일상 존댓말",
    "formal": "하십시오체의 격식 있는 문의(고객센터·공공기관 문의 같은 말투)",
    "elderly": "나이 든 어르신이 묻는 듯한 길고 공손한 말투(말하는 사람 자신을 '어르신'이라고 부르지 않음)",
    "teen": "10대가 쓰는 줄임말·유행어 섞인 말투(뜻이 바뀌는 은어는 쓰지 않음)",
    "office": "바쁜 직장인이 메신저로 짧게 묻는 말투",
    "tourist": "제주에 처음 온 관광객이 조심스럽게 묻는 말투",
    "newcomer": "육지에서 제주로 이주한 지 얼마 안 된 사람이 묻는 말투",
    "jeju": "제주 사투리 어미와 표현을 섞은 말투(예: ~우꽈?, ~수과?, ~쿠과?, ~마씸, 하영=많이, 게메=글쎄)",
    "typo": "급하게 친 듯한 문장. 자판 오타·맞춤법 오류·띄어쓰기 오류를 2개 이상 넣는다(자리표시자는 건드리지 않음)",
    "search": "검색창에 치듯 명사 위주로 아주 짧게 줄인 질문",
    "story": "자기 사정을 한두 문장 먼저 말한 뒤 묻는 말투(사정 설명에 새 장소·시간·조건·숫자·사람을 넣지 않음)",
    "indirect": "'~인지 궁금합니다', '~알 수 있을까요'처럼 돌려서 묻는 말투",
    "command": "'알려줘', '정리해 줘', '찾아봐'처럼 시키는 말투",
    "reorder": "조건을 문장 앞으로 옮기는 등 어순을 바꾼 표준어 질문",
    "mixed": "영어 단어를 몇 개 섞은 질문(자리표시자와 숫자는 그대로, 문장 대부분은 한국어)",
    "emoji": "이모티콘·이모지·감탄사를 많이 붙인 말투",
    "hesitant": "'혹시', '아마', '~것 같은데'처럼 망설이며 묻는 말투",
    "worried": "걱정하거나 불안해하며 묻는 말투",
    "multi": "두세 문장으로 나눠 묻는 말투",
    "gyeongsang": "경상도 사투리 어미를 섞은 말투(예: ~나?, ~노?, ~데이, 억수로=많이)",
    "nominal": "'~여부 확인 요청', '~관련 문의'처럼 명사형으로 끝나는 말투",
    "negative": "'~아니지?', '~안 그래?'처럼 부정형·확인형으로 묻는 말투(묻는 내용과 조건은 같게)",
    "konglish": "외래어와 콩글리시를 섞은 말투(자리표시자와 숫자는 그대로)",
    "jeolla": "전라도 사투리 어미를 섞은 말투(예: ~당가?, ~했당께, ~잉, 겁나=많이)",
    "chungcheong": "충청도 사투리 어미를 섞은 말투(예: ~유?, ~했슈, ~혀?)",
    "dictation": "음성 인식으로 받아 적은 듯한 문장(문장부호 거의 없음, 띄어쓰기 들쭉날쭉, '어', '음' 같은 말버릇 섞임)",
    "learner": "한국어를 배우는 외국인이 쓴 듯한 문장(조사·어미가 조금 어색함)",
    "sms": "문자메시지처럼 줄임말·초성체를 섞은 말투(예: ㅇㅇ, ㄱㄱ, 넘=너무, ㅈㄱ=지금)",
    "kid": "어린아이가 묻는 듯한 쉬운 말투",
    "headline": "뉴스 제목처럼 짧게 요약한 말투(물음표로 끝냄)",
    "complaint": "민원 게시판에 올리는 불만 섞인 말투",
    "rhetorical": "과장하거나 감탄하며 묻는 말투",
    "quoted": "옆 사람의 질문을 그대로 옮겨 적는 말투(예: 옆자리 동료가 이렇게 물어봐요: ...)",
    "researcher": "보고서를 쓰는 연구자가 조건을 정확히 밝히며 묻는 말투",
    "groupchat": "친구 단톡방에 남기는 짧고 가벼운 말투",
}
HELDOUT_STYLES = ("jeolla", "chungcheong", "dictation", "learner", "sms", "kid", "headline", "complaint", "rhetorical",
                  "quoted", "researcher", "groupchat")
DEV_STYLES = tuple(s for s in STYLES if s not in HELDOUT_STYLES)

# ---------------------------------------------------------------- 상황(화자 x 경로 x 목적 x 감정) - 말투·표현에만 반영
SPEAKERS = ("제주 토박이", "육지에서 온 이주민", "외국인 유학생", "프리랜서", "대학원생", "자영업자", "공무원", "기자",
            "택시 기사", "간호사", "은퇴한 교사", "신입 사원")
CHANNELS = ("메신저 대화", "검색창", "음성 비서", "고객센터 문의 게시판", "지역 커뮤니티 게시글", "이메일", "단톡방",
            "전화 상담 메모")
PURPOSES = ("약속 장소를 정하려고", "동네 정보를 모으려고", "블로그 글을 쓰려고", "회사 보고서를 쓰려고",
            "가족에게 알려 주려고", "숙소를 고르려고", "기사를 확인하려고", "발표 자료를 만들려고", "친구에게 추천하려고")
EMOTIONS = ("급함", "걱정", "호기심", "짜증", "설렘", "의심", "피곤함", "기대")
HELDOUT_SITUATION = {"speaker": ("택시 기사", "간호사", "은퇴한 교사"), "channel": ("이메일", "전화 상담 메모"),
                     "purpose": ("발표 자료를 만들려고", "친구에게 추천하려고"), "emotion": ("피곤함", "기대")}


def pick_situation(number: int, heldout: bool) -> dict:
    """호출 번호로 상황 요소 네 가지를 고른다(개발용은 평가용 전용 요소를 빼고 고른다)."""
    picked = {}
    for offset, (name, options) in enumerate((("speaker", SPEAKERS), ("channel", CHANNELS), ("purpose", PURPOSES),
                                              ("emotion", EMOTIONS))):
        allowed = [o for o in options if heldout or o not in HELDOUT_SITUATION[name]]
        picked[name] = allowed[base.compute_mix(number * 7 + offset) % len(allowed)]
    return picked


def describe_situation(situation: dict) -> str:
    return (f"{situation['speaker']}이(가) {situation['channel']}에서 {situation['purpose']} "
            f"'{situation['emotion']}' 마음으로 묻는다")


def check_situation(situation: dict | None) -> bool:
    """상황에 평가용 전용 요소가 들어 있으면 True."""
    return bool(situation) and any(situation.get(name) in options for name, options in HELDOUT_SITUATION.items())


# ---------------------------------------------------------------- 질문 유형(1차 27종 + 새 30종)

FOLLOW = base.FOLLOW_CONTEXT
NEW_FRAMES = [
    {"id": "G01", "family": "profile_multi", "workspace": "life", "text": "<P1> <DAY2> <TIME> <WEATHER> 버스 어때?",
     "slots": {"P1": "area", "DAY2": "day2", "TIME": "time_point", "WEATHER": "weather"},
     "expect": {"intent": "profile", "topic": "commute"}},
    {"id": "G02", "family": "living_profile", "workspace": "life", "text": "<P1>에서 <EVE> 지내기 괜찮아?",
     "slots": {"P1": "area", "EVE": "evening"}, "expect": {"intent": "profile", "topic": "living"}},
    {"id": "G03", "family": "visit_elder_market", "workspace": "life",
     "text": "<OLD>을(를) 모시고 <MARKET>에 <DAY2> 가려는데 몇 시에 가야 덜 붐벼?",
     "slots": {"OLD": "old", "MARKET": "market", "DAY2": "day2"},
     "expect": {"intent": "boundary", "topic": "crowd", "variable": "visit_hour", "user_group": "어르신"}},
    {"id": "G04", "family": "youth_school", "workspace": "life",
     "text": "<YOUNG>이(가) <P1>에서 <MORNING> 버스 타고 학교 가는데 괜찮을까?",
     "slots": {"YOUNG": "young", "P1": "area", "MORNING": "morning"},
     "expect": {"intent": "profile", "topic": "commute", "user_group": "청소년"}},
    {"id": "G05", "family": "night_return", "workspace": "life", "text": "<P1>에서 <AFTER>에 일이 끝나면 집에 갈 버스가 있을까?",
     "slots": {"P1": "area", "AFTER": "after"},
     "expect": {"intent": "boundary", "topic": "commute", "variable": "return_hour"}},
    {"id": "G06", "family": "stay_access", "workspace": "life", "text": "<P1>에 숙소를 잡으려는데 <TIME> 버스 다니기 편해?",
     "slots": {"P1": "area", "TIME": "time_point"}, "expect": {"intent": "profile", "topic": "commute"}},
    {"id": "G07", "family": "compare_three", "workspace": "life", "text": "<P1>, <P2>, <P3> 중에 퇴근 버스 타기 제일 나은 곳은?",
     "slots": {"P1": "emd", "P2": "emd", "P3": "emd"},
     "expect": {"intent": "compare", "topic": "commute", "hour_from": 17, "hour_to": 19}},
    {"id": "G08", "family": "exclusion", "workspace": "life", "text": "<P1> 말고 <P2>은(는) <TIME> 버스 어때?",
     "slots": {"P1": "area", "P2": "area", "TIME": "time_point"}, "exclude_places": ["P1"],
     "expect": {"intent": "profile", "topic": "commute"}},
    {"id": "G09", "family": "claim_percent", "workspace": "lawyer",
     "text": "<P1> 저녁 유동인구가 <PCT>% 줄었다는 기사 사실이야?", "slots": {"P1": "area", "PCT": "pct"},
     "expect": {"intent": "fact_check", "topic": "general", "hour_from": 18, "hour_to": 21}},
    {"id": "G10", "family": "claim_month", "workspace": "lawyer", "text": "<P1> 버스 이용객이 <MONTH>월에 급증했다는 보도 맞아?",
     "slots": {"P1": "area", "MONTH": "month_commute"}, "expect": {"intent": "fact_check", "topic": "commute"}},
    {"id": "G11", "family": "claim_source", "workspace": "lawyer",
     "text": "<SOURCE>에서 <P1> 상권 매출이 줄었다는 얘기를 봤는데 사실이야?", "slots": {"SOURCE": "source", "P1": "area"},
     "expect": {"intent": "fact_check", "topic": "spending"}},
    {"id": "G12", "family": "catalog", "workspace": "life", "text": "<TOPICWORD> 관련해서 어떤 데이터가 있어?",
     "slots": {"TOPICWORD": "topicword"}, "expect": {"intent": "catalog"}},
    {"id": "G13", "family": "abstain_concept", "workspace": "life", "text": "<P1> 쪽 <CONCEPT>은(는) 괜찮은 편이야?",
     "slots": {"P1": "area", "CONCEPT": "concept"}, "expect": {"intent": "profile", "topic": "general", "abstain": True}},
    {"id": "G14", "family": "night_mood", "workspace": "life", "text": "<P1> <NIGHTWORD> 분위기 어때?",
     "slots": {"P1": "area", "NIGHTWORD": "nightword"}, "expect": {"intent": "profile", "topic": "night"}},
    {"id": "G15", "family": "spending_compare", "workspace": "life",
     "text": "<P1>(이)랑 <P2> 중 <EVE> 상권 매출은 어디가 더 커?", "slots": {"P1": "area", "P2": "area", "EVE": "evening"},
     "expect": {"intent": "compare", "topic": "spending"}},
    {"id": "G16", "family": "waste_month_claim", "workspace": "lawyer",
     "text": "<P1> 음식물 쓰레기가 <YM>에 갑자기 늘었다는 거 맞아?", "slots": {"P1": "area", "YM": "ym_waste"},
     "expect": {"intent": "fact_check", "topic": "waste"}},
    {"id": "G17", "family": "wifi_year", "workspace": "lawyer", "text": "<P1> 공공와이파이 이용자가 <YEAR>년에 얼마나 됐는지 확인해 줘",
     "slots": {"P1": "area", "YEAR": "year_wifi"}, "expect": {"intent": "fact_check", "topic": "wifi"}},
    {"id": "G18", "family": "ev_place_claim", "workspace": "lawyer",
     "text": "<P1>에서 전기차 충전하려고 기다리는 시간이 평균 <MIN>분이라는데 맞아?", "slots": {"P1": "area", "MIN": "ev_min"},
     "expect": {"intent": "fact_check", "topic": "ev"}},
    {"id": "G19", "family": "followup_place", "workspace": "life", "text": "그럼 <P2>은(는)?", "context": FOLLOW,
     "slots": {"P1": "area", "P2": "area"}, "exclude_places": ["P1"],
     "expect": {"intent": "profile", "topic": "commute", "hour_from": 17, "hour_to": 19}},
    {"id": "G20", "family": "followup_time", "workspace": "life", "text": "그럼 <TIME>은(는)?", "context": FOLLOW,
     "slots": {"P1": "area", "TIME": "time_point"}, "expect": {"intent": "profile", "topic": "commute"}},
    {"id": "G21", "family": "followup_daycombo", "workspace": "life", "text": "<DAY2>은(는)요?",
     "context": "<P1> <WEATHER> 퇴근 버스 어때?", "slots": {"P1": "area", "WEATHER": "weather", "DAY2": "day2"},
     "expect": {"intent": "profile", "topic": "commute", "hour_from": 17, "hour_to": 19}},
    {"id": "G22", "family": "boundary_wait_dislike", "workspace": "life",
     "text": "<P1>에서 버스를 <MIN>분 넘게 기다리긴 싫은데 몇 시까지 타야 돼?", "slots": {"P1": "area", "MIN": "wait_min"},
     "expect": {"intent": "boundary", "topic": "commute", "variable": "wait_minutes"}},
    {"id": "G23", "family": "hidden_slice_day", "workspace": "life",
     "text": "<STOP> 정류소 평소엔 괜찮은데 <DAY2> 특히 붐비는 시간이 있어?", "slots": {"STOP": "stop", "DAY2": "day2"},
     "expect": {"intent": "hidden_slice", "topic": "commute"}},
    {"id": "G24", "family": "crowd_day_time", "workspace": "life", "text": "<MARKET>은(는) <DAY2> <TIME> 사람 많아?",
     "slots": {"MARKET": "market", "DAY2": "day2", "TIME": "time_point"}, "expect": {"intent": "profile", "topic": "crowd"}},
    {"id": "G25", "family": "range_frequency", "workspace": "life", "text": "<P1>에서 <RANGE> 사이에 버스 자주 와?",
     "slots": {"P1": "area", "RANGE": "range"}, "expect": {"intent": "profile", "topic": "commute"}},
    {"id": "G26", "family": "last_bus", "workspace": "life", "text": "<P1>에서 막차가 몇 시쯤이야?",
     "slots": {"P1": "area"}, "expect": {"intent": "boundary", "topic": "commute", "variable": "return_hour"}},
    {"id": "G27", "family": "compare_group", "workspace": "life",
     "text": "<OLD>이(가) 다니기엔 <P1>(이)랑 <P2> 중 어디가 버스 타기 편해?",
     "slots": {"OLD": "old", "P1": "area", "P2": "area"},
     "expect": {"intent": "compare", "topic": "commute", "user_group": "어르신"}},
    {"id": "G28", "family": "claim_compare_spending", "workspace": "lawyer",
     "text": "<P1>이(가) <P2>보다 <EVE> 상권 매출이 더 높다는 주장 검증해 줘", "slots": {"P1": "area", "P2": "area", "EVE": "evening"},
     "expect": {"intent": "fact_check", "topic": "spending"}},
    {"id": "G29", "family": "visit_day", "workspace": "life", "text": "<DAY2>에 <MARKET> 덜 붐비는 시간대 알려줘",
     "slots": {"DAY2": "day2", "MARKET": "market"},
     "expect": {"intent": "boundary", "topic": "crowd", "variable": "visit_hour"}},
    {"id": "G30", "family": "weather_compare", "workspace": "life",
     "text": "<WEATHER>에는 <P1>(이)랑 <P2> 중 어디가 출근 버스 타기 나아?",
     "slots": {"WEATHER": "weather", "P1": "area", "P2": "area"},
     "expect": {"intent": "compare", "topic": "commute", "hour_from": 7, "hour_to": 9}},
]
PLACEHOLDER_NOTES = {**base.PLACEHOLDER_NOTES,
                     "P3": "세 번째 제주 지명", "DAY2": "요일·장날 조건(예: 주말, 평일, 장날, 토요일)",
                     "MORNING": "아침 시간 표현(예: 아침 7시에, 출근 시간에)", "PCT": "퍼센트 숫자(예: 20)",
                     "MONTH": "월 숫자(예: 6)", "SOURCE": "정보를 본 곳(예: 커뮤니티, 유튜브, 지역 신문)",
                     "TOPICWORD": "알고 싶은 자료의 주제(예: 버스, 음식물 쓰레기)",
                     "NIGHTWORD": "늦은 시간 표현(예: 밤늦게, 심야에)", "YM": "연도와 월(예: 2019년 12월)",
                     "RANGE": "시간 범위 전체(예: 저녁 6시부터 9시까지)"}
UNIT_AFTER = {**base.UNIT_AFTER, "PCT": "%", "MONTH": "월"}


def load_frames() -> list[dict]:
    """1차 기본 틀 27종(자리표시자 틀만) + 새 30종. 벤치마크·시나리오 고정 문장은 이번 시험에 넣지 않는다."""
    frames = [dict(frame, context=frame.get("context")) for frame in base.FRAMES]
    frames += [dict(frame, context=frame.get("context")) for frame in NEW_FRAMES]
    return frames


# ---------------------------------------------------------------- 라벨 검증용 독립 어휘
# 해석기 규칙을 가져오지 않고 뜻으로 새로 적었다. 이 어휘는 틀·값을 '걸러 내기'만 한다(정답을 만들지 않는다).

FLAGS = re.IGNORECASE
LEXICON = {
    "time": {
        "새벽": r"새벽|dawn",
        "아침": r"아침|이른\s*시간|모닝|morning",
        "출근": r"출근",
        "오전": r"오전|(?<![a-z])a\.?m(?![a-z])",
        "점심": r"점심|정오|lunch|noon",
        "낮": r"한낮|(?<![가-힣])낮\s*(?:에|엔|시간|\d)",
        "오후": r"오후|(?<![a-z])p\.?m(?![a-z])|afternoon",
        "퇴근": r"퇴근|칼퇴|(?<![가-힣])일\s*(?:이\s*)?(?:끝나|끝난|마치|마친)|업무\s*(?:가\s*)?(?:끝|종료|마치|마감)|회사\s*(?:가\s*)?(?:끝나|끝난|마치)"
                r"|일과\s*(?:후|끝)|off\s*work",
        "저녁": r"저녁|이브닝|evening|dinner|노을|해질\s*(?:녘|무렵)"
                r"|(?<![가-힣])해\s*(?:가\s*)?(?:질\s*(?:녘|무렵|때)|지는\s*(?:시간|때|무렵)|저물)",
        "밤": r"(?<![가-힣])밤|야간|심야|자정|한밤|늦은\s*(?:시간|밤)|밤늦|나이트|night",
        "귀가": r"귀가\s*(?:시간|길|할|하는|하기|하려|해야|때)|귀갓길",
        "막차": r"막차|첫차|막버스|마지막\s*(?:버스|차|배차|운행)|last\s*bus",
        "시각": r"(?<![가-힣])(?:한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|열한|열두)\s*시(?![간장민청내외설기작대적름])"
                r"|러시\s*아워|rush\s*hour",
    },
    "period": {"기간": r"(?:19|20)\d{2}|(?<![가-힣])봄(?![가-힣])|봄철|여름|가을|겨울|작년|올해|재작년|지난해|내년|상반기"
                     r"|하반기|연말|연초|월초|월말"},
    "weather": {
        "비": r"(?<![가-힣])비\s*(?:가|도|는|만|라도)?\s*(?:오|올|와|옴|온|내리|쏟|그치|많이|조금|좀|소식|예보)"
              r"|(?<![가-힣])빗|우천|우산|장마|소나기|폭우|호우|태풍|\brain",
        "맑음": r"맑|화창|쾌청|(?<![가-힣])비\s*(?:가\s*)?안\s*(?:오|올|와|옴|온)|sunny|\bdry\b",
        "기타": r"(?<![가-힣])눈\s*(?:이|도)?\s*(?:오|올|와|내리|쌓)|눈길|폭설|흐린|흐림|흐려|궂|강풍|바람\s*(?:이\s*)?(?:불|많|세)"
                r"|폭염|한파|무더|더운\s*날|추운\s*날|snow",
        "날씨일반": r"날씨|weather",
    },
    "day": {
        "평일": r"평일|(?<![가-힣])주중|weekday",
        "주말": r"주말|weekend|불금|불토",
        "요일": r"[월화수목금토일]요일|[월화수목금토일]욜",
        "요일일반": r"(?<![월화수목금토일])요일",
        "장날": r"장날|오일장|장\s*(?:이\s*)?(?:서는|설|섰|선\s*날|열리)",
        "휴일": r"공휴일|휴일|명절|연휴|빨간\s*날|holiday",
        "방학": r"방학|개학",
    },
    "group": {
        "어르신": r"어르신|노인|할머니|할아버지|할매|할배|부모님|노부모|고령|연세(?!\s*(?:집|살이|계약))|나이\s*(?:가\s*)?(?:드신|든|많)|시니어|노약자"
                  r"|노년|노령|은퇴|elderly|senior",
        "청소년": r"학생|청소년|아이들|(?<![가-힣])아이(?:가|를|는|도|랑|와|한테|에게|랑은)(?=\s)|아이\s*(?:둘|셋|하나|데리|키우)"
                  r"|애들|애기|아기|자녀|아들"
                  r"|(?<![가-힣])딸(?!기)|초등|중딩|고딩|초딩|중학|고등|10대|십대|미성년|유아|어린이|꼬마|학교(?!\s*(?:평판|순위|알리미|성적))|등교|하교|등굣"
                  r"|하굣|대학원생|\bkids?\b|student|teen",
        "기타": r"장애|휠체어|임산부|유모차",
    },
    "claim": {"주장": r"기사|(?<![가-힣])보도|뉴스|언론|신문|주장|소문|루머|카더라|팩트|검증|사실(?:이|인|일)|사실\s*(?:여부|확인)"
                      r"|진짜(?:야|예요|임|인가|인지|요\s*\?)|정말(?:이야|인가|인지|요\s*\?)|맞는\s*말"
                      r"|(?<!알)맞(?:아|나|지|죠|습|을까|는지|는\s*거|음|냐|니|어|제|수과|우꽈|수꽈|쥬|당가|소|으|느)"
                      r"|믿어|믿을|믿고|믿기|(?:다|라)는\s*(?:말|얘기|이야기|소리|글)"
                      r"|들었는데|들었어|들었습|들은\s*(?:얘기|말|바)|그러던데|하던데|라던데|다던데"
                      r"|다더라|라더라|라는데|다는데|라고들|다고들|그러더라|하더라|발표(?:에\s*따르|했|된|한\s*(?:거|게|내용))"
                      r"|(?:커뮤니티|유튜브|맘카페|인스타|블로그|게시판|단톡방?|카톡|트위터|SNS|방송)에서\s*"
                      r"(?:봤|보니|읽|들|그러|하던|말하|나왔|나온)"},
    "concept": {"개념": r"집값|집\s*값|월세|전세|매매|부동산|아파트|시세|소음|시끄|층간|치안|범죄|우범|위험|안전|학군|평판|명문"
                       r"|미세먼지|대기질|공기\s*(?:질|오염|가\s*좋|가\s*나쁘)|황사|주차|친절(?!하게|히)|만족|서비스\s*질|물가"
                       r"|임대료|땅값"},
    "compare": {"비교": r"비교|(?<![아어해봐])보다(?=\s|$|는|도|더|[?,.!])|어디가|어디\s*(?:가\s*)?(?:더|낫)"
                       r"|어느\s*(?:쪽|곳|동네|데|게)|\bvs\b|(?<![가-힣])대비|차이|(?<![가-힣])낫|나은|나아(?!가|지|진|졌)|나을|견주"},
    "boundary": {"경계": r"몇\s*시|언제|(?<![가-힣])이후|이전|까지|부터|기다|대기(?!질|\s*오염|\s*환경)|웨이팅|기달|방문(?!객)|몇\s*분|막차"},
    "slice": {"숨은혼잡": r"평균|숨은|숨겨|가려진|민원|증편|피크|러시|특정\s*시간"},
    "topic": {
        "commute": r"버스|버쓰|뻐스|대중교통|정류|배차|통근|막차|첫차|막버스|환승|노선|교통|\bbus(?:es)?\b",
        "crowd": r"붐|혼잡|북적|복잡|복작|인파|사람(?:들)?\s*(?:이\s*|도\s*|로\s*)?(?:많|몰|북적|붐|없|적|별로|꽉|있|잇|만|가득|하영)"
                 r"|한산|한적|한가(?:해|한|하|롭|로)|몰리|몰려|미어터|바글|꽉\s*차|\bcrowd|\bbusy\b|피크",
        "spending": r"매출|상권|장사|소비|(?<!교통)카드|(?<![가-힣])가게|손님|매상|지출|씀씀이|쇼핑|결제|돈\s*(?:을\s*)?(?:쓰|써)",
        "living": r"살기|사는\s*(?:게|곳|거|건|데)|살\s*(?:만|곳|데|집)|생활(?!\s*(?:쓰레기|폐기물|정보))|지내기|지낼\s*만|거주|주거|살림|라이프|리빙",
        "living_bg": r"이사|정착",  # '이사 와서 잘 몰라서'처럼 말하는 사람의 사정(덧붙어도 주제는 그대로, 빠짐 검사에서는 생활로 본다)
        "waste": r"쓰레기|음식물|음쓰|클린하우스|분리수거|배출",
        "wifi": r"와이파이|wifi|wi-fi|인터넷|무선",
        "ev": r"전기차|충전|\bEV\b",
        "night": r"(?<![가-힣])밤|야간|심야|새벽|한밤|나이트|밤늦|늦은\s*저녁|늦저녁|저녁\s*늦",
    },
}
LEXICON_PATTERNS = {(category, concept): re.compile(pattern, FLAGS)
                    for category, concepts in LEXICON.items() for concept, pattern in concepts.items()}
# 틀에 그 개념의 자리표시자가 있으면 같은 개념 낱말을 덧붙여도 정답이 바뀌지 않는다(예: <OLD> 틀의 '할머니 모시고')
SLOT_ALLOWED = {"OLD": {("group", "어르신")}, "YOUNG": {("group", "청소년")},
                "WEATHER": {("weather", "날씨일반")}, "COND": {("weather", "날씨일반"), ("day", "요일일반")},
                "OLDCOND": {("weather", "날씨일반"), ("day", "요일일반")}, "DAY": {("day", "요일일반")},
                "DAY2": {("day", "요일일반")}, "MORNING": {("time", "아침"), ("time", "출근")},
                "RANGE": {("boundary", "경계")}}
# 원래 틀에서 빠지면 묻는 뜻이 바뀌는 단서(틀별, 그중 하나는 남아 있어야 함). 자리표시자는 ◇ 로 바꾼 뒤 본다.
WHEN = r"언제|몇\s*시|시간대|어느\s*(?:시간|때|시각|시점)|어떤\s*(?:시간|때|시각|시점)|무슨\s*시간|타이밍|타임|피크|러시"
VISIT_WHEN = WHEN + r"|시간|시각|시점|(?<![가-힣])때"
KEEP_CUES = {
    "F07": VISIT_WHEN, "G23": VISIT_WHEN, "F09": VISIT_WHEN, "G03": VISIT_WHEN, "G29": VISIT_WHEN,
    "F08": VISIT_WHEN + r"|까지|시한|마감", "G22": VISIT_WHEN + r"|까지|시한|마감",
    "F18": r"비교|랑|하고|보다|vs|견주|대보|대비|차이|나아|나은|낫",
    "G08": r"말고|대신|빼고|제외|아니고|아닌",
    "G12": r"데이터|자료|데이타|data|통계|목록|정보",
    "G26": r"막차|마지막\s*(?:버스|차|운행|배차)|막버스|끊기|끊겨|끝나는\s*시간|운행\s*(?:종료|끝)",
    # 전기차 '충전 대기시간'이 '충전에 걸리는 시간'으로 바뀌면 대기시간 라벨이 맞지 않는다
    "F25": r"기다|대기|웨이팅|기달|줄\s*서|순서", "G18": r"기다|대기|웨이팅|기달|줄\s*서|순서",
}
KEEP_PATTERNS = {frame_id: re.compile(pattern, FLAGS) for frame_id, pattern in KEEP_CUES.items()}
# '퇴근 시간'(17~19시)을 뜻하는 틀이 '퇴근 후'(규약상 18~21시)로 바뀌면 시간대가 달라진다
AFTER_WORK_LIKE = re.compile(r"퇴근\s*(?:후|뒤|하고|끝나고|끝난|끝나면|이후)")
NOTATION_LEFT = re.compile(r"\((?:이|은|는|가|을|를|으|과|와|랑|이랑|이나|나)\)")
NEUTRAL = "◇"
PLACE_SLOTS = ("P1", "P2", "P3", "STOP", "MARKET")
AREA_SLOTS = ("P1", "P2", "P3")
CONJ_FORMS = {"랑", "랑은", "랑도", "과", "와", "나"}
SKIP_BY_INTENT = {"compare": {"compare"}, "boundary": {"boundary"}, "hidden_slice": {"boundary", "slice"}}


@lru_cache(maxsize=1)
def load_place_pattern() -> re.Pattern:
    """틀 본문에 덧붙인 지명을 찾는 정규식(사전 표기 전체 + 시 이름). '제주', '제주도'는 섬 이름이라 뺀다."""
    keys = sorted(base.load_place_keys()[0] | {"제주시", "서귀포시", "서귀포"}, key=len, reverse=True)
    return re.compile(r"(?<![가-힣])(?:" + "|".join(re.escape(k) for k in keys) + ")")


def make_plain(text: str) -> str:
    """자리표시자를 ◇ 로 바꾼 본문(조사 표기는 남긴다)."""
    return PLACEHOLDER.sub(NEUTRAL, text)


def find_concepts(text: str, background: bool = False) -> set[tuple[str, str]]:
    """본문에 보이는 (범주, 개념). background=False 면 사정 설명용 생활 낱말('이사')도 생활 주제로 센다."""
    found = {key for key, pattern in LEXICON_PATTERNS.items() if pattern.search(text)}
    if not background and ("topic", "living_bg") in found:
        found = (found - {("topic", "living_bg")}) | {("topic", "living")}
    return found


def check_notation(text: str) -> bool:
    """자리표시자에 붙지 않고 남은 조사 표기('(이)랑(은)'의 '(은)' 등)가 있으면 True."""
    rest, position = [], 0
    for match in PLACEHOLDER.finditer(text):
        rest.append(text[position:match.start()])
        position = match.end() + base.read_particle(text, match.end())[1]
    rest.append(text[position:])
    return bool(NOTATION_LEFT.search(" ".join(rest)))


def read_conj(text: str, name: str) -> bool:
    """자리표시자 바로 뒤가 '(이)랑'·'와(과)'·'보다'처럼 다른 장소와 잇거나 견주는 조사인지."""
    position = text.find(f"<{name}>")
    if position < 0:
        return False
    end = position + len(name) + 2
    form = base.read_particle(text, end)[0]
    return form in CONJ_FORMS or text.startswith("보다", end)


def check_meaning(frame: dict, text: str) -> str | None:
    """LLM 이 쓴 틀이 원래 틀의 뜻(정답 슬롯)을 바꿨는지 독립 어휘로 본다. 통과하면 None, 아니면 탈락 사유."""
    if check_notation(text):
        return "notation_leftover"
    plain, base_plain = make_plain(text), make_plain(frame["text"])
    found, base_found = find_concepts(plain), find_concepts(base_plain)
    allowed = set(base_found)
    for name in frame["slots"]:
        allowed |= SLOT_ALLOWED.get(name, set())
    if frame["expect"].get("topic") == "commute":
        allowed.add(("topic", "crowd"))  # 버스·정류소 질문의 '붐비나'·'혼잡도'는 버스 이용 주제 안의 말이다(F06 과 같은 규약)
    background = find_concepts(plain, background=True)
    if ("topic", "living_bg") in background and ("topic", "living") not in background:
        allowed.add(("topic", "living"))  # '이사 와서 잘 몰라서'는 사정 설명이지 생활 주제 질문이 아니다
    skipped = set(SKIP_BY_INTENT.get(frame["expect"]["intent"], set()))
    if frame["workspace"] == "lawyer":
        skipped |= {"claim", "compare", "boundary", "slice"}
    for category, concept in sorted(found - allowed):
        if category not in skipped:
            return f"added_{category}"
    for category, concept in sorted(base_found - found):
        if category in ("time", "day", "topic"):
            return f"dropped_{category}"
    if load_place_pattern().search(plain):
        return "added_place"
    keep = KEEP_PATTERNS.get(frame["id"])
    if keep and not keep.search(plain):
        return "dropped_cue"
    if frame["expect"]["intent"] != "compare":
        for name in AREA_SLOTS:
            if read_conj(text, name) and not read_conj(frame["text"], name):
                return "particle_changed"
    if (frame["expect"].get("hour_from"), frame["expect"].get("hour_to")) == (17, 19) and AFTER_WORK_LIKE.search(plain):
        return "changed_time"
    return None


def check_format(frame: dict, text: str) -> str | None:
    """형식 검사(1차와 같은 규칙, 단위는 퍼센트·월까지, 길이는 세 문장까지 허용해 240자)."""
    if not 2 <= len(text) <= 240:
        return "length"
    if sorted(PLACEHOLDER.findall(text)) != sorted(PLACEHOLDER.findall(frame["text"])):
        return "placeholder_mismatch"
    rest = PLACEHOLDER.sub("", text)
    if "<" in rest or ">" in rest or "{" in rest or "}" in rest:
        return "broken_placeholder"
    for name, unit in UNIT_AFTER.items():
        if f"<{name}>" in text and not re.search(rf"<{name}>\s*{unit}", text):
            return "unit_missing"
    if sorted(re.findall(r"\d+(?:\.\d+)?", rest)) != sorted(re.findall(r"\d+(?:\.\d+)?", PLACEHOLDER.sub("", frame["text"]))):
        return "numbers_changed"
    return None


# ---------------------------------------------------------------- 끼울 값(1차 값 + 교차 검증한 확장 어휘)

VOCAB_PATH = META / "diverse_vocab.json"
# 시간대 낱말의 시각 범위 = 해석기 설계 규약(src/agent/intent_parser.py extract_hours, README 슬롯 설명과 같음)
PERIOD_WINDOWS = {"새벽": (0, 5), "출근": (7, 9), "아침": (7, 9), "점심": (12, 13), "퇴근": (17, 19), "늦은 저녁": (21, 23),
                  "저녁": (18, 21), "밤": (21, 23), "야간": (21, 23), "심야": (22, 23), "오전": (9, 11), "오후": (13, 17)}
# 시간대 값에서 보일 수 있는 낱말 개념(이 밖의 시간 개념이 보이면 값이 섞인 것으로 보고 뺀다)
PERIOD_CONCEPTS = {"새벽": {"새벽"}, "출근": {"출근", "아침"}, "아침": {"아침", "출근"}, "점심": {"점심"}, "퇴근": {"퇴근"},
                   "늦은 저녁": {"저녁", "밤"}, "저녁": {"저녁"}, "밤": {"밤"}, "야간": {"밤"}, "심야": {"밤"},
                   "오전": {"오전", "아침"}, "오후": {"오후"}}
# 시각 값의 시간 낱말과 맞아야 하는 시각(0~23시)
CONCEPT_HOURS = {"새벽": range(0, 7), "아침": range(5, 12), "출근": range(5, 11), "오전": range(0, 12), "점심": range(11, 14),
                 "낮": range(10, 18), "오후": range(12, 24), "퇴근": range(16, 21), "저녁": range(16, 23),
                 "밤": [*range(19, 24), 0, 1, 2]}
GROUP_LABELS = {"어르신": "어르신", "청소년": "청소년"}
CONCEPT_NAMES = ("부동산 가격", "소음", "치안·범죄", "학군·교육 평판", "대기질", "주차 수요", "주관적 만족도")
TOPIC_NAMES = ("commute", "crowd", "spending", "living", "waste", "wifi", "ev")
VOCAB_ALLOWED = {"time_point": {"time", "topic"}, "after": {"time", "topic", "boundary"},
                 "range": {"time", "topic", "boundary"}, "weather": {"weather"}, "day": {"day", "boundary"},
                 "group": {"group"}, "concept": {"concept"}, "topicword": {"topic", "boundary", "slice"},
                 "source": {"claim", "topic", "concept", "slice"}, "place_form": set()}
AMPM_AFTER = re.compile(r"(?<![:\d])(\d{1,2})(?::\d{2})?\s*([ap])\.?\s*m\b\.?", FLAGS)  # 9pm, 9:00 PM, 11 p.m.
AMPM_BEFORE = re.compile(r"(?<![a-z])([ap])\.?\s*m\.?\s*(\d{1,2})\s*시", FLAGS)  # PM 7시, pm10시
NEGATED = re.compile(r"않|(?<![가-힣])안\s|없|아닌|아닐|아니|0%")
AFTER_LIKE = re.compile(r"넘|지나|지난|이후|경과|(?<![가-힣])후|(?<![가-힣])뒤|다음")
DAY_NEGATED = re.compile(r"(평일|주중|주말)\s*(?:말고|빼고|제외(?:한|하고)?|아닌|아니고)")
DAY_RULES = {"weekday": ({"평일", "요일"}, set("월화수목금")), "weekend": ({"주말", "요일"}, set("토일")),
             "market": ({"장날"}, set()),
             **{d: ({"요일", "평일" if d in "월화수목금" else "주말"}, {d}) for d in "월화수목금토일"}}
TIME_CATEGORIES = ("time_point", "after", "range")
PARTICLE_END = re.compile(r"(?:에|에서|은|는|이|가|을|를|로|으로|와|과|랑|도|만|의|께)$")
KOREAN_DIGITS = {"열두": 12, "열한": 11, "열": 10, "아홉": 9, "여덟": 8, "일곱": 7, "여섯": 6, "다섯": 5, "네": 4, "세": 3,
                 "두": 2, "한": 1}
BASE_FORMS = ("{}", "{} 쪽", "{} 근처", "{} 일대", "{} 주변", "제주 {}", "{} 인근", "{} 부근", "{} 지역")
FORM_WORDS = re.compile(r"\s*(?:쪽|근처|일대|주변|인근|부근|지역|동네)")
BASE_SOURCES = ("커뮤니티", "유튜브", "지역 신문", "단톡방", "맘카페", "인스타")
CONTEXT_FORMS = ("<P1> 퇴근 버스 어때?", "<P1>에서 퇴근할 때 버스 타기 어때?", "<P1> 퇴근 시간 버스 괜찮아?",
                 "<P1> 퇴근길 버스 사정 좀 알려줘", "퇴근 시간에 <P1> 버스 타기 괜찮은 편이야?", "<P1>에서 퇴근길에 버스 타면 어때?",
                 "<P1> 쪽 퇴근 버스 어떤가요?", "<P1> 퇴근 시간대 버스 상황은 어때?", "<P1>에서 퇴근 버스 타기 편해?",
                 "요즘 <P1> 퇴근 버스 어때요?", "<P1> 퇴근할 때 버스 괜찮을까요?", "<P1> 퇴근 무렵 버스는 어때?")


def load_vocab() -> dict:
    if not VOCAB_PATH.exists():
        return {}
    return json.loads(VOCAB_PATH.read_text(encoding="utf-8")).get("categories", {})


def read_hours(surface: str) -> list[int]:
    """값 표기에 적힌 시각 숫자(아라비아·한글 숫자, am/pm 은 24시로)."""
    hours = [int(h) % 12 + (12 if p.lower() == "p" else 0) for h, p in AMPM_AFTER.findall(surface)]
    hours += [int(h) % 12 + (12 if p.lower() == "p" else 0) for p, h in AMPM_BEFORE.findall(surface)]
    if not hours:
        hours = [int(h) for h in re.findall(r"(\d{1,2})\s*(?::\d{2}|시)", surface)]
    hours += [KOREAN_DIGITS[w] for w in re.findall(r"(열두|열한|열|아홉|여덟|일곱|여섯|다섯|네|세|두|한)\s*시(?![간])", surface)]
    return hours


def read_span(surface: str) -> list[int]:
    """시간 범위 표기의 두 끝 시각을 적힌 차례대로(12시간제 숫자 그대로, 정오=12, 자정=0)."""
    hours = []
    pattern = r"정오|자정|(?<![\d:])(\d{1,2})(?::\d{2})?|(열두|열한|열|아홉|여덟|일곱|여섯|다섯|네|세|두|한)\s*시(?!간)"
    for match in re.finditer(pattern, surface):
        word = match.group(0)
        if word in ("정오", "자정"):
            hours.append(12 if word == "정오" else 0)
        else:
            hours.append(int(match.group(1)) if match.group(1) else KOREAN_DIGITS[match.group(2)])
    return hours


def check_hour(surface: str, hour: int) -> bool:
    """표기에 숫자 시각이 있으면 그 시각(12시간제 포함)이 라벨과 맞는지."""
    return all(h % 12 == hour % 12 for h in read_hours(surface))


def convert_item(category: str, item: dict) -> tuple[dict | None, str | None]:
    """검증된 어휘 한 줄 -> 값 묶음 항목(정답 슬롯 포함). 코드 교차 확인에 걸리면 (None, 사유)."""
    surface = re.sub(r"\s+", " ", str(item.get("surface", ""))).strip()
    if not surface or len(surface) > 30 or "<" in surface or ">" in surface:
        return None, "surface"
    found = find_concepts(surface.replace("{}", NEUTRAL))
    allowed = VOCAB_ALLOWED[category] | ({"group"} if item.get("concept") == "학군·교육 평판" else set())
    if {c for c, _ in found} - allowed:
        return None, "other_condition"  # 학군 개념의 '학교·자녀'는 이용자 조건이 아니라 개념 자체다
    if category in TIME_CATEGORIES and {concept for c, concept in found if c == "topic"} - {"night"}:
        return None, "other_condition"  # '막차 시간'·'출근 버스 시간'처럼 시간 값에 주제어가 섞임
    if load_place_pattern().search(surface.replace("{}", NEUTRAL).replace("오일장", "")):
        return None, "place_word"  # 요일 값의 '오일장 서는 날'은 날짜 조건이지 장소가 아니다
    time_found = {concept for c, concept in found if c == "time"}
    labels: dict = {}
    if category == "time_point":
        if AFTER_LIKE.search(surface):
            return None, "after_like"  # '8시 좀 넘어서'는 그 시각인지 '8시 이후'인지 모호하다
        period, hour = item.get("period"), item.get("hour24")
        if isinstance(hour, int) and 0 <= hour <= 23:
            if not check_hour(surface, hour) or any(hour not in CONCEPT_HOURS[c] for c in time_found if c in CONCEPT_HOURS):
                return None, "hour_conflict"
            if not read_hours(surface) and not time_found and hour not in (0, 12):
                return None, "hour_unstated"
            labels = {"hour_from": hour, "hour_to": min(23, hour + 1)}
        elif period in PERIOD_WINDOWS:
            if read_hours(surface) or not time_found <= PERIOD_CONCEPTS[period]:
                return None, "period_conflict"
            labels = dict(zip(("hour_from", "hour_to"), PERIOD_WINDOWS[period]))
        else:
            return None, "label"
    elif category == "after":
        hour = item.get("hour24")
        if not isinstance(hour, int) or not (hour == 0 or 13 <= hour <= 23) or not check_hour(surface, hour):
            return None, "label"
        if any(hour not in CONCEPT_HOURS[c] for c in time_found if c in CONCEPT_HOURS):
            return None, "hour_conflict"
        labels = {"hour_from": 0, "hour_to": 5} if hour == 0 else {"hour_from": hour, "hour_to": 23}
    elif category == "range":
        start, end = item.get("start24"), item.get("end24")
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= 24):
            return None, "label"
        hours = read_span(surface)
        if len(hours) != 2 or hours[0] % 12 != start % 12 or hours[1] % 12 != end % 12:
            return None, "hour_conflict"
        labels = {"hour_from": start, "hour_to": min(23, end)}
    elif category == "weather":
        if item.get("weather") not in ("rain", "dry"):
            return None, "label"
        weather_found = {concept for c, concept in found if c == "weather"}
        rain_word = "비" in weather_found and not NEGATED.search(surface)  # '비가 오지 않는 날'은 비 안 오는 날
        if "기타" in weather_found or ("맑음" in weather_found if item["weather"] == "rain" else rain_word):
            return None, "label_conflict"
        labels = {"weather": item["weather"]}
    elif category == "day":
        code = item.get("day")
        negated = {"평일" if word in ("평일", "주중") else "주말" for word in DAY_NEGATED.findall(surface)}
        day_found = {concept for c, concept in found if c == "day"} - {"요일일반"} - negated
        days = set(re.findall(r"(?<!요)[월화수목금토일](?=[월화수목금토일·,~\-\s]*(?:요일|욜))", surface))
        if code not in DAY_RULES or not day_found <= DAY_RULES[code][0] or not days <= DAY_RULES[code][1]:
            return None, "label_conflict"  # '토요일과 일요일'은 주말, '평일 말고 주말'은 주말
        if code in "월화수목금토일" and code not in days:
            return None, "label_conflict"
        labels = {"day_type": code}
    elif category == "group":
        group = item.get("group")
        group_found = {concept for c, concept in found if c == "group"}
        if group not in GROUP_LABELS or not group_found <= {GROUP_LABELS[group]}:
            return None, "label_conflict"
        labels = {"user_group": group}
    elif category == "concept":
        if item.get("concept") not in CONCEPT_NAMES:
            return None, "label"
        labels = {"abstain": True}
    elif category == "topicword":
        topic = item.get("topic")
        topic_found = {concept for c, concept in found if c == "topic"}
        if topic not in TOPIC_NAMES or not topic_found <= {topic}:
            return None, "label_conflict"
        labels = {"topic": topic}
    elif category == "place_form":
        rest = surface.replace("{}", "").strip()
        if surface.count("{}") != 1 or re.search(r"\d", surface) or not rest or len(rest) > 10 or PARTICLE_END.search(surface):
            return None, "label"  # 조사로 끝나면 틀의 조사와 겹친다('{} 쪽에' + '에서')
    elif category == "source":
        if re.search(r"\d", surface):
            return None, "label"
    return base.mark_value(surface, **labels), None


def convert_vocab() -> tuple[dict[str, list[dict]], dict[str, dict[str, int]]]:
    """확장 어휘 -> 범주별 값 묶음과 코드 교차 확인 탈락 통계."""
    pools: dict[str, list[dict]] = {name: [] for name in VOCAB_ALLOWED}
    dropped: dict[str, dict[str, int]] = {}
    seen = set()
    for category, items in load_vocab().items():
        if category not in VOCAB_ALLOWED:
            continue
        for item in items:
            value, reason = convert_item(category, item)
            key = (category, value["surface"].replace(" ", "")) if value else None
            if value is None or key in seen:
                reason = reason or "duplicate"
                dropped.setdefault(category, {})[reason] = dropped.setdefault(category, {}).get(reason, 0) + 1
                continue
            seen.add(key)
            value["key"] = f"{category}:{value['surface']}"
            value["heldout_only"] = compute_hash(value["key"]) % 4 == 0
            pools[category].append(value)
    return pools, dropped


def extend_pool(pools: dict, name: str, values: list[dict]) -> None:
    """같은 표기(공백 무시)가 없을 때만 덧붙인다."""
    present = {value["surface"].replace(" ", "") for value in pools[name]}
    for value in values:
        if value["surface"].replace(" ", "") not in present:
            present.add(value["surface"].replace(" ", ""))
            pools[name].append(value)


def make_value(pool: str, surface: str, heldout: bool | None = None, **labels) -> dict:
    value = base.mark_value(surface, **labels)
    value["key"] = f"{pool}:{surface}"
    value["heldout_only"] = (compute_hash(value["key"]) % 4 == 0) if heldout is None else heldout
    return value


def build_pools() -> dict[str, list[dict]]:
    pools = base.build_pools()
    vocab = convert_vocab()[0]
    extend_pool(pools, "time_point", vocab["time_point"])
    extend_pool(pools, "after", vocab["after"])
    extend_pool(pools, "weather", vocab["weather"])
    extend_pool(pools, "cond", vocab["weather"] + vocab["day"])
    extend_pool(pools, "day", vocab["day"])
    extend_pool(pools, "old", [v for v in vocab["group"] if v["labels"]["user_group"] == "어르신"])
    extend_pool(pools, "young", [v for v in vocab["group"] if v["labels"]["user_group"] == "청소년"])
    extend_pool(pools, "group", vocab["group"])
    extend_pool(pools, "concept", vocab["concept"])
    evening = [v for v in vocab["time_point"] if (v["labels"]["hour_from"], v["labels"]["hour_to"]) == (18, 21)]
    evening += [v for v in vocab["after"] if 18 <= v["labels"]["hour_from"] <= 20]  # 21시 이후는 '밤'이라 저녁 값에 넣지 않는다
    # 생활·매출 틀에 끼우므로 '늦은 저녁'·'밤 8시'처럼 밤 주제어가 든 값은 빼서 주제(생활 vs 야간)가 모호하지 않게 한다
    extend_pool(pools, "evening", [v for v in evening if ("topic", "night") not in find_concepts(v["surface"])])
    pools["emd"] = [v for v in pools["area"] if len(v["labels"]["emds"]) == 1]
    pools["day2"] = list(pools["day"])
    extend_pool(pools, "day2", [make_value("day", "장날", heldout=False, day_type="market")])
    pools["morning"] = [v for v in pools["time_point"]
                        if v["labels"].get("hour_from") is not None and 5 <= v["labels"]["hour_from"] <= 9
                        and v["labels"]["hour_to"] <= 10]
    pools["nightword"] = [v for v in pools["time_point"]
                          if {concept for c, concept in find_concepts(v["surface"]) if c == "topic"} == {"night"}
                          and (v["labels"]["hour_from"] >= 21 or v["labels"]["hour_to"] <= 5)]
    pools["pct"] = [make_value("pct", str(n)) for n in (5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80)]
    pools["month_commute"] = [make_value("month", str(m), abstain=not 5 <= m <= 10) for m in range(1, 13)]
    pools["source"] = [make_value("source", s) for s in BASE_SOURCES]
    extend_pool(pools, "source", vocab["source"])
    pools["topicword"] = list(vocab["topicword"])
    pools["ym_waste"] = [make_value("ym", f"{y}년 {m}월", abstain=not 201801 <= y * 100 + m <= 202105)
                         for y in range(2016, 2024) for m in (1, 3, 5, 7, 9, 12)]
    pools["range"] = list(vocab["range"])
    pools["form"] = [make_value("form", form, heldout=False if form == "{}" else None) for form in BASE_FORMS]
    extend_pool(pools, "form", vocab["place_form"])
    pools["ctx"] = []
    for number, form in enumerate(CONTEXT_FORMS):
        value = make_value("ctx", "", heldout=False if number == 0 else None)
        value.update(key=f"ctx:{form}", text=form)
        if number:
            value["heldout_only"] = compute_hash(value["key"]) % 4 == 0
        pools["ctx"].append(value)
    return pools


# ---------------------------------------------------------------- 문장틀 불러오기(형식 + 뜻 검사)

MODEL_FILES = {"gemma4": META / "diverse_frames_gemma4.jsonl", "qwen3": META / "diverse_frames_qwen3.jsonl"}
SET_SOURCES = {"dev": ("base", "gemma4"), "heldout": ("qwen3",)}


def load_templates(set_name: str) -> tuple[list[dict], dict]:
    """세트의 문장틀 목록과 탈락 통계. LLM 틀은 형식 검사와 뜻 검사(check_meaning)를 모두 통과해야 쓴다."""
    frames = {frame["id"]: frame for frame in load_frames()}
    templates, dropped, seen, numbers = [], {}, set(), {}
    for source in SET_SOURCES[set_name]:
        rows = []
        if source == "base":
            rows = [{"frame_id": fid, "style": "canonical", "texts": [frame["text"]]} for fid, frame in frames.items()]
        elif MODEL_FILES[source].exists():
            rows = [json.loads(line) for line in MODEL_FILES[source].read_text(encoding="utf-8").splitlines()
                    if line.strip()]
        for row in rows:
            frame = frames[row["frame_id"]]
            for text in row["texts"]:
                normalized = re.sub(r"\s+", " ", text).strip()
                reason = check_format(frame, normalized)
                if reason is None and source != "base":
                    reason = check_meaning(frame, normalized)
                    reason = f"meaning:{reason}" if reason else None
                if reason is None and (frame["id"], normalized) in seen:
                    reason = "duplicate"
                if reason:
                    dropped[reason] = dropped.get(reason, 0) + 1
                    continue
                seen.add((frame["id"], normalized))
                number = numbers.get((source, frame["id"]), 0)
                numbers[(source, frame["id"])] = number + 1
                templates.append({"key": f"{set_name}:{source}:{frame['id']}:{number:05d}", "frame": frame["id"],
                                  "family": frame["family"], "workspace": frame["workspace"], "style": row["style"],
                                  "source": source, "text": normalized, "context": frame.get("context"),
                                  "situation": row.get("situation")})
    return templates, dropped


# ---------------------------------------------------------------- 정답 계산·입력 변형·조합 열거

BLANK = base.BLANK


def compute_expected(frame: dict, values: dict) -> dict:
    """틀의 고정 정답 + 끼운 값의 정답 슬롯. 'X 말고 Y' 틀과 앞 질문에만 나온 지명은 exclude_places 로 뺀다."""
    expected = {**BLANK, **frame["expect"]}
    excluded = set(frame.get("exclude_places", ()))
    emds, places = set(), 0
    for name, value in values.items():
        for slot, label in value["labels"].items():
            if slot == "emds":
                if name not in excluded:
                    emds.update(label)
                    places += 1
            else:
                expected[slot] = label
    if places:
        expected["emds"], expected["area_count"] = sorted(emds), places
    return expected


HELDOUT_KINDS = ("quote", "english_affix", "space_in_place")
KINDS = list(base.KINDS) + ["partial_space", "comma", "greeting", "closing", "mid_emoji", "stretch", "multi_typo",
                            *HELDOUT_KINDS]
KIND_SETS = {"dev": [k for k in KINDS if k not in HELDOUT_KINDS], "heldout": list(KINDS)}
STRESS_KINDS = tuple(base.STRESS_KINDS) + ("stretch", "multi_typo", "space_in_place")
GREETINGS = ("안녕하세요, 수고 많으십니다. ", "좋은 하루 보내고 계시죠? 하나만 여쭤볼게요. ", "바쁘신데 죄송한데요, ",
             "저기요~ 질문 하나 있어요! ", "늘 잘 보고 있어요. ", "처음 질문 올려 봐요. ", "실례합니다. ", "하나 물어봐도 돼? ",
             "급한 건 아닌데 ", "다름이 아니라 ", "궁금해서 그러는데요 ", "질문 드립니다. ", "제가 잘 몰라서요, ",
             "어쩌다 보니 궁금해졌는데 ", "도와주세요! ")
CLOSINGS = (" 답변 주시면 정말 감사하겠습니다.", " 미리 감사드려요!", " 바쁘시겠지만 꼭 답 부탁드립니다~", " 고마워!!",
            " 빠른 답변 부탁드려요.", " 알려 주시면 큰 도움이 될 것 같아요.", " 감사합니다 :)", " 수고하세요!",
            " 좋은 하루 되세요.", " 답 좀 줘~", " 부탁 좀 할게.")
MID_MARKS = (" \U0001F68C", " \U0001F605", " ㅎㅎ", " ;;", " \U0001F64F", " ^^", " ㅠ", " \U0001F914", " (?)")
QUOTES = (('"', '"'), ("'", "'"), ("“", "”"), ("「", "」"), ("『", "』"), ("[", "]"))
EN_PREFIXES = ("Hi, ", "Hello! ", "Hey ", "Q: ", "Quick question: ")
EN_SUFFIXES = (" plz", " thx!", " asap", " lol", " please~")


def apply_typos(text: str, spans: list, pick: int) -> str | None:
    """지명 밖 본문 두세 군데에 자판 오타(글자 수는 그대로)."""
    covered = {i for s, e, _ in spans for i in range(s, e)}
    changed, out = set(), text
    for step in range(2 + pick % 2):
        free = [i for i, c in enumerate(out) if i not in covered and i not in changed and base.split_syllable(c)]
        if not free:
            break
        offset = free[(pick >> (4 + step * 7)) % len(free)]
        mutated = base.apply_typo(out, (offset, offset + 1), pick >> (11 + step * 5), check_place=False)
        changed.add(offset)
        if mutated is not None:
            out = mutated
    return out if out != text and sum(a != b for a, b in zip(out, text)) >= 2 else None


def apply_stretch(text: str, spans: list, pick: int) -> str | None:
    """끝 음절의 모음을 한두 번 더 늘인다('어때?' -> '어때애애?')."""
    covered = {i for s, e, _ in spans for i in range(s, e)}
    for offset in range(len(text) - 1, -1, -1):
        parts = base.split_syllable(text[offset])
        if parts and offset not in covered:
            extra = base.compose_syllable(base.CHO.index("ㅇ"), parts[1], 0) * (1 + (pick >> 6) % 2)
            return text[:offset + 1] + extra + text[offset + 1:]
        if parts:
            return None
    return None


def apply_noise(text: str, spans: list, kind: str, number: int) -> str | None:
    """입력 변형 하나를 적용한다. 적용할 수 없으면 None. 1차 변형은 scale_suite 와 같은 함수(지명 자리에 P3 추가)."""
    pick = base.compute_mix(number)
    gaps = [i for i, c in enumerate(text) if c == " "]
    places = [(s, e) for s, e, name in spans if name in PLACE_SLOTS]
    if kind == "typo_place":
        return base.apply_typo(text, places[pick % len(places)], pick >> 8, check_place=True) if places else None
    if kind == "choseong_place":
        eligible = [(s, e) for s, e in places if text[s:e] in base.load_place_keys()[1]]
        if not eligible:
            return None
        start, end = eligible[pick % len(eligible)]
        return text[:start] + base.extract_choseong(text[start:end]) + text[end:]
    if kind in base.KINDS:
        return base.apply_noise(text, spans, kind, number)
    if kind == "partial_space":
        if len(gaps) < 2:
            return None
        drop = {g for k, g in enumerate(gaps) if (pick >> (k % 60)) & 1}
        if not drop or len(drop) == len(gaps):
            drop = set(gaps[(pick >> 7) % 2::2])
        return "".join(c for i, c in enumerate(text) if i not in drop)
    if kind == "comma":
        usable = [g for g in gaps if g > 0 and text[g - 1] not in ",.?!~"]
        return text[:usable[pick % len(usable)]] + "," + text[usable[pick % len(usable)]:] if usable else None
    if kind == "greeting":
        return GREETINGS[pick % len(GREETINGS)] + text
    if kind == "closing":
        return text + CLOSINGS[pick % len(CLOSINGS)]
    if kind == "mid_emoji":
        if not gaps:
            return None
        cut = gaps[pick % len(gaps)]
        return text[:cut] + MID_MARKS[(pick >> 8) % len(MID_MARKS)] + text[cut:]
    if kind == "stretch":
        return apply_stretch(text, spans, pick)
    if kind == "multi_typo":
        return apply_typos(text, spans, pick)
    if kind == "quote":
        left, right = QUOTES[pick % len(QUOTES)]
        return left + text + right
    if kind == "english_affix":
        if pick % 2:
            return EN_PREFIXES[(pick >> 4) % len(EN_PREFIXES)] + text
        return text + EN_SUFFIXES[(pick >> 4) % len(EN_SUFFIXES)]
    if kind == "space_in_place":
        long_places = [(s, e) for s, e in places if e - s >= 3 and " " not in text[s:e]]
        if not long_places:
            return None
        start, end = long_places[pick % len(long_places)]
        cut = start + 1 + (pick >> 8) % (end - start - 1)
        return text[:cut] + " " + text[cut:]
    raise ValueError(kind)


def build_triples(values: list[dict]) -> list[tuple[dict, dict, dict]]:
    """세 곳 비교용 조합: 같은 시, 서로 겹치는 읍면동 없음."""
    by_city: dict[str, list[dict]] = {}
    for value in values:
        by_city.setdefault(value["city"], []).append(value)
    triples = []
    for group in by_city.values():
        for i, first in enumerate(group):
            for j in range(i + 1, len(group)):
                second = group[j]
                if set(first["labels"]["emds"]) & set(second["labels"]["emds"]):
                    continue
                used = set(first["labels"]["emds"]) | set(second["labels"]["emds"])
                triples += [(first, second, third) for third in group[j + 1:] if not used & set(third["labels"]["emds"])]
    return triples


def plan_dimensions(template: dict, frame: dict, pools: dict, set_name: str) -> list[tuple[tuple, list]]:
    """문장틀의 조합 차원: [(자리표시자 이름들, 값 묶음 목록)]. 개발용은 평가용 전용 값을 뺀다.
    지명 감싸기(FORM)는 본문에 지명 자리표시자가 있는 틀에만, 앞 질문 표현(CTX)은 1차 후속 질문 문맥을 쓰는 틀에만 붙인다.
    FORM 은 그대로 쓰기('{}')를 감싸기 종류 수만큼 넣어 절반쯤은 지명을 그대로 쓰게 한다."""
    def is_allowed(value: dict) -> bool:
        return set_name != "dev" or not value["heldout_only"]

    slots = frame["slots"]
    dims = []
    if all(name in slots for name in AREA_SLOTS):
        dims.append((AREA_SLOTS, build_triples([v for v in pools[slots["P1"]] if is_allowed(v)])))
    elif "P1" in slots and "P2" in slots:
        dims.append((("P1", "P2"), base.build_pairs([v for v in pools[slots["P1"]] if is_allowed(v)])))
    elif "P1" in slots:
        dims.append((("P1",), [(v,) for v in pools[slots["P1"]] if is_allowed(v)]))
    for name, pool_name in slots.items():
        if name in AREA_SLOTS:
            continue
        values = [v for v in pools[pool_name] if is_allowed(v)]
        if check_attached(template["text"], name):
            values = [v for v in values if v["surface"]]
        dims.append(((name,), [(v,) for v in values]))
    if any(f"<{name}>" in template["text"] for name in AREA_SLOTS):
        forms = [v for v in pools["form"] if is_allowed(v)]
        dims.append((("FORM",), [(forms[0],)] * (len(forms) - 1) + [(v,) for v in forms]))
    if frame.get("context") == FOLLOW:
        dims.append((("CTX",), [(v,) for v in pools["ctx"] if is_allowed(v)]))
    return dims


def count_space(dims: list, set_name: str) -> int:
    total = len(KIND_SETS[set_name])
    for _, options in dims:
        total *= len(options)
    return total


def decode_case(index: int, dims: list, kind_count: int) -> tuple[dict, int]:
    """조합 번호 -> ({자리표시자: 값}, 변형 번호)."""
    kind = index % kind_count
    rest = index // kind_count
    values = {}
    for names, options in reversed(dims):
        choice = options[rest % len(options)]
        rest //= len(options)
        for name, value in zip(names, choice):
            values[name] = value
    return values, kind


def apply_forms(template_text: str, values: dict, form: dict | None) -> tuple[dict, dict]:
    """지명 값을 감싸기 형태로 바꾼 값과, 감싼 지명 안 원래 지명의 (앞 글자 수, 길이)."""
    if form is None or form["surface"] == "{}":
        return values, {}
    head = form["surface"].index("{}")
    shown, offsets = dict(values), {}
    for name in AREA_SLOTS:
        marker = f"<{name}>"
        position = template_text.find(marker)
        if name not in values or position < 0 or FORM_WORDS.match(template_text, position + len(marker)):
            continue  # 틀에 이미 '쪽'·'근처'가 붙은 자리는 그대로 둔다
        shown[name] = {**values[name], "surface": form["surface"].replace("{}", values[name]["surface"])}
        offsets[name] = (head, len(values[name]["surface"]))
    return shown, offsets


def build_case(template: dict, frame: dict, dims: list, index: int, set_name: str) -> dict:
    """조합 번호 하나 -> 시험 문항(문장·문맥·정답·실제 적용된 변형·평가용 전용 요소)."""
    kinds = KIND_SETS[set_name]
    values, kind_number = decode_case(index, dims, len(kinds))
    form, ctx = values.pop("FORM", None), values.pop("CTX", None)
    shown, offsets = apply_forms(template["text"], values, form)
    text, spans = base.fill_template(template["text"], shown)
    spans = [(s + offsets[n][0], s + offsets[n][0] + offsets[n][1], n) if n in offsets else (s, e, n) for s, e, n in spans]
    context_text = ctx["text"] if ctx is not None else template.get("context")
    context = base.fill_template(context_text, values)[0] if context_text else None
    noisy, kind = None, ""
    salt = compute_hash(template["key"])
    for step in range(len(kinds)):
        kind = kinds[(kind_number + step) % len(kinds)]
        noisy = apply_noise(text, spans, kind, index ^ salt)
        if noisy is not None and noisy.strip():
            break
    extras = [v for v in (form, ctx) if v is not None]
    novel = [part for part, flag in (
        ("value", any(v["heldout_only"] for v in values.values())), ("form", bool(form and form["heldout_only"])),
        ("context", bool(ctx and ctx["heldout_only"])), ("kind", kind in HELDOUT_KINDS),
        ("style", template["style"] in HELDOUT_STYLES), ("situation", check_situation(template.get("situation"))),
    ) if flag]
    return {"text": noisy, "context": context, "kind": kind, "expected": compute_expected(frame, values),
            "values": [v["key"] for v in list(values.values()) + extras], "unseen": bool(novel), "novel_parts": novel}
