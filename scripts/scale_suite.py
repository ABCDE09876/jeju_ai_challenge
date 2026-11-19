"""대규모 조합 행동 시험(2,500만 문항 이상)의 정의: 문장틀·끼울 값·정답 계산·입력 변형.

CheckList(Ribeiro 외, ACL 2020)식 행동 시험을 제주 질문 해석기에 맞춰 만들었다.
- 문장틀: 견고성 시험 기준 질문의 질문 유형(family)마다 자리표시자(<P1>, <TIME> 등)를 넣은 기본 틀 27개와
  기준 질문 중 벤치마크·시나리오 문장 30개(자리표시자 없음), 그리고 데스크탑 GPU 의 로컬 LLM 이 그 틀을
  7가지 말투로 다시 쓴 틀(scripts/generate_question_frames.py).
- 끼울 값: 지명 사전의 읍면동·별칭·정류소·명소와 시간·날씨·요일·이용자·숫자 표현. 값마다 정답 슬롯이 정해져 있어
  틀에 값을 끼우면 그 문장의 정답(12개 슬롯)이 저절로 정해진다(해석기를 돌려 정답을 만들지 않는다).
- 입력 변형: 엄격(해석이 바뀌면 안 됨) = 그대로·띄어쓰기 생략·문장부호·이모티콘·앞 덧말·뒤 덧말·공백 잡음·전각 문자,
  부하(사람도 헷갈릴 수 있음) = 지명 오타·지명 초성·본문 오타.
- 개발용/평가용 분리: 개발용은 기본 틀과 gemma4 틀에 값의 약 3/4 만 끼운다. 평가용은 qwen3 틀에 모든 값을 끼워
  개발 중에 한 번도 보지 못한 문장틀과 값(약 1/4)을 함께 시험한다.
이 모듈은 표준 라이브러리만 쓴다(LLM 생성 단계는 해석기 없이 돈다). 해석기 채점은 scripts/run_scale_test.py.
"""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from math import gcd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "meta"
SCALE_SEED = 20261120
SLOTS = ["intent", "topic", "emds", "area_count", "hour_from", "hour_to", "weather", "day_type", "user_group",
         "wait_minutes", "abstain", "variable"]
STYLES = {
    "casual": "친구에게 묻는 반말 구어체",
    "typo": "급하게 친 듯한 문장. 자판 오타·맞춤법 오류·띄어쓰기 오류를 2개 이상 넣는다(자리표시자는 건드리지 않음)",
    "jeju": "제주 사투리 어미와 표현을 섞은 말투(예: ~우꽈?, ~수과?, ~쿠과?, ~마씸, 하영=많이, 게메=글쎄)",
    "elderly": "나이 든 어르신이 묻는 듯한 길고 공손한 말투",
    "short": "검색창에 치듯 아주 짧게 줄인 질문",
    "reorder": "조건을 문장 앞으로 옮기는 등 어순을 바꾼 표준어 질문",
    "mixed": "영어 단어를 몇 개 섞은 질문(자리표시자와 숫자는 그대로)",
}
PLACEHOLDER = re.compile(r"<([A-Z][A-Z0-9]*)>")
# 자리표시자 뜻(LLM 지시문에 그대로 넣는다)
PLACEHOLDER_NOTES = {
    "P1": "제주 지명(동네·생활권 이름)", "P2": "또 다른 제주 지명", "STOP": "버스 정류소 이름",
    "MARKET": "시장이나 명소 이름", "COND": "날씨나 요일 조건(예: 비 오는 날, 평일, 주말, 월요일). 빈칸일 수도 있다",
    "EVE": "저녁 시간 조건(예: 퇴근 후, 저녁에). 빈칸일 수도 있다", "TIME": "시간 표현(예: 아침 8시에, 출근 시간에)",
    "AFTER": "'몇 시 이후'를 뜻하는 시간 표현 전체(예: 밤 9시 이후)", "MIN": "분 단위 숫자(예: 10)",
    "OLD": "어르신을 뜻하는 말(예: 어르신, 할머니)", "OLDCOND": "요일·날씨·시간 조건(예: 평일 오전에, 비 오는 날). 빈칸일 수도 있다",
    "YOUNG": "학생·청소년을 뜻하는 말", "WEATHER": "날씨 조건(예: 비 오는 날, 맑은 날)",
    "CONCEPT": "생활과 관련된 개념(예: 집값, 치안)", "PERIOD": "연도와 계절(예: 2022년 여름)",
    "DAY": "요일 조건(예: 주말, 평일)", "CLOCK": "시각(예: 8시)", "GROUP": "이용자 집단(예: 어르신, 학생)",
    "RATIO": "배수 숫자(예: 2)", "YEAR": "연도 숫자(예: 2019)",
}
FOLLOW_CONTEXT = "<P1> 퇴근 버스 어때?"
# 기본 문장틀: 기준 질문 생성기(scripts/build_robustness_seeds.py)의 질문 유형을 자리표시자로 일반화했다.
# slots = 자리표시자 -> 값 묶음, expect = 값과 무관하게 정해진 정답 슬롯
FRAMES = [
    {"id": "F01", "family": "compare_commute", "workspace": "life",
     "text": "<P1>(이)랑 <P2> 중에 <COND> 퇴근 버스 타기 어디가 나아?", "slots": {"P1": "area", "P2": "area", "COND": "cond"},
     "expect": {"intent": "compare", "topic": "commute", "hour_from": 17, "hour_to": 19}},
    {"id": "F02", "family": "compare_living", "workspace": "life",
     "text": "<P1>(이)랑 <P2> 중 <EVE> 생활하기 어디가 나아?", "slots": {"P1": "area", "P2": "area", "EVE": "evening"},
     "expect": {"intent": "compare", "topic": "living"}},
    {"id": "F03", "family": "profile_time", "workspace": "life", "text": "<P1> <TIME> 버스 어때?",
     "slots": {"P1": "area", "TIME": "time_point"}, "expect": {"intent": "profile", "topic": "commute"}},
    {"id": "F04", "family": "boundary_after", "workspace": "life", "text": "<P1>에서 <AFTER>에도 버스 다녀?",
     "slots": {"P1": "area", "AFTER": "after"},
     "expect": {"intent": "boundary", "topic": "commute", "variable": "return_hour"}},
    {"id": "F05", "family": "night_crowd", "workspace": "life", "text": "<P1> <AFTER>에도 사람 많아?",
     "slots": {"P1": "area", "AFTER": "after"}, "expect": {"intent": "profile", "topic": "crowd"}},
    {"id": "F06", "family": "market_crowd", "workspace": "life", "text": "<STOP> 정류소 장날에 붐벼?",
     "slots": {"STOP": "stop"}, "expect": {"intent": "profile", "topic": "commute", "day_type": "market"}},
    {"id": "F07", "family": "hidden_slice", "workspace": "life", "text": "<STOP> 정류소는 평균은 한산한데 언제 붐벼?",
     "slots": {"STOP": "stop"}, "expect": {"intent": "hidden_slice", "topic": "commute"}},
    {"id": "F08", "family": "boundary_wait", "workspace": "life",
     "text": "<P1> 정류소에서 <MIN>분까지 기다릴 수 있으면 몇 시까지 버스 타면 돼?",
     "slots": {"P1": "area", "MIN": "wait_min"},
     "expect": {"intent": "boundary", "topic": "commute", "variable": "wait_minutes"}},
    {"id": "F09", "family": "boundary_visit", "workspace": "life", "text": "<MARKET> 장 보러 몇 시에 가는 게 덜 붐벼?",
     "slots": {"MARKET": "market"}, "expect": {"intent": "boundary", "topic": "crowd", "variable": "visit_hour"}},
    {"id": "F10", "family": "group_elderly", "workspace": "life", "text": "<OLD>이(가) <P1>에서 <OLDCOND> 버스 타기 괜찮을까?",
     "slots": {"OLD": "old", "P1": "area", "OLDCOND": "old_cond"},
     "expect": {"intent": "profile", "topic": "commute", "user_group": "어르신"}},
    {"id": "F11", "family": "group_youth", "workspace": "life", "text": "<YOUNG>이(가) <P1>에서 버스 타고 다니기 어때?",
     "slots": {"YOUNG": "young", "P1": "area"}, "expect": {"intent": "profile", "topic": "commute", "user_group": "청소년"}},
    {"id": "F12", "family": "weather", "workspace": "life", "text": "<WEATHER> <P1> 퇴근 버스 어때?",
     "slots": {"WEATHER": "weather", "P1": "area"},
     "expect": {"intent": "profile", "topic": "commute", "hour_from": 17, "hour_to": 19}},
    {"id": "F13", "family": "spending", "workspace": "life", "text": "<P1> 상권 저녁 매출은 도민이 많이 써?",
     "slots": {"P1": "area"}, "expect": {"intent": "profile", "topic": "spending", "hour_from": 18, "hour_to": 21}},
    {"id": "F14", "family": "abstain_unmeasured", "workspace": "life", "text": "<P1> <CONCEPT> 어때?",
     "slots": {"P1": "area", "CONCEPT": "concept"}, "expect": {"intent": "profile", "topic": "general", "abstain": True}},
    {"id": "F15", "family": "abstain_period", "workspace": "life", "text": "<PERIOD> <P1> 버스 붐볐어?",
     "slots": {"PERIOD": "period", "P1": "area"}, "expect": {"intent": "profile", "topic": "commute"}},
    {"id": "F16", "family": "followup_weather", "workspace": "life", "text": "그럼 <WEATHER>은(는)?",
     "context": FOLLOW_CONTEXT, "slots": {"P1": "area", "WEATHER": "weather"},
     "expect": {"intent": "profile", "topic": "commute", "hour_from": 17, "hour_to": 19}},
    {"id": "F17", "family": "followup_day", "workspace": "life", "text": "<DAY>은(는)요?", "context": FOLLOW_CONTEXT,
     "slots": {"P1": "area", "DAY": "day"}, "expect": {"intent": "profile", "topic": "commute", "hour_from": 17, "hour_to": 19}},
    {"id": "F18", "family": "followup_partner", "workspace": "life", "text": "거기 말고 <P2>(이)랑 비교해 줘",
     "context": FOLLOW_CONTEXT, "slots": {"P1": "area", "P2": "area"},
     "expect": {"intent": "compare", "topic": "commute", "hour_from": 17, "hour_to": 19}},
    {"id": "F19", "family": "followup_clock", "workspace": "life", "text": "그럼 <CLOCK> 이후는?", "context": FOLLOW_CONTEXT,
     "slots": {"P1": "area", "CLOCK": "clock"}, "expect": {"intent": "boundary", "topic": "commute", "variable": "return_hour"}},
    {"id": "F20", "family": "followup_group", "workspace": "life", "text": "<GROUP> 기준으로는?", "context": FOLLOW_CONTEXT,
     "slots": {"P1": "area", "GROUP": "group"}, "expect": {"intent": "profile", "topic": "commute", "hour_from": 17, "hour_to": 19}},
    {"id": "F21", "family": "trend_claim", "workspace": "lawyer", "text": "<P1> 버스 이용객이 <RATIO>배 늘었다는 기사 사실이야?",
     "slots": {"P1": "area", "RATIO": "ratio"}, "expect": {"intent": "fact_check", "topic": "commute"}},
    {"id": "F22", "family": "causal_claim", "workspace": "lawyer", "text": "<P1> 카드 매출이 줄어든 건 관광객 때문이라던데 맞아?",
     "slots": {"P1": "area"}, "expect": {"intent": "fact_check", "topic": "spending"}},
    {"id": "F23", "family": "compare_claim", "workspace": "lawyer", "text": "<P1>은(는) 저녁에 도민이 <P2>보다 많다던데 맞아?",
     "slots": {"P1": "area", "P2": "area"}, "expect": {"intent": "fact_check", "topic": "general", "hour_from": 18, "hour_to": 21}},
    {"id": "F24", "family": "slice_claim", "workspace": "lawyer",
     "text": "<STOP> 정류소는 평균 이용객이 적어서 증편이 필요 없다는 답변, 맞아?", "slots": {"STOP": "stop"},
     "expect": {"intent": "fact_check", "topic": "commute"}},
    {"id": "F25", "family": "ev_claim", "workspace": "lawyer", "text": "전기차 충전 대기시간이 평균 <MIN>분이라는 보도 믿어도 돼?",
     "slots": {"MIN": "ev_min"}, "expect": {"intent": "fact_check", "topic": "ev"}},
    {"id": "F26", "family": "waste_claim", "workspace": "lawyer", "text": "<P1> 음식물 쓰레기가 <YEAR>년에 급증했다는 주장 검증해 줘",
     "slots": {"P1": "area", "YEAR": "year_waste"}, "expect": {"intent": "fact_check", "topic": "waste"}},
    {"id": "F27", "family": "wifi_claim", "workspace": "lawyer", "text": "<P1> 공공와이파이 이용자가 <YEAR>년에 반토막 났다던데 맞아?",
     "slots": {"P1": "area", "YEAR": "year_wifi"}, "expect": {"intent": "fact_check", "topic": "wifi"}},
]
FIXED_FAMILIES = ("benchmark", "scenario")  # 기준 질문 중 그대로 쓰는 문장(자리표시자 없음)
UNIT_AFTER = {"MIN": "분", "RATIO": "배", "YEAR": "년"}  # 자리표시자 바로 뒤에 와야 하는 단위


def load_frames() -> list[dict]:
    """기본 문장틀 27개 + 기준 질문의 벤치마크·시나리오 문장 30개(정답은 기준 질문의 reference)."""
    frames = [dict(frame, context=frame.get("context")) for frame in FRAMES]
    for line in (META / "robustness_seeds.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        seed = json.loads(line)
        if seed["family"] in FIXED_FAMILIES:
            frames.append({"id": f"X{seed['id'][1:]}", "family": seed["family"], "workspace": seed["workspace"],
                           "text": seed["text"], "context": seed.get("context"), "slots": {},
                           "expect": dict(seed["reference"]), "fixed": True})
    return frames


# ---------------------------------------------------------------- 끼울 값(정답 슬롯 포함)

EXCLUDED_ALIASES = ("오일장", "민속오일장", "공항", "시청", "영교", "우도", "추자도")  # 기준 질문 생성기와 같은 제외 목록
RISKY_BASES = {"중앙", "천지", "연", "정방"}
MARKET_NAMES = ("오일장", "민속오일장", "동문시장", "올레시장", "매일올레시장")
KOREAN_NUMERALS = {6: "여섯", 7: "일곱", 8: "여덟", 9: "아홉", 10: "열", 11: "열한"}
DAY_CODES = {"월요일": "월", "화요일": "화", "수요일": "수", "목요일": "목", "금요일": "금", "토요일": "토", "일요일": "일"}
RAIN = ("비 오는 날", "비 올 때", "비가 오는 날", "비 내리는 날", "우천 시", "비 많이 오는 날")
DRY = ("비 안 오는 날", "비 안 올 때", "맑은 날", "비가 오지 않는 날", "비 없는 날", "날씨 맑은 날")
OLD_WORDS = ("어르신", "노인", "할머니", "할아버지", "부모님", "고령자", "연세 드신 분", "나이 드신 분")
YOUNG_WORDS = ("학생", "학생들", "청소년", "중학생", "고등학생", "아이들", "애들", "청소년들")
CONCEPTS = ("집값", "층간소음", "치안", "학군", "미세먼지", "주차난", "월세", "전세", "아파트값", "범죄", "소음", "대기질",
            "주차 공간", "학교 평판", "부동산")
SEASONS = {"봄": (3, 4, 5), "여름": (6, 7, 8), "가을": (9, 10, 11), "겨울": (12, 1, 2)}
COMMUTE_COVERAGE = (202605, 202610)  # 버스·생활 분석 테이블 보유 기간(src/agent/intent_parser.py TABLE_COVERAGE)
WASTE_YEARS, WIFI_YEARS = (2018, 2019, 2020, 2021), (2019, 2020, 2021)
CURRENT_YEAR = 2026


def mark_value(surface: str, **labels) -> dict:
    return {"surface": surface, "labels": labels}


def decide_abstain(year: int, season: str) -> bool:
    """연도·계절이 버스 테이블 보유 기간과 한 달도 겹치지 않으면 판정 보류(True)."""
    keys = [(year + 1 if season == "겨울" and month in (1, 2) else year) * 100 + month for month in SEASONS[season]]
    low, high = COMMUTE_COVERAGE
    return not any(low <= key <= high for key in keys)


@lru_cache(maxsize=1)
def load_gazetteer() -> dict:
    return json.loads((META / "jeju_gazetteer.json").read_text(encoding="utf-8"))


def build_areas() -> list[dict]:
    """일반 지명 묶음: 읍면동 이름·'제주시 OO동' 전체 이름·'노형'처럼 뗀 이름·'일도일동' 숫자 읽기·생활권 별칭."""
    gazetteer = load_gazetteer()
    values, seen = [], set()

    def add_value(surface: str, targets) -> None:
        key = surface.replace(" ", "")
        if key not in seen:
            seen.add(key)
            values.append(mark_value(surface, emds=sorted(targets)))

    for row in gazetteer["regions"]:
        name = row["emd_nm"]
        add_value(name, [name])
        add_value(row["full_nm"], [name])
        base = re.sub(r"(읍|면|동)$", "", name)
        if len(base) >= 2 and base not in RISKY_BASES and not re.search(r"\d", base):
            add_value(base, [name])
        for digit, word in (("1동", "일동"), ("2동", "이동"), ("3동", "삼동")):
            if name.endswith(digit):
                add_value(name[: -len(digit)] + word, [name])
    for row in gazetteer["aliases"]:
        if row["alias"] not in EXCLUDED_ALIASES:
            add_value(row["alias"], row["targets"])
    return values


def build_targets() -> dict[str, list[str]]:
    """표기 -> 읍면동. 해석기 색인과 같은 우선순위(읍면동 > 별칭 > 명소 > 정류소, 먼저 등록된 것)를 따른다.
    예: '삼화지구'는 정류소(삼양동)이기 전에 생활권 별칭(화북동·삼양동)이다(기준 질문 생성기도 같은 규칙)."""
    gazetteer = load_gazetteer()
    targets: dict[str, list[str]] = {}
    for row in gazetteer["regions"]:
        name = row["emd_nm"]
        for surface in [name, row["full_nm"], re.sub(r"(읍|면|동)$", "", name)]:
            targets.setdefault(surface.replace(" ", ""), [name])
    for row in gazetteer["aliases"]:
        targets.setdefault(row["alias"].replace(" ", ""), sorted(row["targets"]))
    for row in gazetteer["landmarks"]:
        targets.setdefault(row["name"].replace(" ", ""), [row["emd_nm"]])
    for row in gazetteer["bus_stations"]:
        targets.setdefault(row["bus_station_name"].replace(" ", ""), [row["emd_nm"]])
    return targets


def build_pools() -> dict[str, list[dict]]:
    gazetteer = load_gazetteer()
    city = {row["emd_nm"]: row["city"] for row in gazetteer["regions"]}
    targets = build_targets()
    pools = {"area": build_areas()}
    pools["stop"] = [mark_value(row["bus_station_name"], emds=targets[row["bus_station_name"].replace(" ", "")])
                     for row in gazetteer["bus_stations"]]
    pools["market"] = [mark_value(name, emds=targets[name.replace(" ", "")])
                       for name in [row["name"] for row in gazetteer["landmarks"]] + list(MARKET_NAMES)]
    for value in pools["area"] + pools["stop"] + pools["market"]:
        value["city"] = city[value["labels"]["emds"][0]]
    weather = ([mark_value(s, weather="rain") for s in RAIN] + [mark_value(s, weather="dry") for s in DRY])
    pools["weather"] = weather
    pools["cond"] = ([mark_value("")] + weather + [mark_value("평일", day_type="weekday"), mark_value("주중", day_type="weekday"),
                     mark_value("주말", day_type="weekend")]
                     + [mark_value(label, day_type=code) for label, code in DAY_CODES.items()])
    pools["evening"] = [mark_value(""), mark_value("퇴근 후", hour_from=18, hour_to=21),
                        mark_value("퇴근하고", hour_from=18, hour_to=21), mark_value("퇴근 끝나고", hour_from=18, hour_to=21),
                        mark_value("저녁에", hour_from=18, hour_to=21), mark_value("저녁 시간에", hour_from=18, hour_to=21),
                        mark_value("저녁 7시 이후", hour_from=19, hour_to=23),
                        mark_value("오후 6시부터 9시까지", hour_from=18, hour_to=21),
                        mark_value("저녁 8시 넘어서", hour_from=20, hour_to=23)]
    points = [mark_value(f"아침 {h}시에", hour_from=h, hour_to=h + 1) for h in (6, 7, 8, 9, 10)]
    points += [mark_value(f"오전 {h}시에", hour_from=h, hour_to=h + 1) for h in (9, 10, 11)]
    points += [mark_value(f"오후 {h}시에", hour_from=h + 12, hour_to=h + 13) for h in (1, 2, 3, 4, 5)]
    points += [mark_value(f"저녁 {h}시쯤", hour_from=h + 12, hour_to=h + 13) for h in (6, 7, 8, 9)]
    points += [mark_value(f"밤 {h}시에", hour_from=h + 12, hour_to=min(23, h + 13)) for h in (9, 10, 11)]
    points += [mark_value(f"{h}시에", hour_from=h, hour_to=h + 1) for h in (13, 15, 17, 19, 21)]
    points += [mark_value(f"새벽 {h}시에", hour_from=h, hour_to=h + 1) for h in (4, 5)]
    for surface, start, end in (("출근 시간에", 7, 9), ("출근길에", 7, 9), ("아침에", 7, 9), ("점심때", 12, 13),
                                ("점심시간에", 12, 13), ("새벽에", 0, 5), ("오전에", 9, 11), ("오후에", 13, 17),
                                ("저녁에", 18, 21), ("밤에", 21, 23), ("심야에", 22, 23), ("늦은 저녁에", 21, 23),
                                ("퇴근 시간에", 17, 19), ("퇴근길에", 17, 19)):
        points.append(mark_value(surface, hour_from=start, hour_to=end))
    pools["time_point"] = points
    after = []
    for h in (7, 8, 9, 10, 11):
        hours = {"hour_from": h + 12, "hour_to": 23}
        after += [mark_value(f"밤 {h}시 이후", **hours), mark_value(f"{h + 12}시 이후", **hours),
                  mark_value(f"밤 {KOREAN_NUMERALS[h]} 시 이후", **hours), mark_value(f"{h}pm 이후", **hours),
                  mark_value(f"밤 {h}시 넘은 시간", **hours), mark_value(f"밤 {h}시 지난 시간", **hours)]
        if h <= 9:
            after.append(mark_value(f"저녁 {h}시 이후", **hours))
    pools["after"] = after
    pools["wait_min"] = [mark_value(str(n), wait_minutes=n) for n in (5, 10, 15, 20, 25, 30, 40, 45)]
    pools["ev_min"] = [mark_value(str(n), wait_minutes=n) for n in (10, 12, 15, 17, 20, 25, 30, 40)]
    pools["old"] = [mark_value(s, user_group="어르신") for s in OLD_WORDS]
    pools["young"] = [mark_value(s, user_group="청소년") for s in YOUNG_WORDS]
    pools["group"] = pools["old"] + pools["young"]
    pools["old_cond"] = [mark_value(""), mark_value("평일 오전에", day_type="weekday", hour_from=9, hour_to=11),
                         mark_value("비 오는 날", weather="rain"),
                         mark_value("주말 오후에", day_type="weekend", hour_from=13, hour_to=17),
                         mark_value("아침에", hour_from=7, hour_to=9), mark_value("오후에", hour_from=13, hour_to=17),
                         mark_value("주말에", day_type="weekend"),
                         mark_value("비 오는 날 오전에", weather="rain", hour_from=9, hour_to=11),
                         mark_value("장날에", day_type="market")]
    pools["concept"] = [mark_value(s, abstain=True) for s in CONCEPTS]
    periods = [mark_value(f"{year}년 {season}", abstain=decide_abstain(year, season))
               for year in range(2018, 2027) for season in SEASONS]
    for word, year in (("재작년", CURRENT_YEAR - 2), ("작년", CURRENT_YEAR - 1), ("올해", CURRENT_YEAR)):
        periods += [mark_value(f"{word} {season}", abstain=decide_abstain(year, season)) for season in SEASONS]
    pools["period"] = periods
    pools["day"] = ([mark_value("주말", day_type="weekend"), mark_value("평일", day_type="weekday"),
                     mark_value("주중", day_type="weekday")]
                    + [mark_value(label, day_type=code) for label, code in DAY_CODES.items()])
    clocks = [mark_value(f"{h}시", hour_from=h + 12, hour_to=23) for h in range(6, 12)]
    clocks += [mark_value(f"{KOREAN_NUMERALS[h]} 시", hour_from=h + 12, hour_to=23) for h in range(6, 12)]
    clocks += [mark_value(f"{h}시", hour_from=h, hour_to=23) for h in range(18, 24)]
    clocks += [mark_value(f"저녁 {h}시", hour_from=h + 12, hour_to=23) for h in (6, 7, 8)]
    clocks += [mark_value(f"{h}pm", hour_from=h + 12, hour_to=23) for h in (7, 8, 9, 10)]
    pools["clock"] = clocks
    pools["ratio"] = [mark_value(r) for r in ("2", "3", "1.5", "2.5", "4", "5", "10", "1.2")]
    pools["year_waste"] = [mark_value(str(y), abstain=y not in WASTE_YEARS) for y in range(2016, 2026)]
    pools["year_wifi"] = [mark_value(str(y), abstain=y not in WIFI_YEARS) for y in range(2016, 2026)]
    for name, values in pools.items():
        for value in values:
            value["key"] = f"{name}:{value['surface']}"
            value["heldout_only"] = bool(value["surface"]) and compute_hash(value["key"]) % 4 == 0
    return pools


def compute_hash(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


# ---------------------------------------------------------------- 문장틀에 값 끼우기(조사 자동 맞춤)

# '(이)랑', '은(는)' 같은 두 형태 표기와 LLM 이 덧붙인 변형 표기('(이)는', '(이)에서')
NOTATION = re.compile(
    r"\(이\)(랑은|랑도|랑|나|야|라던데|라는|라고|라도|라면|고|면|며|든|에요|여|는|은|가|를|을|로|와|과|에서|에|의|도|만|까지|보다)"
    r"|(은\(는\)|는\(은\)|이\(가\)|가\(이\)|을\(를\)|를\(을\)|과\(와\)|와\(과\))|\(으\)(로)")
# 표기 없이 바로 붙인 조사(뒤가 공백·문장부호·끝일 때만 조사로 본다)
BARE = re.compile(r"(이랑은|랑은|이랑도|랑도|이랑|랑|이나|이야|이라던데|라던데|이라는|라는|이라고|라고|이에요|예요|은|는|이|가|을|를|"
                  r"과|와|으로|로)(?=[\s,.?!~]|$)")
PARTICLE_PAIRS = {
    "랑": ("이랑", "랑"), "랑은": ("이랑은", "랑은"), "랑도": ("이랑도", "랑도"), "나": ("이나", "나"), "야": ("이야", "야"),
    "라던데": ("이라던데", "라던데"), "라는": ("이라는", "라는"), "라고": ("이라고", "라고"), "라도": ("이라도", "라도"),
    "라면": ("이라면", "라면"), "고": ("이고", "고"), "면": ("이면", "면"), "며": ("이며", "며"), "든": ("이든", "든"),
    "에요": ("이에요", "예요"), "여": ("이여", "여"), "은": ("은", "는"), "는": ("은", "는"), "이": ("이", "가"),
    "가": ("이", "가"), "을": ("을", "를"), "를": ("을", "를"), "과": ("과", "와"), "와": ("과", "와"),
}
BARE_KEYS = {"이랑은": "랑은", "이랑도": "랑도", "이랑": "랑", "이나": "나", "이야": "야", "이라던데": "라던데",
             "이라는": "라는", "이라고": "라고", "이에요": "에요", "예요": "에요", "으로": "로"}
DIGIT_FINALS = {"0": True, "1": True, "2": False, "3": True, "4": False, "5": False, "6": True, "7": True, "8": True,
                "9": False}


def read_final(word: str) -> tuple[bool, bool]:
    """(받침 있음, 받침이 ㄹ). 숫자는 읽는 소리로 판단한다."""
    last = word[-1] if word else ""
    code = ord(last) - 0xAC00 if last else -1
    if 0 <= code < 11172:
        return code % 28 != 0, code % 28 == 8
    if last in DIGIT_FINALS:
        return DIGIT_FINALS[last], last in "178"
    return False, False


def pick_particle(word: str, form: str) -> str:
    final, rieul = read_final(word)
    if form == "로":
        return "으로" if final and not rieul else "로"
    if form in PARTICLE_PAIRS:
        with_final, without_final = PARTICLE_PAIRS[form]
        return with_final if final else without_final
    return form  # 에서·에·의·도·만·까지·보다처럼 받침과 무관한 조사


def read_particle(template: str, start: int) -> tuple[str | None, int]:
    """자리표시자 바로 뒤의 조사 표기 -> (조사 기본형, 소비한 글자 수)."""
    match = NOTATION.match(template, start)
    if match:
        form = match.group(1) or match.group(3) or match.group(2)[0]
        return form, match.end() - start
    match = BARE.match(template, start)
    if match:
        word = match.group(1)
        return BARE_KEYS.get(word, word), match.end() - start
    return None, 0


def check_attached(template: str, name: str) -> bool:
    """자리표시자 바로 뒤에 조사·글자가 붙어 있으면 빈 값을 끼울 수 없다."""
    position = template.find(f"<{name}>")
    following = template[position + len(name) + 2: position + len(name) + 3]
    return bool(following) and (following == "(" or "가" <= following <= "힣")


def fill_template(template: str, values: dict) -> tuple[str, list[tuple[int, int, str]]]:
    """자리표시자에 값을 끼우고 조사를 맞춘다. 돌려주는 spans 는 끼운 값의 (시작, 끝, 자리표시자 이름)."""
    pieces, position = [], 0
    for match in PLACEHOLDER.finditer(template):
        pieces.append((template[position:match.start()], None))
        name = match.group(1)
        surface = values[name]["surface"]
        form, used = read_particle(template, match.end())
        pieces.append((surface, name))
        if form is not None:
            pieces.append((pick_particle(surface, form) if surface else "", None))
        position = match.end() + used
    pieces.append((template[position:], None))
    text, spans = "", []
    for piece, name in pieces:
        if name is None:
            piece = re.sub(r"\s+", " ", piece)
            if (not text or text.endswith(" ")) and piece.startswith(" "):
                piece = piece[1:]
            text += piece
        elif piece:
            spans.append((len(text), len(text) + len(piece), name))
            text += piece
    return text.rstrip(), spans


BLANK = {"intent": None, "topic": "general", "emds": [], "area_count": 0, "hour_from": None, "hour_to": None,
         "weather": "any", "day_type": "any", "user_group": None, "wait_minutes": None, "abstain": False,
         "variable": None}


def compute_expected(frame: dict, values: dict) -> dict:
    """틀의 고정 정답 + 끼운 값의 정답 슬롯 = 이 문장의 정답(12개 슬롯)."""
    if frame.get("fixed"):
        return dict(frame["expect"])
    expected = {**BLANK, **frame["expect"]}
    emds, places = set(), 0
    for value in values.values():
        for slot, label in value["labels"].items():
            if slot == "emds":
                emds.update(label)
                places += 1
            else:
                expected[slot] = label
    if places:
        expected["emds"], expected["area_count"] = sorted(emds), places
    return expected


# ---------------------------------------------------------------- 입력 변형(잡음)

KINDS = ["none", "nospace", "punct", "emoticon", "prefix", "suffix", "whitespace", "fullwidth",
         "typo_place", "choseong_place", "typo_text"]
STRESS_KINDS = ("typo_place", "choseong_place", "typo_text")
ENDINGS = ["", "?", "??", "?!", "!", ".", "...", "~", "?~", "?!?"]
EMOTICONS = [" ㅋㅋ", "ㅋㅋㅋ", " ㅎㅎ", " ㅠㅠ", " ^^", " ;;", " \U0001F64F", " \U0001F605"]
PREFIXES = ["혹시 ", "저기 ", "음 ", "궁금한 게 있는데 ", "질문: ", "Q. ", "있잖아 ", "근데 "]
SUFFIXES = [" 알려줘", " 알려주세요", " 궁금해요", " 좀 알려줄래?", " 부탁해"]
PLACE_NAMES = ("P1", "P2", "STOP", "MARKET")
CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
        "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]
KEY_ROWS = ["ㅂㅈㄷㄱㅅㅛㅕㅑㅐㅔ", "ㅁㄴㅇㄹㅎㅗㅓㅏㅣ", "ㅋㅌㅊㅍㅠㅜㅡ"]  # 두벌식 자판
SHIFTED = {"ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅈ": "ㅉ", "ㄱ": "ㄲ", "ㅅ": "ㅆ", "ㅐ": "ㅒ", "ㅔ": "ㅖ"}


def build_neighbors() -> dict[str, list[str]]:
    position = {key: (row, col) for row, keys in enumerate(KEY_ROWS) for col, key in enumerate(keys)}
    neighbors = {}
    for key, (row, col) in position.items():
        around = [(row, col - 1), (row, col + 1), (row - 1, col), (row - 1, col + 1), (row + 1, col - 1), (row + 1, col)]
        found = [KEY_ROWS[r][c] for r, c in around if 0 <= r < len(KEY_ROWS) and 0 <= c < len(KEY_ROWS[r])]
        neighbors[key] = [k for k in found if (k in CHO) == (key in CHO)]  # 자음은 자음끼리, 모음은 모음끼리만 바꾼다
    return neighbors


NEIGHBORS = build_neighbors()


def split_syllable(char: str) -> tuple[int, int, int] | None:
    code = ord(char) - 0xAC00
    return (code // 588, (code % 588) // 28, code % 28) if 0 <= code < 11172 else None


def compose_syllable(cho: int, jung: int, jong: int) -> str:
    return chr(0xAC00 + (cho * 21 + jung) * 28 + jong)


def list_mutations(char: str, allow_split: bool) -> list[str]:
    """한 음절에 낼 수 있는 자판 오타(이웃 자판 바꾸기·쌍자음·받침 빼기·자모 풀어 쓰기) 후보."""
    parts = split_syllable(char)
    if parts is None:
        return []
    cho, jung, jong = parts
    out = [compose_syllable(CHO.index(other), jung, jong) for other in NEIGHBORS.get(CHO[cho], [])]
    out += [compose_syllable(cho, JUNG.index(other), jong) for other in NEIGHBORS.get(JUNG[jung], [])]
    if jong and JONG[jong] in NEIGHBORS:
        out += [compose_syllable(cho, jung, JONG.index(other)) for other in NEIGHBORS[JONG[jong]] if other in JONG]
    if CHO[cho] in SHIFTED:
        out.append(compose_syllable(CHO.index(SHIFTED[CHO[cho]]), jung, jong))
    if JUNG[jung] in SHIFTED:
        out.append(compose_syllable(cho, JUNG.index(SHIFTED[JUNG[jung]]), jong))
    if jong:
        out.append(compose_syllable(cho, jung, 0))
    if allow_split:
        out.append(CHO[cho] + JUNG[jung] + JONG[jong])
    return out


def build_exact_keys() -> set[str]:
    """해석기 지명 색인(src/tools/gazetteer_index.PlaceIndex)과 같은 규칙으로 만든 정확 일치 표기 전체."""
    gazetteer = load_gazetteer()
    keys = set()
    for row in gazetteer["regions"]:
        name = row["emd_nm"]
        keys.update({name, row["full_nm"].replace(" ", "")})
        base = re.sub(r"(읍|면|동)$", "", name)
        if len(base) >= 2 and base not in RISKY_BASES:
            keys.add(base)
        for digit, word in (("1동", "일동"), ("2동", "이동"), ("3동", "삼동")):
            if name.endswith(digit):
                keys.add(name[: -len(digit)] + word)
    keys.update(row["alias"].replace(" ", "") for row in gazetteer["aliases"])
    keys.update(row["name"].replace(" ", "") for row in gazetteer["landmarks"])
    keys.update(row["bus_station_name"].replace(" ", "") for row in gazetteer["bus_stations"])
    return keys


def extract_choseong(text: str) -> str:
    return "".join(CHO[(ord(c) - 0xAC00) // 588] if 0 <= ord(c) - 0xAC00 < 11172 else c for c in text)


@lru_cache(maxsize=1)
def load_place_keys() -> tuple[frozenset, frozenset]:
    """(정확 일치 표기, 초성만으로 하나의 지명을 가리키는 표기)."""
    keys = build_exact_keys()
    counts: dict[str, int] = {}
    for key in keys:
        counts[extract_choseong(key)] = counts.get(extract_choseong(key), 0) + 1
    unique = frozenset(k for k in keys if len(k) >= 3 and re.fullmatch(r"[가-힣]+", k) and counts[extract_choseong(k)] == 1)
    return frozenset(keys), unique


def compute_mix(number: int) -> int:
    """splitmix64: 문항 번호로 변형 세부 선택을 정한다(재현 가능)."""
    number = (number + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    number = ((number ^ (number >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    number = ((number ^ (number >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return number ^ (number >> 31)


def apply_typo(text: str, span: tuple[int, int], pick: int, check_place: bool) -> str | None:
    start, end = span
    candidates = [(offset, mutated) for offset in range(start, end)
                  for mutated in list_mutations(text[offset], allow_split=check_place and offset == end - 1)]
    if not candidates:
        return None
    keys = load_place_keys()[0] if check_place else frozenset()
    for step in range(len(candidates)):
        offset, mutated = candidates[(pick + step) % len(candidates)]
        surface = text[start:offset] + mutated + text[offset + 1:end]
        if check_place and surface.replace(" ", "") in keys:
            continue  # 오타가 다른 실제 지명이 되면 정답이 바뀌므로 쓰지 않는다
        return text[:offset] + mutated + text[offset + 1:]
    return None


def apply_noise(text: str, spans: list, kind: str, number: int) -> str | None:
    """입력 변형 하나를 적용한다. 적용할 수 없으면 None."""
    pick = compute_mix(number)
    if kind == "none":
        return text
    if kind == "nospace":
        return re.sub(r"\s+", "", text) if " " in text else None
    if kind == "punct":
        return re.sub(r"[?!.~\s]+$", "", text) + ENDINGS[pick % len(ENDINGS)]
    if kind == "emoticon":
        return text + EMOTICONS[pick % len(EMOTICONS)]
    if kind == "prefix":
        return PREFIXES[pick % len(PREFIXES)] + text
    if kind == "suffix":
        return text + SUFFIXES[pick % len(SUFFIXES)]
    if kind == "whitespace":
        mode = pick % 4
        if mode == 0:
            return text.replace(" ", "  ") if " " in text else None
        if mode == 1:
            return "  " + text + "  "
        gaps = [i for i, c in enumerate(text) if c == " "]
        if not gaps:
            return None
        cut = gaps[(pick >> 8) % len(gaps)]
        return text[:cut] + ("\t" if mode == 2 else "\n") + text[cut + 1:]
    if kind == "fullwidth":
        if not re.search(r"[0-9A-Za-z]", text):
            return None
        return "".join(chr(ord(c) + 0xFEE0) if c.isascii() and c.isalnum() else c for c in text)
    places = [(s, e) for s, e, name in spans if name in PLACE_NAMES]
    if kind == "typo_place":
        if not places:
            return None
        return apply_typo(text, places[pick % len(places)], pick >> 8, check_place=True)
    if kind == "choseong_place":
        eligible = [(s, e) for s, e in places if text[s:e] in load_place_keys()[1]]
        if not eligible:
            return None
        start, end = eligible[pick % len(eligible)]
        return text[:start] + extract_choseong(text[start:end]) + text[end:]
    if kind == "typo_text":
        covered = {i for s, e, _ in spans for i in range(s, e)}
        free = [i for i, c in enumerate(text) if i not in covered and split_syllable(c)]
        if not free:
            return None
        offset = free[pick % len(free)]
        return apply_typo(text, (offset, offset + 1), pick >> 8, check_place=False)
    raise ValueError(kind)


# ---------------------------------------------------------------- 문장틀 검사·불러오기·조합 열거

def check_template(frame: dict, text: str) -> str | None:
    """LLM 이 만든 틀의 형식 검사(두 세트에 같은 규칙). 통과하면 None, 아니면 탈락 사유."""
    if not 2 <= len(text) <= 200:
        return "length"
    if sorted(PLACEHOLDER.findall(text)) != sorted(PLACEHOLDER.findall(frame["text"])):
        return "placeholder_mismatch"
    rest = PLACEHOLDER.sub("", text)
    if "<" in rest or ">" in rest:
        return "broken_placeholder"
    for name, unit in UNIT_AFTER.items():
        if f"<{name}>" in text and not re.search(rf"<{name}>\s*{unit}", text):
            return "unit_missing"
    numbers = sorted(re.findall(r"\d+(?:\.\d+)?", rest))
    if numbers != sorted(re.findall(r"\d+(?:\.\d+)?", PLACEHOLDER.sub("", frame["text"]))):
        return "numbers_changed"
    return None


MODEL_FILES = {"gemma4": META / "scale_frames_gemma4.jsonl", "qwen3": META / "scale_frames_qwen3.jsonl"}
SET_SOURCES = {"dev": ("base", "gemma4"), "heldout": ("qwen3",)}


def load_templates(set_name: str) -> tuple[list[dict], dict]:
    """세트의 문장틀 목록과 탈락 통계. 같은 틀 안에서 공백만 다른 중복은 한 번만 쓴다."""
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
                reason = check_template(frame, text)
                normalized = re.sub(r"\s+", " ", text).strip()
                if reason is None and (frame["id"], normalized) in seen:
                    reason = "duplicate"
                if reason:
                    dropped[reason] = dropped.get(reason, 0) + 1
                    continue
                seen.add((frame["id"], normalized))
                number = numbers.get((source, frame["id"]), 0)
                numbers[(source, frame["id"])] = number + 1
                templates.append({"key": f"{set_name}:{source}:{frame['id']}:{number:04d}", "frame": frame["id"],
                                  "family": frame["family"], "workspace": frame["workspace"], "style": row["style"],
                                  "source": source, "text": normalized, "context": frame.get("context")})
    return templates, dropped


def build_pairs(areas: list[dict]) -> list[tuple[dict, dict]]:
    """비교 질문용 지명 쌍: 같은 시, 겹치는 읍면동 없음(기준 질문 생성기와 같은 조건)."""
    return [(a, b) for a in areas for b in areas
            if a is not b and a["city"] == b["city"] and not set(a["labels"]["emds"]) & set(b["labels"]["emds"])]


def plan_dimensions(template: dict, frame: dict, pools: dict, set_name: str) -> list[tuple[tuple, list]]:
    """문장틀의 조합 차원: [(자리표시자 이름들, 값 묶음 목록)]. 개발용은 평가용 전용 값을 뺀다."""
    def is_allowed(value: dict) -> bool:
        return set_name != "dev" or not value["heldout_only"]

    slots = frame["slots"]
    dims = []
    if "P1" in slots and "P2" in slots:
        dims.append((("P1", "P2"), build_pairs([v for v in pools["area"] if is_allowed(v)])))
    elif "P1" in slots:
        dims.append((("P1",), [(v,) for v in pools["area"] if is_allowed(v)]))
    for name, pool_name in slots.items():
        if name in ("P1", "P2"):
            continue
        values = [v for v in pools[pool_name] if is_allowed(v)]
        if check_attached(template["text"], name):
            values = [v for v in values if v["surface"]]
        dims.append(((name,), [(v,) for v in values]))
    return dims


def count_space(dims: list) -> int:
    total = len(KINDS)
    for _, options in dims:
        total *= len(options)
    return total


def pick_stride(space: int) -> int:
    """조합 공간 전체에 고르게 퍼지도록 공간 크기와 서로소인 보폭을 고른다."""
    stride = max(1, int(space * 0.6180339887)) | 1
    while gcd(stride, space) != 1:
        stride += 2
    return stride


def decode_case(index: int, dims: list) -> tuple[dict, int]:
    """조합 번호 -> ({자리표시자: 값}, 변형 번호)."""
    kind = index % len(KINDS)
    rest = index // len(KINDS)
    values = {}
    for names, options in reversed(dims):
        choice = options[rest % len(options)]
        rest //= len(options)
        for name, value in zip(names, choice):
            values[name] = value
    return values, kind


def build_case(template: dict, frame: dict, dims: list, index: int) -> dict:
    """조합 번호 하나 -> 시험 문항(문장·문맥·정답·실제 적용된 변형)."""
    values, kind_number = decode_case(index, dims)
    text, spans = fill_template(template["text"], values)
    context = fill_template(template["context"], values)[0] if template.get("context") else None
    noisy, kind = None, ""
    salt = compute_hash(template["key"])
    for step in range(len(KINDS)):
        kind = KINDS[(kind_number + step) % len(KINDS)]
        noisy = apply_noise(text, spans, kind, index ^ salt)
        if noisy is not None and noisy.strip():
            break
    return {"text": noisy, "context": context, "kind": kind, "expected": compute_expected(frame, values),
            "values": [value["key"] for value in values.values()],
            "unseen": any(value["heldout_only"] for value in values.values())}
