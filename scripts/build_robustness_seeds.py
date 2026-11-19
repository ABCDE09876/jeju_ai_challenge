"""견고성 시험용 기준 질문(364문항) 생성기.

템플릿마다 의도한 슬롯(intended_slots: 지역·시간대·날씨·요일·이용자·대기시간·판정 보류 등)을 직접 정하고,
나머지 슬롯은 기준 문장을 현재 해석기로 해석한 값으로 채워 reference 를 만든다. 무작위 선택은 시드 20261120 고정.

주의: 제출물의 data/meta/robustness_seeds.jsonl 은 2026-11-16 해석기 수정 전에 이 절차로 만든 뒤 고정한 파일이다
(수정 전 해석기 값이 틀렸던 2문항은 note 를 달고 바로잡음). 이 스크립트를 다시 돌리면 현재 해석기 기준으로 새 파일을
만들므로 기본 출력은 data/processed/robustness_seeds_rebuilt.jsonl 이다(고정 파일을 덮어쓰지 않음).
python scripts/build_robustness_seeds.py [--out 경로]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.agent.intent_parser import merge_dialogue_context, parse_question  # noqa: E402
from src.config import META_DIR, PROCESSED_DIR, SEED  # noqa: E402
from src.tools.gazetteer_index import load_place_index  # noqa: E402

SCENARIOS = {
    "life": ["애월이 아라동보다 살기 활기찬 동네라던데, 도민 기준으로도 맞아?",
             "동문시장 쪽 일도1동이랑 이도1동 중에 밤 9시 이후 생활하기 어디가 나아?",
             "삼화지구로 이사 가면 퇴근 버스 괜찮을까? 아라동이랑 비교해 줘",
             "비 오는 평일 퇴근길에 삼화지구 정류소가 언제 붐벼?",
             "오일장 장날 몇 시에 가면 덜 붐벼? 어르신 모시고 가려고",
             "신제주랑 구제주 중 퇴근 후 저녁 생활 어디가 나아?"],
    "lawyer": ["한림읍 음식물쓰레기가 2019년 12월에 10월보다 58.69배 폭증했다. 관광객 때문이다.",
               "제주 공공와이파이 이용자가 2020년 10월에 반토막 났다던데 맞아?",
               "도두동 오일장 정류소는 평균 이용객이 적어서 버스 증편이 필요 없다는 주장, 반박할 근거 있어?",
               "버스 이용이 줄어든 건 전기차 보급 때문이라는 기사 맞아?",
               "전기차 충전 대기시간은 평균 17분이라 짧다는 기사 맞아?",
               "노형동은 저녁에 도민 유동인구가 연동보다 많다는데 맞아?"],
}
EXCLUDED_ALIASES = ("오일장", "민속오일장", "공항", "시청", "영교", "우도", "추자도")


class SeedBook:
    """템플릿 질문을 모으고 기준 슬롯을 계산한다."""

    def __init__(self):
        gazetteer = json.loads((META_DIR / "jeju_gazetteer.json").read_text(encoding="utf-8"))
        self.index = load_place_index()
        self.rng = random.Random(SEED)
        self.emds = [row["emd_nm"] for row in gazetteer["regions"]]
        self.city = {row["emd_nm"]: row["city"] for row in gazetteer["regions"]}
        self.stations = [row["bus_station_name"] for row in gazetteer["bus_stations"]]
        self.landmarks = [row["name"] for row in gazetteer["landmarks"]]
        aliases = [row["alias"] for row in gazetteer["aliases"]]
        self.area_pool = self.emds + [a for a in aliases if a not in EXCLUDED_ALIASES]
        self.seeds: list[dict] = []

    def find_emds(self, surface: str) -> set:
        entry = self.index.exact.get(surface.replace(" ", ""))
        if entry is None:
            raise ValueError(f"사전에 없는 지명: {surface}")
        return set(entry.targets)

    def collect_emds(self, labels) -> list[str]:
        found = set()
        for label in labels:
            if label in self.index.alias_rows:
                found.update(self.index.alias_rows[label]["targets"])
            elif label in self.index.regions:
                found.add(label)
            else:
                match = self.index.match_token(label)
                found.update(match.targets if match else [f"?{label}"])
        return sorted(found)

    def summarize_plan(self, plan) -> dict:
        return {"intent": plan.intent, "topic": plan.topic, "emds": self.collect_emds(plan.areas),
                "area_count": len(plan.areas), "hour_from": plan.hour_from, "hour_to": plan.hour_to,
                "weather": plan.weather, "day_type": plan.day_type, "user_group": plan.user_group,
                "wait_minutes": plan.wait_minutes, "abstain": plan.needs_abstain is not None, "variable": plan.variable}

    def pick_pair(self) -> tuple[str, str]:
        while True:
            first, second = self.rng.sample(self.area_pool, 2)
            left, right = self.find_emds(first), self.find_emds(second)
            if left & right or self.city.get(sorted(left)[0]) != self.city.get(sorted(right)[0]):
                continue
            return first, second

    def add_seed(self, family: str, workspace: str, text: str, intended: dict | None = None,
                 context: str | None = None) -> None:
        self.seeds.append({"family": family, "workspace": workspace, "text": text, "context": context,
                           "intended": intended or {}})


def attach_josa(word: str, with_final: str, without_final: str) -> str:
    code = ord(word[-1]) - 0xAC00
    return word + (with_final if 0 <= code < 11172 and code % 28 else without_final)


def build_life(book: SeedBook) -> None:
    rng, emds = book.rng, book.find_emds
    for cond, weather, day in [("", "any", "any"), ("비 오는 날 ", "rain", "any"), ("평일 ", "any", "weekday"),
                               ("주말 ", "any", "weekend")] * 4:
        a, b = book.pick_pair()
        book.add_seed("compare_commute", "life", f"{attach_josa(a, '이랑', '랑')} {b} 중에 {cond}퇴근 버스 타기 어디가 나아?",
                      {"intent": "compare", "topic": "commute", "emds": sorted(emds(a) | emds(b)), "area_count": 2,
                       "hour_from": 17, "hour_to": 19, "weather": weather, "day_type": day})
    for cond, hours in [("", (None, None)), ("퇴근 후 ", (18, 21)), ("저녁에 ", (18, 21))] * 5:
        a, b = book.pick_pair()
        book.add_seed("compare_living", "life", f"{attach_josa(a, '이랑', '랑')} {b} 중 {cond}생활하기 어디가 나아?",
                      {"intent": "compare", "topic": "living", "emds": sorted(emds(a) | emds(b)), "area_count": 2,
                       "hour_from": hours[0], "hour_to": hours[1]})
    for phrase, hours in [("출근 시간에", (7, 9)), ("아침 8시에", (8, 9)), ("저녁 7시쯤", (19, 20)), ("점심때", (12, 13)),
                          ("새벽에", (0, 5)), ("오후 3시에", (15, 16))] * 3:
        a = rng.choice(book.area_pool)
        book.add_seed("profile_time", "life", f"{a} {phrase} 버스 어때?",
                      {"intent": "profile", "topic": "commute", "emds": sorted(emds(a)), "hour_from": hours[0],
                       "hour_to": hours[1]})
    for hour in [8, 9, 10] * 5:
        a = rng.choice(book.area_pool)
        book.add_seed("boundary_after", "life", f"{a}에서 밤 {hour}시 이후에도 버스 다녀?",
                      {"intent": "boundary", "topic": "commute", "emds": sorted(emds(a)), "hour_from": hour + 12,
                       "hour_to": 23, "variable": "return_hour"})
    for hour in [9, 10, 11] * 5:
        a = rng.choice(book.area_pool)
        book.add_seed("night_crowd", "life", f"{a} 밤 {hour}시 넘어서도 사람 많아?",
                      {"intent": "profile", "topic": "crowd", "emds": sorted(emds(a)), "hour_from": hour + 12,
                       "hour_to": 23})
    for station in rng.sample(book.stations, 15):
        book.add_seed("market_crowd", "life", f"{station} 정류소 장날에 붐벼?",
                      {"intent": "profile", "topic": "commute", "emds": sorted(emds(station)), "day_type": "market"})
    for station in rng.sample(book.stations, 15):
        book.add_seed("hidden_slice", "life", f"{station} 정류소는 평균은 한산한데 언제 붐벼?",
                      {"intent": "hidden_slice", "topic": "commute", "emds": sorted(emds(station))})
    for minutes in [10, 15, 20, 30] * 4:
        a = rng.choice(book.area_pool)
        book.add_seed("boundary_wait", "life", f"{a} 정류소에서 {minutes}분까지 기다릴 수 있으면 몇 시까지 버스 타면 돼?",
                      {"intent": "boundary", "topic": "commute", "emds": sorted(emds(a)), "wait_minutes": minutes,
                       "variable": "wait_minutes"})
    for place in rng.sample(book.landmarks, 12) + ["오일장", "동문시장", "올레시장"]:
        book.add_seed("boundary_visit", "life", f"{place} 장 보러 몇 시에 가는 게 덜 붐벼?",
                      {"intent": "boundary", "topic": "crowd", "emds": sorted(emds(place)), "variable": "visit_hour"})
    for cond, day, hours in [("평일 오전에 ", "weekday", (9, 11)), ("비 오는 날 ", "any", (None, None)),
                             ("", "any", (None, None))] * 5:
        a = rng.choice(book.area_pool)
        book.add_seed("group_elderly", "life", f"어르신이 {a}에서 {cond}버스 타기 괜찮을까?",
                      {"intent": "profile", "topic": "commute", "emds": sorted(emds(a)), "user_group": "어르신",
                       "day_type": day, "weather": "rain" if "비" in cond else "any", "hour_from": hours[0],
                       "hour_to": hours[1]})
    for a in rng.sample(book.area_pool, 10):
        book.add_seed("group_youth", "life", f"학생들이 {a}에서 버스 타고 다니기 어때?",
                      {"intent": "profile", "topic": "commute", "emds": sorted(emds(a)), "user_group": "청소년"})
    for a in rng.sample(book.area_pool, 12):
        book.add_seed("weather_dry", "life", f"비 안 오는 날 {a} 퇴근 버스 어때?",
                      {"intent": "profile", "topic": "commute", "emds": sorted(emds(a)), "weather": "dry",
                       "hour_from": 17, "hour_to": 19})
    for a in rng.sample(book.area_pool, 12):
        book.add_seed("spending", "life", f"{a} 상권 저녁 매출은 도민이 많이 써?",
                      {"topic": "spending", "emds": sorted(emds(a)), "hour_from": 18, "hour_to": 21})
    for concept in ["집값", "층간소음", "치안", "학군", "미세먼지", "주차난"] * 3:
        a = rng.choice(book.area_pool)
        book.add_seed("abstain_unmeasured", "life", f"{a} {concept} 어때?", {"emds": sorted(emds(a)), "abstain": True})
    for year, season in [(2022, "여름"), (2023, "겨울"), (2024, "봄"), (2024, "가을"), (2023, "여름")] * 3:
        a = rng.choice(book.area_pool)
        book.add_seed("abstain_period", "life", f"{year}년 {season} {a} 버스 붐볐어?",
                      {"topic": "commute", "emds": sorted(emds(a)), "abstain": True})
    for _ in range(8):
        a, b = book.pick_pair()
        previous = f"{a} 퇴근 버스 어때?"
        for follow, extra in [("그럼 비 오는 날은?", {"weather": "rain"}), ("주말은요?", {"day_type": "weekend"}),
                              (f"거기 말고 {attach_josa(b, '이랑', '랑')} 비교해 줘",
                               {"emds": sorted(emds(a) | emds(b)), "area_count": 2}),
                              ("그럼 8시 이후는?", {"hour_from": 20, "hour_to": 23}),
                              ("어르신 기준으로는?", {"user_group": "어르신"})]:
            book.add_seed("followup", "life", follow, {"topic": "commute", "emds": sorted(emds(a)), **extra}, previous)


def build_lawyer(book: SeedBook) -> None:
    rng, emds = book.rng, book.find_emds
    benchmark = json.loads((META_DIR / "benchmark_items.json").read_text(encoding="utf-8"))
    for item in benchmark["items"]:
        book.add_seed("benchmark", item["workspace"], item["question"])
    for workspace, questions in SCENARIOS.items():
        for question in questions:
            if question not in [s["text"] for s in book.seeds]:
                book.add_seed("scenario", workspace, question)
    for ratio in ["2", "3", "1.5"] * 4:
        a = rng.choice(book.area_pool)
        book.add_seed("trend_claim", "lawyer", f"{a} 버스 이용객이 {ratio}배 늘었다는 기사 사실이야?",
                      {"intent": "fact_check", "topic": "commute", "emds": sorted(emds(a))})
    for a in rng.sample(book.area_pool, 12):
        book.add_seed("causal_claim", "lawyer", f"{a} 카드 매출이 줄어든 건 관광객 때문이라던데 맞아?",
                      {"intent": "fact_check", "topic": "spending", "emds": sorted(emds(a))})
    for _ in range(12):
        a, b = book.pick_pair()
        book.add_seed("compare_claim", "lawyer", f"{attach_josa(a, '은', '는')} 저녁에 도민이 {b}보다 많다던데 맞아?",
                      {"intent": "fact_check", "emds": sorted(emds(a) | emds(b)), "area_count": 2,
                       "hour_from": 18, "hour_to": 21})
    for station in rng.sample(book.stations, 12):
        book.add_seed("slice_claim", "lawyer", f"{station} 정류소는 평균 이용객이 적어서 증편이 필요 없다는 답변, 맞아?",
                      {"intent": "fact_check", "topic": "commute", "emds": sorted(emds(station))})
    for minutes in [15, 17, 20, 25, 30, 40]:
        book.add_seed("ev_claim", "lawyer", f"전기차 충전 대기시간이 평균 {minutes}분이라는 보도 믿어도 돼?",
                      {"intent": "fact_check", "topic": "ev", "wait_minutes": minutes})
    for year, abstain in [(2019, False), (2020, False), (2023, True)] * 3:
        a = rng.choice(book.emds)
        book.add_seed("waste_claim", "lawyer", f"{a} 음식물 쓰레기가 {year}년에 급증했다는 주장 검증해 줘",
                      {"intent": "fact_check", "topic": "waste", "emds": [a], "abstain": abstain})
    for year, abstain in [(2020, False), (2021, False), (2024, True)] * 3:
        a = rng.choice(book.emds)
        book.add_seed("wifi_claim", "lawyer", f"{a} 공공와이파이 이용자가 {year}년에 반토막 났다던데 맞아?",
                      {"intent": "fact_check", "topic": "wifi", "emds": [a], "abstain": abstain})


def main() -> int:
    parser = argparse.ArgumentParser(description="견고성 시험 기준 질문 생성")
    parser.add_argument("--out", default=str(PROCESSED_DIR / "robustness_seeds_rebuilt.jsonl"))
    args = parser.parse_args()
    book = SeedBook()
    build_life(book)
    build_lawyer(book)
    rows, issues = [], 0
    for number, seed in enumerate(book.seeds, start=1):
        plan = parse_question(seed["text"], seed["workspace"])
        if seed["context"]:
            plan = merge_dialogue_context(plan, parse_question(seed["context"], seed["workspace"]))
        parsed = book.summarize_plan(plan)
        wrong = [slot for slot, value in seed["intended"].items() if parsed[slot] != value]
        issues += bool(wrong)
        rows.append({"id": f"S{number:04d}", "family": seed["family"], "workspace": seed["workspace"],
                     "text": seed["text"], "context": seed["context"], "reference": {**parsed, **seed["intended"]},
                     "intended_slots": sorted(seed["intended"]),
                     "note": f"기준 문장 해석이 의도와 다른 슬롯: {wrong}" if wrong else ""})
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"기준 질문 {len(rows)}개, 기준 문장 해석이 의도와 다른 질문 {issues}개 -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
