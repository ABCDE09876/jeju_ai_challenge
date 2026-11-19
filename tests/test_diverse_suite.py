"""다양성 대규모 시험(scripts/diverse_suite.py)의 라벨 검증기·값 묶음·정답 계산·입력 변형을 확인한다."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import diverse_suite as suite  # noqa: E402

POOLS = suite.build_pools()
FRAMES = {frame["id"]: frame for frame in suite.load_frames()}


def pick_value(pool: str, surface: str) -> dict:
    return next(value for value in POOLS[pool] if value["surface"] == surface)


def test_meaning_check():
    """LLM 틀이 조건을 덧붙이거나 빼면 걸러 내고, 말투만 바뀐 틀은 통과시킨다."""
    check = suite.check_meaning
    assert check(FRAMES["F03"], "혹시 <P1> <TIME> 버스 타기 괜찮은지 궁금합니다") is None
    assert check(FRAMES["F03"], "<P1> <TIME> 버스 어떵허우꽈?") is None
    assert check(FRAMES["F01"], "<P1>(이)랑 <P2> 중에 <COND> 퇴근 버스 어디가 나아? 이사 와서 잘 몰라요") is None
    assert check(FRAMES["F03"], "<P1> <TIME> 버스 혼잡도 알려줘") is None  # 버스 혼잡은 버스 주제 안의 말
    assert check(FRAMES["F03"], "<P1> <TIME> 아침 버스 어때?") == "added_time"
    assert check(FRAMES["F03"], "비 오는 날 <P1> <TIME> 버스 어때?") == "added_weather"
    assert check(FRAMES["F03"], "주말에 <P1> <TIME> 버스 어때?") == "added_day"
    assert check(FRAMES["F03"], "할머니랑 <P1> <TIME> 버스 타도 돼?") == "added_group"
    assert check(FRAMES["F03"], "<P1> <TIME> 버스 괜찮다던데 맞아?") == "added_claim"
    assert check(FRAMES["F03"], "<P1> <TIME> 버스 몇 시까지 있어?") == "added_boundary"
    assert check(FRAMES["F03"], "공항에서 <P1> <TIME> 버스 어때?") == "added_place"
    assert check(FRAMES["F05"], "<P1> <AFTER>에도 버스에 사람 많아?") == "added_topic"
    assert check(FRAMES["F01"], "<P1>(이)랑 <P2> 중에 <COND> 버스 타기 어디가 나아?") == "dropped_time"
    assert check(FRAMES["F01"], "<P1>(이)랑 <P2> 중에 <COND> 퇴근하고 버스 타기 어디가 나아?") == "changed_time"
    assert check(FRAMES["F06"], "<STOP> 정류소 주말에 붐벼?") == "added_day"
    assert check(FRAMES["G08"], "<P1>(이)랑 <P2>은(는) <TIME> 버스 어때?") == "dropped_cue"
    assert check(FRAMES["F04"], "<P1>(이)랑(은) <AFTER>에도 버스 다녀?") == "notation_leftover"
    assert check(FRAMES["F14"], "<P1> <CONCEPT> 친절하게 알려줘") is None  # 부탁 말투의 '친절'은 만족도 개념이 아니다


def test_vocab_labels():
    """확장 어휘 값의 정답 슬롯이 규약과 맞고, 다른 조건이 섞인 값은 없다."""
    for value in POOLS["time_point"]:
        start, end = value["labels"]["hour_from"], value["labels"]["hour_to"]
        assert 0 <= start <= end <= 23, value
        if end - start <= 1:  # 시각 값: 표기 속 숫자 시각이 라벨과 맞는다(12시간제 포함)
            assert suite.check_hour(value["surface"], start), value
    for value in POOLS["after"]:
        assert value["labels"]["hour_to"] in (5, 23)
    allowed = {"weather": suite.VOCAB_ALLOWED["weather"], "day": suite.VOCAB_ALLOWED["day"],
               "old": suite.VOCAB_ALLOWED["group"], "young": suite.VOCAB_ALLOWED["group"],
               "concept": suite.VOCAB_ALLOWED["concept"] | {"group", "boundary"}}  # 1차 값 '대기질'·'학교 평판' 포함
    for name, categories in allowed.items():
        for value in POOLS[name]:
            assert {c for c, _ in suite.find_concepts(value["surface"])} <= categories, value
    for value in POOLS["form"]:
        assert value["surface"].count("{}") == 1
    assert POOLS["form"][0]["surface"] == "{}" and not POOLS["form"][0]["heldout_only"]
    assert not POOLS["ctx"][0]["heldout_only"] and POOLS["ctx"][0]["text"] == suite.FOLLOW
    for value in POOLS["nightword"]:
        assert ("topic", "night") in suite.find_concepts(value["surface"])
    assert suite.convert_item("time_point", {"surface": "저녁 7시에", "hour24": 7})[1] == "hour_conflict"
    assert suite.convert_item("time_point", {"surface": "비 오는 아침에", "period": "아침"})[1] == "other_condition"
    assert suite.convert_item("weather", {"surface": "비 안 오는 날", "weather": "rain"})[1] == "label_conflict"
    assert suite.convert_item("day", {"surface": "일요일", "day": "토"})[1] == "label_conflict"
    assert suite.convert_item("place_form", {"surface": "{} 쪽에"})[1] == "label"
    assert suite.convert_item("time_point", {"surface": "오후 세 시 반쯤", "hour24": 15})[0]["labels"] == {
        "hour_from": 15, "hour_to": 16}


def test_expected_labels():
    """'X 말고 Y'와 앞 질문에만 나온 지명은 정답 지역에서 빠지고, 세 곳 비교는 지역 셋이다."""
    p1, p2 = pick_value("area", "노형동"), pick_value("area", "연동")
    time = pick_value("time_point", "아침 7시에")
    expected = suite.compute_expected(FRAMES["G08"], {"P1": p1, "P2": p2, "TIME": time})
    assert (expected["emds"], expected["area_count"], expected["hour_from"]) == (["연동"], 1, 7)
    expected = suite.compute_expected(FRAMES["G19"], {"P1": p1, "P2": p2})
    assert (expected["emds"], expected["area_count"], expected["intent"]) == (["연동"], 1, "profile")
    triples = suite.build_triples(POOLS["emd"])
    assert triples and all(len({e for v in t for e in v["labels"]["emds"]}) == 3 for t in triples[:500])
    expected = suite.compute_expected(FRAMES["G07"], dict(zip(("P1", "P2", "P3"), triples[0])))
    assert expected["area_count"] == 3 and expected["intent"] == "compare"


def test_case_building():
    """조합 번호 -> 문항이 결정적이고, 개발용에는 평가용 전용 요소가 하나도 없으며, 지명 감싸기는 지명 자리만 바꾼다."""
    template = {"key": "dev:base:G08:00000", "frame": "G08", "family": "exclusion", "workspace": "life",
                "style": "canonical", "source": "base", "text": FRAMES["G08"]["text"], "context": None}
    dims = suite.plan_dimensions(template, FRAMES["G08"], POOLS, "dev")
    space = suite.count_space(dims, "dev")
    first = [suite.build_case(template, FRAMES["G08"], dims, (i * 7919) % space, "dev") for i in range(300)]
    again = [suite.build_case(template, FRAMES["G08"], dims, (i * 7919) % space, "dev") for i in range(300)]
    assert first == again
    assert all(not case["unseen"] and case["kind"] not in suite.HELDOUT_KINDS for case in first)
    assert {case["kind"] for case in first} <= set(suite.KIND_SETS["dev"])
    form = next(v for v in POOLS["form"] if v["surface"] == "{} 근처")
    shown, offsets = suite.apply_forms(template["text"], {"P1": POOLS["area"][0], "P2": POOLS["area"][1]}, form)
    assert shown["P1"]["surface"].endswith(" 근처") and offsets["P1"] == (0, len(POOLS["area"][0]["surface"]))
    follow = {"key": "heldout:qwen3:F16:00000", "frame": "F16", "family": "followup_weather", "workspace": "life",
              "style": "jeolla", "source": "qwen3", "text": FRAMES["F16"]["text"], "context": suite.FOLLOW,
              "situation": {"speaker": "간호사", "channel": "메신저 대화", "purpose": "숙소를 고르려고", "emotion": "걱정"}}
    dims = suite.plan_dimensions(follow, FRAMES["F16"], POOLS, "heldout")
    case = suite.build_case(follow, FRAMES["F16"], dims, 12345, "heldout")
    assert {"style", "situation"} <= set(case["novel_parts"]) and case["unseen"]
    assert "퇴근" in case["context"]


def test_noise_kinds():
    """새 입력 변형은 지명 글자를 건드리지 않거나(엄격) 지명 안 띄어쓰기만 넣는다(부하)."""
    text, spans = suite.base.fill_template("<P1> 퇴근 버스 어때?", {"P1": pick_value("area", "누웨마루거리")})
    for kind in suite.KINDS:
        for number in range(40):
            noisy = suite.apply_noise(text, spans, kind, number)
            if noisy is None or kind in ("typo_place", "choseong_place", "space_in_place", "nospace", "partial_space"):
                continue
            assert "누웨마루거리" in noisy, (kind, noisy)
    assert suite.apply_noise(text, spans, "space_in_place", 7).replace(" ", "").startswith("누웨마루거리")
    for phrase in suite.GREETINGS + suite.CLOSINGS + suite.EN_PREFIXES + suite.EN_SUFFIXES:
        assert not suite.find_concepts(phrase), phrase
