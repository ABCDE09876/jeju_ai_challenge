"""대규모 조합 행동 시험(scripts/scale_suite.py)의 정답 계산과, 그 시험에서 드러나 고친 해석기 동작을 확인한다."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import scale_suite as suite  # noqa: E402
from src.agent.intent_parser import extract_group, extract_weather, merge_dialogue_context, parse_question  # noqa: E402

POOLS = suite.build_pools()
FRAMES = {frame["id"]: frame for frame in suite.load_frames()}


def pick_value(pool: str, surface: str) -> dict:
    return next(value for value in POOLS[pool] if value["surface"] == surface)


def test_seed_reproduction():
    """기본 틀에 기준 질문의 값을 끼우면 기준 질문 문장과 정답(12개 슬롯)이 그대로 나온다."""
    seeds = {row["id"]: row for row in map(json.loads, (ROOT / "data/meta/robustness_seeds.jsonl").read_text(
        encoding="utf-8").splitlines())}
    cases = [("S0001", "F01", {"P1": ("area", "탑동"), "P2": ("area", "이호테우"), "COND": ("cond", "")}),
             ("S0080", "F06", {"STOP": ("stop", "이호테우해변")}),
             ("S0110", "F08", {"P1": ("area", "일도2동"), "MIN": ("wait_min", "10")}),
             ("S0141", "F10", {"OLD": ("old", "어르신"), "P1": ("area", "신화월드"), "OLDCOND": ("old_cond", "평일 오전에")}),
             ("S0208", "F15", {"PERIOD": ("period", "2022년 여름"), "P1": ("area", "중문단지")}),
             ("S0225", "F18", {"P1": ("area", "천지동"), "P2": ("area", "예래동")}),
             ("S0226", "F19", {"P1": ("area", "천지동"), "CLOCK": ("clock", "8시")}),
             ("S0317", "F23", {"P1": ("area", "바오젠거리"), "P2": ("area", "월정리")}),
             ("S0356", "F27", {"P1": ("area", "대천동"), "YEAR": ("year_wifi", "2020")})]
    for seed_id, frame_id, picks in cases:
        frame = FRAMES[frame_id]
        values = {name: pick_value(pool, surface) for name, (pool, surface) in picks.items()}
        text = suite.fill_template(frame["text"], values)[0]
        context = suite.fill_template(frame["context"], values)[0] if frame.get("context") else None
        assert (text, context) == (seeds[seed_id]["text"], seeds[seed_id]["context"])
        assert suite.compute_expected(frame, values) == seeds[seed_id]["reference"]


def test_particle_choice():
    fill = suite.fill_template
    assert fill("<P1>(이)랑 <P2> 중", {"P1": pick_value("area", "신제주"), "P2": pick_value("area", "애월")})[0] == "신제주랑 애월 중"
    assert fill("<P1>은(는) 좋아", {"P1": pick_value("area", "함덕")})[0] == "함덕은 좋아"
    assert fill("<P1>(으)로 가", {"P1": pick_value("area", "애월")})[0] == "애월로 가"
    assert fill("<P1>(이)로 가", {"P1": pick_value("area", "노형")})[0] == "노형으로 가"
    assert fill("<OLD>이(가) 타", {"OLD": pick_value("old", "할머니")})[0] == "할머니가 타"
    assert fill("<P1>는 어때", {"P1": pick_value("area", "연동")})[0] == "연동은 어때"
    assert fill("중에 <COND> 퇴근", {"COND": pick_value("cond", "")})[0] == "중에 퇴근"


def test_case_determinism():
    templates, _ = suite.load_templates("dev")
    template = next(t for t in templates if t["frame"] == "F01")
    frame = FRAMES["F01"]
    dims = suite.plan_dimensions(template, frame, POOLS, "dev")
    space = suite.count_space(dims)
    first = [suite.build_case(template, frame, dims, (i * 7919) % space) for i in range(50)]
    again = [suite.build_case(template, frame, dims, (i * 7919) % space) for i in range(50)]
    assert first == again
    assert {case["kind"] for case in first} <= set(suite.KINDS)
    assert all(not case["unseen"] for case in first)  # 개발용에는 평가용 전용 값이 없다


def test_typo_safety():
    keys = suite.load_place_keys()[0]
    text, spans = suite.fill_template("<P1> 버스 어때?", {"P1": pick_value("area", "아라동")})
    for number in range(300):
        noisy = suite.apply_noise(text, spans, "typo_place", number)
        assert noisy.split()[0].replace(" ", "") not in keys


def ask(question: str, previous: str | None = None, workspace: str = "life"):
    plan = parse_question(question, workspace)
    return merge_dialogue_context(plan, parse_question(previous, workspace)) if previous else plan


def test_glued_input():
    """띄어쓰기를 모두 뺀 문장, 조사가 붙은 초성 지명, 전각 숫자도 같은 해석이 나온다."""
    plan = ask("노형동이랑외도동중퇴근후생활하기어디가나아?")
    assert (plan.areas, plan.intent, plan.topic, plan.hour_from) == (["노형동", "외도동"], "compare", "living", 18)
    plan = ask("애월은저녁에도민이조천읍보다많다던데맞아?", workspace="lawyer")
    assert (plan.areas, plan.intent) == (["애월읍", "조천읍"], "fact_check")
    assert ask("거기말고제주공항이랑비교해줘", "삼화지구 퇴근 버스 어때?").areas == ["삼화지구", "제주공항"]
    assert ask("ㄴㅎㄷ이랑 외도동 중 퇴근 버스 어디가 나아?").areas == ["노형동", "외도동"]
    plan = ask("일도２동 밤 ９시 이후에도 버스 다녀?")
    assert (plan.areas, plan.hour_from, plan.intent) == (["일도2동"], 21, "boundary")
    assert ask("몇시에가면 덜 붐벼? 동문시장 장보러").variable == "visit_hour"


def test_scale_fixes():
    assert ask("제주시 한경면 대기질 어때?").intent == "profile"  # '대기질'의 '대기'는 버스 대기가 아니다
    assert ask("대천 대기질 어때?").needs_abstain
    assert extract_group("나이 드신 분이 노형에서 버스 타기 괜찮을까?") == "어르신"
    assert extract_weather("비 많이 오는 날 퇴근 버스 어때?") == "rain"
    assert extract_weather("비가 많이 와서 연동 카드 매출이 줄었다는데 맞아?") is None  # 원인 서술은 조건이 아니다
    assert ask("전기차 충전 대기시간이 평균 40분이라는 보도가 사실인지 친절하게 좀 가르쳐줄 수 있겠나?",
               workspace="lawyer").needs_abstain is None  # 부탁 말투의 '친절'은 만족도 질문이 아니다
    assert ask("노형동 버스 기사들 친절해?").needs_abstain


def test_limit_audit():
    """한계 구획의 자료 설명('113행, 전체 2,447행의 4.6%') 속 숫자도 수치 감사의 근거로 인정한다."""
    from src.agent.orchestrator import HybridOrchestrator
    from src.agent.response_cache import ResponseCache

    engine = HybridOrchestrator("lawyer", force_offline=True, cache=ResponseCache(enabled=False))
    for question in ("2023년 전기차 충전 대기시간이 평균 40분이라는 보도 맞아?",
                     "전기차 충전 내기시간이 평균 40분이라는 보도가 사실인지 친절하게 좀 가르쳐줄 수 있겠나?"):
        engine.reset()
        bundle = engine.answer(question)
        assert "4.6%" in bundle["narrative"]
        assert bundle["number_audit"]["ratio"] == 1.0, bundle["number_audit"]
