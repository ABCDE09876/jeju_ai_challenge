"""1억 문항 확장 시험(scripts/mega_suite.py)의 새 값·새 입력 변형·평가용 전용 표시를 확인한다."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import diverse_suite as prior  # noqa: E402
import mega_suite as suite  # noqa: E402

POOLS = suite.build_pools()
FRAMES = {frame["id"]: frame for frame in suite.load_frames()}


def pick_area(surface: str) -> dict:
    return next(value for value in POOLS["area"] if value["surface"] == surface)


def test_kind_sets():
    """새 변형은 개발용 5종·평가용 전용 5종이고, 개발용에는 평가용 전용 변형이 없다."""
    assert len(suite.KIND_SETS["heldout"]) == len(prior.KINDS) + 10
    assert not set(suite.KIND_SETS["dev"]) & set(suite.HELDOUT_KINDS)
    assert set(suite.FRESH_DEV_KINDS) <= set(suite.KIND_SETS["dev"])
    assert set(suite.FRESH_HELDOUT_KINDS) <= set(suite.KIND_SETS["heldout"]) - set(suite.KIND_SETS["dev"])


def test_fresh_noise():
    """새 변형은 값 자리(지명·시각)를 건드리지 않고, 덧붙이는 말에는 시간·주제 같은 조건 낱말이 없다."""
    text, spans = suite.core.fill_template("<P1>에서 퇴근 버스 어때?", {"P1": pick_area("누웨마루거리")})
    for kind in suite.FRESH_DEV_KINDS + suite.FRESH_HELDOUT_KINDS:
        for number in range(60):
            noisy = suite.apply_noise(text, spans, kind, number)
            if noisy is None:
                continue
            assert "누웨마루거리" in noisy, (kind, noisy)
            assert noisy != text, (kind, noisy)
    assert suite.apply_noise(text, spans, "particle_drop", 3) == "누웨마루거리 퇴근 버스 어때?"
    attached, attached_spans = suite.core.fill_template("<P1>이야?", {"P1": pick_area("노형동")})
    assert suite.apply_noise(attached, attached_spans, "particle_drop", 1) is None  # '노형동이야'의 '이'는 조사가 아니다
    assert suite.split_unit("오후 8시쯤", 0) == "오후 8 시쯤"
    assert suite.make_typo("버스", [], 0, suite.convert_initial) == "뻐스"
    for phrase in suite.CALL_PREFIXES + suite.ASIDE_NOTES + suite.REPLY_HEADS:
        assert not prior.find_concepts(phrase), phrase


def test_fresh_contexts():
    """새 앞 질문 표현은 원래 표현과 같은 조건(퇴근 시간대·버스 주제)만 담고, 절반은 평가용 전용이다."""
    expected = prior.find_concepts("<P1> 퇴근 버스 어때?".replace("<P1>", prior.NEUTRAL))
    for form in suite.FRESH_CONTEXTS:
        assert prior.find_concepts(form.replace("<P1>", prior.NEUTRAL)) == expected, form
    fresh = [v for v in POOLS["ctx"] if v.get("fresh")]
    assert len(fresh) == len(suite.FRESH_CONTEXTS) and sum(v["heldout_only"] for v in fresh) == len(fresh) // 2


def test_fresh_values():
    """새 어휘는 같은 변환 규칙을 통과한 값만 들어가고, 개발용 조합에는 평가용 전용 새 값이 없다."""
    vocab, dropped = suite.convert_fresh()
    for category, values in vocab.items():
        for value in values:
            assert value["fresh"] and value["key"].startswith(f"{category}:")
    template = {"key": "dev:base:G08:00000", "frame": "G08", "family": "exclusion", "workspace": "life",
                "style": "canonical", "source": "base", "text": FRAMES["G08"]["text"], "context": None}
    dims = suite.plan_dimensions(template, FRAMES["G08"], POOLS, "dev")
    for _, options in dims:
        for choice in options:
            assert not any(value["heldout_only"] for value in choice)


def test_case_marks():
    """평가용 문항은 새 요소가 들면 fresh 를 표시하고, 개발용 문항은 평가용 전용 표시가 하나도 없다."""
    template = {"key": "dev:base:G08:00000", "frame": "G08", "family": "exclusion", "workspace": "life",
                "style": "canonical", "source": "base", "text": FRAMES["G08"]["text"], "context": None}
    dims = suite.plan_dimensions(template, FRAMES["G08"], POOLS, "dev")
    space = suite.count_space(dims, "dev")
    cases = [suite.build_case(template, FRAMES["G08"], dims, (i * 7919) % space, "dev") for i in range(300)]
    assert cases == [suite.build_case(template, FRAMES["G08"], dims, (i * 7919) % space, "dev") for i in range(300)]
    assert all(not case["unseen"] and case["kind"] in suite.KIND_SETS["dev"] for case in cases)
    held = dict(template, key="heldout:qwen3:G08:00000", source="qwen3")
    dims = suite.plan_dimensions(held, FRAMES["G08"], POOLS, "heldout")
    space = suite.count_space(dims, "heldout")
    marked = [suite.build_case(held, FRAMES["G08"], dims, (i * 104729) % space, "heldout") for i in range(400)]
    for case in marked:
        fresh = [part for part in case["novel_parts"] if part.startswith("fresh_")]
        assert ("fresh" in case["novel_parts"]) == bool(fresh)
        assert ("fresh_kind" in fresh) == (case["kind"] in suite.FRESH_HELDOUT_KINDS)
    assert any("fresh_kind" in case["novel_parts"] for case in marked)
