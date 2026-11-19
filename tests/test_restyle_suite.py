"""6천만 문항 새 말투 시험(scripts/restyle_suite.py)의 말투·상황 분리, 새 입력 변형, 문장틀 거르기, 평가용 전용 표시를 확인한다."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import diverse_suite as earlier  # noqa: E402
import mega_suite  # noqa: E402
import restyle_suite as suite  # noqa: E402

POOLS = suite.build_pools()
FRAMES = {frame["id"]: frame for frame in suite.load_frames()}


def pick_area(surface: str) -> dict:
    return next(value for value in POOLS["area"] if value["surface"] == surface)


def test_style_split():
    """새 말투 24종은 이전 36종과 겹치지 않고, 개발용 말투·상황에는 평가용 전용 요소가 없다."""
    assert len(suite.STYLES) == 24 and len(suite.HELDOUT_STYLES) == 12 and len(suite.DEV_STYLES) == 12
    assert not set(suite.STYLES) & set(earlier.STYLES)
    for number in range(3000):
        assert not suite.check_situation(suite.pick_situation(number, heldout=False))
    assert any(suite.check_situation(suite.pick_situation(number, heldout=True)) for number in range(200))


def test_kind_sets():
    """새 변형은 개발용 5종·평가용 전용 5종이고, 개발용에는 이전·이번 평가용 전용 변형이 하나도 없다."""
    assert len(suite.KIND_SETS["heldout"]) == len(mega_suite.KINDS) + 10
    assert not set(suite.KIND_SETS["dev"]) & set(suite.HELDOUT_KINDS)
    assert set(suite.NEW_DEV_KINDS) <= set(suite.KIND_SETS["dev"])
    assert set(suite.NEW_HELDOUT_KINDS) <= set(suite.KIND_SETS["heldout"]) - set(suite.KIND_SETS["dev"])


def test_new_noise():
    """새 변형은 정해진 대로 바꾸고, 지명 값 자리는 '지명 한영 전환'만 건드린다."""
    assert suite.convert_layout("버스") == "qjtm" and suite.convert_layout("노형동") == "shgudehd"
    assert suite.convert_layout("괜찮아?") == "rhoscksgdk?"
    assert [suite.apply_pairs(w, suite.compute_link) for w in ("많이", "있어", "없어", "괜찮아", "좋아", "같이")] == \
        ["마니", "이써", "업써", "괜차나", "조아", None]
    assert [suite.apply_pairs(w, suite.compute_assimilation) for w in ("합니다", "같이", "좋고", "축하", "학교")] == \
        ["함니다", "가치", "조코", "추카", "학꾜"]
    assert suite.convert_numbers("저녁 8시 30분, 21시") == "저녁 여덟 시 삼십 분, 이십일 시"
    assert suite.convert_numbers("1시간 걸려") is None and suite.convert_numbers("9pm 이후") is None
    text, spans = suite.core.fill_template("<P1>에서 퇴근 버스 타려는데 정류소 사람 많아?", {"P1": pick_area("노형동")})
    for kind in suite.NEW_DEV_KINDS + suite.NEW_HELDOUT_KINDS:
        for number in range(40):
            noisy = suite.apply_noise(text, spans, kind, number)
            if noisy is None:
                continue
            assert noisy != text, (kind, noisy)
            assert ("노형동" in noisy) == (kind != "layout_place"), (kind, noisy)
    assert suite.apply_noise(text, spans, "layout_place", 0).startswith("shgudehd에서")
    assert suite.strip_tail("노형동 버스 어때?", []) == "노형동 버스 어ㄸ?"
    for word in suite.FILLERS:
        assert not earlier.find_concepts(word), word


def test_templates_filtered():
    """새 문장틀은 형식·뜻 검사를 통과하고, 이전 시험에 나온 글과 (평가용은) 이번 개발용 글을 쓰지 않는다."""
    earlier_texts = suite.load_earlier()
    dev, _ = suite.load_templates("dev")
    held, _ = suite.load_templates("heldout")
    dev_texts = {(t["frame"], t["text"]) for t in dev}
    for template in dev + held:
        frame = FRAMES[template["frame"]]
        assert earlier.check_format(frame, template["text"]) is None
        assert earlier.check_meaning(frame, template["text"]) is None
        assert (template["frame"], template["text"]) not in earlier_texts
    assert all(t["style"] in suite.DEV_STYLES and not suite.check_situation(t["situation"]) for t in dev)
    assert not {(t["frame"], t["text"]) for t in held} & dev_texts


def test_case_marks():
    """개발용 문항에는 평가용 전용 표시가 없고, 평가용 문항은 새 요소가 들면 new_any 를 표시한다. 같은 번호는 같은 문항."""
    template = {"key": "dev:gemma4:G08:00000", "frame": "G08", "family": "exclusion", "workspace": "life",
                "style": "memo", "source": "gemma4", "text": FRAMES["G08"]["text"], "context": None,
                "situation": suite.pick_situation(1, heldout=False)}
    dims = suite.plan_dimensions(template, FRAMES["G08"], POOLS, "dev")
    space = suite.count_space(dims, "dev")
    cases = [suite.build_case(template, FRAMES["G08"], dims, (i * 7919) % space, "dev") for i in range(400)]
    assert cases == [suite.build_case(template, FRAMES["G08"], dims, (i * 7919) % space, "dev") for i in range(400)]
    assert all(not case["unseen"] and case["kind"] in suite.KIND_SETS["dev"] for case in cases)
    held = dict(template, key="heldout:qwen3:G08:00000", source="qwen3", style="letter")
    dims = suite.plan_dimensions(held, FRAMES["G08"], POOLS, "heldout")
    space = suite.count_space(dims, "heldout")
    cases = [suite.build_case(held, FRAMES["G08"], dims, (i * 7919) % space, "heldout") for i in range(400)]
    assert all("new_style" in case["novel_parts"] and "new_any" in case["novel_parts"] for case in cases)
    assert any(case["kind"] in suite.NEW_HELDOUT_KINDS and "new_kind" in case["novel_parts"] for case in cases)
