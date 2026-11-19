"""1억 문항 확장 시험(mega)의 정의: 다양성 시험(diverse_suite)의 문장틀·값·입력 변형 위에 새 요소를 더한다.

데스크탑 GPU 를 쓸 수 없어 새 LLM 문장틀은 만들지 않았다(문장틀은 다양성 시험과 같다). 새로 더한 것:
- 새 값: data/meta/mega_vocab.json(범주 10개). 범주 둘씩 맡은 Claude 에이전트가 제안하고 다른 에이전트가 독립 검토했다
  (source_type synthetic_seed). 정답 슬롯은 diverse_suite.convert_item 이 같은 규칙으로 계산·교차 확인한다. 절반은 평가용 전용.
- 새 입력 변형 10종. 개발용 5종: 모음 혼동 오타(ㅐ/ㅔ 등)·지명 뒤 조사 빼기·호출 말머리('@제주봇')·낱말 되풀이·
  숫자와 단위 사이 띄어쓰기. 평가용 전용 5종: 받침 오타·된소리 오타('뻐스')·해시태그·괄호 덧말·답장 머리말.
- 새 앞 질문 표현 8가지(후속 질문 문맥, 절반 평가용 전용).
평가용에만 넣은 이 새 요소(fresh)가 든 문항이 이 시험의 주 지표다. 다양성 시험의 평가용 요소는 그 시험의 공식 채점 뒤
오답을 살펴봤으므로, 보고서는 새 요소가 든 문항과 나머지를 나눠 적는다.
"""
from __future__ import annotations

import json

import diverse_suite as prior
from diverse_suite import (SCALE_SEED, SET_AWARE, SET_SOURCES, SLOTS, check_attached, compute_hash,  # noqa: F401
                           load_frames, load_templates, pick_stride, plan_dimensions)

core = prior.base  # scale_suite(값 끼우기·자모 분해·자판 오타 같은 공통 부품)
META = prior.META
FRESH_PATH = META / "mega_vocab.json"
FRESH_DEV_KINDS = ("vowel_swap", "particle_drop", "call_prefix", "word_repeat", "digit_space")
FRESH_HELDOUT_KINDS = ("batchim_typo", "tense_typo", "hashtag", "aside_note", "reply_quote")
HELDOUT_KINDS = tuple(prior.HELDOUT_KINDS) + FRESH_HELDOUT_KINDS
KINDS = list(prior.KINDS) + list(FRESH_DEV_KINDS) + list(FRESH_HELDOUT_KINDS)
KIND_SETS = {"dev": [k for k in KINDS if k not in HELDOUT_KINDS], "heldout": list(KINDS)}
STRESS_KINDS = tuple(prior.STRESS_KINDS) + ("vowel_swap", "batchim_typo", "tense_typo")
VOWEL_SWAPS = {"ㅐ": "ㅔ", "ㅔ": "ㅐ", "ㅒ": "ㅖ", "ㅖ": "ㅒ", "ㅚ": "ㅙ", "ㅙ": "ㅞ", "ㅞ": "ㅚ", "ㅢ": "ㅣ"}
TENSE_PAIRS = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ"}
ADDED_FINALS = ("ㄴ", "ㄹ", "ㅁ", "ㅇ")
PARTICLES = ("에서는", "에서", "이랑", "랑", "으로", "로", "과", "와", "은", "는", "이", "가", "을", "를", "의", "에")
UNIT_CHARS = "시분%월살세"
CALL_PREFIXES = ("@제주봇 ", "봇아, ", "[질문] ", "(질문) ", "Q. ", "챗봇님, ", "AI야 ", "비서야, ", "도우미님 ")
ASIDE_NOTES = (" (급함)", " (진지하게)", " (궁금해서요)", " (질문이에요)", " (꼭 알려 주세요)", " (혹시 해서)", " (부탁해요)")
REPLY_HEADS = ("RE: ", "Re: ", "[답장] ", "↳ ", "> ", "└ ")
FRESH_CONTEXTS = ("<P1>에서 퇴근 버스 타려는데 어때?", "퇴근길 <P1> 버스 어떤 편이야?", "<P1>에서 퇴근할 때 버스 이용 괜찮을까?",
                  "<P1> 퇴근하는 시간에 버스 어때요?", "<P1> 퇴근 시간 버스 타기 수월해?", "<P1>에서 퇴근 버스 잡기 어때?",
                  "<P1> 쪽 퇴근 시간대 버스는 어때?", "<P1> 퇴근 버스 사정 어때요?")


# ---------------------------------------------------------------- 새 값(어휘)·앞 질문 표현

def load_fresh() -> dict:
    if not FRESH_PATH.exists():
        return {}
    return json.loads(FRESH_PATH.read_text(encoding="utf-8")).get("categories", {})


def convert_fresh() -> tuple[dict[str, list[dict]], dict[str, dict[str, int]]]:
    """새 어휘 -> 범주별 값 묶음(fresh 표시, 절반 평가용 전용)과 코드 교차 확인 탈락 통계."""
    pools: dict[str, list[dict]] = {name: [] for name in prior.VOCAB_ALLOWED}
    dropped: dict[str, dict[str, int]] = {}
    seen = set()
    for category, items in load_fresh().items():
        if category not in prior.VOCAB_ALLOWED:
            continue
        for item in items:
            value, reason = prior.convert_item(category, item)
            key = (category, value["surface"].replace(" ", "")) if value else None
            if value is None or key in seen:
                reason = reason or "duplicate"
                dropped.setdefault(category, {})[reason] = dropped.setdefault(category, {}).get(reason, 0) + 1
                continue
            seen.add(key)
            value["key"] = f"{category}:{value['surface']}"
            value["fresh"] = True
            value["heldout_only"] = compute_hash("mega:" + value["key"]) % 2 == 0
            pools[category].append(value)
    return pools, dropped


def add_fresh(pools: dict, vocab: dict) -> None:
    """새 값을 다양성 시험의 값 묶음 규칙(diverse_suite.build_pools)과 같은 방식으로 각 묶음에 덧붙인다."""
    extend = prior.extend_pool
    extend(pools, "time_point", vocab["time_point"])
    extend(pools, "after", vocab["after"])
    extend(pools, "weather", vocab["weather"])
    extend(pools, "cond", vocab["weather"] + vocab["day"])
    extend(pools, "day", vocab["day"])
    extend(pools, "day2", vocab["day"])
    extend(pools, "old", [v for v in vocab["group"] if v["labels"]["user_group"] == "어르신"])
    extend(pools, "young", [v for v in vocab["group"] if v["labels"]["user_group"] == "청소년"])
    extend(pools, "group", vocab["group"])
    extend(pools, "concept", vocab["concept"])
    evening = [v for v in vocab["time_point"] if (v["labels"]["hour_from"], v["labels"]["hour_to"]) == (18, 21)]
    evening += [v for v in vocab["after"] if 18 <= v["labels"]["hour_from"] <= 20]
    extend(pools, "evening", [v for v in evening if ("topic", "night") not in prior.find_concepts(v["surface"])])
    extend(pools, "morning", [v for v in vocab["time_point"]
                              if v["labels"].get("hour_from") is not None and 5 <= v["labels"]["hour_from"] <= 9
                              and v["labels"]["hour_to"] <= 10])
    extend(pools, "nightword", [v for v in vocab["time_point"]
                                if {c for kind, c in prior.find_concepts(v["surface"]) if kind == "topic"} == {"night"}
                                and (v["labels"]["hour_from"] >= 21 or v["labels"]["hour_to"] <= 5)])
    extend(pools, "source", vocab["source"])
    extend(pools, "topicword", vocab["topicword"])
    extend(pools, "range", vocab["range"])
    extend(pools, "form", vocab["place_form"])


def build_pools() -> dict[str, list[dict]]:
    pools = prior.build_pools()
    add_fresh(pools, convert_fresh()[0])
    for number, form in enumerate(FRESH_CONTEXTS):
        value = prior.make_value("ctx", "", heldout=number % 2 == 1)
        value.update(key=f"ctx:{form}", text=form, fresh=True)
        pools["ctx"].append(value)
    return pools


# ---------------------------------------------------------------- 새 입력 변형

def make_typo(text: str, spans: list, pick: int, change) -> str | None:
    """값 자리 밖(틀 본문) 음절 하나를 change((초성, 중성, 종성) -> 새 세 쌍 또는 None)로 바꾼다."""
    covered = {i for s, e, _ in spans for i in range(s, e)}
    options = []
    for offset, char in enumerate(text):
        parts = core.split_syllable(char)
        if parts is None or offset in covered:
            continue
        changed = change(parts, pick)
        if changed is not None and changed != parts:
            options.append((offset, changed))
    if not options:
        return None
    offset, (cho, jung, jong) = options[pick % len(options)]
    return text[:offset] + core.compose_syllable(cho, jung, jong) + text[offset + 1:]


def convert_vowel(parts: tuple, pick: int) -> tuple | None:
    vowel = core.JUNG[parts[1]]
    return (parts[0], core.JUNG.index(VOWEL_SWAPS[vowel]), parts[2]) if vowel in VOWEL_SWAPS else None


def convert_final(parts: tuple, pick: int) -> tuple:
    if parts[2]:
        return parts[0], parts[1], 0
    return parts[0], parts[1], core.JONG.index(ADDED_FINALS[(pick >> 8) % len(ADDED_FINALS)])


def convert_initial(parts: tuple, pick: int) -> tuple | None:
    initial = core.CHO[parts[0]]
    return (core.CHO.index(TENSE_PAIRS[initial]), parts[1], parts[2]) if initial in TENSE_PAIRS else None


def strip_particle(text: str, spans: list, pick: int) -> str | None:
    """지명 바로 뒤에 붙어 낱말을 끝내는 조사 하나를 뺀다('노형동에서 퇴근' -> '노형동 퇴근', '노형동이야'는 그대로)."""
    options = []
    for _, end, name in spans:
        if name not in prior.PLACE_SLOTS:
            continue
        for particle in PARTICLES:
            after = end + len(particle)
            if text.startswith(particle, end) and (after == len(text) or text[after] in " ?!.,~"):
                options.append((end, after))
                break
    if not options:
        return None
    end, after = options[pick % len(options)]
    return text[:end] + text[after:]


def find_tokens(text: str, spans: list) -> list[tuple[int, int]]:
    """공백으로 나눈 낱말 중 값 자리와 겹치지 않는 낱말의 (시작, 끝)."""
    covered = {i for s, e, _ in spans for i in range(s, e)}
    tokens, start = [], None
    for offset, char in enumerate(text + " "):
        if char == " ":
            if start is not None and not any(i in covered for i in range(start, offset)):
                tokens.append((start, offset))
            start = None
        elif start is None:
            start = offset
    return tokens


def stack_word(text: str, spans: list, pick: int) -> str | None:
    tokens = [t for t in find_tokens(text, spans) if any(core.split_syllable(c) for c in text[t[0]:t[1]])]
    if not tokens:
        return None
    start, end = tokens[pick % len(tokens)]
    word = text[start:end].rstrip("?!.,~")
    return text[:start] + word + " " + text[start:]


def split_unit(text: str, pick: int) -> str | None:
    """숫자와 단위(시·분·%·월·살·세) 사이에 띄어쓰기를 넣는다('8시' -> '8 시'). 값 자리 안도 바꾼다(뜻은 그대로)."""
    places = [i for i in range(len(text) - 1) if text[i].isdigit() and text[i + 1] in UNIT_CHARS]
    if not places:
        return None
    cut = places[pick % len(places)] + 1
    return text[:cut] + " " + text[cut:]


def add_hashtag(text: str, pick: int) -> str:
    """낱말 머리 한두 곳에 '#'을 붙인다('#노형동 퇴근 버스 어때?')."""
    heads = [0] + [i + 1 for i, c in enumerate(text[:-1]) if c == " " and text[i + 1] != " "]
    chosen = {heads[pick % len(heads)]}
    if (pick >> 5) % 2 and len(heads) > 1:
        chosen.add(heads[(pick >> 9) % len(heads)])
    return "".join(("#" if i in chosen else "") + c for i, c in enumerate(text))


def add_aside(text: str, spans: list, pick: int) -> str | None:
    covered = {i for s, e, _ in spans for i in range(s, e)}
    gaps = [i for i, c in enumerate(text) if c == " " and i not in covered]
    if not gaps:
        return None
    cut = gaps[pick % len(gaps)]
    return text[:cut] + ASIDE_NOTES[(pick >> 8) % len(ASIDE_NOTES)] + text[cut:]


def apply_noise(text: str, spans: list, kind: str, number: int) -> str | None:
    """입력 변형 하나를 적용한다. 다양성 시험의 21종은 그대로, 새 10종은 여기서 만든다. 적용할 수 없으면 None."""
    if kind not in FRESH_DEV_KINDS and kind not in FRESH_HELDOUT_KINDS:
        return prior.apply_noise(text, spans, kind, number)
    pick = core.compute_mix(number)
    if kind == "vowel_swap":
        return make_typo(text, spans, pick, convert_vowel)
    if kind == "batchim_typo":
        return make_typo(text, spans, pick, convert_final)
    if kind == "tense_typo":
        return make_typo(text, spans, pick, convert_initial)
    if kind == "particle_drop":
        return strip_particle(text, spans, pick)
    if kind == "call_prefix":
        return CALL_PREFIXES[pick % len(CALL_PREFIXES)] + text
    if kind == "word_repeat":
        return stack_word(text, spans, pick)
    if kind == "digit_space":
        return split_unit(text, pick)
    if kind == "hashtag":
        return add_hashtag(text, pick)
    if kind == "aside_note":
        return add_aside(text, spans, pick)
    if kind == "reply_quote":
        return REPLY_HEADS[pick % len(REPLY_HEADS)] + text
    raise ValueError(kind)


# ---------------------------------------------------------------- 조합 공간·문항 만들기

def count_space(dims: list, set_name: str) -> int:
    total = len(KIND_SETS[set_name])
    for _, options in dims:
        total *= len(options)
    return total


def build_case(template: dict, frame: dict, dims: list, index: int, set_name: str) -> dict:
    """조합 번호 하나 -> 시험 문항. 다양성 시험과 같고, 평가용 전용 새 요소(fresh_*)를 따로 표시한다."""
    kinds = KIND_SETS[set_name]
    values, kind_number = prior.decode_case(index, dims, len(kinds))
    form, ctx = values.pop("FORM", None), values.pop("CTX", None)
    shown, offsets = prior.apply_forms(template["text"], values, form)
    text, spans = core.fill_template(template["text"], shown)
    spans = [(s + offsets[n][0], s + offsets[n][0] + offsets[n][1], n) if n in offsets else (s, e, n) for s, e, n in spans]
    context_text = ctx["text"] if ctx is not None else template.get("context")
    context = core.fill_template(context_text, values)[0] if context_text else None
    noisy, kind = None, ""
    salt = compute_hash(template["key"])
    for step in range(len(kinds)):
        kind = kinds[(kind_number + step) % len(kinds)]
        noisy = apply_noise(text, spans, kind, index ^ salt)
        if noisy is not None and noisy.strip():
            break
    extras = [v for v in (form, ctx) if v is not None]
    marks = (
        ("value", any(v["heldout_only"] for v in values.values())), ("form", bool(form and form["heldout_only"])),
        ("context", bool(ctx and ctx["heldout_only"])), ("kind", kind in HELDOUT_KINDS),
        ("style", template["style"] in prior.HELDOUT_STYLES), ("situation", prior.check_situation(template.get("situation"))),
        ("fresh_value", any(v.get("fresh") and v["heldout_only"] for v in values.values())),
        ("fresh_form", bool(form and form.get("fresh") and form["heldout_only"])),
        ("fresh_context", bool(ctx and ctx.get("fresh") and ctx["heldout_only"])),
        ("fresh_kind", kind in FRESH_HELDOUT_KINDS),
    )
    novel = [part for part, flag in marks if flag]
    if any(part.startswith("fresh_") for part in novel):
        novel.append("fresh")
    return {"text": noisy, "context": context, "kind": kind, "expected": prior.compute_expected(frame, values),
            "values": [v["key"] for v in list(values.values()) + extras], "unseen": bool(novel), "novel_parts": novel}
