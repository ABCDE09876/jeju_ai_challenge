"""6천만 문항 새 말투 시험(restyle)의 정의: 데스크탑 GPU 로컬 LLM 이 새로 쓴 문장틀 x 1억 시험의 값 x 새 입력 변형.

1억 시험(mega_suite)은 데스크탑을 쓸 수 없어 7.11절 문장틀을 다시 썼다. 이번 시험은 데스크탑 GPU 로 문장틀을 새로 만든다.
- 질문 유형: 다양성 시험과 같은 57종(정답 규약이 검증된 틀). 문장틀 글은 모두 이번에 새로 생성했다.
- 새 말투 24종(개발용 12 + 평가용 전용 12, 기존 36종과 겹치지 않음)과 새 상황(화자·경로·목적·감정, 기존과 다른 요소).
  개발용 틀은 gemma4:12b 가 개발용 말투 12종으로, 평가용 틀은 qwen3:14b 가 24종 전부로 쓴다(모델 분리는 이전 시험과 같다).
- 값: 1억 시험과 같은 값 묶음(1차·다양성·1억 어휘, 이전 시험의 개발용/평가용 구분 그대로).
- 입력 변형: 1억 시험 31종 + 새 10종. 개발용 5종: 한영 전환 오류(지명 밖 낱말 하나)·소리 나는 대로 쓰기(연음)·
  숫자를 한글 수사로·말버릇 끼우기·띄어쓰기 한 칸 밀림. 평가용 전용 5종: 지명 한영 전환 오류·소리 나는 대로 쓰기
  (된소리·콧소리·거센소리)·끝 글자 덜 친 채 보내기·얼버무리는 말 끼우기·낱말을 이모지로 바꾸기.
- 라벨 검증: LLM 이 쓴 틀은 다양성 시험과 같은 형식 검사·뜻 검사(diverse_suite.check_meaning, 해석기와 무관한 독립 어휘)를
  통과해야 쓴다.
평가용 틀은 모두 이번에 새로 만든 글이라 평가용 전체가 주 지표이고, 평가용 전용 말투·상황·새 변형이 든 문항을 따로 센다.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

import mega_suite as prior

diverse = prior.prior  # diverse_suite(문장틀 57종·뜻 검사·값 끼우기)
core = prior.core  # scale_suite(자모 분해·자판 오타 같은 공통 부품)
META = prior.META
SCALE_SEED = prior.SCALE_SEED
SLOTS = prior.SLOTS
SET_AWARE = True
compute_hash = prior.compute_hash
pick_stride = prior.pick_stride
check_attached = prior.check_attached
PLACEHOLDER_NOTES = diverse.PLACEHOLDER_NOTES
load_frames = diverse.load_frames

# ---------------------------------------------------------------- 새 말투 24종(개발용 12 + 평가용 전용 12)

STYLES = {
    "eumseum": "인터넷 커뮤니티의 음슴체('~함?', '~임?', '~어떰?')로 짧게 묻는 말투(뜻이 모호한 줄임말은 쓰지 않음)",
    "haoche": "하오체·하게체 같은 옛날식 말투('~하오?', '~한가?', '~이오?')",
    "translationese": "영어 문장을 옮긴 듯한 번역투('~에 대한 정보를 제공해 줄 수 있습니까?', '~하는 것이 가능합니까?')",
    "memo": "메모처럼 낱말을 쉼표나 빗금(/)으로 끊어 적은 짧은 질문",
    "voicecall": "음성 비서를 부르는 말('헤이 봇', '비서야', '도우미야')로 시작해 말하듯 묻는 말투(부르는 말에 지명을 쓰지 않음)",
    "gangwon": "강원도 사투리 어미를 섞은 말투(예: ~래요?, ~드래요, ~나요?, ~우?)",
    "dananka": "군대식 다나까체('~입니까?', '~습니까?', '~지 말입니다')",
    "selftalk": "혼잣말처럼 중얼거리다가 묻는 말투('음... ~려나?', '~하나 궁금하네')",
    "plainspec": "조건을 하나씩 풀어서 밝히는 말투('장소는 ~이고, 시간은 ~인데' 식, 원문 조건만 씀)",
    "staffask": "담당자·선생님 같은 상대를 부르며 공손하게 묻는 말투('담당자님, ~ 여쭤봐도 될까요?')",
    "mixspeech": "존댓말과 반말이 한 문장 안에서 섞인 말투",
    "qna": "질문 게시판 글처럼 '~ 아시는 분?', '~ 답변 부탁드립니다'로 묻는 말투",
    "jejudeep": "제주 어르신처럼 제주어를 짙게 쓰는 말투(예: ~우꽈?, ~수가?, ~ㄴ디, 하영=많이, 가민=가면, 어떵=어떻게)",
    "archaic": "사극처럼 예스러운 말투('~하느냐?', '~하오리까?', '~인고?')",
    "radio": "라디오 사연처럼 진행자를 부르며 사연과 함께 묻는 말투(사연에 새 장소·시간·조건·사람을 넣지 않음)",
    "interview": "인터뷰 질문처럼 '~에 대해 어떻게 보시나요?', '~는 어떤 편인가요?'로 묻는 말투",
    "fragment": "문장을 마침표로 두세 조각 끊어 띄엄띄엄 쓴 말투",
    "humble": "몹시 겸손하게 낮추는 극존칭('송구하오나 ~ 여쭙습니다', '~하신지 여쭙고자 합니다')",
    "lecture": "설명을 곁들이며 묻는 강의식 말투(설명에 새 조건·숫자를 넣지 않음)",
    "machinetrans": "기계 번역기를 거친 듯 어순과 조사가 어색한 직역 말투",
    "letter": "편지처럼 받는 사람을 부르고 인사한 뒤 묻는 말투",
    "tweet": "짧은 감탄 한마디를 붙인 소셜 미디어 한 줄 글 말투",
    "bilingual": "짧은 영어 한마디와 한국어 질문을 섞은 말투(자리표시자와 숫자는 그대로, 묻는 내용은 한국어)",
    "consult": "전문가에게 상담을 청하듯 '~ 상담 부탁드립니다', '~ 조언 구합니다'로 묻는 말투",
}
HELDOUT_STYLES = ("jejudeep", "archaic", "radio", "interview", "fragment", "humble", "lecture", "machinetrans", "letter",
                  "tweet", "bilingual", "consult")
DEV_STYLES = tuple(s for s in STYLES if s not in HELDOUT_STYLES)

# ---------------------------------------------------------------- 새 상황(화자 x 경로 x 목적 x 감정) - 말투·표현에만 반영
SPEAKERS = ("배달 라이더", "카페 사장", "대학 신입생", "출장 온 회사원", "재택근무자", "야간 근무자", "귀촌한 부부",
            "도서관 사서", "주말 부부", "관광 가이드", "버스 동호회 회원", "취업 준비생")
CHANNELS = ("앱 리뷰 댓글", "지도 앱 문의", "휴대폰 문자", "회의 메모", "설문 응답란", "채팅 상담", "라디오 사연", "손편지")
PURPOSES = ("이사 갈 동네를 고르려고", "동료에게 공유하려고", "출퇴근 계획을 세우려고", "여행 일정을 짜려고",
            "모임 장소를 정하려고", "과제를 하려고", "기사 내용을 확인하려고", "주변 사람과 내기를 해서", "글 소재를 찾으려고")
EMOTIONS = ("체념", "조바심", "신중함", "들뜸", "의아함", "다급함", "서운함", "느긋함")
HELDOUT_SITUATION = {"speaker": ("관광 가이드", "버스 동호회 회원", "취업 준비생"), "channel": ("라디오 사연", "손편지"),
                     "purpose": ("주변 사람과 내기를 해서", "글 소재를 찾으려고"), "emotion": ("서운함", "느긋함")}


def pick_situation(number: int, heldout: bool) -> dict:
    """호출 번호로 상황 요소 네 가지를 고른다(개발용은 평가용 전용 요소를 빼고 고른다). 다양성 시험과 다른 소금값을 쓴다."""
    picked = {}
    for offset, (name, options) in enumerate((("speaker", SPEAKERS), ("channel", CHANNELS), ("purpose", PURPOSES),
                                              ("emotion", EMOTIONS))):
        allowed = [o for o in options if heldout or o not in HELDOUT_SITUATION[name]]
        picked[name] = allowed[core.compute_mix(60_000_000 + number * 7 + offset) % len(allowed)]
    return picked


def describe_situation(situation: dict) -> str:
    return (f"{situation['speaker']}이(가) {situation['channel']}에서 {situation['purpose']} "
            f"'{situation['emotion']}' 마음으로 묻는다")


def check_situation(situation: dict | None) -> bool:
    """상황에 평가용 전용 요소가 들어 있으면 True."""
    return bool(situation) and any(situation.get(name) in options for name, options in HELDOUT_SITUATION.items())


# ---------------------------------------------------------------- 문장틀 불러오기(형식 + 뜻 검사 + 이전에 나온 글 빼기)

MODEL_FILES = {"gemma4": META / "restyle_frames_gemma4.jsonl", "qwen3": META / "restyle_frames_qwen3.jsonl"}
SET_SOURCES = {"dev": ("gemma4",), "heldout": ("qwen3",)}
EARLIER_FILES = ("scale_frames_gemma4.jsonl", "scale_frames_qwen3.jsonl", "diverse_frames_gemma4.jsonl",
                 "diverse_frames_qwen3.jsonl")


def read_rows(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def load_earlier() -> frozenset:
    """이전 시험(2,550만·7,100만·1억)에 이미 나온 (질문 유형, 문장틀 글) - 새 시험에서는 뺀다."""
    seen = {(frame["id"], normalize_text(frame["text"])) for frame in load_frames()}
    for name in EARLIER_FILES:
        for row in read_rows(META / name):
            seen.update((row["frame_id"], normalize_text(text)) for text in row["texts"])
    return frozenset(seen)


@lru_cache(maxsize=1)
def load_dev_texts() -> frozenset:
    """이번 개발용(gemma4) 틀 글 - 평가용 틀이 같은 글이면 개발용에서 본 글이라 뺀다."""
    return frozenset((row["frame_id"], normalize_text(text))
                     for row in read_rows(MODEL_FILES["gemma4"]) for text in row["texts"])


def load_templates(set_name: str) -> tuple[list[dict], dict]:
    """세트의 문장틀 목록과 탈락 통계. 다양성 시험과 같은 형식·뜻 검사를 통과하고, 이전 시험에 나온 글이 아니며,
    평가용은 이번 개발용 틀과도 다른 글만 쓴다."""
    frames = {frame["id"]: frame for frame in load_frames()}
    earlier = load_earlier()
    dev_texts = load_dev_texts() if set_name == "heldout" else frozenset()
    templates, dropped, seen, numbers = [], {}, set(), {}
    for source in SET_SOURCES[set_name]:
        for row in read_rows(MODEL_FILES[source]):
            frame = frames[row["frame_id"]]
            for text in row["texts"]:
                normalized = normalize_text(text)
                reason = diverse.check_format(frame, normalized)
                if reason is None:
                    reason = diverse.check_meaning(frame, normalized)
                    reason = f"meaning:{reason}" if reason else None
                if reason is None and (frame["id"], normalized) in seen:
                    reason = "duplicate"
                if reason is None and (frame["id"], normalized) in earlier:
                    reason = "seen_earlier"
                if reason is None and (frame["id"], normalized) in dev_texts:
                    reason = "seen_dev"
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


# ---------------------------------------------------------------- 값 묶음·정답·조합 차원(1억 시험과 같다)

build_pools = prior.build_pools
compute_expected = diverse.compute_expected
plan_dimensions = diverse.plan_dimensions
decode_case = diverse.decode_case
apply_forms = diverse.apply_forms

# ---------------------------------------------------------------- 새 입력 변형 10종

NEW_DEV_KINDS = ("layout_word", "sound_link", "korean_number", "filler_word", "space_shift")
NEW_HELDOUT_KINDS = ("layout_place", "sound_assim", "jamo_tail", "hedge_word", "emoji_word")
HELDOUT_KINDS = tuple(prior.HELDOUT_KINDS) + NEW_HELDOUT_KINDS
KINDS = list(prior.KINDS) + list(NEW_DEV_KINDS) + list(NEW_HELDOUT_KINDS)
KIND_SETS = {"dev": [k for k in KINDS if k not in HELDOUT_KINDS], "heldout": list(KINDS)}
STRESS_KINDS = tuple(prior.STRESS_KINDS) + ("layout_word", "sound_link", "space_shift", "layout_place", "sound_assim",
                                            "jamo_tail", "emoji_word")
# 두벌식 자판: 한글 자모 -> 영문 글쇠(한영 전환을 잊고 친 글)
LAYOUT_KEYS = {"ㄱ": "r", "ㄲ": "R", "ㄴ": "s", "ㄷ": "e", "ㄸ": "E", "ㄹ": "f", "ㅁ": "a", "ㅂ": "q", "ㅃ": "Q", "ㅅ": "t",
               "ㅆ": "T", "ㅇ": "d", "ㅈ": "w", "ㅉ": "W", "ㅊ": "c", "ㅋ": "z", "ㅌ": "x", "ㅍ": "v", "ㅎ": "g",
               "ㅏ": "k", "ㅐ": "o", "ㅑ": "i", "ㅒ": "O", "ㅓ": "j", "ㅔ": "p", "ㅕ": "u", "ㅖ": "P", "ㅗ": "h",
               "ㅘ": "hk", "ㅙ": "ho", "ㅚ": "hl", "ㅛ": "y", "ㅜ": "n", "ㅝ": "nj", "ㅞ": "np", "ㅟ": "nl", "ㅠ": "b",
               "ㅡ": "m", "ㅢ": "ml", "ㅣ": "l", "ㄳ": "rt", "ㄵ": "sw", "ㄶ": "sg", "ㄺ": "fr", "ㄻ": "fa", "ㄼ": "fq",
               "ㄽ": "ft", "ㄾ": "fx", "ㄿ": "fv", "ㅀ": "fg", "ㅄ": "qt"}
# 연음: 받침 -> (남는 받침, 다음 'ㅇ' 자리로 옮겨 가는 소리). ㅎ 은 소리가 없어진다('좋아' -> '조아', '싫어' -> '시러').
LINKED_FINALS = {"ㄱ": ("", "ㄱ"), "ㄲ": ("", "ㄲ"), "ㄴ": ("", "ㄴ"), "ㄷ": ("", "ㄷ"), "ㄹ": ("", "ㄹ"), "ㅁ": ("", "ㅁ"),
                 "ㅂ": ("", "ㅂ"), "ㅅ": ("", "ㅅ"), "ㅆ": ("", "ㅆ"), "ㅈ": ("", "ㅈ"), "ㅊ": ("", "ㅊ"), "ㅋ": ("", "ㅋ"),
                 "ㅌ": ("", "ㅌ"), "ㅍ": ("", "ㅍ"), "ㅎ": ("", "ㅇ"), "ㄳ": ("ㄱ", "ㅆ"), "ㄵ": ("ㄴ", "ㅈ"),
                 "ㄶ": ("", "ㄴ"), "ㄺ": ("ㄹ", "ㄱ"), "ㄻ": ("ㄹ", "ㅁ"), "ㄼ": ("ㄹ", "ㅂ"), "ㄽ": ("ㄹ", "ㅆ"),
                 "ㄾ": ("ㄹ", "ㅌ"), "ㄿ": ("ㄹ", "ㅍ"), "ㅀ": ("", "ㄹ"), "ㅄ": ("ㅂ", "ㅆ")}
# 소리 나는 대로(평가용 전용): 구개음화·거센소리·콧소리·된소리
PALATAL = {"ㄷ": "ㅈ", "ㅌ": "ㅊ"}
ASPIRATED = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅈ": "ㅊ"}
H_FINALS = {"ㅎ": "", "ㄶ": "ㄴ", "ㅀ": "ㄹ"}
BEFORE_H = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅅ": "ㅌ", "ㅂ": "ㅍ", "ㅈ": "ㅊ"}
NASAL_FINALS = {"ㄱ": "ㅇ", "ㄲ": "ㅇ", "ㅋ": "ㅇ", "ㄳ": "ㅇ", "ㄺ": "ㅇ", "ㄷ": "ㄴ", "ㅅ": "ㄴ", "ㅆ": "ㄴ", "ㅈ": "ㄴ",
                "ㅊ": "ㄴ", "ㅌ": "ㄴ", "ㅂ": "ㅁ", "ㅍ": "ㅁ", "ㅄ": "ㅁ", "ㄿ": "ㅁ"}
TENSE = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ"}
STOP_FINALS = ("ㄱ", "ㄲ", "ㄳ", "ㄺ", "ㄷ", "ㅅ", "ㅆ", "ㅈ", "ㅊ", "ㅌ", "ㅂ", "ㅍ", "ㅄ")
NATIVE_HOURS = ("영", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉", "열", "열한", "열두")
SINO_DIGITS = ("", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구")
HOUR_NUMBER = re.compile(r"(?<![\d.:])(\d{1,2})\s*시(?!간)")
MINUTE_NUMBER = re.compile(r"(?<![\d.:])(\d{1,3})\s*분")
FILLERS = ("음", "어", "저기", "뭐랄까", "그러니까", "아 참", "흠", "음음")
HEDGES = ("솔직히", "진짜로", "혹시나 해서", "개인적으로", "아무튼", "어쨌든", "사실은", "왠지")
EMOJI_WORDS = (("와이파이", "\U0001F4F6"), ("정류소", "\U0001F68F"), ("정류장", "\U0001F68F"), ("쓰레기", "\U0001F5D1"),
               ("버스", "\U0001F68C"), ("택시", "\U0001F695"), ("충전", "\U0001F50C"), ("학교", "\U0001F3EB"),
               ("사람", "\U0001F465"))


def convert_layout(word: str) -> str:
    """한글 음절을 두벌식 영문 글쇠로 바꾼다('버스' -> 'qjtm'). 한글이 아닌 글자는 그대로 둔다."""
    out = []
    for char in word:
        parts = core.split_syllable(char)
        if parts is None:
            out.append(char)
            continue
        cho, jung, jong = parts
        out.append(LAYOUT_KEYS[core.CHO[cho]] + LAYOUT_KEYS[core.JUNG[jung]] + (LAYOUT_KEYS[core.JONG[jong]] if jong else ""))
    return "".join(out)


def apply_pairs(word: str, change) -> str | None:
    """이웃한 두 음절 (앞, 뒤)마다 change(앞 세 쌍, 뒤 세 쌍) -> (새 앞, 새 뒤) 또는 None 을 적용한다. 바뀐 게 없으면 None."""
    chars, changed = list(word), False
    for offset in range(len(chars) - 1):
        first, second = core.split_syllable(chars[offset]), core.split_syllable(chars[offset + 1])
        if first is None or second is None:
            continue
        pair = change(first, second)
        if pair is None:
            continue
        chars[offset] = core.compose_syllable(*pair[0])
        chars[offset + 1] = core.compose_syllable(*pair[1])
        changed = True
    return "".join(chars) if changed else None


def compute_link(first: tuple, second: tuple) -> tuple | None:
    """연음: 받침 + 'ㅇ'으로 시작하는 음절 -> 받침을 뒤로 옮긴다('많이' -> '마니', '있어' -> '이써', '없어' -> '업써')."""
    final = core.JONG[first[2]]
    if not final or final == "ㅇ" or core.CHO[second[0]] != "ㅇ":
        return None
    if final in PALATAL and core.JUNG[second[1]] == "ㅣ":
        return None  # '같이'·'굳이'는 구개음화(평가용 전용 변형이 다룬다)
    keep, moved = LINKED_FINALS[final]
    return (first[0], first[1], core.JONG.index(keep)), (core.CHO.index(moved), second[1], second[2])


def compute_assimilation(first: tuple, second: tuple) -> tuple | None:
    """소리 나는 대로(평가용 전용): 구개음화('같이' -> '가치'), 거센소리('좋고' -> '조코', '축하' -> '추카'),
    콧소리('합니다' -> '함니다', '있나' -> '인나'), 된소리('학교' -> '학꾜')."""
    final, initial, vowel = core.JONG[first[2]], core.CHO[second[0]], core.JUNG[second[1]]
    if final in PALATAL and initial == "ㅇ" and vowel == "ㅣ":
        return (first[0], first[1], 0), (core.CHO.index(PALATAL[final]), second[1], second[2])
    if final in H_FINALS and initial in ASPIRATED:
        return ((first[0], first[1], core.JONG.index(H_FINALS[final])),
                (core.CHO.index(ASPIRATED[initial]), second[1], second[2]))
    if final in BEFORE_H and initial == "ㅎ":
        return (first[0], first[1], 0), (core.CHO.index(BEFORE_H[final]), second[1], second[2])
    if final in NASAL_FINALS and initial in ("ㄴ", "ㅁ"):
        return (first[0], first[1], core.JONG.index(NASAL_FINALS[final])), second
    if final in STOP_FINALS and initial in TENSE:
        return first, (core.CHO.index(TENSE[initial]), second[1], second[2])
    return None


def read_sino(number: int) -> str:
    """한자어 수사('25' -> '이십오', '10' -> '십', '0' -> '영')."""
    if number == 0:
        return "영"
    hundreds, rest = divmod(number, 100)
    tens, ones = divmod(rest, 10)
    words = ""
    if hundreds:
        words += ("" if hundreds == 1 else SINO_DIGITS[hundreds]) + "백"
    if tens:
        words += ("" if tens == 1 else SINO_DIGITS[tens]) + "십"
    return words + SINO_DIGITS[ones]


def convert_numbers(text: str) -> str | None:
    """숫자 시각·분을 한글 수사로 쓴다('8시' -> '여덟 시', '21시' -> '이십일 시', '20분' -> '이십 분'). 없으면 None.
    '08시'처럼 0을 붙인 24시 표기는 오전이라는 뜻이 '여덟 시'에서 사라지므로 바꾸지 않는다."""
    def convert_hour(match: re.Match) -> str:
        hour = int(match.group(1))
        if hour > 24 or (len(match.group(1)) == 2 and match.group(1).startswith("0")):
            return match.group(0)
        return f"{NATIVE_HOURS[hour] if hour <= 12 else read_sino(hour)} 시"

    out = HOUR_NUMBER.sub(convert_hour, text)
    out = MINUTE_NUMBER.sub(lambda match: f"{read_sino(int(match.group(1)))} 분", out)
    return out if out != text else None


def find_words(text: str, spans: list) -> list[tuple[int, int]]:
    """값 자리와 겹치지 않고 한글 음절이 든 낱말의 (시작, 끝)."""
    return [(s, e) for s, e in prior.find_tokens(text, spans) if any(core.split_syllable(c) for c in text[s:e])]


def apply_word_change(text: str, spans: list, pick: int, change) -> str | None:
    """값 자리 밖 낱말 하나를 change(낱말) -> 새 낱말 또는 None 으로 바꾼다(바꿀 수 있는 낱말 중 하나를 고름)."""
    options = []
    for start, end in find_words(text, spans):
        changed = change(text[start:end])
        if changed is not None and changed != text[start:end]:
            options.append((start, end, changed))
    if not options:
        return None
    start, end, changed = options[pick % len(options)]
    return text[:start] + changed + text[end:]


def add_word(text: str, spans: list, pick: int, words: tuple) -> str:
    """값 자리 밖 띄어쓰기 자리(또는 문장 맨 앞)에 낱말 하나를 끼운다."""
    covered = {i for s, e, _ in spans for i in range(s, e)}
    gaps = [0] + [i for i, c in enumerate(text) if c == " " and i not in covered]
    cut = gaps[pick % len(gaps)]
    word = words[(pick >> 8) % len(words)]
    return word + " " + text if cut == 0 else text[:cut] + " " + word + text[cut:]


def make_space_shift(text: str, spans: list, pick: int) -> str | None:
    """값 자리 밖 띄어쓰기 한 칸을 옆 글자로 민다('퇴근 버스' -> '퇴 근버스' 또는 '퇴근버 스')."""
    covered = {i for s, e, _ in spans for i in range(s, e)}
    options = []
    for gap, char in enumerate(text):
        if char != " " or gap < 2 or gap > len(text) - 3 or gap - 1 in covered or gap + 1 in covered:
            continue
        if text[gap - 2] != " " and core.split_syllable(text[gap - 1]):
            options.append(text[:gap - 1] + " " + text[gap - 1] + text[gap + 1:])  # 앞 낱말 끝 글자를 뒤로
        if text[gap + 2] != " " and core.split_syllable(text[gap + 1]):
            options.append(text[:gap] + text[gap + 1] + " " + text[gap + 2:])  # 뒤 낱말 첫 글자를 앞으로
    return options[pick % len(options)] if options else None


def strip_tail(text: str, spans: list) -> str | None:
    """마지막 한글 음절(값 자리 밖)을 덜 친 채 보낸다('어때?' -> '어ㄸ?'): 그 음절을 첫소리 자모만 남긴다."""
    covered = {i for s, e, _ in spans for i in range(s, e)}
    for offset in range(len(text) - 1, -1, -1):
        parts = core.split_syllable(text[offset])
        if parts is None:
            continue
        if offset in covered:
            return None
        return text[:offset] + core.CHO[parts[0]] + text[offset + 1:]
    return None


def convert_emoji(text: str, spans: list, pick: int) -> str | None:
    """값 자리 밖 낱말 머리의 '버스'·'정류소' 같은 말 하나를 같은 뜻의 이모지로 바꾼다('버스가' -> '🚌가')."""
    covered = {i for s, e, _ in spans for i in range(s, e)}
    options, taken = [], set()
    for word, emoji in EMOJI_WORDS:
        for match in re.finditer(re.escape(word), text):
            start, end = match.span()
            inside = set(range(start, end))
            if inside & covered or inside & taken or (start > 0 and core.split_syllable(text[start - 1])):
                continue
            taken |= inside
            options.append((start, end, emoji))
    if not options:
        return None
    start, end, emoji = sorted(options)[pick % len(options)]
    return text[:start] + emoji + text[end:]


def apply_noise(text: str, spans: list, kind: str, number: int) -> str | None:
    """입력 변형 하나를 적용한다. 1억 시험의 31종은 그대로, 새 10종은 여기서 만든다. 적용할 수 없으면 None."""
    if kind not in NEW_DEV_KINDS and kind not in NEW_HELDOUT_KINDS:
        return prior.apply_noise(text, spans, kind, number)
    pick = core.compute_mix(number)
    if kind == "layout_word":
        return apply_word_change(text, spans, pick, convert_layout)
    if kind == "layout_place":
        places = [(s, e) for s, e, name in spans if name in diverse.PLACE_SLOTS]
        if not places:
            return None
        start, end = places[pick % len(places)]
        return text[:start] + convert_layout(text[start:end]) + text[end:]
    if kind == "sound_link":
        return apply_word_change(text, spans, pick, lambda word: apply_pairs(word, compute_link))
    if kind == "sound_assim":
        return apply_word_change(text, spans, pick, lambda word: apply_pairs(word, compute_assimilation))
    if kind == "korean_number":
        return convert_numbers(text)
    if kind == "filler_word":
        return add_word(text, spans, pick, FILLERS)
    if kind == "hedge_word":
        return add_word(text, spans, pick, HEDGES)
    if kind == "space_shift":
        return make_space_shift(text, spans, pick)
    if kind == "jamo_tail":
        return strip_tail(text, spans)
    if kind == "emoji_word":
        return convert_emoji(text, spans, pick)
    raise ValueError(kind)


# ---------------------------------------------------------------- 조합 공간·문항 만들기

def count_space(dims: list, set_name: str) -> int:
    total = len(KIND_SETS[set_name])
    for _, options in dims:
        total *= len(options)
    return total


def build_case(template: dict, frame: dict, dims: list, index: int, set_name: str) -> dict:
    """조합 번호 하나 -> 시험 문항. 1억 시험과 같은 방식이고, 이번 평가용 전용 새 요소(new_*)를 따로 표시한다."""
    kinds = KIND_SETS[set_name]
    values, kind_number = decode_case(index, dims, len(kinds))
    form, ctx = values.pop("FORM", None), values.pop("CTX", None)
    shown, offsets = apply_forms(template["text"], values, form)
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
        ("context", bool(ctx and ctx["heldout_only"])), ("kind", kind in prior.HELDOUT_KINDS),
        ("new_style", template["style"] in HELDOUT_STYLES), ("new_situation", check_situation(template.get("situation"))),
        ("new_kind", kind in NEW_HELDOUT_KINDS),
    )
    novel = [part for part, flag in marks if flag]
    if any(part.startswith("new_") for part in novel):
        novel.append("new_any")
    return {"text": noisy, "context": context, "kind": kind, "expected": compute_expected(frame, values),
            "values": [v["key"] for v in list(values.values()) + extras], "unseen": bool(novel), "novel_parts": novel}
