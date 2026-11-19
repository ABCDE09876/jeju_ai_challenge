"""제주 지명 자모(Jamo) 퍼지 색인.

- 한글 음절을 초·중·종성 자모로 분해해 편집거리를 계산하므로 '노형똥', '신졔주', '아라ㄷㅗㅇ' 같은 오타도 잡는다.
- 초성만 입력('ㄴㅎㄷ', 'ㅅㅈㅈ')해도 지명을 찾는다.
- 조사('이랑', '에서', '은/는' 등)를 떼어 내고, 별칭(신제주, 삼화지구 등)과 정류소·상권명을 함께 색인한다.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field, replace
from functools import lru_cache

from src.config import GAZETTEER_PATH

CHOSEONG = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNGSEONG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONGSEONG = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")
PARTICLES = sorted([
    "에서는", "에서도", "으로는", "이랑은", "하고는", "이라는", "이랑", "하고", "에서", "으로", "까지", "부터", "보다",
    "처럼", "만큼", "이나", "이며", "이고", "와는", "과는", "에는", "에도", "쪽은", "쪽이", "쪽", "근처", "주변", "일대",
    "동네", "지역", "은", "는", "이", "가", "을", "를", "에", "의", "와", "과", "랑", "로", "도", "만", "요", "께", "엔",
    # 붙여 쓴 '중(에)'·'아니면'과 서술격 조사('애월읍이야 아니면 이도2동이야'): LLM 변형 질문 견고성 시험에서 보강
    "중에서", "중에", "중", "아니면", "이에요", "예요", "이야", "야", "이냐", "냐", "인가", "보다는", "보단",
], key=len, reverse=True)
# '연동카드', '삼화지구정류소', '이호테우밤'처럼 지명 뒤에 붙여 쓴 명사·시간어
GLUED_NOUNS = ["정류소", "정류장", "버스", "카드", "상권", "매출", "밤", "새벽", "저녁", "아침", "점심", "오전", "오후"]
PHRASE_THRESHOLD = 0.88  # 띄어 쓴 별칭('서귀포 신시가지')의 오타 허용 기준
GLUED_MIN = 4  # 이 길이 이상인 어절이 지명으로 해석되지 않으면 붙여 쓴 지명을 찾아본다
NUMERAL_DONG = {"일동": "1동", "이동": "2동", "삼동": "3동"}
RISKY_BASES = {"중앙", "천지", "연", "정방"}
STOPWORDS = {
    "버스", "퇴근", "출근", "시간", "저녁", "아침", "점심", "주말", "평일", "오늘", "내일", "어디", "어디가", "언제", "비교",
    "비교해", "해줘", "알려줘", "나아", "좋아", "살기", "이사", "외출", "동네", "사람", "많아", "붐벼", "혼잡", "그럼",
    "거기", "말고", "이후", "이전", "밤", "새벽", "장날", "우천", "비가", "오는", "안", "올", "때는", "기준", "어르신",
    "청소년", "주장", "맞아", "사실", "평균", "때문", "때문에", "관광객", "도민", "카드", "매출", "소비", "쓰레기",
    "와이파이", "전기차", "충전", "대기", "기다릴", "몇", "시", "분", "정도", "근데", "그리고", "진짜", "정말",
    "전입", "전출",  # '전입 인구'가 '건입'으로 읽히지 않게(다양성 시험 개발용에서 확인)
    "변동", "환경",  # '매출 변동'·'환경에서'가 연동·한경으로 읽히지 않게(1억 시험 개발용에서 확인)
    "한결",  # '한결 편할 것 같습니다'가 한경으로 읽히지 않게(다양성 시험 개발용에서 확인)
}
# 불용어는 조사를 뗀 꼴로도 보지만, 부사 '한결'에는 조사가 붙지 않으므로 '한결에서'는 한경의 오타로 본다
# (1억 시험 평가용 공식 채점 뒤 사후 수정)
PARTICLE_STOPWORDS = STOPWORDS - {"한결"}
# 섬 전체를 가리키는 말('제주도에 있는 우도면')은 특정 동네가 아니다. 근사 비교하면 '제주도'가 '제주대'로 읽힌다
# (다양성 대규모 시험 개발용 세트에서 확인).
ISLAND_WORDS = {"제주", "제주도", "제주섬", "탐라", "제주특별자치도"}
GLUE_MASK = re.compile(r"정류소|정류장|일대|일원|근처|주변|부근|인근|권역|대중교통|호텔|경우(?!체)|서귀포시청"
                       r"|오라(?:고|구|는데|던|더라|며|면서|니까|길래|네(?!거리)|해서|해요|했|하는데|하던|하셔서|하시|셔서|셨|신다|시는"
                       r"|시던|그래|그러|그랬)")  # '집에오라고'·'필요한경우'·'이호텔'·'제주대중교통'은 지명이 아니다(검토 워크플로에서 보강)
GLUE_BEFORE = ("이랑", "랑", "하고", "과", "와", "에서", "에", "은", "는", "이", "가", "을", "를", "도", "만", "중", "쪽", "□", "제주",
               "제주도", "제주시", "서귀포시", "서귀포", "시", "면", "읍", "동", "말고", "보다", "vs", "VS", "및", "?", ".", "!", "요",
               "그럼", "근데", "요즘", "오늘", "내일", "주말", "평일", "지금", "이나", "부터", "대비",
               "거기", "저기", "여기")  # '거기우도랑'(1억 시험 개발용에서 보강. '그'는 '그건입니다'와 겹쳐 뺌)


@lru_cache(maxsize=100_000)
def decompose_jamo(text: str) -> str:
    """'노형동' -> 'ㄴㅗㅎㅕㅇㄷㅗㅇ'. 한글 이외 문자는 그대로 둔다(같은 입력은 기억해 둔 값을 쓴다)."""
    out = []
    for char in unicodedata.normalize("NFC", text):
        code = ord(char) - 0xAC00
        if 0 <= code < 11172:
            out.append(CHOSEONG[code // 588])
            out.append(JUNGSEONG[(code % 588) // 28])
            tail = JONGSEONG[code % 28]
            if tail:
                out.append(tail)
        else:
            out.append(char)
    return "".join(out)


def extract_choseong(text: str) -> str:
    out = []
    for char in text:
        code = ord(char) - 0xAC00
        out.append(CHOSEONG[code // 588] if 0 <= code < 11172 else char)
    return "".join(out)


def is_choseong_only(text: str) -> bool:
    return bool(text) and all("ㄱ" <= ch <= "ㅎ" for ch in text)


def measure_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, start=1):
        current = [i]
        for j, b in enumerate(right, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def collect_grams(jamo: str) -> frozenset:
    return frozenset(jamo[i:i + 2] for i in range(len(jamo) - 1)) or frozenset({jamo})


def score_similarity(left: str, right: str) -> float:
    """자모 편집거리 유사도(85%) + 자모 2-gram 자카드(15%)."""
    a, b = decompose_jamo(left), decompose_jamo(right)
    if not a or not b:
        return 0.0
    return score_parts(a, collect_grams(a), b, collect_grams(b))


def score_parts(a: str, grams_a: frozenset, b: str, grams_b: frozenset, factor: float = 1.0, beat: float = -1.0,
                reach: float = -1.0) -> float:
    """score_similarity(a, b) * factor 와 같은 값(같은 연산 순서). 다만 점수 상한이 beat 이하이거나 reach 미만이면
    편집거리 계산을 건너뛰고 -1.0 을 돌려준다(결과가 바뀌지 않는 가지치기).

    상한: 편집 한 번은 2-gram 을 최대 2개 없애므로 편집거리 >= ceil(|A-B|/2), ceil(|B-A|/2), 길이 차이.
    상한이 지금까지의 최고점 이하이면 최고점을 바꿀 수 없고, 기준 미만이면 채택될 수 없다.
    """
    longest = max(len(a), len(b))
    jaccard = len(grams_a & grams_b) / len(grams_a | grams_b)
    lower = max(abs(len(a) - len(b)), -(-len(grams_a - grams_b) // 2), -(-len(grams_b - grams_a) // 2))
    upper = (0.85 * (1 - lower / longest) + 0.15 * jaccard) * factor
    if upper <= beat or upper < reach:
        return -1.0
    edit = 1 - measure_distance(a, b) / longest
    return (0.85 * edit + 0.15 * jaccard) * factor


def strip_particles(token: str, deep: bool = True) -> list[str]:
    """조사·접미어를 떼어 낸 후보 목록(원형 포함, 긴 것부터).

    deep 이면 '예래동이랑도'처럼 조사가 두 겹이거나 '연동카드'처럼 명사가 붙은 말을 한 번 더 뗀다(정확 일치용).
    근사 비교는 후보마다 사전 전체를 훑으므로 한 겹 후보만 쓴다.
    """
    token = token.strip()
    candidates = [token]
    for particle in PARTICLES + GLUED_NOUNS:
        if token.endswith(particle) and len(token) - len(particle) >= 1:
            candidates.append(token[: -len(particle)])
    for stem in candidates[1:] if deep else []:
        for particle in PARTICLES + GLUED_NOUNS:
            if stem.endswith(particle) and len(stem) - len(particle) >= 2:
                candidates.append(stem[: -len(particle)])
    for word, digit in NUMERAL_DONG.items():
        for candidate in list(candidates):
            if candidate.endswith(word) and len(candidate) > 2:
                candidates.append(candidate[: -len(word)] + digit)
    seen, ordered = set(), []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return ordered


@dataclass
class PlaceEntry:
    surface: str
    kind: str            # emd | alias | landmark | station
    targets: list[str]
    label: str
    weight: float = 1.0
    jamo: str = ""       # 미리 분해해 둔 자모(근사 비교용)
    grams: frozenset = frozenset()


@dataclass
class PlaceMatch:
    query: str
    surface: str
    kind: str
    targets: list[str]
    label: str
    score: float
    corrected: bool = False
    position: int = 0


@dataclass
class AreaUnit:
    label: str
    emds: list[str]
    kind: str
    matched_from: str = ""
    score: float = 1.0
    corrected: bool = False
    notes: list[str] = field(default_factory=list)

    def export_unit(self) -> dict:
        return {"label": self.label, "emds": self.emds, "kind": self.kind, "matched_from": self.matched_from,
                "score": round(self.score, 3), "corrected": self.corrected}


class PlaceIndex:
    def __init__(self, gazetteer: dict):
        self.gazetteer = gazetteer
        self.regions = {row["emd_nm"]: row for row in gazetteer["regions"]}
        self.alias_rows = {row["alias"]: row for row in gazetteer["aliases"]}
        self.entries: list[PlaceEntry] = []
        self.exact: dict[str, PlaceEntry] = {}
        self.spaced: list[PlaceEntry] = []  # 원래 띄어 쓴 이름(2어절 근사 일치 후보)
        self.token_memo: dict = {}  # 어절별 해석 결과(같은 어절의 근사 비교 반복을 없앤다)
        self.phrase_memo: dict = {}  # 2어절 구절별 근사 비교 결과
        self.glued_keys: dict[str, list[str]] = {}  # 첫 글자 -> 표기(긴 것부터), 붙여 쓴 어절 분해용
        self.glued_tails: tuple[str, ...] = ()  # 두 글자 이상 표기(붙여 쓴 지명 바로 뒤의 두 글자 지명 허용용)
        self.build_entries()

    def add_entry(self, surface: str, kind: str, targets: list[str], label: str, weight: float = 1.0) -> None:
        key = surface.replace(" ", "")
        if not key or key in self.exact:
            return
        jamo = decompose_jamo(key)
        entry = PlaceEntry(key, kind, targets, label, weight, jamo, collect_grams(jamo))
        self.entries.append(entry)
        self.exact[key] = entry
        if " " in surface.strip():
            self.spaced.append(entry)

    def match_phrase(self, phrase: str) -> tuple[PlaceEntry | None, float]:
        """2어절 구절을 띄어 쓴 이름과 근사 비교한다('서귀포 신시가기' -> '서귀포 신시가지'). 같은 구절은 기억해 둔다."""
        if phrase not in self.phrase_memo:
            if len(self.phrase_memo) > 50_000:
                self.phrase_memo.clear()
            self.phrase_memo[phrase] = self.search_phrase(phrase)
        return self.phrase_memo[phrase]

    def search_phrase(self, phrase: str) -> tuple[PlaceEntry | None, float]:
        best, best_score = None, 0.0
        for candidate in strip_particles(phrase):
            if len(candidate) < 4:
                continue
            jamo = decompose_jamo(candidate)
            grams = collect_grams(jamo)
            for entry in self.spaced:
                if abs(len(entry.surface) - len(candidate)) > 1:
                    continue
                score = score_parts(jamo, grams, entry.jamo, entry.grams, beat=best_score, reach=PHRASE_THRESHOLD)
                if score > best_score:
                    best, best_score = entry, score
        return (best, best_score) if best_score >= PHRASE_THRESHOLD else (None, 0.0)

    def build_entries(self) -> None:
        for name, row in self.regions.items():
            scale = 1.0
            self.add_entry(name, "emd", [name], name, scale)
            self.add_entry(row["full_nm"], "emd", [name], name, scale)
            base = re.sub(r"(읍|면|동)$", "", name)
            if len(base) >= 2 and base not in RISKY_BASES:
                self.add_entry(base, "emd", [name], name, 0.95)
            for digit, word in (("1동", "일동"), ("2동", "이동"), ("3동", "삼동")):
                if name.endswith(digit):
                    self.add_entry(name[: -len(digit)] + word, "emd", [name], name, 0.95)
        for alias, row in self.alias_rows.items():
            self.add_entry(alias, "alias", row["targets"], alias, 1.0)
        for row in self.gazetteer.get("landmarks", []):
            self.add_entry(row["name"], "landmark", [row["emd_nm"]], row["name"], 0.9)
        for row in self.gazetteer.get("bus_stations", []):
            self.add_entry(row["bus_station_name"], "station", [row["emd_nm"]], row["bus_station_name"], 0.85)

    # ------------------------------------------------------------ 단일 토큰

    def match_token(self, token: str, threshold: float = 0.75) -> PlaceMatch | None:
        """한 어절을 지명으로 해석한다. 같은 어절은 기억해 둔 결과의 복사본을 돌려준다(호출부가 position 을 바꿈)."""
        key = (token, threshold)
        if key not in self.token_memo:
            if len(self.token_memo) > 50_000:
                self.token_memo.clear()
            self.token_memo[key] = self.search_token(token, threshold)
        found = self.token_memo[key]
        return replace(found) if found else None

    def search_token(self, token: str, threshold: float = 0.75) -> PlaceMatch | None:
        cleaned = re.sub(r"[^\w가-힣ㄱ-ㅎㅏ-ㅣ]", "", token)
        if not cleaned:
            return None
        for candidate in strip_particles(cleaned):
            entry = self.exact.get(candidate)
            if entry:
                return PlaceMatch(token, entry.surface, entry.kind, entry.targets, entry.label, 1.0, corrected=False)
        if cleaned in STOPWORDS or any(candidate in PARTICLE_STOPWORDS for candidate in strip_particles(cleaned)) \
                or re.fullmatch(r"\d+.*", cleaned):
            return None
        if any(candidate in ISLAND_WORDS for candidate in strip_particles(cleaned)):
            return None
        if any(candidate in ("전입", "전출") for candidate in strip_particles(cleaned)):
            return None  # '전입이 많은 동네'의 전입이 건입으로 읽히지 않게('전입동' 오타는 건입동으로 둔다)
        # 초성만 친 지명. 'ㄴㅎㄷ이랑'처럼 조사가 붙어도 조사를 뗀 초성으로 찾는다(대규모 행동 시험에서 보강).
        choseong = next((c for c in strip_particles(cleaned) if len(c) >= 3 and is_choseong_only(c)), None)
        if choseong:
            hits = [e for e in self.entries if extract_choseong(e.surface) == choseong]
            if hits:
                best = max(hits, key=lambda e: (e.kind == "alias", e.weight, -len(e.surface)))
                return PlaceMatch(token, best.surface, best.kind, best.targets, best.label, 0.9, corrected=True)
            if choseong == cleaned:
                return None
        best_match, best_score = None, 0.0
        limit = threshold if len(cleaned) > 3 else max(threshold, 0.8)
        for candidate in strip_particles(cleaned, deep=False):
            if len(candidate) < 2:
                continue
            jamo = decompose_jamo(candidate)
            grams = collect_grams(jamo)
            for entry in self.entries:
                if abs(len(entry.surface) - len(candidate)) > 2:
                    continue
                score = score_parts(jamo, grams, entry.jamo, entry.grams, factor=0.98 + 0.02 * entry.weight,
                                    beat=best_score, reach=limit)
                if score > best_score:
                    best_match, best_score = entry, score
        if best_match and best_score >= limit:
            return PlaceMatch(token, best_match.surface, best_match.kind, best_match.targets, best_match.label,
                              best_score, corrected=True)
        return None

    # ------------------------------------------------------------ 문장 전체

    def find_places(self, text: str) -> list[PlaceMatch]:
        """문장에서 지명을 등장 순서대로 찾는다(2어절 별칭 우선)."""
        if not text:
            return []
        words = re.findall(r"[가-힣ㄱ-ㅎㅏ-ㅣA-Za-z0-9]+", text)
        found: list[PlaceMatch] = []
        used = set()
        for size in (2, 1):
            for i in range(len(words) - size + 1):
                if any(j in used for j in range(i, i + size)):
                    continue
                phrase = "".join(words[i:i + size])
                if size == 2:
                    entry, score, corrected = None, 1.0, False
                    for candidate in strip_particles(phrase):
                        entry = self.exact.get(candidate)
                        if entry:
                            break
                    if entry and entry.surface in strip_particles(words[i]):
                        continue  # 첫 어절만으로 이미 지명이면('제주민속오일장 상권') 1어절 단계에 맡긴다
                    if not entry:
                        entry, score = self.match_phrase(phrase)
                        corrected = entry is not None
                    if not entry or entry.kind not in ("alias", "landmark", "station"):
                        continue
                    match = PlaceMatch(phrase, entry.surface, entry.kind, entry.targets, entry.label, score, corrected, i)
                else:
                    match = self.match_token(phrase)
                    if not match:
                        continue
                    match.position = i
                used.update(range(i, i + size))
                found.append(match)
        for i, word in enumerate(words):
            if i not in used and len(word) >= GLUED_MIN:
                for match in self.scan_glued(word, i):
                    found.append(match)
                    used.add(i)
        found.sort(key=lambda m: m.position)
        unique, seen = [], set()
        for match in found:
            key = (match.label, tuple(match.targets))
            if key not in seen:
                seen.add(key)
                unique.append(match)
        return unique

    def scan_glued(self, word: str, position: int) -> list[PlaceMatch]:
        """띄어쓰기 없이 붙여 쓴 어절('노형동이랑외도동중에…')에서 사전 표기를 긴 것부터 겹치지 않게 찾는다.

        어절 전체가 지명으로 해석되지 않을 때만 쓴다(대규모 행동 시험의 띄어쓰기 생략 변형에서 보강).
        """
        if not self.glued_keys:
            for key in sorted(self.exact, key=len, reverse=True):
                if len(key) >= 2:
                    self.glued_keys.setdefault(key[0], []).append(key)
        # '쪽일대정류소에서'의 '대정', '급건입니다'의 '건입'처럼 다른 낱말에 걸친 두 글자 지명은 지명이 아니다(다양성 시험에서 보강):
        # 정류소·일대 같은 낱말을 먼저 가리고, 두 글자 표기는 어절 첫머리나 조사·꾸밈말 바로 뒤에서만 찾는다.
        if not self.glued_tails:
            self.glued_tails = tuple(key for key in self.exact if len(key) >= 2)
        word = GLUE_MASK.sub(lambda match: "□" * len(match.group(0)), word)
        # 두 글자 표기는 다른 지명 바로 뒤에도 허용한다('노형연동'). 조사 '이'+'호~'(이호)와 '도두'+수량 글자('도두 시')는 뺀다.
        spans = [(start, start + len(key), key) for start in range(len(word))
                 for key in self.glued_keys.get(word[start], ()) if word.startswith(key, start)
                 and (len(key) > 2 or start == 0 or word[:start].endswith(GLUE_BEFORE)
                      or (key != "이호" and word[:start].endswith(self.glued_tails)))
                 and not (start > 0 and key == "도두" and word[start + 2:start + 3] in ("시", "번", "개", "명", "대", "분", "배"))]
        chosen: list[tuple[int, int, str]] = []
        # 긴 표기부터, 같은 길이면 읍면동 표기('제주대정읍'은 대정읍), 겹치면 버림
        for start, end, key in sorted(spans, key=lambda s: (s[0] - s[1], self.exact[s[2]].kind != "emd", s[0])):
            if all(end <= s or start >= e for s, e, _ in chosen):
                chosen.append((start, end, key))  # '인구제주시오라동'은 '구제주'가 아니라 '제주시오라동'
        matches = []
        for _, _, key in sorted(chosen):
            entry = self.exact[key]
            matches.append(PlaceMatch(key, entry.surface, entry.kind, entry.targets, entry.label, 1.0, False, position))
        return matches

    # ------------------------------------------------------------ 도구 입력

    def resolve_areas(self, names) -> tuple[list[AreaUnit], list[str]]:
        """도구 입력 areas(문자열 목록)를 분석 단위로 바꾼다. 못 찾은 이름은 두 번째 값으로 돌려준다."""
        units: list[AreaUnit] = []
        unknown: list[str] = []
        if isinstance(names, str):
            names = [names]
        for raw in names or []:
            text = str(raw).strip()
            if not text:
                continue
            match = self.match_token(text.replace(" ", ""))
            if not match:
                places = self.find_places(text)
                match = places[0] if places else None
            if not match:
                unknown.append(text)
                continue
            label = match.label if match.kind in ("alias", "emd") else f"{match.label}({match.targets[0]})"
            unit = AreaUnit(label, list(dict.fromkeys(match.targets)), match.kind, text, match.score, match.corrected)
            if not any(u.label == unit.label for u in units):
                units.append(unit)
        return units, unknown

    def suggest_names(self, text: str, limit: int = 3) -> list[str]:
        scored = sorted(self.entries, key=lambda e: score_similarity(text, e.surface), reverse=True)
        names = []
        for entry in scored:
            if entry.label not in names:
                names.append(entry.label)
            if len(names) >= limit:
                break
        return names

    def pick_partner(self, unit: AreaUnit, exclude: set[str] | None = None) -> AreaUnit:
        """단일 지역 질문에 자동 비교 상대를 붙인다(사전의 생활권 비교 힌트 우선)."""
        exclude = exclude or set()
        hint = ""
        if unit.kind == "alias" and unit.label in self.alias_rows:
            hint = self.alias_rows[unit.label]["partner_hint"]
        elif unit.emds and unit.emds[0] in self.regions:
            hint = self.regions[unit.emds[0]]["partner_hint"]
        if hint and hint not in exclude:
            if hint in self.alias_rows:
                return AreaUnit(hint, list(self.alias_rows[hint]["targets"]), "alias", "auto_partner")
            if hint in self.regions:
                return AreaUnit(hint, [hint], "emd", "auto_partner")
        city = self.regions.get(unit.emds[0], {}).get("city", "제주시") if unit.emds else "제주시"
        for name, row in self.regions.items():
            if row["city"] == city and name not in unit.emds and name not in exclude:
                return AreaUnit(name, [name], "emd", "auto_partner")
        return AreaUnit("노형동", ["노형동"], "emd", "auto_partner")


@lru_cache(maxsize=1)
def load_place_index() -> PlaceIndex:
    gazetteer = json.loads(GAZETTEER_PATH.read_text(encoding="utf-8"))
    return PlaceIndex(gazetteer)
