"""자가 치유(Self-Healing) 데이터 계층.

- load_any_csv        : 인코딩(utf-8-sig > cp949 > euc-kr > utf-8 > latin1)·구분자(, \\t ; |) 자동 판별,
                        천 단위 콤마 숫자, 마스킹('*', '**', '***'), 'NULL' 문자열, float32 잡음,
                        엑셀 날짜 자동변환으로 오염된 노선번호('02월 01일' -> '2-1')를 정제하고, 요일·성별 컬럼의
                        혼재 표기('토요일'/'토', 'M'/'남')를 통일한다. 예외를 밖으로 던지지 않는다.
- relax_sparse_filter : 교차 조건 표본이 N<5 이거나 분모가 0이면 조건을 한 단계씩 넓혀 유효 통계를 만든다.
- divide_safely / average_safely : ZeroDivisionError, NaN 을 None 으로 흡수한다.
"""
from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from src.config import MIN_SLICE_ROWS

ENCODINGS = ("utf-8-sig", "cp949", "euc-kr", "utf-8", "latin1")
DELIMITERS = (",", "\t", ";", "|")
NULL_TOKENS = {"", "null", "none", "nan", "na", "n/a", "#n/a", "-", "nil"}
MASK_PATTERN = re.compile(r"^\*+$")
NUMBER_PATTERN = re.compile(r"^[+-]?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?%?$")
MANGLED_CODE = re.compile(r"^(\d{1,2})월\s*(\d{1,2})일$")
CODE_TOKENS = {"id", "cd", "code", "no", "num", "number", "date", "ymd", "month", "year", "zip", "mac", "tag",
               "grid", "age", "datetime"}
KOREAN_CODE_HINTS = ("번호", "코드", "일자", "일시", "날짜", "년월", "연월", "우편", "아이디")
WEEKDAY_MAP = {
    "월": "월", "월요일": "월", "mon": "월", "monday": "월",
    "화": "화", "화요일": "화", "tue": "화", "tuesday": "화",
    "수": "수", "수요일": "수", "wed": "수", "wednesday": "수",
    "목": "목", "목요일": "목", "thu": "목", "thursday": "목",
    "금": "금", "금요일": "금", "fri": "금", "friday": "금",
    "토": "토", "토요일": "토", "sat": "토", "saturday": "토",
    "일": "일", "일요일": "일", "sun": "일", "sunday": "일",
}
SEX_MAP = {"남": "남성", "남성": "남성", "남자": "남성", "m": "남성", "male": "남성",
           "여": "여성", "여성": "여성", "여자": "여성", "f": "여성", "female": "여성"}
# 공식 카드 시트(1-12)는 요일 컬럼 이름이 week. 숫자형 주차 컬럼은 앞 단계에서 숫자로 바뀌어 여기까지 오지 않는다.
WEEKDAY_COLUMN = re.compile(r"(요일|weekday|day_?of_?week|^dow$|^week$|^day_?week$|^week_?day$)", re.IGNORECASE)
SEX_COLUMN = re.compile(r"(성별|^sex$|^gender$|_sex$|_gender$|^sex_|^gender_)", re.IGNORECASE)


@dataclass
class LoadedTable:
    frame: pd.DataFrame
    ok: bool
    encoding: str = ""
    delimiter: str = ""
    notes: list[str] = field(default_factory=list)
    masked_cells: int = 0
    null_cells: int = 0
    restored_codes: int = 0
    skipped_lines: int = 0
    numeric_columns: list[str] = field(default_factory=list)

    def count_rows(self) -> int:
        return int(len(self.frame))


@dataclass
class RelaxedSlice:
    frame: pd.DataFrame
    level: int
    label: str
    rows: int
    relaxed: bool
    note: str = ""
    requested_rows: int = 0


# ---------------------------------------------------------------- 안전 연산

def divide_safely(numerator, denominator, digits: int | None = None):
    """분모가 0·NaN·None 이면 None. 예외를 던지지 않는다."""
    try:
        top = float(numerator)
        bottom = float(denominator)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(top) or not math.isfinite(bottom) or bottom == 0:
        return None
    value = top / bottom
    if not math.isfinite(value):
        return None
    return round(value, digits) if digits is not None else value


def average_safely(values: Iterable, digits: int | None = None):
    clean = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            clean.append(number)
    if not clean:
        return None
    mean = sum(clean) / len(clean)
    return round(mean, digits) if digits is not None else mean


def tidy_number(value, digits: int = 2):
    """JSON 직렬화용: NaN/inf -> None, numpy 스칼라 -> 파이썬 숫자."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if not math.isfinite(number):
        return None
    if number.is_integer() and abs(number) < 1e15:
        return int(number)
    return round(number, digits)


# ---------------------------------------------------------------- 표기 정규화

def normalize_spaces(text) -> str:
    if text is None:
        return ""
    value = str(text).replace("\xa0", " ").replace("　", " ").replace("﻿", "")
    return re.sub(r"\s+", " ", value).strip()


def normalize_weekday(text) -> str:
    key = normalize_spaces(text).lower()
    return WEEKDAY_MAP.get(key, normalize_spaces(text))


def normalize_sex(text) -> str:
    key = normalize_spaces(text).lower()
    return SEX_MAP.get(key, normalize_spaces(text))


def restore_mangled_code(text) -> tuple[str, bool]:
    """엑셀이 '2-1' 같은 노선번호를 '02월 01일'로 바꿔 저장한 값을 되돌린다."""
    value = normalize_spaces(text)
    match = MANGLED_CODE.match(value)
    if not match:
        return value, False
    return f"{int(match.group(1))}-{int(match.group(2))}", True


def parse_number(text):
    """'1,045.23' -> 1045.23, '12.5%' -> 12.5, 마스킹/NULL -> None."""
    value = normalize_spaces(text)
    if value.lower() in NULL_TOKENS or MASK_PATTERN.match(value):
        return None
    if not NUMBER_PATTERN.match(value):
        return None
    number = float(value.replace(",", "").replace("%", ""))
    return round(number, 6)


def looks_like_code(column: str) -> bool:
    """식별자·날짜·코드성 컬럼은 숫자처럼 보여도 문자열로 보존한다(선행 0, 코드 의미 보호)."""
    name = str(column).lower()
    tokens = [token for token in re.split(r"[_\s\-]+", name) if token]
    if any(token in CODE_TOKENS for token in tokens):
        return True
    return any(hint in name for hint in KOREAN_CODE_HINTS)


# ---------------------------------------------------------------- CSV 로더

def decode_bytes(raw: bytes) -> tuple[str, str]:
    for encoding in ENCODINGS:
        try:
            return raw.decode(encoding), encoding
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("latin1", errors="replace"), "latin1"


def sniff_delimiter(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()][:30]
    if not lines:
        return ","
    try:
        dialect = csv.Sniffer().sniff("\n".join(lines[:10]), delimiters="".join(DELIMITERS))
        if dialect.delimiter in DELIMITERS:
            return dialect.delimiter
    except csv.Error:
        pass
    best, best_score = ",", -1.0
    for candidate in DELIMITERS:
        counts = [line.count(candidate) for line in lines]
        if min(counts) == 0:
            continue
        spread = max(counts) - min(counts)
        score = min(counts) - 0.5 * spread
        if score > best_score:
            best, best_score = candidate, score
    return best


def read_source_bytes(source) -> bytes:
    if source is None:
        return b""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if hasattr(source, "getvalue"):
        return source.getvalue()
    if hasattr(source, "read"):
        return source.read()
    raise TypeError(f"지원하지 않는 입력 형식: {type(source).__name__}")


def clean_frame(frame: pd.DataFrame, table: LoadedTable) -> pd.DataFrame:
    renamed = {}
    for position, column in enumerate(frame.columns):
        label = normalize_spaces(column)
        if not label or label.lower().startswith("unnamed"):
            label = f"col_{position + 1}"
        renamed[column] = label
    frame = frame.rename(columns=renamed)
    if frame.columns.duplicated().any():
        seen: dict[str, int] = {}
        labels = []
        for column in frame.columns:
            seen[column] = seen.get(column, 0) + 1
            labels.append(column if seen[column] == 1 else f"{column}_{seen[column]}")
        frame.columns = labels
        table.notes.append("중복 컬럼명을 번호를 붙여 구분했습니다.")

    for column in frame.columns:
        texts = (frame[column].astype(str)
                 .str.replace("﻿", "", regex=False)
                 .str.replace(r"[\s\xa0　]+", " ", regex=True)
                 .str.strip())
        masked = texts.str.fullmatch(r"\*+").fillna(False)
        nulls = texts.str.lower().isin(NULL_TOKENS)
        table.masked_cells += int(masked.sum())
        table.null_cells += int(nulls.sum())
        code_like = looks_like_code(column) or "노선" in str(column)

        mangled = texts.str.fullmatch(MANGLED_CODE.pattern).fillna(False)
        if code_like and mangled.any():
            parts = texts[mangled].str.extract(MANGLED_CODE.pattern)
            texts = texts.copy()
            texts[mangled] = parts[0].astype(int).astype(str) + "-" + parts[1].astype(int).astype(str)
            table.restored_codes += int(mangled.sum())

        if not code_like:
            usable = (~masked) & (~nulls)
            numeric_shape = texts.str.fullmatch(NUMBER_PATTERN.pattern).fillna(False)
            if usable.sum() > 0 and numeric_shape[usable].mean() >= 0.9:
                stripped = texts.where(numeric_shape).str.replace(",", "", regex=False).str.rstrip("%")
                numbers = pd.to_numeric(stripped, errors="coerce").round(6)
                broken = int((usable & numbers.isna()).sum())
                if broken:
                    table.notes.append(f"'{column}' 컬럼의 숫자가 아닌 값 {broken}개를 결측으로 처리했습니다.")
                frame[column] = numbers
                table.numeric_columns.append(column)
                continue
            texts = normalize_labels(texts, column, ~(masked | nulls), table)
        frame[column] = texts.where(~(masked | nulls), other=None)
    return frame


def normalize_labels(texts: pd.Series, column, usable: pd.Series, table: LoadedTable) -> pd.Series:
    """요일·성별 컬럼의 혼재 표기('토요일'/'토', 'M'/'남')를 normalize_weekday / normalize_sex 로 통일한다."""
    label = str(column)
    convert = normalize_weekday if WEEKDAY_COLUMN.search(label) else normalize_sex if SEX_COLUMN.search(label) else None
    if convert is None or not usable.any():
        return texts
    unified = texts.where(~usable, texts.map(convert))
    changed = int((unified != texts).sum())
    if changed:
        example = "월요일 -> 월" if convert is normalize_weekday else "M -> 남성"
        table.notes.append(f"'{label}' 컬럼의 혼재 표기 {changed}개를 표준값으로 통일했습니다(예: {example}).")
    return unified


def load_any_csv(source, max_rows: int | None = None) -> LoadedTable:
    """어떤 CSV 가 들어와도 예외 없이 LoadedTable 을 돌려준다."""
    empty = pd.DataFrame()
    try:
        raw = read_source_bytes(source)
    except Exception as exc:  # 파일 접근 실패도 결과로 보고한다
        return LoadedTable(frame=empty, ok=False, notes=[f"파일을 읽지 못했습니다: {exc}"])
    if not raw or not raw.strip():
        return LoadedTable(frame=empty, ok=False, notes=["빈 파일입니다."])

    text, encoding = decode_bytes(raw)
    delimiter = sniff_delimiter(text)
    table = LoadedTable(frame=empty, ok=False, encoding=encoding, delimiter=delimiter)
    if encoding != "utf-8-sig":
        table.notes.append(f"인코딩을 {encoding}(으)로 자동 판별해 복구했습니다.")
    if delimiter != ",":
        label = {"\t": "탭", ";": "세미콜론", "|": "파이프"}.get(delimiter, delimiter)
        table.notes.append(f"구분자를 '{label}'(으)로 자동 판별했습니다.")

    bad_lines: list = []

    def skip_line(bad):
        bad_lines.append(bad)
        return None

    try:
        frame = pd.read_csv(
            io.StringIO(text), sep=delimiter, dtype=str, keep_default_na=False,
            engine="python", on_bad_lines=skip_line, nrows=max_rows, skip_blank_lines=True,
        )
    except Exception as exc:
        table.notes.append(f"표 구조를 해석하지 못했습니다: {type(exc).__name__}")
        return table

    table.skipped_lines = len(bad_lines)
    if bad_lines:
        table.notes.append(f"열 개수가 맞지 않는 {len(bad_lines)}개 행을 건너뛰었습니다.")
    try:
        frame = clean_frame(frame, table)
    except Exception as exc:
        table.notes.append(f"정제 중 일부 단계를 생략했습니다: {type(exc).__name__}")
    if table.masked_cells:
        table.notes.append(f"개인정보 보호 마스킹('*') {table.masked_cells}개 셀을 결측으로 처리했습니다(0으로 간주하지 않음).")
    if table.restored_codes:
        table.notes.append(f"엑셀 날짜 자동변환으로 오염된 코드 {table.restored_codes}건을 복원했습니다(예: '02월 01일' -> '2-1').")
    if frame.empty:
        table.notes.append("데이터 행이 없습니다(헤더만 존재).")
    table.frame = frame
    table.ok = len(frame.columns) > 0
    return table


# ---------------------------------------------------------------- 점진적 조건 완화

def relax_sparse_filter(
    frame: pd.DataFrame,
    stages: Sequence[tuple[str, Callable[[pd.DataFrame], pd.Series]]],
    min_rows: int = MIN_SLICE_ROWS,
    value_column: str | None = None,
) -> RelaxedSlice:
    """stages 는 (설명, 마스크 함수) 목록이며 좁은 조건에서 넓은 조건 순서로 준다.

    표본 수가 min_rows 미만이거나 value_column 합(분모)이 0이면 다음 단계로 넘어간다.
    """
    if frame is None or frame.empty or not stages:
        return RelaxedSlice(frame=pd.DataFrame(), level=-1, label="데이터 없음", rows=0, relaxed=False,
                            note="해당 조건에 맞는 원자료가 없습니다.")
    requested_rows = 0
    last = None
    for level, (label, mask_fn) in enumerate(stages):
        try:
            subset = frame[mask_fn(frame)]
        except Exception:
            continue
        rows = int(len(subset))
        if level == 0:
            requested_rows = rows
        denominator_ok = True
        if value_column and value_column in subset.columns:
            total = pd.to_numeric(subset[value_column], errors="coerce").fillna(0).sum()
            denominator_ok = bool(total > 0)
        last = (level, label, subset, rows)
        if rows >= min_rows and denominator_ok:
            note = ""
            if level > 0:
                note = (f"요청하신 세부 조건(N={requested_rows}<{min_rows})의 표본이 적어 "
                        f"{label}(N={rows}) 기준으로 자동 확장해 분석했습니다.")
            return RelaxedSlice(frame=subset, level=level, label=label, rows=rows, relaxed=level > 0,
                                note=note, requested_rows=requested_rows)
    if last is None:
        return RelaxedSlice(frame=pd.DataFrame(), level=-1, label="데이터 없음", rows=0, relaxed=False,
                            note="조건식을 적용할 수 없어 분석을 보류했습니다.")
    level, label, subset, rows = last
    return RelaxedSlice(frame=subset, level=level, label=label, rows=rows, relaxed=True,
                        note=f"가장 넓은 조건({label})에서도 표본이 N={rows}로 적어 결과 해석에 주의가 필요합니다.",
                        requested_rows=requested_rows)
