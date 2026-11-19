"""수치 근거 감사기: 해설 문장 속 숫자가 도구 결과(또는 질문)에 실제로 있는 값인지 대조한다.

Claude 가 숫자를 지어내면 '근거 없는 수치'로 표시해 화면에 경고 배지를 띄운다(환각 차단 2중 장치).
"""
from __future__ import annotations

import math
import re

NUMBER = re.compile(r"(?<![\w.])[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])[-+]?\d+(?:\.\d+)?")
DATE_UNITS = ("년", "월", "일")
COUNT_UNITS = ("번", "개", "곳", "대", "단계", "쪽", "층", "호", "차", "위", "턴", "회", "주")
SKIP_BEFORE = ("Step", "step", "STEP", "#")


def collect_numbers(payload, bucket: set[float]) -> None:
    if isinstance(payload, dict):
        for value in payload.values():
            collect_numbers(value, bucket)
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            collect_numbers(value, bucket)
    elif isinstance(payload, bool) or payload is None:
        return
    elif isinstance(payload, (int, float)):
        if math.isfinite(float(payload)):
            bucket.add(float(payload))
    elif isinstance(payload, str):
        for match in NUMBER.finditer(payload):
            try:
                bucket.add(float(match.group(0).replace(",", "")))
            except ValueError:
                continue


def expand_forms(values: set[float]) -> set[float]:
    forms = set()
    for value in values:
        forms.add(value)
        forms.add(round(value, 2))
        forms.add(round(value, 1))
        forms.add(float(round(value)))
        if abs(value) <= 10:
            forms.add(round(value * 100, 1))   # 비율 -> 퍼센트
            forms.add(round((value - 1) * 100, 1))  # 배율 -> 증감률
        if abs(value) >= 1000:
            forms.add(round(value / 10000, 1))  # 만 단위
    return forms


def is_grounded(number: float, forms: set[float]) -> bool:
    for form in forms:
        tolerance = max(0.051, abs(form) * 0.005)
        if abs(number - form) <= tolerance:
            return True
    return False


def is_near(number: float, forms: set[float]) -> bool:
    """반올림 꼴까지만 같은 값으로 본다(상대 오차 없음). 앞 질문에서 본 값을 그대로 다시 인용했는지 볼 때 쓴다."""
    return any(abs(number - form) <= 0.051 for form in forms)


def audit_numbers(text: str, evidence, question: str = "", exact=None) -> dict:
    """evidence(이번 질문의 도구 결과)는 반올림·퍼센트 꼴과 0.5% 상대 오차까지, exact(같은 대화의 앞 질문 도구 결과)는
    반올림·퍼센트 꼴만 근거로 인정한다. 앞 질문 값까지 넓게 맞추면 지어낸 큰 수가 우연히 맞아떨어질 수 있기 때문이다.
    부호는 보지 않고 크기만 대조한다: 해설은 방향을 말로 쓰므로('38.6% 적습니다') 도구의 -38.6 과 같은 값이다."""
    values: set[float] = set()
    collect_numbers(evidence, values)
    collect_numbers(question, values)
    forms = {abs(form) for form in expand_forms(values)}
    earlier: set[float] = set()
    collect_numbers(exact, earlier)
    earlier_forms = {abs(form) for form in expand_forms(earlier)}
    checked, grounded, ungrounded = 0, 0, []
    for match in NUMBER.finditer(text or ""):
        raw = match.group(0)
        tail = text[match.end(): match.end() + 2]
        head = text[max(0, match.start() - 5): match.start()]
        try:
            number = float(raw.replace(",", ""))
        except ValueError:
            continue
        whole = "." not in raw
        if tail.startswith(DATE_UNITS) or any(head.endswith(p) for p in SKIP_BEFORE):
            continue  # 날짜·단계 번호
        if tail.startswith("시") and whole and 0 <= number <= 24:
            continue  # 시각(18시 등). 분·명·배·% 같은 측정값은 모두 감사한다
        if whole and tail.startswith(COUNT_UNITS):
            continue  # 정류소 3곳, 도구 4회 같은 소수 개수
        if number in (0.0, 1.0, 2.0, 3.0) and whole:
            continue
        checked += 1
        if is_grounded(abs(number), forms) or is_near(abs(number), earlier_forms):
            grounded += 1
        else:
            ungrounded.append(raw)
    return {"checked": checked, "grounded": grounded, "ungrounded": ungrounded,
            "ratio": round(grounded / checked, 4) if checked else 1.0}
