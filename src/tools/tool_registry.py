"""4대 도구를 Anthropic Messages API tools(JSON Schema) 규격으로 노출하고 실행한다.

같은 파이썬 함수를 로컬 엔진과 Claude Tool Use 가 함께 쓰므로 두 경로의 수치는 항상 같다.
"""
from __future__ import annotations

import json
import time
import traceback

from src.tools.boundary_simulator import simulate_reversal_boundary
from src.tools.catalog_resolver import resolve_spatial_catalog
from src.tools.confounder_decomposer import decompose_confounders
from src.tools.slice_excavator import excavate_masked_slice

TOPIC_ENUM = ["commute", "night", "living", "spending", "crowd", "waste", "wifi", "ev", "general"]
AREA_FIELD = {
    "type": "array", "items": {"type": "string"}, "maxItems": 4,
    "description": "제주 읍면동명 또는 생활권 별칭(예: 노형동, 신제주, 구제주, 삼화지구, 애월, 오일장). 오타 허용.",
}

TOOL_SPECS = [
    {
        "name": "resolve_spatial_catalog",
        "description": ("질문 주제에 맞는 제주 공공데이터 2~4종을 고르고 공간 키(행정동-법정동-정류소 반경-기상관측소)를 "
                        "자동 결합한다. 지역명 해석·오타 교정 결과, 사용할 테이블의 공식 예시 시트 컬럼 일치 여부와 "
                        "출처(source_type, dataset_id, reference_date)를 돌려준다. 분석 첫 단계에서 호출한다."),
        "input_schema": {
            "type": "object",
            "properties": {
                "areas": AREA_FIELD,
                "topic": {"type": "string", "enum": TOPIC_ENUM, "description": "질문 주제"},
                "question": {"type": "string", "description": "사용자 원문 질문"},
            },
            "required": ["areas", "topic"],
        },
    },
    {
        "name": "decompose_confounders",
        "description": ("단순 전체 평균에서 3대 허수(①관광객·도외 거주자 거품 ②시간대 쏠림·야간 공동화 ③악천후)를 차례로 "
                        "벗겨 지역 간 순위 역전을 검출한다. 지역이 1곳이면 비교 상대를 자동으로 붙인다. "
                        "verdict: reversal / conditional_reversal / no_reversal."),
        "input_schema": {
            "type": "object",
            "properties": {
                "areas": AREA_FIELD,
                "topic": {"type": "string", "enum": TOPIC_ENUM},
                "hour_from": {"type": "integer", "minimum": 0, "maximum": 23, "description": "시작 시각(24시간제)"},
                "hour_to": {"type": "integer", "minimum": 0, "maximum": 23, "description": "끝 시각(24시간제)"},
                "weather": {"type": "string", "enum": ["any", "rain", "dry"]},
                "day_type": {"type": "string",
                             "enum": ["any", "weekday", "weekend", "market", "월", "화", "수", "목", "금", "토", "일"]},
            },
            "required": ["areas"],
        },
    },
    {
        "name": "excavate_masked_slice",
        "description": ("연·월 평균으로는 정상인데 특정 [요일×시간대×날씨/장날] 조각에서 폭증하는 은폐 구간을 발굴하고, "
                        "월초 계단식 집계 단절(수집 방식 변경 의심)과 원자료 컬럼에 없는 미측정 인과 주장을 잡아 "
                        "중립 교정문을 만든다. 주장 검증(팩트체크) 질문이면 claim 에 원문을 넣는다."),
        "input_schema": {
            "type": "object",
            "properties": {
                "areas": AREA_FIELD,
                "topic": {"type": "string", "enum": TOPIC_ENUM},
                "claim": {"type": "string", "description": "검증할 주장 원문(없으면 빈 문자열)"},
                "mode": {"type": "string", "enum": ["all", "slice", "break", "causal"]},
                "user_group": {"type": "string", "enum": ["어르신", "청소년", "일반"]},
            },
            "required": ["areas"],
        },
    },
    {
        "name": "simulate_reversal_boundary",
        "description": ("귀가 시각·방문 시각·허용 대기시간·우천 여부를 움직이며 결론이 뒤집히는 경계선을 계산한다. "
                        "원자료 한계로 남는 현장 확인 질문 1개, 1장짜리 증거 카드, 추천 후속 질문 3개를 함께 돌려준다."),
        "input_schema": {
            "type": "object",
            "properties": {
                "areas": AREA_FIELD,
                "variable": {"type": "string", "enum": ["return_hour", "visit_hour", "wait_minutes", "rain"]},
                "topic": {"type": "string", "enum": TOPIC_ENUM},
                "weather": {"type": "string", "enum": ["any", "rain", "dry"]},
                "day_type": {"type": "string",
                             "enum": ["any", "weekday", "weekend", "market", "월", "화", "수", "목", "금", "토", "일"]},
                "user_group": {"type": "string", "enum": ["어르신", "청소년", "일반"]},
            },
            "required": ["areas"],
        },
    },
]

TOOL_FUNCTIONS = {
    "resolve_spatial_catalog": resolve_spatial_catalog,
    "decompose_confounders": decompose_confounders,
    "excavate_masked_slice": excavate_masked_slice,
    "simulate_reversal_boundary": simulate_reversal_boundary,
}
TOOL_LABELS = {
    "resolve_spatial_catalog": "Step 1 · 데이터셋·공간 키 결합",
    "decompose_confounders": "Step 2 · 3대 허수 박리·순위 역전 검출",
    "excavate_masked_slice": "Step 3 · 은폐 구간·집계 단절·인과 과장 감사",
    "simulate_reversal_boundary": "Step 4 · 역전 경계선·현장 확인 질문",
}
COMPACT_KEYS = ("tool", "ok", "verdict", "facts", "reversals", "relaxation", "neutral_rewrite", "field_question",
                "followups", "unknown_areas", "suggestions", "message", "auto_partner", "naive_leader", "final_leader",
                "stress_leader", "boundary", "number_checks", "hours", "day_type", "weather", "variable")


def list_claude_tools() -> list[dict]:
    """Messages API 에 넘길 tools 배열(스트리밍 중 입력 즉시 전송: eager_input_streaming)."""
    return [{**spec, "eager_input_streaming": True} for spec in TOOL_SPECS]


def validate_input(name: str, payload) -> tuple[dict | None, str | None]:
    """eager 스트리밍은 서버 검증을 생략하므로 클라이언트에서 스키마를 확인한다."""
    if name not in TOOL_FUNCTIONS:
        return None, f"알 수 없는 도구: {name}"
    if not isinstance(payload, dict):
        return None, "도구 입력이 JSON 객체가 아닙니다"
    spec = next(s for s in TOOL_SPECS if s["name"] == name)
    properties = spec["input_schema"]["properties"]
    clean = {}
    for key, value in payload.items():
        if key not in properties or value is None:
            continue
        expected = properties[key].get("type")
        if expected == "array":
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                return None, f"{key} 형식 오류"
            value = [str(v) for v in value][:4]
        elif expected == "integer":
            try:
                value = int(value)
            except (TypeError, ValueError):
                return None, f"{key} 는 정수여야 합니다"
        elif expected == "string":
            value = str(value)
            allowed = properties[key].get("enum")
            if allowed and value not in allowed:
                return None, f"{key} 값은 {allowed} 중 하나여야 합니다"
        clean[key] = value
    for key in spec["input_schema"].get("required", []):
        if key not in clean:
            if key == "areas":
                clean["areas"] = []
            else:
                return None, f"필수 입력 누락: {key}"
    return clean, None


def run_tool(name: str, payload: dict) -> dict:
    """도구 실행. 어떤 예외도 밖으로 던지지 않고 ok=False 결과로 돌려준다."""
    started = time.perf_counter()
    clean, problem = validate_input(name, payload)
    if problem:
        return {"tool": name, "ok": False, "error": problem}
    try:
        output = TOOL_FUNCTIONS[name](**clean)
    except Exception as exc:  # 도구 내부 오류도 대화를 멈추지 않는다
        return {"tool": name, "ok": False, "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=3)[-800:]}
    output.setdefault("elapsed_ms", round((time.perf_counter() - started) * 1000, 1))
    return output


def compact_output(output: dict) -> dict:
    """Claude 에 돌려줄 요약본(핵심 수치 문장·판정·출처). 대시보드는 전체 결과를 쓴다."""
    compact = {key: output[key] for key in COMPACT_KEYS if key in output}
    if "metrics" in output:
        compact["metrics"] = [
            {k: v for k, v in m.items() if k in ("label", "naive_total", "resident", "tourist_share", "window_resident",
                                                 "stress_resident", "night_retention", "rain_delta", "bus_rain_delta",
                                                 "card_resident_share", "window_days", "stress_days")}
            for m in output["metrics"]]
    if "provenance" in output:
        compact["provenance"] = [{k: p.get(k) for k in ("table", "source_type", "dataset_id", "reference_date")}
                                 for p in output["provenance"]]
    if "areas" in output:
        compact["areas"] = [{k: a.get(k) for k in ("label", "emds", "corrected", "matched_from") if k in a}
                            for a in output["areas"]]
    if "causal_audit" in output:
        compact["causal_audit"] = {k: output["causal_audit"].get(k) for k in ("status", "effect_label", "assessments")}
    return compact


def serialize_output(output: dict) -> str:
    return json.dumps(compact_output(output), ensure_ascii=False, default=str)
