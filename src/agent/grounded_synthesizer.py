"""근거 묶음(AnswerBundle) 조립 + 로컬 스트리밍 해설 합성기.

두 브레인(Claude / 로컬)이 같은 도구 결과에서 같은 대시보드·증거 카드를 만든다. 해설만 다를 뿐 수치는 동일하다.
모든 답변은 세 구획을 분리해 보여 준다.
  [원자료로 확인된 수치 사실] / [교란 제거 후 역전 구간] / [원자료 컬럼 부재로 인해 단정할 수 없는 한계 및 현장 확인 질문 1개]
"""
from __future__ import annotations

import re

from src.agent.prompts import SECTION_FACTS, SECTION_LIMIT, SECTION_REVERSAL
from src.tools.analysis_cubes import load_meta, stamp_db
from src.tools.gazetteer_index import load_place_index
from src.tools.tool_common import WEATHER_LABELS, attach_josa
from src.tools.tool_registry import TOOL_LABELS


def quote(label) -> str:
    return f"'{label}'"


def mark_subject(label) -> str:
    """'노형동'이 / '연동'이 / '아라동'이 처럼 받침에 맞는 주격 조사를 붙인다."""
    return attach_josa(quote(label), "이/가")

POSITIVE_STEMS = re.compile(r"(많|높|붐비|붐빈|붐벼|붐볐|북적|활기|크다|큰|커|길다|긴|앞서|우세|더\s*많)")
NEGATIVE_STEMS = re.compile(r"(적다|적어|적은|적고|적지|적네|적습|적더|낮|한산|조용|작다|작아|작은|짧)")
NEGATION = re.compile(r"(지\s*않|지\s*못|안\s*(많|높|붐|크|길|적|낮)|없다|없어|아니)")
TOOL_ORDER = ("resolve_spatial_catalog", "decompose_confounders", "excavate_masked_slice", "simulate_reversal_boundary")
SYNTHETIC_NOTICE = "공식 예시데이터 스키마를 보존한 시드 고정 합성(시연용) 데이터 기준이며, data/raw 에 원자료를 넣으면 같은 절차로 재계산됩니다."


# ---------------------------------------------------------------- 주장 검증(정상 대조군)

def read_claim_direction(tail: str) -> int | None:
    """'보다' 뒤 서술의 방향: 1(많다·붐빈다), -1(적다·한산하다), None(판단 불가). '비교적'·'적지 않다'·'덜' 처리."""
    text = tail.replace("비교적", "").replace("상대적", "")
    if "덜" in text:
        direction = -1
    else:
        positive = bool(POSITIVE_STEMS.search(text))
        negative = bool(NEGATIVE_STEMS.search(text))
        if positive == negative:
            return None
        direction = 1 if positive else -1
    if NEGATION.search(text):
        direction = -direction
    return direction


def split_comparison(plan, metrics: list[dict]) -> tuple[dict | None, dict | None]:
    """'A는 B보다'·'B보다 A가' 어느 어순이든 '보다' 바로 앞 지역을 비교 대상(B), 나머지를 주어(A)로 본다."""
    claim = plan.claim or plan.question
    labels = {m["label"] for m in metrics}
    before = claim.split("보다", 1)[0]
    object_label = None
    places = load_place_index().find_places(before)
    if places:
        last = places[-1]
        object_label = next((name for name in (last.label, *last.targets) if name in labels), None)
    if object_label is None and len(plan.areas) >= 2:
        object_label = plan.areas[1]
    target = next((m for m in metrics if m["label"] == object_label), None)
    subject = next((m for m in metrics if m["label"] != object_label), None)
    return subject, target


def check_comparative_claim(plan, outputs: dict) -> dict | None:
    """'A는 B보다 ~가 많다' 형태의 주장을 교란 제거 후 수치로 확인한다."""
    claim = plan.claim or plan.question
    decompose = outputs.get("decompose_confounders") or {}
    metrics = decompose.get("metrics") or []
    if "보다" not in claim or len(metrics) < 2 or len(plan.areas) < 2:
        return None
    direction = read_claim_direction(claim.split("보다", 1)[1])
    if direction is None:
        return None
    labels = [m["label"] for m in metrics]
    first, second = split_comparison(plan, metrics)
    if first is None or second is None:
        return None
    use_bus = any(w in claim for w in ("버스", "승객", "정류소", "승차"))
    key = "bus_board_window" if use_bus else "window_resident"
    left, right = first.get(key), second.get(key)
    if left is None or right is None or max(left, right) == 0:
        return {"status": "inconclusive", "metric": key, "labels": labels}
    margin = (left - right) / max(left, right)
    if abs(margin) < 0.03:
        status = "inconclusive"
    elif (margin > 0 and direction > 0) or (margin < 0 and direction < 0):
        status = "confirmed"
    else:
        status = "refuted"
    return {"status": status, "metric": key, "first": first["label"], "second": second["label"],
            "first_value": left, "second_value": right, "margin": round(margin, 4)}


def check_slice_claim(plan, outputs: dict) -> dict | None:
    """'X는 장날 오전에 평소보다 붐빈다' 같은 단일 지역 조건부 주장을 은폐 구간 결과로 확인한다."""
    claim = plan.claim or plan.question
    if not any(w in claim for w in ("평소보다", "평균보다", "보통보다")):
        return None
    slices = (outputs.get("excavate_masked_slice") or {}).get("slices") or []
    if plan.areas:
        # 주장의 주어(첫 지역)에 속한 조각만 근거로 쓴다(다른 지역의 과부하로 오경보를 내지 않도록).
        slices = [s for s in slices if s.get("unit") in (None, plan.areas[0])]
    wanted = []
    if "장날" in claim or "오일장" in claim:
        wanted.append("장날")
    if "비" in claim or "우천" in claim:
        wanted.append("비 오는")
    if "주말" in claim:
        wanted.append("주말")
    for label in ("월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"):
        if label in claim or label[0] + "요일" in claim:
            wanted.append(label)
    hits = [s for s in slices if any(w in s["day_group"] for w in wanted)] if wanted else slices
    positive = read_claim_direction(claim.split("보다", 1)[-1]) != -1
    if hits and positive:
        return {"status": "confirmed", "slice": hits[0]}
    if hits and not positive:
        return {"status": "refuted", "slice": hits[0]}
    return {"status": "inconclusive"}


# ---------------------------------------------------------------- 판정

def decide_verdict(plan, outputs: dict) -> tuple[str, dict]:
    if plan.needs_abstain:
        return "abstain", {}
    if plan.intent in ("help", "catalog"):
        return "info", {}
    decompose = outputs.get("decompose_confounders") or {}
    excavate = outputs.get("excavate_masked_slice") or {}
    boundary = outputs.get("simulate_reversal_boundary") or {}
    catalog = outputs.get("resolve_spatial_catalog") or {}
    if not plan.areas and not catalog.get("areas") and plan.intent not in ("fact_check",):
        return "insufficient_area", {}
    if plan.intent == "fact_check":
        status = excavate.get("verdict")
        if status in ("unmeasured_cause", "structural_break", "thin_denominator"):
            return status, {}
        comparison = check_comparative_claim(plan, outputs)
        if comparison and comparison["status"] in ("confirmed", "refuted"):
            return comparison["status"], {"claim_check": comparison}
        slice_claim = check_slice_claim(plan, outputs)
        if slice_claim:
            return slice_claim["status"], {"claim_check": slice_claim}
        if status == "masked_overload":
            return "masked_overload", {}
        if comparison:
            return comparison["status"], {"claim_check": comparison}
        causal = excavate.get("causal_audit") or {}
        if causal.get("status") == "correlation_only":
            return "correlation_only", {}
        return "inconclusive", {}
    # 판정을 내려야 할 도구가 실패(예외·입력 오류)했으면 '역전 없음' 같은 기본값으로 덮지 않고 유보한다.
    if plan.intent == "hidden_slice":
        if not excavate.get("verdict"):
            return "inconclusive", {"tool_error": explain_failure(excavate, "excavate_masked_slice")}
        return ("masked_overload" if excavate["verdict"] == "masked_overload" else "none"), {}
    if plan.intent == "boundary":
        if not boundary.get("verdict"):
            return "inconclusive", {"tool_error": explain_failure(boundary, "simulate_reversal_boundary")}
        return boundary["verdict"], {}
    if not decompose.get("verdict"):
        return "inconclusive", {"tool_error": explain_failure(decompose, "decompose_confounders")}
    return decompose["verdict"], {}


def explain_failure(output: dict, name: str) -> str:
    reason = "계산 오류" if output else "실행 결과 없음"
    return f"{TOOL_LABELS.get(name, name)} 도구가 결과를 내지 못했습니다({reason})"


def compose_headline(verdict: str, plan, outputs: dict, extra: dict) -> str:
    decompose = outputs.get("decompose_confounders") or {}
    excavate = outputs.get("excavate_masked_slice") or {}
    boundary = outputs.get("simulate_reversal_boundary") or {}
    slices = excavate.get("slices") or []
    breaks = excavate.get("breaks") or []
    reversal_line = (boundary.get("reversals") or [""])[0]
    if verdict == "abstain":
        return f"판정 보류: {plan.needs_abstain}. 억지 추론 대신 확인 가능한 범위만 안내하고 현장 확인 질문을 드립니다."
    if verdict == "info":
        return "질문 예시: '애월이 아라동보다 살기 활기찬 동네라던데 맞아?', '오일장 장날 몇 시에 가면 덜 붐벼?'"
    if verdict == "insufficient_area":
        return "질문에서 제주 지역명을 찾지 못했습니다. 동네 이름(예: 노형동, 신제주, 삼화지구)을 함께 적어 주세요."
    if verdict == "reversal":
        return (f"단순 평균으로는 {mark_subject(decompose.get('naive_leader'))} 앞서 보이지만, 교란을 걷어낸 도민 실생활 "
                f"기준으로는 {mark_subject(decompose.get('final_leader'))} 앞섭니다(순위 역전).")
    if verdict == "conditional_reversal":
        return (f"평소에는 {mark_subject(decompose.get('final_leader'))} 앞서지만 "
                f"{WEATHER_LABELS.get(decompose.get('stress_weather'), '악천후')}에는 "
                f"{attach_josa(quote(decompose.get('stress_leader')), '으로/로')} 뒤집힙니다.")
    if verdict == "no_reversal":
        line = f"교란을 걷어내도 {mark_subject(decompose.get('final_leader') or decompose.get('naive_leader'))} 계속 앞섭니다."
        if boundary.get("verdict") == "boundary":
            line += f" 다만 {reversal_line}"
        return line
    if verdict == "masked_overload" and slices:
        top = slices[0]
        return (f"평균만 보면 평범하지만 '{top['target']} · {top['day_group']} · {top['window']}'에는 같은 시각 평소의 "
                f"{top['lift']:.2f}배로 붐빕니다(평균에 가려진 과부하).")
    if verdict == "none":
        return "요일·시간대·날씨 조각을 모두 훑었지만 평균에 가려진 뚜렷한 과부하 구간은 없습니다."
    if verdict == "structural_break" and breaks:
        top = breaks[0]
        return (f"{top['month'][:4]}년 {int(top['month'][4:])}월 초에 계단식 집계 단절이 있어, 이 변화를 실제 증감으로 "
                f"단정할 수 없습니다({top['explanation']}).")
    if verdict == "unmeasured_cause":
        base = "수치 변화 자체는 확인되지만, 주장한 원인은 원자료 컬럼으로 측정되지 않아 단정할 수 없습니다."
        if breaks:
            base += f" 또한 {breaks[0]['month'][:4]}-{breaks[0]['month'][4:]} 월초 집계 단절이 감지되었습니다."
        return base
    if verdict == "thin_denominator":
        audit = excavate.get("denominator_audit") or {}
        return (f"주장된 평균은 전체 {audit.get('rows', 0):,}행 중 {audit.get('valid_time_rows', 0):,}행으로만 계산 가능해 "
                f"제주 전체 대표값으로 쓸 수 없습니다(분모 함정).")
    if verdict in ("confirmed", "refuted"):
        check = extra.get("claim_check") or {}
        word = "원자료와 일치합니다(정상 주장)" if verdict == "confirmed" else "원자료와 반대입니다"
        if verdict == "refuted" and decompose.get("verdict") == "reversal":
            word = (f"관광객이 포함된 단순 평균({quote(decompose.get('naive_leader'))} 우위)으로만 맞고, "
                    f"도민 실생활 기준으로는 반대입니다")
        if check.get("first"):
            return (f"주장은 {word}: 교란 제거 후 '{check['first']}' {check['first_value']:,.0f} vs "
                    f"'{check['second']}' {check['second_value']:,.0f}.")
        if check.get("slice"):
            top = check["slice"]
            return (f"주장은 {word}: '{top['target']} · {top['day_group']} · {top['window']}'이 같은 시각 평소의 "
                    f"{top['lift']:.2f}배입니다.")
        return f"주장은 {word}."
    if verdict == "correlation_only":
        return "두 현상이 함께 변했는지는 볼 수 있지만, 인과는 별도 검증 없이 단정할 수 없습니다."
    if verdict == "inconclusive" and extra.get("tool_error"):
        return f"{extra['tool_error']}. 계산되지 않은 결과를 추정하지 않고 판정을 유보합니다."
    check = extra.get("claim_check") or {}
    if verdict == "inconclusive" and check.get("status") == "inconclusive":
        if check.get("first") and check.get("second"):
            return (f"두 지역 차이가 3% 미만이라 어느 쪽이 더 많다고 단정하기 어렵습니다(판정 유보): "
                    f"'{check['first']}' {check['first_value']:,.0f} vs '{check['second']}' {check['second_value']:,.0f}.")
        if check.get("metric"):
            return "비교에 필요한 수치를 충분히 확보하지 못해 판정을 유보합니다."
        return ("주장한 조건(요일·장날·날씨·시간대)에서 '평소보다 붐빈다'는 근거는 원자료에서 확인되지 않습니다. "
                "확인되지 않은 효과를 사실로 인정하지 않습니다.")
    if verdict == "boundary":
        winners = (boundary.get("boundary") or {}).get("winners") or {}
        asked = plan.hour_from
        if asked is not None and str(asked) in {str(k) for k in winners}:
            winner = next(v for k, v in winners.items() if str(k) == str(asked))
            return f"요청하신 {asked}시 기준으로는 {mark_subject(winner)} 유리합니다. {reversal_line}"
        return reversal_line or "조건을 바꾸면 결론이 뒤집히는 경계가 있습니다."
    if verdict == "no_boundary":
        return reversal_line or "검토한 범위에서는 결론이 뒤집히지 않습니다."
    return "확보한 근거로는 어느 쪽으로도 단정하기 어렵습니다(판정 유보)."


def pick_facts(plan, outputs: dict, verdict: str) -> list[str]:
    decompose = outputs.get("decompose_confounders") or {}
    excavate = outputs.get("excavate_masked_slice") or {}
    boundary = outputs.get("simulate_reversal_boundary") or {}
    catalog = outputs.get("resolve_spatial_catalog") or {}
    facts: list[str] = []
    if verdict in ("structural_break", "unmeasured_cause", "thin_denominator", "masked_overload", "correlation_only"):
        facts += excavate.get("facts", [])[:4]
        facts += [f for f in decompose.get("facts", []) if "관광객" in f or "도민 체류" in f][:2]
    elif verdict in ("reversal", "conditional_reversal", "no_reversal", "confirmed", "refuted", "inconclusive"):
        wanted = ("시간당 평균 체류", "관광객) 체류 비중", "도민 체류(", "야간 잔류율", "비 오는 날")
        facts += [f for f in decompose.get("facts", []) if any(w in f for w in wanted)][:6]
        facts += excavate.get("facts", [])[:1]
    elif verdict in ("boundary", "no_boundary"):
        facts += boundary.get("facts", [])[:3]
        if plan.user_group or plan.day_type == "market":
            facts += excavate.get("facts", [])[:1]
        facts += [f for f in decompose.get("facts", []) if "도민 체류(" in f][:2]
    elif verdict == "abstain":
        facts += catalog.get("facts", [])[:3]
    else:
        facts += catalog.get("facts", [])[:2]
    if verdict not in ("abstain", "info") and boundary.get("facts") and verdict not in ("boundary", "no_boundary"):
        facts += boundary["facts"][:1]
    unique = []
    for fact in facts:
        if fact not in unique:
            unique.append(fact)
    return unique[:7]


def pick_reversals(plan, outputs: dict, verdict: str, extra: dict | None = None) -> list[str]:
    if verdict == "abstain":
        return [f"판정 보류(Abstention): {plan.needs_abstain}. 원자료에 없는 값을 추정해 결론 내리지 않습니다."]
    lines: list[str] = []
    decompose = (outputs.get("decompose_confounders") or {}).get("reversals", [])
    excavate = (outputs.get("excavate_masked_slice") or {}).get("reversals", [])
    boundary = (outputs.get("simulate_reversal_boundary") or {}).get("reversals", [])
    if plan.intent in ("compare", "profile", "boundary"):
        # 비교 질문은 순위 역전·귀가 경계를 먼저, 은폐 구간은 가장 강한 1건만 덧붙인다.
        lines = decompose + boundary[:2] + excavate[:1]
    else:
        lines = excavate + decompose + boundary[:1]
    unique = []
    for line in lines:
        if line not in unique:
            unique.append(line)
    if not unique and (extra or {}).get("tool_error"):
        return ["판정 도구가 결과를 내지 못해 교란 제거 후 역전 여부를 판단하지 않았습니다."]
    return unique[:6] or ["교란을 걷어내도 결론이 바뀌는 구간은 확인되지 않았습니다."]


def pick_limitations(plan, outputs: dict, verdict: str) -> list[str]:
    lines = []
    for name in TOOL_ORDER:
        lines += [n for n in (outputs.get(name) or {}).get("relaxation", [])]
    causal = (outputs.get("excavate_masked_slice") or {}).get("causal_audit") or {}
    for assessment in causal.get("assessments", []) or []:
        if not assessment.get("joinable"):
            lines.append(f"'{assessment['cause_word']}' 영향: {assessment['reason']}")
    used = set()
    for name in TOOL_ORDER:
        for row in (outputs.get(name) or {}).get("provenance", []) or []:
            used.add(row.get("table"))
    try:
        notes = load_meta(stamp_db())["limitations"]
    except Exception:
        notes = []
    priority = {"structural_break": "food_waste_monthly", "unmeasured_cause": "food_waste_monthly",
                "thin_denominator": "ev_charging_wait", "boundary": "bus_service_hourly",
                "masked_overload": "bus_stop_hourly"}
    first = priority.get(verdict)
    ordered = sorted(notes, key=lambda n: (n["table_name"] != first, n["table_name"] not in used))
    for note in ordered[:2]:
        lines.append(note["limitation"])
    unique = []
    for line in lines:
        if line not in unique:
            unique.append(line)
    return unique[:4]


def pick_field_question(plan, outputs: dict, verdict: str) -> str:
    """원자료로 끝내 확인할 수 없는 부분을 현장에서 확인하는 질문 정확히 1개."""
    if verdict == "abstain" and plan.abstain_question:
        return plan.abstain_question
    if verdict == "info":
        return "어느 동네의 어떤 생활 고민(출퇴근·외출·이사·민원)인지 알려 주실 수 있을까요?"
    notes: dict[str, str] = {}
    try:
        for note in load_meta(stamp_db())["limitations"]:
            notes.setdefault(note["table_name"], note["field_question"])
    except Exception:
        notes = {}
    excavate = outputs.get("excavate_masked_slice") or {}
    breaks = excavate.get("breaks") or []
    slices = excavate.get("slices") or []
    causal = excavate.get("causal_audit") or {}
    if verdict in ("structural_break", "unmeasured_cause") and breaks:
        table = breaks[0].get("table")
        if notes.get(table):
            return notes[table]
    if verdict == "unmeasured_cause":
        missing = [a for a in causal.get("assessments", []) if not a.get("joinable")]
        if missing:
            cause = missing[0]["cause_word"]
            return (f"{attach_josa(quote(cause), '이/가')} {causal.get('effect_label', '이 변화')}에 영향을 줬는지 보려면 "
                    f"같은 기간의 {cause} 관련 수치를 관계 기관에서 받아 볼 수 있을까요?")
    if verdict == "thin_denominator" and notes.get("ev_charging_wait"):
        return notes["ev_charging_wait"]
    if verdict == "masked_overload" and slices:
        top = slices[0]
        return (f"{top['target'].replace(' 승하차', '')}에서 {top['day_group']} {top['window']}에 버스가 만차로 그냥 "
                f"지나간 적이 있는지 기사님이나 이용자께 확인할 수 있을까요?")
    if plan.topic in ("commute",) or verdict in ("boundary", "no_boundary"):
        boundary = outputs.get("simulate_reversal_boundary") or {}
        if boundary.get("field_question"):
            return boundary["field_question"]
    topic_table = {"spending": "card_time_monthly", "crowd": "bus_stop_hourly", "waste": "food_waste_monthly",
                   "wifi": "wifi_monthly_users", "ev": "ev_charging_wait"}.get(plan.topic, "floating_pop_hourly")
    return notes.get(topic_table) or "이 결론이 실제 생활과 맞는지 주민께 직접 확인할 수 있을까요?"


def pick_followups(plan, outputs: dict, verdict: str) -> list[str]:
    boundary = outputs.get("simulate_reversal_boundary") or {}
    if verdict in ("info", "insufficient_area"):
        return ["애월이 아라동보다 살기 활기찬 동네라던데 맞아?", "오일장 장날 몇 시에 가면 덜 붐벼?",
                "삼화지구 퇴근 버스 괜찮아? 아라동이랑 비교해 줘"]
    if verdict == "abstain":
        area = plan.areas[0] if plan.areas else "노형동"
        return [f"{area} 퇴근 시간 버스 혼잡은 어때?", f"{area} 밤 9시 이후 생활 인구는?", "어떤 데이터를 쓸 수 있어?"]
    if plan.intent == "fact_check":
        area = plan.areas[0] if plan.areas else "한림읍"
        options = [f"{area} 관련 다른 주장도 검증해 줘", "평균에 가려진 혼잡 시간대는 언제야?", "이 결론을 민원 증거 카드로 정리해 줘"]
        return options
    followups = boundary.get("followups") or []
    defaults = ["그럼 퇴근 시간을 밤 9시 이후로 바꾸면?", "비 안 올 때는?", "거기 말고 아라동이랑 비교해 줘"]
    merged = []
    for item in followups + defaults:
        if item not in merged:
            merged.append(item)
    return merged[:3]


def collect_provenance(outputs: dict) -> list[dict]:
    seen, rows = set(), []
    for name in TOOL_ORDER:
        for row in (outputs.get(name) or {}).get("provenance", []) or []:
            key = row.get("table")
            if key and key not in seen:
                seen.add(key)
                rows.append(row)
    return rows


def build_bundle(plan, outputs: dict, mode: str, mode_reason: str = "", tool_sources: dict | None = None) -> dict:
    verdict, extra = decide_verdict(plan, outputs)
    provenance = collect_provenance(outputs)
    synthetic = any(str(p.get("source_type", "")).startswith(("synthetic", "calibrated")) for p in provenance)
    excavate = outputs.get("excavate_masked_slice") or {}
    bundle = {
        "question": plan.question,
        "plan": plan.export_plan(),
        "mode": mode,
        "mode_reason": mode_reason,
        "verdict": verdict,
        "headline": compose_headline(verdict, plan, outputs, extra),
        "facts": pick_facts(plan, outputs, verdict),
        "reversals": pick_reversals(plan, outputs, verdict, extra),
        "limitations": ([extra["tool_error"]] if extra.get("tool_error") else [])
                       + pick_limitations(plan, outputs, verdict)[:4 - bool(extra.get("tool_error"))],
        "field_question": pick_field_question(plan, outputs, verdict),
        "neutral_rewrite": excavate.get("neutral_rewrite"),
        "followups": pick_followups(plan, outputs, verdict),
        "claim_check": extra.get("claim_check"),
        "provenance": provenance,
        "data_notice": SYNTHETIC_NOTICE if synthetic else "원자료(official_raw) 기준",
        "outputs": outputs,
        "tool_runs": [
            {"name": name, "ok": (outputs.get(name) or {}).get("ok", False),
             "elapsed_ms": (outputs.get(name) or {}).get("elapsed_ms"),
             "verdict": (outputs.get(name) or {}).get("verdict"),
             "source": (tool_sources or {}).get(name, mode)}
            for name in TOOL_ORDER if name in outputs
        ],
    }
    if plan.corrections:
        bundle["facts"].insert(0, "지명 자동 교정: " + ", ".join(plan.corrections))
    return bundle


# ---------------------------------------------------------------- 로컬 해설

def compose_narrative(bundle: dict) -> str:
    plan = bundle.get("plan") or {}
    lines = [f"**결론**: {bundle['headline']}", ""]
    if plan.get("inherited"):
        lines.append(f"_이전 대화에서 이어받은 조건: {', '.join(plan['inherited'])}_")
        lines.append("")
    lines.append(SECTION_FACTS)
    lines += [f"- {fact}" for fact in bundle["facts"]] or ["- (확인된 수치 없음)"]
    lines.append("")
    lines.append(SECTION_REVERSAL)
    lines += [f"- {line}" for line in bundle["reversals"]]
    if bundle.get("neutral_rewrite"):
        lines.append(f"- 중립 교정: {bundle['neutral_rewrite']}")
    lines.append("")
    lines.append(SECTION_LIMIT)
    lines += [f"- 한계: {line}" for line in bundle["limitations"][:3]]
    lines.append(f"- 현장 확인 질문: {bundle['field_question']}")
    lines.append("")
    lines.append(f"_데이터: {bundle['data_notice']}_")
    return "\n".join(lines)


def stream_narrative(text: str):
    """문장 단위로 잘라 흘려보낸다(지연 없이 즉시)."""
    for piece in re.split(r"(?<=\n)", text):
        if piece:
            yield piece


QUESTION_LINE = re.compile(r"현장\s*확인\s*질문\s*[:：]")


def ensure_sections(narrative: str, bundle: dict) -> tuple[str, str]:
    """Claude 해설이 3구획 + 현장 확인 질문 정확히 1개 형식을 지키도록 빠진 것만 보완한다.

    반환: (보완된 전체 해설, 화면에 이어 붙일 추가분 - 없으면 빈 문자열)
    """
    text = (narrative or "").rstrip()
    lines = text.split("\n")
    question_rows = [i for i, line in enumerate(lines) if QUESTION_LINE.search(line)]
    if len(question_rows) > 1:
        text = "\n".join(line for i, line in enumerate(lines) if i not in question_rows[1:])
    blocks = {
        SECTION_FACTS: [f"- {fact}" for fact in bundle.get("facts", [])] or ["- (확인된 수치 없음)"],
        SECTION_REVERSAL: [f"- {line}" for line in bundle.get("reversals", [])],
        SECTION_LIMIT: [f"- 한계: {line}" for line in bundle.get("limitations", [])[:2]],
    }
    additions = ["\n".join([header] + body) for header, body in blocks.items() if header not in text]
    if not question_rows:
        additions.append(f"- 현장 확인 질문: {bundle['field_question']}")
    appended = ""
    if additions:
        appended = ("\n\n---\n_형식 보완: 빠진 근거 구획만 로컬 엔진이 같은 도구 결과로 덧붙였습니다._\n"
                    + "\n\n".join(additions))
        text += appended
    return text, appended
