"""하이브리드 Auto-Fallback 오케스트레이터.

[Primary]  Claude API(claude-opus-5-5, effort xhigh 고정)가 의도·슬롯 해석, 4대 도구 연쇄 호출, 한국어 해설을 맡는다.
[Core]     모든 수치는 로컬 파이썬 도구(SQLite + pandas)가 계산한다. Claude 에게는 도구 결과의 숫자만 인용하게 하고,
           해설 속 숫자는 number_auditor 가 다시 대조해 근거 없는 숫자를 표시한다.
[Fallback] 키 미설정·강제 오프라인·인증/네트워크 오류·429·3.5초 무응답·거절이면 화면 오류 없이 즉시
           로컬 신경-기호 엔진(자모 퍼지 파서 + 문맥 추적 + 결정론적 ReAct 상태 머신 + 스트리밍 합성기)으로 전환한다.
이벤트(dict)를 순서대로 내보내므로 Streamlit·CLI·테스트가 같은 흐름을 쓴다.
"""
from __future__ import annotations

import time

from src.agent.claude_bridge import ClaudeBridge, ClaudeUnavailable, REASON_LABELS
from src.agent.dialogue_memory import DialogueMemory
from src.agent.grounded_synthesizer import build_bundle, compose_narrative, ensure_sections, stream_narrative
from src.agent.intent_parser import QueryPlan, merge_dialogue_context, parse_question
from src.agent.number_auditor import audit_numbers
from src.agent.prompts import build_user_turn
from src.agent.response_cache import ResponseCache
from src.config import CLAUDE_EFFORT, CLAUDE_MODEL
from src.tools.tool_registry import TOOL_LABELS, run_tool

HYBRID_BADGE = "🟣 하이브리드 부스터 모드 가동 중 (Anthropic Claude API 추론 + 로컬 정밀 연산 | 오프라인 자동 폴백 대기)"
LOCAL_BADGE = "🟢 100% 로컬 온프레미스 자립 모드 가동 중 (외부 API 비용 0원 | 센터 PC 완벽 호환)"
LIVE_TEXT_THRESHOLD = 120


def plan_tool_calls(plan: QueryPlan) -> list[tuple[str, dict]]:
    """결정론적 ReAct 상태 머신: 의도별 도구 호출 순서와 인자."""
    base = {"areas": plan.areas, "topic": plan.topic}
    calls = [("resolve_spatial_catalog", {**base, "question": plan.question})]
    if plan.intent in ("help", "catalog") or plan.needs_abstain:
        return calls
    day = plan.day_type if plan.day_type != "any" else None
    decompose = ("decompose_confounders", {**base, "hour_from": plan.hour_from, "hour_to": plan.hour_to,
                                           "weather": plan.weather, "day_type": plan.day_type})
    excavate = ("excavate_masked_slice", {**base, "claim": plan.claim, "mode": "all", "user_group": plan.user_group})
    variable = plan.variable
    if variable is None:
        variable = "visit_hour" if (plan.topic == "crowd" or plan.day_type == "market") else "return_hour"
    market_words = ("장날", "오일장", "민속오일장")
    market = variable == "visit_hour" and any(word in plan.question for word in market_words)
    boundary = ("simulate_reversal_boundary", {**base, "variable": variable, "weather": plan.weather,
                                               "day_type": day or ("market" if market else "weekday"),
                                               "user_group": plan.user_group})
    if plan.intent == "fact_check":
        calls += [excavate]
        if plan.areas:
            calls += [decompose]
        calls += [boundary]
    elif plan.intent == "hidden_slice":
        calls += [excavate, decompose, boundary]
    else:
        calls += [decompose, excavate, boundary]
    return calls


EARLIER_TURNS = 8  # 수치 감사 근거로 함께 보는 같은 대화의 앞 질문 수(앞 답의 도구 값을 다시 인용하는 후속 질문용)


def collect_evidence(outputs: dict, bundle: dict, runs: list | None = None) -> dict:
    """수치 감사의 근거: 도구 결과 + 답변 한계 구획에 넣은 문구(도구의 완화 내역과 DB limitation_notes 의 자료 설명).

    한계 문구는 자료에서 온 고정 설명이므로 그 안의 숫자('113행, 전체 2,447행의 4.6%')도 근거로 인정한다.
    Claude 가 같은 도구를 조건을 바꿔 여러 번 부르면 모든 호출 결과(runs)도 근거로 본다(로컬 도구가 계산한 값).
    같은 대화의 앞 질문 도구 결과는 audit_numbers 의 exact 로 따로 넘겨 반올림 꼴만 인정한다.
    """
    return {"tools": outputs, "limitations": bundle.get("limitations", []),
            "runs": [run.get("output") for run in runs or []]}


def pick_primary(outputs: dict, name: str, output) -> None:
    """같은 도구를 여러 번 부르면 첫 성공 호출(질문 조건)을 답변 묶음에 쓴다. 앞 호출이 실패했으면 다음 성공 호출로 바꾼다.

    예: '평소' 와 '비 오는 날' 로 귀가 경계를 두 번 구하면 증거 카드·판정은 질문 조건인 첫 호출을 따른다.
    """
    current = outputs.get(name)
    current_ok = isinstance(current, dict) and current.get("ok", False)
    if current is None or (not current_ok and isinstance(output, dict) and output.get("ok", False)):
        outputs[name] = output


class HybridOrchestrator:
    def __init__(self, workspace: str = "life", api_key: str = "", force_offline: bool = False,
                 bridge: ClaudeBridge | None = None, cache: ResponseCache | None = None):
        self.memory = DialogueMemory(workspace=workspace)
        self.api_key = (api_key or "").strip()
        self.force_offline = force_offline
        self.bridge = bridge
        self.cache = cache or ResponseCache()
        self.last_fallback: str | None = None
        self.streamed_chars = 0
        self.earlier_tools: list = []  # 같은 대화의 앞 질문 도구 결과(수치 감사에서 반올림 꼴만 인정하는 근거)

    # ------------------------------------------------------------ 상태

    def configure(self, api_key: str = "", force_offline: bool = False) -> None:
        key = (api_key or "").strip()
        if key != self.api_key:
            self.bridge = None
        self.api_key = key
        self.force_offline = force_offline

    def report_status(self) -> dict:
        if self.force_offline:
            return {"mode": "local", "badge": LOCAL_BADGE, "reason": REASON_LABELS["forced_offline"]}
        if not self.api_key and self.bridge is None:
            return {"mode": "local", "badge": LOCAL_BADGE, "reason": REASON_LABELS["no_key"]}
        return {"mode": "hybrid", "badge": HYBRID_BADGE, "reason": f"{CLAUDE_MODEL} · effort {CLAUDE_EFFORT}"}

    def reset(self) -> None:
        self.memory.reset()
        self.earlier_tools = []

    # ------------------------------------------------------------ 로컬 경로

    def run_local_tools(self, plan: QueryPlan, done: dict | None = None):
        outputs = dict(done or {})
        for name, args in plan_tool_calls(plan):
            if name in outputs and outputs[name].get("ok", False):
                continue
            yield {"kind": "step", "label": TOOL_LABELS[name], "detail": "로컬 정밀 연산"}
            outputs[name] = run_tool(name, args)
            yield {"kind": "tool", "name": name, "output": outputs[name], "source": "local"}
        return outputs

    def answer_locally(self, plan: QueryPlan, mode_reason: str, done: dict | None = None,
                       sources: dict | None = None):
        cache_key = ResponseCache.make_key(plan.export_plan(), "local")
        cached = self.cache.load(cache_key) if not done else None
        if cached:
            yield {"kind": "step", "label": "스마트 캐시 적중", "detail": "같은 질문·조건의 로컬 결과 재사용"}
            cached["mode_reason"] = f"{mode_reason} · 캐시 재사용"
            for chunk in stream_narrative(cached.get("narrative", "")):
                yield {"kind": "text", "text": chunk}
            return cached
        outputs = yield from self.run_local_tools(plan, done)
        source_map = dict(sources or {})
        for name in outputs:
            source_map.setdefault(name, "local")
        bundle = build_bundle(plan, outputs, "local", mode_reason, source_map)
        narrative = compose_narrative(bundle)
        for chunk in stream_narrative(narrative):
            yield {"kind": "text", "text": chunk}
        bundle["narrative"] = narrative
        bundle["number_audit"] = audit_numbers(narrative, collect_evidence(outputs, bundle), plan.question,
                                               exact=self.earlier_tools)
        self.cache.save(cache_key, plan.export_plan(), "local", bundle)
        return bundle

    # ------------------------------------------------------------ Claude 경로

    def answer_with_claude(self, plan: QueryPlan, question: str):
        if self.bridge is None:
            self.bridge = ClaudeBridge(self.api_key)
        user_turn = build_user_turn(question, plan.export_plan(), self.memory.workspace)
        claude_outputs: dict = {}
        pending: list[str] = []
        live = False
        tool_in_round = False
        final_event = None
        for event in self.bridge.run_turn(user_turn, self.memory.transcript):
            kind = event["kind"]
            if kind == "thinking":
                yield {"kind": "thinking", "text": event["text"]}
            elif kind == "tool_start":
                tool_in_round = True
                if pending and not live:
                    yield {"kind": "thinking", "text": "".join(pending)}
                    pending = []
                yield {"kind": "step", "label": TOOL_LABELS.get(event["name"], event["name"]),
                       "detail": "Claude 가 도구 호출"}
            elif kind == "tool_result":
                pick_primary(claude_outputs, event["name"], event["output"])
                yield {"kind": "tool", "name": event["name"], "output": event["output"], "source": "claude",
                       "input": event.get("input")}
            elif kind == "text":
                if live:
                    self.streamed_chars += len(event["text"])
                    yield {"kind": "text", "text": event["text"]}
                else:
                    pending.append(event["text"])
                    if not tool_in_round and sum(len(p) for p in pending) >= LIVE_TEXT_THRESHOLD:
                        live = True
                        chunk = "".join(pending)
                        self.streamed_chars += len(chunk)
                        yield {"kind": "text", "text": chunk}
                        pending = []
            elif kind == "round_end":
                tool_round = event["stop_reason"] in ("tool_use", "pause_turn")
                if not tool_round and pending:
                    chunk = "".join(pending)
                    self.streamed_chars += len(chunk)
                    yield {"kind": "text", "text": chunk}
                elif pending:
                    yield {"kind": "thinking", "text": "".join(pending)}
                if tool_round and live:
                    yield {"kind": "text", "text": "\n\n"}
                pending = []
                live = False
                tool_in_round = False
            elif kind == "done":
                final_event = event
        return claude_outputs, final_event, user_turn

    # ------------------------------------------------------------ 진입점

    def stream_answer(self, question: str):
        started = time.perf_counter()
        plan = parse_question(question, self.memory.workspace)
        plan = merge_dialogue_context(plan, self.memory.recall_plan())
        detail = f"의도={plan.intent}, 지역={', '.join(plan.areas) or '미지정'}, 주제={plan.topic}"
        if plan.corrections:
            detail += f", 지명 교정 {', '.join(plan.corrections)}"
        if plan.inherited:
            detail += f", 이전 대화에서 이어받음: {', '.join(plan.inherited)}"
        yield {"kind": "step", "label": "의도·지명 해석(자모 퍼지 파서 + 문맥 추적)", "detail": detail}
        status = self.report_status()
        bundle = None
        turn_runs: list = []  # 이번 질문에서 Claude 가 부른 도구 호출 전부(같은 도구 여러 번 포함)
        if status["mode"] == "hybrid":
            yield {"kind": "mode", "mode": "claude", "text": status["reason"]}
            self.streamed_chars = 0
            try:
                claude_outputs, final_event, user_turn = yield from self.answer_with_claude(plan, question)
                if final_event is None or not final_event.get("final_text", "").strip():
                    raise ClaudeUnavailable("unexpected", "빈 응답")
                sources = {name: "claude" for name in claude_outputs}
                outputs = yield from self.run_local_tools(plan, claude_outputs)
                for name in outputs:
                    sources.setdefault(name, "local-fill")
                runs = final_event.get("tool_runs") or []
                bundle = build_bundle(plan, outputs, "claude", f"{final_event.get('model', CLAUDE_MODEL)} · "
                                      f"effort {CLAUDE_EFFORT} · 도구 {len(runs) or len(claude_outputs)}회", sources)
                narrative, appended = ensure_sections(final_event["final_text"], bundle)
                if appended:
                    yield {"kind": "text", "text": appended}
                bundle["narrative"] = narrative
                bundle["claude"] = {"rounds": final_event.get("rounds"), "seconds": final_event.get("seconds"),
                                    "usage": final_event.get("usage"), "model": final_event.get("model"),
                                    "tool_calls": [{"name": run.get("name"), "input": run.get("input"),
                                                    "ok": isinstance(run.get("output"), dict) and run["output"].get("ok", False)}
                                                   for run in runs]}
                bundle["number_audit"] = audit_numbers(narrative, collect_evidence(outputs, bundle, runs), plan.question,
                                                       exact=self.earlier_tools)
                turn_runs = runs
                self.memory.extend_transcript(final_event["messages"])
                self.cache.save(ResponseCache.make_key(plan.export_plan(), "claude"), plan.export_plan(), "claude", bundle)
                self.last_fallback = None
            except ClaudeUnavailable as exc:
                self.last_fallback = exc.explain()
                yield {"kind": "mode", "mode": "local", "text": f"{exc.explain()} -> 로컬 엔진 자동 전환"}
                if self.streamed_chars:
                    yield {"kind": "text", "text": "\n\n---\n_Claude 연결이 중간에 끊겨, 아래부터 로컬 엔진이 같은 도구 "
                                                   "결과로 다시 답합니다(위 내용은 무시해도 됩니다)._\n\n"}
                cached = self.cache.load(ResponseCache.make_key(plan.export_plan(), "claude"))
                if cached and cached.get("narrative"):
                    for chunk in stream_narrative(cached["narrative"]):
                        yield {"kind": "text", "text": chunk}
                    bundle = cached
                    bundle["mode_reason"] = f"{exc.explain()} · 같은 질문의 이전 Claude 해설(캐시) 표시"
                else:
                    bundle = yield from self.answer_locally(plan, f"자동 폴백: {exc.explain()}")
                self.memory.add_local_turn(build_user_turn(question, plan.export_plan(), self.memory.workspace),
                                           bundle.get("narrative", ""))
        else:
            yield {"kind": "mode", "mode": "local", "text": status["reason"]}
            bundle = yield from self.answer_locally(plan, status["reason"])
        bundle["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
        self.memory.remember(plan, bundle)
        turn = {"outputs": bundle.get("outputs") or {}, "runs": [run.get("output") for run in turn_runs]}
        self.earlier_tools = (self.earlier_tools + [turn])[-EARLIER_TURNS:]
        yield {"kind": "final", "bundle": bundle}

    def answer(self, question: str) -> dict:
        """이벤트를 모두 소비하고 최종 묶음만 돌려준다(테스트·벤치마크용)."""
        bundle = None
        for event in self.stream_answer(question):
            if event["kind"] == "final":
                bundle = event["bundle"]
        return bundle
