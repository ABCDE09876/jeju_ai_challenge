"""Primary Brain: Anthropic Claude API 브릿지.

- 모델·추론 강도 고정: claude-opus-5-5 + output_config.effort="xhigh" + adaptive thinking(요약 표시).
- 비용 상한 장치 없음: 호출 횟수 제한·토큰 축소·저가 모델 대체를 하지 않는다. max_tokens 는 API 필수값이라
  모델 최대 출력(128,000)을 넣는다. temperature 는 Opus 5.5 가 허용하지 않으므로 보내지 않는다.
- 멀티턴 Tool Use 루프: stop_reason == "tool_use" 인 동안 모든 tool_use 블록을 로컬 도구로 실행하고
  {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(...)} 목록을 user 메시지로 돌려준다.
- 가용성 판정: 첫 스트림 이벤트가 3.5초 안에 오지 않거나 인증·네트워크·서버 오류, 거절(refusal)이면
  ClaudeUnavailable 을 던져 오케스트레이터가 0.1초 안에 로컬 엔진으로 전환하게 한다.
- 대화 이력은 추가만 한다(append-only). 앞선 턴을 고치지 않으므로 사고 블록 서명이 유지된다.
"""
from __future__ import annotations

import queue
import threading
import time

from src.agent.prompts import SYSTEM_PROMPT
from src.config import (CLAUDE_EFFORT, CLAUDE_MAX_TOKENS, CLAUDE_MODEL, CLAUDE_THINKING, FIRST_EVENT_TIMEOUT_S,
                        STREAM_IDLE_TIMEOUT_S)
from src.tools.tool_registry import list_claude_tools, run_tool, serialize_output

REASON_LABELS = {
    "no_key": "API Key 미설정",
    "forced_offline": "강제 로컬 모드",
    "auth_error": "API Key 인증 실패",
    "permission": "API 권한 없음",
    "model_unavailable": "모델 접근 불가",
    "rate_limited": "요청 한도 초과(429)",
    "network": "네트워크 연결 실패",
    "timeout": "응답 지연(3.5초 내 첫 응답 없음)",
    "idle_timeout": "스트림 중단",
    "server_error": "Claude 서버 오류",
    "bad_request": "요청 형식 오류",
    "refusal": "Claude 안전 정책상 응답 거절",
    "max_tokens": "출력 길이 한도 도달",
    "sdk_missing": "anthropic 패키지 미설치",
    "unexpected": "예상하지 못한 오류",
}


class ClaudeUnavailable(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail

    def explain(self) -> str:
        return REASON_LABELS.get(self.reason, self.reason)


def classify_error(exc: BaseException) -> str:
    try:
        import anthropic
    except ImportError:
        return "sdk_missing"
    checks = [
        (anthropic.AuthenticationError, "auth_error"),
        (anthropic.PermissionDeniedError, "permission"),
        (anthropic.NotFoundError, "model_unavailable"),
        (anthropic.RateLimitError, "rate_limited"),
        (anthropic.BadRequestError, "bad_request"),
        (anthropic.APITimeoutError, "timeout"),
        (anthropic.APIConnectionError, "network"),
        (anthropic.APIStatusError, "server_error"),
    ]
    for kind, reason in checks:
        if isinstance(exc, kind):
            return reason
    if isinstance(exc, (ConnectionError, OSError, TimeoutError)):
        return "network"
    return "unexpected"


class ClaudeBridge:
    def __init__(self, api_key: str, client=None):
        self.api_key = (api_key or "").strip()
        self.client = client
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}

    def build_client(self):
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise ClaudeUnavailable("no_key")
        try:
            import anthropic
        except ImportError as exc:
            raise ClaudeUnavailable("sdk_missing", str(exc)) from exc
        self.client = anthropic.Anthropic(
            api_key=self.api_key, max_retries=0,
            timeout=anthropic.Timeout(STREAM_IDLE_TIMEOUT_S, connect=FIRST_EVENT_TIMEOUT_S),
        )
        return self.client

    @staticmethod
    def build_request(messages: list) -> dict:
        """모든 요청이 지나가는 단일 지점: 모델·effort·thinking 을 여기서 고정한다."""
        return {
            "model": CLAUDE_MODEL,
            "max_tokens": CLAUDE_MAX_TOKENS,
            "thinking": dict(CLAUDE_THINKING),
            "output_config": {"effort": CLAUDE_EFFORT},
            "system": SYSTEM_PROMPT,
            "tools": list_claude_tools(),
            "messages": messages,
        }

    def stream_once(self, messages: list):
        """한 번의 요청을 별도 스레드에서 스트리밍하고 이벤트를 넘긴다. 반환값: 최종 Message."""
        client = self.build_client()
        request = self.build_request(messages)
        inbox: queue.Queue = queue.Queue()
        cancel = threading.Event()
        live_stream: dict = {}

        def pump_stream():
            try:
                with client.messages.stream(**request) as stream:
                    live_stream["active"] = stream
                    for event in stream:
                        if cancel.is_set():
                            # 화면이 떠났거나(재실행·중단) 시간 초과로 버려진 응답: with 블록을 빠져나가며 연결을 닫는다.
                            return
                        inbox.put(("event", event))
                    inbox.put(("final", stream.get_final_message()))
            except BaseException as exc:  # 스레드 밖으로 전달해 분류
                inbox.put(("error", exc))

        threading.Thread(target=pump_stream, daemon=True, name="claude-stream").start()
        self.usage["requests"] += 1
        try:
            try:
                kind, item = inbox.get(timeout=FIRST_EVENT_TIMEOUT_S)
            except queue.Empty:
                raise ClaudeUnavailable("timeout", f"{FIRST_EVENT_TIMEOUT_S}s 내 첫 이벤트 없음") from None
            while True:
                if kind == "error":
                    raise ClaudeUnavailable(classify_error(item), f"{type(item).__name__}: {item}") from item
                if kind == "final":
                    usage = getattr(item, "usage", None)
                    if usage is not None:
                        self.usage["input_tokens"] += int(getattr(usage, "input_tokens", 0) or 0)
                        self.usage["output_tokens"] += int(getattr(usage, "output_tokens", 0) or 0)
                    return item
                yield item
                try:
                    kind, item = inbox.get(timeout=STREAM_IDLE_TIMEOUT_S)
                except queue.Empty:
                    raise ClaudeUnavailable("idle_timeout") from None
        finally:
            # 정상 종료·예외·소비자 이탈(GeneratorExit, Streamlit 재실행) 어느 경우든 백그라운드 스트림을 멈춘다.
            cancel.set()
            closer = getattr(live_stream.get("active"), "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass

    def request_round(self, messages: list, round_no: int):
        """스트림 이벤트를 화면용 이벤트로 바꿔 내보내고 최종 Message 를 돌려준다."""
        final = yield from self.relay_events(self.stream_once(messages), round_no)
        return final

    @staticmethod
    def relay_events(source, round_no: int):
        while True:
            try:
                event = next(source)
            except StopIteration as stop:
                return stop.value
            kind = getattr(event, "type", "")
            if kind == "content_block_start":
                block = getattr(event, "content_block", None)
                if getattr(block, "type", "") == "tool_use":
                    yield {"kind": "tool_start", "name": getattr(block, "name", ""), "round": round_no}
            elif kind == "content_block_delta":
                delta = getattr(event, "delta", None)
                delta_type = getattr(delta, "type", "")
                if delta_type == "thinking_delta" and getattr(delta, "thinking", ""):
                    yield {"kind": "thinking", "text": delta.thinking, "round": round_no}
                elif delta_type == "text_delta" and getattr(delta, "text", ""):
                    yield {"kind": "text", "text": delta.text, "round": round_no}

    def run_turn(self, user_content: str, transcript: list):
        """한 사용자 턴 전체(도구 루프 포함). 마지막에 {'kind': 'done', ...} 이벤트로 새 메시지들을 넘긴다."""
        messages = list(transcript) + [{"role": "user", "content": user_content}]
        added = [messages[-1]]
        round_no = 1
        tool_runs = []
        started = time.perf_counter()
        response = yield from self.request_round(messages, round_no)
        yield {"kind": "round_end", "round": round_no, "stop_reason": response.stop_reason}
        while response.stop_reason in ("tool_use", "pause_turn"):
            assistant_turn = {"role": "assistant", "content": response.content}
            messages.append(assistant_turn)
            added.append(assistant_turn)
            if response.stop_reason == "tool_use":
                replies = []
                for block in response.content:
                    if getattr(block, "type", "") != "tool_use":
                        continue
                    output = run_tool(block.name, block.input)
                    tool_runs.append({"name": block.name, "input": block.input, "output": output, "source": "claude"})
                    yield {"kind": "tool_result", "name": block.name, "input": block.input, "output": output}
                    reply = {"type": "tool_result", "tool_use_id": block.id, "content": serialize_output(output)}
                    if not output.get("ok", True):
                        reply["is_error"] = True
                    replies.append(reply)
                user_turn = {"role": "user", "content": replies}
                messages.append(user_turn)
                added.append(user_turn)
            round_no += 1
            response = yield from self.request_round(messages, round_no)
            yield {"kind": "round_end", "round": round_no, "stop_reason": response.stop_reason}
        if response.stop_reason == "refusal":
            raise ClaudeUnavailable("refusal", str(getattr(response, "stop_details", "")))
        if response.stop_reason == "max_tokens":
            raise ClaudeUnavailable("max_tokens")
        final_turn = {"role": "assistant", "content": response.content}
        added.append(final_turn)
        final_text = "".join(getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text")
        yield {"kind": "done", "messages": added, "final_text": final_text, "tool_runs": tool_runs,
               "rounds": round_no, "seconds": round(time.perf_counter() - started, 1), "usage": dict(self.usage),
               "model": getattr(response, "model", CLAUDE_MODEL)}


def probe_connection(api_key: str, client=None) -> tuple[bool, str]:
    """사이드바 '연결 확인' 버튼용: 짧은 요청 한 번으로 키·네트워크 상태를 본다(같은 고정 모델·effort)."""
    bridge = ClaudeBridge(api_key, client)
    try:
        events = bridge.run_turn("연결 확인입니다. 도구 없이 '연결됨' 한 단어로만 답하세요.", [])
        for event in events:
            if event.get("kind") == "done":
                return True, f"{event.get('model')} 응답 확인"
    except ClaudeUnavailable as exc:
        return False, exc.explain()
    return False, REASON_LABELS["unexpected"]
