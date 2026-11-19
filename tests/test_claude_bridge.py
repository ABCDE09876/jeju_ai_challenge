"""Claude 브릿지: 모델·effort 고정, 비용 상한 부재, Tool Use 루프 규약, 자동 폴백(네트워크 없이 모의 클라이언트)."""
import json
import time
from types import SimpleNamespace

import pytest

from src.agent.claude_bridge import ClaudeBridge, ClaudeUnavailable
from src.agent.orchestrator import HybridOrchestrator
from src.agent.response_cache import ResponseCache
from src.config import CLAUDE_MAX_TOKENS


class ScriptedStream:
    """라운드별로 미리 정한 이벤트·최종 메시지를 돌려주는 모의 스트림."""

    def __init__(self, final, delay=0.0, fault=None):
        self.final, self.delay, self.fault = final, delay, fault

    def __enter__(self):
        if self.fault:
            raise self.fault
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        if self.delay:
            time.sleep(self.delay)
        yield SimpleNamespace(type="message_start")
        for block in self.final.content:
            if block.type == "text":
                yield SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=block.text))
            if block.type == "tool_use":
                yield SimpleNamespace(type="content_block_start", content_block=block)

    def get_final_message(self):
        return self.final


class ScriptedClient:
    def __init__(self, finals, delay=0.0, fault=None):
        self.finals, self.delay, self.fault, self.requests = list(finals), delay, fault, []
        self.messages = SimpleNamespace(stream=self.open_stream)

    def open_stream(self, **request):
        self.requests.append(request)
        final = self.finals.pop(0) if self.finals else self.finals
        return ScriptedStream(final, self.delay, self.fault)


def make_message(blocks, stop_reason):
    return SimpleNamespace(content=blocks, stop_reason=stop_reason, usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                           model="claude-opus-5-5", stop_details=None)


def test_model_lock(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
    request = ClaudeBridge.build_request([{"role": "user", "content": "x"}])
    assert request["model"] == "claude-opus-5-5"
    assert request["output_config"] == {"effort": "xhigh"}
    assert request["thinking"]["type"] == "adaptive"
    assert request["max_tokens"] == CLAUDE_MAX_TOKENS == 128000
    assert "temperature" not in request and "top_p" not in request


def test_tool_loop():
    tool_block = SimpleNamespace(type="tool_use", id="toolu_01", name="decompose_confounders",
                                 input={"areas": ["애월읍", "아라동"], "topic": "living"})
    text_block = SimpleNamespace(type="text", text="**결론**: 테스트 해설")
    client = ScriptedClient([make_message([tool_block], "tool_use"), make_message([text_block], "end_turn")])
    bridge = ClaudeBridge("sk-ant-test", client=client)
    events = list(bridge.run_turn("애월이랑 아라동 비교", []))
    done = events[-1]
    assert done["kind"] == "done" and done["final_text"] == "**결론**: 테스트 해설"
    assert [e["name"] for e in events if e["kind"] == "tool_result"] == ["decompose_confounders"]
    second_request = client.requests[1]["messages"]
    reply = second_request[-1]["content"][0]
    assert second_request[-1]["role"] == "user" and reply["type"] == "tool_result"
    assert reply["tool_use_id"] == "toolu_01"
    assert json.loads(reply["content"])["verdict"] == "reversal"
    assert all(r["model"] == "claude-opus-5-5" and r["output_config"]["effort"] == "xhigh" for r in client.requests)


def test_timeout_fallback():
    slow = ScriptedClient([make_message([SimpleNamespace(type="text", text="늦은 답")], "end_turn")], delay=6.0)
    orchestrator = HybridOrchestrator("life", bridge=ClaudeBridge("sk-ant-test", client=slow),
                                      cache=ResponseCache(enabled=False))
    started = time.perf_counter()
    bundle = orchestrator.answer("노형동이랑 아라동 비교해줘")
    assert time.perf_counter() - started < 5.0
    assert bundle["mode"] == "local" and "3.5초" in bundle["mode_reason"]


@pytest.mark.parametrize("fault_name", ["AuthenticationError", "RateLimitError", "InternalServerError"])
def test_error_fallback(fault_name):
    import anthropic
    import httpx2

    status = {"AuthenticationError": 401, "RateLimitError": 429, "InternalServerError": 529}[fault_name]
    response = httpx2.Response(status, request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages"))
    fault = getattr(anthropic, fault_name)("mock", response=response, body=None)
    client = ScriptedClient([], fault=fault)
    orchestrator = HybridOrchestrator("life", bridge=ClaudeBridge("sk-ant-test", client=client),
                                      cache=ResponseCache(enabled=False))
    bundle = orchestrator.answer("애월이 아라동보다 활기차다던데 맞아?")
    assert bundle["mode"] == "local" and bundle["verdict"] in ("refuted", "reversal")


def test_refusal_fallback():
    client = ScriptedClient([make_message([], "refusal")])
    with pytest.raises(ClaudeUnavailable) as caught:
        list(ClaudeBridge("sk-ant-test", client=client).run_turn("x", []))
    assert caught.value.reason == "refusal"


def test_missing_key():
    orchestrator = HybridOrchestrator("life", api_key="", cache=ResponseCache(enabled=False))
    assert orchestrator.report_status()["mode"] == "local"
    assert orchestrator.answer("노형동 저녁 생활 어때?")["mode"] == "local"
