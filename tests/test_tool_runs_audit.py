"""Claude 가 같은 도구를 여러 번 부르거나 앞 질문의 도구 값을 다시 인용할 때의 답변 묶음·수치 감사(네트워크 없이 모의 클라이언트).

2026-11-19 실제 연결 40문항 시험에서 찾은 결함의 회귀 시험:
- 같은 도구를 조건을 바꿔 두 번 부르면(평소·비 오는 날) 마지막 호출만 남아, 증거 카드·판정이 질문 조건이 아닌 변형 조건을
  따랐고(귀가 경계 20시 대신 18시), 앞 호출에서 온 실제 도구 값이 '근거 없는 수치'로 표시됐다.
- 후속 질문에서 앞 질문의 도구 값을 다시 인용해도 '근거 없는 수치'로 표시됐다.
Claude 가 직접 지어낸 값은 여전히 근거 없는 수치로 남아야 한다.
"""
from types import SimpleNamespace

from src.agent.claude_bridge import ClaudeBridge
from src.agent.intent_parser import parse_question
from src.agent.number_auditor import audit_numbers, collect_numbers, expand_forms, is_grounded
from src.agent.orchestrator import HybridOrchestrator, collect_evidence, pick_primary, plan_tool_calls
from src.agent.response_cache import ResponseCache
from src.tools.tool_registry import run_tool

QUESTION = "삼화지구로 이사 가면 퇴근 버스 괜찮을까? 아라동이랑 비교해 줘"


class ScriptedStream:
    def __init__(self, final):
        self.final = final

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        yield SimpleNamespace(type="message_start")
        for block in self.final.content:
            if block.type == "text":
                yield SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=block.text))
            if block.type == "tool_use":
                yield SimpleNamespace(type="content_block_start", content_block=block)

    def get_final_message(self):
        return self.final


class ScriptedClient:
    def __init__(self, finals):
        self.finals = list(finals)
        self.messages = SimpleNamespace(stream=lambda **request: ScriptedStream(self.finals.pop(0)))


def make_message(blocks, stop_reason):
    return SimpleNamespace(content=blocks, stop_reason=stop_reason, usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                           model="claude-opus-5-5", stop_details=None)


def pick_unique(own, other) -> float:
    """own 에만 있고 other 의 어떤 값(반올림·퍼센트 꼴 포함)으로도 설명되지 않는 소수 하나."""
    mine, theirs = set(), set()
    collect_numbers(own, mine)
    collect_numbers(other, theirs)
    forms = expand_forms(theirs)
    for value in sorted(mine):
        if value != int(value) and 1 < value < 100 and not is_grounded(value, forms):
            return value
    raise AssertionError("두 결과를 가르는 숫자가 없음")


def build_variants():
    args = dict(plan_tool_calls(parse_question(QUESTION, "life")))["simulate_reversal_boundary"]
    return args, {**args, "weather": "rain"}


def test_repeated_calls():
    usual, rain = build_variants()
    first, second = run_tool("simulate_reversal_boundary", usual), run_tool("simulate_reversal_boundary", rain)
    assert first["boundary"]["boundary"] != second["boundary"]["boundary"]  # 평소와 비 오는 날의 귀가 경계가 다르다
    only_second = pick_unique(second, first)
    blocks = [SimpleNamespace(type="tool_use", id="toolu_01", name="simulate_reversal_boundary", input=usual),
              SimpleNamespace(type="tool_use", id="toolu_02", name="simulate_reversal_boundary", input=rain)]
    text = SimpleNamespace(type="text", text=f"**결론**: 비 오는 날에는 {only_second}분입니다. 지어낸 값은 987.6분입니다.")
    client = ScriptedClient([make_message(blocks, "tool_use"), make_message([text], "end_turn")])
    orchestrator = HybridOrchestrator("life", bridge=ClaudeBridge("sk-ant-test", client=client), cache=ResponseCache(enabled=False))
    bundle = orchestrator.answer(QUESTION)
    assert bundle["mode"] == "claude"
    # 증거 카드·판정은 질문 조건인 첫 호출(평소)을 따른다
    assert bundle["outputs"]["simulate_reversal_boundary"]["boundary"]["boundary"] == first["boundary"]["boundary"]
    assert "도구 2회" in bundle["mode_reason"] and [call["ok"] for call in bundle["claude"]["tool_calls"]] == [True, True]
    # 둘째 호출의 값은 첫 호출만 보면 근거가 없지만, 이번 질문의 모든 호출을 근거로 보면 근거가 있다. 지어낸 값만 남는다
    assert audit_numbers(f"{only_second}분", collect_evidence(bundle["outputs"], bundle))["ungrounded"]
    assert bundle["number_audit"]["ungrounded"] == ["987.6"]


def test_failed_call():
    outputs = {}
    pick_primary(outputs, "simulate_reversal_boundary", {"ok": False, "error": "인자 오류"})
    pick_primary(outputs, "simulate_reversal_boundary", {"ok": True, "boundary": {"boundary": 20}})
    pick_primary(outputs, "simulate_reversal_boundary", {"ok": True, "boundary": {"boundary": 18}})
    assert outputs["simulate_reversal_boundary"] == {"ok": True, "boundary": {"boundary": 20}}


def test_earlier_exact():
    # 이번 질문의 도구 값은 0.5% 상대 오차까지, 앞 질문의 도구 값은 반올림 꼴만 인정한다(지어낸 큰 수가 우연히 맞지 않게)
    assert audit_numbers("876.5분", {"tools": {"wait": 872.5}})["ungrounded"] == []
    assert audit_numbers("876.5분", {}, exact=[{"wait": 872.5}])["ungrounded"] == ["876.5"]
    assert audit_numbers("872.5분, 61.9%", {}, exact=[{"wait": 872.48, "lift": 1.619}])["ungrounded"] == []
    # 감소를 말로 쓰고 숫자는 양수로 적어도('38.6% 적습니다', '4.8% 줄어듭니다') 도구의 음수 변화율과 같은 값으로 본다
    assert audit_numbers("승차가 38.6% 적고 도민이 4.8% 줄어듭니다", {"tools": {"change": -38.6, "ratio": -0.0483}})["ungrounded"] == []


def test_earlier_turns():
    _, rain = build_variants()
    first_turn = run_tool("simulate_reversal_boundary", rain)
    offline = HybridOrchestrator("life", force_offline=True, cache=ResponseCache(enabled=False))
    offline.answer(QUESTION)
    followup = offline.answer("주말엔?")  # 둘째 턴에 로컬 도구가 채울 결과(같은 대화 문맥)
    value = pick_unique(first_turn, followup["outputs"])
    block = SimpleNamespace(type="tool_use", id="toolu_01", name="simulate_reversal_boundary", input=rain)
    turns = [make_message([block], "tool_use"), make_message([SimpleNamespace(type="text", text="**결론**: 첫 답")], "end_turn"),
             make_message([SimpleNamespace(type="text", text=f"**결론**: 앞서 본 비 오는 날 {value}분보다 덜합니다. 지어낸 값 876.5분.")],
                          "end_turn")]
    orchestrator = HybridOrchestrator("life", bridge=ClaudeBridge("sk-ant-test", client=ScriptedClient(turns)),
                                      cache=ResponseCache(enabled=False))
    orchestrator.answer(QUESTION)
    bundle = orchestrator.answer("주말엔?")
    assert bundle["mode"] == "claude"
    assert audit_numbers(f"{value}분", collect_evidence(bundle["outputs"], bundle))["ungrounded"]  # 이번 턴 결과만으로는 근거 없음
    assert bundle["number_audit"]["ungrounded"] == ["876.5"]
    orchestrator.reset()
    assert orchestrator.earlier_tools == []
