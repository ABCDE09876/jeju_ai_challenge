"""멀티턴 문맥 메모리.

- plans/bundles : 로컬 엔진용 슬롯 계승(merge_dialogue_context)에 쓰는 이전 질문 계획과 답변 묶음
- transcript    : Claude 대화 이력. 추가만 한다(append-only) - 앞선 턴을 고치면 사고 블록 서명이 무효화된다.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DialogueMemory:
    workspace: str = "life"
    plans: list = field(default_factory=list)
    bundles: list = field(default_factory=list)
    transcript: list = field(default_factory=list)

    def recall_plan(self):
        return self.plans[-1] if self.plans else None

    def remember(self, plan, bundle: dict) -> None:
        self.plans.append(plan)
        self.bundles.append(bundle)

    def extend_transcript(self, messages: list) -> None:
        self.transcript.extend(messages)

    def add_local_turn(self, user_content: str, answer_text: str) -> None:
        """Claude 가 응답하지 못한 턴은 로컬 답변을 평문으로 덧붙여 다음 Claude 턴의 문맥을 잇는다."""
        self.transcript.append({"role": "user", "content": user_content})
        self.transcript.append({"role": "assistant", "content": answer_text or "(로컬 엔진이 답변함)"})

    def reset(self) -> None:
        self.plans.clear()
        self.bundles.clear()
        self.transcript.clear()

