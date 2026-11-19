"""50가지 가혹한 엣지케이스 퍼징 스트레스 테스트(런타임 에러 0건 검증).

범주: 지명 오타·초성·줄임말 / 극단적으로 좁은 조건(N=0 유도) / 이상 입력(빈 값·이모지·SQL 주입·초장문) /
      영문 혼용 / 5턴 연속 후속 질문 / 깨진·이상 CSV / API 장애(잘못된 키·네트워크 차단·첫 응답 지연·
      거절·서버 오류·도구 예외 주입) / 판정 보류·미지원 지역·복합 주장.
통과 조건: 예외가 밖으로 새지 않고, 결론·세 구획(수치 사실/역전 구간/한계)·현장 확인 질문 1개를 갖춘 답이 나온다.
API 장애 문항은 로컬 엔진으로 자동 전환(mode=local)되어야 통과.

python scripts/run_fuzz_stress_test.py            # 50문항(오프라인 + 장애 모의)
python scripts/run_fuzz_stress_test.py --live 5   # 추가로 실제 API Key 로 연결 상태 5문항 실행
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.claude_bridge import ClaudeBridge  # noqa: E402
from src.agent.orchestrator import HybridOrchestrator  # noqa: E402
from src.agent.response_cache import ResponseCache  # noqa: E402
from src.config import PROCESSED_DIR, SAMPLES_DIR, read_api_key  # noqa: E402
from src.tools import tool_registry  # noqa: E402
from src.tools.analysis_cubes import warm_cubes  # noqa: E402
from src.tools.csv_auditor import audit_uploaded_csv  # noqa: E402

REPORT = PROCESSED_DIR / "fuzz_report.json"

CHAT_CASES = [
    ("typo", "노형똥 퇴근 버스 어때?"), ("typo", "아라ㄷㅗㅇ이랑 노형 비교해줘"), ("typo", "신졔주 구졔주 비교해줘"),
    ("typo", "ㅅㅈㅈ ㄱㅈㅈ 저녁에 어디가 나아"), ("typo", "삼화지꾸 퇴근길 괜찮아?"), ("typo", "애월이랑 아라 비교"),
    ("typo", "중뮨동 사람 많아?"), ("typo", "일도일동이랑 이도일동 밤에 비교"), ("typo", "서귀포 원도심 저녁 생활 어때"),
    ("typo", "영교 출퇴근 버스 괜찮아?"),
    ("narrow", "우도면 새벽 3시 비 오는 장날 어르신 버스 붐벼?"), ("narrow", "추자면 토요일 새벽 4시 비 오는 날 유동인구 비교해줘"),
    ("narrow", "한경면 금요일 23시 비 오는 장날 청소년 많아?"), ("narrow", "노형동 0시부터 1시 사이 생활 인구는?"),
    ("narrow", "삼화지구 99시 이후 귀가하면?"), ("narrow", "아라동 25시에 버스 있어?"), ("narrow", "중문동 -3시 생활 인구"),
    ("narrow", "표선면 월요일 새벽 2시 비 안 올 때 비교해줘"),
    ("weird", ""), ("weird", "     "), ("weird", "😀😀😀🍊🍊"), ("weird", "'; DROP TABLE regions; --"),
    ("weird", "가" * 5000), ("weird", "1234567890"), ("weird", "제주 " * 300), ("weird", "<script>alert(1)</script> 노형동"),
    ("english", "Compare Nohyeong and Ara for commuting"), ("english", "노형동 vs 연동 which is better at night?"),
    ("abstain", "노형동 층간소음 심해?"), ("abstain", "2023년 아라동 버스 이용객은?"), ("abstain", "중문동 치안 괜찮아?"),
    ("unknown", "서울 강남역 퇴근 버스 어때?"),
    ("compound", "관광객 때문에 한림읍 쓰레기가 늘고 전기차 때문에 버스가 줄었다는 기사 맞아?"),
]
MULTI_TURN = ["삼화지구 퇴근 버스 어때?", "그럼 8시 이후는?", "비 오는 날에는?", "거기 말고 노형동이랑 비교해 줘", "주말엔?"]


def build_csv_cases() -> list[tuple[str, bytes]]:
    hallim = (SAMPLES_DIR / "hallim_food_waste_daily_cp949.csv").read_bytes()
    wide = ",".join(f"c{i}" for i in range(200)) + "\n" + ",".join(str(i) for i in range(200)) + "\n"
    return [
        ("cp949 깨짐 위험 표본", hallim),
        ("cp949 중간 절단", hallim[: len(hallim) // 2 + 1]),
        ("모든 값 결측", "기준일자,배출량\n2020-01-01,\n2020-01-02,NULL\n2020-01-03,***\n".encode("utf-8")),
        ("헤더만", "기준일자,배출량\n".encode("utf-8")),
        ("빈 파일", b""),
        ("바이너리 쓰레기", bytes(range(256)) * 20),
        ("세미콜론 구분", "date;value\n2020-01-01;1,000\n2020-02-01;3,000\n".encode("utf-8")),
        ("파이프 구분+중복 컬럼", "a|a|b\n1|2|***\n3|4|5\n".encode("utf-8")),
        ("latin1 문자", "name,value\ncafé,1\nnaïve,2\n".encode("latin1")),
        ("200열 와이드", wide.encode("utf-8")),
    ]


class FakeStream:
    def __init__(self, mode: str):
        self.mode = mode

    def __enter__(self):
        if self.mode == "server_error":
            import anthropic
            import httpx2
            request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
            raise anthropic.InternalServerError("overloaded", response=httpx2.Response(529, request=request), body=None)
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        if self.mode == "slow":
            time.sleep(6)
        yield SimpleNamespace(type="message_start")

    def get_final_message(self):
        return SimpleNamespace(stop_reason="refusal", content=[], usage=None, stop_details="mock", model="claude-opus-5-5")


class FakeClient:
    def __init__(self, mode: str):
        self.messages = SimpleNamespace(stream=lambda **kwargs: FakeStream(mode))


def check_bundle(bundle: dict | None) -> str | None:
    if not bundle:
        return "답변 묶음 없음"
    for key in ("verdict", "headline", "field_question"):
        if not bundle.get(key):
            return f"{key} 누락"
    for key in ("facts", "reversals", "limitations"):
        if not isinstance(bundle.get(key), list):
            return f"{key} 구획 누락"
    if not bundle.get("narrative"):
        return "해설 누락"
    return None


def run_chat(question: str, orchestrator: HybridOrchestrator | None = None) -> tuple[dict | None, float]:
    orchestrator = orchestrator or HybridOrchestrator("life", force_offline=True, cache=ResponseCache(enabled=False))
    started = time.perf_counter()
    bundle = orchestrator.answer(question)
    return bundle, (time.perf_counter() - started) * 1000


def run_case(case_id: str, category: str, label: str, check) -> dict:
    started = time.perf_counter()
    try:
        outcome = check()
        problem = outcome.get("problem")
        passed = problem is None
    except Exception as exc:  # 테스트 대상이 예외를 밖으로 던지면 실패
        outcome = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=4)}
        passed, problem = False, outcome["error"]
    return {"id": case_id, "category": category, "input": label[:80], "passed": passed, "problem": problem,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
            **{k: v for k, v in outcome.items() if k not in ("problem",)}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", type=int, default=0, help="실제 API Key 로 추가 실행할 연결 상태 문항 수")
    args = parser.parse_args()
    warm_cubes()
    rows = []
    number = 0

    for category, question in CHAT_CASES:
        number += 1

        def run_check(q=question):
            bundle, _ = run_chat(q)
            return {"verdict": bundle.get("verdict") if bundle else None, "mode": bundle.get("mode") if bundle else None,
                    "problem": check_bundle(bundle)}
        rows.append(run_case(f"F{number:02d}", category, question or "(빈 입력)", run_check))

    number += 1

    def run_dialogue():
        orchestrator = HybridOrchestrator("life", force_offline=True, cache=ResponseCache(enabled=False))
        verdicts = []
        for question in MULTI_TURN:
            bundle, _ = run_chat(question, orchestrator)
            problem = check_bundle(bundle)
            if problem:
                return {"problem": f"'{question}': {problem}"}
            verdicts.append(bundle["verdict"])
        inherited = orchestrator.memory.plans[-1].inherited
        return {"verdict": verdicts, "inherited_last": inherited, "problem": None if inherited else "문맥 계승 실패"}
    rows.append(run_case(f"F{number:02d}", "multiturn", " -> ".join(MULTI_TURN), run_dialogue))

    for label, payload in build_csv_cases():
        number += 1

        def audit_sample(p=payload):
            report = audit_uploaded_csv(io.BytesIO(p), "2019년 12월에 10월보다 58.69배 늘었다. 관광객 때문이다.", label)
            problem = None if report.get("verdict") and report.get("field_question") else "감사 결과 형식 누락"
            return {"verdict": report.get("verdict"), "problem": problem}
        rows.append(run_case(f"F{number:02d}", "csv", label, audit_sample))

    def build_fallback(bridge=None, api_key="", expect_ms: float | None = None):
        def run_check():
            orchestrator = HybridOrchestrator("life", api_key=api_key, bridge=bridge, cache=ResponseCache(enabled=False))
            bundle, elapsed = run_chat("노형동이랑 아라동 퇴근 후 생활 비교해줘", orchestrator)
            problem = check_bundle(bundle)
            if not problem and bundle.get("mode") != "local":
                problem = f"로컬 폴백 미발동(mode={bundle.get('mode')})"
            if not problem and expect_ms and elapsed > expect_ms:
                problem = f"폴백 지연 {elapsed:.0f}ms > {expect_ms:.0f}ms"
            return {"verdict": bundle.get("verdict") if bundle else None, "mode": bundle.get("mode") if bundle else None,
                    "mode_reason": bundle.get("mode_reason") if bundle else None, "fallback_ms": round(elapsed, 1),
                    "problem": problem}
        return run_check

    import anthropic
    api_cases = [
        ("잘못된 API Key(실제 401 응답)", build_fallback(api_key="sk-ant-invalid-fuzz-key", expect_ms=8000)),
        ("네트워크 강제 차단(닫힌 포트)", build_fallback(bridge=ClaudeBridge("sk-ant-fuzz", client=anthropic.Anthropic(
            api_key="sk-ant-fuzz", base_url="http://127.0.0.1:9", max_retries=0,
            timeout=anthropic.Timeout(5.0, connect=3.5))), expect_ms=6000)),
        ("첫 응답 3.5초 초과(지연 모의)", build_fallback(bridge=ClaudeBridge("sk-ant-fuzz", client=FakeClient("slow")),
                                                expect_ms=4800)),
        ("Claude 거절(refusal) 모의", build_fallback(bridge=ClaudeBridge("sk-ant-fuzz", client=FakeClient("refusal")),
                                              expect_ms=3000)),
        ("서버 과부하 529 모의", build_fallback(bridge=ClaudeBridge("sk-ant-fuzz", client=FakeClient("server_error")),
                                         expect_ms=3000)),
    ]
    for label, check in api_cases:
        number += 1
        rows.append(run_case(f"F{number:02d}", "api_failure", label, check))

    number += 1
    original = tool_registry.TOOL_FUNCTIONS["decompose_confounders"]

    def raise_fault(**kwargs):
        raise RuntimeError("fuzz: 도구 내부 예외 주입")

    def inject_fault():
        tool_registry.TOOL_FUNCTIONS["decompose_confounders"] = raise_fault
        try:
            bundle, _ = run_chat("애월이랑 아라동 비교해줘")
        finally:
            tool_registry.TOOL_FUNCTIONS["decompose_confounders"] = original
        failed = [r for r in bundle.get("tool_runs", []) if not r.get("ok")]
        return {"verdict": bundle.get("verdict"), "failed_tools": [r["name"] for r in failed],
                "problem": check_bundle(bundle)}
    rows.append(run_case(f"F{number:02d}", "api_failure", "도구 예외 주입(decompose_confounders)", inject_fault))

    live_rows = []
    key = read_api_key()
    if args.live and key:
        live_questions = ["노형똥 퇴근 버스 어때?", "우도면 새벽 3시 비 오는 장날 어르신 버스 붐벼?", "'; DROP TABLE regions; --",
                          "전기차 충전 대기시간이 평균 20분이라는 보도, 믿어도 돼?", "작년 겨울 노형동 밤 버스 붐볐어?"][: args.live]
        for index, question in enumerate(live_questions, start=1):
            def ask_live(q=question):
                orchestrator = HybridOrchestrator("life", api_key=key, cache=ResponseCache(enabled=False))
                bundle, elapsed = run_chat(q, orchestrator)
                return {"verdict": bundle.get("verdict"), "mode": bundle.get("mode"),
                        "mode_reason": bundle.get("mode_reason"), "seconds": round(elapsed / 1000, 1),
                        "number_audit": bundle.get("number_audit"), "problem": check_bundle(bundle)}
            live_rows.append(run_case(f"L{index:02d}", "live_api", question, ask_live))

    passed = sum(1 for r in rows if r["passed"])
    summary = {"total": len(rows), "passed": passed, "failed": len(rows) - passed,
               "pass_rate": round(passed / len(rows), 4), "text": f"{passed} / {len(rows)}",
               "runtime_errors": sum(1 for r in rows if r.get("error")),
               "by_category": {}}
    for row in rows:
        bucket = summary["by_category"].setdefault(row["category"], {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(row["passed"])
    if live_rows:
        live_passed = sum(1 for r in live_rows if r["passed"])
        summary["live_api"] = {"text": f"{live_passed} / {len(live_rows)}",
                               "modes": [r.get("mode") for r in live_rows]}
    report = {"generated_by": "scripts/run_fuzz_stress_test.py", "summary": summary, "cases": rows, "live_cases": live_rows}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"퍼징 결과: {summary['text']} 통과 (런타임 에러 {summary['runtime_errors']}건)")
    for category, bucket in summary["by_category"].items():
        print(f"  {category:12s} {bucket['passed']}/{bucket['total']}")
    for row in rows:
        if not row["passed"]:
            print(f"  [실패] {row['id']} {row['category']} {row['input'][:40]} -> {row['problem']}")
    for row in rows:
        if row["category"] == "api_failure":
            print(f"  [폴백] {row['input']}: mode={row.get('mode')} {row.get('fallback_ms', '')}ms {row.get('mode_reason', '')}")
    if live_rows:
        print(f"연결 상태 실호출: {summary['live_api']['text']} 통과, 모드 {summary['live_api']['modes']}")
        for row in live_rows:
            print(f"  [live] {row['input'][:30]} -> {row.get('verdict')} {row.get('mode')} {row.get('seconds')}s "
                  f"audit={row.get('number_audit')} {row.get('problem') or ''}")
    print(f"저장: {REPORT.relative_to(ROOT)}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
