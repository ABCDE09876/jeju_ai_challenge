"""제주의 두 얼굴 (Jeju True-Life & Citizen Lawyer Agent) - Streamlit 앱.

실행: streamlit run src/app.py
"""
from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.agent.claude_bridge import probe_connection  # noqa: E402
from src.agent.orchestrator import HybridOrchestrator  # noqa: E402
from src.config import PROCESSED_DIR, SAMPLES_DIR, describe_model_lock, load_env, read_api_key  # noqa: E402
from src.tools.analysis_cubes import warm_cubes  # noqa: E402
from src.tools.csv_auditor import audit_uploaded_csv  # noqa: E402
from src.tools.db_connector import ensure_database, fetch_frame  # noqa: E402
from src.tools.tool_registry import TOOL_LABELS  # noqa: E402
from src.ui.chart_style import SERIES, STATUS, apply_korean_font, style_figure  # noqa: E402
from src.ui.components import render_dashboard, render_followups, show_badge  # noqa: E402
from src.ui.evidence_card import build_card_html, describe_verdict  # noqa: E402

st.set_page_config(page_title="제주의 두 얼굴", layout="wide")
apply_korean_font()
load_env()

SCENARIOS = {
    "life": [
        ("관광객 거품", "애월이 아라동보다 살기 활기찬 동네라던데, 도민 기준으로도 맞아?"),
        ("야간 공동화", "동문시장 쪽 일도1동이랑 이도1동 중에 밤 9시 이후 생활하기 어디가 나아?"),
        ("이사·귀가 경계", "삼화지구로 이사 가면 퇴근 버스 괜찮을까? 아라동이랑 비교해 줘"),
        ("우천 은폐 혼잡", "비 오는 평일 퇴근길에 삼화지구 정류소가 언제 붐벼?"),
        ("장날 방문 시각", "오일장 장날 몇 시에 가면 덜 붐벼? 어르신 모시고 가려고"),
        ("생활권 비교", "신제주랑 구제주 중 퇴근 후 저녁 생활 어디가 나아?"),
    ],
    "lawyer": [
        ("집계 단절", "한림읍 음식물쓰레기가 2019년 12월에 10월보다 58.69배 폭증했다. 관광객 때문이다."),
        ("집계 단위 전환", "제주 공공와이파이 이용자가 2020년 10월에 반토막 났다던데 맞아?"),
        ("평균 은폐 민원", "도두동 오일장 정류소는 평균 이용객이 적어서 버스 증편이 필요 없다는 주장, 반박할 근거 있어?"),
        ("미측정 인과", "버스 이용이 줄어든 건 전기차 보급 때문이라는 기사 맞아?"),
        ("분모 함정", "전기차 충전 대기시간은 평균 17분이라 짧다는 기사 맞아?"),
        ("정상 주장 대조군", "노형동은 저녁에 도민 유동인구가 연동보다 많다는데 맞아?"),
    ],
}
SAMPLES = {
    "한림읍 음식물쓰레기 일별(CP949·천단위 콤마·마스킹)": (
        "hallim_food_waste_daily_cp949.csv", "한림읍 음식물쓰레기 배출량이 2019년 12월에 10월보다 58.69배 폭증했다. 관광객 때문이다."),
    "전기차 충전 기록 2,447행(UTF-8 BOM)": ("ev_charging_wait_2447.csv", "제주 전기차 충전 대기시간은 평균 17분이다."),
    "와이파이 월별 이용자(EUC-KR·탭 구분)": (
        "wifi_monthly_users_tab_euckr.csv", "와이파이 이용자가 2020년 10월에 0.55배로 줄었다. 관광객이 줄었기 때문이다."),
    "버스 노선 정보(엑셀 날짜 자동변환 오염)": ("bus_route_info_mangled.csv", ""),
}
ENGINE_NAMES = {"claude": "Claude Opus 5.5 (effort xhigh)", "local": "로컬 신경-기호 엔진"}
ERROR_LOG = PROCESSED_DIR / "app_errors.log"


def log_error(exc: BaseException) -> None:
    """원시 오류는 화면 대신 로컬 로그 파일에만 남긴다."""
    try:
        ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ERROR_LOG.open("a", encoding="utf-8") as log_file:
            log_file.write(f"[{datetime.now().isoformat(timespec='seconds')}] {type(exc).__name__}: {exc}\n"
                         f"{traceback.format_exc()}\n")
    except OSError:
        pass


def boot() -> None:
    if st.session_state.get("booted"):
        return
    try:
        with st.spinner("최초 실행 준비 중: 로컬 DB 확인·구축과 메모리 인덱스 적재(처음 한 번만, 약 10~40초)"):
            ensure_database()
            warm_cubes()
    except Exception as exc:  # DB 구축 실패도 원시 오류 대신 안내로 보여 준다
        log_error(exc)
        st.error("로컬 데이터베이스를 준비하지 못했습니다. 터미널에서 `python scripts/build_jeju_db.py` 를 실행한 뒤 "
                 "새로고침해 주세요(상세 내용: data/processed/app_errors.log).")
        st.stop()
    st.session_state["booted"] = True


def get_orchestrator(workspace: str) -> HybridOrchestrator:
    key = f"orch_{workspace}"
    if key not in st.session_state:
        st.session_state[key] = HybridOrchestrator(workspace)
    return st.session_state[key]


def render_sidebar() -> tuple[str, bool]:
    with st.sidebar:
        st.header("브레인 컨트롤러")
        typed = st.text_input("Anthropic API Key (선택)", type="password", key="api_key_input",
                              placeholder="비워 두어도 100% 로컬 모드로 작동합니다")
        offline = st.toggle("강제 로컬 오프라인 모드 테스트", value=False, key="force_offline",
                            help="켜면 API Key 가 있어도 외부 호출 없이 로컬 엔진만 씁니다(두 모드 비교용).")
        st.caption(f"Claude 고정 정책: {describe_model_lock()}")
        st.caption("호출 횟수·토큰 상한 없음. 3.5초 무응답·오류·거절 시 로컬 엔진으로 자동 전환.")
        api_key = read_api_key(typed)
        if st.button("Claude 연결 확인", width="stretch"):
            if offline or not api_key:
                st.warning("강제 로컬 모드이거나 API Key 가 없어 확인을 건너뛰었습니다.")
            else:
                with st.spinner("Claude 응답 확인 중"):
                    ok, message = probe_connection(api_key)
                (st.success if ok else st.warning)(f"{'연결됨' if ok else '연결 실패'} · {message}")
        st.divider()
        st.subheader("데이터 상태")
        try:
            registry = fetch_frame("SELECT source_type, COUNT(*) AS tables, SUM(row_count) AS rows FROM table_registry "
                                   "GROUP BY source_type ORDER BY rows DESC")
            st.dataframe(registry, hide_index=True, width="stretch")
        except Exception as exc:
            log_error(exc)
            st.caption("DB 상태를 읽지 못했습니다(자동 재구축 대기).")
        stats = get_orchestrator("life").cache.report_stats()
        st.caption(f"스마트 캐시: {stats['entries']}건 저장 · 재사용 {stats['hits']}회")
        if st.button("대화 초기화", width="stretch"):
            for workspace in ("life", "lawyer"):
                get_orchestrator(workspace).reset()
                st.session_state[f"history_{workspace}"] = []
            st.rerun()
    return api_key, offline


def run_question(orchestrator: HybridOrchestrator, workspace: str, question: str) -> None:
    history = st.session_state.setdefault(f"history_{workspace}", [])
    history.append({"question": question, "bundle": None})  # 답변 도중 화면이 다시 그려져도 질문이 사라지지 않게
    with st.chat_message("user"):
        st.markdown(question if question.strip() else "_(빈 질문)_")
    with st.chat_message("assistant"):
        status = st.status("에이전트 분석 중: Step 1 -> Step 4", expanded=True)
        holder: dict = {}
        thinking = {"text": "", "box": None, "shown": 0}
        started = time.perf_counter()

        def stream():
            try:
                for event in orchestrator.stream_answer(question):
                    kind = event["kind"]
                    if kind == "step":
                        status.markdown(f"- **{event['label']}** · {event.get('detail', '')}")
                    elif kind == "mode":
                        status.markdown(f"- 엔진: **{ENGINE_NAMES.get(event['mode'], event['mode'])}** · {event['text']}")
                    elif kind == "thinking":
                        thinking["text"] += event["text"]
                        if thinking["box"] is None:
                            thinking["box"] = status.empty()
                        if len(thinking["text"]) - thinking["shown"] >= 160:
                            thinking["shown"] = len(thinking["text"])
                            thinking["box"].caption("Claude 추론 요약: " + thinking["text"][-500:])
                    elif kind == "tool":
                        output = event["output"]
                        status.markdown(f"  - 완료: {TOOL_LABELS.get(event['name'], event['name'])} "
                                        f"({output.get('elapsed_ms', 0)} ms · 판정 {output.get('verdict', '-')} · "
                                        f"{event.get('source')})")
                    elif kind == "text":
                        yield event["text"]
                    elif kind == "final":
                        holder["bundle"] = event["bundle"]
            except Exception as exc:  # 어떤 경우에도 원시 오류 화면을 띄우지 않는다
                log_error(exc)
                holder["error"] = exc

        st.write_stream(stream())
        bundle = holder.get("bundle")
        if bundle is None:
            backup = HybridOrchestrator(workspace, force_offline=True)
            try:
                bundle = backup.answer(question)
                st.markdown(bundle.get("narrative", ""))
            except Exception as exc:
                log_error(exc)
                status.update(label="답변을 만들지 못했습니다", state="error", expanded=False)
                st.warning("일시적인 문제로 답변을 만들지 못했습니다. 질문을 조금 바꿔 다시 시도해 주세요.")
                return
        seconds = time.perf_counter() - started
        status.update(label=f"분석 완료 · {seconds:.1f}초 · {ENGINE_NAMES.get(bundle.get('mode'), bundle.get('mode'))}",
                      state="complete", expanded=False)
        index = len(history) - 1
        history[index] = {"question": question, "bundle": bundle}
        render_dashboard(bundle, key=f"{workspace}_{index}")
        render_followups(bundle, f"{workspace}_{index}", workspace)


def render_workspace(workspace: str, api_key: str, offline: bool, intro: str, placeholder: str) -> None:
    orchestrator = get_orchestrator(workspace)
    orchestrator.configure(api_key, offline)
    st.caption(intro)
    st.markdown("**원클릭 시연 시나리오**")
    rows = [SCENARIOS[workspace][:3], SCENARIOS[workspace][3:]]
    for row_no, row in enumerate(rows):
        columns = st.columns(3)
        for col_no, (column, (label, question)) in enumerate(zip(columns, row)):
            if column.button(f"{label}", key=f"scenario_{workspace}_{row_no}_{col_no}", help=question, width="stretch"):
                st.session_state[f"pending_{workspace}"] = question
    history = st.session_state.setdefault(f"history_{workspace}", [])
    for index, turn in enumerate(history):
        with st.chat_message("user"):
            st.markdown(turn["question"] if turn["question"].strip() else "_(빈 질문)_")
        with st.chat_message("assistant"):
            if turn["bundle"] is None:
                st.caption("이 질문의 답변은 화면 이동으로 중단되었습니다. 같은 질문을 다시 보내 주세요.")
                continue
            st.markdown(turn["bundle"].get("narrative", ""))
            render_dashboard(turn["bundle"], key=f"{workspace}_{index}")
            if index == len(history) - 1:
                render_followups(turn["bundle"], f"{workspace}_{index}", workspace)
    typed = st.chat_input(placeholder, key=f"chat_{workspace}")
    question = typed if typed is not None else st.session_state.pop(f"pending_{workspace}", None)
    if question is not None:
        run_question(orchestrator, workspace, question)


def build_csv_bundle(report: dict, claim: str) -> dict:
    label, _ = describe_verdict(report.get("verdict", ""))
    headline = {
        "structural_break": "월 경계에서 계단식 집계 단절이 보여, 이 변화를 실제 증감으로 단정할 수 없습니다.",
        "unmeasured_cause": "수치 변화는 재계산으로 확인되지만, 주장한 원인을 측정한 컬럼이 파일에 없습니다.",
        "thin_denominator": "평균이 일부 행으로만 계산되어 대표값으로 쓸 수 없습니다(분모 함정).",
        "figure_mismatch": "주장 속 수치가 재계산 결과와 다릅니다.",
        "no_issue": "재계산·단절·인과 감사에서 이상이 발견되지 않았습니다.",
        "unreadable": "파일을 표로 해석하지 못했습니다.",
    }.get(report.get("verdict", ""), label)
    schema = report.get("schema_match") or {}
    return {
        "question": claim or "(주장 문장 없음)", "verdict": report.get("verdict", ""), "headline": headline,
        "facts": report.get("facts", []), "reversals": report.get("reversals", []),
        "limitations": report.get("limitations", []), "field_question": report.get("field_question", ""),
        "neutral_rewrite": report.get("neutral_rewrite"), "mode": "local", "mode_reason": "CSV 실시간 감사",
        "provenance": [{"table": report.get("file", "업로드 파일"), "source_type": "user_upload",
                        "dataset_id": schema.get("sheet", "공식 시트 대응 없음"),
                        "reference_date": datetime.now().strftime("%Y-%m-%d")}],
        "data_notice": "사용자가 올린 파일 기준 재계산", "number_audit": {}, "outputs": {},
    }


def render_csv_auditor() -> None:
    st.caption("아무 CSV 나 끌어 놓고 주장 문장을 쓰면 인코딩을 자동 복구한 뒤 ① 수치 재계산 ② 월 경계 집계 단절 탐지 "
               "③ 컬럼 스키마 대조 인과 과장 교정을 한 번에 수행합니다.")
    left, right = st.columns([1, 1])
    with left:
        uploaded = st.file_uploader("CSV 파일 끌어 놓기(UTF-8, CP949, EUC-KR, 탭·세미콜론·파이프 구분 모두 가능)",
                                    type=["csv", "txt", "tsv"], key="csv_upload")
        choice = st.selectbox("또는 내장 표본으로 시연", ["(선택 안 함)"] + list(SAMPLES), key="csv_sample")
    default_claim = SAMPLES[choice][1] if choice in SAMPLES else ""
    with right:
        claim = st.text_area("검증할 주장 문장(선택)", value=default_claim, key=f"claim_{choice}", height=110)
        run = st.button("감사 실행", type="primary", width="stretch")
    if run:
        if uploaded is not None:
            source, name = uploaded.getvalue(), uploaded.name
        elif choice in SAMPLES:
            path = SAMPLES_DIR / SAMPLES[choice][0]
            if not path.exists():
                ensure_database()
            source, name = path.read_bytes() if path.exists() else b"", path.name
        else:
            st.warning("CSV 파일을 올리거나 내장 표본을 선택해 주세요.")
            return
        with st.spinner("인코딩 복구·재계산·단절 탐지 중"):
            try:
                st.session_state["csv_report"] = (audit_uploaded_csv(source, claim, name), claim)
            except Exception as exc:
                log_error(exc)
                st.warning("파일을 감사하는 중 문제가 생겼습니다. 다른 형식으로 저장해 다시 올려 주세요.")
                return
    stored = st.session_state.get("csv_report")
    if not stored:
        return
    report, claim_used = stored
    bundle = build_csv_bundle(report, claim_used)
    label, color = describe_verdict(bundle["verdict"])
    show_badge(label, color)
    st.markdown(f"**{bundle['headline']}**")
    a, b, c = st.columns([1, 1.2, 1])
    with a:
        st.markdown("**[원자료로 확인된 수치 사실]**")
        st.markdown("\n".join(f"- {line}" for line in bundle["facts"]))
    with b:
        st.markdown("**[교란 제거 후 역전 구간]**")
        st.markdown("\n".join(f"- {line}" for line in bundle["reversals"]))
        monthly = (report.get("chart") or {}).get("monthly") or []
        if monthly:
            figure = go.Figure(go.Scatter(x=[m["month"] for m in monthly], y=[m["value"] for m in monthly],
                                          mode="lines+markers", line={"color": SERIES[0], "width": 2},
                                          marker={"size": 8}, name=report.get("value_column") or "값"))
            for jump in report.get("monthly_jumps", []):
                figure.add_vline(x=jump["to"], line_dash="dot", line_color=STATUS["critical"])
            st.plotly_chart(style_figure(figure, "월별 재계산(점선: 계단식 점프)", 260), width="stretch", key="csv_chart")
    with c:
        st.markdown("**[원자료 컬럼 부재로 인해 단정할 수 없는 한계 및 현장 확인 질문 1개]**")
        st.markdown("\n".join(f"- 한계: {line}" for line in bundle["limitations"]))
        st.markdown(f"- 현장 확인 질문: {bundle['field_question']}")
        if bundle.get("neutral_rewrite"):
            st.markdown(f"> **중립 교정문** {bundle['neutral_rewrite']}")
        st.download_button("1페이지 증거 카드 내려받기(HTML)", build_card_html(bundle), file_name="csv_evidence_card.html",
                           mime="text/html", key="csv_card", width="stretch")
    with st.expander("복구 내역·스키마 대조·재계산 상세"):
        st.write({k: report.get(k) for k in ("file", "encoding", "delimiter", "rows", "columns", "repair_notes",
                                             "schema_match", "durations", "claim_checks", "month_edges",
                                             "monthly_jumps", "registered_changes")})


def main() -> None:
    boot()
    api_key, offline = render_sidebar()
    life_orchestrator = get_orchestrator("life")
    life_orchestrator.configure(api_key, offline)
    brain = life_orchestrator.report_status()
    st.title("제주의 두 얼굴")
    st.markdown("관광객·기상·평균의 3대 허수를 벗겨내는 **도민 실생활 교차검증 및 시민 데이터 변호사 에이전트**")
    show_badge(brain["badge"], "#6250d6" if brain["mode"] == "hybrid" else STATUS["good"])
    st.caption(f"현재 엔진: {brain['reason']}")
    tabs = st.tabs(["제주의 두 얼굴 (도민 실생활·이사·외출 교차검증)",
                    "평균의 거짓말, 도민의 증거 (시민 데이터 변호사 & 통계 팩트체커)",
                    "새 제주 CSV 업로드 실시간 감사기"])
    with tabs[0]:
        render_workspace("life", api_key, offline,
                         "이사·출퇴근·외출 고민을 물어보면 관광객·시간대·날씨 교란을 걷어낸 도민 기준 결론을 드립니다.",
                         "예: 노형동이랑 아라동 중 퇴근 후 생활하기 어디가 나아?")
    with tabs[1]:
        render_workspace("lawyer", api_key, offline,
                         "기사·주장·민원 문장을 넣으면 평균 은폐, 집계 단절, 미측정 인과, 분모 함정을 감사합니다.",
                         "예: OO동은 관광객 때문에 쓰레기가 늘었다는데 맞아?")
    with tabs[2]:
        render_csv_auditor()


main()
