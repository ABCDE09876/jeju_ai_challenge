"""3단 대조 대시보드: [기존 단순 평균의 착시] vs [3대 허수 박리·은폐 구간 발굴 차트] vs [최종 결론·교정문·증거 카드]."""
from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

from src.ui.chart_style import INK, SERIES, STATUS, style_figure
from src.ui.evidence_card import build_card_html, describe_verdict

STAGE_SHORT = {"S0": "단순<br>평균", "S1": "①관광객<br>제거", "S2": "②생활<br>시간대", "S3": "③악천후"}


def show_badge(text: str, color: str) -> None:
    st.markdown(f"<span style='display:inline-block;padding:3px 10px;border-radius:999px;background:{color};"
                f"color:#fff;font-size:12px;font-weight:700'>{html.escape(text)}</span>", unsafe_allow_html=True)


def assign_colors(labels: list[str]) -> dict[str, str]:
    return {label: SERIES[i % len(SERIES)] for i, label in enumerate(labels)}


def build_naive_figure(bundle: dict):
    outputs = bundle.get("outputs") or {}
    decompose = outputs.get("decompose_confounders") or {}
    excavate = outputs.get("excavate_masked_slice") or {}
    stages = decompose.get("stages") or []
    if stages and stages[0].get("values"):
        values = stages[0]["values"]
        colors = assign_colors(list(values))
        figure = go.Figure(go.Bar(x=list(values), y=list(values.values()), marker_color=[colors[k] for k in values],
                                  marker_cornerradius=4, hovertemplate="%{x}<br>시간당 %{y:,.0f}명<extra></extra>"))
        return style_figure(figure, "단순 평균(관광객 포함)", 280), "단순 평균 1위: " + str(stages[0].get("leader"))
    breaks = excavate.get("breaks") or []
    if breaks:
        top = breaks[0]
        figure = go.Figure(go.Bar(x=[top["prev_month"], top["month"]], y=[top["prev_value"], top["value"]],
                                  marker_color=[SERIES[0], SERIES[1]], marker_cornerradius=4,
                                  hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>"))
        return (style_figure(figure, "그대로 읽으면 '폭증/급감'", 280),
                f"{top['prev_month']} -> {top['month']} {top['month_ratio']:.2f}배")
    slices = excavate.get("slices") or []
    if slices:
        top = slices[0]
        figure = go.Figure(go.Bar(x=["전체 기간 시간당 평균"], y=[top["overall_hourly_mean"]], marker_color=[SERIES[0]],
                                  marker_cornerradius=4, hovertemplate="%{y:,.1f}명<extra></extra>"))
        return style_figure(figure, f"{top['target'][:16]} 평균", 280), "평균만 보면 평범해 보입니다"
    return None, ""


def build_agent_figure(bundle: dict):
    outputs = bundle.get("outputs") or {}
    verdict = bundle.get("verdict")
    decompose = outputs.get("decompose_confounders") or {}
    excavate = outputs.get("excavate_masked_slice") or {}
    boundary = outputs.get("simulate_reversal_boundary") or {}
    if verdict in ("masked_overload",) and excavate.get("slices"):
        top = excavate["slices"][0]
        profile = top["profile"]
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=profile["hours"], y=profile["baseline"], name="같은 시각 평소", mode="lines",
                                    line={"color": SERIES[0], "width": 2}))
        figure.add_trace(go.Scatter(x=profile["hours"], y=profile["slice"], name=top["day_group"], mode="lines+markers",
                                    line={"color": SERIES[1], "width": 2}, marker={"size": 8}))
        figure.add_vrect(x0=top["hours"][0] - 0.5, x1=top["hours"][-1] + 0.5, fillcolor=STATUS["serious"], opacity=0.12,
                         line_width=0)
        figure.update_xaxes(title="시")
        return style_figure(figure, f"은폐 구간: {top['target'][:16]}", 300)
    if verdict in ("structural_break", "unmeasured_cause") and excavate.get("breaks"):
        top = excavate["breaks"][0]
        figure = go.Figure(go.Bar(x=["단절 전 3개월 평균", "단절 후 3개월 평균"], y=[top["before_mean"], top["after_mean"]],
                                  marker_color=[SERIES[0], SERIES[1]], marker_cornerradius=4,
                                  hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>"))
        figure.add_annotation(text=top["explanation"][:46], showarrow=False, xref="paper", yref="paper", x=0, y=1.12,
                              xanchor="left", font={"size": 11, "color": INK["secondary"]})
        return style_figure(figure, f"{top['month'][:4]}-{top['month'][4:]} 월초 계단식 단절", 300)
    if verdict in ("boundary", "no_boundary") and boundary.get("curves"):
        figure = go.Figure()
        colors = assign_colors(list(boundary["curves"]))
        for label, curve in boundary["curves"].items():
            figure.add_trace(go.Scatter(x=[c["hour"] for c in curve], y=[c["discomfort_min"] for c in curve], name=label,
                                        mode="lines+markers", line={"color": colors[label], "width": 2},
                                        marker={"size": 8}, hovertemplate="%{x}시<br>%{y:.1f}분<extra>" + label + "</extra>"))
        point = (boundary.get("boundary") or {}).get("boundary")
        if point is not None:
            figure.add_vline(x=point, line_dash="dot", line_color=STATUS["critical"])
        figure.update_xaxes(title="시")
        figure.update_yaxes(title="불편 지수(분)")
        return style_figure(figure, "역전 경계(불편 지수)", 300)
    stages = decompose.get("stages") or []
    if stages:
        labels = list(stages[0].get("values", {}))
        colors = assign_colors(labels)
        figure = go.Figure()
        for label in labels:
            figure.add_trace(go.Bar(x=[STAGE_SHORT.get(s["stage"], s["stage"]) for s in stages],
                                    y=[s["values"].get(label) for s in stages], name=label,
                                    marker_color=colors[label], marker_cornerradius=4,
                                    hovertemplate="%{x}<br>시간당 %{y:,.0f}명<extra>" + label + "</extra>"))
        figure.update_layout(barmode="group")
        return style_figure(figure, "3대 허수 단계별 순위", 300)
    audit = excavate.get("denominator_audit") or {}
    if audit:
        figure = go.Figure(go.Bar(x=["전체 기록", "미충전", "대기시간 계산 가능"],
                                  y=[audit["rows"], audit["not_charged"], audit["valid_time_rows"]],
                                  marker_color=[SERIES[0], STATUS["warning"], SERIES[1]], marker_cornerradius=4))
        return style_figure(figure, "분모 함정: 계산 가능한 행", 300)
    return None


def render_dashboard(bundle: dict, key: str) -> None:
    left, middle, right = st.columns([1, 1.25, 1.1])
    with left:
        st.markdown("**[기존 단순 평균의 착시]**")
        figure, caption = build_naive_figure(bundle)
        if figure is not None:
            st.plotly_chart(figure, width="stretch", key=f"naive_{key}")
            st.caption(caption)
        else:
            st.caption("이 질문은 단순 평균 비교 대상이 없습니다.")
    with middle:
        st.markdown("**[에이전트의 3대 허수 박리·은폐 구간 발굴 차트]**")
        figure = build_agent_figure(bundle)
        if figure is not None:
            st.plotly_chart(figure, width="stretch", key=f"agent_{key}")
        else:
            st.caption("시각화할 교란·은폐 구간이 없습니다.")
    with right:
        st.markdown("**[최종 역전 결론 · 중립 교정문 · 1페이지 증거 카드]**")
        label, color = describe_verdict(bundle.get("verdict", ""))
        show_badge(label, color)
        st.markdown(bundle.get("headline", ""))
        if bundle.get("neutral_rewrite"):
            st.markdown(f"> **중립 교정문** {bundle['neutral_rewrite']}")
        st.markdown(f"**현장 확인 질문** {bundle.get('field_question', '')}")
        audit = bundle.get("number_audit") or {}
        if audit:
            ok = not audit.get("ungrounded")
            show_badge(f"수치 근거 검증 {audit.get('grounded', 0)}/{audit.get('checked', 0)} 일치",
                       STATUS["good"] if ok else STATUS["critical"])
            if not ok:
                st.caption("도구 결과에서 찾지 못한 수치: " + ", ".join(audit["ungrounded"][:6]))
        st.download_button("1페이지 증거 카드 내려받기(HTML)", build_card_html(bundle), file_name=f"evidence_card_{key}.html",
                           mime="text/html", key=f"card_{key}", width="stretch")
    with st.expander("에이전트 실행 기록(도구·출처·완화 내역)"):
        runs = bundle.get("tool_runs", [])
        if runs:
            st.dataframe(runs, width="stretch", hide_index=True)
        st.caption(f"엔진: {bundle.get('mode')} · {bundle.get('mode_reason', '')} · 전체 {bundle.get('elapsed_ms', 0):,.0f} ms")
        st.dataframe(bundle.get("provenance", []), width="stretch", hide_index=True)
        st.caption(bundle.get("data_notice", ""))


def render_followups(bundle: dict, key: str, workspace: str) -> None:
    followups = bundle.get("followups") or []
    if not followups:
        return
    st.caption("이어서 물어보기")
    columns = st.columns(len(followups))
    for index, (column, text) in enumerate(zip(columns, followups)):
        if column.button(text, key=f"follow_{key}_{index}", width="stretch"):
            st.session_state[f"pending_{workspace}"] = text
            st.rerun()
