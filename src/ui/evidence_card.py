"""1페이지 증거 카드(HTML). 외부 라이브러리 없이 인라인 SVG 막대 그래프를 넣어 센터 PC 에서 바로 인쇄(PDF 저장)할 수 있다."""
from __future__ import annotations

import html
from datetime import datetime

from src.agent.prompts import SECTION_FACTS, SECTION_LIMIT, SECTION_REVERSAL
from src.ui.chart_style import SERIES, STATUS

VERDICT_LABELS = {
    "reversal": ("순위 역전", "serious"), "conditional_reversal": ("조건부 역전", "warning"),
    "no_reversal": ("역전 없음", "good"), "masked_overload": ("평균에 가려진 과부하", "serious"),
    "structural_break": ("집계 단절", "critical"), "unmeasured_cause": ("미측정 인과", "critical"),
    "thin_denominator": ("분모 함정", "warning"), "confirmed": ("정상 주장 확인", "good"),
    "refuted": ("주장과 반대", "critical"), "boundary": ("역전 경계 있음", "warning"),
    "no_boundary": ("경계 없음", "good"), "abstain": ("판정 보류", "warning"), "inconclusive": ("판정 유보", "warning"),
    "correlation_only": ("연관만 확인", "warning"), "none": ("과부하 없음", "good"), "info": ("안내", "good"),
    "insufficient_area": ("지역 미지정", "warning"), "figure_mismatch": ("수치 불일치", "critical"),
    "no_issue": ("이상 없음", "good"), "unreadable": ("판독 불가", "warning"),
}


def describe_verdict(verdict: str) -> tuple[str, str]:
    label, status = VERDICT_LABELS.get(verdict, (verdict, "warning"))
    return label, STATUS[status]


def pick_bars(bundle: dict) -> tuple[str, list[tuple[str, float, str]]]:
    outputs = bundle.get("outputs") or {}
    decompose = outputs.get("decompose_confounders") or {}
    excavate = outputs.get("excavate_masked_slice") or {}
    stages = decompose.get("stages") or []
    if bundle.get("verdict") in ("masked_overload",) and excavate.get("slices"):
        top = excavate["slices"][0]
        return (f"{top['target']} · {top['day_group']} {top['window']}",
                [("같은 시각 평소", top["baseline_mean"], SERIES[0]), ("숨은 조각", top["slice_mean"], SERIES[1])])
    if bundle.get("verdict") in ("structural_break", "unmeasured_cause") and excavate.get("breaks"):
        top = excavate["breaks"][0]
        return (f"{top['emd'] or '제주 전역'} 월초 단절",
                [(f"{top['prev_month']}", top["prev_value"], SERIES[0]), (f"{top['month']}", top["value"], SERIES[1])])
    if len(stages) >= 3:
        first, final = stages[0], next((s for s in stages if s["stage"] == decompose.get("final_stage")), stages[-1])
        bars = []
        for index, label in enumerate(first.get("values", {})):
            bars.append((f"{label} · 단순 평균", first["values"][label], SERIES[index % len(SERIES)]))
        for index, label in enumerate(final.get("values", {})):
            bars.append((f"{label} · 교란 제거", final["values"][label], SERIES[index % len(SERIES)]))
        return "단순 평균 vs 교란 제거 후(시간당 인구)", bars
    return "", []


def draw_svg(bars: list[tuple[str, float, str]]) -> str:
    usable = [(label, float(value), color) for label, value, color in bars if isinstance(value, (int, float))]
    if not usable:
        return ""
    peak = max(value for _, value, _ in usable) or 1.0
    row = 30
    width = 640
    rows = []
    for index, (label, value, color) in enumerate(usable):
        y = 10 + index * row
        length = max(2.0, 380 * value / peak)
        rows.append(
            f'<text x="0" y="{y + 15}" font-size="12" fill="#52514e">{html.escape(label)}</text>'
            f'<rect x="210" y="{y + 3}" width="{length:.1f}" height="16" rx="4" fill="{color}"/>'
            f'<text x="{216 + length:.1f}" y="{y + 15}" font-size="12" fill="#0b0b0b">{value:,.1f}</text>')
    height = 20 + row * len(usable)
    return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">{"".join(rows)}</svg>'


def build_card_html(bundle: dict) -> str:
    label, color = describe_verdict(bundle.get("verdict", ""))
    title, bars = pick_bars(bundle)
    svg = draw_svg(bars)

    def list_items(lines):
        return "".join(f"<li>{html.escape(str(line))}</li>" for line in lines)

    provenance = "".join(
        f"<tr><td>{html.escape(str(p.get('table')))}</td><td>{html.escape(str(p.get('source_type')))}</td>"
        f"<td>{html.escape(str(p.get('dataset_id')))}</td><td>{html.escape(str(p.get('reference_date')))}</td></tr>"
        for p in bundle.get("provenance", []))
    audit = bundle.get("number_audit") or {}
    neutral = bundle.get("neutral_rewrite")
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>증거 카드</title>
<style>
@page {{ size: A4; margin: 14mm; }}
body {{ font-family: 'Malgun Gothic','Apple SD Gothic Neo','NanumGothic',sans-serif; color:#0b0b0b; background:#fcfcfb; margin:0; }}
.card {{ max-width: 760px; margin: 0 auto; padding: 18px 22px; border:1px solid rgba(11,11,11,.12); border-radius:12px; background:#fff; }}
h1 {{ font-size:19px; margin:0 0 4px; }} .meta {{ color:#52514e; font-size:12px; }}
.badge {{ display:inline-block; padding:3px 10px; border-radius:999px; color:#fff; font-size:12px; font-weight:700; background:{color}; }}
.headline {{ font-size:15px; font-weight:700; margin:12px 0; line-height:1.5; }}
h2 {{ font-size:13px; margin:14px 0 4px; color:#256abf; }} ul {{ margin:0; padding-left:18px; font-size:12.5px; line-height:1.6; }}
table {{ border-collapse:collapse; font-size:11px; width:100%; margin-top:6px; }} td,th {{ border-bottom:1px solid #e1e0d9; padding:3px 4px; text-align:left; }}
.q {{ background:#f0efec; border-radius:8px; padding:8px 10px; font-size:13px; margin-top:6px; }}
</style></head><body><div class="card">
<h1>제주의 두 얼굴 · 1페이지 증거 카드</h1>
<div class="meta">질문: {html.escape(bundle.get('question', ''))}</div>
<div class="meta">생성: {datetime.now().strftime('%Y-%m-%d %H:%M')} · 엔진: {html.escape(str(bundle.get('mode', '')))} · {html.escape(str(bundle.get('mode_reason', '')))}</div>
<p><span class="badge">{html.escape(label)}</span></p>
<div class="headline">{html.escape(bundle.get('headline', ''))}</div>
{f'<h2>{html.escape(title)}</h2>{svg}' if svg else ''}
<h2>{SECTION_FACTS}</h2><ul>{list_items(bundle.get('facts', []))}</ul>
<h2>{SECTION_REVERSAL}</h2><ul>{list_items(bundle.get('reversals', []))}</ul>
{f'<h2>중립 교정문</h2><ul><li>{html.escape(neutral)}</li></ul>' if neutral else ''}
<h2>{SECTION_LIMIT}</h2><ul>{list_items(bundle.get('limitations', []))}</ul>
<div class="q"><b>현장 확인 질문</b> {html.escape(bundle.get('field_question', ''))}</div>
<h2>출처 태그</h2><table><tr><th>테이블</th><th>source_type</th><th>dataset_id</th><th>reference_date</th></tr>{provenance}</table>
<p class="meta">수치 근거 검증: {audit.get('grounded', 0)}/{audit.get('checked', 0)} 일치 · {html.escape(bundle.get('data_notice', ''))}</p>
</div></body></html>"""
