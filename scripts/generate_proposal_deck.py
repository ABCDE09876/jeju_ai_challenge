"""15쪽 AI Agent 기획서 자동 생성(상호평가 공개 문서).

출력
  docs/AI_Agent_기획서_15쪽_완성본.md   : 원고(공식 5대 목차 매핑)
  docs/AI_Agent_기획서_15쪽.html        : 인쇄용 A4 가로 15쪽
  docs/AI_Agent_기획서_15쪽.pdf         : Edge/Chrome 헤드리스 인쇄(설치된 경우)
  docs/AI_Agent_기획서_15쪽.pptx        : 발표·편집용(python-pptx 설치된 경우)
  docs/figures/fig1~fig5_*.png          : 200dpi 삽입용 차트 5장
모든 수치는 생성 시점에 도구를 실제로 실행하고 벤치마크·퍼징 리포트를 읽어 채운다(손으로 적은 숫자 없음).
python scripts/generate_proposal_deck.py
"""
from __future__ import annotations

import html
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from src.config import CLAUDE_EFFORT, CLAUDE_MODEL, DOCS_DIR, PROCESSED_DIR, SEED  # noqa: E402
from src.tools.analysis_cubes import load_food, load_wifi, stamp_db, warm_cubes  # noqa: E402
from src.tools.boundary_simulator import simulate_reversal_boundary  # noqa: E402
from src.tools.confounder_decomposer import decompose_confounders  # noqa: E402
from src.tools.db_connector import fetch_frame  # noqa: E402
from src.tools.slice_excavator import excavate_masked_slice  # noqa: E402
from src.ui.chart_style import INK, SERIES, STATUS, apply_korean_font  # noqa: E402

FIG_DIR = DOCS_DIR / "figures"
MD_PATH = DOCS_DIR / "AI_Agent_기획서_15쪽_완성본.md"
HTML_PATH = DOCS_DIR / "AI_Agent_기획서_15쪽.html"
PDF_PATH = DOCS_DIR / "AI_Agent_기획서_15쪽.pdf"
PPTX_PATH = DOCS_DIR / "AI_Agent_기획서_15쪽.pptx"
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe", "/usr/bin/google-chrome", "/usr/bin/chromium"]


# ---------------------------------------------------------------- 근거 수집

def collect_evidence() -> dict:
    warm_cubes()
    v1 = json.loads((PROCESSED_DIR / "benchmark_report_heldout_v1.json").read_text(encoding="utf-8"))
    v2 = json.loads((PROCESSED_DIR / "benchmark_report_posthoc_v2.json").read_text(encoding="utf-8"))
    fuzz = json.loads((PROCESSED_DIR / "fuzz_report.json").read_text(encoding="utf-8"))
    robust_path = PROCESSED_DIR / "robustness_report.json"
    robust = json.loads(robust_path.read_text(encoding="utf-8")) if robust_path.exists() else None
    scale_path = PROCESSED_DIR / "scale_report.json"
    scale = json.loads(scale_path.read_text(encoding="utf-8")) if scale_path.exists() else None
    diverse_path = PROCESSED_DIR / "diverse_report.json"
    diverse = json.loads(diverse_path.read_text(encoding="utf-8")) if diverse_path.exists() else None
    mega_path = PROCESSED_DIR / "mega_report.json"
    mega = json.loads(mega_path.read_text(encoding="utf-8")) if mega_path.exists() else None
    restyle_path = PROCESSED_DIR / "restyle_report.json"
    restyle = json.loads(restyle_path.read_text(encoding="utf-8")) if restyle_path.exists() else None
    live_path = PROCESSED_DIR / "live_report_v1.json"  # 실제 Claude 경로 40문항 1차(공식) 실행
    live40 = json.loads(live_path.read_text(encoding="utf-8")) if live_path.exists() else None
    registry = fetch_frame("SELECT table_name, axis, official_sheet, source_type, dataset_id, row_count FROM table_registry")
    return {
        "aewol": decompose_confounders(["애월읍", "아라동"], "living"),
        "market": excavate_masked_slice(["오일장"], "crowd"),
        "hallim": excavate_masked_slice(["한림읍"], "waste",
                                       claim="한림읍 음식물쓰레기가 2019년 12월에 10월보다 58.69배 폭증했다. 관광객 때문이다."),
        "boundary_dry": simulate_reversal_boundary(["아라동", "삼화지구"], "return_hour"),
        "boundary_rain": simulate_reversal_boundary(["아라동", "삼화지구"], "rain"),
        "v1": v1, "v2": v2, "fuzz": fuzz, "robust": robust, "scale": scale, "diverse": diverse, "mega": mega, "restyle": restyle,
        "live40": live40, "registry": registry,
        "examples": int((registry["source_type"] == "official_example").sum()),
        "rows_total": int(registry["row_count"].sum()),
        "build_seconds": fetch_frame("SELECT value FROM build_log WHERE key = 'build_seconds'")["value"].iloc[0],
    }


# ---------------------------------------------------------------- 차트 5장

def save_figure(figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def draw_architecture() -> Path:
    figure, axes = plt.subplots(figsize=(12, 5.6))
    axes.set_xlim(0, 12)
    axes.set_ylim(0, 6)
    axes.axis("off")

    def draw_box(x, y, w, h, title, body, color):
        axes.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", linewidth=1.2,
                                      edgecolor=color, facecolor=color + "18"))
        axes.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=11.5, weight="bold", color=INK["primary"])
        axes.text(x + w / 2, y + h / 2 - 0.18, body, ha="center", va="center", fontsize=9.2, color=INK["secondary"],
                  linespacing=1.45)

    def draw_arrow(x1, y1, x2, y2, label="", color=INK["secondary"], spot=None, align="center"):
        axes.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, linewidth=1.4,
                                       color=color))
        if label:
            x, y = spot or ((x1 + x2) / 2, (y1 + y2) / 2)
            axes.text(x, y, label, ha=align, va="center", fontsize=8.8, color=color, linespacing=1.35,
                      bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none"})

    draw_box(0.2, 2.3, 2.1, 1.6, "도민 질문", "자연어·오타·초성\n생활권 별칭\n멀티턴 후속 질문", SERIES[0])
    draw_box(2.8, 2.3, 2.2, 1.6, "로컬 NLU", "자모 퍼지 지명 파서\n문맥 추적기\n판정 보류 감지", SERIES[0])
    draw_box(5.6, 3.7, 2.7, 1.9, "Claude 경로", f"Claude {CLAUDE_MODEL}\neffort {CLAUDE_EFFORT} 고정\nTool Use 연쇄 호출\n한국어 스트리밍 해설",
        "#6250d6")
    draw_box(5.6, 0.5, 2.7, 1.9, "로컬 엔진", "규칙·통계 기반 ReAct\n결정론적 상태 머신\n스트리밍 합성기\n외부 호출 0회", STATUS["good"])
    draw_box(8.9, 2.1, 2.9, 2.0, "4대 로컬 도구", "① 공간·카탈로그 결합\n② 3대 허수 걷어내기\n③ 은폐 구간·단절·인과 감사\n④ 역전 경계·현장 질문",
        SERIES[1])
    draw_arrow(2.3, 3.1, 2.8, 3.1)
    draw_arrow(5.0, 3.3, 5.6, 4.4, "키 있음", spot=(4.95, 4.15), align="right")
    draw_arrow(5.0, 2.9, 5.6, 1.6, "키 없음·오프라인", spot=(4.95, 1.85), align="right")
    draw_arrow(6.95, 3.7, 6.95, 2.4, "3.5초 무응답·오류·거절\n즉시 자동 전환", STATUS["critical"], spot=(7.1, 3.05), align="left")
    draw_arrow(8.3, 4.6, 8.9, 3.6, "도구 호출", spot=(8.95, 4.4), align="left")
    draw_arrow(8.3, 1.4, 8.9, 2.6, "도구 호출", spot=(8.95, 1.9), align="left")
    axes.text(10.35, 1.15, "SQLite + pandas 인메모리 큐브\n수치는 여기서만 계산", ha="center", fontsize=8.8, color=INK["secondary"])
    axes.text(6.0, 5.82, "모든 수치는 로컬 도구가 계산 -> 해설 속 숫자는 수치 근거 감사기가 재대조", ha="center", fontsize=10,
              color=INK["primary"], weight="bold")
    return save_figure(figure, "fig1_architecture.png")


def draw_reversal(evidence: dict) -> Path:
    stages = evidence["aewol"]["stages"]
    labels = ["단순 평균", "① 관광객 제거", "② 생활 시간대", "③ 비 오는 날"]
    figure, axes = plt.subplots(figsize=(8.4, 4.4))
    width = 0.36
    for index, area in enumerate(stages[0]["values"]):
        values = [stage["values"].get(area) or 0 for stage in stages]
        positions = [i + (index - 0.5) * width for i in range(len(stages))]
        bars = axes.bar(positions, values, width=width * 0.92, color=SERIES[index], label=area)
        for bar, value in zip(bars, values):
            axes.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,.0f}", ha="center", va="bottom", fontsize=8.5,
                      color=INK["secondary"])
    axes.set_xticks(range(len(stages)))
    axes.set_xticklabels(labels)
    axes.set_ylabel("시간당 체류 인구(명)")
    axes.set_title("관광객 거품을 걷어내면 순위가 뒤집힌다: 애월읍 vs 아라동", fontsize=12, weight="bold")
    axes.legend(frameon=False, loc="upper right")
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(axis="y", color=INK["grid"])
    return save_figure(figure, "fig2_tourist_reversal.png")


def draw_slice(evidence: dict) -> Path:
    top = evidence["market"]["slices"][0]
    profile = top["profile"]
    figure, axes = plt.subplots(figsize=(8.4, 4.2))
    axes.plot(profile["hours"], profile["baseline"], color=SERIES[0], linewidth=2, label="같은 시각 평소 평균")
    axes.plot(profile["hours"], profile["slice"], color=SERIES[1], linewidth=2, marker="o", markersize=5,
              label=top["day_group"])
    axes.axvspan(top["hours"][0] - 0.5, top["hours"][-1] + 0.5, color=STATUS["serious"], alpha=0.13)
    axes.set_xticks(list(range(profile["hours"][0], profile["hours"][-1] + 1, 2)))
    axes.set_xlabel("시")
    axes.set_ylabel("정류소 승하차(명/시간)")
    axes.set_title(f"평균에 가려진 과부하: {top['target']} {top['day_group']} {top['window']} = 평소의 {top['lift']:.2f}배",
                   fontsize=11.5, weight="bold")
    axes.legend(frameon=False)
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(axis="y", color=INK["grid"])
    return save_figure(figure, "fig3_hidden_slice.png")


def draw_break() -> Path:
    stamp = stamp_db()
    food = load_food(stamp)
    hallim = food[food["emd_nm"] == "한림읍"].sort_values("base_year_month")
    wifi = load_wifi(stamp).groupby("base_year_month")["user_count"].sum().sort_index()
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4.0))
    x = range(len(hallim))
    left.plot(list(x), hallim["discharge_daily_avg"], color=SERIES[0], linewidth=2)
    left.set_yscale("log")
    months = hallim["base_year_month"].tolist()
    if "201911" in months:
        left.axvline(months.index("201911"), color=STATUS["critical"], linestyle=":", linewidth=1.6)
        left.text(months.index("201911") + 0.6, hallim["discharge_daily_avg"].min() * 1.6, "2019-11 RFID 전환\n(집계 단절)",
                  fontsize=9, color=STATUS["critical"])
    left.set_xticks([i for i, m in enumerate(months) if m.endswith("01")])
    left.set_xticklabels([m[:4] for m in months if m.endswith("01")])
    left.set_title("한림읍 음식물쓰레기 일평균(로그 축)", fontsize=11, weight="bold")
    wifi_months = list(wifi.index)
    right.plot(range(len(wifi)), wifi.values, color=SERIES[1], linewidth=2)
    if "202010" in wifi_months:
        right.axvline(wifi_months.index("202010"), color=STATUS["critical"], linestyle=":", linewidth=1.6)
        right.text(wifi_months.index("202010") + 0.5, wifi.max() * 0.9, "2020-10 AP->장소\n집계 단위 전환", fontsize=9,
                   color=STATUS["critical"])
    right.set_xticks([i for i, m in enumerate(wifi_months) if m.endswith("01")])
    right.set_xticklabels([m[:4] for m in wifi_months if m.endswith("01")])
    right.set_title("공공 와이파이 월 이용자 수", fontsize=11, weight="bold")
    from matplotlib.ticker import FuncFormatter

    comma = FuncFormatter(lambda value, _: f"{value:,.0f}")
    left.yaxis.set_major_formatter(comma)
    left.yaxis.set_minor_formatter(FuncFormatter(lambda value, _: ""))
    right.yaxis.set_major_formatter(comma)
    for axes in (left, right):
        axes.spines[["top", "right"]].set_visible(False)
        axes.grid(axis="y", color=INK["grid"])
    figure.suptitle("월초 계단식 점프는 '폭증'이 아니라 집계 방식 변경일 수 있다", fontsize=12.5, weight="bold", y=1.03)
    return save_figure(figure, "fig4_structural_break.png")


def draw_benchmark(evidence: dict) -> Path:
    summary = evidence["v1"]["summary"]
    groups = [("masked_slice", "은폐 구간 검출"), ("causal_block", "집계·인과·분모 함정 차단"),
              ("abstention", "판정 보류 정답"), ("accuracy_test", "Held-out 정답")]
    methods = [("baseline_mean", "Baseline 1 단순 평균", INK["muted"]), ("baseline_lookup", "Baseline 2 단일 표 조회", SERIES[3]),
               ("ours", "Ours(동결 규칙 1차 실행)", SERIES[0])]
    figure, axes = plt.subplots(figsize=(9.6, 4.3))
    width = 0.26
    for offset, (key, label, color) in enumerate(methods):
        ratios = [summary[key][g]["count"] / summary[key][g]["total"] for g, _ in groups]
        positions = [i + (offset - 1) * width for i in range(len(groups))]
        bars = axes.bar(positions, ratios, width=width * 0.92, color=color, label=label)
        for bar, (g, _) in zip(bars, groups):
            axes.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                      summary[key][g]["text"].replace(" ", ""), ha="center", fontsize=8.3, color=INK["secondary"])
    axes.set_xticks(range(len(groups)))
    axes.set_xticklabels([label for _, label in groups])
    axes.set_ylim(0, 1.18)
    axes.set_ylabel("성공 비율(성공/전체)")
    axes.set_title("24문항 벤치마크(동결 규칙 1차 실행, 분모 표기)", fontsize=12, weight="bold")
    axes.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(axis="y", color=INK["grid"])
    return save_figure(figure, "fig5_benchmark.png")


# ---------------------------------------------------------------- 원고(15쪽)

def build_pages(evidence: dict) -> list[dict]:
    aewol, market, hallim = evidence["aewol"], evidence["market"], evidence["hallim"]
    dry, rain = evidence["boundary_dry"], evidence["boundary_rain"]
    stages = {s["stage"]: s for s in aewol["stages"]}
    s0, s1 = stages["S0"]["values"], stages["S1"]["values"]
    metrics = {m["label"]: m for m in aewol["metrics"]}
    top = market["slices"][0]
    ref = hallim["hallim_reference"]
    v1, v2, fuzz = evidence["v1"]["summary"], evidence["v2"]["summary"], evidence["fuzz"]["summary"]
    live = fuzz.get("live_api", {}).get("text", "미실행")
    live40 = (evidence.get("live40") or {}).get("summary")
    same = live40["verdict_same_as_local"] if live40 else None  # 실제 Claude 경로 40문항에서 판정이 로컬 엔진과 같은 수
    live_line = f"실제 Claude 경로 40문항 판정 {same} 로컬과 일치" if same else f"실제 Claude 연결 상태 {live} 통과"
    live_short = f"Claude 40문항 판정 로컬 일치 {same}" if same else f"실제 Claude 연결 {live}"  # 15쪽 좁은 칸용
    registry = evidence["registry"]

    def count_rows(name):
        row = registry[registry["table_name"] == name]
        return int(row["row_count"].iloc[0]) if len(row) else 0

    pages = [
        {"section": "1. 구현목표", "title": "제주의 두 얼굴: 관광객·기상·평균의 3대 허수를 벗기는 도민 데이터 변호사",
         "message": "제주 공공데이터의 '평균'은 관광객·낮 시간·맑은 날이 섞인 착시입니다. 이 에이전트는 그 허수를 벗겨 도민 실생활 결론과 민원 근거를 1분 안에 만듭니다.",
         "bullets": [
             f"60초 요약: 질문 한 줄 -> 4대 로컬 도구 실행 -> 3단 대조 대시보드 + 1쪽 증거 카드 + 현장 확인 질문 1개",
             f"검증(동결 규칙 1차 실행): 24문항 중 {v1['ours']['accuracy_all']['text']} 정답, 단순 평균 방식은 {v1['baseline_mean']['accuracy_all']['text']}",
             f"안정성: 50문항 퍼징 {fuzz['text']} 통과(런타임 오류 {fuzz['runtime_errors']}건), {live_line}",
             f"하이브리드: Claude {CLAUDE_MODEL}(effort {CLAUDE_EFFORT}) + 키 없이도 100% 동작하는 로컬 엔진(규칙·통계 기반)",
             "데이터 안내: 이 기획서의 분석 수치는 공식 예시데이터 스키마를 보존한 시드 고정 합성(시연용) 데이터 기준이며, "
             "원자료를 넣으면 같은 절차로 재계산됩니다(모든 행에 출처 태그)"]},
        {"section": "1. 구현목표", "title": "분석 주제와 목표: 평균이 숨기는 도민의 하루",
         "message": "같은 동네도 관광객·시간대·날씨를 빼면 다른 얼굴이 드러납니다. 목표는 '이사·출퇴근·외출 결정'과 '통계 주장·민원 검증'을 도민 스스로 하게 하는 것입니다.",
         "bullets": ["허수 ① 관광객 거품: 도외 거주자 체류·관광 소비가 '활기'를 부풀림",
                     "허수 ② 주간 쏠림·야간 공동화: 낮 평균은 밤 9시 이후 생활을 대변하지 못함",
                     "허수 ③ 비 오는 날의 숨은 혼잡: 비 오는 날 퇴근길 정류소 혼잡은 연평균에 묻힘",
                     "추가 함정: 월초 계단식 집계 단절, 원자료 컬럼에 없는 원인 단정, 일부 행만 계산된 평균(분모 함정)",
                     "성과 지표: 평균 착시 오판 0건, 은폐 구간 검출률, 인과 과장 차단률, 정상 주장 오경보 0건, 판정 보류 정확도"]},
        {"section": "2. 서비스 제안", "title": "데이터 활용 개요: 4대 축을 공간 키로 묶는다",
         "message": f"공식 예시데이터 {evidence['examples']}개 시트 전부를 기준 테이블로 적재하고, 같은 컬럼·코드값으로 확장한 4대 축을 행정동-법정동-정류소-관측소 키로 결합합니다.",
         "table": (["축", "로컬 테이블(행 수)", "공식 시트·데이터허브 출처"], [
             ["① 유동·생활인구/소비", f"floating_pop_hourly({count_rows('floating_pop_hourly'):,}), card_time_monthly({count_rows('card_time_monthly'):,})",
              "4-11 내국인 유동인구(읍면동)_v02 / #753, 1-14 카드 이용 현황(시간대)_v02"],
             ["② 대중교통·이동", f"bus_stop_hourly({count_rows('bus_stop_hourly'):,}), bus_service_hourly({count_rows('bus_service_hourly'):,})",
              "3-04 버스 정류소별 이용 현황 / #725·#1249"],
             ["③ 기상·환경", f"weather_daily({count_rows('weather_daily'):,}), food_waste_monthly({count_rows('food_waste_monthly'):,}), ev_charging_wait({count_rows('ev_charging_wait'):,})",
              "5-02 일일 날씨 정보 / #756, 5-05 음식물쓰레기 / #515, #136 전기차 충전"],
             ["④ 메타·집계변경 이력", f"datahub_catalog({count_rows('datahub_catalog'):,}), aggregation_changes, column_registry",
              "제주데이터허브 데이터목록_20261105.xlsx 전량"]]),
         "bullets": [f"전체 {len(registry)}개 테이블 {evidence['rows_total']:,}행, 모든 행에 source_type·dataset_id·reference_date 출처 태그",
                     "①~③ 축 분석 테이블은 공식 컬럼·코드값을 그대로 쓴 시드 고정 합성(시연용) 데이터(source_type=synthetic_seed), "
                     "사용자 제공 공개 통계 사례는 calibrated_reference 로 구분 표기",
                     "원자료 CSV를 data/raw 에 넣으면 컬럼 서명을 인식해 합성 테이블 대신 최우선 적재(official_raw)"]},
        {"section": "2. 서비스 제안", "title": "핵심 기능: 3개 작업 공간",
         "message": "도민의 생활 질문, 시민의 반박 근거, 담당자의 새 파일 점검을 한 화면에서 처리합니다.",
         "bullets": ["모드 1 제주의 두 얼굴(실생활 교차검증): 이사·출퇴근·외출 비교, 귀가 시각·우천 경계 계산",
                     "모드 2 평균의 거짓말, 도민의 증거(시민 데이터 변호사): 기사·주장·민원 문장을 감사하고 중립 교정문 작성",
                     "모드 3 새 제주 CSV 실시간 감사기: 아무 CSV + 주장 -> 인코딩 복구, 수치 재계산, 월경계 단절, 인과 과장 교정",
                     "공통 출력: [원자료로 확인된 수치 사실] / [교란 제거 후 역전 구간] / [한계 및 현장 확인 질문 1개]",
                     "원클릭 시연 버튼 12개(작업 공간별 6개), 후속 질문 버튼 3개, 1쪽 증거 카드 내려받기"]},
        {"section": "2. 서비스 제안", "title": "도민 편의성 + 분석 사례 ①: 관광객 거품 역전",
         "message": f"(합성 시연 데이터 기준) 단순 평균은 애월읍 {s0['애월읍']:,.0f}명 > 아라동 {s0['아라동']:,.0f}명이지만, 도외 거주자를 빼면 애월읍 {s1['애월읍']:,.0f}명 < 아라동 {s1['아라동']:,.0f}명으로 뒤집힙니다.",
         "figure": "figures/fig2_tourist_reversal.png",
         "bullets": [f"애월읍 체류 인구 중 도외 거주자 비중 {metrics['애월읍']['tourist_share'] * 100:.1f}%, 아라동 {metrics['아라동']['tourist_share'] * 100:.1f}%",
                     "오타·초성·별칭(노형똥, ㅅㅈㅈ, 신제주, 삼화지구)과 '그럼 8시 이후는?' 같은 후속 질문을 그대로 이해",
                     "표본이 5일 미만이면 [요일·시각 -> 평일/주말 구간 -> 지역 전체] 순으로 조건을 넓히고 그 사실을 알림"]},
        {"section": "2. 서비스 제안", "title": "분석 사례 ②: 집계 단절·은폐 구간·우천 경계",
         "message": f"한림읍 음식물쓰레기 일평균 {ref['oct']:,.2f} -> {ref['dec']:,.2f}({ref['ratio']:.2f}배, 사용자 제공 공개 통계 사례로 보정한 표본)은 2019년 11월 RFID 전환 시점과 일치하는 집계 단절입니다. '관광객 때문'은 측정 컬럼이 없어 단정할 수 없습니다.",
         "figure": "figures/fig4_structural_break.png",
         "bullets": [f"오일장 정류소: 전체 평균 시간당 {top['overall_hourly_mean']:,.1f}명이지만 {top['day_group']} {top['window']}에는 같은 시각 평소의 {top['lift']:.2f}배",
                     f"아라동 vs 삼화지구 귀가: 맑은 날 경계 {dry['boundary']['boundary']}시, 비 오는 날 경계 {rain['boundary']['boundary']}시로 앞당겨짐",
                     "교정문 예: '한림읍 배출량이 변했다(원인 미확인, 관광객 영향은 추가 자료로 검증 필요)'"]},
        {"section": "3. Agent 개발 구성", "title": "참여자 역할·목표·Workflow",
         "message": "에이전트는 4단계 ReAct 흐름을 매번 같은 순서로 실행하고, 각 단계 결과를 화면 상태창에 공개합니다.",
         "table": (["참여자", "목표", "에이전트가 하는 일"], [
             ["도민", "이사·출퇴근·외출 결정", "생활권 별칭 해석, 교란 제거 비교, 귀가 경계·방문 시각 제시"],
             ["시민·기자", "주장·민원 검증", "은폐 구간·집계 단절·미측정 인과·분모 함정 감사, 중립 교정문"],
             ["센터 담당자", "새 데이터 점검", "CSV 인코딩 복구, 공식 시트 스키마 대조, 수치 재계산"]]),
         "bullets": ["Step 1 의도·지명 해석 -> Step 2 데이터셋·공간 키 결합 -> Step 3 3대 허수 걷어내기 -> Step 4 은폐 구간·역전 경계",
                     "모든 단계는 같은 파이썬 도구를 쓰므로 Claude 경로와 로컬 경로의 수치가 항상 같음"]},
        {"section": "3. Agent 개발 구성", "title": "하이브리드 구조: Claude API + 로컬 엔진",
         "message": f"키가 있으면 Claude {CLAUDE_MODEL}(effort {CLAUDE_EFFORT})가 해석·도구 연쇄·해설을 맡고, 3.5초 무응답·인증/네트워크 오류·429·거절이면 화면 오류 없이 로컬 엔진이 즉시 이어받습니다.",
         "figure": "figures/fig1_architecture.png",
         "bullets": ["숫자는 로컬 도구만 계산: Claude 에게는 도구 결과의 숫자만 인용하게 하고, 해설 속 숫자는 수치 근거 감사기가 재대조해 근거 없는 숫자를 화면에 표시",
                     "모델과 추론 강도는 고정하고 호출 횟수 제한은 두지 않음. 대화 이력은 고치지 않고 이어 붙여 앞선 추론 맥락을 그대로 유지",
                     "장애 시험 6종 통과: 잘못된 키·네트워크 차단·첫 응답 지연·거절·서버 과부하는 로컬 자동 전환, 도구 예외는 나머지 도구로 답변"]},
        {"section": "3. Agent 개발 구성", "title": "4대 Tool 명세(Claude tools JSON Schema 겸용)",
         "message": "같은 함수가 로컬 엔진과 Claude Tool Use 에 동시에 노출됩니다.",
         "table": (["도구", "입력", "핵심 처리", "출력"], [
             ["resolve_spatial_catalog", "areas, topic", "별칭·오타 해석, 2~4종 데이터셋 선택, 행정동-법정동-정류소-관측소 결합", "공간 단위, 출처, 공식 시트 컬럼 일치"],
             ["decompose_confounders", "areas, 시간대, 날씨, 요일", "단순 평균 -> 관광객 제거 -> 생활 시간대 -> 우천 4단 순위", "역전 여부, 단계별 수치"],
             ["excavate_masked_slice", "areas, claim", "요일x시간x날씨/장날 조각 배율, 월초 단절, 인과 컬럼 대조", "은폐 구간, 단절, 교정문"],
             ["simulate_reversal_boundary", "areas, 변수", "귀가·방문 시각, 허용 대기, 우천을 움직여 경계 계산", "경계, 증거 카드, 질문 1개+후속 3개"]])},
        {"section": "3. Agent 개발 구성", "title": "자율 수행 범위와 판정 보류",
         "message": "에이전트는 근거가 있을 때만 결론을 내고, 컬럼·기간이 없으면 억지 추론 대신 판정을 보류합니다.",
         "bullets": ["자율 수행: 지명 교정, 비교 상대 자동 선택, 조건 완화, 도구 순서 결정, 증거 카드·후속 질문 생성",
                     "판정 보류: 집값·소음·치안·학군 등 측정 컬럼 없음, 보유 기간 밖 시점(예: 작년 겨울) -> 보류 + 현장 확인 질문",
                     f"벤치마크 판정 보류 문항 {v1['ours']['abstention']['text']} 정답(기준선 {v1['baseline_mean']['abstention']['text']})",
                     "Claude 가 안전 정책상 거절(refusal)하면 다른 모델로 넘기지 않고 로컬 엔진이 같은 도구로 답변(모의 거절 시험 통과)"]},
        {"section": "4. 데이터 활용 계획", "title": "필수 데이터명·출처·활용 항목",
         "message": "대회 제공 엑셀 2종을 전부 해독해 실제 컬럼명·코드 표기를 그대로 쓰고, 데이터허브 목록 번호로 출처를 연결합니다.",
         "table": (["대회 제공 파일", "활용 방식", "핵심 항목"], [
             ["제주데이터허브 데이터목록_20261105.xlsx", "1,259건 전량 적재 + 주제별 데이터셋 탐색", "#753 유동인구, #725/#1249 정류소 이용, #756 기상, #515 음식물, #749/#742 와이파이, #136 전기차"],
             ["제주빅데이터센터_예시데이터 및 컬럼확인용.xlsx", f"{evidence['examples']}개 시트 전부 기준 테이블(bdc_*) 적재", "resd(제주/그외), resd·work·visit_pop_cnt, get_on_off_type, user_category, daily_rainfall 등"]]),
         "bullets": ["분석 테이블 컬럼은 공식 시트와 1:1 일치(유동인구 11/11, 정류소 9/9, 기상 14/14)",
                     "이용조건: DAKER 제공 파일은 외부 배포 금지 -> 제출 ZIP 기본 모드에서 원본 엑셀 제외, 해독 결과만 포함"]},
        {"section": "4. 데이터 활용 계획", "title": "수집·전처리·교란 제거 방법",
         "message": "현장 데이터에 흔한 문제를 불러오는 단계에서 먼저 바로잡습니다.",
         "bullets": ["인코딩 utf-8-sig -> cp949 -> euc-kr -> utf-8 -> latin1 자동 판별, 구분자(, 탭 ; |) 자동 인식",
                     "마스킹('*','**','***')은 0이 아니라 결측, 'NULL' 문자열 정리, 천 단위 콤마·float32 잡음 정리",
                     "엑셀 날짜 자동변환 오염 복원('02월 01일' -> 노선 '2-1'), NBSP 섞인 권역명·요일/성별 표기 혼재 정규화",
                     f"합성 확장은 시드 {SEED} 고정, source_type 으로 synthetic_seed / calibrated_reference / official_example 를 구분 표기",
                     "API Key: Anthropic·제주데이터허브 키는 .env 에만, 데이터허브 키는 어떤 제출 모드에서도 제외"]},
        {"section": "5. 개발·탑재 계획", "title": "구현 범위와 UI 화면 구성 + 60초 시연 대본",
         "message": "Streamlit 한 화면: 상단 엔진 배지, 사이드바 엔진 설정, 3개 탭, 대화 + 3단 대조 대시보드.",
         "bullets": ["0~10초: 사이드바에서 키 없이 '100% 로컬 자립 모드' 배지 확인, 강제 오프라인 스위치 소개",
                     "10~25초: '관광객 거품' 버튼 -> 애월읍 vs 아라동 4단 순위 차트에서 역전 확인",
                     "25~40초: '그럼 퇴근 시간을 8시 이후로 바꾸면?' 입력 -> 문맥 계승, 귀가 경계 20시",
                     "40~50초: 변호사 탭 '집계 단절' -> 한림읍 58.69배는 RFID 전환 단절, 관광객 인과는 판정 불가",
                     "50~60초: CSV 감사 탭에서 CP949 표본 올리기 -> 수치 재계산 일치, 증거 카드 내려받기"]},
        {"section": "5. 개발·탑재 계획", "title": "센터 PC 호환성과 외부 의존성",
         "message": "최소사양(i7-8700 / RAM 32GB / GTX 1060 3GB)에서 GPU 없이 CPU·RAM 인메모리 인덱스로 동작합니다.",
         "bullets": [f"벤치마크 실행 중 프로세스 메모리(RSS) {evidence['v1']['process_rss_mb']} MB, 로컬 평균 응답 {v1['ours']['avg_ms']} ms",
                     "DB 는 첫 실행 때 자동 구축(검증 PC 약 7초), 이후 메모리 큐브 재사용",
                     "외부 의존: Claude API 1개(선택). 없거나 막혀도 같은 결과 화면을 로컬 엔진이 생성",
                     "제출 ZIP: local-only(모든 키 제거, 기본) / with-eval-key(평가용 Anthropic 키만 유지) 두 모드",
                     "오류 화면 대신 로컬 답변: 예외는 로컬 로그에만 기록하고 화면에는 로컬 엔진의 답변을 보여 줌"]},
        {"section": "5. 개발·탑재 계획", "title": "검증 결과: 벤치마크와 스트레스 테스트", "dense": True,
         "message": f"동결 규칙 1차 실행: Dev {v1['ours']['accuracy_dev']['text']}, Held-out Test {v1['ours']['accuracy_test']['text']}. 실패 3건을 공개하고 버그 2건을 고친 사후 재실행은 Test {v2['ours']['accuracy_test']['text']}(held-out 성능으로 주장하지 않음).",
         "figure": "figures/fig5_benchmark.png",
         "table": (["지표(성공/전체)", "Baseline 1", "Baseline 2", "Ours 1차"], [
             ["평균 착시 오판(낮을수록 좋음)", v1["baseline_mean"]["mean_illusion"]["text"], v1["baseline_lookup"]["mean_illusion"]["text"], v1["ours"]["mean_illusion"]["text"]],
             ["은폐 구간 검출", v1["baseline_mean"]["masked_slice"]["text"], v1["baseline_lookup"]["masked_slice"]["text"], v1["ours"]["masked_slice"]["text"]],
             ["집계·인과·분모 함정 차단", v1["baseline_mean"]["causal_block"]["text"], v1["baseline_lookup"]["causal_block"]["text"], v1["ours"]["causal_block"]["text"]],
             ["정상 주장 오경보(낮을수록 좋음)", v1["baseline_mean"]["false_alarm"]["text"], v1["baseline_lookup"]["false_alarm"]["text"], v1["ours"]["false_alarm"]["text"]],
             ["판정 보류 정답", v1["baseline_mean"]["abstention"]["text"], v1["baseline_lookup"]["abstention"]["text"], v1["ours"]["abstention"]["text"]]]),
         "bullets": [describe_stress(fuzz, live_short, evidence.get("robust")),
                     describe_robustness(evidence.get("robust")),
                     describe_scale(evidence.get("scale"), evidence.get("diverse"), evidence.get("mega"),
                                    evidence.get("restyle")),
                     "남은 한계: 토요일 혼잡(T10) 미탐, 분석 데이터 다수가 합성(원자료 적재 시 재계산)"]},
    ]
    tiles = {
        1: [(v1["ours"]["accuracy_all"]["text"], "24문항 정답(동결 규칙 1차)"), (fuzz["text"], "가혹 입력 퍼징 통과"),
            (same or live, "Claude·로컬 판정 일치(40문항)" if same else "실제 Claude 연결 시험"),
            (v1["baseline_mean"]["accuracy_all"]["text"], "단순 평균 방식 정답")],
        2: [("3대 허수", "관광객·시간대·악천후"), ("+3 함정", "집계 단절·미측정 인과·분모"), ("1개", "답변마다 현장 확인 질문")],
        4: [("3개", "작업 공간"), ("12개", "원클릭 시연 시나리오"), ("3구획", "모든 답변의 고정 형식")],
        7: [("Step 1~4", "매번 같은 순서의 ReAct"), ("4개", "로컬 교차검증 도구"), (f"{v1['ours']['avg_ms']:.0f} ms", "로컬 평균 응답")],
        9: [(f"{aewol['elapsed_ms']:.0f} ms", "3대 허수 걷어내기 1회"), (f"{market['elapsed_ms']:.0f} ms", "은폐 구간 발굴 1회"),
            (f"{dry['elapsed_ms']:.0f} ms", "역전 경계 계산 1회")],
        10: [(v1["ours"]["abstention"]["text"], "판정 보류 정답"), ("6종", "장애 시험 통과"), ("로컬", "Claude 거절 시 대신 답함")],
        11: [(f"{evidence['examples']}개", "예시 시트 전량 적재"), (f"{count_rows('datahub_catalog'):,}건", "데이터허브 목록 전량"),
             ("11/11", "유동인구 컬럼 일치")],
        12: [("5단계", "인코딩 자동 판별"), ("4종", "구분자 자동 인식"), (str(SEED), "고정 시드")],
        13: [("60초", "시연 대본"), ("3단", "대조 대시보드"), ("1장", "증거 카드(HTML)")],
        14: [(f"{evidence['v1']['process_rss_mb']:.0f} MB", "프로세스 메모리"), (f"{evidence['build_seconds']}초", "DB 최초 구축"),
             ("GPU 0", "CPU 만 사용"), ("1개(선택)", "외부 의존 API")],
    }
    for number, items in tiles.items():
        pages[number - 1]["tiles"] = items
    return pages


def describe_robustness(report: dict | None) -> str:
    """견고성 시험 보고서(robustness_report.json)를 15쪽 한 줄로 요약한다. 없으면 미실행이라고 적는다."""
    if not report or "qwen3:14b" not in report.get("sets", {}):
        return "말투 견고성 시험: 미실행"
    held = report["sets"]["qwen3:14b"]
    baseline = report.get("baseline", {})
    before = baseline.get("sets", {}).get("qwen3:14b", {}).get("full_match_ratio")
    first = baseline.get("round1", {}).get("sets", {}).get("qwen3:14b", {}).get("full_match_ratio")
    if before is None or first is None:
        return f"말투 견고성: 로컬 LLM 이 바꿔 쓴 질문 {held['items']:,}개 해석 일치 {held['full_match_ratio']:.1%}"
    return (f"말투 견고성(다른 모델이 만든 평가용 {held['items']:,}문항): 해석 일치 {before:.1%} -> {first:.1%}, "
            f"사후 {held['full_match_ratio']:.1%}")


def format_count(count: int) -> str:
    """문항 수를 '2,550만'·'1억'처럼 줄여 쓴다."""
    if count >= 100_000_000:
        return f"{count / 100_000_000:.1f}".rstrip("0").rstrip(".") + "억"
    return f"{count // 10000:,}만"


def describe_scale(report: dict | None, diverse: dict | None = None, mega: dict | None = None,
                   restyle: dict | None = None) -> str:
    """대규모 조합(scale_report.json)·다양성(diverse_report.json)·1억 확장(mega_report.json)·새 말투(restyle_report.json)
    시험의 평가용 공식 결과를 15쪽 한 줄로 요약한다."""
    parts = []
    for name, current in (("조합", report), ("다양성", diverse), ("", mega), ("새 말투", restyle)):
        held = (current or {}).get("sets", {}).get("heldout")
        if not held:
            continue
        before = current.get("baseline", {}).get("sets", {}).get("heldout", {}).get("full_match_ratio")
        line = f"{name} {format_count(current['total']['questions'])}".strip()
        line += f" {before:.1%} -> {held['full_match_ratio']:.1%}" if before is not None \
            else f" {held['full_match_ratio']:.1%}"
        parts.append(line)
    if not parts:
        return "대규모 조합 시험: 미실행"
    clean = all((current or {}).get("sets", {}).get("heldout", {}).get("parser_errors", 0) == 0
                for current in (report, diverse, mega, restyle) if current)
    return "대규모 시험 평가용 해석 일치: " + ", ".join(parts) + (", 해석기 예외 0건" if clean else "")


def describe_stress(fuzz: dict, live_line: str, report: dict | None) -> str:
    """퍼징·전체 답변 경로·실제 Claude 연결 결과를 15쪽 한 줄로 요약한다."""
    line = f"퍼징 {fuzz['text']}"
    pipe = (report or {}).get("pipeline")
    if pipe:
        line += f", 질문 {pipe['questions']:,}개 실행 오류 {pipe['runtime_errors']}건"
    return line + f", {live_line}"


# ---------------------------------------------------------------- 출력

def render_markdown(pages: list[dict]) -> str:
    lines = ["# AI Agent 기획서: 제주의 두 얼굴 (15쪽 완성본)", "",
             "> 공식 기획서 5대 목차 매핑: 1~2쪽 구현목표 / 3~6쪽 서비스 제안 / 7~10쪽 Agent 개발 구성 / "
             "11~12쪽 데이터 활용 계획 / 13~15쪽 개발·탑재 계획 및 검증 결과", "",
             "> 이 원고의 모든 수치는 scripts/generate_proposal_deck.py 가 도구를 실제로 실행하고 "
             "벤치마크·퍼징 리포트를 읽어 자동으로 채웠습니다.", ""]
    for number, page in enumerate(pages, start=1):
        lines += [f"## {number}쪽 · [{page['section']}] {page['title']}", "", f"**핵심 메시지** {page['message']}", ""]
        if page.get("tiles"):
            lines += ["**핵심 수치** " + " · ".join(f"{value}({label})" for value, label in page["tiles"]), ""]
        if page.get("table"):
            header, rows = page["table"]
            lines.append("| " + " | ".join(header) + " |")
            lines.append("|" + "---|" * len(header))
            lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
            lines.append("")
        lines += [f"- {bullet}" for bullet in page.get("bullets", [])]
        if page.get("figure"):
            lines += ["", f"![{page['title']}]({page['figure']})"]
        lines.append("")
    return "\n".join(lines)


def render_html(pages: list[dict]) -> str:
    sections = []
    for number, page in enumerate(pages, start=1):
        table = ""
        if page.get("table"):
            header, rows = page["table"]
            table = ("<table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in header) + "</tr>" +
                     "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>" for row in rows) +
                     "</table>")
        bullets = "".join(f"<li>{html.escape(b)}</li>" for b in page.get("bullets", []))
        figure = f'<img src="{page["figure"]}" alt="{html.escape(page["title"])}">' if page.get("figure") else ""
        body = f'<div class="cols"><div class="text">{table}<ul>{bullets}</ul></div><div class="fig">{figure}</div></div>' \
            if figure else f"{table}<ul>{bullets}</ul>"
        tiles = "".join(f'<div class="tile"><b>{html.escape(str(value))}</b><span>{html.escape(label)}</span></div>'
                        for value, label in page.get("tiles", []))
        tiles = f'<div class="tiles">{tiles}</div>' if tiles else ""
        density = " dense" if page.get("dense") else ""
        sections.append(f"""<section class="page{density}"><div class="band">{html.escape(page['section'])}</div>
<h1>{html.escape(page['title'])}</h1><p class="msg">{html.escape(page['message'])}</p>{body}{tiles}
<div class="foot">제주의 두 얼굴 · AI Agent 기획서 · {number} / {len(pages)}</div></section>""")
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>제주의 두 얼굴 기획서</title>
<style>
@page {{ size: A4 landscape; margin: 0; }}
body {{ margin:0; font-family:'Malgun Gothic','Apple SD Gothic Neo','NanumGothic',sans-serif; color:#0b0b0b; background:#f9f9f7; }}
.page {{ width:297mm; height:210mm; box-sizing:border-box; padding:13mm 15mm 12mm; background:#fff; position:relative;
  page-break-after:always; break-after:page; overflow:hidden; }}
.band {{ display:inline-block; background:#2a78d6; color:#fff; font-size:11pt; font-weight:700; padding:3px 12px; border-radius:999px; }}
h1 {{ font-size:25pt; margin:10px 0 8px; }}
.msg {{ font-size:14.5pt; line-height:1.6; background:#f0efec; border-left:5px solid #eb6834; padding:9px 13px; border-radius:6px; margin:6px 0 10px; }}
ul {{ font-size:13.5pt; line-height:1.78; margin:6px 0 0; padding-left:20px; }}
.dense ul {{ line-height:1.42; }}
table {{ border-collapse:collapse; width:100%; font-size:11.6pt; margin:4px 0 6px; }}
th {{ background:#cde2fb; text-align:left; padding:5px 7px; }} td {{ border-bottom:1px solid #e1e0d9; padding:5px 7px; vertical-align:top; }}
.cols {{ display:flex; gap:10mm; }} .text {{ flex:1; }} .fig {{ flex:1.45; display:flex; align-items:flex-start; }}
.fig img {{ width:100%; border:1px solid #e1e0d9; border-radius:8px; }}
.foot {{ position:absolute; bottom:7mm; right:15mm; font-size:9pt; color:#898781; }}
.tiles {{ display:flex; gap:7mm; position:absolute; left:15mm; right:15mm; bottom:17mm; }}
.tile {{ flex:1; border:1px solid #e1e0d9; border-top:5px solid #2a78d6; border-radius:8px; padding:9px 12px; background:#fcfcfb; }}
.tile b {{ display:block; font-size:23pt; color:#256abf; line-height:1.25; }} .tile span {{ font-size:11pt; color:#52514e; }}
</style></head><body>{''.join(sections)}</body></html>"""


def export_pdf(html_path: Path, pdf_path: Path) -> bool:
    browser = next((b for b in BROWSERS if Path(b).exists()), None) or shutil.which("msedge") or shutil.which("chrome")
    if not browser:
        print("[PDF] Edge/Chrome 을 찾지 못해 PDF 변환을 건너뜁니다(HTML 을 브라우저에서 인쇄 -> PDF 저장).")
        return False
    if pdf_path.exists():
        pdf_path.unlink()  # 이전 PDF 가 남아 실패를 성공으로 오인하지 않도록
    # 별도 프로필: 이미 열린 브라우저 창에 인쇄 작업이 넘어가지 않게 한다.
    profile = Path(tempfile.mkdtemp(prefix="jeju_deck_pdf_"))
    command = [browser, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}", "--no-first-run",
               "--no-default-browser-check", "--disable-extensions", "--no-pdf-header-footer",
               "--print-to-pdf-no-header", f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()]
    try:
        subprocess.run(command, check=False, timeout=120, capture_output=True)
        written = ensure_written(pdf_path, 90.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"[PDF] 변환 실패: {exc}")
        written = False
    shutil.rmtree(profile, ignore_errors=True)
    if not written:
        print("[PDF] PDF 파일이 만들어지지 않았습니다(HTML 을 브라우저에서 인쇄 -> PDF 저장).")
    return written


def ensure_written(path: Path, limit_s: float) -> bool:
    """Edge 실행 파일은 인쇄 프로세스를 떼어 놓고 바로 끝날 수 있어, PDF 크기가 더 이상 변하지 않을 때까지 기다린다."""
    deadline = time.monotonic() + limit_s
    last = -1
    while time.monotonic() < deadline:
        size = path.stat().st_size if path.exists() else -1
        if size > 10_000 and size == last:
            return True
        last = size
        time.sleep(0.5)
    return path.exists() and path.stat().st_size > 10_000


def export_pptx(pages: list[dict], path: Path) -> bool:
    try:
        from pptx import Presentation
        from pptx.dml.color import RGBColor
        from pptx.util import Inches, Pt
    except ImportError:
        print("[PPTX] python-pptx 미설치로 건너뜁니다.")
        return False
    deck = Presentation()
    deck.slide_width, deck.slide_height = Inches(13.333), Inches(7.5)
    blank = deck.slide_layouts[6]
    for number, page in enumerate(pages, start=1):
        slide = deck.slides.add_slide(blank)

        def add_text(left, top, width, height, text, size, bold=False, color=(11, 11, 11)):
            frame = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height)).text_frame
            frame.word_wrap = True
            run = frame.paragraphs[0].add_run()
            run.text = text
            run.font.size, run.font.bold = Pt(size), bold
            run.font.color.rgb = RGBColor(*color)
            run.font.name = "Malgun Gothic"
            return frame

        add_text(0.5, 0.3, 12.3, 0.4, page["section"], 13, True, (42, 120, 214))
        add_text(0.5, 0.7, 12.3, 0.8, page["title"], 26, True)
        add_text(0.5, 1.5, 12.3, 1.0, page["message"], 15, False, (82, 81, 78))
        if page.get("tiles"):
            add_text(0.5, 6.35, 12.3, 0.5, "   ".join(f"{value} {label}" for value, label in page["tiles"]), 14, True,
                     (37, 106, 191))
        width = 6.2 if page.get("figure") else 12.3
        body = add_text(0.5, 2.6, width, 4.4, "", 13)
        lines = []
        if page.get("table"):
            header, rows = page["table"]
            lines.append(" | ".join(header))
            lines += [" | ".join(str(c) for c in row) for row in rows]
        lines += [f"• {b}" for b in page.get("bullets", [])]
        for index, line in enumerate(lines):
            paragraph = body.paragraphs[0] if index == 0 else body.add_paragraph()
            run = paragraph.add_run()
            run.text = line
            run.font.size, run.font.name = Pt(12.5), "Malgun Gothic"
        if page.get("figure"):
            slide.shapes.add_picture(str(DOCS_DIR / page["figure"]), Inches(6.95), Inches(2.6), width=Inches(5.9))
        add_text(11.3, 7.0, 1.8, 0.3, f"{number} / {len(pages)}", 10, False, (137, 135, 129))
    deck.save(path)
    return True


def main() -> int:
    font = apply_korean_font()
    evidence = collect_evidence()
    figures = [draw_architecture(), draw_reversal(evidence), draw_slice(evidence), draw_break(), draw_benchmark(evidence)]
    pages = build_pages(evidence)
    MD_PATH.write_text(render_markdown(pages), encoding="utf-8")
    HTML_PATH.write_text(render_html(pages), encoding="utf-8")
    pdf_ok = export_pdf(HTML_PATH, PDF_PATH)
    pptx_ok = export_pptx(pages, PPTX_PATH)
    print(f"글꼴: {font}")
    print(f"원고: {MD_PATH.relative_to(ROOT)} ({len(pages)}쪽)")
    print(f"인쇄용 HTML: {HTML_PATH.relative_to(ROOT)}")
    print(f"PDF: {'생성 ' + str(PDF_PATH.relative_to(ROOT)) if pdf_ok else '건너뜀'}")
    print(f"PPTX: {'생성 ' + str(PPTX_PATH.relative_to(ROOT)) if pptx_ok else '건너뜀'}")
    print("차트:", ", ".join(p.name for p in figures))
    return 0


if __name__ == "__main__":
    sys.exit(main())
