"""차트 공통 스타일: OS 별 한글 글꼴 자동 지정(깨짐 '□□□' 방지) + 검증된 기본 팔레트.

Windows -> Malgun Gothic, Linux -> NanumGothic / UnDotum, macOS -> AppleGothic.
matplotlib 는 axes.unicode_minus=False 를 강제해 음수 부호 깨짐을 막는다.
팔레트: 범주 1 파랑(#2a78d6), 2 주황(#eb6834), 3 청록(#1baf7a) - 3계열까지 색각이상 구분 검증된 순서.
"""
from __future__ import annotations

import platform
from functools import lru_cache

FONT_CANDIDATES = {
    "Windows": ["Malgun Gothic", "맑은 고딕", "Gulim"],
    "Linux": ["NanumGothic", "UnDotum", "Noto Sans CJK KR", "Noto Sans KR"],
    "Darwin": ["AppleGothic", "Apple SD Gothic Neo"],
}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
       "surface": "#fcfcfb"}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


@lru_cache(maxsize=1)
def pick_korean_font() -> str:
    system = platform.system()
    candidates = FONT_CANDIDATES.get(system, []) + [f for names in FONT_CANDIDATES.values() for f in names]
    try:
        from matplotlib import font_manager
        installed = {font.name for font in font_manager.fontManager.ttflist}
        for name in candidates:
            if name in installed:
                return name
    except Exception:
        pass
    return candidates[0] if candidates else "sans-serif"


def apply_korean_font() -> str:
    """matplotlib 전역 글꼴을 한글 글꼴로 바꾸고 음수 부호 깨짐을 막는다. 선택된 글꼴명을 돌려준다."""
    font = pick_korean_font()
    try:
        import matplotlib
        matplotlib.rcParams["font.family"] = font
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    return font


def build_font_family() -> str:
    return f"{pick_korean_font()}, 'Malgun Gothic', 'Apple SD Gothic Neo', 'NanumGothic', 'Noto Sans KR', sans-serif"


def style_figure(figure, title: str = "", height: int = 300):
    """얇은 막대·2px 선·옅은 격자·범례 상단 배치."""
    figure.update_layout(
        title={"text": title, "font": {"size": 13}} if title else None,
        height=height, margin={"l": 8, "r": 8, "t": 36 if title else 12, "b": 8},
        font={"family": build_font_family(), "size": 12},
        legend={"orientation": "h", "yanchor": "top", "y": -0.22, "x": 0},
        hoverlabel={"font": {"family": build_font_family()}},
        bargap=0.35,
    )
    figure.update_xaxes(showgrid=False, linecolor=INK["axis"], tickfont={"color": INK["muted"]})
    figure.update_yaxes(gridcolor=INK["grid"], zeroline=False, tickfont={"color": INK["muted"]})
    return figure
