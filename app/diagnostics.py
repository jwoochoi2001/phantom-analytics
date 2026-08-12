"""효과 결과보다 먼저 보여줄 매칭 품질 진단 (성향점수 분포, 공통지지영역, 매칭률,
매칭 전후 SMD)을 계산하고 Plotly figure로 만든다.

results.json에는 진단에 필요한 가구별 값(개별 p_score 등)이 저장되어 있지 않으므로,
analysis_data.csv를 입력으로 pipeline.estimate_effect의 순수 함수(fit_propensity,
check_common_support, run_matching, compute_balance, smd_continuous)를 그대로 다시
호출해 재현한다. 같은 함수·같은 입력이므로 results.json에 저장된 공통영역·매칭률·
SMD 수치와 동일한 값이 나온다 — 로직을 중복 구현하지 않는다.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

PIPELINE_DIR = Path(__file__).resolve().parent.parent / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))
from prepare_data import FINAL_PROPENSITY_VARS  # noqa: E402
from estimate_effect import (  # noqa: E402
    check_common_support,
    compute_balance,
    fit_propensity,
    run_matching,
    smd_continuous,
)

COLOR_TREATMENT = "#d62728"
COLOR_CONTROL = "#1f77b4"
COLOR_SUPPORT = "rgba(46, 160, 67, 0.12)"
SMD_THRESHOLD = 0.1

VAR_LABELS = {
    "pre_recency_capped": "최근성 (최근 구매 이후 일수, 상한 적용)",
    "log_pre_baskets": "발행 전 장바구니 수 (log)",
    "log_pre_sales": "발행 전 구매금액 (log)",
    "pre_target_share": "발행 전 대상상품 구매 비중",
    "pre_coupon_user": "발행 전 쿠폰 사용 경험",
    "pre_campaign_count_c": "과거 캠페인 수신 횟수 (상한 적용)",
    "has_demographic": "인구통계 보유 여부",
}


@dataclass
class DiagnosticsBundle:
    scored: pd.DataFrame       # 전체 표본 + p_score, logit_p, in_support
    lo: float
    hi: float
    pairs: list
    caliper: float
    smd_after: pd.DataFrame    # 매칭 후 SMD (변수, SMD)
    smd_compare: pd.DataFrame  # 변수별 매칭 전/후 SMD 비교표


def compute_diagnostics(analysis_df: pd.DataFrame) -> DiagnosticsBundle:
    """analysis_data.csv를 읽어 진단용 성향점수·공통영역·매칭·SMD를 재계산한다."""
    scored = fit_propensity(analysis_df)
    scored, lo, hi, _gate = check_common_support(scored)
    pairs, caliper, _gate = run_matching(scored)

    t_all = scored.loc[scored["group"] == "처치"]
    c_all = scored.loc[scored["group"] == "대조"]
    before_rows = [
        {"변수": v, "SMD_매칭전": round(smd_continuous(t_all[v].to_numpy(dtype=float), c_all[v].to_numpy(dtype=float)), 3)}
        for v in FINAL_PROPENSITY_VARS
    ]
    before_df = pd.DataFrame(before_rows)

    if pairs:
        smd_after, _max_abs = compute_balance(scored, pairs)
        smd_after = smd_after.rename(columns={"SMD": "SMD_매칭후"})
    else:
        smd_after = pd.DataFrame({"변수": FINAL_PROPENSITY_VARS, "SMD_매칭후": [None] * len(FINAL_PROPENSITY_VARS)})

    smd_compare = before_df.merge(smd_after, on="변수", how="left")
    smd_compare["변수명"] = smd_compare["변수"].map(VAR_LABELS).fillna(smd_compare["변수"])

    return DiagnosticsBundle(scored=scored, lo=lo, hi=hi, pairs=pairs, caliper=caliper,
                              smd_after=smd_after, smd_compare=smd_compare)


def propensity_distribution_figure(bundle: DiagnosticsBundle) -> go.Figure:
    t = bundle.scored.loc[bundle.scored["group"] == "처치", "p_score"]
    c = bundle.scored.loc[bundle.scored["group"] == "대조", "p_score"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=c, name=f"대조 (n={len(c)})", histnorm="probability density",
                                marker_color=COLOR_CONTROL, opacity=0.55,
                                xbins=dict(start=0, end=1, size=0.025)))
    fig.add_trace(go.Histogram(x=t, name=f"처치 (n={len(t)})", histnorm="probability density",
                                marker_color=COLOR_TREATMENT, opacity=0.55,
                                xbins=dict(start=0, end=1, size=0.025)))
    fig.add_vrect(x0=bundle.lo, x1=bundle.hi, fillcolor=COLOR_SUPPORT, line_width=0,
                  annotation_text="공통지지영역", annotation_position="top left")
    fig.update_layout(
        barmode="overlay", height=340, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="성향점수 (캠페인 수신확률)", yaxis_title="밀도",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
    )
    return fig


def common_support_table(bundle: DiagnosticsBundle) -> pd.DataFrame:
    df = bundle.scored
    rows = []
    for g in ["처치", "대조"]:
        sub = df.loc[df["group"] == g]
        n_total = len(sub)
        n_in = int(sub["in_support"].sum())
        rows.append({
            "집단": g, "전체 표본": n_total, "공통영역 내": n_in,
            "공통영역 밖": n_total - n_in,
            "공통영역 밖 비율": f"{(n_total - n_in) / n_total:.1%}" if n_total else "-",
        })
    return pd.DataFrame(rows)


def love_plot_figure(bundle: DiagnosticsBundle) -> go.Figure:
    d = bundle.smd_compare.sort_values("SMD_매칭전", key=lambda s: s.abs())
    fig = go.Figure()
    for x_col, name, color, symbol in [
        ("SMD_매칭전", "매칭 전", "#9aa7b5", "circle-open"),
        ("SMD_매칭후", "매칭 후", "#2f7d70", "circle"),
    ]:
        fig.add_trace(go.Scatter(
            x=d[x_col], y=d["변수명"], mode="markers", name=name,
            marker=dict(size=11, color=color, symbol=symbol, line=dict(width=2)),
        ))
    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.add_vline(x=SMD_THRESHOLD, line_color="grey", line_dash="dash", line_width=1)
    fig.add_vline(x=-SMD_THRESHOLD, line_color="grey", line_dash="dash", line_width=1)
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="표준화 평균차이 (SMD, 처치-대조)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        template="plotly_white",
    )
    return fig
