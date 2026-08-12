"""캠페인 18: 처치/대조 p_score 분포를 겹쳐서 그리고, 공통지지영역(common support)을
계산한다. 공통지지영역 안/밖 가구 수를 집단별로 세고, 영역 밖 가구의 발행 전 특성을
안쪽 가구와 비교해 왜 벗어났는지 살펴본다.

공통지지영역 정의(표준 min-max 겹침 방식):
    [max(처치 p_score 최솟값, 대조 p_score 최솟값), min(처치 p_score 최댓값, 대조 p_score 최댓값)]
이 구간 밖의 가구는 반대 집단에서 비슷한 p_score를 가진 짝이 존재하지 않는 가구다.

입력: outputs/campaign_18/analysis_data.csv (p_score 포함, campaign18_propensity_score.py 결과)
출력: outputs/campaign_18/p_score_common_support.png
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = Path(__file__).resolve().parent.parent / "outputs" / "campaign_18" / "analysis_data.csv"
OUT_DIR = DATA_PATH.parent

PRE_VARS_RAW = [
    "recency_days", "pre_baskets", "pre_sales", "pre_quantity",
    "pre_target_purchase_count", "pre_coupon_redemptions", "pre_campaign_count",
]


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    assert "p_score" in df.columns, "analysis_data.csv에 p_score가 없음 — campaign18_propensity_score.py 먼저 실행 필요"

    t = df.loc[df["group"] == "처치", "p_score"]
    c = df.loc[df["group"] == "대조", "p_score"]

    # -----------------------------------------------------------------
    # 공통지지영역 계산
    # -----------------------------------------------------------------
    lo = max(t.min(), c.min())
    hi = min(t.max(), c.max())
    df["in_support"] = df["p_score"].between(lo, hi)

    print(f"{'='*72}\n캠페인 18: p_score 공통지지영역\n{'='*72}")
    print(f"처치 p_score 범위: [{t.min():.4f}, {t.max():.4f}]")
    print(f"대조 p_score 범위: [{c.min():.4f}, {c.max():.4f}]")
    print(f"공통지지영역 = [max({t.min():.4f}, {c.min():.4f}), min({t.max():.4f}, {c.max():.4f})] "
          f"= [{lo:.4f}, {hi:.4f}]\n")

    # -----------------------------------------------------------------
    # 집단별 안/밖 가구 수
    # -----------------------------------------------------------------
    counts = df.groupby(["group", "in_support"]).size().unstack(fill_value=0)
    counts = counts.rename(columns={True: "영역 안", False: "영역 밖"})
    counts["합계"] = counts.sum(axis=1)
    counts["영역 밖 비율"] = (counts["영역 밖"] / counts["합계"] * 100).round(1)
    print("[집단별 공통지지영역 안/밖 가구 수]")
    print(counts.to_string())
    print()

    outside = df.loc[~df["in_support"]].copy()
    inside = df.loc[df["in_support"]].copy()
    print(f"영역 밖 가구 총 {len(outside)}명 (처치 {(outside['group']=='처치').sum()}, "
          f"대조 {(outside['group']=='대조').sum()}) — 전체 {len(df)}명 중 {100*len(outside)/len(df):.1f}%\n")

    # -----------------------------------------------------------------
    # 영역 밖 가구의 발행 전 특성: 같은 집단의 영역 안 가구와 비교
    # -----------------------------------------------------------------
    print("[영역 밖 vs 영역 안: 발행 전 특성 평균 비교 (집단별)]")
    rows = []
    for g in ["처치", "대조"]:
        out_g = outside.loc[outside["group"] == g]
        in_g = inside.loc[inside["group"] == g]
        if out_g.empty:
            continue
        for v in PRE_VARS_RAW:
            rows.append(
                {
                    "집단": g,
                    "변수": v,
                    "영역 밖 평균": round(out_g[v].mean(), 2),
                    "영역 안 평균": round(in_g[v].mean(), 2),
                    "영역 밖 n": len(out_g),
                }
            )
    compare_df = pd.DataFrame(rows)
    print(compare_df.to_string(index=False))
    print()

    print("[공통지지영역 밖 가구 개별 목록 (p_score, 주요 발행 전 특성)]")
    list_cols = ["household_key", "group", "p_score"] + PRE_VARS_RAW
    print(outside[list_cols].sort_values(["group", "p_score"]).to_string(index=False))
    print()

    # -----------------------------------------------------------------
    # 해석 메모 자동 생성 (수치 기반)
    # -----------------------------------------------------------------
    out_t = outside.loc[outside["group"] == "처치"]
    out_c = outside.loc[outside["group"] == "대조"]
    print("[해석 메모]")
    if not out_t.empty:
        in_t = inside.loc[inside["group"] == "처치"]
        print(
            f"  - 영역 밖 처치가구 {len(out_t)}명: p_score가 대조집단 최댓값({c.max():.4f})보다 높아 "
            f"대응할 대조가구가 없다. pre_baskets 평균 {out_t['pre_baskets'].mean():.0f}회로 "
            f"영역 안 처치가구 평균({in_t['pre_baskets'].mean():.0f}회)보다 "
            f"{'높다' if out_t['pre_baskets'].mean() > in_t['pre_baskets'].mean() else '낮다'} "
            "→ 극단적으로 활발한(혹은 비활동) 소수 가구로 보인다."
        )
    if not out_c.empty:
        in_c = inside.loc[inside["group"] == "대조"]
        print(
            f"  - 영역 밖 대조가구 {len(out_c)}명: p_score가 처치집단 최솟값({t.min():.4f})보다 낮아 "
            f"대응할 처치가구가 없다. pre_baskets 평균 {out_c['pre_baskets'].mean():.0f}회로 "
            f"영역 안 대조가구 평균({in_c['pre_baskets'].mean():.0f}회)보다 "
            f"{'높다' if out_c['pre_baskets'].mean() > in_c['pre_baskets'].mean() else '낮다'}, "
            f"recency_days 평균 {out_c['recency_days'].mean():.0f}일로 영역 안 대조가구"
            f"({in_c['recency_days'].mean():.0f}일)보다 "
            f"{'김(비활동적)' if out_c['recency_days'].mean() > in_c['recency_days'].mean() else '짧음(활동적)'} "
            "→ 활동성이 매우 낮아 캠페인 18을 받았을 가능성 자체가 낮게 추정된 가구로 보인다."
        )
    print("  - 이 가구들은 매칭 단계에서 caliper·공통지지영역 기준으로 자연스럽게 제외되며, "
          "제외 자체가 '비교 불가능한 극단 가구를 걸러내는' 정상적인 절차다.")

    # -----------------------------------------------------------------
    # 그래프: 처치/대조 p_score를 겹쳐서 그리고 공통지지영역을 음영으로 표시
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 41)
    ax.hist(c, bins=bins, alpha=0.55, label=f"대조 (n={len(c)})", color="#1f77b4", density=True)
    ax.hist(t, bins=bins, alpha=0.55, label=f"처치 (n={len(t)})", color="#d62728", density=True)
    ax.axvspan(lo, hi, color="green", alpha=0.08, label=f"공통지지영역 [{lo:.3f}, {hi:.3f}]")
    ax.axvline(lo, color="green", linestyle="--", linewidth=1)
    ax.axvline(hi, color="green", linestyle="--", linewidth=1)
    ax.set_xlabel("p_score (캠페인 18 수신확률)")
    ax.set_ylabel("밀도")
    ax.set_title("캠페인 18: 처치/대조 p_score 분포 겹침 + 공통지지영역")
    ax.legend()
    fig.tight_layout()
    plot_path = OUT_DIR / "p_score_common_support.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n저장: {plot_path}")


if __name__ == "__main__":
    main()
