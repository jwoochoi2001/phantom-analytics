"""캠페인 18 처치/대조 가구의 '발행 전' 특성(pre_*)을 비교해 두 집단이 매칭 전에
얼마나 다른지 확인한다.

- 각 변수의 처치/대조 평균(또는 비율)과 평균차이를 계산한다.
- 표준화 평균차이(SMD, Standardized Mean Difference)로 척도가 다른 변수들의 차이 크기를
  동일 기준으로 비교한다. |SMD| >= 0.1 을 "분포 차이가 크다"는 기준으로 쓴다
  (성향점수 문헌에서 흔히 쓰는 불균형 판단 기준).
- SMD가 큰 변수는 그래프(Love plot + 분포 비교)로 보여준다.

주의: 이 단계는 아직 매칭 전이다. 여기서 SMD가 큰 변수는 "캠페인 수신 여부와 관련되어
보이는 특성"의 후보일 뿐이며, 이 자체가 인과관계를 의미하지 않는다
(DATA_DICTIONARY.md: 성향점수는 캠페인 수신 가능성을 나타낸다).

입력: outputs/campaign_18/analysis_data.csv (household_key, group, pre_* 변수 포함)
출력: outputs/campaign_18/pre_period_balance_love_plot.png
      outputs/campaign_18/pre_period_balance_distributions.png
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Malgun Gothic"  # Windows 한글 폰트 (한글 깨짐 방지)
matplotlib.rcParams["axes.unicode_minus"] = False

sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = Path(__file__).resolve().parent.parent / "outputs" / "campaign_18" / "analysis_data.csv"
OUT_DIR = DATA_PATH.parent

# 발행 전 특성만 사용한다 (결과변수·캠페인 기간 정보는 성향점수 입력에서 제외 — DATA_DICTIONARY.md 규칙)
CONTINUOUS_VARS = [
    "recency_days", "pre_baskets", "pre_sales", "pre_quantity",
    "pre_target_purchase_count", "pre_target_quantity", "pre_target_sales",
    "pre_coupon_redemptions", "pre_campaign_count",
]
BINARY_VARS = ["pre_target_any"]

VAR_LABELS = {
    "recency_days": "recency_days (최근 구매 이후 기간)",
    "pre_baskets": "pre_baskets (발행 전 장바구니 수)",
    "pre_sales": "pre_sales (발행 전 구매금액)",
    "pre_quantity": "pre_quantity (발행 전 구매수량)",
    "pre_target_any": "pre_target_any (대상 상품 구매 경험 비율)",
    "pre_target_purchase_count": "pre_target_purchase_count (대상 상품 구매건수)",
    "pre_target_quantity": "pre_target_quantity (대상 상품 구매수량)",
    "pre_target_sales": "pre_target_sales (대상 상품 구매금액)",
    "pre_coupon_redemptions": "pre_coupon_redemptions (과거 쿠폰 사용횟수)",
    "pre_campaign_count": "pre_campaign_count (과거 캠페인 수신횟수)",
}

SMD_FLAG_THRESHOLD = 0.1  # 이 이상이면 "분포 차이가 크다"고 판단 (성향점수 문헌 관례)


def smd_continuous(t: pd.Series, c: pd.Series) -> float:
    pooled_sd = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
    return 0.0 if pooled_sd == 0 else (t.mean() - c.mean()) / pooled_sd


def smd_binary(t: pd.Series, c: pd.Series) -> float:
    p_t, p_c = t.mean(), c.mean()
    pooled = np.sqrt((p_t * (1 - p_t) + p_c * (1 - p_c)) / 2)
    return 0.0 if pooled == 0 else (p_t - p_c) / pooled


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    t_mask, c_mask = df["group"] == "처치", df["group"] == "대조"
    n_t, n_c = t_mask.sum(), c_mask.sum()

    print(f"{'='*72}\n캠페인 18 처치 vs 대조: 발행 전 특성 균형 점검 (매칭 전)\n{'='*72}")
    print(f"분석표: {DATA_PATH.name} | 처치 {n_t}명 / 대조 {n_c}명\n")

    rows = []
    for col in CONTINUOUS_VARS:
        t_vals, c_vals = df.loc[t_mask, col], df.loc[c_mask, col]
        mean_t, mean_c = t_vals.mean(), c_vals.mean()
        rows.append(
            {
                "변수": col,
                "구분": "연속형(평균)",
                "처치": round(mean_t, 2),
                "대조": round(mean_c, 2),
                "평균차이": round(mean_t - mean_c, 2),
                "SMD": round(smd_continuous(t_vals, c_vals), 3),
            }
        )
    for col in BINARY_VARS:
        t_vals, c_vals = df.loc[t_mask, col], df.loc[c_mask, col]
        p_t, p_c = t_vals.mean(), c_vals.mean()
        rows.append(
            {
                "변수": col,
                "구분": "이진형(비율)",
                "처치": round(p_t, 3),
                "대조": round(p_c, 3),
                "평균차이": round(p_t - p_c, 3),
                "SMD": round(smd_binary(t_vals, c_vals), 3),
            }
        )

    balance = pd.DataFrame(rows).sort_values("SMD", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    balance["분포차이큼(|SMD|>=0.1)"] = balance["SMD"].abs() >= SMD_FLAG_THRESHOLD

    print("[발행 전 특성 균형표] SMD 절대값 큰 순")
    print(balance.to_string(index=False))
    print()

    flagged = balance.loc[balance["분포차이큼(|SMD|>=0.1)"], "변수"].tolist()
    print(f"|SMD| >= {SMD_FLAG_THRESHOLD} 로 분포 차이가 큰 변수 {len(flagged)}개: {flagged}\n")

    # -----------------------------------------------------------------
    # 그래프 1: Love plot (변수별 SMD)
    # -----------------------------------------------------------------
    plot_df = balance.sort_values("SMD", key=lambda s: s.abs())
    colors = ["#d62728" if v else "#888888" for v in plot_df["분포차이큼(|SMD|>=0.1)"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hlines(y=plot_df["변수"], xmin=0, xmax=plot_df["SMD"], color=colors, linewidth=2)
    ax.scatter(plot_df["SMD"], plot_df["변수"], color=colors, zorder=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(SMD_FLAG_THRESHOLD, color="grey", linestyle="--", linewidth=1)
    ax.axvline(-SMD_FLAG_THRESHOLD, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("표준화 평균차이 (SMD, 처치-대조)")
    ax.set_title("캠페인 18: 발행 전 특성 처치-대조 SMD (매칭 전)")
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels([VAR_LABELS[v] for v in plot_df["변수"]], fontsize=8)
    fig.tight_layout()
    love_path = OUT_DIR / "pre_period_balance_love_plot.png"
    fig.savefig(love_path, dpi=150)
    plt.close(fig)
    print(f"저장: {love_path}")

    # -----------------------------------------------------------------
    # 그래프 2: SMD 큰 변수의 분포 비교 (박스플롯, 상위 4개까지)
    #   금액/수량형 변수는 왜곡이 커서 시각화에서만 99분위로 절단한다(원본 미변경).
    # -----------------------------------------------------------------
    top_vars = [v for v in balance.loc[balance["SMD"].abs().sort_values(ascending=False).index, "변수"]][:4]
    top_vars = [v for v in top_vars if v in flagged] or top_vars[:2]

    fig, axes = plt.subplots(1, len(top_vars), figsize=(4.2 * len(top_vars), 4.5))
    if len(top_vars) == 1:
        axes = [axes]
    for ax, col in zip(axes, top_vars):
        plot_data = df[[col, "group"]].copy()
        cap = plot_data[col].quantile(0.99)
        capped = plot_data[col].astype(float).clip(upper=cap)
        n_capped = int((plot_data[col] > cap).sum())
        data_by_group = [capped[df["group"] == g] for g in ["처치", "대조"]]
        ax.boxplot(data_by_group, tick_labels=["처치", "대조"], showmeans=True)
        title = VAR_LABELS[col]
        if n_capped:
            title += f"\n(상위 1%, {n_capped}건 시각화용 절단)"
        ax.set_title(title, fontsize=9)
    fig.suptitle("캠페인 18: 처치 vs 대조 발행 전 분포 비교 (SMD 큰 변수)")
    fig.tight_layout()
    dist_path = OUT_DIR / "pre_period_balance_distributions.png"
    fig.savefig(dist_path, dpi=150)
    plt.close(fig)
    print(f"저장: {dist_path}\n")

    print("[해석 메모]")
    print("  - 위 SMD는 아직 매칭을 하지 않은 원표본 기준이다.")
    print("  - |SMD| >= 0.1인 변수는 처치집단과 대조집단 사이에서 분포가 다르게 나타난 특성으로,")
    print("    '캠페인 18을 받을 가능성'과 관련되어 보이는 후보 변수다(상관 관찰일 뿐 인과 아님).")
    print("  - 이 변수들은 성향점수 모형의 입력 후보이며, 매칭 후에는 SMD가 0에 가깝게")
    print("    줄어드는지 다시 확인해야 한다(CLAUDE.md: 매칭 전후 표준화 평균차이 확인).")


if __name__ == "__main__":
    main()
