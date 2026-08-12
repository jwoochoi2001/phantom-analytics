"""캠페인 18: 확정된 7개 변수로 성향점수(p_score, 캠페인 수신확률)를 계산하고
analysis_data.csv에 추가한 뒤 집단별 분포를 확인한다.

이 스크립트는 아직 pipeline/으로 묶지 않은 순차 실행 스크립트다. 각 단계 결과의
행·열 수를 화면에 출력해 중간 산출물을 눈으로 확인할 수 있게 하고, 나중에
pipeline/prepare_data.py·pipeline/estimate_effect.py로 옮길 "처리규칙"은
# [PIPELINE 이동 대상] 주석으로 코드에 표시해 둔다.

변수 출처: analysis/campaign18_propensity_variables.md (7개 확정 변수)
전제: 모집단은 처치 510 + 대조 335 = 845가구 (campaign18_build_analysis_data.py 결과).
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_PATH = Path(__file__).resolve().parent.parent / "outputs" / "campaign_18" / "analysis_data.csv"
OUT_DIR = DATA_PATH.parent

TARGET_CAMPAIGN = 18
START_DAY, END_DAY = 587, 642
FINAL_VARS = [
    "pre_recency_capped", "log_pre_baskets", "log_pre_sales",
    "pre_target_share", "pre_coupon_user", "pre_campaign_count_c", "has_demographic",
]


def step(title: str) -> None:
    print(f"\n{'-'*72}\n{title}\n{'-'*72}")


def main() -> None:
    # ===================================================================
    # 0단계: 분석표 로드
    # ===================================================================
    step("0단계: analysis_data.csv 로드")
    df = pd.read_csv(DATA_PATH)
    print(f"  shape = {df.shape[0]}행 x {df.shape[1]}열")
    print(f"  열 목록: {list(df.columns)}")

    # ===================================================================
    # 1단계: 변수 계산에 필요한 원본 보조 파일 로드
    # ===================================================================
    step("1단계: 보조 원본 파일 로드")
    desc = pd.read_csv(RAW / "campaign_desc.csv")
    table = pd.read_csv(RAW / "campaign_table.csv")
    hh_demo = pd.read_csv(RAW / "hh_demographic.csv")
    print(f"  campaign_desc.csv   shape = {desc.shape}")
    print(f"  campaign_table.csv  shape = {table.shape}")
    print(f"  hh_demographic.csv  shape = {hh_demo.shape}")

    # ===================================================================
    # 2단계: [PIPELINE 이동 대상 → prepare_data.py] 전처리 — 7개 입력변수 생성
    #   출처: analysis/campaign18_propensity_variables.md 확정 내용 그대로.
    # ===================================================================
    step("2단계: 전처리 — 성향점수 입력변수 7개 생성")

    # (1) pre_recency_capped: 587-마지막구매일, 결측(발행 전 구매 없음)은 586, 상한 365
    df["pre_recency_capped"] = df["recency_days"].fillna(START_DAY - 1).clip(upper=365)

    # (2)(3) 로그변환: 오른쪽 꼬리 완화
    df["log_pre_baskets"] = np.log1p(df["pre_baskets"])
    df["log_pre_sales"] = np.log1p(df["pre_sales"])

    # (4) pre_target_share: 대상 상품 구매금액 비중, 분모 0이면 0
    df["pre_target_share"] = np.where(df["pre_sales"] > 0, df["pre_target_sales"] / df["pre_sales"], 0.0)

    # (5) pre_coupon_user: 과거 쿠폰 사용 경험 이진화
    df["pre_coupon_user"] = (df["pre_coupon_redemptions"] > 0).astype(int)

    # (6) pre_campaign_count_c: "완전히 종료된"(END_DAY < 587) 과거 캠페인 수, 상한 6
    completed_campaign_ids = set(
        desc.loc[(desc["CAMPAIGN"] != TARGET_CAMPAIGN) & (desc["END_DAY"] < START_DAY), "CAMPAIGN"]
    )
    households = set(df["household_key"])
    table_completed = table.loc[
        table["household_key"].isin(households) & table["CAMPAIGN"].isin(completed_campaign_ids)
    ]
    pre_campaign_count_raw = table_completed.groupby("household_key")["CAMPAIGN"].nunique()
    df = df.set_index("household_key")
    df["pre_campaign_count_c"] = pre_campaign_count_raw.reindex(df.index).fillna(0).clip(upper=6)
    df = df.reset_index()

    # (7) has_demographic: hh_demographic.csv 존재 여부 (0/1, 값 자체는 미사용)
    demo_households = set(hh_demo["household_key"])
    df["has_demographic"] = df["household_key"].isin(demo_households).astype(int)

    feature_df = df[["household_key", "group"] + FINAL_VARS].copy()
    print(f"  전처리 결과 feature_df shape = {feature_df.shape[0]}행 x {feature_df.shape[1]}열")
    print(f"  입력변수 7개: {FINAL_VARS}")
    assert feature_df[FINAL_VARS].isna().sum().sum() == 0, "입력변수에 결측이 남아 있음"
    print("  결측 확인: 0건 (통과)")

    # ===================================================================
    # 3단계: [PIPELINE 이동 대상 → prepare_data.py] 표준화
    #   로지스틱 회귀 계수 크기를 변수 간에 비교 가능하게 하고 수렴을 돕는다.
    #   표준화는 845가구 전체(분석 표본 전체)를 기준으로 적합한다 — 예측용
    #   train/test 분리가 아니라 "이 표본에서의 수신확률 추정"이 목적이기 때문이다.
    # ===================================================================
    step("3단계: 표준화 (StandardScaler)")
    X_raw = feature_df[FINAL_VARS].to_numpy(dtype=float)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)
    print(f"  X_std shape = {X_std.shape[0]}행 x {X_std.shape[1]}열")
    print(f"  변수별 평균(표준화 후, 0에 근접해야 함): {np.round(X_std.mean(axis=0), 4)}")
    print(f"  변수별 표준편차(표준화 후, 1에 근접해야 함): {np.round(X_std.std(axis=0), 4)}")

    # ===================================================================
    # 4단계: [PIPELINE 이동 대상 → estimate_effect.py] 로지스틱 회귀 학습
    #   y = treatment(18번 수신=1 / 대조=0), X = 표준화된 7개 발행 전 변수.
    #   class_weight는 균형을 맞추지 않는다 — 성향점수는 이 표본에서 실제 관측된
    #   수신확률을 추정하는 것이 목적이며, 인위적으로 50:50으로 맞추면 확률의
    #   의미가 바뀐다.
    # ===================================================================
    step("4단계: 로지스틱 회귀 학습")
    y = (df["group"] == "처치").astype(int).to_numpy()
    print(f"  학습 데이터: X={X_std.shape}, y={y.shape} (y=1: 처치 {y.sum()}명, y=0: 대조 {len(y)-y.sum()}명)")

    model = LogisticRegression(max_iter=2000)
    model.fit(X_std, y)
    coef_table = pd.DataFrame({"변수": FINAL_VARS, "표준화 계수": np.round(model.coef_[0], 3)})
    print(f"  절편(intercept): {model.intercept_[0]:.3f}")
    print(coef_table.to_string(index=False))

    # ===================================================================
    # 5단계: [PIPELINE 이동 대상 → estimate_effect.py] p_score 계산
    # ===================================================================
    step("5단계: p_score(캠페인 수신확률) 계산")
    p_score = model.predict_proba(X_std)[:, 1]
    print(f"  p_score shape = {p_score.shape} (845가구 각각에 대한 확률)")
    print(f"  p_score 범위: {p_score.min():.4f} ~ {p_score.max():.4f}")

    # ===================================================================
    # 6단계: 분석표에 p_score 추가 후 저장
    # ===================================================================
    step("6단계: analysis_data.csv에 p_score 추가")
    n_cols_before = 19  # campaign18_build_analysis_data.py가 저장한 원래 열 수
    df["p_score"] = p_score
    n_new_cols = df.shape[1] - n_cols_before
    print(f"  갱신된 분석표 shape = {df.shape[0]}행 x {df.shape[1]}열")
    print(
        f"  (원래 {n_cols_before}열 + 신규 {n_new_cols}열: 2단계 입력변수 7개 "
        f"{FINAL_VARS} + p_score 1개)"
    )
    df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")
    print(f"  저장 완료: {DATA_PATH}")

    # ===================================================================
    # 7단계: p_score 집단별 분포 확인 (매칭 전 공통지지영역 예비 점검)
    # ===================================================================
    step("7단계: p_score 집단별 분포")
    dist = df.groupby("group")["p_score"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
    print(dist.round(4).to_string())

    t_range = df.loc[df["group"] == "처치", "p_score"]
    c_range = df.loc[df["group"] == "대조", "p_score"]
    overlap_lo, overlap_hi = max(t_range.min(), c_range.min()), min(t_range.max(), c_range.max())
    n_t_in = ((t_range >= overlap_lo) & (t_range <= overlap_hi)).sum()
    n_c_in = ((c_range >= overlap_lo) & (c_range <= overlap_hi)).sum()
    print(f"\n  공통지지영역(단순 min-max 겹침, 예비 점검용): [{overlap_lo:.4f}, {overlap_hi:.4f}]")
    print(f"  처치 {n_t_in}/{len(t_range)}명 ({100*n_t_in/len(t_range):.1f}%), "
          f"대조 {n_c_in}/{len(c_range)}명 ({100*n_c_in/len(c_range):.1f}%) 이 구간 안에 위치")
    print("  ※ 정식 공통지지영역·매칭률 판정은 이후 매칭 단계에서 별도로 수행한다.")

    # ===================================================================
    # 8단계: 분포 그래프 (히스토그램 + 밀도)
    # ===================================================================
    step("8단계: p_score 분포 그래프 저장")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].hist(t_range, bins=30, alpha=0.6, label=f"처치 (n={len(t_range)})", color="#d62728", density=True)
    axes[0].hist(c_range, bins=30, alpha=0.6, label=f"대조 (n={len(c_range)})", color="#1f77b4", density=True)
    axes[0].set_xlabel("p_score (캠페인 18 수신확률)")
    axes[0].set_ylabel("밀도")
    axes[0].set_title("p_score 분포 (히스토그램)")
    axes[0].legend()

    box_data = [t_range, c_range]
    axes[1].boxplot(box_data, tick_labels=["처치", "대조"], showmeans=True)
    axes[1].set_ylabel("p_score")
    axes[1].set_title("p_score 분포 (박스플롯)")

    fig.suptitle("캠페인 18: 처치/대조 p_score 분포 (매칭 전)")
    fig.tight_layout()
    plot_path = OUT_DIR / "p_score_distribution.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  저장: {plot_path}")

    print(f"\n{'='*72}\n완료 — 다음 단계는 p_score 기반 매칭(caliper 등)이며, 아직 수행하지 않았다.\n{'='*72}")


if __name__ == "__main__":
    main()
