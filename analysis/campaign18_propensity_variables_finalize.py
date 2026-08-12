"""캠페인 18 성향점수 입력변수를 최종 확정한다.

배경: 사용자가 제공한 참고 파일(ps_variables.md)의 변수 엔지니어링 아이디어
(로그변환, 대상상품 비중 변수, 결측 지시자, 상한 절단 등)를 채택하되, 그 파일은
대조군을 1,251명(캠페인을 한 번도 안 받은 916명 포함)으로 잡아 저희가 앞서 확정한
335명(campaign_table.csv 상 캠페인 수신 이력이 있으나 18번·겹치는 캠페인은 안 받은
가구) 기준과 다르다. 사용자가 기존 335명 기준을 유지하기로 결정했으므로, 아래는
같은 845가구(처치 510 + 대조 335) 위에서 참고 파일의 기법을 재계산·재검증한다.

입력: outputs/campaign_18/analysis_data.csv, data/raw/hh_demographic.csv,
      data/raw/campaign_desc.csv, data/raw/campaign_table.csv
출력: analysis/campaign18_propensity_variables.md (최종본으로 갱신)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_PATH = Path(__file__).resolve().parent.parent / "outputs" / "campaign_18" / "analysis_data.csv"
MD_OUT = Path(__file__).resolve().parent / "campaign18_propensity_variables.md"

TARGET_CAMPAIGN = 18
START_DAY, END_DAY = 587, 642


def univariate_auc(x: pd.Series, y: pd.Series) -> float:
    x_arr = x.to_numpy().reshape(-1, 1)
    if np.nanstd(x_arr) == 0:
        return 0.5
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_arr, y)
    p = clf.predict_proba(x_arr)[:, 1]
    return roc_auc_score(y, p)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    desc = pd.read_csv(RAW / "campaign_desc.csv")
    table = pd.read_csv(RAW / "campaign_table.csv")
    hh_demo = pd.read_csv(RAW / "hh_demographic.csv")

    n_t, n_c = (df["group"] == "처치").sum(), (df["group"] == "대조").sum()
    print(f"{'='*72}\n캠페인 18 성향점수 변수 최종 확정 (845가구: 처치 {n_t} / 대조 {n_c})\n{'='*72}\n")

    # -----------------------------------------------------------------
    # 1) pre_recency_capped: 587 - 마지막 구매일, 상한 365, 결측(구매이력 없음)은 586
    # -----------------------------------------------------------------
    # analysis_data.csv의 recency_days = START_DAY - last_purchase_day 이미 계산됨
    n_missing_recency = int(df["recency_days"].isna().sum())
    df["pre_recency_capped"] = df["recency_days"].fillna(START_DAY - 1).clip(upper=365)
    print(f"[1] pre_recency_capped: 결측(발행 전 구매이력 없음) {n_missing_recency}건 → {START_DAY - 1}로 대체, 상한 365일 절단")
    print(f"    범위: {df['pre_recency_capped'].min():.0f} ~ {df['pre_recency_capped'].max():.0f}\n")

    # -----------------------------------------------------------------
    # 2) log_pre_baskets, log_pre_sales : log1p 변환 (오른쪽 꼬리 완화)
    # -----------------------------------------------------------------
    df["log_pre_baskets"] = np.log1p(df["pre_baskets"])
    df["log_pre_sales"] = np.log1p(df["pre_sales"])
    print("[2] log_pre_baskets = log1p(pre_baskets), log_pre_sales = log1p(pre_sales)\n")

    # -----------------------------------------------------------------
    # 3) pre_target_share : 대상 상품 구매금액 / 전체 구매금액 (분모 0이면 0)
    # -----------------------------------------------------------------
    df["pre_target_share"] = np.where(df["pre_sales"] > 0, df["pre_target_sales"] / df["pre_sales"], 0.0)
    corr_share_sales = df["pre_target_share"].corr(df["log_pre_sales"])
    print(f"[3] pre_target_share = pre_target_sales / pre_sales (분모 0 → 0)")
    print(f"    log_pre_sales와 상관계수: {corr_share_sales:.3f} (금액 규모와 낮은 상관이면 채택 근거 성립)\n")

    # -----------------------------------------------------------------
    # 4) pre_coupon_user : 과거 쿠폰 사용 경험 0/1 (원시 횟수는 0이 대다수인 극단분포)
    # -----------------------------------------------------------------
    df["pre_coupon_user"] = (df["pre_coupon_redemptions"] > 0).astype(int)
    zero_share = (df["pre_coupon_redemptions"] == 0).mean()
    print(f"[4] pre_coupon_user = (pre_coupon_redemptions > 0). 원시값 0 비중: {zero_share:.1%} → 이진화로 대체\n")

    # -----------------------------------------------------------------
    # 5) pre_campaign_count_c : "완전히 종료된" 과거 캠페인 수 (END_DAY < 587), 상한 6
    #    주의: START_DAY<587 기준으로 하면 14,15,16,17(587 이전 시작, 이후까지 진행)이
    #    섞여 사후 정보가 유입될 수 있다. 저희 처치/대조는 구조적으로 14~17,19~22를
    #    받지 않지만, 기준 자체는 더 엄격한 END_DAY<587로 다시 계산해 재검증한다.
    # -----------------------------------------------------------------
    completed_campaign_ids = set(
        desc.loc[(desc["CAMPAIGN"] != TARGET_CAMPAIGN) & (desc["END_DAY"] < START_DAY), "CAMPAIGN"]
    )
    started_not_completed = set(
        desc.loc[
            (desc["CAMPAIGN"] != TARGET_CAMPAIGN) & (desc["START_DAY"] < START_DAY) & (desc["END_DAY"] >= START_DAY),
            "CAMPAIGN",
        ]
    )
    print(f"    END_DAY<587 완전종료 캠페인: {sorted(completed_campaign_ids)}")
    print(f"    START_DAY<587이지만 END_DAY>=587(종료 안 됨, 겹침 캠페인): {sorted(started_not_completed)}")

    households = set(df["household_key"])
    table_completed = table.loc[
        table["household_key"].isin(households) & table["CAMPAIGN"].isin(completed_campaign_ids)
    ]
    assert not (set(table_completed["CAMPAIGN"].unique()) & started_not_completed), (
        "완전종료 캠페인 집계에 미종료(겹침) 캠페인이 섞여 있음"
    )
    pre_campaign_count_raw = table_completed.groupby("household_key")["CAMPAIGN"].nunique()
    df = df.set_index("household_key")
    df["pre_campaign_count_c"] = pre_campaign_count_raw.reindex(df.index).fillna(0)
    max_before_cap = df["pre_campaign_count_c"].max()
    df["pre_campaign_count_c"] = df["pre_campaign_count_c"].clip(upper=6)
    df = df.reset_index()
    # 기존 pre_campaign_count(START_DAY 기준)와 값이 동일한지 확인 (저희 표본에서는 14~17이 아예
    # 없으므로 두 기준이 같아야 정상)
    mismatch = (df["pre_campaign_count_c"].clip(upper=6) != df["pre_campaign_count"].clip(upper=6)).sum()
    print(f"    상한 절단 전 최대값: {max_before_cap:.0f} → 상한 6 적용")
    print(f"    기존 pre_campaign_count(START_DAY 기준)과 값이 다른 가구 수: {mismatch}명 "
          f"(0이어야 정상 — 저희 표본은 14~17,19~22를 구조적으로 받지 않으므로 두 기준이 같음)\n")

    # -----------------------------------------------------------------
    # 6) has_demographic : 인구통계 보유 여부 0/1 (값 자체 대신 결측 지시자만 사용)
    # -----------------------------------------------------------------
    demo_households = set(hh_demo["household_key"])
    df["has_demographic"] = df["household_key"].isin(demo_households).astype(int)
    cov_by_group = df.groupby("group")["has_demographic"].mean()
    print(f"[6] has_demographic: 처치 {cov_by_group['처치']:.1%} vs 대조 {cov_by_group['대조']:.1%} 보유")
    p_t, p_c = cov_by_group["처치"], cov_by_group["대조"]
    smd_demo = (p_t - p_c) / np.sqrt((p_t * (1 - p_t) + p_c * (1 - p_c)) / 2)
    print(f"    SMD = {smd_demo:.3f} (결측 여부 자체가 강한 신호인지 확인)\n")

    # -----------------------------------------------------------------
    # 검토 후 제외: pre_sales_per_basket, pre_target_purchase(=pre_target_any) 재검증
    # -----------------------------------------------------------------
    df["pre_sales_per_basket"] = np.where(df["pre_baskets"] > 0, df["pre_sales"] / df["pre_baskets"], np.nan)
    aov_t = df.loc[df["group"] == "처치", "pre_sales_per_basket"].mean()
    aov_c = df.loc[df["group"] == "대조", "pre_sales_per_basket"].mean()
    aov_auc = univariate_auc(df["pre_sales_per_basket"].fillna(df["pre_sales_per_basket"].median()), (df["group"] == "처치").astype(int))
    print(f"[검토] pre_sales_per_basket(객단가): 처치 {aov_t:.2f} vs 대조 {aov_c:.2f}, 단변량 AUC={aov_auc:.3f} → 변별력 낮으면 제외")

    target_any_t = (df.loc[df["group"] == "처치", "pre_target_purchase_count"] > 0).mean()
    target_any_c = (df.loc[df["group"] == "대조", "pre_target_purchase_count"] > 0).mean()
    print(f"[검토] pre_target_any: 처치 {target_any_t:.1%} vs 대조 {target_any_c:.1%} → 변별력 없으면 제외\n")

    # -----------------------------------------------------------------
    # 최종 후보 7개: 단변량 AUC + 상관행렬 + VIF
    # -----------------------------------------------------------------
    final_vars = [
        "pre_recency_capped", "log_pre_baskets", "log_pre_sales",
        "pre_target_share", "pre_coupon_user", "pre_campaign_count_c", "has_demographic",
    ]
    y = (df["group"] == "처치").astype(int)

    print("[최종 후보 7개 단변량 AUC]")
    auc_rows = []
    for v in final_vars:
        auc = univariate_auc(df[v], y)
        auc_rows.append({"변수": v, "단변량 AUC": round(auc, 3)})
    auc_df = pd.DataFrame(auc_rows)
    print(auc_df.to_string(index=False))
    print()

    print("[최종 후보 7개 상관행렬 (Pearson)]")
    corr = df[final_vars].corr().round(2)
    print(corr.to_string())
    print()

    X = df[final_vars].to_numpy(dtype=float)
    X_std = StandardScaler().fit_transform(X)
    vif_rows = []
    for i, v in enumerate(final_vars):
        vif = variance_inflation_factor(X_std, i)
        vif_rows.append({"변수": v, "VIF": round(vif, 2)})
    vif_df = pd.DataFrame(vif_rows)
    print("[VIF]")
    print(vif_df.to_string(index=False))
    print()

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_std, y)
    p_hat = clf.predict_proba(X_std)[:, 1]
    full_auc = roc_auc_score(y, p_hat)
    print(f"[다변량 로지스틱 회귀] 표준화 7변수 모형 AUC = {full_auc:.3f}")
    print(f"  완전분리 여부(예측확률 0/1 근접치 존재): "
          f"{'주의 필요' if ((p_hat < 0.001) | (p_hat > 0.999)).any() else '없음'}\n")

    summary = auc_df.merge(vif_df, on="변수")
    print("[최종 요약]")
    print(summary.to_string(index=False))

    # -----------------------------------------------------------------
    # 결측치 처리 요약표 (변수별 1행)
    # -----------------------------------------------------------------
    n_zero_denom = int((df["pre_sales"] == 0).sum())
    missing_rows = [
        {"변수": "pre_recency_capped", "결측 처리": f"발행 전 구매 없으면 {START_DAY - 1}(최댓값)로 대체 후 365일 절단 (해당 {n_missing_recency}건)"},
        {"변수": "log_pre_baskets", "결측 처리": "결측 없음 — 활동 없으면 원값 0 → log1p(0)=0"},
        {"변수": "log_pre_sales", "결측 처리": "결측 없음 — 활동 없으면 원값 0 → log1p(0)=0"},
        {"변수": "pre_target_share", "결측 처리": f"분모(pre_sales)가 0이면 0으로 처리 (해당 {n_zero_denom}건)"},
        {"변수": "pre_coupon_user", "결측 처리": "결측 없음 — 사용 이력 없으면 0"},
        {"변수": "pre_campaign_count_c", "결측 처리": "결측 없음 — 과거 완료 캠페인 없으면 0, 상한 6 절단"},
        {"변수": "has_demographic", "결측 처리": "결측 개념 자체가 없음 — hh_demographic.csv 존재 여부가 곧 값(0/1)"},
        {"변수": "(미채택) 인구통계 값", "결측 처리": "hh_demographic.csv에 없는 가구는 결측 = 조사 안 됨. 0/평균 대체 금지, 값 자체는 이번 모형에 미포함"},
        {"변수": "결과변수(target_*, any_purchase 등)", "결측 처리": "캠페인 기간 거래 없으면 0으로 확정(관찰된 사실, 결측 아님) — 이미 analysis_data.csv에 반영됨"},
    ]
    missing_df = pd.DataFrame(missing_rows)
    print("\n[결측치 처리 요약]")
    print(missing_df.to_string(index=False))

    write_markdown(df, final_vars, summary, corr, full_auc, n_t, n_c, cov_by_group,
                    aov_t, aov_c, aov_auc, target_any_t, target_any_c,
                    completed_campaign_ids, started_not_completed, corr_share_sales, missing_df)
    print(f"\n저장 완료: {MD_OUT}")


def write_markdown(df, final_vars, summary, corr, full_auc, n_t, n_c, cov_by_group,
                    aov_t, aov_c, aov_auc, target_any_t, target_any_c,
                    completed_campaign_ids, started_not_completed, corr_share_sales, missing_df) -> None:
    lines = []
    lines.append("# 캠페인 18 성향점수 입력변수 확정 (최종)\n")
    lines.append(
        f"- 대상: 캠페인 18 (TypeA, DAY {START_DAY}~{END_DAY})\n"
        f"- 분석표: `outputs/campaign_18/analysis_data.csv` (처치 {n_t} / 대조 {n_c} = {n_t + n_c}가구)\n"
        "- 대조군은 `campaign_table.csv` 상 캠페인 수신 이력이 있으나 18번·겹치는 8개 캠페인은 "
        "받지 않은 335가구로 고정한다(캠페인을 한 번도 받지 않은 916가구는 제외 — 앞 단계에서 "
        "사용자가 직접 확정한 기준).\n"
        "- 참고: 사용자가 제공한 `ps_variables.md`는 대조군을 1,251명(위 916명 포함)으로 다르게 "
        "잡았기 때문에 이 문서와 모집단이 다르다. 아래는 그 파일의 변수 엔지니어링 방식을 "
        "저희 845가구 기준으로 재계산·재검증한 결과다.\n"
    )

    lines.append("## 1. 변수 엔지니어링\n")
    lines.append(
        f"1. **pre_recency_capped** = `{START_DAY} - 마지막 구매일`, 상한 365일 절단, 발행 전 구매이력이 "
        f"없는 가구는 {START_DAY - 1}(관찰 가능한 최댓값)로 대체. 현재 845가구 모두 발행 전 구매가 있어 "
        "결측 0건.\n"
        "2. **log_pre_baskets**, **log_pre_sales** = `log1p()` 변환. 오른쪽 꼬리가 긴 금액·빈도 변수의 "
        "왜도를 완화한다.\n"
        f"3. **pre_target_share** = `pre_target_sales / pre_sales` (분모 0이면 0). log_pre_sales와의 상관 "
        f"{corr_share_sales:.3f}로, 대상 상품 관련성을 전체 구매 규모와 어느 정도 독립적으로 표현한다.\n"
        "4. **pre_coupon_user** = `pre_coupon_redemptions > 0` (0/1). 원시 횟수는 0이 대다수인 극단 분포라 "
        "이진화가 더 안정적이다.\n"
        f"5. **pre_campaign_count_c** = 캠페인 시작일(587) 기준 **완전히 종료된**(`END_DAY < 587`) 과거 "
        f"캠페인 수신 횟수, 상한 6. 완전종료 캠페인: {sorted(completed_campaign_ids)}. "
        f"시작은 587 이전이지만 아직 안 끝난(겹치는) 캠페인 {sorted(started_not_completed)}은 제외해 "
        "사후 정보 유입을 원천 차단한다. 저희 표본은 이 겹치는 캠페인들을 애초에 받지 않으므로 "
        "`START_DAY<587` 기준으로 계산했던 이전 값과 결과는 동일하지만, 기준 자체는 이 방식이 더 "
        "엄격하고 일반적으로 옳다.\n"
        "6. **has_demographic** = `household_key`가 `hh_demographic.csv`에 존재하는지 (0/1). 인구통계 "
        f"값 자체(연령·소득 등)는 처치 {cov_by_group['처치']:.1%} vs 대조 {cov_by_group['대조']:.1%}로 "
        "확보율 차이가 커서 회귀 표본이 크게 줄어드는 문제가 있었다. 값 대신 **결측 여부만** 지시자로 "
        "넣어 이 정보를 표본 손실 없이 활용한다.\n"
    )

    lines.append("## 2. 검토 후 제외한 변수 (재검증)\n")
    lines.append(
        f"- **pre_sales_per_basket(객단가)**: 처치 {aov_t:.2f} vs 대조 {aov_c:.2f}, 단변량 AUC "
        f"{aov_auc:.3f} → 이미 균형에 가까워 변별력이 낮다. 제외.\n"
        f"- **pre_target_any(대상 상품 구매 경험 유무)**: 처치 {target_any_t:.1%} vs 대조 "
        f"{target_any_c:.1%} → 두 집단 모두 사실상 100%라 변별력이 없다. 제외.\n"
        "- **pre_quantity**: 이전 검토에서 확인한 특정 상품(무게 단위 판매 추정)의 이상치 문제로 제외 "
        "(변경 없음).\n"
        "- **pre_target_quantity, pre_target_sales**: `pre_target_share`로 대체되어 별도 입력 불필요.\n"
    )

    lines.append("## 3. 최종 확정 변수 (7개)\n")
    lines.append(df_to_markdown_table(summary))
    lines.append(f"\n**다변량 로지스틱 회귀(표준화, 7변수) AUC = {full_auc:.3f}**\n")

    lines.append("\n## 4. 변수 간 상관행렬 (Pearson)\n")
    lines.append("```\n" + corr.to_string() + "\n```\n")

    lines.append("## 5. 모형 사양\n")
    lines.append("```\n")
    lines.append("treatment ~ pre_recency_capped + log_pre_baskets + log_pre_sales\n")
    lines.append("          + pre_target_share + pre_coupon_user + pre_campaign_count_c\n")
    lines.append("          + has_demographic\n")
    lines.append("```\n")

    lines.append("\n## 6. 성향점수 입력에서 제외해야 하는 변수 (사후 정보)\n")
    lines.append(
        "캠페인 시작일(DAY 587) 이후에 결정되는 값은 어떤 형태로도 입력에 넣지 않는다.\n\n"
        "| 구분 | 변수 | 제외 사유 |\n|---|---|---|\n"
        "| 주요 결과 | `target_purchase`, `target_sales`, `target_quantity` | 추정 대상 그 자체 |\n"
        "| 보조 결과 | `any_purchase`, `total_sales`, `baskets` | 추정 대상 그 자체 |\n"
        "| 캠페인 기간 쿠폰 사용 | `coupon_redempt.csv`의 `587<=DAY<=642` 행, CAMPAIGN=18 | 처치의 결과이지 원인이 아님(post-treatment) |\n"
        "| 캠페인 이후 정보 | `DAY>642`의 모든 거래·쿠폰·캠페인 수신 | 처치 이후 발생, 시간상 원인이 될 수 없음 |\n"
        "| 겹치는 캠페인 수신 | 14,15,16,17,19,20,21,22 수신 이력 | 처치/대조 집단 정의에서 이미 전원 제외됨 |\n"
        "| 집단 라벨 | `treatment`, `group` | 모형의 종속변수 자체 |\n"
    )

    lines.append("\n## 7. 변수별 결측치 처리 요약 (한눈에 보기)\n")
    lines.append(df_to_markdown_table(missing_df))
    lines.append("")

    lines.append("\n## 8. 확정 사항 요약\n")
    lines.append(
        "1. 입력변수는 발행 전 구매행동 6개(변환 포함) + 인구통계 결측 지시자 1개 = **7개**로 확정한다.\n"
        "2. 대조군은 335가구(캠페인 수신 이력은 있으나 18번·겹치는 캠페인은 안 받은 가구)로 고정하며, "
        "캠페인을 한 번도 안 받은 916가구는 포함하지 않는다.\n"
        "3. 금액·빈도는 로그 변환, 최근성은 상한 365일 절단, 과거 캠페인 수는 '완전 종료' 기준으로 "
        "다시 계산하고 상한 6으로 절단한다.\n"
        "4. 대상 상품 관련성은 절대 금액이 아니라 지출 비중(`pre_target_share`)으로 넣는다.\n"
        "5. 인구통계 값 자체는 넣지 않고 결측 여부(`has_demographic`)만 넣는다 — 확보율 자체가 "
        "처치/대조 간 크게 달라(SMD 큼) 정보가 있지만, 값 자체를 넣으면 표본이 크게 줄어들기 때문이다.\n"
        "6. 결과변수, 캠페인 기간 쿠폰 사용, 캠페인 이후 정보, 겹치는 캠페인 수신 이력은 어떤 형태로도 "
        "넣지 않는다.\n"
        "7. 이 변수 목록은 결과변수를 보지 않고 확정했으며, 효과 추정 후 변경하지 않는다.\n"
    )

    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def df_to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body_lines = []
    for _, row in df.iterrows():
        cells = [str(row[c]).replace("\n", " ").replace("|", "\\|") for c in cols]
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + body_lines)


if __name__ == "__main__":
    main()
