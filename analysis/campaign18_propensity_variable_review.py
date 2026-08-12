"""캠페인 18 성향점수 모형에 사용할 변수를 검토한다.

- 후보 1: 발행 전 구매행동 변수 (outputs/campaign_18/analysis_data.csv의 pre_* 열)
- 후보 2: 인구통계 변수 (hh_demographic.csv) — 845가구(처치+대조) 중 실제 확보된 비율을 확인한다.
- 다중공선성 확인: 구매행동 변수 간 상관관계를 계산해 중복 변수를 가려낸다.
- 제외 대상: 결과변수(analysis_data.csv의 target_*/any_purchase/total_sales/baskets),
  캠페인 기간 쿠폰 사용(coupon_redempt.csv, CAMPAIGN=18, 캠페인 기간 내 DAY),
  캠페인 이후 정보(END_DAY 이후 모든 원본 데이터) — 이 세 가지는 값 자체를 계산하지 않고
  "왜 제외해야 하는지"만 표에 남긴다.

이 스크립트는 변수 선택 근거(결측치 의미 포함)를 표로 정리해 출력하고,
analysis/campaign18_propensity_variables.md 로 저장한다.
"""

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_PATH = Path(__file__).resolve().parent.parent / "outputs" / "campaign_18" / "analysis_data.csv"
MD_OUT = Path(__file__).resolve().parent / "campaign18_propensity_variables.md"

START_DAY, END_DAY = 587, 642  # campaign_desc.csv CAMPAIGN=18 기준 (이미 검증됨)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    hh_demo = pd.read_csv(RAW / "hh_demographic.csv")

    households = set(df["household_key"])
    demo_covered = households & set(hh_demo["household_key"])
    coverage_pct = 100 * len(demo_covered) / len(households)

    print(f"{'='*72}\n캠페인 18 성향점수 모형 변수 검토\n{'='*72}")
    print(f"분석 대상 가구: {len(households)}명 (처치+대조)")
    print(f"이 중 hh_demographic.csv에 인구통계가 있는 가구: {len(demo_covered)}명 ({coverage_pct:.1f}%)")
    print(f"인구통계 없는 가구: {len(households) - len(demo_covered)}명 ({100 - coverage_pct:.1f}%)\n")

    # -----------------------------------------------------------------
    # 처치/대조별 인구통계 확보율도 따로 확인 (표본 편향 여부 점검)
    # -----------------------------------------------------------------
    df_demo_flag = df[["household_key", "group"]].copy()
    df_demo_flag["has_demo"] = df_demo_flag["household_key"].isin(demo_covered)
    print("[그룹별 인구통계 확보율]")
    print(
        df_demo_flag.groupby("group")["has_demo"].agg(["sum", "count", "mean"]).rename(
            columns={"sum": "확보 가구", "count": "전체 가구", "mean": "확보율"}
        ).round(3).to_string()
    )
    print()

    # -----------------------------------------------------------------
    # 발행 전 구매행동 변수 간 상관관계 (다중공선성 확인)
    # -----------------------------------------------------------------
    behavior_cols = [
        "recency_days", "pre_baskets", "pre_sales", "pre_quantity",
        "pre_target_purchase_count", "pre_target_quantity", "pre_target_sales",
        "pre_coupon_redemptions", "pre_campaign_count",
    ]
    corr = df[behavior_cols].corr(method="spearman").round(2)
    print("[발행 전 구매행동 변수 간 상관관계 (Spearman)]")
    print(corr.to_string())
    print()

    # -----------------------------------------------------------------
    # 변수 검토표
    # -----------------------------------------------------------------
    rows = []

    def add(변수, 원본, 구분, 이유, 결측치의미):
        rows.append({"변수": 변수, "원본": 원본, "구분": 구분, "선택/제외 이유": 이유, "결측치의 의미": 결측치의미})

    # --- 포함 추천: 발행 전 구매행동 ---
    add(
        "recency_days", "transaction_data.csv (DAY)", "포함(사용)",
        "캠페인 시작일 기준 최근 구매 경과일. 처치-대조 SMD=-0.37로 두 집단을 가장 잘 가르는 변수 중 하나.",
        "발행 전(DAY<587) 구매가 전혀 없는 가구는 계산 불가 → NaN. '최근성이 0'이 아니라 "
        "'발행 전 구매 이력 자체 없음'을 의미하므로 0으로 채우면 안 됨(현재 845가구 모두 발행 전 구매 있어 결측 0건).",
    )
    add(
        "pre_baskets", "transaction_data.csv (BASKET_ID)", "포함(사용)",
        "구매 빈도 지표. SMD=0.74로 처치-대조 차이가 가장 큰 변수.",
        "발행 전 거래가 없으면 0 (활동 없음을 의미, 결측 아님).",
    )
    add(
        "pre_sales", "transaction_data.csv (SALES_VALUE)", "포함(사용)",
        "구매력 지표. SMD=0.65로 차이가 큼. pre_quantity보다 이상치에 덜 민감해 pre_baskets와 함께 핵심 변수로 채택.",
        "발행 전 거래가 없으면 0 (활동 없음을 의미, 결측 아님).",
    )
    add(
        "pre_quantity", "transaction_data.csv (QUANTITY)", "제외 추천(데이터 품질)",
        "이전 단계에서 확인한 대로 특정 상품(PRODUCT_ID 6534178, 무게 단위 판매 추정)의 QUANTITY가 "
        "수만 단위로 찍혀 845가구 중 502명(59%)에 영향. 중앙값까지 왜곡되어 구매력 지표로 신뢰 불가. "
        "같은 목적은 pre_sales(금액)로 이미 대체 가능.",
        "발행 전 거래가 없으면 0. (값 자체는 있으나 이상치로 신뢰 불가)",
    )
    add(
        "pre_target_purchase_count", "transaction_data.csv x coupon.csv(PRODUCT_ID)", "포함(사용)",
        "캠페인 18 대상 상품군에 대한 발행 전 관심도. DATA_DICTIONARY.md의 'pre_target_purchases'에 해당하는 "
        "핵심 사전 관련성 변수. SMD=0.66로 차이가 큼.",
        "발행 전 대상 상품 구매가 없으면 0 (활동 없음을 의미, 결측 아님).",
    )
    add(
        "pre_target_quantity", "transaction_data.csv x coupon.csv(PRODUCT_ID)", "제외 추천(다중공선성)",
        "pre_target_purchase_count·pre_target_sales와 상관계수가 매우 높아(중복 정보) 함께 넣으면 "
        "다중공선성만 키움. pre_target_purchase_count 하나로 대상 상품 관심도를 대표.",
        "발행 전 대상 상품 구매가 없으면 0.",
    )
    add(
        "pre_target_sales", "transaction_data.csv x coupon.csv(PRODUCT_ID)", "제외 추천(다중공선성)",
        "pre_target_purchase_count·pre_sales와 상관이 높아 중복. 대상 상품 특유의 신호는 "
        "pre_target_purchase_count로 충분히 반영됨.",
        "발행 전 대상 상품 구매가 없으면 0.",
    )
    add(
        "pre_target_any", "transaction_data.csv x coupon.csv(PRODUCT_ID)", "제외(변별력 없음)",
        "대상 상품군이 전체 상품의 38%(35,513/92,353)로 넓어 처치·대조 모두 100% 구매 경험 "
        "(SMD=0.000). 분산이 없어 모형에 정보를 주지 않음.",
        "발행 전 대상 상품 구매가 없으면 0.",
    )
    add(
        "pre_coupon_redemptions", "coupon_redempt.csv (DAY, CAMPAIGN)", "포함(사용)",
        "과거(발행 전) 쿠폰 반응성. 마케팅 반응 성향을 보여주는 직접적 지표이며 SMD=0.31.",
        "발행 전 쿠폰 사용 이력이 없으면 0 (활동 없음을 의미, 결측 아님).",
    )
    add(
        "pre_campaign_count", "campaign_table.csv x campaign_desc.csv(START_DAY<587)", "포함(사용)",
        "과거 캠페인 수신 횟수 = 마케팅 대상으로 자주 선정되는 가구인지 나타내는 지표. SMD=0.28.",
        "과거에 시작한 캠페인을 하나도 못 받았으면 0 (활동 없음을 의미, 결측 아님).",
    )

    # --- 인구통계 (검토 필요) ---
    demo_note = (
        f"hh_demographic.csv는 845가구 중 {len(demo_covered)}명({coverage_pct:.1f}%)만 커버. "
        "확보 안 된 가구는 무작위 결측이 아니라 '인구통계 조사 자체가 안 된 가구'로, 0이나 평균으로 "
        "대체하면 안 됨(정보 없음과 특정 값은 다른 의미)."
    )
    for col, label in [
        ("AGE_DESC", "연령대"), ("MARITAL_STATUS_CODE", "결혼상태"), ("INCOME_DESC", "소득구간"),
        ("HOMEOWNER_DESC", "주택소유"), ("HH_COMP_DESC", "가구구성"),
        ("HOUSEHOLD_SIZE_DESC", "가구원수"), ("KID_CATEGORY_DESC", "자녀구분"),
    ]:
        add(
            f"{col} ({label})", "hh_demographic.csv", "검토 필요(커버리지 제한)",
            f"인구통계 확보 가구에서만 사용 가능한 보조 변수. {demo_note} "
            "포함 시 분석 표본이 인구통계 확보 가구로 크게 줄어들거나, 결측 더미를 별도로 둬야 함.",
            "household_key가 hh_demographic.csv에 없으면 결측 = '인구통계 미확보'. "
            "MARITAL_STATUS_CODE='U'처럼 파일 내 'Unknown' 코드가 있는 경우는 다른 의미로, "
            "'응답은 했으나 알 수 없음'에 해당해 파일에 없는 결측과 구분해야 함.",
        )

    # --- 제외해야 할 변수: 결과변수 / 캠페인 기간 쿠폰 사용 / 캠페인 이후 정보 ---
    for col in ["target_purchase", "target_sales", "target_quantity"]:
        add(
            col, "outputs/campaign_18/analysis_data.csv (캠페인 기간 결과)", "제외(결과변수)",
            "캠페인 기간(DAY 587~642) 대상 상품 구매 결과. 성향점수는 '수신 가능성'을 예측하는 모형이며, "
            "결과를 입력에 넣으면 미래 정보 누출(target leakage)로 매칭 자체가 무의미해짐.",
            "해당 없음 (모형에서 완전히 제외).",
        )
    for col in ["any_purchase", "total_sales", "baskets"]:
        add(
            col, "outputs/campaign_18/analysis_data.csv (캠페인 기간 보조 결과)", "제외(결과변수)",
            "캠페인 기간 전체 상품 구매 결과. 위와 동일한 이유로 성향점수 입력에서 제외.",
            "해당 없음 (모형에서 완전히 제외).",
        )
    add(
        "캠페인 기간 쿠폰 사용", "coupon_redempt.csv (household_key, DAY, COUPON_UPC, CAMPAIGN=18, START_DAY<=DAY<=END_DAY)",
        "제외(캠페인 기간 정보)",
        "캠페인을 받은 가구만 그 쿠폰을 쓸 수 있으므로 처치를 받았다는 사실 자체를 그대로 드러내는 "
        "변수(post-treatment). 수신 가능성이 아니라 수신 결과이므로 입력 후보에서 원천 배제.",
        "해당 가구가 쿠폰을 안 썼으면 사용 이력 자체가 없음(활동 없음). 성향점수 모형에 넣지 않으므로 "
        "결측 처리 자체가 불필요.",
    )
    add(
        "캠페인 이후 정보 (DAY > 642)", "transaction_data.csv, coupon_redempt.csv 등 모든 원본의 END_DAY 이후 행",
        "제외(캠페인 이후 정보)",
        "캠페인 종료 이후에 발생한 모든 구매·쿠폰 정보는 처치 이후 시점이라 시간 순서상 "
        "'원인'이 될 수 없음. CLAUDE.md 규칙: 발행 전 특성에는 캠페인 시작일 이전 정보만 사용.",
        "해당 없음 (애초에 조회·집계하지 않음).",
    )
    add(
        "group / treatment", "outputs/campaign_18/analysis_data.csv", "제외(모형의 목표변수)",
        "성향점수 모형이 예측해야 할 대상(Y) 자체. 입력 변수(X)가 아니라 모형의 종속변수.",
        "해당 없음.",
    )

    review = pd.DataFrame(rows)

    print("[변수 검토표]")
    print(review.to_string(index=False))
    print()

    final_included = review.loc[review["구분"] == "포함(사용)", "변수"].tolist()
    print(f"최종 포함 추천 변수 {len(final_included)}개: {final_included}")

    demo_by_group = df_demo_flag.groupby("group")["has_demo"].agg(["sum", "count", "mean"])
    write_markdown(review, final_included, len(households), len(demo_covered), coverage_pct, corr, demo_by_group)
    print(f"\n저장 완료: {MD_OUT}")


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """tabulate 의존성 없이 DataFrame을 마크다운 표 문자열로 변환한다."""
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body_lines = []
    for _, row in df.iterrows():
        cells = [str(row[c]).replace("\n", " ").replace("|", "\\|") for c in cols]
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + body_lines)


def write_markdown(review: pd.DataFrame, final_included, n_total, n_demo, coverage_pct, corr, demo_by_group) -> None:
    lines = []
    lines.append("# 캠페인 18 성향점수 모형 변수 검토\n")
    lines.append(
        "분석 대상 가구(처치+대조) {}명을 기준으로, `outputs/campaign_18/analysis_data.csv`의 발행 전 "
        "구매행동 변수와 `data/raw/hh_demographic.csv`의 인구통계 변수를 검토했다. "
        "성향점수는 '캠페인 18을 받을 가능성'을 예측하는 모형이므로, 캠페인 시작일(DAY 587) 이전에 "
        "확정된 정보만 입력으로 쓸 수 있다(CLAUDE.md 분석 설계 규칙).\n".format(n_total)
    )

    lines.append("## 1. 인구통계 커버리지\n")
    lines.append(
        f"- 분석 대상 {n_total}가구 중 `hh_demographic.csv`에 인구통계가 있는 가구: **{n_demo}명 "
        f"({coverage_pct:.1f}%)**\n"
        f"- 인구통계가 없는 가구: {n_total - n_demo}명 ({100 - coverage_pct:.1f}%)\n"
    )
    t_row = demo_by_group.loc["처치"]
    c_row = demo_by_group.loc["대조"]
    lines.append(
        f"- **그룹별 확보율이 다르다**: 처치 {int(t_row['sum'])}/{int(t_row['count'])}명 "
        f"({t_row['mean']*100:.1f}%) vs 대조 {int(c_row['sum'])}/{int(c_row['count'])}명 "
        f"({c_row['mean']*100:.1f}%). 인구통계 확보 여부 자체가 처치군에서 두 배 이상 높아, "
        "이 결측이 완전 무작위(MCAR)가 아니라 처치 배정과도 연관되어 있음을 시사한다. "
        "결측 가구를 임의로 채우거나 빼고 비교하면 새로운 편향을 만들 수 있다.\n"
    )
    lines.append(
        "- 이 결측은 무작위가 아니라 '애초에 조사되지 않은 가구'다. 0이나 평균으로 대체하면 존재하지 "
        "않는 정보를 만들어내는 것이므로, 인구통계 변수는 이번 1차 모형에서는 **보류**하고 커버리지 "
        "문제를 해결한 뒤(결측 더미 추가, 또는 확보된 가구만의 민감도 분석 등) 별도로 검토한다.\n"
    )

    lines.append("## 2. 발행 전 구매행동 변수 간 상관관계 (Spearman)\n")
    lines.append("```\n" + corr.to_string() + "\n```\n")
    lines.append(
        "`pre_target_purchase_count`, `pre_target_quantity`, `pre_target_sales`는 서로 상관이 매우 높고, "
        "`pre_baskets`·`pre_sales`와도 상관이 높다. 세 대상 상품 변수를 모두 넣으면 다중공선성만 커지므로 "
        "`pre_target_purchase_count` 하나만 남긴다.\n"
    )

    lines.append("## 3. 변수 검토표\n")
    lines.append(df_to_markdown_table(review))
    lines.append("")

    lines.append("\n## 4. 최종 확정 변수\n")
    lines.append("성향점수 모형(처치 여부를 예측하는 로지스틱 회귀 등)의 입력(X)으로 아래 변수를 확정한다.\n")
    for v in final_included:
        lines.append(f"- `{v}`")
    lines.append("")
    lines.append(
        "\n인구통계 변수는 커버리지 문제(약 {:.0f}% 결측, 처치/대조 간 확보율도 상이)로 이번 1차 모형에서는 "
        "**미포함**하며, 필요 시 확보된 가구만 대상으로 한 별도 민감도 분석에서 다룬다.\n".format(100 - coverage_pct)
    )
    lines.append(
        "**남은 다중공선성 caveat**: `pre_sales`와 `pre_target_purchase_count`의 상관계수도 0.86으로 "
        "여전히 높다. 두 변수를 동시에 남긴 이유는 '전체 구매력'과 '캠페인 대상 카테고리 관심도'라는 "
        "개념이 달라서지만, 로지스틱 회귀 계수의 개별 해석은 불안정할 수 있다는 점을 감안해야 한다. "
        "성향점수(예측 확률) 자체의 품질에는 큰 문제가 없더라도, 계수 부호·크기를 개별적으로 해석하지 "
        "않는다.\n"
    )

    lines.append("## 5. 제외 변수 요약\n")
    lines.append("| 구분 | 변수 | 이유 |\n|---|---|---|\n")
    excluded = review.loc[review["구분"].str.startswith("제외")]
    for _, r in excluded.iterrows():
        lines.append(f"| {r['구분']} | `{r['변수']}` | {r['선택/제외 이유']} |\n")

    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
