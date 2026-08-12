"""18번 캠페인 처치/대조 가구의 발행 전(pre-period) 구매 특성을 만든다.

사용 원본 파일과 열:
- campaign_desc.csv   : CAMPAIGN, START_DAY, END_DAY        (캠페인 18 시작일, 과거 캠페인 판정)
- campaign_table.csv  : household_key, CAMPAIGN             (처치/대조 가구 재구성, 과거 캠페인 수신 횟수)
- coupon.csv          : CAMPAIGN, PRODUCT_ID                (18번 대상 상품 목록)
- coupon_redempt.csv  : household_key, DAY, CAMPAIGN         (과거 쿠폰 사용 횟수)
- transaction_data.csv: household_key, BASKET_ID, DAY, PRODUCT_ID, QUANTITY, SALES_VALUE
                         (recency, 장바구니 수, 구매금액, 구매수량, 대상 상품 구매 이력)

CLAUDE.md 분석 설계 규칙: 발행 전 특성에는 캠페인 시작일(START_DAY) 이전 정보만 사용한다.
모든 pre_* 변수는 DAY <= (18번 START_DAY - 1) 구간만 사용하고, assert로 이 경계를 넘는 행이
섞이지 않았는지 코드에서 직접 검증한다. 분석표는 저장하지 않고 터미널에만 출력한다
(CLAUDE.md 파일 관리 규칙).
"""

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
TARGET_CAMPAIGN = 18


def build_groups(desc: pd.DataFrame, table: pd.DataFrame):
    """18번 캠페인 처치/대조 가구 집합을 재구성한다 (campaign18_profile.py와 동일 로직)."""
    row18 = desc.loc[desc["CAMPAIGN"] == TARGET_CAMPAIGN].iloc[0]
    start_day = int(row18["START_DAY"])
    end_day = int(row18["END_DAY"])

    overlap_mask = (
        (desc["CAMPAIGN"] != TARGET_CAMPAIGN) & (desc["START_DAY"] <= end_day) & (desc["END_DAY"] >= start_day)
    )
    overlap_ids = desc.loc[overlap_mask, "CAMPAIGN"].tolist()

    recipients18 = set(table.loc[table["CAMPAIGN"] == TARGET_CAMPAIGN, "household_key"].unique())
    overlap_recipients = set(table.loc[table["CAMPAIGN"].isin(overlap_ids), "household_key"].unique())
    campaign_universe = set(table["household_key"].unique())

    treatment = recipients18 - overlap_recipients
    control = campaign_universe - recipients18 - overlap_recipients

    return start_day, end_day, overlap_ids, treatment, control


def main() -> None:
    desc = pd.read_csv(RAW / "campaign_desc.csv")
    table = pd.read_csv(RAW / "campaign_table.csv")

    start_day, end_day, overlap_ids, treatment, control = build_groups(desc, table)
    pre_end_day = start_day - 1  # 발행 전 관찰 종료일: 캠페인 시작일 하루 전

    households = sorted(treatment | control)
    group_map = {hh: "처치" for hh in treatment}
    group_map.update({hh: "대조" for hh in control})

    print(f"{'='*70}\n캠페인 18번 처치/대조 가구 발행 전 구매 특성\n{'='*70}")
    print(f"분석 대상 가구: 처치 {len(treatment)}명 + 대조 {len(control)}명 = 총 {len(households)}명")
    print(f"캠페인 18 START_DAY={start_day} → 발행 전 관찰 종료일 기준: DAY <= {pre_end_day}\n")

    # -----------------------------------------------------------------
    # 18번 캠페인 대상 상품 (coupon.csv)
    # -----------------------------------------------------------------
    coupon = pd.read_csv(RAW / "coupon.csv")
    target_products = set(coupon.loc[coupon["CAMPAIGN"] == TARGET_CAMPAIGN, "PRODUCT_ID"].unique())

    # -----------------------------------------------------------------
    # transaction_data.csv: 대상 가구 + DAY <= pre_end_day 만 사용
    # -----------------------------------------------------------------
    txn = pd.read_csv(RAW / "transaction_data.csv")
    txn_full_min, txn_full_max = int(txn["DAY"].min()), int(txn["DAY"].max())
    txn_pre = txn.loc[txn["household_key"].isin(households) & (txn["DAY"] <= pre_end_day)].copy()
    assert txn_pre["DAY"].max() <= pre_end_day, "transaction_data 발행 전 구간에 캠페인 시작일 이후 행이 섞여 있음"

    last_purchase_day = txn_pre.groupby("household_key")["DAY"].max()
    pre_baskets = txn_pre.groupby("household_key")["BASKET_ID"].nunique()
    pre_sales = txn_pre.groupby("household_key")["SALES_VALUE"].sum()
    pre_quantity = txn_pre.groupby("household_key")["QUANTITY"].sum()

    txn_pre_target = txn_pre.loc[txn_pre["PRODUCT_ID"].isin(target_products)]
    pre_target_purchase_count = txn_pre_target.groupby("household_key").size()
    pre_target_quantity = txn_pre_target.groupby("household_key")["QUANTITY"].sum()
    pre_target_sales = txn_pre_target.groupby("household_key")["SALES_VALUE"].sum()

    # -----------------------------------------------------------------
    # coupon_redempt.csv: 대상 가구 + DAY <= pre_end_day 만 사용
    # -----------------------------------------------------------------
    redempt = pd.read_csv(RAW / "coupon_redempt.csv")
    redempt_full_min, redempt_full_max = int(redempt["DAY"].min()), int(redempt["DAY"].max())
    redempt_pre = redempt.loc[redempt["household_key"].isin(households) & (redempt["DAY"] <= pre_end_day)]
    if not redempt_pre.empty:
        assert redempt_pre["DAY"].max() <= pre_end_day, "coupon_redempt 발행 전 구간에 캠페인 시작일 이후 행이 섞여 있음"
    pre_coupon_redemptions = redempt_pre.groupby("household_key").size()

    # -----------------------------------------------------------------
    # campaign_table.csv + campaign_desc.csv: START_DAY < 18번 START_DAY인 "과거" 캠페인만 카운트
    # -----------------------------------------------------------------
    past_campaign_ids = set(
        desc.loc[(desc["CAMPAIGN"] != TARGET_CAMPAIGN) & (desc["START_DAY"] < start_day), "CAMPAIGN"]
    )
    assert TARGET_CAMPAIGN not in past_campaign_ids, "18번 캠페인 자신이 과거 캠페인 집합에 포함됨"
    assert desc.loc[desc["CAMPAIGN"].isin(past_campaign_ids), "START_DAY"].max() < start_day, (
        "과거 캠페인 집합에 캠페인 18 시작일 이후 START_DAY가 섞여 있음"
    )
    table_pre = table.loc[table["household_key"].isin(households) & table["CAMPAIGN"].isin(past_campaign_ids)]
    # 처치/대조 가구는 애초에 겹치는 8개 캠페인을 받지 않으므로, 그중 과거에 시작한 14,15,16,17이
    # 여기 섞여 있으면 안 된다 (가구 구성 로직 자체의 정합성 재확인).
    assert not (set(table_pre["CAMPAIGN"].unique()) & set(overlap_ids)), "과거 캠페인 집계에 겹치는 캠페인이 포함됨"
    pre_campaign_count = table_pre.groupby("household_key")["CAMPAIGN"].nunique()

    # -----------------------------------------------------------------
    # 관찰 구간 표 출력 (캠페인 시작일 이후 정보 미포함 확인)
    # -----------------------------------------------------------------
    window_rows = [
        {
            "변수": "recency_days (최근 구매 이후 기간)",
            "원본 파일.열": "transaction_data.csv: DAY",
            "관찰 시작일": txn_full_min,
            "관찰 종료일": pre_end_day,
            "실사용 최대 DAY": int(txn_pre["DAY"].max()) if not txn_pre.empty else None,
        },
        {
            "변수": "pre_baskets (장바구니 수)",
            "원본 파일.열": "transaction_data.csv: BASKET_ID, DAY",
            "관찰 시작일": txn_full_min,
            "관찰 종료일": pre_end_day,
            "실사용 최대 DAY": int(txn_pre["DAY"].max()) if not txn_pre.empty else None,
        },
        {
            "변수": "pre_sales (구매금액)",
            "원본 파일.열": "transaction_data.csv: SALES_VALUE, DAY",
            "관찰 시작일": txn_full_min,
            "관찰 종료일": pre_end_day,
            "실사용 최대 DAY": int(txn_pre["DAY"].max()) if not txn_pre.empty else None,
        },
        {
            "변수": "pre_quantity (구매수량)",
            "원본 파일.열": "transaction_data.csv: QUANTITY, DAY",
            "관찰 시작일": txn_full_min,
            "관찰 종료일": pre_end_day,
            "실사용 최대 DAY": int(txn_pre["DAY"].max()) if not txn_pre.empty else None,
        },
        {
            "변수": "pre_target_* (대상 상품 구매 이력)",
            "원본 파일.열": "transaction_data.csv(DAY,PRODUCT_ID) x coupon.csv(PRODUCT_ID, CAMPAIGN=18)",
            "관찰 시작일": txn_full_min,
            "관찰 종료일": pre_end_day,
            "실사용 최대 DAY": int(txn_pre_target["DAY"].max()) if not txn_pre_target.empty else None,
        },
        {
            "변수": "pre_coupon_redemptions (과거 쿠폰 사용 횟수)",
            "원본 파일.열": "coupon_redempt.csv: DAY",
            "관찰 시작일": redempt_full_min,
            "관찰 종료일": pre_end_day,
            "실사용 최대 DAY": int(redempt_pre["DAY"].max()) if not redempt_pre.empty else None,
        },
        {
            "변수": "pre_campaign_count (과거 캠페인 수신 횟수)",
            "원본 파일.열": "campaign_table.csv(CAMPAIGN) x campaign_desc.csv(START_DAY)",
            "관찰 시작일": "-",
            "관찰 종료일": f"CAMPAIGN.START_DAY < {start_day}",
            "실사용 최대 DAY": int(desc.loc[desc["CAMPAIGN"].isin(past_campaign_ids), "START_DAY"].max()) if past_campaign_ids else None,
        },
    ]
    window_df = pd.DataFrame(window_rows)

    print("[변수별 관찰 시작일·종료일]")
    print(window_df.to_string(index=False))
    print()
    print("[검증] 캠페인 시작일 이후 정보 혼입 여부")
    print(f"  transaction_data 발행 전 구간 최대 DAY = {int(txn_pre['DAY'].max())} <= {pre_end_day} → 통과")
    if not redempt_pre.empty:
        print(f"  coupon_redempt 발행 전 구간 최대 DAY = {int(redempt_pre['DAY'].max())} <= {pre_end_day} → 통과")
    else:
        print("  coupon_redempt 발행 전 구간: 대상 가구의 사용 이력 없음")
    print(
        f"  과거 캠페인 집합의 최대 START_DAY = "
        f"{int(desc.loc[desc['CAMPAIGN'].isin(past_campaign_ids), 'START_DAY'].max())} < {start_day} → 통과"
    )
    print("  과거 캠페인 집계에 겹치는 8개 캠페인(14,15,16,17,19,20,21,22) 미포함 → 통과\n")

    # -----------------------------------------------------------------
    # 가구 x 변수 테이블 조합
    # -----------------------------------------------------------------
    df = pd.DataFrame({"household_key": households}).set_index("household_key")
    df["group"] = pd.Series(group_map)
    df["recency_days"] = start_day - last_purchase_day
    df["pre_baskets"] = pre_baskets
    df["pre_sales"] = pre_sales
    df["pre_quantity"] = pre_quantity
    df["pre_target_purchase_count"] = pre_target_purchase_count
    df["pre_target_quantity"] = pre_target_quantity
    df["pre_target_sales"] = pre_target_sales
    df["pre_coupon_redemptions"] = pre_coupon_redemptions
    df["pre_campaign_count"] = pre_campaign_count

    count_cols = [
        "pre_baskets", "pre_sales", "pre_quantity",
        "pre_target_purchase_count", "pre_target_quantity", "pre_target_sales",
        "pre_coupon_redemptions", "pre_campaign_count",
    ]
    df[count_cols] = df[count_cols].fillna(0)
    df["pre_target_any"] = (df["pre_target_purchase_count"] > 0).astype(int)
    df = df.reset_index()

    n_no_pre_purchase = int((df["pre_baskets"] == 0).sum())

    print("[가구 x 변수 테이블 미리보기 (상위 10행)]")
    print(df.head(10).to_string(index=False))
    print()

    print("[그룹별 요약 통계 (mean / median)]")
    summary_cols = [
        "recency_days", "pre_baskets", "pre_sales", "pre_quantity",
        "pre_target_any", "pre_target_purchase_count", "pre_target_sales",
        "pre_coupon_redemptions", "pre_campaign_count",
    ]
    summary = df.groupby("group")[summary_cols].agg(["mean", "median"]).round(2)
    print(summary.to_string())
    print("  (pre_quantity는 아래 이상치 경고 참고 — 중앙값도 이상치 영향을 받을 만큼 해당 상품 노출 가구 비중이 큼)")
    print()

    print(f"발행 전 구매 이력이 전혀 없는 가구 수: {n_no_pre_purchase}명 / {len(df)}명 (recency_days 결측)")
    if n_no_pre_purchase:
        print("  → recency_days는 NaN으로 남겨두었으며, 이후 매칭 단계에서 별도 처리 필요")
    print()

    # -----------------------------------------------------------------
    # 데이터 품질 경고: QUANTITY 이상치 (DATA_DICTIONARY.md에 명시된 점검 항목)
    #   특정 PRODUCT_ID(무게 단위 판매 등으로 추정)에서 QUANTITY가 수만 단위로 찍혀
    #   pre_quantity 평균이 소수 가구에 의해 크게 좌우된다. 원본 값을 수정하지 않고
    #   사실만 보고한다.
    # -----------------------------------------------------------------
    q99 = txn_pre["QUANTITY"].quantile(0.99)
    extreme_rows = txn_pre.loc[txn_pre["QUANTITY"] > q99 * 20]
    if not extreme_rows.empty:
        n_hh_affected = extreme_rows["household_key"].nunique()
        pct_hh_affected = 100 * n_hh_affected / len(households)
        top_product = extreme_rows["PRODUCT_ID"].value_counts().idxmax()
        print("[데이터 품질 경고] pre_quantity 이상치")
        print(
            f"  발행 전 거래 중 QUANTITY가 99분위({q99:.0f})의 20배를 초과하는 행 {len(extreme_rows)}건, "
            f"영향 가구 {n_hh_affected}명/{len(households)}명 ({pct_hh_affected:.0f}%)"
        )
        print(f"  가장 많이 나타나는 PRODUCT_ID: {int(top_product)} (무게/벌크 단위 판매 상품으로 추정, coupon.csv 대상 상품 아님 여부는 별도 확인 필요)")
        print(
            "  → 원본 값은 수정하지 않았음. 영향 가구 비중이 커서 평균은 물론 중앙값도 왜곡될 수 있으므로, "
            "이후 단계에서 이 상품(들)을 제외하거나 별도 처리할지 결정이 필요함(현재는 결정하지 않음)"
        )


if __name__ == "__main__":
    main()
