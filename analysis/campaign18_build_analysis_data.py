"""18번 캠페인 처치/대조 가구의 발행 전 특성 + 캠페인 기간 결과변수를 합쳐
가구 단위 분석표(analysis_data.csv)를 만든다.

사용 원본 파일과 열:
- campaign_desc.csv   : CAMPAIGN, START_DAY, END_DAY        (캠페인 18 기간, 과거/겹치는 캠페인 판정)
- campaign_table.csv  : household_key, CAMPAIGN             (처치/대조 가구, 과거 캠페인 수신 횟수)
- coupon.csv          : CAMPAIGN, PRODUCT_ID                (18번 대상 상품 목록)
- coupon_redempt.csv  : household_key, DAY, CAMPAIGN         (과거 쿠폰 사용 횟수)
- transaction_data.csv: household_key, BASKET_ID, DAY, PRODUCT_ID, QUANTITY, SALES_VALUE
                         (발행 전 특성 + 캠페인 기간 결과변수)

발행 전 특성(pre_*)은 DAY <= START_DAY-1, 결과변수는 START_DAY <= DAY <= END_DAY 구간만 사용한다
(CLAUDE.md: 발행 전 특성에는 캠페인 시작일 이전 정보만 사용, 주요 결과는 대상 상품, 보조 결과는
전체 상품 — DATA_DICTIONARY.md 정의와 동일).

거래가 없는 가구도 분석표에 남기고 결과값은 0으로 채운다. 최종 결과는
outputs/campaign_18/analysis_data.csv 하나로 저장한다(CLAUDE.md 파일 관리 규칙).
"""

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "campaign_18"
TARGET_CAMPAIGN = 18


def build_groups(desc: pd.DataFrame, table: pd.DataFrame):
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
    pre_end_day = start_day - 1

    households = sorted(treatment | control)
    group_map = {hh: "처치" for hh in treatment}
    group_map.update({hh: "대조" for hh in control})

    print(f"{'='*70}\n캠페인 18번 가구 분석표(analysis_data) 생성\n{'='*70}")
    print(f"분석 대상 가구: 처치 {len(treatment)}명 + 대조 {len(control)}명 = 총 {len(households)}명")
    print(f"발행 전 구간: DAY 1~{pre_end_day} | 캠페인 기간(결과변수 구간): DAY {start_day}~{end_day}\n")

    coupon = pd.read_csv(RAW / "coupon.csv")
    target_products = set(coupon.loc[coupon["CAMPAIGN"] == TARGET_CAMPAIGN, "PRODUCT_ID"].unique())

    txn = pd.read_csv(RAW / "transaction_data.csv")
    txn_hh = txn.loc[txn["household_key"].isin(households)].copy()

    # -----------------------------------------------------------------
    # 발행 전 특성 (pre_*) : DAY <= pre_end_day
    # -----------------------------------------------------------------
    txn_pre = txn_hh.loc[txn_hh["DAY"] <= pre_end_day]
    assert txn_pre["DAY"].max() <= pre_end_day, "발행 전 구간에 캠페인 시작일 이후 행이 섞여 있음"

    last_purchase_day = txn_pre.groupby("household_key")["DAY"].max()
    pre_baskets = txn_pre.groupby("household_key")["BASKET_ID"].nunique()
    pre_sales = txn_pre.groupby("household_key")["SALES_VALUE"].sum()
    pre_quantity = txn_pre.groupby("household_key")["QUANTITY"].sum()

    txn_pre_target = txn_pre.loc[txn_pre["PRODUCT_ID"].isin(target_products)]
    pre_target_purchase_count = txn_pre_target.groupby("household_key").size()
    pre_target_quantity = txn_pre_target.groupby("household_key")["QUANTITY"].sum()
    pre_target_sales = txn_pre_target.groupby("household_key")["SALES_VALUE"].sum()

    redempt = pd.read_csv(RAW / "coupon_redempt.csv")
    redempt_pre = redempt.loc[redempt["household_key"].isin(households) & (redempt["DAY"] <= pre_end_day)]
    if not redempt_pre.empty:
        assert redempt_pre["DAY"].max() <= pre_end_day, "coupon_redempt 발행 전 구간에 캠페인 시작일 이후 행이 섞여 있음"
    pre_coupon_redemptions = redempt_pre.groupby("household_key").size()

    past_campaign_ids = set(
        desc.loc[(desc["CAMPAIGN"] != TARGET_CAMPAIGN) & (desc["START_DAY"] < start_day), "CAMPAIGN"]
    )
    table_pre = table.loc[table["household_key"].isin(households) & table["CAMPAIGN"].isin(past_campaign_ids)]
    assert not (set(table_pre["CAMPAIGN"].unique()) & set(overlap_ids)), "과거 캠페인 집계에 겹치는 캠페인이 포함됨"
    pre_campaign_count = table_pre.groupby("household_key")["CAMPAIGN"].nunique()

    # -----------------------------------------------------------------
    # 캠페인 기간 결과변수 : START_DAY <= DAY <= END_DAY
    #   주요 결과 = 대상 상품(target_*) / 보조 결과 = 전체 상품(any_purchase, total_sales, baskets)
    # -----------------------------------------------------------------
    txn_camp = txn_hh.loc[(txn_hh["DAY"] >= start_day) & (txn_hh["DAY"] <= end_day)]
    if not txn_camp.empty:
        assert txn_camp["DAY"].min() >= start_day and txn_camp["DAY"].max() <= end_day, (
            "캠페인 기간 구간에 START_DAY~END_DAY 밖의 행이 섞여 있음"
        )

    # 보조 결과: 전체 상품
    total_sales = txn_camp.groupby("household_key")["SALES_VALUE"].sum()
    baskets = txn_camp.groupby("household_key")["BASKET_ID"].nunique()

    # 주요 결과: 캠페인 대상 상품
    txn_camp_target = txn_camp.loc[txn_camp["PRODUCT_ID"].isin(target_products)]
    target_sales = txn_camp_target.groupby("household_key")["SALES_VALUE"].sum()
    target_quantity = txn_camp_target.groupby("household_key")["QUANTITY"].sum()
    target_purchase_count = txn_camp_target.groupby("household_key").size()

    print("[결과변수 관찰 구간 확인]")
    print(f"  주요 결과(target_purchase, target_sales, target_quantity): transaction_data.csv DAY {start_day}~{end_day}")
    print(f"  보조 결과(any_purchase, total_sales, baskets): transaction_data.csv DAY {start_day}~{end_day}")
    if not txn_camp.empty:
        print(f"  실사용 DAY 범위: {int(txn_camp['DAY'].min())}~{int(txn_camp['DAY'].max())} → 캠페인 기간 내 확인됨\n")
    else:
        print("  캠페인 기간 거래 없음\n")

    # -----------------------------------------------------------------
    # 최종 분석표 조합
    # -----------------------------------------------------------------
    df = pd.DataFrame({"household_key": households}).set_index("household_key")
    df["group"] = pd.Series(group_map)
    df["treatment"] = (df["group"] == "처치").astype(int)

    # 발행 전 특성
    df["recency_days"] = start_day - last_purchase_day
    df["pre_baskets"] = pre_baskets
    df["pre_sales"] = pre_sales
    df["pre_quantity"] = pre_quantity
    df["pre_target_purchase_count"] = pre_target_purchase_count
    df["pre_target_quantity"] = pre_target_quantity
    df["pre_target_sales"] = pre_target_sales
    df["pre_coupon_redemptions"] = pre_coupon_redemptions
    df["pre_campaign_count"] = pre_campaign_count

    pre_count_cols = [
        "pre_baskets", "pre_sales", "pre_quantity",
        "pre_target_purchase_count", "pre_target_quantity", "pre_target_sales",
        "pre_coupon_redemptions", "pre_campaign_count",
    ]
    df[pre_count_cols] = df[pre_count_cols].fillna(0)
    df["pre_target_any"] = (df["pre_target_purchase_count"] > 0).astype(int)

    # 주요 결과 (캠페인 대상 상품)
    df["target_sales"] = target_sales
    df["target_quantity"] = target_quantity
    df["target_purchase_count"] = target_purchase_count

    # 보조 결과 (전체 상품)
    df["total_sales"] = total_sales
    df["baskets"] = baskets

    result_cols = ["target_sales", "target_quantity", "target_purchase_count", "total_sales", "baskets"]
    df[result_cols] = df[result_cols].fillna(0)
    df["target_purchase"] = (df["target_purchase_count"] > 0).astype(int)
    df["any_purchase"] = (df["baskets"] > 0).astype(int)

    df = df.reset_index()

    # 열 순서 정리: 식별/처치 -> 발행 전 특성 -> 주요 결과 -> 보조 결과
    ordered_cols = [
        "household_key", "group", "treatment",
        "recency_days", "pre_baskets", "pre_sales", "pre_quantity",
        "pre_target_any", "pre_target_purchase_count", "pre_target_quantity", "pre_target_sales",
        "pre_coupon_redemptions", "pre_campaign_count",
        # 주요 결과 (대상 상품)
        "target_purchase", "target_sales", "target_quantity",
        # 보조 결과 (전체 상품)
        "any_purchase", "total_sales", "baskets",
    ]
    df = df[ordered_cols]

    # -----------------------------------------------------------------
    # 검증
    # -----------------------------------------------------------------
    n_no_campaign_txn = int((df["any_purchase"] == 0).sum())
    result_value_cols = ["target_purchase", "target_sales", "target_quantity", "any_purchase", "total_sales", "baskets"]
    assert df[result_value_cols].isna().sum().sum() == 0, "결과변수에 결측이 남아 있음 (0 처리 누락)"
    assert len(df) == len(households), "분석표 가구 수가 처치+대조 가구 수와 다름"
    assert df["household_key"].is_unique, "household_key 중복 존재"

    print("[검증]")
    print(f"  분석표 행 수: {len(df)}명 (처치 {int(df['treatment'].sum())} + 대조 {len(df) - int(df['treatment'].sum())})")
    print(f"  결과변수 6개 열({', '.join(result_value_cols)}) 결측 0건 확인됨")
    print(f"  캠페인 기간 거래가 없어 결과값이 전부 0으로 채워진 가구: {n_no_campaign_txn}명 (분석표에서 제외하지 않음)")
    print()

    print("[가구 x 변수 분석표 미리보기 (상위 10행)]")
    print(df.head(10).to_string(index=False))
    print()

    print("[그룹별 결과변수 요약 (mean)]")
    print(df.groupby("group")[["target_purchase", "target_sales", "target_quantity", "any_purchase", "total_sales", "baskets"]].mean().round(3).to_string())
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "analysis_data.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {out_path}  ({len(df)}행 x {len(df.columns)}열)")


if __name__ == "__main__":
    main()
