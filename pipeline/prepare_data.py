"""캠페인 입력형 데이터 준비 파이프라인.

캠페인 18 분석(analysis/campaign18_*.py)에서 검증한 절차를 일반화한다. 입력은
`campaign_id`와 `pre_days`(발행 전 관찰 기간 길이, 일)뿐이며 나머지 처리 규칙은
아래 CONSTANTS로 고정한다 — 캠페인 18에서 검증된 값이므로 실행마다 바뀌지 않는다.

실행 순서: 캠페인 정보조회 → 집단 구성 → 사전(및 결과) 변수 계산 → 결측치 처리 → 인코딩

CLAUDE.md 규칙:
- 처치집단 = 선택 캠페인을 받고 같은 기간 겹치는 다른 캠페인은 받지 않은 가구.
- 대조집단 = 선택 캠페인을 받지 않고 같은 기간 겹치는 다른 캠페인도 받지 않은 가구.
  모집단은 campaign_table.csv에 등장하는(=캠페인을 1회 이상 받은) 가구로 한정한다.
  캠페인을 한 번도 받지 않은 가구는 대조군에 포함하지 않는다(캠페인 18에서 확정한 규칙).
- 발행 전 특성에는 캠페인 시작일 이전 정보만 사용한다.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"

# ===========================================================================
# 제약사항(CONSTANTS): 캠페인 18에서 검증한 처리 규칙. campaign_id, pre_days
# 외에는 실행 인자로 바꾸지 않는다 — 다른 값으로 실험하려면 이 상수를 명시적으로
# 수정하고 analysis/campaign18_*.py 결과와 다시 대조해야 한다.
# ===========================================================================
RECENCY_CAP_DAYS = 365          # pre_recency_capped 상한 (campaign18_propensity_variables_finalize.py)
CAMPAIGN_COUNT_CAP = 6           # pre_campaign_count_c 상한 (ps_variables.md 검증 결과 채택)
FINAL_PROPENSITY_VARS = [
    "pre_recency_capped", "log_pre_baskets", "log_pre_sales",
    "pre_target_share", "pre_coupon_user", "pre_campaign_count_c", "has_demographic",
]
OUTCOME_PRIMARY = ["target_purchase", "target_sales", "target_quantity"]
OUTCOME_SECONDARY = ["any_purchase", "total_sales", "baskets"]


@dataclass
class CampaignMeta:
    campaign_id: int
    description: str
    start_day: int
    end_day: int
    pre_days: int
    pre_start_day: int
    pre_end_day: int
    overlap_ids: list = field(default_factory=list)
    n_universe: int = 0
    n_treatment: int = 0
    n_control: int = 0


def _log(step: str, df: pd.DataFrame | None = None, extra: str = "") -> None:
    if df is not None:
        print(f"[prepare_data] {step}: shape = {df.shape[0]}행 x {df.shape[1]}열  {extra}")
    else:
        print(f"[prepare_data] {step}  {extra}")


# ---------------------------------------------------------------------------
# 1) 캠페인 정보조회
# ---------------------------------------------------------------------------
def get_campaign_info(campaign_id: int, pre_days: int, desc: pd.DataFrame) -> CampaignMeta:
    row = desc.loc[desc["CAMPAIGN"] == campaign_id]
    if row.empty:
        raise ValueError(f"campaign_desc.csv에 CAMPAIGN={campaign_id}가 없습니다.")
    row = row.iloc[0]
    start_day, end_day = int(row["START_DAY"]), int(row["END_DAY"])

    overlap_mask = (
        (desc["CAMPAIGN"] != campaign_id) & (desc["START_DAY"] <= end_day) & (desc["END_DAY"] >= start_day)
    )
    overlap_ids = sorted(desc.loc[overlap_mask, "CAMPAIGN"].tolist())

    pre_start_day = max(1, start_day - pre_days)
    pre_end_day = start_day - 1

    meta = CampaignMeta(
        campaign_id=campaign_id, description=row["DESCRIPTION"], start_day=start_day, end_day=end_day,
        pre_days=pre_days, pre_start_day=pre_start_day, pre_end_day=pre_end_day, overlap_ids=overlap_ids,
    )
    _log(
        "1단계 캠페인 정보조회", None,
        f"CAMPAIGN={campaign_id} ({meta.description}) DAY {start_day}~{end_day}, "
        f"발행전 관찰구간 DAY {pre_start_day}~{pre_end_day}, 겹치는 캠페인 {overlap_ids}",
    )
    return meta


# ---------------------------------------------------------------------------
# 2) 집단 구성 (처치/대조)
# ---------------------------------------------------------------------------
def build_treatment_control(meta: CampaignMeta, table: pd.DataFrame) -> tuple[set, set, pd.DataFrame]:
    recipients = set(table.loc[table["CAMPAIGN"] == meta.campaign_id, "household_key"].unique())
    overlap_recipients = set(
        table.loc[table["CAMPAIGN"].isin(meta.overlap_ids), "household_key"].unique()
    )
    campaign_universe = set(table["household_key"].unique())

    treatment = recipients - overlap_recipients
    control = campaign_universe - recipients - overlap_recipients

    meta.n_universe = len(campaign_universe)
    meta.n_treatment = len(treatment)
    meta.n_control = len(control)

    households = sorted(treatment | control)
    group_df = pd.DataFrame({"household_key": households})
    group_df["group"] = group_df["household_key"].apply(lambda h: "처치" if h in treatment else "대조")
    group_df["treatment"] = (group_df["group"] == "처치").astype(int)

    _log(
        "2단계 집단 구성", group_df,
        f"(모집단 {meta.n_universe}명 중 처치 {meta.n_treatment} + 대조 {meta.n_control})",
    )
    return treatment, control, group_df


# ---------------------------------------------------------------------------
# 3) 사전/결과 변수 계산 및 생성
# ---------------------------------------------------------------------------
def compute_pre_period_features(
    group_df: pd.DataFrame, meta: CampaignMeta, txn: pd.DataFrame, coupon: pd.DataFrame,
    redempt: pd.DataFrame, desc: pd.DataFrame, table: pd.DataFrame, hh_demo: pd.DataFrame,
) -> pd.DataFrame:
    households = set(group_df["household_key"])
    target_products = set(coupon.loc[coupon["CAMPAIGN"] == meta.campaign_id, "PRODUCT_ID"].unique())

    txn_hh = txn.loc[txn["household_key"].isin(households)]
    txn_pre = txn_hh.loc[(txn_hh["DAY"] >= meta.pre_start_day) & (txn_hh["DAY"] <= meta.pre_end_day)]
    assert txn_pre.empty or (
        txn_pre["DAY"].min() >= meta.pre_start_day and txn_pre["DAY"].max() <= meta.pre_end_day
    ), "발행 전 구간에 관찰창 밖 행이 섞여 있음"

    last_purchase_day = txn_pre.groupby("household_key")["DAY"].max()
    pre_baskets = txn_pre.groupby("household_key")["BASKET_ID"].nunique()
    pre_sales = txn_pre.groupby("household_key")["SALES_VALUE"].sum()
    pre_quantity = txn_pre.groupby("household_key")["QUANTITY"].sum()

    txn_pre_target = txn_pre.loc[txn_pre["PRODUCT_ID"].isin(target_products)]
    pre_target_purchase_count = txn_pre_target.groupby("household_key").size()
    pre_target_sales = txn_pre_target.groupby("household_key")["SALES_VALUE"].sum()

    redempt_pre = redempt.loc[
        redempt["household_key"].isin(households)
        & (redempt["DAY"] >= meta.pre_start_day) & (redempt["DAY"] <= meta.pre_end_day)
    ]
    pre_coupon_redemptions = redempt_pre.groupby("household_key").size()

    # 과거 캠페인 수신 횟수: "완전히 종료된"(END_DAY < start_day) 캠페인만 카운트.
    # pre_days 윈도로 제한하지 않는다 — 마케팅 노출 이력은 최근 구매행동과는 다른
    # 성격의 장기 신호이기 때문(캠페인 18 검증 시 결정한 규칙).
    completed_campaign_ids = set(
        desc.loc[(desc["CAMPAIGN"] != meta.campaign_id) & (desc["END_DAY"] < meta.start_day), "CAMPAIGN"]
    )
    assert not (completed_campaign_ids & set(meta.overlap_ids)), "완전종료 캠페인 집합에 겹치는 캠페인이 섞여 있음"
    table_completed = table.loc[
        table["household_key"].isin(households) & table["CAMPAIGN"].isin(completed_campaign_ids)
    ]
    pre_campaign_count_raw = table_completed.groupby("household_key")["CAMPAIGN"].nunique()

    demo_households = set(hh_demo["household_key"])

    # 캠페인 기간 결과변수: START_DAY <= DAY <= END_DAY
    txn_camp = txn_hh.loc[(txn_hh["DAY"] >= meta.start_day) & (txn_hh["DAY"] <= meta.end_day)]
    if not txn_camp.empty:
        assert txn_camp["DAY"].min() >= meta.start_day and txn_camp["DAY"].max() <= meta.end_day, (
            "캠페인 기간 구간에 밖 행이 섞여 있음"
        )
    total_sales = txn_camp.groupby("household_key")["SALES_VALUE"].sum()
    baskets = txn_camp.groupby("household_key")["BASKET_ID"].nunique()
    txn_camp_target = txn_camp.loc[txn_camp["PRODUCT_ID"].isin(target_products)]
    target_sales = txn_camp_target.groupby("household_key")["SALES_VALUE"].sum()
    target_quantity = txn_camp_target.groupby("household_key")["QUANTITY"].sum()
    target_purchase_count = txn_camp_target.groupby("household_key").size()

    df = group_df.set_index("household_key").copy()
    df["recency_days"] = meta.start_day - last_purchase_day
    df["pre_baskets"] = pre_baskets
    df["pre_sales"] = pre_sales
    df["pre_quantity"] = pre_quantity
    df["pre_target_purchase_count"] = pre_target_purchase_count
    df["pre_target_sales"] = pre_target_sales
    df["pre_coupon_redemptions"] = pre_coupon_redemptions
    df["pre_campaign_count_raw"] = pre_campaign_count_raw
    df["has_demographic"] = df.index.to_series().isin(demo_households).astype(int)
    df["target_sales"] = target_sales
    df["target_quantity"] = target_quantity
    df["target_purchase_count"] = target_purchase_count
    df["total_sales"] = total_sales
    df["baskets"] = baskets
    df = df.reset_index()

    _log(
        "3단계 사전/결과 변수 계산", df,
        f"(발행전 DAY {meta.pre_start_day}~{meta.pre_end_day}, 캠페인기간 DAY {meta.start_day}~{meta.end_day})",
    )
    return df


# ---------------------------------------------------------------------------
# 4) 결측치 처리
# ---------------------------------------------------------------------------
def handle_missing(df: pd.DataFrame, meta: CampaignMeta) -> pd.DataFrame:
    df = df.copy()
    n_missing_recency = int(df["recency_days"].isna().sum())

    # 활동 지표: 발행 전/캠페인 기간에 해당 활동이 없으면 0 (결측이 아니라 관찰된 사실)
    zero_fill_cols = [
        "pre_baskets", "pre_sales", "pre_quantity",
        "pre_target_purchase_count", "pre_target_sales", "pre_coupon_redemptions",
        "pre_campaign_count_raw",
        "target_sales", "target_quantity", "target_purchase_count", "total_sales", "baskets",
    ]
    df[zero_fill_cols] = df[zero_fill_cols].fillna(0)

    # recency: 발행 전 구매가 전혀 없으면 "관찰 가능한 최댓값"(pre_days)으로 대체 —
    # 0(오늘 구매)과 정반대 의미이므로 별도 처리.
    df["recency_days_filled"] = df["recency_days"].fillna(meta.pre_days)

    df["target_purchase"] = (df["target_purchase_count"] > 0).astype(int)
    df["any_purchase"] = (df["baskets"] > 0).astype(int)

    result_cols = OUTCOME_PRIMARY + OUTCOME_SECONDARY
    assert df[result_cols].isna().sum().sum() == 0, "결과변수에 결측이 남아 있음"

    _log(
        "4단계 결측치 처리", df,
        f"(recency 결측 {n_missing_recency}건 → pre_days({meta.pre_days})로 대체, 나머지는 0으로 채움)",
    )
    return df


# ---------------------------------------------------------------------------
# 5) 인코딩 (성향점수 입력변수 7개 생성)
# ---------------------------------------------------------------------------
def encode_features(df: pd.DataFrame, meta: CampaignMeta) -> pd.DataFrame:
    df = df.copy()

    effective_recency_cap = min(RECENCY_CAP_DAYS, meta.pre_days)
    df["pre_recency_capped"] = df["recency_days_filled"].clip(upper=effective_recency_cap)

    df["log_pre_baskets"] = np.log1p(df["pre_baskets"])
    df["log_pre_sales"] = np.log1p(df["pre_sales"])
    df["pre_target_share"] = np.where(df["pre_sales"] > 0, df["pre_target_sales"] / df["pre_sales"], 0.0)
    df["pre_coupon_user"] = (df["pre_coupon_redemptions"] > 0).astype(int)
    df["pre_campaign_count_c"] = df["pre_campaign_count_raw"].clip(upper=CAMPAIGN_COUNT_CAP)
    # has_demographic은 3단계에서 이미 생성됨

    assert df[FINAL_PROPENSITY_VARS].isna().sum().sum() == 0, "인코딩된 입력변수에 결측이 남아 있음"
    _log(
        "5단계 인코딩", df,
        f"(성향점수 입력변수 7개 생성 완료: {FINAL_PROPENSITY_VARS}, "
        f"recency cap = min({RECENCY_CAP_DAYS}, pre_days={meta.pre_days}) = {effective_recency_cap})",
    )
    return df


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
def prepare_data(campaign_id: int, pre_days: int, output_dir: Path | None = None) -> tuple[pd.DataFrame, CampaignMeta]:
    """output_dir: 결과 저장 위치(기본값 outputs/). 테스트 등에서 실제 캐시를 건드리지
    않으려면 임시 디렉터리를 넘긴다."""
    desc = pd.read_csv(RAW / "campaign_desc.csv")
    table = pd.read_csv(RAW / "campaign_table.csv")
    coupon = pd.read_csv(RAW / "coupon.csv")
    redempt = pd.read_csv(RAW / "coupon_redempt.csv")
    hh_demo = pd.read_csv(RAW / "hh_demographic.csv")
    txn = pd.read_csv(RAW / "transaction_data.csv")

    meta = get_campaign_info(campaign_id, pre_days, desc)
    treatment, control, group_df = build_treatment_control(meta, table)

    if meta.n_treatment == 0 or meta.n_control == 0:
        # 표본이 아예 없으면 이후 단계(사전변수·인코딩 등)를 계산할 수 없다. 예외를 던지지
        # 않고 빈 분석표를 반환해, 호출부(run_pipeline.py)의 표본 부족 게이트가 정상적인
        # status/reason 경로로 처리하게 한다.
        _log(
            "중단", None,
            f"처치({meta.n_treatment}) 또는 대조({meta.n_control}) 가구가 0명 — 사전변수 계산 단계로 진행하지 않음",
        )
        return group_df, meta

    df = compute_pre_period_features(group_df, meta, txn, coupon, redempt, desc, table, hh_demo)
    df = handle_missing(df, meta)
    df = encode_features(df, meta)

    out_dir = (output_dir or OUTPUTS) / f"campaign_{campaign_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "analysis_data.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    _log("저장", None, f"analysis_data.csv → {out_path}")

    return df, meta


if __name__ == "__main__":
    import argparse

    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="캠페인 데이터 준비 파이프라인")
    parser.add_argument("--campaign_id", type=int, required=True)
    parser.add_argument("--pre_days", type=int, required=True)
    args = parser.parse_args()
    prepare_data(args.campaign_id, args.pre_days)
