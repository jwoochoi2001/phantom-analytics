"""18번 캠페인의 기간, 수신 가구, 대상 상품, 겹치는 캠페인을 확인한다.

사용 원본 파일 3개와 그 안에서 사용하는 열:
- campaign_desc.csv : CAMPAIGN, DESCRIPTION, START_DAY, END_DAY   (캠페인 자체의 정보)
- campaign_table.csv: household_key, CAMPAIGN                    (캠페인을 받은 가구)
- coupon.csv        : CAMPAIGN, PRODUCT_ID, COUPON_UPC           (캠페인이 겨냥한 상품)

각 단계마다 어떤 파일·열에서 값이 나왔는지 print로 함께 설명한다.
분석표 저장 없이 터미널 출력만 수행한다(CLAUDE.md 파일 관리 규칙).
"""

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
TARGET_CAMPAIGN = 18


def main() -> None:
    print(f"{'='*70}\n캠페인 {TARGET_CAMPAIGN}번 처치 대상 프로파일\n{'='*70}\n")

    # -----------------------------------------------------------------
    # 1단계: campaign_desc.csv에서 캠페인 기본 정보(기간) 조회
    # -----------------------------------------------------------------
    desc = pd.read_csv(RAW / "campaign_desc.csv")
    row18 = desc.loc[desc["CAMPAIGN"] == TARGET_CAMPAIGN]
    if row18.empty:
        raise ValueError(f"campaign_desc.csv에 CAMPAIGN={TARGET_CAMPAIGN}이 없습니다.")
    row18 = row18.iloc[0]

    camp_type = row18["DESCRIPTION"]
    start_day = int(row18["START_DAY"])
    end_day = int(row18["END_DAY"])
    duration = end_day - start_day + 1

    print("[1단계] 캠페인 기간")
    print(f"  출처: campaign_desc.csv, CAMPAIGN=={TARGET_CAMPAIGN} 행의 START_DAY / END_DAY / DESCRIPTION 열")
    print(f"  캠페인 유형(DESCRIPTION): {camp_type}")
    print(f"  시작일(START_DAY): {start_day}  (달력 날짜가 아닌 데이터 기준 상대 일자)")
    print(f"  종료일(END_DAY): {end_day}")
    print(f"  기간 길이: {duration}일 (END_DAY - START_DAY + 1)")
    print()

    # -----------------------------------------------------------------
    # 2단계: campaign_table.csv에서 수신 가구 조회
    # -----------------------------------------------------------------
    table = pd.read_csv(RAW / "campaign_table.csv")
    recipients18 = table.loc[table["CAMPAIGN"] == TARGET_CAMPAIGN, "household_key"].unique()
    n_recipients = len(recipients18)

    print("[2단계] 수신 가구")
    print(f"  출처: campaign_table.csv, CAMPAIGN=={TARGET_CAMPAIGN} 행의 household_key 열(중복 제거)")
    print(f"  수신 가구 수: {n_recipients}명")
    print(f"  household_key 예시(최대 10개): {[int(x) for x in sorted(recipients18)[:10]]}")
    print()

    # -----------------------------------------------------------------
    # 3단계: coupon.csv에서 대상 상품 조회
    # -----------------------------------------------------------------
    coupon = pd.read_csv(RAW / "coupon.csv")
    coupon18 = coupon.loc[coupon["CAMPAIGN"] == TARGET_CAMPAIGN]
    target_products = coupon18["PRODUCT_ID"].unique()
    target_coupons = coupon18["COUPON_UPC"].unique()

    print("[3단계] 대상 상품")
    print(f"  출처: coupon.csv, CAMPAIGN=={TARGET_CAMPAIGN} 행의 PRODUCT_ID 열(중복 제거)")
    print(f"  대상 상품(PRODUCT_ID) 수: {len(target_products)}개")
    print(f"  대상 쿠폰(COUPON_UPC) 수: {len(target_coupons)}개  (coupon.csv의 COUPON_UPC 열, 참고용)")
    print(f"  PRODUCT_ID 예시(최대 10개): {[int(x) for x in sorted(target_products)[:10]]}")
    print()

    # -----------------------------------------------------------------
    # 4단계: campaign_desc.csv 전체와 비교해 기간이 겹치는 다른 캠페인 탐색
    # -----------------------------------------------------------------
    overlap_mask = (
        (desc["CAMPAIGN"] != TARGET_CAMPAIGN)
        & (desc["START_DAY"] <= end_day)
        & (desc["END_DAY"] >= start_day)
    )
    overlapping = desc.loc[overlap_mask, ["CAMPAIGN", "DESCRIPTION", "START_DAY", "END_DAY"]].sort_values("CAMPAIGN")

    print("[4단계] 기간이 겹치는 다른 캠페인")
    print(
        f"  출처: campaign_desc.csv 전체 행 중 CAMPAIGN!={TARGET_CAMPAIGN}이면서 "
        f"START_DAY <= {end_day}(18번 END_DAY) AND END_DAY >= {start_day}(18번 START_DAY)인 행"
    )
    print(f"  겹치는 캠페인 수: {len(overlapping)}개")
    if overlapping.empty:
        print("  없음")
    else:
        print(overlapping.to_string(index=False))
    print()

    # -----------------------------------------------------------------
    # 5단계: 겹치는 캠페인까지 받은 가구를 제외한 순수 처치 가구 (campaign_table.csv 재조회)
    # -----------------------------------------------------------------
    overlap_ids = overlapping["CAMPAIGN"].tolist()
    overlap_recipients = set(
        table.loc[table["CAMPAIGN"].isin(overlap_ids), "household_key"].unique()
    )
    pure_treatment = set(recipients18) - overlap_recipients
    n_pure_treatment = len(pure_treatment)

    print("[5단계] 순수 처치 가구 (18번만 받고 겹치는 캠페인은 안 받은 가구)")
    print(
        "  출처: campaign_table.csv에서 18번 수신 가구 집합 - (4단계 겹치는 캠페인 수신 가구 집합)"
    )
    print(f"  겹치는 캠페인 중 하나라도 받은 18번 수신 가구 수: {len(set(recipients18) & overlap_recipients)}명")
    print(f"  순수 처치 가구 수: {n_pure_treatment}명 (전체 18번 수신 {n_recipients}명 중)")
    print()

    # -----------------------------------------------------------------
    # 6단계: 대조군(비처치 가구) 확정
    #   - 모집단은 campaign_table.csv에 등장하는 전체 가구(=캠페인을 1회 이상 받은 가구)로 한정한다.
    #   - 대조군 = 이 모집단 중 18번도, 18번과 겹치는 8개 캠페인도 받지 않은 가구.
    #   - "캠페인을 아예 한 번도 받지 않은 가구"(transaction_data.csv 전체 가구 - campaign_table.csv
    #     가구)는 애초에 마케팅 대상이 아니었던 별개 집단이므로 대조군에서 제외한다.
    # -----------------------------------------------------------------
    campaign_universe = set(table["household_key"].unique())
    control = campaign_universe - set(recipients18) - overlap_recipients
    n_control = len(control)

    txn_households = set(
        pd.read_csv(RAW / "transaction_data.csv", usecols=["household_key"])["household_key"].unique()
    )
    never_campaigned = txn_households - campaign_universe
    n_never_campaigned = len(never_campaigned)

    # 검증: 대조군 가구가 실제로 받은 캠페인이 겹치는 8개(overlap_ids)와 하나도 안 겹치는지 확인
    control_actual_campaigns = set(table.loc[table["household_key"].isin(control), "CAMPAIGN"].unique())
    assert not (control_actual_campaigns & set(overlap_ids)), "대조군에 겹치는 캠페인 수신 가구가 섞여 있음"
    assert TARGET_CAMPAIGN not in control_actual_campaigns, "대조군에 18번 수신 가구가 섞여 있음"

    print("[6단계] 대조군(비처치) 가구 확정")
    print(
        "  출처: campaign_table.csv 전체 가구(household_key 고유값) - 18번 수신 가구 - "
        "겹치는 8개 캠페인 수신 가구"
    )
    print(f"  캠페인 수신 이력이 있는 전체 가구(모집단): {len(campaign_universe)}명")
    print(f"  대조군(비처치) 가구 수: {n_control}명")
    print(
        f"  검증: 대조군이 실제 받은 캠페인 ID = {sorted(int(c) for c in control_actual_campaigns)[:5]}...(생략), "
        f"겹치는 8개 캠페인과 교집합 없음 확인됨"
    )
    print()
    print("  참고: transaction_data.csv 기준 거래 가구 전체와 비교")
    print(f"    거래 이력이 있는 전체 가구: {len(txn_households)}명 (transaction_data.csv household_key)")
    print(
        f"    이 중 캠페인을 한 번도 받지 않은 가구: {n_never_campaigned}명 "
        "(campaign_table.csv에 아예 등장하지 않는 가구) → 대조군에서 제외"
    )
    print("    사유: 처음부터 마케팅 대상이 아니었던 별개 모집단이라 18번 수신 가구와 비교 대상이 아님")
    print()

    # -----------------------------------------------------------------
    # 7단계: 분석집단 3분류 (처치 / 대조 / 제외) 및 최종 검증
    #   모집단은 6단계와 동일한 campaign_table.csv 전체 가구(1,584명)로 고정한다.
    #   각 가구는 아래 세 그룹 중 정확히 하나에만 속하도록 정의한다(상호배타·모두포함).
    #     1) 18번만 받은 가구            = 처치집단   = pure_treatment (5단계)
    #     2) 같은 기간 어떤 캠페인도 안 받은 가구 = 대조집단   = control (6단계)
    #     3) 겹치는 캠페인에도 노출되어 제외된 가구 = overlap_recipients (18번 수신 여부 무관)
    # -----------------------------------------------------------------
    group_treatment = pure_treatment
    group_control = control
    group_excluded = overlap_recipients & campaign_universe  # 겹치는 캠페인 수신 가구(18번 동시수신 포함)

    groups_df = pd.DataFrame(
        [
            {"그룹": "1) 18번만 받음 (처치집단)", "가구 수": len(group_treatment)},
            {"그룹": "2) 동기간 어떤 캠페인도 안 받음 (대조집단)", "가구 수": len(group_control)},
            {"그룹": "3) 겹치는 캠페인에도 노출되어 제외", "가구 수": len(group_excluded)},
        ]
    )
    groups_df.loc[len(groups_df)] = ["합계(모집단, campaign_table.csv 전체 가구)", len(campaign_universe)]

    print("[7단계] 분석집단 3분류")
    print("  출처: campaign_table.csv (household_key, CAMPAIGN) 기준 2·4·5·6단계 결과를 재구성")
    print(groups_df.to_string(index=False))
    print()

    # 검증 1: 세 그룹의 합이 모집단과 정확히 일치하는지 (상호배타 + 모두포함)
    union_check = group_treatment | group_control | group_excluded
    sum_matches_union = len(group_treatment) + len(group_control) + len(group_excluded) == len(union_check)
    assert union_check == campaign_universe, "세 그룹의 합집합이 모집단과 다름"
    assert sum_matches_union, "그룹 간 중복이 존재해 단순 합산 수치가 합집합 크기와 다름"

    # 검증 2: 최종 처치집단과 대조집단에 같은 가구가 없는지 확인
    treatment_control_overlap = group_treatment & group_control
    assert not treatment_control_overlap, "처치집단과 대조집단에 중복 가구가 있음"

    print("[검증] 최종 처치집단과 대조집단 중복 여부")
    print(f"  처치집단({len(group_treatment)}명) ∩ 대조집단({len(group_control)}명) = {len(treatment_control_overlap)}명")
    print("  → 중복 없음 확인됨" if not treatment_control_overlap else "  → 경고: 중복 가구 존재")
    print(f"  세 그룹 합({len(group_treatment)}+{len(group_control)}+{len(group_excluded)}="
          f"{len(group_treatment)+len(group_control)+len(group_excluded)}) = 모집단({len(campaign_universe)}) 일치 확인됨")
    print()

    print(f"{'='*70}\n요약\n{'='*70}")
    print(f"캠페인 18 ({camp_type}) | 기간: DAY {start_day}~{end_day} ({duration}일)")
    print(f"수신 가구: {n_recipients}명 | 순수 처치 가구: {n_pure_treatment}명")
    print(f"대상 상품: {len(target_products)}개 | 겹치는 캠페인: {overlap_ids if overlap_ids else '없음'}")
    print(f"대조군(비처치) 가구: {n_control}명  (참고: 캠페인 자체를 받은 적 없는 가구 {n_never_campaigned}명은 대조군에서 제외)")
    print(f"최종 처치집단 {len(group_treatment)}명 / 대조집단 {len(group_control)}명 / 제외 {len(group_excluded)}명"
          f" (처치·대조 중복 0명 확인)")


if __name__ == "__main__":
    main()
