"""캠페인별 기간, 수신 가구 수, 기간이 겹치는 다른 캠페인, 처치/대조 후보 수를 확인한다.

- 처치 가구: 선택 캠페인을 받았고 같은 기간에 겹치는 다른 캠페인은 받지 않은 가구.
- 대조 후보 가구: 선택 캠페인을 받지 않았고 같은 기간에 겹치는 다른 캠페인도 받지 않은 가구.
  (CLAUDE.md 분석 설계 규칙과 동일한 정의)
- 가구 모집단은 campaign_table.csv에 등장하는 전체 가구(캠페인을 1회 이상 받은 가구, 1,584명)로 한정한다.
  campaign_table.csv 자체가 "캠페인 수신 가구" 단위 파일이라 이 파일만으로는 캠페인을 한 번도
  받지 않은 가구를 알 수 없기 때문이다. 대조 후보 수는 이 모집단 안에서만 계산한 값이다.
- 표본이 작다고 판단하는 기준은 임계값(MIN_N, 기본 30명)이며 상황에 맞게 조정한다.
"""

import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
MIN_N = 30  # 처치/대조 후보 수가 이 값보다 작으면 표본 부족으로 표시 (조정 가능한 임계값)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def main() -> None:
    desc = pd.read_csv(RAW / "campaign_desc.csv")
    table = pd.read_csv(RAW / "campaign_table.csv")

    universe = set(table["household_key"].unique())
    n_universe = len(universe)

    recipients = table.groupby("CAMPAIGN")["household_key"].apply(set).to_dict()

    desc = desc.sort_values("CAMPAIGN").reset_index(drop=True)

    rows = []
    for _, r in desc.iterrows():
        c = int(r["CAMPAIGN"])
        s, e = int(r["START_DAY"]), int(r["END_DAY"])

        overlap_mask = (desc["CAMPAIGN"] != c) & (desc["START_DAY"] <= e) & (desc["END_DAY"] >= s)
        overlapping = desc.loc[overlap_mask, "CAMPAIGN"].astype(int).tolist()

        recip_c = recipients.get(c, set())
        n_received = len(recip_c)

        overlap_recip = set()
        for oc in overlapping:
            overlap_recip |= recipients.get(oc, set())

        treatment = recip_c - overlap_recip
        n_treatment = len(treatment)

        control_candidates = universe - recip_c - overlap_recip
        n_control = len(control_candidates)

        rows.append(
            {
                "CAMPAIGN": c,
                "TYPE": r["DESCRIPTION"],
                "START_DAY": s,
                "END_DAY": e,
                "DURATION_DAYS": e - s + 1,
                "N_RECEIVED": n_received,
                "OVERLAPPING_CAMPAIGNS": overlapping if overlapping else "-",
                "N_TREATMENT": n_treatment,
                "N_CONTROL_CANDIDATES": n_control,
            }
        )

    result = pd.DataFrame(rows)
    result["SMALL_TREATMENT"] = result["N_TREATMENT"] < MIN_N
    result["INSUFFICIENT_CONTROL"] = result["N_CONTROL_CANDIDATES"] < MIN_N

    print(f"가구 모집단(campaign_table.csv 기준, 캠페인을 1회 이상 받은 가구): {n_universe}명")
    print(f"표본 부족 판단 임계값(MIN_N): {MIN_N}명")
    print()
    print("=== 캠페인별 기간, 수신 가구, 겹치는 캠페인, 처치/대조 후보 수 ===")
    print(
        result[
            [
                "CAMPAIGN",
                "TYPE",
                "START_DAY",
                "END_DAY",
                "DURATION_DAYS",
                "N_RECEIVED",
                "OVERLAPPING_CAMPAIGNS",
                "N_TREATMENT",
                "N_CONTROL_CANDIDATES",
            ]
        ].to_string(index=False)
    )

    flagged = result[result["SMALL_TREATMENT"] | result["INSUFFICIENT_CONTROL"]]
    print()
    print(f"=== 표본 부족 또는 대조군 부족 캠페인 ({len(flagged)}/{len(result)}건, 기준: N < {MIN_N}) ===")
    if flagged.empty:
        print("해당 없음")
    else:
        for _, r in flagged.iterrows():
            reasons = []
            if r["SMALL_TREATMENT"]:
                reasons.append(f"처치 가구 부족(N_TREATMENT={r['N_TREATMENT']})")
            if r["INSUFFICIENT_CONTROL"]:
                reasons.append(f"대조 후보 부족(N_CONTROL_CANDIDATES={r['N_CONTROL_CANDIDATES']})")
            print(f"  CAMPAIGN {r['CAMPAIGN']} ({r['TYPE']}): " + ", ".join(reasons))


if __name__ == "__main__":
    main()
