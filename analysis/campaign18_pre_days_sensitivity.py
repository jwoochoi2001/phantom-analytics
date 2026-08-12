"""캠페인 18을 여러 pre_days(발행 전 관찰 기간)로 돌려 결과가 얼마나 민감한지 확인한다.

발행 전 특성 계산 구간의 길이를 30/60/90/180/365/586(전체 이력)일로 바꿔가며
집단 구성(캠페인 18/대조군 정의는 pre_days와 무관해 동일), 매칭, 균형, 효과 추정치가
어떻게 달라지는지 비교한다.

출력: outputs/campaign18_pre_days_sensitivity.csv, 터미널 요약표
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from run_pipeline import run  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "campaign18_pre_days_sensitivity.csv"
PRE_DAYS_VALUES = [30, 60, 90, 180, 365, 586]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    for pd_ in PRE_DAYS_VALUES:
        print(f"\n{'#'*70}\npre_days={pd_}\n{'#'*70}")
        res = run(campaign_id=18, pre_days=pd_)
        row = {
            "pre_days": pd_,
            "status": res["status"],
            "n_matched_pairs": res.get("n_matched_pairs"),
            "match_rate": res.get("match_rate"),
            "max_abs_smd": res.get("max_abs_smd"),
        }
        if res["status"] == "ok":
            ts = res["outcomes"]["primary"]["target_sales"]
            row["raw_diff"] = ts["raw"]["diff"]
            row["matched_diff"] = ts["matched"]["diff"]
            row["matched_ci_lo"] = ts["matched"]["ci_lo"]
            row["matched_ci_hi"] = ts["matched"]["ci_hi"]
            row["significant"] = not (ts["matched"]["ci_lo"] <= 0 <= ts["matched"]["ci_hi"])
        rows.append(row)

    sens = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sens.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n{'='*70}\n캠페인 18 pre_days 민감도 요약\n{'='*70}")
    print(sens.to_string(index=False))
    print(f"\n저장 완료: {OUT_PATH}")

    print("\n[해석 메모]")
    print("  - status·매칭쌍수·SMD·매칭후 차이가 pre_days에 따라 어떻게 바뀌는지 위 표로 확인.")
    print("  - 모든 pre_days에서 결론(유의성)이 같다면 결과가 이 선택에 민감하지 않다는 뜻이고,")
    print("    바뀐다면 pre_days 선택 자체가 결론에 영향을 준다는 뜻이므로 근거를 명확히 밝혀야 한다.")


if __name__ == "__main__":
    main()
