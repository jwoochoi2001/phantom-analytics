"""검증된 18번 캠페인 분석표(outputs/campaign_18/analysis_data.csv)에서
처치집단과 대조집단의 결과변수를 단순 비교한다.

주의: 이 단계는 매칭이나 보정을 거치지 않은 처치/대조 단순 비교이며,
여기서 나오는 차이는 "보정 전 관찰된 차이"일 뿐 인과효과가 아니다
(CLAUDE.md: 단순 집단 차이를 인과효과라고 표현하지 않는다).
매칭 후 추정치와는 반드시 구분해서 봐야 한다.
"""

import sys
from pathlib import Path

import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = Path(__file__).resolve().parent.parent / "outputs" / "campaign_18" / "analysis_data.csv"

PRIMARY = ["target_purchase", "target_sales", "target_quantity"]
SECONDARY = ["any_purchase", "total_sales", "baskets"]
BINARY = {"target_purchase", "any_purchase"}


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # -----------------------------------------------------------------
    # 분석표 재검증 (원본에서 새로 계산하지는 않되, 파일 내부 정합성만 확인)
    # -----------------------------------------------------------------
    assert set(df["group"].unique()) == {"처치", "대조"}, "group 값이 처치/대조 두 종류가 아님"
    n_t, n_c = (df["group"] == "처치").sum(), (df["group"] == "대조").sum()
    result_cols = PRIMARY + SECONDARY
    assert df[result_cols].isna().sum().sum() == 0, "결과변수에 결측이 있음"

    print(f"{'='*72}\n캠페인 18번 처치 vs 대조 단순 비교 (보정 전 관찰된 차이)\n{'='*72}")
    print(f"분석표: {DATA_PATH.name} ({len(df)}행) | 처치 {n_t}명 / 대조 {n_c}명\n")
    print("※ 아래 수치는 매칭·보정을 거치지 않은 단순 평균 비교이며 인과효과가 아니다.")
    print("  (성향점수 매칭 이후 추정치와 반드시 구분해서 해석할 것)\n")

    def compare(col: str) -> dict:
        t_vals = df.loc[df["group"] == "처치", col]
        c_vals = df.loc[df["group"] == "대조", col]
        mean_t, mean_c = t_vals.mean(), c_vals.mean()
        diff = mean_t - mean_c
        pct_diff = (diff / mean_c * 100) if mean_c != 0 else float("nan")

        if col in BINARY:
            # 두 비율 차이의 95% CI (정규근사, Wald)
            p_t, p_c = mean_t, mean_c
            se = ((p_t * (1 - p_t) / n_t) + (p_c * (1 - p_c) / n_c)) ** 0.5
            ci_low, ci_high = diff - 1.96 * se, diff + 1.96 * se
        else:
            # Welch 두 표본 t-검정 기반 평균차 95% CI
            tstat, pvalue = stats.ttest_ind(t_vals, c_vals, equal_var=False)
            se = (t_vals.var(ddof=1) / n_t + c_vals.var(ddof=1) / n_c) ** 0.5
            ci_low, ci_high = diff - 1.96 * se, diff + 1.96 * se

        return {
            "변수": col,
            "처치 평균": round(mean_t, 3),
            "대조 평균": round(mean_c, 3),
            "차이(처치-대조)": round(diff, 3),
            "차이 95% CI": f"[{ci_low:.3f}, {ci_high:.3f}]",
            "상대차이(%)": round(pct_diff, 1) if pct_diff == pct_diff else None,
        }

    primary_rows = [compare(c) for c in PRIMARY]
    secondary_rows = [compare(c) for c in SECONDARY]

    print("[주요 결과] 캠페인 대상 상품, 캠페인 기간 (DAY 587~642)")
    print(pd.DataFrame(primary_rows).to_string(index=False))
    print()
    print("[보조 결과] 전체 상품, 캠페인 기간 (DAY 587~642)")
    print(pd.DataFrame(secondary_rows).to_string(index=False))
    print()

    print("[해석 시 유의사항]")
    print("  - 처치/대조 가구는 아직 성향점수로 매칭되지 않은 원표본이다.")
    print("  - 위 차이 95% CI는 두 집단 평균/비율 차이 자체의 불확실성만 나타내며,")
    print("    관찰되지 않은 교란요인(가구 특성 차이 등)을 통제하지 않았다.")
    print("  - 따라서 이 표의 수치는 '처치집단과 대조집단이 보정 없이 다르게 관찰된 정도'이며,")
    print("    캠페인 18의 인과효과 추정치가 아니다. 인과효과는 매칭 이후 별도로 산출한다.")


if __name__ == "__main__":
    main()
