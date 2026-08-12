"""캠페인 18: 선택한 매칭 조건(caliper = 0.2 x SD(logit p_score))의 매칭 쌍에서
캠페인 효과를 추정한다.

- 주요 결과: 대상 상품 구매율(target_purchase), 구매금액(target_sales), 구매수량(target_quantity)
- 보조 결과: 전체 구매율(any_purchase), 구매금액(total_sales), 장바구니 수(baskets)
- 보정 전 단순차이(원표본 845명, 독립 2표본)와 매칭 후 차이(232쌍, 대응표본)를 함께 제시하고
  각각 95% CI를 계산한다.

매칭은 이전 단계(campaign18_nn_matching.py)와 동일한 규칙 — 공통지지영역 내 1:1 최근접이웃,
대조가구 재사용 금지, 시드 42로 고정 — 을 그대로 재현해 같은 232쌍을 얻는다.

주의(CLAUDE.md): 매칭 후 차이도 관찰되지 않은 교란을 통제한 것은 아니므로 "추정된 효과"라고
표현하되 확정된 인과효과로 단정하지 않는다.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = Path(__file__).resolve().parent.parent / "outputs" / "campaign_18" / "analysis_data.csv"

CALIPER_MULTIPLIER = 0.2  # 지난 단계에서 확정한 기본 caliper
RNG_SEED = 42

PRIMARY = ["target_purchase", "target_sales", "target_quantity"]
SECONDARY = ["any_purchase", "total_sales", "baskets"]
LABELS = {
    "target_purchase": "대상 상품 구매율",
    "target_sales": "대상 상품 구매금액",
    "target_quantity": "대상 상품 구매수량",
    "any_purchase": "전체 구매율",
    "total_sales": "전체 구매금액",
    "baskets": "장바구니 수",
}


def nn_match(treat: pd.DataFrame, ctrl: pd.DataFrame, caliper: float, seed: int = RNG_SEED):
    ctrl_logit = ctrl["logit_p"].to_numpy()
    ctrl_ids = ctrl["household_key"].to_numpy()
    used = np.zeros(len(ctrl_ids), dtype=bool)
    rng = np.random.default_rng(seed)
    order = rng.permutation(treat.index.to_numpy())
    pairs = []
    for idx in order:
        t_id = treat.loc[idx, "household_key"]
        t_logit = treat.loc[idx, "logit_p"]
        avail_idx = np.where(~used)[0]
        if avail_idx.size == 0:
            continue
        dists = np.abs(ctrl_logit[avail_idx] - t_logit)
        j = np.argmin(dists)
        if dists[j] <= caliper:
            chosen = avail_idx[j]
            used[chosen] = True
            pairs.append((t_id, ctrl_ids[chosen]))
    return pairs


def raw_diff(df: pd.DataFrame, col: str) -> dict:
    t = df.loc[df["group"] == "처치", col]
    c = df.loc[df["group"] == "대조", col]
    mean_t, mean_c = t.mean(), c.mean()
    diff = mean_t - mean_c
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    return {
        "처치 평균": mean_t, "대조 평균": mean_c, "차이": diff,
        "CI_lo": diff - 1.96 * se, "CI_hi": diff + 1.96 * se, "n_t": len(t), "n_c": len(c),
    }


def matched_diff(pair_df: pd.DataFrame, col_t: str, col_c: str) -> dict:
    diffs = pair_df[col_t] - pair_df[col_c]
    n = len(diffs)
    mean_diff = diffs.mean()
    se = diffs.std(ddof=1) / np.sqrt(n)
    return {
        "처치 평균": pair_df[col_t].mean(), "대조 평균": pair_df[col_c].mean(), "차이": mean_diff,
        "CI_lo": mean_diff - 1.96 * se, "CI_hi": mean_diff + 1.96 * se, "n": n,
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    p_t = df.loc[df["group"] == "처치", "p_score"]
    p_c = df.loc[df["group"] == "대조", "p_score"]
    lo, hi = max(p_t.min(), p_c.min()), min(p_t.max(), p_c.max())
    df["in_support"] = df["p_score"].between(lo, hi)
    df["logit_p"] = np.log(df["p_score"] / (1 - df["p_score"]))

    common = df.loc[df["in_support"]].copy()
    treat_all = common.loc[common["group"] == "처치"].reset_index(drop=True)
    ctrl_all = common.loc[common["group"] == "대조"].reset_index(drop=True)
    sd_logit = common["logit_p"].std(ddof=1)
    caliper = CALIPER_MULTIPLIER * sd_logit

    pairs = nn_match(treat_all, ctrl_all, caliper)
    print(f"{'='*78}\n캠페인 18: 매칭 쌍(caliper {CALIPER_MULTIPLIER} x SD = {caliper:.4f}) 기반 효과 추정\n{'='*78}")
    print(f"매칭 쌍 수: {len(pairs)}쌍 (처치 {len(pairs)}명 vs 대조 {len(pairs)}명, 원 처치 510명 중 {len(pairs)/510:.1%})\n")

    outcome_cols = ["household_key"] + PRIMARY + SECONDARY
    t_lookup = df.set_index("household_key")[PRIMARY + SECONDARY]
    pair_rows = []
    for t_id, c_id in pairs:
        row = {"t_id": t_id, "c_id": c_id}
        for col in PRIMARY + SECONDARY:
            row[f"{col}_t"] = t_lookup.loc[t_id, col]
            row[f"{col}_c"] = t_lookup.loc[c_id, col]
        pair_rows.append(row)
    pair_df = pd.DataFrame(pair_rows)
    print(f"매칭쌍 데이터 shape = {pair_df.shape[0]}행 x {pair_df.shape[1]}열\n")

    def build_table(cols, title):
        rows = []
        for col in cols:
            raw = raw_diff(df, col)
            mat = matched_diff(pair_df, f"{col}_t", f"{col}_c")
            rows.append(
                {
                    "변수": f"{col} ({LABELS[col]})",
                    "보정전 처치평균": round(raw["처치 평균"], 3),
                    "보정전 대조평균": round(raw["대조 평균"], 3),
                    "보정전 차이": round(raw["차이"], 3),
                    "보정전 95%CI": f"[{raw['CI_lo']:.3f}, {raw['CI_hi']:.3f}]",
                    "매칭후 처치평균": round(mat["처치 평균"], 3),
                    "매칭후 대조평균": round(mat["대조 평균"], 3),
                    "매칭후 차이": round(mat["차이"], 3),
                    "매칭후 95%CI": f"[{mat['CI_lo']:.3f}, {mat['CI_hi']:.3f}]",
                }
            )
        table = pd.DataFrame(rows)
        print(f"[{title}]  (보정전 n=845(처치510/대조335, 독립2표본) vs 매칭후 n={len(pair_df)}쌍(대응표본))")
        print(table.to_string(index=False))
        print()
        return table

    primary_table = build_table(PRIMARY, "주요 결과 — 캠페인 대상 상품, 캠페인 기간(DAY 587~642)")
    secondary_table = build_table(SECONDARY, "보조 결과 — 전체 상품, 캠페인 기간(DAY 587~642)")

    print("[해석 메모]")
    print("  - '매칭후 차이'는 232쌍의 대응표본(paired) 차이이며, 발행 전 특성(7개 변수) 분포가")
    print("    맞춰진 상태에서의 차이다. 보정전 단순차이와의 격차가 발행 전 불균형이 결과에 얼마나")
    print("    섞여 있었는지를 보여준다.")
    print("  - 95% CI가 0을 포함하지 않으면 매칭 표본 내에서는 관찰된 차이가 우연으로 보기 어렵다는")
    print("    뜻이다. 다만 이는 여전히 '관찰된 발행 전 특성'만 통제한 결과이며, 관찰되지 않은")
    print("    교란요인(예: 오프라인 접촉, 개인 성향)은 통제되지 않았으므로 확정된 인과효과로")
    print("    단정하지 않는다.")
    print("  - 매칭 표본은 원 처치 510명 중 232명(45.5%)만 포함하므로, 이 결과는 '대조군과 비교")
    print("    가능했던 처치 가구'에 대한 추정이며 전체 처치 가구로 일반화하지 않는다.")


if __name__ == "__main__":
    main()
