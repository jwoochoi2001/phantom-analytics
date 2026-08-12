"""캠페인 18: 공통지지영역 안의 가구만 대상으로 1:1 최근접이웃(NN) 매칭을 수행한다.

- 매칭 변수: logit(p_score) = log(p/(1-p))  (성향점수 매칭의 표준 관행 — 0~1로 압축된
  확률보다 꼬리가 완만해 거리 계산에 적합하다)
- caliper = {0.1, 0.2, 0.3} x SD(logit(p_score))  (Austin 2011 등에서 흔히 쓰는 절사폭.
  SD는 공통지지영역 내 처치+대조 전체 풀링 표본 기준으로 계산한다)
- 대조가구는 재사용하지 않는다(1:1, without replacement).
- 처치 가구를 매칭하는 순서는 결과가 순서에 좌우되지 않도록 고정 시드(42)로
  무작위화한다 — 이 규칙 전체가 나중에 pipeline/estimate_effect.py로 옮길 대상이다.

입력: outputs/campaign_18/analysis_data.csv (p_score, 7개 성향점수 입력변수 포함)
출력: 화면 출력만 (matched_data.csv 등 파일 저장은 최종 단계에서 별도 수행)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = Path(__file__).resolve().parent.parent / "outputs" / "campaign_18" / "analysis_data.csv"

FINAL_VARS = [
    "pre_recency_capped", "log_pre_baskets", "log_pre_sales",
    "pre_target_share", "pre_coupon_user", "pre_campaign_count_c", "has_demographic",
]
CALIPER_MULTIPLIERS = [0.1, 0.2, 0.3]
RNG_SEED = 42


def smd_continuous(t: np.ndarray, c: np.ndarray) -> float:
    pooled_sd = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
    return 0.0 if pooled_sd == 0 else (t.mean() - c.mean()) / pooled_sd


def compute_smd_table(matched_t: pd.DataFrame, matched_c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v in FINAL_VARS:
        smd = smd_continuous(matched_t[v].to_numpy(dtype=float), matched_c[v].to_numpy(dtype=float))
        rows.append({"변수": v, "SMD": round(smd, 3)})
    return pd.DataFrame(rows)


def nn_match(treat: pd.DataFrame, ctrl: pd.DataFrame, caliper: float, seed: int = RNG_SEED):
    """1:1 최근접이웃 매칭, 대조가구 재사용 금지. (household_key 쌍 리스트) 반환."""
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
        min_dist = dists[j]
        if min_dist <= caliper:
            chosen = avail_idx[j]
            used[chosen] = True
            pairs.append((t_id, ctrl_ids[chosen], min_dist))
    return pairs


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    assert "p_score" in df.columns, "p_score 없음 — campaign18_propensity_score.py 먼저 실행 필요"

    # -----------------------------------------------------------------
    # 공통지지영역 재계산 (campaign18_common_support.py와 동일 정의)
    # -----------------------------------------------------------------
    p_t = df.loc[df["group"] == "처치", "p_score"]
    p_c = df.loc[df["group"] == "대조", "p_score"]
    lo, hi = max(p_t.min(), p_c.min()), min(p_t.max(), p_c.max())
    df["in_support"] = df["p_score"].between(lo, hi)
    df["logit_p"] = np.log(df["p_score"] / (1 - df["p_score"]))

    common = df.loc[df["in_support"]].copy()
    treat_all = common.loc[common["group"] == "처치"].reset_index(drop=True)
    ctrl_all = common.loc[common["group"] == "대조"].reset_index(drop=True)

    sd_logit = common["logit_p"].std(ddof=1)

    print(f"{'='*78}\n캠페인 18: 공통지지영역 내 1:1 최근접이웃 매칭 (caliper별 비교)\n{'='*78}")
    print(f"공통지지영역: [{lo:.4f}, {hi:.4f}] → 처치 {len(treat_all)}명 / 대조 {len(ctrl_all)}명 (매칭 대상)")
    print(f"logit(p_score) 표준편차(공통지지영역 풀링): {sd_logit:.4f}\n")

    summary_rows = []
    for m in CALIPER_MULTIPLIERS:
        caliper = m * sd_logit
        pairs = nn_match(treat_all, ctrl_all, caliper)

        n_pairs = len(pairs)
        match_rate = n_pairs / len(treat_all)
        n_excluded_treat = len(treat_all) - n_pairs
        n_unused_ctrl = len(ctrl_all) - n_pairs

        matched_t_ids = [p[0] for p in pairs]
        matched_c_ids = [p[1] for p in pairs]
        matched_t = df.loc[df["household_key"].isin(matched_t_ids)]
        matched_c = df.loc[df["household_key"].isin(matched_c_ids)]

        smd_table = compute_smd_table(matched_t, matched_c)
        max_abs_smd = smd_table["SMD"].abs().max()
        worst_var = smd_table.loc[smd_table["SMD"].abs().idxmax(), "변수"]

        print(f"--- caliper = {m} x SD = {caliper:.4f} ---")
        print(f"  매칭 쌍 수: {n_pairs} | 처치 매칭률: {match_rate:.1%} | "
              f"제외 처치가구: {n_excluded_treat}명 | 미사용 대조가구: {n_unused_ctrl}명")
        print(f"  매칭 후 SMD:")
        print("   " + smd_table.to_string(index=False).replace("\n", "\n   "))
        print(f"  매칭 후 최대 |SMD|: {max_abs_smd:.3f} ({worst_var})\n")

        summary_rows.append(
            {
                "caliper 배수": m,
                "caliper(logit)": round(caliper, 4),
                "매칭 쌍 수": n_pairs,
                "처치 매칭률": f"{match_rate:.1%}",
                "제외 처치가구": n_excluded_treat,
                "미사용 대조가구": n_unused_ctrl,
                "매칭후 최대|SMD|": round(max_abs_smd, 3),
                "최대 SMD 변수": worst_var,
            }
        )

    summary = pd.DataFrame(summary_rows)
    print(f"{'='*78}\ncaliper 조건별 비교 요약\n{'='*78}")
    print(summary.to_string(index=False))

    # -----------------------------------------------------------------
    # 기본 caliper 추천 규칙:
    #   1순위: 매칭 후 최대 |SMD|가 가장 작은 조건
    #   2순위(동점 또는 근소 차일 때): 표본을 과도하게 잃지 않는지 확인
    #     (원 처치 표본 510명 대비 매칭률로 비교)
    # 이 규칙 자체가 pipeline/estimate_effect.py로 옮길 의사결정 로직이다.
    # -----------------------------------------------------------------
    n_treatment_total = int((df["group"] == "처치").sum())
    summary["매칭률(원 처치 510명 대비)"] = (summary["매칭 쌍 수"] / n_treatment_total * 100).round(1).astype(str) + "%"

    best_idx = summary["매칭후 최대|SMD|"].idxmin()
    best = summary.loc[best_idx]
    print(f"\n{'='*78}\n기본 caliper 추천\n{'='*78}")
    print(f"1순위 기준(매칭 후 최대 |SMD| 최소): caliper {best['caliper 배수']}배 "
          f"(최대|SMD|={best['매칭후 최대|SMD|']}, 매칭 쌍 {best['매칭 쌍 수']}개)")
    for _, r in summary.iterrows():
        tag = " ← 최소" if r.name == best_idx else ""
        print(f"  caliper {r['caliper 배수']}배: 최대|SMD|={r['매칭후 최대|SMD|']:.3f}, "
              f"쌍수={r['매칭 쌍 수']}, 원표본 대비 매칭률={r['매칭률(원 처치 510명 대비)']}{tag}")

    max_pairs = summary["매칭 쌍 수"].max()
    print(f"\n2순위 확인(표본 손실): caliper {best['caliper 배수']}배의 매칭 쌍 {best['매칭 쌍 수']}개는 "
          f"세 조건 중 최대치({max_pairs}개, caliper {summary.loc[summary['매칭 쌍 수'].idxmax(), 'caliper 배수']}배) "
          f"대비 {max_pairs - best['매칭 쌍 수']}쌍({(max_pairs - best['매칭 쌍 수'])/max_pairs*100:.1f}%) 적을 뿐이라, "
          "표본을 과도하게 희생하지 않고 최선의 균형을 얻는다.")


if __name__ == "__main__":
    main()
