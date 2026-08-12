"""성향점수 추정 · 공통지지영역 확인 · 매칭 · 균형/효과 추정 파이프라인.

캠페인 18 분석(analysis/campaign18_propensity_score.py, campaign18_common_support.py,
campaign18_nn_matching.py, campaign18_matched_effect.py)에서 검증한 절차를 일반화한다.

실행 순서: 성향점수 → 공통영역 확인 → 매칭 → 균형과 효과 추정.
각 단계에 표본/영역/불균형 기준(제약사항)을 두고, 기준 미달 시 효과를 계산하지 않고
status와 reason을 반환한다(CLAUDE.md: 표본 부족·공통영역 부족·잔여 불균형 시 상태와
이유 반환).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import FINAL_PROPENSITY_VARS, OUTCOME_PRIMARY, OUTCOME_SECONDARY, CampaignMeta  # noqa: E402

OUTPUTS = Path(__file__).resolve().parent.parent / "outputs"

# ===========================================================================
# 제약사항(CONSTANTS): 캠페인 18에서 검증한 매칭·품질 기준.
# ===========================================================================
CALIPER_MULTIPLIER = 0.2          # campaign18_nn_matching.py에서 확정한 기본 caliper
RNG_SEED = 42                     # 매칭 순서 고정 시드
MIN_COMMON_SUPPORT_N = 30         # 공통지지영역 내 최소 표본(집단별)
MIN_MATCHED_PAIRS = 30            # 매칭 후 최소 쌍 수
MAX_ALLOWED_SMD = 0.1             # 매칭 후 허용 가능한 최대 |SMD| (관례적 임계값)


def _log(step: str, extra: str = "") -> None:
    print(f"[estimate_effect] {step}  {extra}")


def result(status: str, reason: str = "", **kwargs) -> dict[str, Any]:
    return {"status": status, "reason": reason, **kwargs}


# ---------------------------------------------------------------------------
# 6) 성향점수
# ---------------------------------------------------------------------------
def fit_propensity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    X = df[FINAL_PROPENSITY_VARS].to_numpy(dtype=float)
    y = df["treatment"].to_numpy()

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_std, y)
    df["p_score"] = model.predict_proba(X_std)[:, 1]
    df["logit_p"] = np.log(df["p_score"] / (1 - df["p_score"]))

    _log(
        "6단계 성향점수", f"n={len(df)}, p_score 범위=[{df['p_score'].min():.4f}, {df['p_score'].max():.4f}]"
    )
    return df


# ---------------------------------------------------------------------------
# 7) 공통영역 확인
# ---------------------------------------------------------------------------
def check_common_support(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float, dict | None]:
    p_t = df.loc[df["group"] == "처치", "p_score"]
    p_c = df.loc[df["group"] == "대조", "p_score"]
    lo, hi = max(p_t.min(), p_c.min()), min(p_t.max(), p_c.max())
    df = df.copy()
    df["in_support"] = df["p_score"].between(lo, hi)

    n_t_in = int(((df["group"] == "처치") & df["in_support"]).sum())
    n_c_in = int(((df["group"] == "대조") & df["in_support"]).sum())
    _log("7단계 공통영역 확인", f"[{lo:.4f}, {hi:.4f}] → 처치 {n_t_in}명 / 대조 {n_c_in}명")

    if n_t_in < MIN_COMMON_SUPPORT_N or n_c_in < MIN_COMMON_SUPPORT_N:
        return df, lo, hi, result(
            "insufficient_overlap",
            f"공통지지영역 내 표본 부족: 처치 {n_t_in}명, 대조 {n_c_in}명 "
            f"(기준 각 {MIN_COMMON_SUPPORT_N}명 이상). 공통지지영역=[{lo:.4f},{hi:.4f}]",
            common_support=[round(lo, 4), round(hi, 4)], n_treatment_in_support=n_t_in, n_control_in_support=n_c_in,
        )
    return df, lo, hi, None


# ---------------------------------------------------------------------------
# 8) 매칭
# ---------------------------------------------------------------------------
def nn_match(treat: pd.DataFrame, ctrl: pd.DataFrame, caliper: float, seed: int = RNG_SEED) -> list[tuple]:
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
            pairs.append((t_id, ctrl_ids[chosen], dists[j]))
    return pairs


def run_matching(df: pd.DataFrame) -> tuple[list[tuple], float, dict | None]:
    common = df.loc[df["in_support"]].copy()
    treat_all = common.loc[common["group"] == "처치"].reset_index(drop=True)
    ctrl_all = common.loc[common["group"] == "대조"].reset_index(drop=True)
    sd_logit = common["logit_p"].std(ddof=1)
    caliper = CALIPER_MULTIPLIER * sd_logit

    pairs = nn_match(treat_all, ctrl_all, caliper)
    n_pairs = len(pairs)
    match_rate = n_pairs / len(treat_all) if len(treat_all) else 0.0
    _log(
        "8단계 매칭",
        f"caliper={CALIPER_MULTIPLIER} x SD({sd_logit:.4f})={caliper:.4f} → {n_pairs}쌍 "
        f"(공통영역 처치 {len(treat_all)}명 대비 매칭률 {match_rate:.1%})",
    )

    if n_pairs < MIN_MATCHED_PAIRS:
        return pairs, caliper, result(
            "insufficient_sample",
            f"매칭 쌍 부족: {n_pairs}쌍 (기준 {MIN_MATCHED_PAIRS}쌍 이상)",
            n_matched_pairs=n_pairs, match_rate=round(match_rate, 4),
        )
    return pairs, caliper, None


# ---------------------------------------------------------------------------
# 9) 균형과 효과 추정
# ---------------------------------------------------------------------------
def smd_continuous(t: np.ndarray, c: np.ndarray) -> float:
    pooled_sd = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2)
    return 0.0 if pooled_sd == 0 else (t.mean() - c.mean()) / pooled_sd


def _smd_table(t_df: pd.DataFrame, c_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v in FINAL_PROPENSITY_VARS:
        smd = smd_continuous(t_df[v].to_numpy(dtype=float), c_df[v].to_numpy(dtype=float))
        rows.append({"변수": v, "SMD": round(smd, 3)})
    return pd.DataFrame(rows)


def compute_balance_before(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """매칭 전 SMD — 공통지지영역 내 처치 전체 vs 대조 전체 기준."""
    common = df.loc[df["in_support"]]
    smd_df = _smd_table(common.loc[common["group"] == "처치"], common.loc[common["group"] == "대조"])
    max_abs_smd = float(smd_df["SMD"].abs().max())
    _log("9단계 균형 확인(매칭 전)", f"공통영역 내 최대 |SMD| = {max_abs_smd:.3f}")
    return smd_df, max_abs_smd


def compute_balance(df: pd.DataFrame, pairs: list[tuple]) -> tuple[pd.DataFrame, float]:
    """매칭 후 SMD — 매칭된 쌍만 기준."""
    t_ids = [p[0] for p in pairs]
    c_ids = [p[1] for p in pairs]
    matched_t = df.loc[df["household_key"].isin(t_ids)]
    matched_c = df.loc[df["household_key"].isin(c_ids)]
    smd_df = _smd_table(matched_t, matched_c)
    max_abs_smd = float(smd_df["SMD"].abs().max())
    _log("9단계 균형 확인(매칭 후)", f"매칭 후 최대 |SMD| = {max_abs_smd:.3f}")
    return smd_df, max_abs_smd


def raw_diff(df: pd.DataFrame, col: str) -> dict:
    t = df.loc[df["group"] == "처치", col]
    c = df.loc[df["group"] == "대조", col]
    diff = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
    return {"mean_t": t.mean(), "mean_c": c.mean(), "diff": diff, "ci_lo": diff - 1.96 * se, "ci_hi": diff + 1.96 * se}


def matched_diff(pair_df: pd.DataFrame, col_t: str, col_c: str) -> dict:
    diffs = pair_df[col_t] - pair_df[col_c]
    n = len(diffs)
    mean_diff = diffs.mean()
    se = diffs.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return {
        "mean_t": pair_df[col_t].mean(), "mean_c": pair_df[col_c].mean(), "diff": mean_diff,
        "ci_lo": mean_diff - 1.96 * se, "ci_hi": mean_diff + 1.96 * se,
    }


def estimate_outcomes(df: pd.DataFrame, pairs: list[tuple]) -> dict:
    lookup = df.set_index("household_key")[OUTCOME_PRIMARY + OUTCOME_SECONDARY]
    pair_rows = []
    for t_id, c_id, _ in pairs:
        row = {}
        for col in OUTCOME_PRIMARY + OUTCOME_SECONDARY:
            row[f"{col}_t"] = lookup.loc[t_id, col]
            row[f"{col}_c"] = lookup.loc[c_id, col]
        pair_rows.append(row)
    pair_df = pd.DataFrame(pair_rows)

    def build(cols):
        out = {}
        for col in cols:
            raw = raw_diff(df, col)
            mat = matched_diff(pair_df, f"{col}_t", f"{col}_c")
            out[col] = {"raw": {k: round(v, 4) for k, v in raw.items()},
                        "matched": {k: round(v, 4) for k, v in mat.items()}}
        return out

    return {"primary": build(OUTCOME_PRIMARY), "secondary": build(OUTCOME_SECONDARY)}


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------
def estimate_effect(df: pd.DataFrame, meta: CampaignMeta) -> dict:
    base_ctx = dict(
        campaign_id=meta.campaign_id, pre_days=meta.pre_days,
        start_day=meta.start_day, end_day=meta.end_day, overlap_ids=meta.overlap_ids,
        n_treatment=meta.n_treatment, n_control=meta.n_control,
    )

    df = fit_propensity(df)

    df, lo, hi, gate = check_common_support(df)

    # analysis_data.csv를 p_score/logit_p/in_support가 포함된 상태로 갱신 저장한다.
    # 게이트에 걸려 이후 단계를 진행하지 못하더라도, 성향점수 분포·공통영역 진단은
    # 화면(app)에서 항상 조회할 수 있어야 하기 때문이다.
    out_dir = OUTPUTS / f"campaign_{meta.campaign_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "analysis_data.csv", index=False, encoding="utf-8-sig")
    _log("저장", f"analysis_data.csv (p_score 포함) → {out_dir / 'analysis_data.csv'}")

    if gate is not None:
        return {**base_ctx, **gate}

    support_ctx = {
        **base_ctx,
        "common_support": [round(lo, 4), round(hi, 4)],
        "n_common_support_treatment": int(((df["group"] == "처치") & df["in_support"]).sum()),
        "n_common_support_control": int(((df["group"] == "대조") & df["in_support"]).sum()),
    }

    smd_before_df, max_abs_smd_before = compute_balance_before(df)

    pairs, caliper, gate = run_matching(df)
    if gate is not None:
        return {**support_ctx, **gate, "balance_smd_before": smd_before_df.to_dict(orient="records"),
                "max_abs_smd_before": round(max_abs_smd_before, 4)}

    smd_df, max_abs_smd = compute_balance(df, pairs)
    balance_ok = max_abs_smd <= MAX_ALLOWED_SMD

    outcomes = estimate_outcomes(df, pairs)

    out_dir = OUTPUTS / f"campaign_{meta.campaign_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    matched_ids = [p[0] for p in pairs] + [p[1] for p in pairs]
    matched_df = df.loc[df["household_key"].isin(matched_ids)].copy()
    pair_id_map = {}
    for t_id, c_id, _ in pairs:
        pair_id_map[t_id] = f"pair_{t_id}_{c_id}"
        pair_id_map[c_id] = f"pair_{t_id}_{c_id}"
    matched_df["pair_id"] = matched_df["household_key"].map(pair_id_map)
    matched_path = out_dir / "matched_data.csv"
    matched_df.to_csv(matched_path, index=False, encoding="utf-8-sig")
    _log("저장", f"matched_data.csv → {matched_path}")

    status = "ok" if balance_ok else "residual_imbalance"
    reason = "" if balance_ok else (
        f"매칭 후 최대 |SMD|={max_abs_smd:.3f}가 기준({MAX_ALLOWED_SMD}) 초과 — "
        "효과 추정치는 참고용으로만 사용하고 확정하지 않음"
    )
    return result(
        status, reason,
        campaign_id=meta.campaign_id, pre_days=meta.pre_days,
        start_day=meta.start_day, end_day=meta.end_day, overlap_ids=meta.overlap_ids,
        n_treatment=meta.n_treatment, n_control=meta.n_control,
        common_support=[round(lo, 4), round(hi, 4)],
        n_common_support_treatment=int(((df["group"] == "처치") & df["in_support"]).sum()),
        n_common_support_control=int(((df["group"] == "대조") & df["in_support"]).sum()),
        caliper_multiplier=CALIPER_MULTIPLIER, caliper_logit=round(caliper, 4),
        n_matched_pairs=len(pairs), match_rate=round(len(pairs) / meta.n_treatment, 4),
        balance_smd_before=smd_before_df.to_dict(orient="records"), max_abs_smd_before=round(max_abs_smd_before, 4),
        balance_smd_after=smd_df.to_dict(orient="records"), max_abs_smd=round(max_abs_smd, 4),
        outcomes=outcomes,
    )


if __name__ == "__main__":
    import argparse

    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="성향점수·매칭·효과 추정 파이프라인")
    parser.add_argument("--campaign_id", type=int, required=True)
    args = parser.parse_args()

    data_path = OUTPUTS / f"campaign_{args.campaign_id}" / "analysis_data.csv"
    df = pd.read_csv(data_path)
    meta = CampaignMeta(
        campaign_id=args.campaign_id, description="", start_day=0, end_day=0, pre_days=0,
        pre_start_day=0, pre_end_day=0, n_treatment=int((df["group"] == "처치").sum()),
        n_control=int((df["group"] == "대조").sum()),
    )
    res = estimate_effect(df, meta)
    print(res)
