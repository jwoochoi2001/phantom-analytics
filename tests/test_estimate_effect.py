"""pipeline/estimate_effect.py 핵심 함수 단위 테스트 (합성 데이터)."""

import numpy as np
import pandas as pd
import pytest

from estimate_effect import (
    MIN_COMMON_SUPPORT_N,
    check_common_support,
    compute_balance,
    compute_balance_before,
    nn_match,
    smd_continuous,
)
from prepare_data import FINAL_PROPENSITY_VARS


def test_smd_continuous_known_value():
    t = np.array([10.0, 12.0, 14.0])  # mean 12, var 4
    c = np.array([8.0, 10.0, 12.0])  # mean 10, var 4
    smd = smd_continuous(t, c)
    # pooled_sd = sqrt((4+4)/2) = 2, diff = 2 -> smd = 1.0
    assert smd == pytest.approx(1.0)


def test_smd_continuous_zero_variance_returns_zero():
    t = np.array([5.0, 5.0])
    c = np.array([5.0, 5.0])
    assert smd_continuous(t, c) == 0.0


def test_nn_match_no_duplicate_control_and_respects_caliper():
    rng = np.random.default_rng(0)
    treat = pd.DataFrame(
        {"household_key": range(100, 150), "logit_p": rng.normal(0, 1, 50)}
    ).reset_index(drop=True)
    ctrl = pd.DataFrame(
        {"household_key": range(200, 230), "logit_p": rng.normal(0, 1, 30)}
    ).reset_index(drop=True)

    caliper = 0.3
    pairs = nn_match(treat, ctrl, caliper, seed=42)

    matched_ctrl_ids = [p[1] for p in pairs]
    assert len(matched_ctrl_ids) == len(set(matched_ctrl_ids)), "대조가구가 중복 매칭됨"
    assert len(pairs) <= len(ctrl)  # 대조 풀 크기를 넘을 수 없음

    for t_id, c_id, dist in pairs:
        assert dist <= caliper

    matched_treat_ids = [p[0] for p in pairs]
    assert len(matched_treat_ids) == len(set(matched_treat_ids)), "처치가구가 중복 매칭됨"


def test_nn_match_is_deterministic_given_seed():
    rng = np.random.default_rng(1)
    treat = pd.DataFrame({"household_key": range(10), "logit_p": rng.normal(0, 1, 10)})
    ctrl = pd.DataFrame({"household_key": range(10, 20), "logit_p": rng.normal(0, 1, 10)})

    pairs1 = nn_match(treat, ctrl, caliper=0.5, seed=7)
    pairs2 = nn_match(treat, ctrl, caliper=0.5, seed=7)
    assert pairs1 == pairs2


def test_nn_match_empty_control_pool_returns_no_pairs():
    treat = pd.DataFrame({"household_key": [1, 2, 3], "logit_p": [0.1, 0.2, 0.3]})
    ctrl = pd.DataFrame({"household_key": [], "logit_p": []})
    pairs = nn_match(treat, ctrl, caliper=1.0, seed=1)
    assert pairs == []


def _make_group_df(n_t: int, n_c: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_t + n_c
    df = pd.DataFrame({"household_key": range(n)})
    df["group"] = ["처치"] * n_t + ["대조"] * n_c
    df["treatment"] = (df["group"] == "처치").astype(int)
    for v in FINAL_PROPENSITY_VARS:
        df[v] = rng.normal(0, 1, n)
    return df


def test_check_common_support_flags_insufficient_when_below_threshold():
    df = _make_group_df(n_t=5, n_c=5)  # MIN_COMMON_SUPPORT_N(30)보다 훨씬 작음
    df["p_score"] = np.random.default_rng(0).uniform(0, 1, len(df))
    _, lo, hi, gate = check_common_support(df)
    assert gate is not None
    assert gate["status"] == "insufficient_overlap"
    assert gate["n_treatment_in_support"] < MIN_COMMON_SUPPORT_N


def test_check_common_support_passes_when_enough_overlap():
    df = _make_group_df(n_t=50, n_c=50)
    # 두 집단이 동일한 min/max(0.3~0.7)를 공유하도록 결정론적으로 구성 —
    # 그래야 공통지지영역이 [0.3, 0.7] 전체가 되어 모든 행이 영역 안에 들어온다.
    df.loc[df["group"] == "처치", "p_score"] = np.linspace(0.3, 0.7, 50)
    df.loc[df["group"] == "대조", "p_score"] = np.linspace(0.3, 0.7, 50)
    out_df, lo, hi, gate = check_common_support(df)
    assert gate is None
    assert lo == pytest.approx(0.3)
    assert hi == pytest.approx(0.7)
    assert out_df["in_support"].sum() == len(df)


def test_compute_balance_zero_after_matching_identical_pairs():
    df = _make_group_df(n_t=5, n_c=5, seed=3)
    # 매칭 쌍: household 0~4(처치) <-> 5~9(대조)를 동일한 값으로 강제
    for v in FINAL_PROPENSITY_VARS:
        df.loc[5:9, v] = df.loc[0:4, v].to_numpy()
    pairs = [(i, i + 5, 0.0) for i in range(5)]
    smd_df, max_abs = compute_balance(df, pairs)
    assert max_abs == pytest.approx(0.0)


def test_compute_balance_before_uses_common_support_population():
    df = _make_group_df(n_t=20, n_c=20, seed=5)
    df["in_support"] = True
    smd_df, max_abs = compute_balance_before(df)
    assert set(smd_df["변수"]) == set(FINAL_PROPENSITY_VARS)
    assert max_abs >= 0
