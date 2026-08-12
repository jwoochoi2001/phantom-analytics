"""pipeline/sensitivity.py (Rosenbaum bounds) 단위 테스트."""

import numpy as np
import pytest
from scipy.stats import wilcoxon

from sensitivity import breakdown_gamma, rosenbaum_bounds


def test_gamma1_matches_standard_wilcoxon():
    rng = np.random.default_rng(0)
    diffs = rng.normal(loc=2.0, scale=3.0, size=100)
    bounds = rosenbaum_bounds(diffs, gammas=[1.0])
    w = wilcoxon(diffs, alternative="greater")
    row = bounds.iloc[0]
    assert row["p_lower"] == pytest.approx(w.pvalue, abs=1e-9)
    assert row["p_upper"] == pytest.approx(w.pvalue, abs=1e-9)


def test_p_upper_monotonically_increases_with_gamma():
    rng = np.random.default_rng(1)
    diffs = rng.normal(loc=1.0, scale=2.0, size=60)
    bounds = rosenbaum_bounds(diffs, gammas=[1.0, 1.5, 2.0, 3.0, 5.0])
    p_upper = bounds["p_upper"].to_numpy()
    assert np.all(np.diff(p_upper) >= -1e-12)


def test_strong_effect_is_more_robust_than_weak_effect():
    rng = np.random.default_rng(2)
    strong = rng.normal(loc=5.0, scale=2.0, size=100)
    weak = rng.normal(loc=0.05, scale=2.0, size=100)
    bg_strong = breakdown_gamma(strong)
    bg_weak = breakdown_gamma(weak)
    # 강한 효과는 끝까지 안 무너지거나(None), 무너지더라도 약한 효과보다 큰 Gamma에서 무너져야 함
    assert bg_weak is not None
    assert bg_strong is None or bg_strong > bg_weak


def test_zero_diff_pairs_are_excluded_not_crash():
    diffs = np.array([1.0, -1.0, 0.0, 0.0, 2.0, -0.5])
    bounds = rosenbaum_bounds(diffs)
    assert not bounds.empty


def test_all_zero_diffs_raises_clear_error():
    with pytest.raises(ValueError):
        rosenbaum_bounds(np.zeros(10))
