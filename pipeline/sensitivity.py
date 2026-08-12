"""매칭된 쌍에 대한 Rosenbaum 민감도 분석(관찰되지 않은 교란에 대한 강건성 점검).

매칭은 발행 전에 관찰된 특성(7개 변수)만 균형을 맞춘다. 관찰되지 않은 교란요인
(예: 오프라인 접촉, 개인 성향)이 남아 있을 수 있는데, Rosenbaum bounds는 "숨겨진
교란이 처치 배정 확률을 최대 Gamma배까지 왜곡시킬 수 있다고 가정할 때, 결론이
여전히 유지되는가"를 정량화한다.

방법: 매칭쌍 차이(d_i = 처치-대조)에 대한 Wilcoxon 부호순위검정을 뒤틀어, 각
Gamma에 대해 p-value의 하한/상한을 계산한다(Gamma=1이면 숨겨진 교란 없음 = 일반
Wilcoxon 검정과 동일). p_upper가 0.05를 넘는 최소 Gamma를 "붕괴점(breakdown Gamma)"
이라 부른다 — 이 값이 클수록 결론이 숨겨진 교란에 강건하다는 뜻이다.

참고: Rosenbaum, P.R. (2002), *Observational Studies*, 2nd ed., Ch.4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

DEFAULT_GAMMAS = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0]


def rosenbaum_bounds(diffs: np.ndarray, gammas: list[float] = DEFAULT_GAMMAS) -> pd.DataFrame:
    """매칭쌍 차이(diffs = 처치값 - 대조값) 배열에 대한 Gamma별 p-value 상/하한.

    차이가 0인 쌍(동률)은 부호순위검정 관행대로 제외한다.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        raise ValueError("모든 쌍의 차이가 0이라 Rosenbaum bounds를 계산할 수 없음")

    abs_d = np.abs(d)
    ranks = pd.Series(abs_d).rank(method="average").to_numpy()
    signs = np.sign(d)
    T_plus = ranks[signs > 0].sum()
    sum_ranks_sq = (ranks**2).sum()

    rows = []
    for gamma in gammas:
        p_plus = gamma / (1 + gamma)   # 숨겨진 편향이 "양의 차이"를 실제보다 부풀릴 수 있는 최대 확률
        p_minus = 1 / (1 + gamma)      # 최소 확률

        # p_upper (p-value 상한, 유의성에 가장 불리한 최악의 경우): 귀무가설 하에서도
        # 양의 부호가 나올 확률을 p_plus(더 큼)로 가정 -> 기대순위합이 커져 관측치가
        # 덜 극단적으로 보임 -> p-value가 커짐(가장 보수적인 해석).
        mu_for_upper = p_plus * ranks.sum()
        var_for_upper = p_plus * (1 - p_plus) * sum_ranks_sq
        z_for_upper = (T_plus - mu_for_upper) / np.sqrt(var_for_upper)
        p_upper = 1 - norm.cdf(z_for_upper)

        # p_lower (p-value 하한, 가장 유리한 경우): p_minus(더 작음)를 가정 -> 기대순위합이
        # 작아져 관측치가 더 극단적으로 보임 -> p-value가 작아짐.
        mu_for_lower = p_minus * ranks.sum()
        var_for_lower = p_minus * (1 - p_minus) * sum_ranks_sq
        z_for_lower = (T_plus - mu_for_lower) / np.sqrt(var_for_lower)
        p_lower = 1 - norm.cdf(z_for_lower)

        rows.append({"Gamma": gamma, "p_lower": p_lower, "p_upper": p_upper})

    return pd.DataFrame(rows)


def breakdown_gamma(diffs: np.ndarray, alpha: float = 0.05, gammas: list[float] = DEFAULT_GAMMAS) -> float | None:
    """p_upper가 alpha를 처음 넘는 Gamma(붕괴점)를 반환한다. 처음부터 넘으면 1.0,
    끝까지 안 넘으면 None(강건함)을 반환한다."""
    bounds = rosenbaum_bounds(diffs, gammas)
    over = bounds.loc[bounds["p_upper"] > alpha]
    if over.empty:
        return None
    return float(over["Gamma"].iloc[0])
