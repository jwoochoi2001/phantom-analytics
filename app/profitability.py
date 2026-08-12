"""쿠폰 후보안 수익성 비교.

같은 캠페인의 매칭 후 효과(target_sales 증분, 가구당 — 캠페인 분석에서 얻은 모델
추정치)를 사용해, 사용자가 입력한 후보 쿠폰(할인액·예상 사용률·발행 수·운영비 —
사용자 입력 수익성 가정)별로 예상 증분매출·총비용·증분이익·ROI·손익분기 충족 여부를
계산한다.

CLAUDE.md 규칙:
- 관찰된 사실/모델 추정치(캠페인 효과)와 사용자가 입력한 수익성 가정(할인액 등)을 구분한다.
- 후보 시나리오 중 가장 높은 결과를 전역 최적 쿠폰이라고 표현하지 않는다 —
  추천은 사용자가 입력한 후보군 안에서만 이루어진다.

계산 규칙(고정):
    예상 사용 건수 = 발행 수 x 예상 사용률
    예상 증분매출 = 예상 사용 건수 x ATE(target_sales, 매칭후)
    총 비용        = 예상 사용 건수 x 할인액 + 운영비
    증분이익       = 예상 증분매출 - 총비용
    ROI            = 증분이익 / 총비용  (총비용이 0이면 정의하지 않음)
    손익분기 충족   = 증분이익 >= 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CANDIDATE_COLS = ["후보명", "할인액", "예상사용률", "발행수", "운영비"]


def empty_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"후보명": "후보 A", "할인액": 5.0, "예상사용률": 0.15, "발행수": 1000, "운영비": 500.0},
            {"후보명": "후보 B", "할인액": 10.0, "예상사용률": 0.25, "발행수": 1000, "운영비": 500.0},
        ]
    )


def losing_candidates_example() -> pd.DataFrame:
    """손익분기 미충족 UI 동작을 확인하기 위한 예시(할인액·운영비를 지나치게 크게 잡음)."""
    return pd.DataFrame(
        [
            {"후보명": "적자 후보 X", "할인액": 100.0, "예상사용률": 0.5, "발행수": 1000, "운영비": 10000.0},
            {"후보명": "적자 후보 Y", "할인액": 200.0, "예상사용률": 0.8, "발행수": 1000, "운영비": 20000.0},
        ]
    )


def validate_candidates(df: pd.DataFrame) -> list[str]:
    """입력 오류를 사람이 읽을 수 있는 문구 리스트로 반환한다 (표본 부족·0비용·음수 등)."""
    warnings: list[str] = []
    if df.empty:
        warnings.append("입력된 후보가 없습니다.")
        return warnings

    for col in CANDIDATE_COLS:
        if col not in df.columns:
            warnings.append(f"필수 열 누락: {col}")
    if warnings:
        return warnings

    if df["후보명"].isna().any() or (df["후보명"].astype(str).str.strip() == "").any():
        warnings.append("후보명이 비어 있는 행이 있습니다.")
    if df["후보명"].duplicated().any():
        dup = df.loc[df["후보명"].duplicated(), "후보명"].tolist()
        warnings.append(f"후보명이 중복되었습니다: {dup}")
    if (df["할인액"] < 0).any():
        warnings.append("할인액이 음수인 후보가 있습니다.")
    if ((df["예상사용률"] < 0) | (df["예상사용률"] > 1)).any():
        warnings.append("예상사용률은 0~1(0%~100%) 범위여야 합니다.")
    if (df["발행수"] <= 0).any():
        warnings.append("발행수가 0 이하인 후보가 있습니다.")
    if (df["운영비"] < 0).any():
        warnings.append("운영비가 음수인 후보가 있습니다.")
    return warnings


def compute_candidates(df: pd.DataFrame, ate_per_household: float) -> pd.DataFrame:
    """후보별 증분매출·비용·이익·ROI·손익분기 충족 여부를 계산한다.

    ate_per_household: 선택 캠페인의 매칭 후 target_sales 차이(가구당, $) — 모델 추정치.
    """
    out = df.copy()
    out["예상사용건수"] = out["발행수"] * out["예상사용률"]
    out["예상증분매출"] = out["예상사용건수"] * ate_per_household
    out["총비용"] = out["예상사용건수"] * out["할인액"] + out["운영비"]
    out["증분이익"] = out["예상증분매출"] - out["총비용"]
    out["ROI"] = np.where(out["총비용"] > 0, out["증분이익"] / out["총비용"], np.nan)
    out["손익분기충족"] = out["증분이익"] >= 0
    return out


def recommend(df: pd.DataFrame) -> tuple[str | None, str]:
    """입력한 후보군 안에서만 추천한다. (candidate_name, 근거 설명) 반환."""
    if df.empty:
        return None, "후보가 없습니다."

    passing = df.loc[df["손익분기충족"]]
    if not passing.empty:
        best_idx = passing["증분이익"].idxmax()
        name = df.loc[best_idx, "후보명"]
        return name, (
            f"손익분기를 충족한 후보 {len(passing)}개 중 증분이익이 가장 큰 '{name}'을 "
            "입력한 후보군 내에서 추천합니다. (전역 최적 쿠폰을 의미하지 않습니다.)"
        )

    best_idx = df["증분이익"].idxmax()
    name = df.loc[best_idx, "후보명"]
    return name, (
        f"입력한 후보 중 손익분기를 충족한 후보가 없습니다. 그중 증분손실이 가장 적은 "
        f"'{name}'을 참고용으로 표시하며, 실제 집행은 권장하지 않습니다."
    )
