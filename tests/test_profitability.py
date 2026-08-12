"""app/profitability.py 단위 테스트 (합성 데이터, 수작업 계산값과 대조)."""

import math

import pandas as pd
import pytest

from profitability import compute_candidates, empty_candidates, recommend, validate_candidates


def test_compute_candidates_matches_hand_calculation():
    df = pd.DataFrame(
        [{"후보명": "A", "할인액": 5.0, "예상사용률": 0.15, "발행수": 1000, "운영비": 500.0}]
    )
    out = compute_candidates(df, ate_per_household=100.0)
    row = out.iloc[0]

    # 사용건수=150, 매출=150*100=15000, 비용=150*5+500=1250, 이익=13750, ROI=11.0
    assert row["예상사용건수"] == 150
    assert row["예상증분매출"] == 15000
    assert row["총비용"] == 1250
    assert row["증분이익"] == 13750
    assert row["ROI"] == pytest.approx(11.0)
    assert row["손익분기충족"]


def test_zero_cost_roi_is_nan_not_crash():
    df = pd.DataFrame(
        [{"후보명": "Z", "할인액": 0.0, "예상사용률": 0.0, "발행수": 1000, "운영비": 0.0}]
    )
    out = compute_candidates(df, ate_per_household=100.0)
    assert out.iloc[0]["총비용"] == 0
    assert math.isnan(out.iloc[0]["ROI"])
    assert out.iloc[0]["손익분기충족"]  # 이익 0 >= 0


def test_negative_ate_all_candidates_lose_money():
    df = empty_candidates()
    out = compute_candidates(df, ate_per_household=-10.0)
    assert (out["증분이익"] < 0).all()
    assert not out["손익분기충족"].any()


def test_recommend_picks_highest_profit_among_breakeven_candidates():
    df = pd.DataFrame(
        [
            {"후보명": "A", "할인액": 5.0, "예상사용률": 0.15, "발행수": 1000, "운영비": 500.0},
            {"후보명": "B", "할인액": 10.0, "예상사용률": 0.25, "발행수": 1000, "운영비": 500.0},
        ]
    )
    out = compute_candidates(df, ate_per_household=100.0)
    name, reason = recommend(out)
    # A: 사용건수150, 이익=150*100-(150*5+500)=13750
    # B: 사용건수250, 이익=250*100-(250*10+500)=22000  -> B가 더 큼
    assert name == "B"
    assert "입력한 후보군 내에서 추천" in reason


def test_recommend_when_no_candidate_breaks_even():
    df = pd.DataFrame(
        [
            {"후보명": "적자1", "할인액": 100.0, "예상사용률": 0.5, "발행수": 1000, "운영비": 10000.0},
            {"후보명": "적자2", "할인액": 200.0, "예상사용률": 0.8, "발행수": 1000, "운영비": 20000.0},
        ]
    )
    out = compute_candidates(df, ate_per_household=5.0)
    assert not out["손익분기충족"].any()
    name, reason = recommend(out)
    assert name == "적자1"  # 손실이 더 적은 쪽
    assert "충족한 후보가 없습니다" in reason


def test_recommend_empty_dataframe():
    name, reason = recommend(pd.DataFrame(columns=["후보명", "증분이익", "손익분기충족"]))
    assert name is None


@pytest.mark.parametrize(
    "bad_row,expected_substr",
    [
        ({"후보명": "A", "할인액": -1.0, "예상사용률": 0.1, "발행수": 100, "운영비": 0.0}, "할인액이 음수"),
        ({"후보명": "A", "할인액": 1.0, "예상사용률": 1.5, "발행수": 100, "운영비": 0.0}, "예상사용률은 0~1"),
        ({"후보명": "A", "할인액": 1.0, "예상사용률": 0.1, "발행수": 0, "운영비": 0.0}, "발행수가 0 이하"),
        ({"후보명": "A", "할인액": 1.0, "예상사용률": 0.1, "발행수": 100, "운영비": -5.0}, "운영비가 음수"),
    ],
)
def test_validate_candidates_catches_bad_inputs(bad_row, expected_substr):
    df = pd.DataFrame([bad_row])
    warnings = validate_candidates(df)
    assert any(expected_substr in w for w in warnings)


def test_validate_candidates_detects_duplicate_names():
    df = pd.DataFrame(
        [
            {"후보명": "A", "할인액": 1.0, "예상사용률": 0.1, "발행수": 100, "운영비": 0.0},
            {"후보명": "A", "할인액": 2.0, "예상사용률": 0.2, "발행수": 200, "운영비": 0.0},
        ]
    )
    warnings = validate_candidates(df)
    assert any("중복" in w for w in warnings)
