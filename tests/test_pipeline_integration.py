"""실제 데이터 기반 통합 회귀 테스트.

캠페인 18(pre_days=586)을 파이프라인으로 돌려 analysis/campaign18_*.py에서
검증한 기준값을 그대로 재현하는지 확인한다(pipeline/validate_campaign18.py의
공식 pytest 버전). 실제 CSV를 읽으므로 다른 테스트보다 느리다.

모든 테스트는 임시 디렉터리(output_dir)에 결과를 저장한다 — 실제
outputs/campaign_{id}/ 캐시(앱·리포트가 참조하는 "정식" 결과)를 테스트가
덮어쓰지 않게 하기 위함이다.
"""

import pytest

from run_pipeline import run

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def campaign18_result(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("campaign18_outputs")
    return run(campaign_id=18, pre_days=586, output_dir=out_dir)


def test_campaign18_group_sizes(campaign18_result):
    assert campaign18_result["n_treatment"] == 510
    assert campaign18_result["n_control"] == 335


def test_campaign18_common_support(campaign18_result):
    assert campaign18_result["n_common_support_treatment"] == 507
    assert campaign18_result["n_common_support_control"] == 313


def test_campaign18_matching(campaign18_result):
    assert campaign18_result["n_matched_pairs"] == 232


def test_campaign18_balance_passes(campaign18_result):
    assert campaign18_result["status"] == "ok"
    assert campaign18_result["max_abs_smd"] == pytest.approx(0.08, abs=0.001)


def test_campaign18_period_and_overlap(campaign18_result):
    assert campaign18_result["start_day"] == 587
    assert campaign18_result["end_day"] == 642
    assert campaign18_result["overlap_ids"] == [14, 15, 16, 17, 19, 20, 21, 22]


def test_small_sample_campaign_returns_insufficient_sample_status(tmp_path):
    # 캠페인 3: 순수 처치가구 0명으로 이미 확인된 캠페인 — 예외 없이 status 반환해야 함
    res = run(campaign_id=3, pre_days=90, output_dir=tmp_path)
    assert res["status"] == "insufficient_sample"
    assert res["n_treatment"] == 0


def test_different_campaign_does_not_leak_campaign18_period(tmp_path):
    res = run(campaign_id=13, pre_days=90, output_dir=tmp_path)
    assert res["start_day"] == 504
    assert res["end_day"] == 551
    assert res["overlap_ids"] == [11, 12, 14, 15]
    assert res["start_day"] != 587  # 캠페인 18 값이 섞여 들어오지 않았는지 확인
