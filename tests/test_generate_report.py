"""pipeline/generate_report.py 통합 테스트 (실제 results.json 필요)."""

import pytest

from generate_report import generate_report

pytestmark = pytest.mark.integration


def test_generate_report_produces_html_with_key_sections():
    from run_pipeline import run

    run(campaign_id=18, pre_days=90)  # results.json 최신화
    path = generate_report(18)
    assert path.exists()

    html = path.read_text(encoding="utf-8")
    assert "캠페인 18 분석 보고서" in html
    assert "target_sales" in html
    assert "매칭 전후 균형" in html
    assert "<table>" in html


def test_generate_report_missing_results_raises():
    with pytest.raises(FileNotFoundError):
        generate_report(campaign_id=999999)
