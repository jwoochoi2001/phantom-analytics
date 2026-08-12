"""app/streamlit_app.py의 캐시 전용(원본 데이터 없는 배포 환경) 폴백 동작 테스트.

실제로 data/raw/를 지우지 않고 RAW_DATA_AVAILABLE 플래그만 monkeypatch해서 검증한다.
"""

import streamlit_app as app


def test_cache_only_campaign_list_has_entries(monkeypatch):
    monkeypatch.setattr(app, "RAW_DATA_AVAILABLE", False)
    df = app._load_campaign_list_from_cache()
    assert not df.empty
    assert {"CAMPAIGN", "DESCRIPTION", "START_DAY", "END_DAY"} <= set(df.columns)


def test_cache_hit_works_without_raw_data(monkeypatch):
    monkeypatch.setattr(app, "RAW_DATA_AVAILABLE", False)
    res, from_cache = app.get_or_run(18, 90)
    assert from_cache is True
    assert res["campaign_id"] == 18
    assert res["pre_days"] == 90


def test_cache_miss_returns_no_raw_data_status_instead_of_crashing(monkeypatch):
    monkeypatch.setattr(app, "RAW_DATA_AVAILABLE", False)
    res, from_cache = app.get_or_run(18, 999999)  # 캐시에 없는 조합
    assert from_cache is False
    assert res["status"] == "no_raw_data"
    assert "reason" in res
