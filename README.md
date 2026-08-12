# 쿠폰 캠페인 효과와 수익성 분석

이 폴더는 Kaggle의 dunnhumby The Complete Journey 데이터를 이용해 캠페인별 비교집단을 구성하고, 성향점수 매칭으로 캠페인 효과를 추정한 뒤 쿠폰 후보안의 수익성을 비교하는 프로젝트 작업 공간이다.

## 핵심 결과 (요약)

30개 캠페인 전체에 파이프라인을 돌린 결과, 표본·공통지지영역·매칭 후 균형 기준을 모두 통과한 캠페인은 3개(8, 18, 30)뿐이었고, **이 3개 캠페인 어디에서도 매칭 후 통계적으로 유의한 양의 효과를 찾지 못했다.** 보정 전 단순 비교에서 보이던 "효과"는 대부분 발행 전 구매행동 차이(선택편향)로 설명되며, 매칭으로 이를 통제하면 대부분 0에 가깝게 줄어들거나(캠페인 8은 부호까지 반전) 신뢰구간이 0을 포함했다.

유일하게 유의했던 사례(캠페인 18, 발행 전 관찰 기간을 30일로 짧게 잡았을 때)도 (1) 관찰 기간을 60일 이상으로 늘리면 유의성이 사라지고, (2) Rosenbaum 민감도 분석상 아주 작은 숨겨진 교란(Gamma=1.2)만으로도 결론이 뒤집혀, 신뢰할 만한 근거로 보기 어렵다.

**해석**: "이 캠페인들은 효과가 없다"는 확정이 아니라, "이 데이터와 방법으로는 신뢰할 만한 양의 인과효과를 찾지 못했다"는 뜻이다. 상세 검증 과정과 수치는 [`outputs/validation_report.md`](outputs/validation_report.md), 캠페인별 상세 결과는 `outputs/campaign_{id}/report.html`을 참고한다.

## 최종 결과물

- [x] 캠페인 번호와 발행 전 관찰 기간(pre_days)을 입력받는 재사용 가능한 분석 파이프라인(`pipeline/`) — 표본 부족·공통지지영역 부족·잔여 불균형 시 효과를 확정하지 않고 상태·사유를 반환
- [x] 비교집단 품질(성향점수 분포·공통지지영역·매칭률·매칭 전후 SMD)을 효과보다 먼저 보여주는 Streamlit 앱(`app/streamlit_app.py`)
- [x] 손익분기점과 사용자가 입력한 쿠폰 후보안을 비교하는 기능(`app/profitability.py`) — 추천은 입력한 후보군 안에서만 판단
- [x] 30개 캠페인 일괄 비교, pre_days 민감도 분석, Rosenbaum 민감도 분석, pytest 회귀 테스트

## 작업 방식

파이프라인·앱·검증까지 완료된 상태이며, 이후 이 프로젝트를 이어서 쓰거나 확장할 때는 아래 순서를 따른다.

1. **새 캠페인·조건으로 분석**: `python pipeline/run_pipeline.py --campaign_id {id} --pre_days {N}` 또는 앱에서 선택해 실행한다. 같은 조건(`campaign_id`, `pre_days`)으로 이미 실행한 결과가 있으면 `outputs/campaign_{id}/results.json`을 그대로 재사용하며, 처리 규칙(caliper, 변수 목록, 품질 기준 등)을 바꿀 때만 재실행한다(`CLAUDE.md` 규칙).
2. **결과 신뢰 여부는 status로 판단**: `results.json`의 `status`가 `"ok"`가 아니면(표본 부족·공통지지영역 부족·잔여 불균형) 효과 수치를 그대로 믿지 않는다 — `reason`을 먼저 확인한다.
3. **처리 규칙을 바꾸는 코드 수정 후에는 반드시 `python -m pytest`를 실행**해 캠페인 18 기준값(`tests/test_pipeline_integration.py`)이 그대로 재현되는지 확인한 뒤 커밋한다.
4. **원본 데이터(`data/raw/`)는 수정하지 않는다.** 캠페인 번호·기간·대상 상품은 코드에 고정하지 않고 항상 원본 테이블에서 조회한다.
5. 캠페인별 최종 산출물은 `outputs/campaign_{id}/`의 4개 파일(`analysis_data.csv`, `matched_data.csv`, `results.json`, `report.html`)로 남기고, 다른 캠페인의 결과를 덮어쓰지 않는다.
6. 전체 프로젝트 결론과 한계는 `outputs/validation_report.md`를 최신 상태로 유지한다 — 새 캠페인을 분석하거나 처리 규칙을 바꾸면 관련 섹션을 갱신한다.

## 폴더 구조

- `data/raw/`: 분석에 필요한 Kaggle 원본 CSV 7개
- `data/processed/`: 검증된 가구 단위 분석표
- `analysis/`: 데이터 점검 문서와 탐색 코드, 캠페인 18 단계별 검증 스크립트, 30개 캠페인 일괄 실행·pre_days/Rosenbaum 민감도 분석
- `pipeline/`: 캠페인 입력형 데이터 준비(`prepare_data.py`)·효과 추정(`estimate_effect.py`)·민감도 분석(`sensitivity.py`)·실행기(`run_pipeline.py`)·보고서 생성(`generate_report.py`)
- `outputs/`: 캠페인별 결과(`campaign_{id}/analysis_data.csv`, `matched_data.csv`, `results.json`, `report.html`), 캠페인 일괄 비교표, `validation_report.md`(전체 검증 요약)
- `app/`: Streamlit 앱(`streamlit_app.py`)과 수익성 계산 코드(`profitability.py`)
- `tests/`: pytest 회귀·단위 테스트 (`pytest.ini`로 설정)
- `CLAUDE.md`: Claude Code가 항상 따라야 할 작업 규칙
- `DATA_DICTIONARY.md`: 원본 테이블과 핵심 변수 정의
- `PROJECT_CHECKLIST.md`: 단계별 검증 기준
- `SOURCE_AND_LICENSE.md`: 데이터 출처와 포함 범위

## 실행 환경

```bash
python -m pip install -r requirements.txt

# 앱 실행 (캠페인·pre_days 선택 → 진단 → 품질 통과 시 효과·수익성 비교)
streamlit run app/streamlit_app.py

# 파이프라인 단독 실행 (예: 캠페인 18, 발행 전 90일)
python pipeline/run_pipeline.py --campaign_id 18 --pre_days 90
python pipeline/generate_report.py --campaign_id 18

# 회귀 테스트 (빠른 단위 테스트만: -m "not integration")
python -m pytest
```
