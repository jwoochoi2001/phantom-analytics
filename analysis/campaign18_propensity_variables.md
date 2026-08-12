# 캠페인 18 성향점수 입력변수 확정 (최종)

- 대상: 캠페인 18 (TypeA, DAY 587~642)
- 분석표: `outputs/campaign_18/analysis_data.csv` (처치 510 / 대조 335 = 845가구)
- 대조군은 `campaign_table.csv` 상 캠페인 수신 이력이 있으나 18번·겹치는 8개 캠페인은 받지 않은 335가구로 고정한다(캠페인을 한 번도 받지 않은 916가구는 제외 — 앞 단계에서 사용자가 직접 확정한 기준).
- 참고: 사용자가 제공한 `ps_variables.md`는 대조군을 1,251명(위 916명 포함)으로 다르게 잡았기 때문에 이 문서와 모집단이 다르다. 아래는 그 파일의 변수 엔지니어링 방식을 저희 845가구 기준으로 재계산·재검증한 결과다.

## 1. 변수 엔지니어링

1. **pre_recency_capped** = `587 - 마지막 구매일`, 상한 365일 절단, 발행 전 구매이력이 없는 가구는 586(관찰 가능한 최댓값)로 대체. 현재 845가구 모두 발행 전 구매가 있어 결측 0건.
2. **log_pre_baskets**, **log_pre_sales** = `log1p()` 변환. 오른쪽 꼬리가 긴 금액·빈도 변수의 왜도를 완화한다.
3. **pre_target_share** = `pre_target_sales / pre_sales` (분모 0이면 0). log_pre_sales와의 상관 -0.207로, 대상 상품 관련성을 전체 구매 규모와 어느 정도 독립적으로 표현한다.
4. **pre_coupon_user** = `pre_coupon_redemptions > 0` (0/1). 원시 횟수는 0이 대다수인 극단 분포라 이진화가 더 안정적이다.
5. **pre_campaign_count_c** = 캠페인 시작일(587) 기준 **완전히 종료된**(`END_DAY < 587`) 과거 캠페인 수신 횟수, 상한 6. 완전종료 캠페인: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 26, 27, 28, 29, 30]. 시작은 587 이전이지만 아직 안 끝난(겹치는) 캠페인 [14, 15, 16, 17]은 제외해 사후 정보 유입을 원천 차단한다. 저희 표본은 이 겹치는 캠페인들을 애초에 받지 않으므로 `START_DAY<587` 기준으로 계산했던 이전 값과 결과는 동일하지만, 기준 자체는 이 방식이 더 엄격하고 일반적으로 옳다.
6. **has_demographic** = `household_key`가 `hh_demographic.csv`에 존재하는지 (0/1). 인구통계 값 자체(연령·소득 등)는 처치 47.5% vs 대조 21.5%로 확보율 차이가 커서 회귀 표본이 크게 줄어드는 문제가 있었다. 값 대신 **결측 여부만** 지시자로 넣어 이 정보를 표본 손실 없이 활용한다.

## 2. 검토 후 제외한 변수 (재검증)

- **pre_sales_per_basket(객단가)**: 처치 28.60 vs 대조 29.64, 단변량 AUC 0.515 → 이미 균형에 가까워 변별력이 낮다. 제외.
- **pre_target_any(대상 상품 구매 경험 유무)**: 처치 100.0% vs 대조 100.0% → 두 집단 모두 사실상 100%라 변별력이 없다. 제외.
- **pre_quantity**: 이전 검토에서 확인한 특정 상품(무게 단위 판매 추정)의 이상치 문제로 제외 (변경 없음).
- **pre_target_quantity, pre_target_sales**: `pre_target_share`로 대체되어 별도 입력 불필요.

## 3. 최종 확정 변수 (7개)

| 변수 | 단변량 AUC | VIF |
|---|---|---|
| pre_recency_capped | 0.618 | 1.12 |
| log_pre_baskets | 0.744 | 1.7 |
| log_pre_sales | 0.725 | 1.74 |
| pre_target_share | 0.548 | 1.1 |
| pre_coupon_user | 0.539 | 1.11 |
| pre_campaign_count_c | 0.565 | 1.4 |
| has_demographic | 0.63 | 1.25 |

**다변량 로지스틱 회귀(표준화, 7변수) AUC = 0.785**


## 4. 변수 간 상관행렬 (Pearson)

```
                      pre_recency_capped  log_pre_baskets  log_pre_sales  pre_target_share  pre_coupon_user  pre_campaign_count_c  has_demographic
pre_recency_capped                  1.00            -0.27          -0.21              0.02            -0.09                 -0.07            -0.23
log_pre_baskets                    -0.27             1.00           0.55             -0.25             0.14                  0.44             0.33
log_pre_sales                      -0.21             0.55           1.00             -0.21             0.26                  0.45             0.39
pre_target_share                    0.02            -0.25          -0.21              1.00            -0.00                 -0.02            -0.08
pre_coupon_user                    -0.09             0.14           0.26             -0.00             1.00                  0.22             0.22
pre_campaign_count_c               -0.07             0.44           0.45             -0.02             0.22                  1.00             0.19
has_demographic                    -0.23             0.33           0.39             -0.08             0.22                  0.19             1.00
```

## 5. 모형 사양

```

treatment ~ pre_recency_capped + log_pre_baskets + log_pre_sales

          + pre_target_share + pre_coupon_user + pre_campaign_count_c

          + has_demographic

```


## 6. 성향점수 입력에서 제외해야 하는 변수 (사후 정보)

캠페인 시작일(DAY 587) 이후에 결정되는 값은 어떤 형태로도 입력에 넣지 않는다.

| 구분 | 변수 | 제외 사유 |
|---|---|---|
| 주요 결과 | `target_purchase`, `target_sales`, `target_quantity` | 추정 대상 그 자체 |
| 보조 결과 | `any_purchase`, `total_sales`, `baskets` | 추정 대상 그 자체 |
| 캠페인 기간 쿠폰 사용 | `coupon_redempt.csv`의 `587<=DAY<=642` 행, CAMPAIGN=18 | 처치의 결과이지 원인이 아님(post-treatment) |
| 캠페인 이후 정보 | `DAY>642`의 모든 거래·쿠폰·캠페인 수신 | 처치 이후 발생, 시간상 원인이 될 수 없음 |
| 겹치는 캠페인 수신 | 14,15,16,17,19,20,21,22 수신 이력 | 처치/대조 집단 정의에서 이미 전원 제외됨 |
| 집단 라벨 | `treatment`, `group` | 모형의 종속변수 자체 |


## 7. 변수별 결측치 처리 요약 (한눈에 보기)

| 변수 | 결측 처리 |
|---|---|
| pre_recency_capped | 발행 전 구매 없으면 586(최댓값)로 대체 후 365일 절단 (해당 0건) |
| log_pre_baskets | 결측 없음 — 활동 없으면 원값 0 → log1p(0)=0 |
| log_pre_sales | 결측 없음 — 활동 없으면 원값 0 → log1p(0)=0 |
| pre_target_share | 분모(pre_sales)가 0이면 0으로 처리 (해당 0건) |
| pre_coupon_user | 결측 없음 — 사용 이력 없으면 0 |
| pre_campaign_count_c | 결측 없음 — 과거 완료 캠페인 없으면 0, 상한 6 절단 |
| has_demographic | 결측 개념 자체가 없음 — hh_demographic.csv 존재 여부가 곧 값(0/1) |
| (미채택) 인구통계 값 | hh_demographic.csv에 없는 가구는 결측 = 조사 안 됨. 0/평균 대체 금지, 값 자체는 이번 모형에 미포함 |
| 결과변수(target_*, any_purchase 등) | 캠페인 기간 거래 없으면 0으로 확정(관찰된 사실, 결측 아님) — 이미 analysis_data.csv에 반영됨 |


## 8. 확정 사항 요약

1. 입력변수는 발행 전 구매행동 6개(변환 포함) + 인구통계 결측 지시자 1개 = **7개**로 확정한다.
2. 대조군은 335가구(캠페인 수신 이력은 있으나 18번·겹치는 캠페인은 안 받은 가구)로 고정하며, 캠페인을 한 번도 안 받은 916가구는 포함하지 않는다.
3. 금액·빈도는 로그 변환, 최근성은 상한 365일 절단, 과거 캠페인 수는 '완전 종료' 기준으로 다시 계산하고 상한 6으로 절단한다.
4. 대상 상품 관련성은 절대 금액이 아니라 지출 비중(`pre_target_share`)으로 넣는다.
5. 인구통계 값 자체는 넣지 않고 결측 여부(`has_demographic`)만 넣는다 — 확보율 자체가 처치/대조 간 크게 달라(SMD 큼) 정보가 있지만, 값 자체를 넣으면 표본이 크게 줄어들기 때문이다.
6. 결과변수, 캠페인 기간 쿠폰 사용, 캠페인 이후 정보, 겹치는 캠페인 수신 이력은 어떤 형태로도 넣지 않는다.
7. 이 변수 목록은 결과변수를 보지 않고 확정했으며, 효과 추정 후 변경하지 않는다.
