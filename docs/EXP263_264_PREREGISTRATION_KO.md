# Exp263-264 전체 결측 인지 통합 검증 사전등록

## 목적

기존 자동 경보를 보존하면서 결측 우선 전처리와 네 점수원 합의 정책을
1,117개 데이터셋 전체에 적용한다. Exp263은 라벨 없는 prediction을 만들고,
Exp264는 prediction을 SHA-256으로 동결한 뒤 TEST 라벨을 사용해 자동 경보와
사람 검토 lane을 분리 집계한다.

## 범위와 고정 입력

- 모집단은 `/Users/minho/Documents/Dataset/univariate_ts.db`의 전체 1,117개 데이터셋이다.
- 산출물은 `/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration`에 저장한다.
- 유의수준은 `ALPHA = 0.005`로 고정한다.
- 기존 lane은 Exp253의 `frozen_predictions.jsonl`에서 읽고 기존 Hard alert는 그대로 유지한다.
- 통계, 구조, 변화점, 마스킹 복원 점수원은 각각 고정 variant
  `combined_40`, `structural_combined_440`, `change_point_combined_33`,
  `mask_025__seed_intersection`을 사용한다.
- 결측 영향 20개 데이터셋에는 Exp260-262의 `corrected_four_sources` 결과를
  override로 사용한다. 나머지 1,097개에는 동일한 기존 점수원 결과를 사용한다.
- 마스킹 복원 점수는 `n_train >= 100`인 데이터셋에만 적용한다.

## 고정 라우팅 규칙

- 새 Hard 승격은 기존 Standard 후보 중 `n_train >= 100`, 네 점수원 중 3개 이상
  동의, 결측 품질 통과를 모두 만족할 때만 허용한다.
- 두 점수원 동의 또는 결측 품질 차단 후보는 Priority review로 이동한다.
- 0~1개 동의 후보는 Standard review에 유지한다.
- 기존 Hard alert는 삭제하거나 낮은 lane으로 이동하지 않는다.
- 결과를 확인한 뒤 alpha, variant, threshold, 동의 수, `n_train` 기준을 바꾸지 않는다.

## 금지된 정보

Prediction 단계에서는 TEST 라벨, 이상 위치, family별 TEST 성능을 읽지 않는다.
TEST 성능, family 정보, 결과 후의 threshold 조정, 후보 수 조정, tail replacement,
수동 데이터셋 제외를 정책 선택에 사용하지 않는다. Evaluation 단계는 동결된
prediction SHA-256을 먼저 확인한 뒤에만 TEST 라벨을 읽는다.

## 해석 한계

이번 결과는 retrospective integration validation이다. prospective 검증이나
end-to-end strict TRAIN-only 운영 검증으로 표현하지 않는다. Exp253의
`selected_from_prior_test_review=True` 및 `deployment_eligible=False` 한계를
그대로 유지한다. 자동 경보 지표와 human-assisted review 지표는 서로 합치거나
운영 성능 하나로 해석하지 않는다.
