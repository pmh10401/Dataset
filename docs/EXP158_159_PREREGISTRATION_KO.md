# Exp158–159 사전등록

## 연구 구분 및 연구 계약

- **연구 구분**: 소표본 데이터셋을 위한 retrospective counterfactual sensitivity (사후적 반사실적 민감도 분석)
- **제한 정책**: 자동 운영 정책(automatic operational policy) 또는 prospective validation(예측적 검사)으로 표현하거나 보고하지 않는다.

## 고정 연구 계약 (Fixed Contract)

1. **Alpha 그리드 유지**: `ALPHAS = (0.005, 0.01, 0.02, 0.05)`를 유지하며, 최종 TEST 결과에 따라 특정 alpha를 임의 선택하지 않는다.
2. **기존 캐시 재사용**: `outputs/exp153_157_train_only_improvements/01_score_cache_manifest.csv`와 `cache/*.npz`를 재사용하며 캐시 SHA-256 해시를 엄격히 검증한다.
3. **Pooling 대상**: target pool 생성은 `5 <= n_train < 20` 조건의 데이터셋으로만 국한한다.
4. **소표본 기권 (Abstention)**: `n_train < 5` 데이터셋은 기존 계약대로 complete calibration abstention을 적용하고 후보를 생성하지 않는다.
5. **대형 표본 제외**: `n_train >= 20` 데이터셋은 이 1차 실험의 similarity pooling target에서 제외한다.
6. **TRAIN-only 정보 사용**: 이웃 선택 및 pool 생성에는 target 데이터셋의 TRAIN OOF score만 사용한다. TEST score, TEST label, 실제 이상 위치, TEST 길이, family 이름, dataset 이름 기반 정책 분기, 과거 TEST 성능은 일절 사용하지 않는다.
7. **데이터셋 명칭의 역할**: dataset 이름은 조회 key와 출력 식별자로만 사용한다.
8. **자가 참조 배제 (Target Exclusion)**: target dataset의 TRAIN OOF score는 pooled conformal reference에서 반드시 제외한다.
9. **최소 레퍼런스 크기**: pool의 최소 reference 수는 기존 계약과 동일하게 `MIN_POOL_REFERENCE = 200`으로 고정한다.
10. **사후 튜닝 금지**: TEST 결과를 보고 neighbor 수, descriptor 구성, alpha, threshold, pool 크기 또는 lane 규칙을 사후 조정하지 않는다.
11. **규칙 엄격성**: Priority review를 Hard alert로 승격하지 않으며, tail replacement와 TEST-length top-N budget을 사용하지 않는다.

## Exp158: Score-similarity 설계

네 TRAIN OOF score source (`rocket_exp40`, `exp55_best`, `exp56_best`, `exp84`)에서 다음 단계를 거쳐 descriptor를 만든다.

1. Source별 finite TRAIN OOF score를 해당 데이터셋의 TRAIN median/MAD 기반 robust z-score로 변환한다.
2. Source별 `q10, q25, q50, q75, q90, mean, std` 7개 통계치를 계산한다.
3. 네 source의 결과를 연결하여 28차원 descriptor를 생성한다.
4. 전체 eligible donor descriptor의 각 차원을 TRAIN-only median/IQR로 robust scale한다.
5. Euclidean distance로 target과 donor의 거리를 계산한다. (target 자신은 제외)
6. 거리 오름차순으로 donor를 추가해 각 source의 pooled reference가 최소 200개가 될 때까지 모은다.
7. 최소 200개를 처음 넘는 거리와 같은 거리에 있는 donor는 모두 포함한다. (dataset rename이나 CSV row order가 tie 결과를 바꾸지 않도록 보장)
8. donor가 부족하여 최소 200개를 채우지 못하면 `pool_unavailable`로 기록하고 후보를 생성하지 않는다.
9. Target TEST score는 target 자신의 TRAIN OOF median/MAD로만 z-score화한 뒤, source별 pooled reference에 대해 기존 `+1` empirical conformal p-value를 계산한다.
10. 동일 index에서 네 source 중 2개 이상이 `p <= alpha`이면 Exp158 후보로 기록한다.

Descriptor를 계산할 수 없는 source가 있는 데이터셋은 donor와 target 모두에서 제외하고 그 이유를 manifest에 기록한다.

## Exp159: Lane 통합

기준 lane은 frozen `outputs/exp151_152_virtual_run_conformal_v2/04_exp152_b2_source_predictions.jsonl`이다. 해당 파일과 `05_prediction_manifest.json`의 SHA256 해시를 검증한다.

- baseline Hard, Standard, Priority는 그대로 보존한다.
- Exp158의 새 similarity 후보 중 baseline 어느 lane에도 없는 index만 Exp159 Standard review에 추가한다.
- Exp159 Hard와 Priority는 baseline과 exact match여야 한다. (새 후보를 Hard로 올리지 않음)
- 모든 lane은 상호 배타적(mutually exclusive)이어야 한다.
- `candidate = Hard union Standard union Priority`가 만족해야 한다.
- Exp159 prediction에는 TEST label 또는 TP/FP/F1/combined_f1을 넣지 않는다.
- evaluation은 frozen prediction SHA256을 확인한 뒤 별도 단계에서만 TEST label을 읽는다.
- `mean_combined_f1`은 human-assisted diagnostic으로만 표시한다.
