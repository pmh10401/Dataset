# Exp167/168 Score-Calibration Parity Design

## 목적

Exp165는 소표본 TRAIN-normal score-similarity pooling으로 threshold 해상도와 정상 pseudo-run false-review를 개선했지만, Exp166 TEST 평가에서는 TP/FP 균형과 Mean F1을 함께 개선하지 못했다. 가장 유력한 원인은 calibration score가 cross-fit 모델에서 생성된 반면 TEST score는 full-TRAIN 모델에서 생성되어 점수의 fit 크기, prototype, center와 scale 조건이 달랐다는 점이다.

Exp167/168은 feature를 추가하거나 alpha를 다시 선택하지 않는다. 같은 fold 모델이 만든 held-out TRAIN 정상 점수와 TEST 점수를 직접 비교하여 score-calibration parity의 영향을 분리한다.

## 범위와 명칭

- 대상 population: Exp165/166과 동일한 `5 <= n_train < 20` 데이터셋 316개
- `n_train < 5` 데이터셋 162개: 완전 기권 유지
- 점수원:
  - `spectral`: 주파수 변화 점수
  - `shapelet`: 정상 패턴 거리 점수
  - `interval`: 구간 통계 점수
- 고정 alpha: `(0.005, 0.01, 0.02, 0.05)`
- 모든 결과: retrospective research sensitivity
- 후보 lane: research review candidate
- autonomous Hard alert: 평가하지 않음
- TEST 성능을 이용한 alpha, source pair 또는 정책 선택: 금지

## Exp167: Fold-Matched Parity Audit

### 데이터 흐름

각 데이터셋의 TRAIN 정상 표본에 Exp164의 결정적 fold plan을 그대로 적용한다. 이 population은 `n_train < 20`이므로 leave-one-out fold를 사용한다.

각 fold에서 다음을 한 번에 계산한다.

1. `n_train - 1`개 TRAIN 정상 trace로 feature extractor와 score model을 fit한다.
2. 제외된 TRAIN 정상 trace 하나를 held-out query로 점수화한다.
3. 전체 TEST trace를 같은 fold model로 점수화한다.
4. 세 점수원 각각에 대해 fit score, held-out score, TEST score를 저장한다.

TEST label은 이 단계에서 읽지 않는다.

### Fold-paired empirical p-value

TEST index `j`, source `s`, fold `i`에 대해:

```text
heldout_score[i, s] = fold i에서 제외된 TRAIN 정상의 점수
test_score[i, j, s] = 같은 fold i 모델이 만든 TEST j의 점수

p[j, s] =
  (1 + sum_i I(heldout_score[i, s] >= test_score[i, j, s]))
  / (n_train + 1)
```

낮은 p-value일수록 fold 전반에서 TEST 점수가 held-out 정상보다 크다는 뜻이다. `+1` finite-sample correction을 유지하며 TEST 결과를 위해 제거하지 않는다.

이 값은 source별 fold-paired empirical p-value로 부른다. end-to-end conformal guarantee 또는 실제 설비 false-alarm guarantee라고 부르지 않는다.

### 감사 지표

Exp167은 label-free audit만 수행하고 다음을 기록한다.

- dataset/source별 fold 수와 유효 비교 수
- OOF held-out score와 full-fit TRAIN score의 location/scale 차이
- fold별 TEST score의 median, IQR, max와 fold 간 변동성
- 기존 Exp166 full-fit TEST score와 fold-aggregated TEST score의 rank correlation
- source별 최소 가능한 p-value `1 / (n_train + 1)`
- alpha별 resolution-blocked 여부
- non-finite score, shape mismatch, index 범위 오류
- prediction에 TEST label 또는 평가 metric이 없는지

Exp167의 목적은 성능 결론이 아니라 score 생성 조건 불일치의 크기와 p-value 해상도를 확인하는 것이다.

## Exp168: Frozen Parity Candidate Evaluation

### 사전 고정 후보 규칙

Exp167에서 생성한 source별 p-value를 사용해 다음 네 source 구성과 두 agreement 규칙을 모두 보고한다.

Source 구성:

1. `spectral+shapelet`
2. `spectral+interval`
3. `shapelet+interval`
4. `spectral+shapelet+interval`

두 source 구성은 두 source 모두 `p <= alpha`일 때 후보로 한다. 세 source 구성은 다음을 각각 보고한다.

- `2of3`: 세 source 중 둘 이상이 `p <= alpha`
- `3of3`: 세 source 모두 `p <= alpha`

네 alpha와 모든 source 구성을 함께 유지한다. TEST 결과가 좋은 하나를 primary, optimal 또는 recommended로 선택하지 않는다.

### Prediction과 evaluation 분리

Prediction 함수는 다음 입력만 받는다.

```text
predict_dataset(
    x_train_normal,
    x_test,
    frozen_config,
)
```

Prediction 단계는 TEST values를 점수화하지만 다음 정보는 받지 않는다.

- `y_test`
- 실제 이상 위치
- family 이름에 따른 정책 분기
- dataset 이름에 따른 정책 분기
- TEST 길이에 따른 threshold 또는 budget
- 과거 TEST 성능

Prediction JSONL을 SHA-256으로 동결한 뒤 evaluation 단계에서만 TEST label을 읽는다.

### 평가 지표

각 alpha/source 구성마다 다음을 보고한다.

- datasets
- review candidates, TP, FP, micro precision
- macro recall, Mean F1
- zero-F1 dataset 수
- candidate-zero dataset 수
- dataset별 candidate mean/median/p90/p95/max
- Exp166 local baseline 대비 added/removed TP와 FP
- candidate exact Jaccard
- prediction coverage, lane overlap, index range 오류
- resolution-blocked dataset 수

Hard alert 지표와 combined F1을 생성하지 않는다. 모든 지표는 research review candidate의 후향 평가로 명시한다.

## 불변성 및 안전 계약

다음 테스트를 구현 전에 실패하도록 먼저 작성한다.

1. fold-paired p-value가 같은 fold의 held-out/test score만 비교한다.
2. finite-sample `+1` correction과 최소 p-value를 보존한다.
3. fold 순서가 바뀌어도 p-value가 동일하다.
4. TEST label을 바꿔도 prediction이 동일하다.
5. prediction 함수 signature에 `y_test`가 없다.
6. prediction JSONL에 TP/FP/F1/combined F1이 없다.
7. alpha와 source 구성 전체가 빠짐없이 출력된다.
8. candidate index 중복, 범위 오류, count 불일치가 없다.
9. 같은 seed와 입력에서 prediction hash가 동일하다.
10. Exp165/166 결과 파일을 덮어쓰지 않는다.

## 출력 구조

```text
experiments/exp167_168/
  __init__.py
  parity_core.py
  run_experiment_167_168_score_calibration_parity.py
  test_exp167_168_parity_core.py
  test_exp167_168_structure.py

outputs/exp167_168_score_calibration_parity/
  01_contract.json
  02_parity_audit.csv
  03_fold_score_summary.csv
  04_frozen_predictions.jsonl
  05_prediction_manifest.json
  06_exp168_dataset_metrics.csv
  07_exp168_global_summary.csv
  08_candidate_transition.csv
  09_invariance_report.md
  10_result_analysis.md
  MANIFEST.csv
```

대용량 fold score cache와 checkpoint는 output 하위 cache 디렉터리에 저장하며 GitHub 릴리스에서는 제외한다.

## 실행 순서

1. 구조 및 core 단위 테스트 작성, 예상 실패 확인
2. 최소 core 구현 후 테스트 통과
3. 10-dataset smoke audit/predict/evaluate
4. Exp167 전체 label-free parity audit
5. Exp168 전체 label-free prediction
6. prediction SHA-256 동결 및 독립 무결성 확인
7. 마지막에만 TEST label 평가
8. 독립 재집계와 결과 분석
9. 대시보드 catalog와 순차 큐 등록

## 중단 조건

- 기존 Exp165 target 316개 coverage를 재현하지 못함
- 같은 fold model로 held-out TRAIN과 TEST를 점수화하지 못함
- prediction 단계가 TEST label을 읽음
- non-finite score 또는 dataset 누락을 임의로 제외해야 함
- 결과를 본 뒤 alpha, source 구성, correction 또는 threshold를 바꿔야 함

중단 시 일부 결과를 성공으로 처리하지 않고 blocker와 재개 조건을 기록한다.

## 성공 해석

Exp168이 Exp166 local baseline보다 FP를 줄이면서 TP와 Mean F1을 유지하면, score-calibration parity가 유효한 개선 신호라고 해석한다. 성능이 유지되지 않으면 fit-condition mismatch만으로 Exp166 실패를 설명할 수 없으며, 다음 단계는 source별 score 안정성 또는 새로운 feature가 된다.

어떤 결과도 prospective validation, 운영 alpha 확정, autonomous Hard alert 성능 또는 end-to-end strict TRAIN-only 완료를 의미하지 않는다.
