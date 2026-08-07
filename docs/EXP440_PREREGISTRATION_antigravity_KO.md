# Exp440 사전등록: TRAIN 전용 저주파 정밀도 문턱

## 목적

Exp439의 저주파 점수는 기존에 놓친 이상을 233개 더 찾았지만 정상도 996개나 이상으로 잘못 골랐다. Exp440은 저주파 신호 자체를 버리지 않고, **여러 분해 깊이에서 반복해서 강하게 나타나며 TRAIN 기준이 안정적인 후보만** 남긴다.

## 동결 신원과 입력

```text
experiment_id = Exp440
run_identity = exp440-train-only-low-frequency-precision-gate-antigravity-20260808
dataset_count = 1117
database_sha256 = 8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c
exp439_prediction_sha256 = 500f6956b8640ea329cd47c8cdb504debaeff6ec67c44d0c56b96c1ecd77f3ef
hard_prediction_sha256 = 437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70
nearest_k = 3
maximum_train_rows_per_exact_length_group = 256
quantile = 0.99, method = higher, comparison = strict greater
stability_ratio_max = 2.0
consensus_depths_min = 2
outside_hard_budget = 100
bootstrap_seed = 44020260808
bootstrap_samples = 20000
```

실행 루트는 `/Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808`로 고정한다.

## 예측 규칙

1. 정상 TRAIN 행만 사용한다. 비유한 TRAIN 행은 행 전체를 제외하고 기록한다. 한 exact-length 그룹의 유한 정상 TRAIN이 256행을 넘으면 임의 추출하지 않고 그 그룹 전체를 `train_group_over_compute_cap`으로 abstain한다. 이 규칙은 TEST를 보기 전에 고정한 정밀도·실행 가능성 보호 규칙이다.
2. 각 길이 그룹에서 Exp439가 고른 깊이 `d`와 경계 안의 `d-1`, `d+1`을 사용한다. 유효 깊이가 두 개보다 적으면 후보를 만들지 않는다.
3. 각 깊이에서 한 TRAIN 행을 빼고 나머지 TRAIN만으로 변환과 최근접 정상 거리 `k=3`을 다시 계산한다.
4. OOF 점수의 0.99 분위수를 `higher` 방식으로 문턱으로 삼으며, TEST 점수는 문턱보다 **엄격히 클 때만** 넘은 것으로 본다.
5. TRAIN 행 하나를 지울 때마다 문턱을 다시 계산한다. 가장 큰 문턱을 `max(가장 작은 문턱, 1e-12)`로 나눈 값이 2.0을 넘으면 그 깊이는 불안정하므로 쓰지 않는다.
6. 안정적인 깊이 중 최소 두 곳에서 문턱을 넘은 TEST 좌표만 임시 후보가 된다.
7. Hard 좌표는 먼저 제외한다. 남은 후보는 합의 깊이 수, 두 번째 초과 비율, 두 번째 경험 순위, 데이터셋 이름, 행 번호 순으로 정렬해 최대 100개만 남긴다.

## 금지 사항

- 예측 중 TEST label, 실제 이상 위치, Exp439 평가 파일, 과거 family별 TEST 성능을 읽지 않는다.
- 결측값 대체, 길이 맞춤, tail replacement를 하지 않는다.
- TEST 결과를 보고 같은 Exp440의 깊이, 문턱, 예산 또는 점수원을 바꾸지 않는다.
- Exp439 산출물은 수정하지 않는다.
- 결과는 review-only이며 자동 경보로 승격하지 않는다.

## 가속 규칙

CPU는 논리 코어 수에서 하나를 뺀 worker 수를 사용한다(현재 12코어에서 11 worker, 각 worker 수치 스레드 1개). MLX와 Numba는 같은 입력·seed에서 유한 출력, 후보 좌표 완전 일치, 최대 오차 `1e-5` 이하, 전체 단계 속도 1.1배 이상을 모두 만족할 때만 채택한다.

## 사후 성공 기준

prediction SHA256을 먼저 동결한 뒤에만 TEST label을 연다. 후보 수 100 이하, 신규 복구 데이터셋 10개 이상, 추가 FP 30 이하, 후보 precision 0.35 이상, micro/macro F1 모두 증가, 두 paired bootstrap 95% 신뢰구간 하한 모두 0 초과, baseline F1 0.8 이상인 강한 데이터셋에서 0.2 초과 하락 없음이 모두 성립할 때만 승자로 판단한다.
