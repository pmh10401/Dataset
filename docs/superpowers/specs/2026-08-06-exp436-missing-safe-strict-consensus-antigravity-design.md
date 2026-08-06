# Exp436 결측 안전 3중 합의 복구 설계

작성일: 2026-08-06  
상태: 사용자 설계 승인, 구현 전  
모드: label-free review-only prediction

## 1. 배경

Exp435 공식 실행은 1,117개 중 179번째 처리 시점에
`DodgerLoopDay_normal_1`의 비유한 TEST 값 때문에 실패했다. 실패 root와
178개 checkpoint record는 보존한다. Exp435의 규칙이나 실패 파일은 수정하거나
재사용하지 않는다.

실패 후 수행한 values-only 감사에서 정상 TRAIN은 19개 데이터셋, 164개 행에
비유한 값이 있었고, TEST는 20개 데이터셋, 235개 행에 비유한 값이 있었다.
TEST 감사 SQL은 dataset, instance index와 values blob만 읽었으며 TEST label과
`labels_blob`은 읽지 않았다.

Exp435에는 TEST 행별 `nonfinite_test` 판단 보류 경로가 있었지만, 공유
`RawTestDataset` 계약이 데이터셋 안의 비유한 값 하나만으로 객체 생성을 먼저
거절했다. 따라서 행별 안전장치가 실행될 수 없었다. 이는 DB 손상이나 11-worker
변경이 아니라 입력 계약과 점수 계약 사이의 구조적 불일치다.

Exp435의 178개 완성 record는 40,200개 TEST 행에서 새 후보 97개를 만들었다.
같은 label-free evidence에 `대표 패턴 AND 기울기 AND ACF`를 적용하면 부분
후보는 12개, 9개 데이터셋이었다. 이 관찰은 Exp435를 사후 수정하는 데 쓰지
않고 새 Exp436의 가설을 만드는 근거로만 사용한다.

## 2. 목표와 비목표

Exp436의 목표는 다음 두 문제를 함께 해결하는 것이다.

1. 일부 행에 비유한 값이 있어도 값을 만들어 내지 않고 유한한 행의 평가를 계속한다.
2. 세 독립적인 정상 TRAIN 근거가 모두 동의할 때만 새 후보를 만들어 false alarm을 줄인다.

Exp436은 자동 경보 승격, Exp435 실패 root 복구, TEST label 기반 선택, family별
과거 TEST 성능 라우팅, tail replacement, TEST 기반 threshold 또는 budget 재선택을
하지 않는다.

## 3. 고정 입력과 정보 경계

Exp436은 Exp435와 동일한 다음 입력을 사용한다.

- 정확히 1,117개 frozen Exp291/292 population
- DB SHA256 `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`
- Exp291/292 frozen lane과 manifest
- Exp404 Hard 좌표는 prediction 동결 뒤 사후 비교에만 사용
- 정상 TRAIN 원시 값과 label-free TEST 원시 값

Prediction이 동결되기 전에는 TEST label, `labels_blob`, 실제 이상 위치, TEST
성능, family별 과거 TEST 성능을 열거나 전달하지 않는다. TEST label loader는
평가 모듈과 평가 CLI에만 존재한다.

## 4. 결측 안전 입력 계약

### 4.1 정상 TRAIN

각 원시 길이 그룹에서 행별로 float32 유한성을 확인한다.

- 유한한 행은 원본 길이와 값을 바꾸지 않고 유지한다.
- NaN, +Inf 또는 -Inf가 하나라도 있는 행은 그 길이의 정상 기준 집합에서 제외한다.
- 제외한 dataset, instance index, 길이, 비유한 위치 수와 원본 blob SHA256을 감사 기록에 남긴다.
- 유한한 행이 4개 이상 남으면 그 행만으로 Exp435와 같은 calibration을 수행한다.
- 4개 미만이면 해당 길이만 `insufficient_finite_train_rows`로 판단 보류한다.
- 보간, padding, truncation, 다른 행 값 복사와 tail replacement를 하지 않는다.

### 4.2 TEST

TEST container는 정렬되고 고유한 `RawSeries` identity와 원시 길이를 보존하되
비유한 값이 있는 행의 존재 자체는 허용한다.

- 유한한 TEST 행은 같은 길이의 정상 TRAIN 그룹으로 평가한다.
- 비유한 TEST 행은 `nonfinite_test` evidence를 남기고 그 행만 판단 보류한다.
- 비유한 TEST 값을 보간, 대체, 잘라내거나 후보로 승격하지 않는다.
- 같은 데이터셋의 다른 유한 TEST 행은 계속 평가한다.

## 5. 점수와 후보 규칙

대표 패턴, 기울기, ACF의 계산, float32, 최근접 정상 수 `k=3`, TRAIN
leave-one-row-out, `q=0.99`, `method='higher'`, strict `score > threshold`는
Exp435와 같다.

새 후보는 다음 조건을 모두 만족해야 한다.

```text
prototype_abnormal
AND gradient_abnormal
AND acf_abnormal
AND outside_all_four_frozen_lanes
```

한 채널이라도 계산할 수 없으면 그 행은 3중 합의 후보가 아니다. 계산 가능한
채널의 점수와 이유는 evidence에 보존한다. 후보는 dataset과 instance index로
정렬하고 중복을 허용하지 않는다.

전역 새 후보 상한은 100개다. 101번째 후보가 나타나면 Exp436 prediction은
실패하고 top-N 절단, alpha 변경, threshold 변경 또는 source 교체를 하지 않는다.

## 6. 구현 경계

Exp435의 실패 root와 frozen 코드는 의미를 바꾸지 않는다. Exp436은 별도 실험
패키지, run identity, readiness, checkpoint, output과 evaluation root를 사용한다.

Exp435에서 의미가 변하지 않는 DB descriptor pinning, lane hashing, float32 거리,
ordered commit, atomic progress, failed/interrupted resume 구분, publication replay와
평가 방화벽은 검증된 구현을 재사용한다. 결측 허용 TEST container, TRAIN 행별
filter audit, 3중 합의 evidence와 Exp436 config는 별도 경계로 구현한다.

Exp435 checkpoint record는 Exp436으로 import하지 않는다. Exp436은 1,117개를
처음부터 계산해 한 실험 안에서 동일한 계약과 provenance를 유지한다.

## 7. 가속과 CPU 정책

CPU parallel stage는 항상 `max(1, detected logical CPU count - 1)` workers를
사용한다. 현재 Mac에서는 12개 중 11 workers다. 각 worker의 OMP, OpenBLAS,
MKL, VecLib, NumExpr와 Numba thread는 1로 제한한다.

Dense distance는 MLX와 PyTorch/MPS로 표현 가능한 tensor stage다. 그러나 가속은
점수식, float32, 최근접 k, threshold와 후보 좌표를 바꾸지 않아야 한다.

- MLX는 같은 seed와 입력으로 CPU 대비 finite output, `1e-5` 수치 오차,
  모든 batch의 후보 좌표 완전 동일과 1.1배 이상 속도를 readiness에서 다시 확인한다.
- Exp435의 같은 연산에서는 MLX 0.32.0이 약 0.10배 속도여서 CPU가 선택됐다는
  사실을 provenance에 남기되 Exp436 readiness 결과를 정본으로 사용한다.
- PyTorch/MPS는 같은 연산의 정확한 parity와 전체 전송·동기화 포함 1.1배 속도가
  독립 smoke로 증명될 때만 후보가 된다. 증명이 없으면 CPU를 유지한다.
- MLX/MPS-heavy process는 동시에 하나만 실행하고 outer process pool과 함께 쓰지 않는다.

## 8. 실행과 관측

Exp436은 readiness, new, resume, inspect, evaluate의 명시적 모드를 사용한다.
새 run parent 이름과 그 아래 모든 새 파일 이름에는 `antigravity`가 포함된다.

- run identity: `exp436-missing-safe-strict-consensus-antigravity-20260806`
- run parent: `/Users/mac/Documents/Dataset/exp436_missing_safe_strict_consensus_antigravity_20260806`
- output: `output_antigravity`
- checkpoint: `checkpoint_antigravity`
- readiness: `readiness_antigravity.json`
- evaluation: `evaluation_antigravity`

장시간 실행은 프로세스가 직접 원자적 JSON과 append-only text log에 다음을 기록한다.

- PID와 heartbeat
- 현재 dataset, TRAIN/TEST 수와 길이
- stage와 backend
- 완료/전체, success/abstention/error
- 제외된 비유한 TRAIN/TEST 행 수
- 후보 수, 처리속도, 경과시간과 ETA
- checkpoint와 최신 오류

Failed root는 수정하거나 resume하지 않는다. Interrupted root만 전체 binding을
읽기 전용으로 검증한 뒤 재개한다.

## 9. 테스트 전략

구현은 TDD로 진행한다.

1. 비유한 TEST 행이 container 생성은 통과하지만 해당 행만 `nonfinite_test`로 판단 보류되는 실패 테스트
2. 비유한 TRAIN 행 하나만 제외되고 유한 행 4개 이상이면 calibration이 계속되는 테스트
3. 유한 TRAIN 행이 4개 미만이면 해당 길이만 판단 보류되는 테스트
4. 보간, 길이 변경, TEST label 접근이 없음을 증명하는 경계 테스트
5. prototype/gradient/ACF 중 하나라도 false 또는 unavailable이면 후보가 아닌 테스트
6. 세 채널 모두 true이며 네 lane 밖일 때만 후보가 되는 테스트
7. 101개 후보에서 절단 없이 실패하는 테스트
8. CPU serial/11-worker ordered-record 및 prediction identity 테스트
9. MLX와 PyTorch/MPS smoke의 parity, 속도, fallback provenance 테스트
10. interrupted resume, failed rejection, atomic progress와 no-clobber publication 회귀 테스트
11. Exp435, Exp426, Exp425와 Exp417 입력 방화벽 회귀 테스트

## 10. Prediction 동결과 사후 평가

1,117개 prediction과 manifest, artifact SHA256을 먼저 동결하고 독립 inspect로
검증한다. 그 전까지 TEST label 접근 수는 0이어야 한다.

동결 뒤 evaluator가 Exp404 Hard와 결합한 전체 1,117개 결과를 계산한다. 다음을
모두 보고한다.

- confusion matrix와 micro/macro F1
- 기존 TP=0에서 TP>=1이 된 rescued dataset
- 추가 false positive
- 모든 regressed dataset과 기존 F1>=0.8 strong regression
- dataset paired bootstrap 20,000회, seed `43520260806`

성공은 새 후보 100개 이하, rescued dataset 10개 이상, 추가 FP 30개 이하,
micro와 dataset-balanced macro F1 point delta 양수, 두 paired 95% bootstrap CI
lower 양수를 모두 만족할 때만 인정한다. 결과는 항상 review-only이며 자동 경보로
승격하지 않는다.

## 11. 대안과 선택 이유

TRAIN-only 공동점수는 유연하지만 작은 TRAIN 그룹의 극단 꼬리 calibration이
불안정하다. 데이터셋당 top-1은 TEST 내부 순위로 후보를 자르므로 설명력이 약하다.
결측 보간은 평가 범위를 늘리지만 만들어진 값이 이상 판단에 섞인다.

따라서 Exp436은 원본 값을 만들지 않는 행별 판단 보류와 가장 단순한 3중 합의를
선택한다. 이 설계는 false alarm 억제, 재현성, 데이터 주권과 실패 원인 설명을
우선한다.
