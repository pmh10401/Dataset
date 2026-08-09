# Exp444 TRAIN-only 안정성 정밀화 사전등록

Exp444는 이미 동결된 Exp442의 30개 좌표를 그대로 두고, 각 후보의 정상 TRAIN
구성을 결정적으로 조금씩 바꾸었을 때도 동일한 Exp440 규칙으로 후보가 남는지를
기록하는 label-free review 실험이다. 새로운 좌표를 만들거나 후보를 보충하지
않으며, stable/unstable 두 review lane을 설명적으로 보존한다.

## 동결 설계와 실행 신원

승인된 설계의 bytes와 SHA256을 다음과 같이 묶는다.

```text
design_source = docs/superpowers/specs/2026-08-10-exp444-train-only-stability-precision-gate-antigravity-design.md
design_source_sha256 = 1bb76f19d07ef312d5ac5498414d0c914ce94b99c5912e0bbcf3f0197c1400e8
experiment_id = Exp444
run_identity = exp444-train-only-stability-precision-gate-antigravity-20260810
run_root = /Users/mac/Documents/Dataset/exp444_train_only_stability_precision_gate_antigravity_20260810
isolated_interpreter = /Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314
dataset_count = 1117
source_candidate_count = 30
candidate_per_dataset = 1
maximum_stability_views = 16
stability_fraction = 0.875
mask_seed = 44420260810
bootstrap_seed = 44420260811
bootstrap_samples = 20000
acceleration_tolerance = 1e-5
acceleration_speedup_min = 1.1
expected_cpu_workers_on_current_mac = 11
```

공식 root는 readiness 전까지 없어야 한다. 기존 Exp440·Exp442·Exp443 root와
artifact는 읽기 전용이며 복사·수정·resume하지 않는다. Exp444는 별도 namespace,
checkpoint, publication, evaluation을 사용한다.

## 동결 입력과 SHA256

prediction이 참조하는 좌표는 Exp442의 동결된 30개 prediction row이며 dataset마다
최대 하나다. Exp444는 후보를 추가·교체·이동·padding하지 않는다. readiness는 경로가
아닌 전체 bytes, artifact map, dataset population, 좌표 순서와 선택 identity를
검증한다.

```text
database_sha256 = 8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c
exp404_hard_prediction_sha256 = 437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70
exp442_prediction_sha256 = b79bad6012ad6bc07050ade9968e815796a2cc871b5c85d63f0bd7dbdf6f81f4
exp442_candidate_ranking_sha256 = d8d858cd74553aa64f6f0a5424064872b59f682fed1c8b2ace63fffba6ad20cc
exp442_selection_sha256 = 06004f18dbd9b0df3fa857bb8b1a9fd7eafd4d927f3736fb8cfb25db56b72408
exp442_selected_coordinate_identity_sha256 = db0b9ed00ed5b9313fef10306d04bb539136c5b28b97ebcd325662d9302a3048
```

정상 TRAIN grouping, whole-row nonfinite 제외, FFT 저주파 feature, robust
center/scale, nearest `k=3`, OOF quantile `0.99` (`higher`), strict
`score > threshold`, 두 depth strict consensus는 Exp440과 동일하다. 먼저 전체
정상 TRAIN으로 Exp442 30개 control을 재현하고, 좌표·agreeing depth·candidate가
정확히 맞지 않으면 stability view를 만들지 않는다.

## 결정적 omission view와 lane 규칙

정확한 길이 group의 TRAIN 행 수를 `n`이라 한다. 각 view는 다음 수식을 그대로
사용한다.

```text
omit_count = max(1, floor(0.10 * n))
view_count = min(16, combination(n, omit_count))
required_survival_count = ceil(0.875 * view_count)
```

mask는 dataset, raw length, TRAIN instance identity, `mask_seed`, view counter를
canonical JSON으로 묶어 SHA256 순서를 만든다. 동일 mask는 다시 쓰지 않으며 가능한
mask가 16개보다 적으면 모두 사용한다. 각 view는 남은 TRAIN만으로 Exp440
calibration을 처음부터 만들고, 원래 Exp442 TEST 후보 한 행만 다시 점수화한다.
view가 살아남으려면 (1) 변경하지 않은 Exp440 최소 2-depth strict consensus를
만족하고 (2) 원래 Exp442 agreeing depth 중 적어도 두 개가 해당 view threshold를
넘어야 한다. `survival_count >= required_survival_count`이면 `stable_review`,
그보다 작으면 `unstable_review`로 기록한다. 이 식은 후보 분류 규칙이며 실험
성공 기준이 아니다. 두 lane의 합은 항상 원본 30개다.

## label firewall와 publication 순서

prediction, readiness, checkpoint, publication은 TEST `label`·`labels_blob`,
실제 anomaly position, 과거 TEST 지표, family TEST 성능, TEST missingness를
선택 gate로 쓰는 행위, candidate index 위치, tail 규칙, 평가 후 재선택을
허용하지 않는다. 정상 TRAIN 선택에 필요한 기존 `TRAIN label = 0` 조건만
허용한다. TEST `values_blob`와 TEST identity는 동결된 30개 좌표에 한해 읽을 수
있다.

먼저 1,117개 dataset summary, 원본 30개 좌표, candidate/view mask identity와
SHA256, view별 TRAIN 사용·제외 identity와 score/threshold/rank, survival count/rate,
stable/unstable lane, source/control/backend audit, progress/checkpoint/PID identity,
manifest와 self-excluding hash map을 canonical create-only publication으로
동결한다. publication 전후 label-access count는 0이어야 하고 `inspect`는 SQLite나
worker를 열지 않고 bytes만 재검증한다. publication과 hash map을 두 번 확인한 뒤
별도 evaluator가 한 번만 TEST `instance_index, label`을 읽어 baseline·Exp442·stable
지표, lane별 TP/FP, 전환 dataset, paired micro/macro 변화와 20,000회 bootstrap
구간을 그대로 기록한다. 평가는 prediction bytes를 바꾸지 않는 별도 artifact다.

## backend·thread·실행 계약

전체 실행 전에 동일 input·seed의 정확한 stability stage를 CPU와 MLX/Numba로
비교한다. mask, candidate/view survival, stable/unstable 좌표, finite output이
같고 최대 scalar 절대차가 `1e-5` 이하이며 transfer·synchronization을 포함한
end-to-end stage가 CPU보다 `1.1x` 이상 빠를 때만 가속 경로를 선택한다. manifest에는
MLX/mlx-metal 또는 Numba 버전, device, timing, parity와 fallback 이유를 남긴다.
조건 하나라도 충족하지 않으면 CPU `max(1, os.cpu_count() - 1)` worker를 사용하며
현재 Mac의 기대값은 11이다. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS`,
`BLIS_NUM_THREADS`, `NUMBA_NUM_THREADS`는 NumPy import 전에 모두 `1`이다. MLX/MPS
heavy process는 동시에 하나만 실행하고 outer multiprocessing과 함께 돌리지 않는다.

긴 실행은 부모 process가 atomic progress/PID JSON, append-only events JSONL과 text
log에 current dataset/candidate/view dimensions, counters, throughput, elapsed,
ETA, checkpoint, latest error를 직접 쓴다. Codex는 반복 polling하지 않는다. 실패한
attempt와 staging은 덮어쓰지 않으며 inactive interrupted attempt만 ordered prefix
검증 후 resume한다.

## 거버넌스 해석 (flat block)

```text
scientific_success_criterion = none
evaluation_status = descriptive_complete_no_promotion
adaptive_followup = true
independent_validation = false
scientific_winner_selected = false
review_only = true
operational_alert_eligible = false
same_experiment_reselection_allowed = false
```

Exp444 결과는 수치와 lane을 있는 그대로 남기는 descriptive review evidence다.
어떤 결과도 과학적 승자, champion, 독립 검증, 외부 일반화 또는 운영 경보의 근거로
해석하지 않는다. 같은 Exp444 안에서 결과를 본 뒤 view 수, `0.875`, source,
threshold 또는 budget을 다시 고르지 않는다. 다음 실험 여부는 별도 방향 논의다.
