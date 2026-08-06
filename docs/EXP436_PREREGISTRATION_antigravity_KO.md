# Exp436 결측 안전 엄격 3중 합의 복구 사전등록

## 정본과 상태

이 문서는 과학 정본이며 운영 정본은 `docs/EXP436_RUNBOOK_antigravity_KO.md`다.

- 실험: `Exp436`; 모집단: frozen Exp291/292의 정확히 **1,117개** dataset (`1117`)
- run identity: `exp436-missing-safe-strict-consensus-antigravity-20260806`
- run parent: `/Users/mac/Documents/Dataset/exp436_missing_safe_strict_consensus_antigravity_20260806`
- 상태: product implementation/readiness/prediction/evaluation 미시작; TEST label 미개봉
- mode: label-free `review_only`; 자동 경보 승격, Exp435 root 복구/수정/import 금지

Exp436은 새 root에서 1,117개 전체를 처음부터 재계산한다. Exp435의 failed root와 178개
checkpoint record는 보존하고 재사용하지 않는다. 목표는 비유한 행을 만들지 않고 행별로
안전하게 판단 보류하며 prototype·gradient·ACF가 모두 동의한 새 review 후보만 내는 것이다.

## 동결 입력과 정보 경계

| 입력 | 동결 값 |
| --- | --- |
| DB | `/Users/mac/Documents/Dataset/univariate_ts.db` |
| DB SHA256 | `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c` |
| Exp291 prediction SHA256 | `73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848` |
| Exp291 manifest SHA256 | `761466253cccbb7b91fe36cdbd6d298de5d141c90e01e9fe646a8ae242ca9c3e` |
| Exp292 prediction SHA256 | `f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b` |
| Exp292 manifest SHA256 | `0672b2f84a3b2883628bac3c12ad8a25ebfb48efd929e4c0a448c5418267b0d0` |
| Exp404 Hard comparator SHA256 | `437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70` |
| Exp404 Hard comparator | `/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl` |

Prediction은 normal TRAIN raw values (`split='TRAIN' AND label='0'`), label-free TEST raw
values, dataset/instance identity, raw length, TRAIN/TEST count, frozen population/lane/manifest만
사용한다. TEST label, `labels_blob`, anomaly position, TEST metric/F1/confusion, family별 과거
TEST 성능과 human-assisted 지표는 prediction SHA256·manifest SHA256 동결 전 SELECT, open,
전달, 접근하지 않는다. Exp404 Hard 좌표는 동결 뒤 비교에만 쓴다.

TEST 결과로 alpha, threshold, score source, backend, budget, routing을 재선택하지 않는다.
DB 전체 이름 수는 모집단이 아니며 population 밖 이름·누락·중복은 실패다. tail replacement,
보간, padding, truncation, 행 값 복사도 금지한다.

## 결측 안전 계약

입력은 little-endian `float32`이고 original identity와 length를 보존한다.

- NaN, `+Inf`, `-Inf`가 하나라도 있는 normal TRAIN 행은 해당 length reference에서 행 전체를
  제외하고 dataset/instance/length/nonfinite-position-count/original values-blob SHA256 audit을 남긴다.
- 같은 length의 finite normal TRAIN 행이 4개 이상이면 그 행만으로 calibration한다. 4개 미만이면
  그 length는 `insufficient_finite_train_rows` 판단 보류다.
- TEST container는 정렬·고유 `RawSeries` identity와 raw length를 지키며 nonfinite TEST 행 존재를
  허용한다. nonfinite TEST 행은 `nonfinite_test` evidence를 남기고 그 행만 판단 보류하며 후보가
  아니다. finite sibling TEST 행은 계속 평가한다.
- 값 대체나 생성은 없고 nonfinite score, fingerprint change, checkpoint corruption, write failure는
  전체 prediction 실패다.

## 점수와 후보

- dtype `float32`, `candidate_alpha=0.01`, nearest normal `k=3`, TRAIN leave-one-row-out
- quantile `q=0.99`, `method='higher'`, channel abnormal은 strict `score > threshold`
- shared robust transform: `center=median(matrix)`,
  `scale=float32(1.4826 * median(abs(matrix-center)))`, scale floor `1e-12`
- prototype/raw와 gradient는 각각 자신의 TRAIN-group matrix로 위 transform을 계산한다. ACF는 이미
  동결한 ACF formula를 바꾸지 않는다.
- distance `sqrt(mean((x-reference)**2))`, smallest 3 distance median
- gradient `diff(values, prepend=values[0])`; ACF lag `1..min(32, floor(length/4))`, 최소 length
  16, variance floor `1e-12`

후보는 다음이 모두 참일 때만 허용된다.

```text
prototype_abnormal
AND gradient_abnormal
AND acf_abnormal
AND outside_all_four_frozen_lanes
```

four frozen lane은 Hard, Standard, Priority, automatic-normal이다. channel 하나라도 false 또는
unavailable이면 후보가 아니며 evidence를 보존한다. 후보는 `(dataset, instance_index)`로 정렬·고유다.
global candidate budget은 **100**이고 101번째는 `candidate_budget_exceeded` 전체 실패다. top-N
절단, threshold 변경, source 교체는 없다.

## 실행과 가속

readiness가 정확한 RUN_PARENT를 생성·소유하고 그 아래
`readiness_antigravity.json`을 남긴다. `new`는 그 **같은 existing RUN_PARENT**와 exact verified
readiness artifact를 요구하며 `output_antigravity`, `checkpoint_antigravity`, `evaluation_antigravity`가
모두 부재일 때만 허용한다. `new`는 RUN_PARENT 또는 readiness root를 삭제·덮어쓰지 않는다. 모든 새
artifact filename은 `antigravity`를 포함한다.

CPU worker는 `max(1, (os.cpu_count() or 1) - 1)`이며 numerical-library thread는 worker당 1이다.
현재 Mac은 12 logical CPU, 11 workers다. MLX/PyTorch-MPS smoke seed는 정확히 `43601`이며 각 backend는
batch `(16, 32, 64, 128, 256)`마다 untimed warm-up 정확히 2회와 timed paired CPU/device run 정확히 5회를
수행한다. device work는 timing 안에서 synchronize하고 각 median은 5회 결과의 median이다. finite output,
모든 preregistered candidate coordinate CPU 동일성, max absolute 및 normalized score error 각각 `<=1e-5`,
transfer/synchronization 포함 `>=1.1x` speedup을 모두 보여야만 그 batch가 통과한다. backend 안에서는
통과 batch 중 device median이 가장 낮은 것을 고르고 tie는 더 작은 batch다. MLX/MPS 둘 다 통과하면 더
낮은 device median을 선택하고 exact tie는 lexical backend name(`mlx`가 `mps`보다 먼저)이다. 둘 다
실패하면 CPU다. MLX/MPS-heavy process는 하나뿐이고 채택 시 outer process pool을 쓰지 않는다. manifest는
MLX/mlx-metal/version/device, seed, warm-up/timed-run count, batch medians, parity, 선택과 fallback reason을 기록한다.

프로세스는 `progress_antigravity.json`, `events_antigravity.jsonl`, `parent_antigravity.log`,
`run_antigravity.pid`, ordered canonical checkpoint를 기록한다. PID, heartbeat, current dataset,
TRAIN/TEST counts·lengths, backend, completed/total, success/abstention/error, nonfinite counts,
candidate count, rate, elapsed, ETA, checkpoint, latest error가 필수다. persisted stage는 `loading`,
`dispatching`, `awaiting_worker_result`, `committing`, `idle`뿐이다.

`interrupted` root만 frozen identity/input/progress binding read-only 검증 뒤 resume한다. `failed`
root는 input/PID/progress/event mutation 또는 temp cleanup 전에 read-only 거절하고 수정·삭제·재개하지
않는다. complete resume/inspect는 DB나 worker를 열지 않는 read-only publication replay다.

## 동결 뒤 평가와 성공 기준

prediction은 `review_only=true`, `operational_alert_eligible=false`다. completion은 1,117 처리,
missing/duplicate/out-of-population/failed dataset 0, candidate `<=100`, inactive PID,
`current_dataset=null`, TEST label access 0, checkpoint/event/summary/prediction 및 SHA256 재검증
일치를 요구한다. 그 뒤에만 prediction/manifest/artifact SHA256을 동결한다.

publication artifact에는 `train_exclusion_audit_antigravity.jsonl`도 포함하며 manifest와
`artifact_sha256_antigravity.json`이 그 SHA256을 bind한다. evaluation은 descriptor-bound read-only connection 하나로 TEST `instance_index,label`을 읽기 전
Exp404/Exp291 equality gate와 output-byte snapshot/no-clobber를 검증한다. artifact는
`evaluation_antigravity.json`, `dataset_metrics_antigravity.jsonl`, `rescued_datasets_antigravity.jsonl`,
`regressed_datasets_antigravity.jsonl`, `confusion_antigravity.json`, `bootstrap_antigravity.json`,
`decision_antigravity.json`, `artifact_sha256_antigravity.json`다.

paired dataset-cluster bootstrap은 1,117개 dataset을 **20,000**회 resample하며 seed `43520260806`,
percentile 2.5%/97.5%를 쓴다. 성공은 candidate <=100, rescued dataset **10개 이상**, added false
positive **30개 이하**, paired micro 및 dataset-balanced macro F1 point delta 양수, 두 95% CI lower
양수를 모두 요구한다. 하나라도 실패하면 `FAILURE_LEDGER.json`, 성공이면 `CHAMPION_LEDGER.json`에
review-only로 기록하며 자동 경보·frozen constant는 바꾸지 않는다.
