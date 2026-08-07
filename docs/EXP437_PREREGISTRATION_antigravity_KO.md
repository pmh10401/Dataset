# Exp437 메타데이터 왕복 수리 사전등록

이 문서는 Exp437의 과학 정본이다. 운영 명령과 고정 경로는
`docs/EXP437_RUNBOOK_antigravity_KO.md`에 있다. Exp437은 Exp436의
값·점수·임계값·lane·backend 선택·후보 규칙을 다시 선택하지 않는 얇은
successor이며, 변경은 실행 정체성과 canonical length-histogram transport에만
한정한다.

## 동결 신원과 현재 상태

다음 항목은 readiness 전에 고정하며 임의 인자나 commit에서 파생하지 않는다.

```text
Experiment: Exp437
Run identity: exp437-metadata-roundtrip-antigravity-20260807
Dataset population: 1,117 sorted unique datasets
Candidate alpha: 0.01
Candidate budget: 100
CPU workers: max(1, logical_cpus - 1), expected 11 on this Mac
Scientific delta from Exp436: none
Transport delta: canonical length-histogram validation ignores insertion order and normalizes lexicographically
Prediction label access: zero
Exp436 checkpoint reuse: forbidden
Evaluation: only after prediction, manifest, and artifact hashes are frozen
```

- 상태: `implementation_complete_readiness_pending`; prediction not started
- `prediction_phase_started: false`; `test_labels_opened: false`
- mode: `review_only`; `operational_alert_eligible: false`
- run parent: `/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807`
- readiness, prediction, publication, evaluation은 아직 생성하지 않는다.
- Exp436 failed root, readiness artifact hash, 47-record authority, 1 orphan
  record, 그리고 readiness/history hash는 읽기 전용 immutable history다. 삭제,
  수정, import, resume, checkpoint 재사용을 하지 않는다.

## Prediction boundary

이 절의 모든 scientific rule과 숫자는 Exp436의 `Prediction boundary`와
byte-for-byte 동일하다. Exp437은 다음의 label-free 입력만 사용한다.

### Frozen inputs (Exp436 값 unchanged)

- DB: `/Users/mac/Documents/Dataset/univariate_ts.db`
- DB SHA256: `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`
- Exp291 prediction SHA256: `73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848`
- Exp291 manifest SHA256: `761466253cccbb7b91fe36cdbd6d298de5d141c90e01e9fe646a8ae242ca9c3e`
- Exp292 prediction SHA256: `f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b`
- Exp292 manifest SHA256: `0672b2f84a3b2883628bac3c12ad8a25ebfb48efd929e4c0a448c5418267b0d0`
- Exp404 Hard comparator SHA256: `437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70`
- Exp404 Hard comparator: `/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl`

- `split='TRAIN' AND label='0'`인 normal TRAIN raw values
- TEST raw values와 dataset/instance identity, raw length, TRAIN/TEST count
- frozen 1,117-dataset population, four frozen lane과 frozen manifest

Prediction SHA256과 manifest SHA256이 얼기 전에는 TEST `label`, `labels_blob`,
anomaly position, TEST metric/F1/confusion, family별 과거 TEST 성능,
human-assisted 지표를 SELECT·open·전달·접근하지 않는다. TEST 결과로 alpha,
threshold, score source, backend, budget 또는 routing을 다시 고르지 않는다.
Exp404 Hard 좌표는 hash freeze 이후의 평가 비교에서만 읽는다. population 밖
dataset, 누락, 중복, tail replacement, 보간, padding, truncation, 값 복사는
실패다. Prediction label access는 정확히 zero다.

입력은 little-endian `float32`이고 original identity와 length를 보존한다. NaN,
`+Inf`, `-Inf`가 있는 normal TRAIN 행은 length reference에서 행 전체를 제외하고
dataset/instance/length/nonfinite-position-count/original values-blob SHA256
audit을 남긴다. finite normal TRAIN 행이 같은 length에 4개 이상이면 그것으로
calibration하고, 4개 미만이면 `insufficient_finite_train_rows`로 판단 보류한다.
nonfinite TEST 행은 `nonfinite_test` evidence와 함께 그 행만 판단 보류하며 finite
sibling은 계속 평가한다. 값 대체나 생성은 없으며 nonfinite score, fingerprint
change, checkpoint corruption, write failure는 전체 prediction 실패다.

Exp436의 실패한 47개 record나 orphan record를 읽어 결과로 쓰지 않는다. Exp437은
새 root에서 1,117개를 처음부터 계산한다. `Exp436 checkpoint reuse: forbidden`은
실행 코드·checkpoint reader·publication 모두에 적용된다.

## Frozen scientific rule

이 절은 Exp436의 `Frozen scientific rule`과 동일하며 숫자·수식·threshold·lane·
routing을 바꾸지 않는다.

- dtype `float32`, `candidate_alpha=0.01`, nearest normal `k=3`, TRAIN
  leave-one-row-out
- quantile `q=0.99`, `method='higher'`, channel abnormal은 strict
  `score > threshold`
- shared robust transform: `center=median(matrix)`,
  `scale=float32(1.4826 * median(abs(matrix-center)))`, scale floor `1e-12`
- prototype/raw와 gradient는 각자의 TRAIN-group matrix로 transform을 계산하고,
  ACF는 이미 동결한 Exp436 formula를 그대로 사용한다.
- distance는 `sqrt(mean((x-reference)**2))`, smallest 3 distance median이다.
- gradient는 `diff(values, prepend=values[0])`; ACF lag는
  `1..min(32, floor(length/4))`, 최소 length 16, variance floor `1e-12`다.

후보는 아래 조건을 모두 만족할 때만 허용된다. 네 frozen lane은 Hard,
Standard, Priority, automatic-normal이며, channel 하나라도 false 또는
unavailable이면 candidate가 아니다. 후보는 `(dataset, instance_index)`로
정렬·고유다.

```text
prototype_abnormal
AND gradient_abnormal
AND acf_abnormal
AND outside_all_four_frozen_lanes
```

`candidate_alpha=0.01`, quantile/threshold, k, lane set, seed, bootstrap,
rescue, false-positive limit과 score source에는 Exp436 대비 scientific delta가
없다. TEST 결과로 이 값을 바꾸지 않는다.

## Backend readiness

Exp436의 `Backend readiness` 계약과 numeric values는 unchanged다. readiness는
현재 clean HEAD와 frozen inputs를 bind하고 MLX와 PyTorch-MPS를 각각 같은
TRAIN-only smoke로 평가하되, CPU fallback을 기본 안전 경로로 둔다. Exp437의
metadata repair는 score computation을 바꾸지 않으며 MLX/MPS를 주장하기 위해
official CPU transformer를 다시 쓰지 않는다.

- CPU worker: `max(1, (os.cpu_count() or 1) - 1)`; 이 Mac은 12 logical CPU라서
  expected 11 workers; worker당 numerical-library thread는 1이다.
- smoke seed: 정확히 `43601`; batch sizes `(16, 32, 64, 128, 256)`.
- 각 batch는 untimed warm-up 정확히 2회와 transfer/synchronization을 포함한
  timed paired CPU/device run 정확히 5회다. device synchronization은 timed
  callable 안에서 수행하고 device median은 5회 결과의 median이다.
- finite output, preregistered candidate coordinate CPU 동일성, max absolute
  error와 normalized score error 각각 `<=1e-5`, transfer/synchronization 포함
  `>=1.1x` speedup을 모두 만족한 batch만 통과한다.
- backend 안에서는 passing batch 중 device median이 가장 낮은 것을 고르고,
  tie는 더 작은 batch다. MLX와 MPS가 모두 통과하면 더 낮은 median을 고르고
  exact tie는 lexical backend name (`mlx` before `mps`)이다. 둘 다 실패하면
  CPU다. CPU fallback은 유효한 결과이며 현재 예상 선택이다.
- MLX/MPS-heavy process는 하나뿐이고 채택 시 outer process pool을 쓰지 않는다.
  manifest에는 MLX/mlx-metal/version/device, seed, warm-up/timed-run count,
  batch medians, parity, selection과 fallback reason을 남긴다.

이 평가는 실행 profile이나 transport repair를 이유로 scientific value를
재선택하지 않는다. MLX/MPS parity·속도 gate가 하나라도 실패하면 existing CPU
implementation을 유지한다.

## Candidate budget

Exp436의 `Candidate budget`와 값은 unchanged다. global candidate budget은
**100**이다. 101번째 candidate는 `candidate_budget_exceeded` 전체 prediction
실패이며 top-N 절단, threshold 변경, source 교체, rescue 또는 false-positive
재설정은 금지한다. 후보 0..100개만 canonical review-only prediction freeze로
진행할 수 있다.

## Transport-only repair

Exp437의 유일한 implementation/scientific-boundary 밖 변경은 canonical
metadata transport다. Exp436의 length histogram 의미를 그대로 유지하되 JSON
object-key insertion order에 의존하지 않는다.

- histogram key는 positive canonical decimal length string이다(`"1"`, `"29"`,
  `"102"`; leading zero, unknown key, label-like key는 거부).
- histogram value는 nonnegative integer이고, values의 합은 TRAIN 또는 TEST
  row count와 정확히 같다.
- checkpoint record, progress snapshot, completion event를 읽을 때 caller
  insertion order를 요구하지 않는다.
- 허용한 map은 `dict(sorted(histogram.items()))`로 lexicographic key order로
  normalize한 뒤 canonical JSON으로 쓴다. 따라서 numeric order `29, 102`와
  canonical JSON order `102, 29`가 같은 의미로 왕복된다.

이 lexical histogram normalization 외에는 feature, score, alpha, threshold,
lane, budget, seed, bootstrap, rescue, false-positive gate, routing, population,
or information boundary가 바뀌지 않는다. No scientific delta from Exp436.

## 실행, root, progress와 immutable failure

readiness가 previously absent 정확한 `RUN_PARENT`를 원자적으로 생성·소유하고
그 아래 `readiness_antigravity.json`을 남긴다. `new`는 같은 existing
`RUN_PARENT`와 exact verified readiness artifact를 요구하며
`output_antigravity`, `checkpoint_antigravity`, `evaluation_antigravity`가 모두
부재일 때만 허용한다. readiness와 `new`는 root나 readiness를 덮어쓰거나 지우지
않는다. 모든 새 runtime artifact filename은 `antigravity`를 포함한다.

runtime은 다음을 기록한다.

- `output_antigravity/progress_antigravity.json`
- `output_antigravity/events_antigravity.jsonl`
- `output_antigravity/parent_antigravity.log`
- `output_antigravity/run_antigravity.pid`
- ordered canonical checkpoint records under `checkpoint_antigravity`

PID, heartbeat, current dataset, TRAIN/TEST counts·lengths, backend,
completed/total, success/abstention/error, nonfinite counts, candidate count,
rate, elapsed, ETA, checkpoint, latest error가 필수다. persisted stage는
`loading`, `dispatching`, `awaiting_worker_result`, `committing`, `idle`뿐이다.

`resume`은 **interrupted Exp437 root에만** 합법적이다. frozen identity/input/
progress binding을 read-only 검증한 뒤에만 resume하며, complete/inspect는 DB나
worker를 열지 않는 read-only publication replay다. `failed` root는 immutable이고
input/PID/progress/event mutation, temp cleanup, 삭제, 덮어쓰기, resume를 모두
금지한다. failed root는 새 experiment number로 새 run parent를 사용해야 한다.
Exp436 failed root에도 동일한 immutable rule을 적용한다.

## Post-freeze evaluation

이 절의 `Post-freeze evaluation` rule은 Exp436과 unchanged다. prediction은
`review_only=true`, `operational_alert_eligible=false`이며 completion은 정확히
1,117 processed, missing/duplicate/out-of-population/failed dataset 0,
candidate `<=100`, inactive PID, `current_dataset=null`, TEST label access 0,
checkpoint/event/summary/prediction 및 SHA256 재검증 일치를 요구한다. 그 뒤에만
prediction, manifest, artifact hashes를 freeze한다.

publication은 `train_exclusion_audit_antigravity.jsonl`을 포함하고 manifest와
`artifact_sha256_antigravity.json`이 그 hash를 bind한다. evaluation은 output/
publication bytes와 Exp404/Exp291 equality gate를 검증한 뒤 descriptor-bound
read-only connection 하나로 TEST `instance_index,label`을 읽는다. TEST label은
이 시점 이전에는 정확히 zero다. evaluation artifact는 다음과 같다.

- `evaluation_antigravity.json`
- `dataset_metrics_antigravity.jsonl`
- `rescued_datasets_antigravity.jsonl`
- `regressed_datasets_antigravity.jsonl`
- `confusion_antigravity.json`
- `bootstrap_antigravity.json`
- `decision_antigravity.json`
- `artifact_sha256_antigravity.json`

paired dataset-cluster bootstrap은 1,117개 dataset을 **20,000**회 resample하며
seed `43520260806`, percentile 2.5%/97.5%를 쓴다. success에는 candidate `<=100`,
rescued dataset **10개 이상**, added false positive **30개 이하**, paired micro와
dataset-balanced macro F1 point delta 양수, 두 95% CI lower 양수가 모두 필요하다.
하나라도 실패하면 `FAILURE_LEDGER.json`, 성공하면 `CHAMPION_LEDGER.json`에
review-only로 기록하며 automatic alert나 frozen constant를 바꾸지 않는다.

Evaluation: only after prediction, manifest, and artifact hashes are frozen.
Exp437 prediction has not started and no TEST label has been opened in this
implementation-complete/readiness-pending state.
