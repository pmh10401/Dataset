# Exp436 결측 안전 엄격 3중 합의 실행 안내서

## 권한과 현재 상태

이 문서는 운영 정본이며 과학 상수·정보 경계는
`docs/EXP436_PREREGISTRATION_antigravity_KO.md`가 정본이다. 현재 product implementation 전이므로
readiness, prediction, evaluation, TEST label access를 시작하지 않는다. 구현과 focused test 완료 뒤
clean HEAD에서만 다음 wrapper를 쓴다.

```bash
cd '/Users/mac/Documents/Time Series/.worktrees/exp425-selective-recovery'
python3 scratch/run_exp436_missing_safe_strict_consensus_antigravity.py --help
```

허용 모드는 `readiness`, `new`, `resume`, `inspect`, `evaluate` 다섯 개뿐이다. run identity는
`exp436-missing-safe-strict-consensus-antigravity-20260806`이고 commit에서 파생하지 않는다.

## 고정 경로, 해시, artifact

```text
DB=/Users/mac/Documents/Dataset/univariate_ts.db
RUN_PARENT=/Users/mac/Documents/Dataset/exp436_missing_safe_strict_consensus_antigravity_20260806
OUTPUT=/Users/mac/Documents/Dataset/exp436_missing_safe_strict_consensus_antigravity_20260806/output_antigravity
CHECKPOINT=/Users/mac/Documents/Dataset/exp436_missing_safe_strict_consensus_antigravity_20260806/checkpoint_antigravity
READINESS=/Users/mac/Documents/Dataset/exp436_missing_safe_strict_consensus_antigravity_20260806/readiness_antigravity.json
EVALUATION=/Users/mac/Documents/Dataset/exp436_missing_safe_strict_consensus_antigravity_20260806/evaluation_antigravity
```

DB SHA256은 `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`다. Exp291
prediction/manifest SHA256은 `73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848` /
`761466253cccbb7b91fe36cdbd6d298de5d141c90e01e9fe646a8ae242ca9c3e`, Exp292는
`f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b` /
`0672b2f84a3b2883628bac3c12ad8a25ebfb48efd929e4c0a448c5418267b0d0`, Exp404 comparator는
`437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70`다.

publication은 `prediction_antigravity.jsonl`, `dataset_summary_antigravity.jsonl`,
`score_evidence_antigravity.jsonl`, `abstention_ledger_antigravity.jsonl`,
`backend_decision_antigravity.json`, `manifest_antigravity.json`, `status_antigravity.json`,
`artifact_sha256_antigravity.json`이다. evaluation은 `evaluation_antigravity.json`,
`dataset_metrics_antigravity.jsonl`, `rescued_datasets_antigravity.jsonl`,
`regressed_datasets_antigravity.jsonl`, `confusion_antigravity.json`, `bootstrap_antigravity.json`,
`decision_antigravity.json`, `artifact_sha256_antigravity.json`만 만든다.

## 구현 후 readiness

```bash
git status --short --branch
git rev-parse HEAD
python3 -m unittest discover -s experiments/exp436_missing_safe_strict_consensus_antigravity/tests -p 'test_*_antigravity.py'
python3 -m unittest experiments.exp425_selective_recovery.tests.test_database -v
python3 -m unittest discover -s experiments/exp435_train_only_consensus_recovery_antigravity/tests -p 'test_*_antigravity.py'
python3 -m unittest discover -s experiments/exp426_verified_baseline/tests -p 'test_*_antigravity.py'
python3 -m unittest discover -s experiments/exp417_causal_multihorizon_shadow/tests -p 'test_*.py'
git diff --check
```

`new`는 **RUN_PARENT가 부재일 때만** 허용한다. 따라서 existing/failed/complete parent를 지우거나
덮어쓰지 않는다. readiness는 그 직전의 비어 있는 parent를 한 번 만든 뒤 output/checkpoint/readiness/
evaluation fixed root 모두가 없을 때만 수행한다.

```bash
mkdir /Users/mac/Documents/Dataset/exp436_missing_safe_strict_consensus_antigravity_20260806
python3 scratch/run_exp436_missing_safe_strict_consensus_antigravity.py readiness
```

readiness는 clean HEAD, exact hash, **1,117** (`1117`) population, TEST label/`labels_blob` access 0,
candidate fixture와 root absence를 기록한다. CPU는 `max(1,(os.cpu_count() or 1)-1)` workers(현재 12
logical CPU에서 11)와 worker당 numerical thread 1이다. MLX/PyTorch-MPS exact dense-distance smoke는
finite output, candidate coordinate CPU identity, absolute/normalized error 각각 `<=1e-5`, transfer 및
synchronization 포함 `>=1.1x` speedup을 모두 만족할 때만 채택한다. accelerator 하나만 heavy process로
실행하며 채택 시 outer pool을 쓰지 않는다. MLX/mlx-metal/device/version/timing/parity/fallback reason은
manifest에 남긴다. 실패 readiness 뒤 `new`를 실행하지 않는다.

## label-free new와 진행

```bash
python3 scratch/run_exp436_missing_safe_strict_consensus_antigravity.py new
```

prediction은 normal TRAIN values와 label-free TEST values만 쓴다. TEST label, `labels_blob`, anomaly
position, TEST metric, family performance는 열지 않는다. nonfinite normal TRAIN 행은 original blob SHA256
audit을 남기고 제외하며 finite 행 4개 미만 length는 `insufficient_finite_train_rows`다. nonfinite TEST
행은 `nonfinite_test` evidence로 그 행만 abstain하고 finite sibling은 계속한다. imputation/padding/
truncation/copy/tail replacement는 없다.

점수는 float32, `k=3`, TRAIN LOO, `q=0.99`, `method='higher'`, strict `score > threshold`다. 후보는
`prototype AND gradient AND ACF` 그리고 four frozen lane 밖일 때만 생긴다. unavailable channel은 true가
아니다. budget은 **100**, 101번째는 top-N 절단 없이 `candidate_budget_exceeded` 전체 실패다. 항상
review-only이며 operational alert가 아니다.

```bash
python3 -m json.tool /Users/mac/Documents/Dataset/exp436_missing_safe_strict_consensus_antigravity_20260806/output_antigravity/progress_antigravity.json
tail -n 40 /Users/mac/Documents/Dataset/exp436_missing_safe_strict_consensus_antigravity_20260806/output_antigravity/parent_antigravity.log
```

progress는 PID/heartbeat/current dataset/TRAIN·TEST count 및 length/backend/stage/completed-total/
success-abstention-error/nonfinite count/candidate/rate/elapsed/ETA/checkpoint/latest error를 표시한다.
persisted stage는 `loading`, `dispatching`, `awaiting_worker_result`, `committing`, `idle`뿐이다.

## 중단, resume, inspect

```bash
python3 scratch/run_exp436_missing_safe_strict_consensus_antigravity.py resume
python3 scratch/run_exp436_missing_safe_strict_consensus_antigravity.py inspect
```

두 번째 writer는 active PID가 있으면 시작하지 않는다. `interrupted` root만 frozen binding을 read-only로
검증 뒤 resume한다. `failed` root(101st candidate, fingerprint, score, checkpoint/write failure 포함)는
input/PID/progress/event mutation 또는 temp cleanup 전에 read-only 거절하며 수정·삭제·재개하지 않는다.
complete resume/inspect는 DB나 worker를 열지 않는 read-only publication replay다. inspect 완료 조건은
`complete_review_only`, completed=1,117, errors=0, candidate=0..100, inactive PID,
`current_dataset=null`, TEST label access=0, prediction/manifest/artifact SHA256 재검증 일치다.

## 동결 뒤 evaluate와 종료

RESEARCH_STATE/EXPERIMENT_CATALOG에 prediction/manifest/artifact SHA256과 1,117 coverage가 동결되고
inspect가 성공한 뒤에만 평가한다.

```bash
python3 scratch/run_exp436_missing_safe_strict_consensus_antigravity.py evaluate
```

evaluator는 publication, Exp404/Exp291 Hard equality gate, output byte snapshot/no-clobber를 확인한 뒤
descriptor-bound read-only connection 하나로 TEST `instance_index,label`을 읽는다. paired dataset-cluster
bootstrap은 **20,000**회, seed `43520260806`, percentile 2.5%/97.5%다. success는 candidate <=100,
rescued dataset **10개 이상**, added FP **30개 이하**, micro와 dataset-balanced macro F1 point delta 양수,
두 95% CI lower 양수를 모두 만족할 때다. 모든 regression/strong baseline(F1>=0.8) regression을 공개하고,
실패는 `FAILURE_LEDGER.json`, 성공은 `CHAMPION_LEDGER.json`에 보존하며 automatic alert로 승격하지 않는다.
