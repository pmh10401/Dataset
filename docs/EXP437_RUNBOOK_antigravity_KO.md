# Exp437 메타데이터 왕복 수리 실행 안내서

이 문서는 Exp437 운영 정본이다. 과학 상수와 정보 경계는
`docs/EXP437_PREREGISTRATION_antigravity_KO.md`가 정본이며 Exp436과의
scientific delta는 없다. 이 worktree의 implementation은 완료되었고 현재
상태는 `implementation_complete_readiness_pending`; prediction not started,
`test_labels_opened=false`다.

## 고정 실행 identity와 경로

모든 명령은 clean HEAD의 다음 worktree에서 실행한다. wrapper는 outside-CWD에도
동작하지만 runbook은 이 고정 경로를 먼저 명시한다.

```text
WORKTREE=/Users/mac/Documents/Time Series/.worktrees/exp425-selective-recovery
EXPERIMENT=Exp437
RUN_IDENTITY=exp437-metadata-roundtrip-antigravity-20260807
DB=/Users/mac/Documents/Dataset/univariate_ts.db
RUN_PARENT=/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807
READINESS=/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807/readiness_antigravity.json
OUTPUT=/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807/output_antigravity
CHECKPOINT=/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807/checkpoint_antigravity
PUBLICATION=/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807/output_antigravity/publication_antigravity
EVALUATION=/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807/evaluation_antigravity
```

실험 population은 sorted unique **1,117** datasets다. candidate alpha는 `0.01`,
candidate budget은 **100**, CPU는 `max(1, logical_cpus - 1)`이며 이 Mac의
expected topology는 12 logical CPU/11 workers다. prediction은 normal TRAIN과
label-free TEST raw values만 사용하고 TEST label access는 zero다. Exp436
checkpoint reuse는 forbidden이다. 새 artifact filename에는 `antigravity`가
들어간다.

## 정확한 고정 명령

아래 명령과 redirection path를 변경하지 않는다.

```bash
cd '/Users/mac/Documents/Time Series/.worktrees/exp425-selective-recovery'
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp437_metadata_roundtrip_antigravity.py readiness
PYTHONDONTWRITEBYTECODE=1 nohup python3 scratch/run_exp437_metadata_roundtrip_antigravity.py new \
  > /Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807/launcher_antigravity.log 2>&1 &
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp437_metadata_roundtrip_antigravity.py inspect
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp437_metadata_roundtrip_antigravity.py resume
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp437_metadata_roundtrip_antigravity.py evaluate
```

허용 command는 `readiness`, `new`, `resume`, `inspect`, `evaluate` 다섯 개뿐이며
path, worker, backend, seed, threshold, alpha, lane, budget override flag는 없다.
`new`는 readiness의 **same existing** `RUN_PARENT`와 exact verified
`readiness_antigravity.json`을 요구한다. output/checkpoint/evaluation fixed root가
이미 있으면 새 실행을 시작하지 않는다.

## readiness와 backend 기록

readiness는 previously absent 정확한 run parent를 원자적으로 생성·소유하고
`readiness_antigravity.json`을 기록한다. clean HEAD, frozen DB/input hashes,
sorted unique population, zero TEST-label access, four lanes, fixed roots absence,
and the Exp437 execution identity를 확인한다.

Exp436 `Backend readiness`와 numeric values는 그대로다. smoke seed는 `43601`,
batch sizes는 `(16, 32, 64, 128, 256)`, batch별 warm-up은 2회, timed paired
CPU/device run은 5회다. finite output, exact candidate coordinate parity,
absolute/normalized error 각각 `<=1e-5`, transfer/synchronization 포함 speedup
`>=1.1x`를 모두 만족할 때만 MLX/MPS batch를 통과시킨다. backend tie는 lower
device median, then smaller batch, cross-backend exact tie는 lexical `mlx` before
`mps`다. 두 accelerator가 실패하면 CPU fallback이며 기존 CPU implementation을
유지한다. MLX/MPS-heavy process는 하나뿐이고 CPU worker는 11 expected다.

readiness 실패는 `new`를 허용하지 않는다. readiness 또는 이후 artifact에서
Exp436 identity/root/checkpoint가 보이면 Exp437 readiness를 실패시킨다.

## prediction, logs와 고정 artifact 목록

`new`는 새 output/checkpoint를 만들고 1,117개 dataset을 처음부터 계산한다. Exp436
47-record authority나 orphan record를 import하지 않는다. lexical histogram
transport repair는 positive decimal key/nonnegative integer/count-sum 의미를
검증한 뒤 `dict(sorted(histogram.items()))`으로 normalize할 뿐, score/threshold/
candidate/routing을 바꾸지 않는다. prediction은 항상
`review_only=true`, `operational_alert_eligible=false`다.

운영자가 확인해야 할 파일은 다음과 같다.

- progress JSON: `$OUTPUT/progress_antigravity.json`
- event JSONL: `$OUTPUT/events_antigravity.jsonl`
- parent log: `$OUTPUT/parent_antigravity.log`
- PID file: `$OUTPUT/run_antigravity.pid`
- checkpoint directory: `$CHECKPOINT` (ordered canonical records)
- publication directory: `$PUBLICATION`
- evaluation directory: `$EVALUATION`

progress에는 PID, heartbeat, current dataset, TRAIN/TEST counts·lengths, backend,
stage, completed/total, success/abstention/error, nonfinite counts, candidate count,
rate, elapsed, ETA, checkpoint, latest error가 있어야 한다. persisted stage는
`loading`, `dispatching`, `awaiting_worker_result`, `committing`, `idle`만 허용한다.

publication이 만들어질 때 `prediction_antigravity.jsonl`,
`dataset_summary_antigravity.jsonl`, `score_evidence_antigravity.jsonl`,
`abstention_ledger_antigravity.jsonl`, `train_exclusion_audit_antigravity.jsonl`,
`backend_decision_antigravity.json`, `manifest_antigravity.json`,
`status_antigravity.json`, `artifact_sha256_antigravity.json`를 bind한다. 모든
prediction/manifest/artifact hash는 post-freeze evaluation 전에 immutable하다.

## 중단, resume, inspect와 immutable failure

`resume`은 **interrupted Exp437 root에만 legal**하다. 먼저 frozen identity, input
hashes, progress binding, PID/event/checkpoint authority를 read-only로 검증하고
그 뒤에만 재개한다. Exp437의 `checkpoint_complete` root도 `resume` 대상이
아니며 실행·publication·mutation 전에 거절한다. 오직 `inspect`만 DB와 worker를
열지 않는 read-only publication replay다.

`failed` root는 immutable이다. candidate budget 초과, score/fingerprint 오류,
checkpoint/write failure, metadata corruption, label boundary 위반 등 어떤
실패도 failed root의 input/PID/progress/event를 수정하거나 temp를 지우거나
삭제·덮어쓰기·resume하지 않는다. failed Exp437은 반드시 **새 experiment number**
와 새 fixed run parent로 다시 사전등록한다. Exp436 failed root는 47-record
authority, one orphan, readiness hash와 함께 같은 방식으로 영구 보존한다.

## Post-freeze evaluation

`inspect`가 1,117 completed, errors 0, candidate `<=100`, inactive PID,
`current_dataset=null`, zero TEST-label access, checkpoint/event/summary/prediction
hash equality를 확인하고 prediction/manifest/artifact hashes를 고정한 뒤에만
`evaluate`를 실행한다. 평가 전 Exp404/Exp291 equality gate와 output-byte
snapshot/no-clobber를 확인하고 descriptor-bound read-only SQLite connection 하나로
TEST `instance_index,label`을 연다. 이 시점 전에는 TEST label access가 정확히
zero다.

평가는 1,117 dataset의 paired cluster bootstrap 20,000회, seed `43520260806`,
percentile 2.5%/97.5%를 사용한다. candidate `<=100`, rescued dataset `>=10`,
added false positive `<=30`, paired micro와 dataset-balanced macro F1 delta > 0,
두 95% CI lower > 0을 모두 요구하며, 실패는 `FAILURE_LEDGER.json`, 성공은
`CHAMPION_LEDGER.json`에 review-only로 남긴다. evaluation은 frozen scientific
values를 다시 선택하거나 automatic alert로 승격하지 않는다.

## 상태 handoff

이 문서가 동결하는 상태는 `implementation_complete_readiness_pending`이고
prediction not started다. Task 7의 verified commit이 생기기 전에는 final
implementation commit을 state/catalog에 기록하지 않는다. readiness는 clean HEAD
review와 exact command validation을 통과한 뒤에만 실행한다.
