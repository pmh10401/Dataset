# Exp438 주파수·복잡도 근거 실행 안내서

과학 규칙은 `docs/EXP438_PREREGISTRATION_antigravity_KO.md`가 정본이다. 이
문서는 고정 명령, 경로, 재개와 확인 방법만 설명한다. 현재 구현과 병행 실행
보완은 완료됐다. Exp437 stale-PID 복구 preflight가 통과했으며, 정확한 기존
runtime API로 Exp437를 복구한 뒤 살아 있는 CPU 실행을 확인하고 Exp438을
병행 시작한다.

## 고정 경로

```text
WORKTREE=/Users/mac/Documents/Time Series/.worktrees/exp438-frequency-complexity-evidence
DB=/Users/mac/Documents/Dataset/univariate_ts.db
RUN_PARENT=/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807
READINESS=/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807/readiness_antigravity.json
OUTPUT=/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807/output_antigravity
CHECKPOINT=/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807/checkpoint_antigravity
PUBLICATION=/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807/publication_antigravity
EVALUATION=/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807/evaluation_antigravity
LAUNCHER_LOG=/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807/launcher_antigravity.log
EXP437_RECOVERY_LOG=/Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807/recovery_launcher_antigravity.log
```

population은 sorted unique 1,117개다. CPU worker는 항상
`max(1, logical_cpus - 1)`이고 현재 Mac에서는 11개, worker당 numerical thread는
1개다. backend는 readiness 전에는 **미정**이다.

## 실행 전 필수 gate

다음을 모두 만족해야 readiness를 실행할 수 있다.

1. Exp437가 terminal/inactive이거나, `running` CPU·실제 PID 생존·heartbeat
   300초 이내·오류 0·TEST-label 접근 0을 모두 만족한다.
2. 병행 상태라면 Exp437 identity/root/total이 고정값이고 progress와 checkpoint
   카운터가 모순되지 않는다.
3. stale PID 복구는 exact clean commit
   `cc0f507f290dc7f613763d3653498f8f54cad88a`의 기존 공개 runtime API만 사용한다.
4. 이 worktree가 clean HEAD다.
5. 고정 Exp438 run parent가 아직 없다.
6. DB, 네 lane, source와 population hash가 동결값과 같다.
7. 사용자 병행 승인이 동결되어 있다.

하나라도 실패하면 run parent를 만들거나 readiness/backend benchmark를 실행하지
않는다. Exp438 readiness/prediction은 Exp437 root를 읽기만 한다. 별도 복구
런처만 Exp437 기존 owner API를 호출하며 파일을 직접 수정·삭제하지 않는다.

## Exp437 복구 명령

```bash
cd '/Users/mac/Documents/Time Series/.worktrees/exp438-frequency-complexity-evidence'
python3 operational/recover_exp437_checkpoint_antigravity.py --preflight

nohup python3 operational/recover_exp437_checkpoint_antigravity.py \
  > /Users/mac/Documents/Dataset/exp437_metadata_roundtrip_antigravity_20260807/recovery_launcher_antigravity.log \
  2>&1 &
```

launcher log와 shell PID는 `output_antigravity` 밖의 run parent에 둔다. Exp437
output은 엄격한 네 파일 inventory이므로 외부 로그를 넣으면 복구가 거부된다.

## 정확한 다섯 명령

```bash
cd '/Users/mac/Documents/Time Series/.worktrees/exp438-frequency-complexity-evidence'

PYTHONDONTWRITEBYTECODE=1 python3 \
  scratch/run_exp438_frequency_complexity_evidence_antigravity.py readiness

PYTHONDONTWRITEBYTECODE=1 nohup python3 \
  scratch/run_exp438_frequency_complexity_evidence_antigravity.py new \
  > /Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807/launcher_antigravity.log 2>&1 &

PYTHONDONTWRITEBYTECODE=1 python3 \
  scratch/run_exp438_frequency_complexity_evidence_antigravity.py inspect

PYTHONDONTWRITEBYTECODE=1 python3 \
  scratch/run_exp438_frequency_complexity_evidence_antigravity.py resume

PYTHONDONTWRITEBYTECODE=1 python3 \
  scratch/run_exp438_frequency_complexity_evidence_antigravity.py evaluate
```

path, workers, backend, batch, seed, depth, threshold, alpha, lane 또는 budget을
바꾸는 옵션은 없다.

## readiness가 확인하고 기록하는 것

readiness는 absent run parent를 원자적으로 생성하고 그 안에 한 개의 canonical
`readiness_antigravity.json`을 쓴다. clean reviewed commit, DB SHA256, source
path/hash, 1,117 dataset order, 네 lane, fixed config, Exp437 terminal/inactive
또는 fresh live CPU snapshot, 집중 테스트와 prediction label access 0을 다시
확인한다. 병행 snapshot에는 PID, OS 생존, backend, heartbeat age, progress와
checkpoint 카운터, 오류, publication/evaluation 존재 여부가 남는다.

backend smoke는 seed `43801`, batch `(16, 32, 64, 128, 256)`, warm-up 2회,
paired timing 5회다. Numba compile amortization과 MLX/MPS transfer·synchronize를
포함한다. finite output, candidate identity, score 오차 `<=1e-5`, speedup
`>=1.1x`를 모두 통과한 정확한 stage만 선택한다. readiness에는 Python, NumPy,
Numba, MLX/mlx-metal, PyTorch 버전, device, batch별 시간, parity, 선택 결과와
fallback 이유가 남는다. CPU 선택은 정상이다. 실제 readiness 전에는 문서나
manifest에 backend가 선택됐다고 쓰지 않는다.

## 공식 실행과 직접 확인할 로그

`new`는 exact readiness와 Exp437의 현재 상태를 다시 검증한 뒤 새
output/checkpoint를 만든다. Exp437 카운터 전진, 새로 복구된 생존 PID 또는 정상
terminal 전환만 허용한다. CPU가
선택되면 11 dataset workers를 사용하고 parent가 frozen order대로만 commit한다.
Numba/MLX/MPS exact stage가 선택되면 사전등록 scheduler 규칙에 따라 단일
accelerator 경로를 사용한다.

사용자가 직접 볼 파일은 다음이다.

```text
/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807/output_antigravity/progress_antigravity.json
/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807/output_antigravity/events_antigravity.jsonl
/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807/output_antigravity/parent_antigravity.log
/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807/output_antigravity/run_antigravity.pid
/Users/mac/Documents/Dataset/exp438_frequency_complexity_evidence_antigravity_20260807/launcher_antigravity.log
```

progress JSON은 PID, heartbeat, 현재 dataset, TRAIN/TEST 행 수·길이·값 수·bytes,
stage, backend, completed/total, success/abstention/error, 처리속도, 경과시간, ETA,
checkpoint, 채널별 후보 수와 최신 오류를 원자적으로 기록한다. heartbeat는
프로세스가 직접 쓰므로 Codex가 반복 감시하지 않는다.

## 중단·재개·실패

- `inspect`: DB와 worker를 열지 않는 read-only 상태/publication replay다.
- `resume`: PID가 비활성인 interrupted root만 허용한다. identity, config, DB,
  readiness, checkpoint ordered prefix, progress/event binding이 모두 같아야 한다.
- complete, checkpoint-complete 또는 failed root는 resume하지 않는다.
- failed root는 immutable이다. 파일 수정, temp 정리, 삭제, 덮어쓰기, 재시작을
  하지 않는다. 다음 시도는 새 실험 번호와 새 run parent를 사전등록한다.
- 후보 budget, nonfinite output, hash mismatch, label boundary, worker/checkpoint
  오류는 fail closed한다. top-N 절단이나 같은 실험의 재조정은 없다.

## publication과 평가 순서

공식 실행이 끝난 뒤 먼저 `inspect`한다. 정확히 1,117 completed, missing/extra/
failed 0, inactive PID, `current_dataset=null`, budget 준수, label access 0,
checkpoint/event/summary/prediction/manifest/hash 일치가 확인되어야 publication을
동결할 수 있다.

publication에는 다음 8개 파일이 있다.

```text
prediction_antigravity.jsonl
dataset_summary_antigravity.jsonl
abstention_ledger_antigravity.jsonl
train_exclusion_audit_antigravity.jsonl
backend_decision_antigravity.json
manifest_antigravity.json
status_antigravity.json
artifact_sha256_antigravity.json
```

그 뒤에만 `evaluate`한다. 평가기는 Exp438 publication/output/checkpoint,
Exp437 frozen publication, Exp404/Exp291 Hard equality, DB와 provenance를 먼저
검증하고 그 후 TEST label을 한 번 연다. frozen source bytes는 평가 전후 같아야
한다.

evaluation에는 다음 8개 파일이 원자적 no-clobber 방식으로 생긴다.

```text
evaluation_antigravity.json
dataset_metrics_antigravity.jsonl
rescued_datasets_antigravity.jsonl
regressed_datasets_antigravity.jsonl
confusion_antigravity.json
bootstrap_antigravity.json
decision_antigravity.json
artifact_sha256_antigravity.json
```

세 채널 결과를 모두 보고하며 Exp438 안에서 winner를 고르지 않는다. 기존
evaluation root가 있거나 hash replay가 실패하면 덮어쓰지 않는다.

## 현재 handoff

현재 상태는 `implementation_complete_concurrent_readiness_pending`이다. Exp437
복구 preflight는 진행 938, event 938, 공식 checkpoint 938, intent와 함께 남은
물리 record 939, 오류 0, label 접근 0으로 통과했다. 아직 두 복구/공식 프로세스는
시작하지 않았다. 위 복구 명령으로 Exp437를 살리고 한 번 생존 확인한 뒤 정확한
Exp438 `readiness`와 `new`를 순서대로 실행한다.
