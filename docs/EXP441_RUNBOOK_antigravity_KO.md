# Exp441 Hard-empty 단발 복구 실행 안내서

## 동결 원본

이 runbook은 다음 reviewed design 내용 SHA에 고정된다. SHA가 달라지면 아래
명령을 실행하지 말고 새 사전등록을 만든다.

```text
design_source = docs/superpowers/specs/2026-08-08-exp441-hard-empty-single-shot-recovery-antigravity-design.md
design_source_sha256 = c4d692297290869be90815d115c1fda7f71120326b4be504ebf685619b6545da
experiment_id = Exp441
run_identity = exp441-hard-empty-single-shot-recovery-antigravity-20260808
run_root = /Users/mac/Documents/Dataset/exp441_hard_empty_single_shot_recovery_antigravity_20260808
```

## 고정 산출물 경로

모든 경로는 위 run root 아래의 고정 절대 경로다. 경로, source, seed, budget,
backend, worker를 명령줄에서 바꾸는 옵션은 없다.

```text
/Users/mac/Documents/Dataset/exp441_hard_empty_single_shot_recovery_antigravity_20260808/readiness_antigravity.json
/Users/mac/Documents/Dataset/exp441_hard_empty_single_shot_recovery_antigravity_20260808/attempts_antigravity/attempt_0000_antigravity
/Users/mac/Documents/Dataset/exp441_hard_empty_single_shot_recovery_antigravity_20260808/attempts_antigravity/attempt_0000_antigravity/progress_antigravity.json
/Users/mac/Documents/Dataset/exp441_hard_empty_single_shot_recovery_antigravity_20260808/attempts_antigravity/attempt_0000_antigravity/events_antigravity.jsonl
/Users/mac/Documents/Dataset/exp441_hard_empty_single_shot_recovery_antigravity_20260808/attempts_antigravity/attempt_0000_antigravity/text_antigravity.log
/Users/mac/Documents/Dataset/exp441_hard_empty_single_shot_recovery_antigravity_20260808/attempts_antigravity/attempt_0000_antigravity/pid_antigravity
/Users/mac/Documents/Dataset/exp441_hard_empty_single_shot_recovery_antigravity_20260808/publication_antigravity
/Users/mac/Documents/Dataset/exp441_hard_empty_single_shot_recovery_antigravity_20260808/evaluation_antigravity
```

`progress_antigravity.json`에는 PID, 현재 dataset/record, 단계, backend, 완료/전체,
candidate 수, 처리속도, 경과시간, ETA, checkpoint와 최신 오류를 원자적으로 쓴다.
events와 text log는 같은 process가 소유한다. readiness와 inspect는 읽기 전용이며
prediction 명령은 SQLite와 TEST label-capable evaluator를 import하지 않는다.

## 정확히 고정된 네 명령

작업 디렉터리는 `/Users/mac/Documents/Time Series/.worktrees/exp438-frequency-complexity-evidence`로
고정한다. 아래 네 줄 외의 CLI 명령, flag, 경로 override는 제공하지 않는다.

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp441_hard_empty_single_shot_recovery_antigravity.py readiness
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp441_hard_empty_single_shot_recovery_antigravity.py new
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp441_hard_empty_single_shot_recovery_antigravity.py inspect
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp441_hard_empty_single_shot_recovery_antigravity.py evaluate
```

`readiness`는 reviewed design/source SHA, clean HEAD, fixed inputs, backend 상태와
CPU topology를 확인하고 readiness JSON을 한 번 만든다. `new`는 readiness를 다시
검증한 뒤 `$RUN_ROOT/attempts_antigravity`에서 **가장 작은 absent sequence**를
할당한다. 첫 attempt는 `attempt_0000_antigravity`이고, 빈 sequence가 있으면 그
sequence를 우선한다. `new`는 오래된 attempt를 resume하지 않는다. 중단·실패한
attempt와 staging은 변경 없이 보존하고 다음 `new`가 새 sequence에서 처음부터
검증한다. 성공한 attempt 하나만 `publication_antigravity`로 no-clobber 원자
승격된다. 별도 `resume` command는 없다.

`new` always allocates the smallest absent attempt sequence and never resumes an older attempt.

`inspect`는 publication과 source bytes, candidate identity, label access count를
재계산하지만 파일을 수정하지 않는다. `evaluate`만 publication/prediction SHA가
동결된 뒤 lazy evaluator를 import하고, DB descriptor hash와 `PRAGMA quick_check`
이후 frozen context 1,117개 dataset의 parameterized TEST `instance_index,label`
query를 정확히 한 번 수행한다. prediction 단계에는 label query가 없고 count는
정확히 0이어야 한다.

## publication/evaluation 의미

publication에는 prediction, candidate ranking, source audit, manifest, status,
artifact SHA map만 create-only로 기록한다. prediction은 dataset, instance_index와
label-free 근거만 포함한다. evaluation은 `evaluation_antigravity`,
`bootstrap_antigravity`, `decision_antigravity` 등 별도 산출물로 만들며 기존
Exp440 evaluation을 읽거나 덮어쓰지 않는다. 모든 governing artifact는 아래
flat governance block을 그대로 유지한다.

```text
official_run_purpose = deterministic_retrospective_reproduction
bootstrap_interpretation = descriptive_post_selection
adaptive_followup = true
independent_validation = false
external_generalization_supported = false
confirmatory_statistical_claim = false
scientific_winner_selected = false
review_only = true
operational_alert_eligible = false
```

어떤 기술적 재현 상태에서도 Exp404 Hard/Exp416 감사본은 검증 기준이고 Exp441은
CHAMPION_LEDGER의 `pending_candidate`나 champion으로 저장하지 않는다. TEST 결과를
본 뒤 선택한 후속 규칙이라는 사실은 바뀌지 않으므로 결과는 descriptive
post-selection evidence로만 보고한다.

## 실패 시 보존 규칙

source SHA, population, record digest, schema, 범위, symlink/regular-file 검사가
하나라도 실패하면 label을 열기 전에 종료한다. readiness 이후 HEAD/source 바이트가
바뀌면 실행하지 않는다. 실패 root, attempt, staging, progress/event/text/PID 로그는
감사 가능한 증거로 보존하며 publication/evaluation은 create-only라 기존 디렉터리를
덮어쓰지 않는다. 규칙이나 고정 identity 변경은 이 runbook 수정으로 우회하지 않고
새 실험 번호와 새 사전등록으로 처리한다.
