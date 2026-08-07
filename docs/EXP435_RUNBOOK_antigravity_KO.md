# Exp435 실행 안내서

## 1. 목적

Exp435는 정상 TRAIN으로 만든 대표 패턴·기울기·주기 점수의 합의로 동결된 1,117개
데이터셋, 총 195,900개 TEST row를 label-free 평가한다. 새 후보는 검토용이며
기존 Hard 경보를 바꾸지 않는다.

모든 명령은 다음 clean worktree에서 실행한다.

```bash
cd '/Users/mac/Documents/Time Series/.worktrees/exp425-selective-recovery'
```

실행 파일은 구현 후 다음 경로에 있다.

```text
scratch/run_exp435_train_only_consensus_recovery_antigravity.py
```

직접 실행 wrapper는 저장소 밖의 현재 디렉터리에서도 저장소 root를 자동으로
찾는다. 구현 완료 후 다음 도움말에는 정확히 다섯 모드가 보인다.

```bash
python3 scratch/run_exp435_train_only_consensus_recovery_antigravity.py --help
```

`readiness`, `new`, `resume`, `inspect`, `evaluate` 외의 모드는 없다.

## 2. 고정 경로

```text
DB=/Users/mac/Documents/Dataset/univariate_ts.db
EXP291_PREDICTION=/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl
EXP291_MANIFEST=/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_prediction_manifest.json
EXP292_PREDICTION=/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/exp292_frozen_review_tiers.jsonl
EXP292_MANIFEST=/Users/mac/Documents/Dataset/exp292_standard_review_evidence_tiering/exp292_prediction_manifest.json
RUN_PARENT=/Users/mac/Documents/Dataset/exp435_train_only_consensus_recovery_antigravity_20260806
OUTPUT=/Users/mac/Documents/Dataset/exp435_train_only_consensus_recovery_antigravity_20260806/output_antigravity
CHECKPOINT=/Users/mac/Documents/Dataset/exp435_train_only_consensus_recovery_antigravity_20260806/checkpoint_antigravity
READINESS=/Users/mac/Documents/Dataset/exp435_train_only_consensus_recovery_antigravity_20260806/readiness_antigravity.json
EVALUATION=/Users/mac/Documents/Dataset/exp435_train_only_consensus_recovery_antigravity_20260806/evaluation_antigravity
RUN_IDENTITY=exp435-train-only-consensus-recovery-antigravity-20260806
```

readiness, runtime/checkpoint/progress/publication은 모두 이 정확한 frozen identity를
기록·검증하며 Git commit에서 identity를 만들지 않는다.

## 3. 구현 완료 전 검증

```bash
git status --short --branch
git rev-parse HEAD
python3 -m unittest discover \
  -s experiments/exp435_train_only_consensus_recovery_antigravity/tests \
  -p 'test_*antigravity.py'
python3 -m unittest discover \
  -s experiments/exp426_verified_baseline/tests \
  -p 'test_*antigravity.py'
python3 scratch/run_exp435_train_only_consensus_recovery_antigravity.py --help
git diff --check
```

전체 Exp417/425 회귀도 통과하고 코드맵 세 파일이 같은 commit과 구조를 나타내야
한다. 실제 실행 commit에는 추적되지 않은 파일이나 수정 파일이 없어야 한다.

## 4. readiness

run parent만 먼저 만든다. DB snapshot을 연 동안 `/Users/mac/Documents/Dataset`
부모에 새 항목을 만들지 않는다.

```bash
mkdir /Users/mac/Documents/Dataset/exp435_train_only_consensus_recovery_antigravity_20260806
python3 scratch/run_exp435_train_only_consensus_recovery_antigravity.py readiness
```

readiness는 다음을 확인한다.

- clean Git와 reviewed commit
- DB와 Exp291/292 입력 SHA256
- 공식 population 1,117개와 lane 수
- TEST label/labels_blob 접근 0
- TRAIN-only CPU/MLX parity와 속도 결정
- candidate 규칙의 고정 synthetic fixture 결과
- output, checkpoint, readiness artifact, evaluation의 네 고정 root가 모두 부재
  (run parent는 존재하더라도 비어 있어야 함)

MLX smoke는 seed `43501`로 만든 동일한 float32 입력을 CPU와 MLX에 전달한다.
fixture는 production처럼 channel 하나의 TEST query row 정확히 하나를 score하고,
reference batch sizes `16, 32, 64, 128, 256`은 변경하지 않는다.
각 값은 query 묶음이 아니라 reference 묶음 크기다. 각 묶음에서 찾은 최근접
후보를 전역으로 다시 합쳐 정확한 k=3 median을 계산하며, readiness에서 선택한
크기가 production 점수 계산에도 그대로 사용된다.
각 유한 score 쌍의 정규화오차는
`abs(cpu_score - mlx_score) / max(abs(cpu_score), abs(mlx_score), 1e-12)`이고,
그 최댓값과 최대 절대오차가 각각 `1e-5` 이하여야 한다. parity, 공식 alpha 후보
좌표와 전송·동기화를 포함한 1.1배 속도 향상을 모두 만족하지 못하면 CPU로
고정한다.

readiness가 MLX를 채택했더라도 runtime device/version/parity/output 오류는 CPU로
fallback하지 않고 run을 `failed`로 만든다. MLX manifest와 CPU score의 혼합은 없다.

성공하면 `readiness_antigravity.json`의 `state`가 `ready`이고 backend가 `cpu` 또는
`mlx`로 한 번 고정된다. 실패하면 `new`를 실행하지 않는다.

readiness는 `ready`만 기록한다. `new`는 `initializing`에서 시작해
`running`을 거쳐 checkpoint가 완성되면 `checkpoint_complete`를 만들고,
검증된 publication은 `complete_review_only`가 된다. `inspect`와 이미 끝난
`resume`은 이 publication을 read-only로 재생한다. `evaluate`는 동결 후에만
`evaluated` 결과를 반환한다. 사용자가 중단하면 `interrupted`, 무결성·budget
오류는 `failed`로 남으며 `failed` root는 재개하지 않는다. CLI는
readiness/new/resume/inspect가 성공 상태를
반환할 때 0, 평가가 `evaluated`/`success`/`complete`일 때 0을 반환하며,
argparse 오류나 처리 실패는 2, 미완료 상태는 1을 반환한다.

## 5. 새 label-free 실행

```bash
python3 scratch/run_exp435_train_only_consensus_recovery_antigravity.py new
```

프로그램은 output과 checkpoint를 새로 만들고 1,117개를 처리한다. candidate가
101개 이상 관측되면 top-N을 만들지 않고 전체 실패로 종료한다. TEST label은
열지 않는다. 공식 candidate budget은 100개이며 초과 결과를 잘라내지 않는다.

## 6. 진행 상태 확인

Codex가 반복 감시하지 않아도 프로세스가 다음 파일을 갱신한다.

```text
$OUTPUT/progress_antigravity.json
$OUTPUT/events_antigravity.jsonl
$OUTPUT/parent_antigravity.log
$OUTPUT/run_antigravity.pid
```

사람이 확인할 명령은 다음과 같다.

```bash
python3 -m json.tool \
  /Users/mac/Documents/Dataset/exp435_train_only_consensus_recovery_antigravity_20260806/output_antigravity/progress_antigravity.json
tail -n 40 \
  /Users/mac/Documents/Dataset/exp435_train_only_consensus_recovery_antigravity_20260806/output_antigravity/parent_antigravity.log
```

`current_dataset`, `dataset_metadata`, `stage`, `backend`, `counts`, `throughput`,
`elapsed_seconds`, `eta_seconds`, `checkpoint`와 `latest_error`를 확인한다.

CPU parallel에서는 submitted/in-flight future가 감지된 논리 CPU 수보다 하나 적다.
현재 공식 실행 Mac에서는 최대 11개다. `current_dataset`은
parent가 loading/dispatching 중이거나 아직 result를 수집하지 않은
`awaiting_worker_result`, 또는 `committing` 중인 dataset만 가리킨다. queued dataset과
이미 complete/commit된 dataset을 `scoring`으로 표시하지 않으며 parent가 worker
activity event를 받지 않으므로 parallel stage는 `scoring`을 주장하지 않는다.
progress와 event에 저장되는 모든 stage는 `loading`, `dispatching`,
`awaiting_worker_result`, `committing`, `idle`의 다섯 가지뿐이다. 시작·복구·완료는
현재 dataset이 없으므로 `idle`로 기록하며, 다른 값은 파일을 쓰기 전에 거절한다.

## 7. 중단과 재개

같은 PID가 살아 있으면 두 번째 실행을 시작하지 않는다. 프로세스가 없고 상태가
`interrupted` 또는 재개 가능한 미완료 상태일 때만 실행한다.

```bash
python3 scratch/run_exp435_train_only_consensus_recovery_antigravity.py resume
```

계산·무결성 오류나 candidate budget 초과로 `failed`가 된 root는 재개하지 않는다.
`resume`은 failed durable state를 input session open, PID rewrite, progress write,
event append 전에 read-only로 거절하므로 root tree bytes는 불변이다. 파일을 직접
고치거나 삭제하지 않는다. `interrupted` root만 frozen identity와 binding을 다시
검증한 뒤 재개할 수 있다. progress/PID/event stage와 binding 검사가 먼저 끝난 뒤에만
허용된 crash temp를 정리하며, invalid event가 있으면 temp도 그대로 둔다.

완료 checkpoint가 이미 있으면 `resume`은 DB나 worker를 다시 열지 않고 publication을
검증한다. 이미 존재하는 publication이 완전히 같으면 읽기 전용으로 검증해 그대로
반환한다. publication이 다르거나 게시 경쟁 중 destination이 새로 생긴 경우, 또는
evaluation destination이 이미 존재하는 경우에는 no-clobber 규칙으로 실패한다.
재사용 가능한 publication은 publish 함수 호출 시작 때 이미 있던 것뿐이다. 호출 뒤
replay/staging 중 나타난 destination은 bytes가 같아도 경주로 실패한다.

## 8. 읽기 전용 완료 확인

```bash
python3 scratch/run_exp435_train_only_consensus_recovery_antigravity.py inspect
```

완료 조건은 다음과 같다.

- state `complete_review_only`
- completed 1,117, error 0 (총 TEST row 195,900)
- candidate 0~100
- current dataset `null`, inactive PID
- TEST label 접근 0
- prediction/manifest/artifact SHA256 재계산 일치
- review-only true, operational alert eligible false

## 9. prediction 결과

```text
$OUTPUT/publication_antigravity/prediction_antigravity.jsonl
$OUTPUT/publication_antigravity/dataset_summary_antigravity.jsonl
$OUTPUT/publication_antigravity/score_evidence_antigravity.jsonl
$OUTPUT/publication_antigravity/abstention_ledger_antigravity.jsonl
$OUTPUT/publication_antigravity/backend_decision_antigravity.json
$OUTPUT/publication_antigravity/manifest_antigravity.json
$OUTPUT/publication_antigravity/status_antigravity.json
$OUTPUT/publication_antigravity/artifact_sha256_antigravity.json
```

prediction과 manifest SHA256을 RESEARCH_STATE와 EXPERIMENT_CATALOG에 기록한 뒤
prediction 단계를 닫는다.

## 10. 동결 후 사후 평가

`inspect`가 성공하고 prediction/manifest SHA256이 연구 상태 파일에 고정된 뒤에만
다음 명령을 실행한다.

```bash
python3 scratch/run_exp435_train_only_consensus_recovery_antigravity.py evaluate
```

`evaluate`는 먼저 동결된 artifact를 다시 검증한 뒤 TEST label을 연다. 결과는
`evaluation_antigravity` 아래에 새로 만들고 기존 prediction을 바꾸지 않는다.
production 경로는 run parent 바로 아래의 고정 output/evaluation 형제만 허용한다.
TEST label을 열기 전 Exp404 comparator
(`/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl`,
SHA256 `437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70`)와
Exp291의 `automatic_hard_indices`가 888개 non-empty 데이터셋, 2,218개 좌표,
229개 empty complement를 포함해 1,117개 전체에서 정확히 같은지 확인한다. 그
뒤 `mode=ro` SQLite 연결 하나만 열어 `instance_index`와 `label`을 읽고 닫는다.
Exp404 comparator와 Exp291 artifact는 각각 하나의 `O_NOFOLLOW` descriptor에서
bytes+SHA256+parse를 끝내며, SQLite는 Exp417 descriptor-bound verified read-only
connection 하나로 정확한 hashed inode만 연다.
label 접근 전 output 전체 트리 바이트를 snapshot하고, 평가 파일을 staging한 뒤
publish 직전 다시 전체 바이트를 검사한다. 변경이 있으면 no-clobber publish를
실패시킨다.

평가 root에는 정확히 다음 8개 파일만 생성된다.

- `evaluation_antigravity.json`
- `dataset_metrics_antigravity.jsonl`
- `rescued_datasets_antigravity.jsonl`
- `regressed_datasets_antigravity.jsonl`
- `confusion_antigravity.json`
- `bootstrap_antigravity.json`
- `decision_antigravity.json`
- `artifact_sha256_antigravity.json`

평가 결과에는 confusion matrix, micro/macro F1, rescued dataset, 추가 FP,
regression, strong baseline regression과 20,000회 paired dataset bootstrap이
포함된다.

## 11. 최종 판단

다음을 모두 만족하면 Exp435는 개선된 review-only 후보다.

- 1,117개 prediction/evaluation population 보존
- rescued dataset 10개 이상
- 추가 FP 30개 이하
- micro F1 point delta > 0 및 paired 95% CI lower > 0
- dataset-balanced macro F1 point delta > 0 및 paired 95% CI lower > 0
- prediction 동결 전 TEST label 접근 0, 같은 Exp435 안의 TEST 기반 재선택 0
- Exp404/Exp291 좌표 equality gate와 output snapshot/no-clobber 검증 통과
- `review_only=true`, `operational_alert_eligible=false` 유지

하나라도 실패하면 FAILURE_LEDGER에 기록하고 Exp435 설정을 바꾸지 않는다. 다음
연구는 새 실험 번호와 새 사전등록으로 시작한다.
