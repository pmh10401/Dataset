# Exp435 실행 안내서

## 1. 목적

Exp435는 정상 TRAIN으로 만든 대표 패턴·기울기·주기 점수의 합의로 1,117개 TEST
instance를 label-free 평가한다. 새 후보는 검토용이며 기존 Hard 경보를 바꾸지
않는다.

모든 명령은 다음 clean worktree에서 실행한다.

```bash
cd '/Users/mac/Documents/Time Series/.worktrees/exp425-selective-recovery'
```

실행 파일은 구현 후 다음 경로에 있다.

```text
scratch/run_exp435_train_only_consensus_recovery_antigravity.py
```

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
```

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
- output/checkpoint root 부재

성공하면 `readiness_antigravity.json`의 `state`가 `ready`이고 backend가 `cpu` 또는
`mlx`로 한 번 고정된다. 실패하면 `new`를 실행하지 않는다.

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

## 7. 중단과 재개

같은 PID가 살아 있으면 두 번째 실행을 시작하지 않는다. 프로세스가 없고 상태가
`interrupted` 또는 재개 가능한 미완료 상태일 때만 실행한다.

```bash
python3 scratch/run_exp435_train_only_consensus_recovery_antigravity.py resume
```

계산·무결성 오류나 candidate budget 초과로 `failed`가 된 root는 재개하지 않는다.
파일을 직접 고치거나 삭제하지 않는다.

## 8. 읽기 전용 완료 확인

```bash
python3 scratch/run_exp435_train_only_consensus_recovery_antigravity.py inspect
```

완료 조건은 다음과 같다.

- state `complete_review_only`
- completed 1,117, error 0
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

평가 결과에는 confusion matrix, micro/macro F1, rescued dataset, 추가 FP,
regression, strong baseline regression과 20,000회 paired dataset bootstrap이
포함된다.

## 11. 최종 판단

다음을 모두 만족하면 Exp435는 개선된 review-only 후보다.

- rescued dataset 10개 이상
- 추가 FP 30개 이하
- micro F1 95% CI lower > 0
- macro F1 95% CI lower > 0

하나라도 실패하면 FAILURE_LEDGER에 기록하고 Exp435 설정을 바꾸지 않는다. 다음
연구는 새 실험 번호와 새 사전등록으로 시작한다.
