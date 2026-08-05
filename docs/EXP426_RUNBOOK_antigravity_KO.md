# Exp426 실행 안내서

## 1. 이 안내서의 목적

Exp426은 Exp424의 검증된 983개 결과와 TRAIN 근거 부족 134개를 합쳐 1,117개
전체 비교 기준을 만든다. 새 모델을 학습하거나 TEST 정답을 보는 실행이 아니다.
모든 명령은 저장소 root에서 실행한다.

```bash
cd '/Users/mac/Documents/Time Series/.worktrees/exp425-selective-recovery'
```

실행 파일은 다음 하나다.

```text
scratch/run_exp426_verified_baseline_antigravity.py
```

worker 수는 12로 고정되어 있고 명령행 옵션으로 바꿀 수 없다.

## 2. 고정 경로

```text
DB=/Users/mac/Documents/Dataset/univariate_ts.db
SOURCE=/Users/mac/Documents/Dataset/exp424_mps_score_cache_20260802-153831.checkpoint
OUTPUT=/Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805
CHECKPOINT=/Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805.checkpoint
READINESS=/Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_readiness_20260805.json
```

다른 경로는 CLI가 거부한다. 기존 output, checkpoint 또는 readiness 파일을
덮어쓰지 않는다.

## 3. 실행 전 확인

먼저 branch가 깨끗하고 Task 4 commit에 있는지 확인한다.

```bash
git status --short --branch
git rev-parse HEAD
python3 -m unittest discover \
  -s experiments/exp426_verified_baseline/tests \
  -p 'test_*antigravity.py'
```

실제 실행 전에는 전체 Exp425와 지정된 Exp417 회귀도 통과해야 한다. readiness는
clean Git, DB·lane·Exp424 지문, 1대12 과학 결과 동일성, 실제 12 worker PID,
DB 1회 해시 수명주기와 집중 테스트를 다시 기록한다.

## 4. readiness

readiness 파일과 실제 output/checkpoint가 모두 없을 때 한 번만 실행한다.

```bash
python3 scratch/run_exp426_verified_baseline_antigravity.py readiness \
  --readiness-output \
  /Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_readiness_20260805.json
```

성공 조건은 다음과 같다.

- `state = ready`
- source `records/importable/terminal = 1117/983/134`
- `workers = 12`, `observed_worker_processes = 12`
- `scientific_parity = true`
- DB `full_hash_count = 1`, cleanup `removed`
- TEST 값·정답 접근 수 0
- output/checkpoint root absent

readiness는 지정한 JSON 하나만 남긴다. DB snapshot이 잠시 만드는 private owner
link와 directory는 종료 전에 제거되어야 한다. readiness 실패 시 실제 실행을
시작하지 않는다.

## 5. 새 실행

fresh Sol/High 검토와 readiness가 통과한 뒤에만 실행한다.

```bash
python3 scratch/run_exp426_verified_baseline_antigravity.py new \
  --output-root \
  /Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805 \
  --checkpoint-root \
  /Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805.checkpoint
```

프로그램은 먼저 Exp424 1,117개를 12 CPU worker로 확인하고 983개를 가져온다.
그 다음 134개만 DB의 정상 TRAIN으로 다시 확인한다. TEST 값과 정답 loader는
사용하지 않는다. 완료되면 `publication_antigravity` 결과 묶음을 원자적으로
공개하고 prediction과 manifest SHA256을 출력한다.

## 6. 진행 상황 확인

Codex가 반복 감시하지 않아도 프로그램이 30초마다 진행 파일을 갱신한다.

```text
/Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805/progress.json
/Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805/events.jsonl
/Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805/parent.log
/Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805/run.pid
```

사람이 볼 때는 다음처럼 읽으면 된다.

```bash
python3 -m json.tool \
  /Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805/progress.json
tail -n 30 \
  /Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805/parent.log
```

`progress.json`의 핵심은 `state`, `current_dataset`, `dataset_metadata`,
`counts`, `throughput_datasets_per_second`, `elapsed_seconds`, `eta_seconds`,
`checkpoint`, `latest_error`다. `run.pid`의 `active=true`인 같은 프로세스가
살아 있으면 두 번째 실행을 시작하지 않는다.

## 7. 중단 후 재개

프로세스가 끝났고 root가 미완료일 때만 다음 명령을 쓴다.

```bash
python3 scratch/run_exp426_verified_baseline_antigravity.py resume \
  --output-root \
  /Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805 \
  --checkpoint-root \
  /Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805.checkpoint
```

이미 checkpoint에 확정된 데이터셋은 다시 계산하지 않는다. record 저장 뒤 event
또는 progress 기록 전에 중단됐으면 누락된 로그만 복원한다. 완료된 root에서
`resume`을 다시 실행하면 DB·Exp424 출처·worker를 열지 않고 결과를 읽기 전용으로
재검증해 반환한다.

손상된 파일을 직접 고치거나 삭제하지 않는다. 오류와 마지막 checkpoint를 보존한
뒤 원인을 조사한다.

## 8. 읽기 전용 확인

다음 명령은 파일 bytes, inode, mtime을 바꾸지 않는다.

```bash
python3 scratch/run_exp426_verified_baseline_antigravity.py inspect \
  --output-root \
  /Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805 \
  --checkpoint-root \
  /Users/mac/Documents/Dataset/exp426_verified_baseline_antigravity_20260805.checkpoint
```

완료 시 다음을 요구한다.

- `state = complete_review_only`
- completed/imported/verified/failed = `1117/983/134/0`
- `current_dataset = null`
- backend/workers = `cpu/12`
- prediction과 manifest 경로 및 SHA256 존재
- `review_only = true`
- `operational_alert_eligible = false`
- inactive PID와 checkpoint/runtime/publication 일치

미완료 상태는 exit code 1, 정상 완료는 0, 잘못된 명령·경로·손상은 2다.

## 9. 결과 파일

```text
OUTPUT/publication_antigravity/prediction_antigravity.jsonl
OUTPUT/publication_antigravity/dataset_summary_antigravity.jsonl
OUTPUT/publication_antigravity/import_ledger_antigravity.jsonl
OUTPUT/publication_antigravity/train_abstention_ledger_antigravity.jsonl
OUTPUT/publication_antigravity/backend_decision_antigravity.json
OUTPUT/publication_antigravity/manifest_antigravity.json
OUTPUT/publication_antigravity/status_antigravity.json
OUTPUT/publication_antigravity/artifact_sha256_antigravity.json
```

`inspect`는 체크포인트 record SHA256, 완료 event, 정렬된 후보, manifest hash와
artifact hash를 처음부터 다시 계산한다. 결과 묶음의 일부만 있거나 한 byte라도
달라지면 실패한다.

## 10. 실행 후 멈출 지점

prediction과 manifest SHA256을 `RESEARCH_STATE.json`과
`EXPERIMENT_CATALOG.json`에 기록하고 여기서 멈춘다. Exp426 실행 안에서 TEST
정답, 혼동행렬, precision, recall 또는 F1을 계산하지 않는다. 별도 사후 평가를
시작하기 전에 사용자와 다음 연구 방향을 다시 논의한다.
