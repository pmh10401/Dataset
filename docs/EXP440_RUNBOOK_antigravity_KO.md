# Exp440 실행 안내서

## 고정 경로

```text
RUN_PARENT=/Users/mac/Documents/Dataset/exp440_train_only_low_frequency_precision_gate_antigravity_20260808
READINESS=$RUN_PARENT/readiness_antigravity.json
OUTPUT=$RUN_PARENT/output_antigravity
CHECKPOINT=$RUN_PARENT/checkpoint_antigravity
PUBLICATION=$RUN_PARENT/publication_antigravity
EVALUATION=$RUN_PARENT/evaluation_antigravity
PROGRESS=$OUTPUT/progress_antigravity.json
EVENTS=$OUTPUT/events_antigravity.jsonl
PARENT_LOG=$OUTPUT/parent_antigravity.log
PID_FILE=$OUTPUT/run_antigravity.pid
```

진행 로그에는 PID, 현재 데이터셋, TRAIN/TEST 행 수와 길이, 현재 단계와 backend, 완료/전체, import/recompute/success/abstention/error, 임시/최종 후보 수, 처리속도, 경과시간, ETA, checkpoint, heartbeat, 최신 오류가 기록된다.

## 고정 명령

작업 디렉터리:

```bash
cd '/Users/mac/Documents/Time Series/.worktrees/exp438-frequency-complexity-evidence'
```

사전 점검:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp440_train_only_low_frequency_precision_gate_antigravity.py readiness
```

새 전체 실행:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp440_train_only_low_frequency_precision_gate_antigravity.py new
```

중단된 checkpoint 재개:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp440_train_only_low_frequency_precision_gate_antigravity.py resume
```

완료 산출물 읽기 전용 검증:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp440_train_only_low_frequency_precision_gate_antigravity.py inspect
```

prediction 동결 후 사후 평가:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scratch/run_exp440_train_only_low_frequency_precision_gate_antigravity.py evaluate
```

경로, worker, backend, seed, 깊이, 문턱, quantile, 예산을 명령줄에서 바꾸는 옵션은 제공하지 않는다.

## 상태 확인

Codex가 반복 감시하지 않아도 프로세스가 직접 다음 파일을 갱신한다.

```bash
python3 -m json.tool "$PROGRESS"
tail -n 30 "$PARENT_LOG"
tail -n 10 "$EVENTS"
cat "$PID_FILE"
```

오류가 나면 실패 루트는 보존한다. `resume`은 죽은 PID와 정확한 `interrupted` 순서 checkpoint에만 허용한다. `failed`, `running`, 변경된 record 또는 이미 존재하는 publication/evaluation은 덮어쓰지 않는다.
