# Exp422-IR 실행 런북

이 런북은 Exp419가 `143(SIGTERM)`으로 중단된 사실을 확인한 뒤, 별도 등록된
Exp422-IR TRAIN-only CPU/MPS 진단을 한 번 실행하는 방법이다.

## 실행 명령

```bash
WORKTREE="/Users/mac/Documents/Time Series/.worktrees/exp422-interrupted419-diagnostic"
PYTHON="/Users/mac/Documents/Time Series/.venv/bin/python"
DB="/Users/mac/Documents/Dataset/univariate_ts.db"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="/Users/mac/Documents/Dataset/exp422_ir_attempt_${STAMP}"
PIDFILE="/Users/mac/Documents/Dataset/exp422_ir_${STAMP}.pid"
RUNLOG="/Users/mac/Documents/Dataset/exp422_ir_${STAMP}.run.log"

PYTHONPATH="$WORKTREE" PYTHONUNBUFFERED=1 "$PYTHON" \
  "$WORKTREE/scratch/run_exp422_interrupted419_diagnostic.py" \
  --database "$DB" \
  --output-root "$OUT" \
  --exp419-pidfile "/Users/mac/Documents/Dataset/exp419_official_20260802-001822.pid" \
  --exp419-exitfile "/Users/mac/Documents/Dataset/exp419_official_20260802-001822.exit" \
  --repeats 3 \
  >"$RUNLOG" 2>&1 &
printf '%s\n' "$!" >"$PIDFILE"
printf 'PID=%s\nOUTPUT=%s\nPROGRESS=%s/progress.jsonl\nRUNLOG=%s\n' \
  "$(cat "$PIDFILE")" "$OUT" "$OUT" "$RUNLOG"
```

## 진행 확인

```bash
cat "/Users/mac/Documents/Dataset/exp422_ir_${STAMP}.pid"
tail -f "/Users/mac/Documents/Dataset/exp422_ir_attempt_${STAMP}/progress.jsonl"
tail -f "/Users/mac/Documents/Dataset/exp422_ir_${STAMP}.run.log"
```

`progress.jsonl`의 `heartbeat`에는 현재 단계, 완료 수, 전체 수, 경과 시간이
기록된다. `core_finished`는 60행, `score_finished`는 5개 batch 정책의 완료
수를 기록한다. 완료되면 같은 폴더에 `manifest.json`이 생긴다.

## 중단

Exp422-IR PID에만 SIGTERM을 보내고, Exp419 PID·checkpoint·공식 output에는
신호를 보내지 않는다. 부분 output과 progress log는 증거 보존을 위해 삭제하지
않는다.
