# Exp422-IR-E 실행 런북

Exp422-IR의 core parity 실패를 보존하면서 TRAIN-only score evidence만
추가 수집한다. 이 실행은 review-only이며 운영 적용을 하지 않는다.

```bash
WORKTREE="/Users/mac/Documents/Time Series/.worktrees/exp422-ir-exploratory-score"
PYTHON="/Users/mac/Documents/Time Series/.venv/bin/python"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="/Users/mac/Documents/Dataset/exp422_ir_e_attempt_${STAMP}"
PIDFILE="/Users/mac/Documents/Dataset/exp422_ir_e_${STAMP}.pid"
RUNLOG="/Users/mac/Documents/Dataset/exp422_ir_e_${STAMP}.run.log"

printf '%s\n' "$$" >"$PIDFILE"
exec env PYTHONPATH="$WORKTREE" PYTHONUNBUFFERED=1 "$PYTHON" \
  "$WORKTREE/scratch/run_exp422_ir_exploratory_score.py" \
  --database "/Users/mac/Documents/Dataset/univariate_ts.db" \
  --output-root "$OUT" \
  --exp419-pidfile "/Users/mac/Documents/Dataset/exp419_official_20260802-001822.pid" \
  --exp419-exitfile "/Users/mac/Documents/Dataset/exp419_official_20260802-001822.exit" \
  >"$RUNLOG" 2>&1
```

확인은 다음과 같이 한다.

```bash
tail -f "/Users/mac/Documents/Dataset/exp422_ir_e_attempt_${STAMP}/progress.jsonl"
ps -p "$(cat "/Users/mac/Documents/Dataset/exp422_ir_e_${STAMP}.pid")" -o pid,etime,pcpu,command
```

`core_manifest.json`은 parity 실패를 보존한 canonical Exp422 evidence이고,
`exploratory_manifest.json`은 core 실패 뒤 score를 수집했다는 review-only
묶음이다. 두 파일 모두 운영 의사결정에 사용하지 않는다.
