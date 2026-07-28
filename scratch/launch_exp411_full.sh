#!/bin/bash
set -euo pipefail

usage() {
  echo "usage: $0 [--print-command]" >&2
}

print_only=0
case "${1-}" in
  "")
    ;;
  --print-command)
    print_only=1
    ;;
  *)
    usage
    exit 2
    ;;
esac

repo="/Users/mac/Documents/Time Series/.worktrees/exp409-prospective-readiness"
python_path="$repo/.venv/bin/python"
runner_path="$repo/scratch/run_exp411_near_repeat_qualification.py"
db_path="/Users/mac/Documents/Dataset/univariate_ts.db"
population_manifest="/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_prediction_manifest.json"
checkpoint_dir="/Users/mac/Documents/Dataset/exp411_near_repeat_qualification.checkpoint"
benchmark_report="/Users/mac/Documents/Dataset/exp411_acceleration_benchmark.json"
output_dir="/Users/mac/Documents/Dataset/exp411_near_repeat_qualification"
stdout_log="/private/tmp/exp411-full.stdout.log"
stderr_log="/private/tmp/exp411-full.stderr.log"
job_label="com.openai.exp411.full"

command=(
  launchctl submit
  -l "$job_label"
  -o "$stdout_log"
  -e "$stderr_log"
  --
  /usr/bin/env
  NUMBA_NUM_THREADS=12
  OMP_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1
  VECLIB_MAXIMUM_THREADS=1
  NUMEXPR_NUM_THREADS=1
  "$python_path"
  "$runner_path"
  --full-qualification
  --db-path "$db_path"
  --population-manifest "$population_manifest"
  --checkpoint-dir "$checkpoint_dir"
  --benchmark-report "$benchmark_report"
  --output-dir "$output_dir"
)

if [[ "$print_only" -eq 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

for required_file in \
  "$python_path" \
  "$runner_path" \
  "$db_path" \
  "$population_manifest" \
  "$benchmark_report"
do
  if [[ ! -f "$required_file" ]]; then
    echo "required file is missing: $required_file" >&2
    exit 1
  fi
done

if [[ -e "$output_dir" ]]; then
  echo "final output already exists: $output_dir" >&2
  exit 1
fi
if [[ -n "$(git -C "$repo" status --porcelain)" ]]; then
  echo "worktree must be clean before full execution" >&2
  exit 1
fi
current_commit="$(git -C "$repo" rev-parse HEAD)"
database_sha256="$(shasum -a 256 "$db_path" | awk '{print $1}')"
if [[ "$database_sha256" != "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c" ]]; then
  echo "database SHA256 mismatch" >&2
  exit 1
fi
if [[ "$(sqlite3 "$db_path" "PRAGMA quick_check;")" != "ok" ]]; then
  echo "database quick_check failed" >&2
  exit 1
fi
"$python_path" - "$benchmark_report" "$current_commit" <<'PY'
from pathlib import Path
import sys
from experiments.exp411_near_repeat_qualification.runner import (
    load_benchmark_report,
)
load_benchmark_report(Path(sys.argv[1]), expected_commit=sys.argv[2])
PY
"$python_path" - "$population_manifest" <<'PY'
from pathlib import Path
import sys
from experiments.exp411_near_repeat_qualification.runner import (
    load_population_manifest,
)
load_population_manifest(Path(sys.argv[1]))
PY

"${command[@]}"
echo "Submitted $job_label"
echo "checkpoint: $checkpoint_dir"
echo "stdout: $stdout_log"
echo "stderr: $stderr_log"
