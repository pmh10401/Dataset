#!/usr/bin/env bash
set -euo pipefail

# Create/verify only the fixed Exp444 environment.  The repository .venv and
# all other experiment environments are intentionally outside this script's
# scope.
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
DIRECT_REQUIREMENTS="${REPO_ROOT}/experiments/exp444_train_only_stability_precision_gate_antigravity/requirements_antigravity.txt"
LOCK_REQUIREMENTS="${REPO_ROOT}/experiments/exp444_train_only_stability_precision_gate_antigravity/requirements_lock_antigravity.txt"
VENV_ROOT="/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314"
PYTHON_BIN="/usr/local/bin/python3"
BOOTSTRAP_FLAG="--bootstrap-lock-antigravity"
EXP444_FREEZE_PATH_ANTIGRAVITY=""

cleanup_antigravity() {
  local status="${1:-$?}"
  trap - EXIT HUP INT TERM
  if [[ -n "${EXP444_FREEZE_PATH_ANTIGRAVITY:-}" ]]; then
    rm -f -- "$EXP444_FREEZE_PATH_ANTIGRAVITY" || status=1
    EXP444_FREEZE_PATH_ANTIGRAVITY=""
  fi
  return "$status"
}

signal_antigravity() {
  local status="$1"
  cleanup_antigravity "$status" || true
  exit "$status"
}

install_traps_antigravity() {
  trap 'cleanup_antigravity' EXIT
  trap 'signal_antigravity 129' HUP
  trap 'signal_antigravity 130' INT
  trap 'signal_antigravity 143' TERM
}

validate_freeze_antigravity() {
  local venv_python="$1"
  local lock_path="$2"
  local actual_path="$3"
  EXP444_FREEZE_PATH_ANTIGRAVITY="$actual_path"
  install_traps_antigravity
  if ! "$venv_python" -m pip freeze --all | LC_ALL=C sort > "$actual_path"; then
    printf '%s\n' "Exp444 freeze collection failed" >&2
    cleanup_antigravity 1 || true
    return 1
  fi
  if ! cmp -s "$lock_path" "$actual_path"; then
    printf '%s\n' "Exp444 environment freeze does not match tracked lock" >&2
    diff -u "$lock_path" "$actual_path" >&2 || true
    cleanup_antigravity 1 || true
    return 1
  fi
  cleanup_antigravity 0
}

bootstrap_lock_antigravity() {
  local venv_python="$1"
  local lock_path="$2"
  if [[ -s "$lock_path" ]]; then
    printf '%s\n' "bootstrap is allowed only while the tracked lock is empty" >&2
    return 1
  fi
  local temporary
  temporary="$(mktemp "${lock_path}.tmp.antigravity.XXXXXX")"
  EXP444_FREEZE_PATH_ANTIGRAVITY="$temporary"
  install_traps_antigravity
  if ! "$venv_python" -m pip freeze --all | LC_ALL=C sort > "$temporary"; then
    printf '%s\n' "Exp444 bootstrap freeze failed" >&2
    cleanup_antigravity 1 || true
    return 1
  fi
  if [[ ! -s "$temporary" ]]; then
    printf '%s\n' "Exp444 bootstrap lock is empty" >&2
    cleanup_antigravity 1 || true
    return 1
  fi
  # The empty tracked lock is a one-time bootstrap sentinel.  Remove only that
  # exact empty file, then publish via a same-directory hard link so an
  # existing non-empty winner can never be replaced.
  if [[ -e "$lock_path" ]]; then
    if [[ -s "$lock_path" ]]; then
      printf '%s\n' "lock appeared concurrently; refusing overwrite" >&2
      cleanup_antigravity 1 || true
      return 1
    fi
    rm -f -- "$lock_path"
  fi
  if ! ln "$temporary" "$lock_path"; then
    printf '%s\n' "lock appeared concurrently; refusing overwrite" >&2
    cleanup_antigravity 1 || true
    return 1
  fi
  rm -f -- "$temporary"
  EXP444_FREEZE_PATH_ANTIGRAVITY=""
  cleanup_antigravity 0
}

validate_existing_antigravity() {
  local venv_root="$1"
  local lock_path="$2"
  if [[ ! -s "$lock_path" ]]; then
    printf '%s\n' "tracked lock is absent or empty; use --bootstrap-lock-antigravity once" >&2
    return 1
  fi
  if [[ -L "$venv_root" || ! -d "$venv_root" || ! -x "$venv_root/bin/python" ]]; then
    printf '%s\n' "existing Exp444 environment is not a valid Python venv; refusing overwrite" >&2
    return 1
  fi
  local mode
  mode="$(stat -f '%Lp' "$venv_root" 2>/dev/null || stat -c '%a' "$venv_root")"
  if [[ "$mode" != "700" ]]; then
    printf '%s\n' "existing Exp444 venv mode is $mode, expected 700; refusing overwrite" >&2
    return 1
  fi
  local venv_python="$venv_root/bin/python"
  local version prefix base
  version="$($venv_python -c 'import platform; print(platform.python_version())' 2>/dev/null || true)"
  if [[ "$version" != "3.14.6" ]]; then
    printf '%s\n' "existing Exp444 venv has Python $version; refusing overwrite" >&2
    return 1
  fi
  prefix="$($venv_python -c 'import sys; print(sys.prefix)' 2>/dev/null || true)"
  base="$($venv_python -c 'import sys; print(sys.base_prefix)' 2>/dev/null || true)"
  if [[ "$prefix" != "$venv_root" || "$base" == "$prefix" ]]; then
    printf '%s\n' "existing Exp444 environment is not isolated; refusing overwrite" >&2
    return 1
  fi
  if ! "$venv_python" - "$lock_path" <<'PY'
import importlib.metadata
from pathlib import Path
import sys

lock = Path(sys.argv[1])
for raw in lock.read_text(encoding="utf-8").splitlines():
    if not raw or raw.startswith("#"):
        continue
    name, separator, version = raw.partition("==")
    if not separator or not name or not version:
        raise SystemExit(f"invalid lock entry: {raw}")
    try:
        actual = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit(f"missing locked distribution: {name}")
    if actual != version:
        raise SystemExit(f"locked distribution mismatch: {name} {actual} != {version}")
PY
  then
    printf '%s\n' "existing Exp444 environment does not match lock metadata; refusing overwrite" >&2
    return 1
  fi
  "$venv_python" -m pip check
  validate_freeze_antigravity "$venv_python" "$lock_path" "${lock_path}.preflight.$$"
}

main_antigravity() {
  local bootstrap=0
  if [[ "$#" -gt 1 || ( "$#" -eq 1 && "$1" != "$BOOTSTRAP_FLAG" ) ]]; then
    printf 'usage: %s [%s]\n' "$0" "$BOOTSTRAP_FLAG" >&2
    return 2
  fi
  if [[ "$#" -eq 1 ]]; then bootstrap=1; fi
  if [[ ! -x "$PYTHON_BIN" ]]; then
    printf '%s\n' "Exp444 requires $PYTHON_BIN (Python 3.14.6)" >&2
    return 1
  fi
  local python_version
  python_version="$($PYTHON_BIN -c 'import platform; print(platform.python_version())')"
  if [[ "$python_version" != "3.14.6" ]]; then
    printf '%s\n' "Exp444 requires Python 3.14.6, got $python_version" >&2
    return 1
  fi
  local expected_direct=$'mlx==0.32.0\nnumba==0.66.0\nnumpy==2.4.6\n'
  if [[ ! -f "$DIRECT_REQUIREMENTS" || "$(<"$DIRECT_REQUIREMENTS")"$'\n' != "$expected_direct" ]]; then
    printf '%s\n' "direct requirements do not match the frozen three-line contract" >&2
    return 1
  fi
  if (( bootstrap == 0 )) && [[ ! -s "$LOCK_REQUIREMENTS" ]]; then
    printf '%s\n' "tracked lock is absent or empty; use --bootstrap-lock-antigravity once" >&2
    return 1
  fi
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
  umask 077
  local parent
  parent="$(dirname -- "$VENV_ROOT")"
  if [[ -e "$parent" && ! -d "$parent" ]]; then
    printf '%s\n' "Exp444 venv parent is not a directory; refusing overwrite" >&2
    return 1
  fi
  [[ -d "$parent" ]] || mkdir -p -m 700 "$parent"
  if [[ -e "$VENV_ROOT" ]]; then
    if (( bootstrap == 1 )); then
      printf '%s\n' "bootstrap is creation-only; an existing Exp444 environment is never mutated" >&2
      return 1
    fi
    validate_existing_antigravity "$VENV_ROOT" "$LOCK_REQUIREMENTS"
    printf '%s\n' "EXP444_ENVIRONMENT_READY"
    return 0
  fi
  "$PYTHON_BIN" -m venv "$VENV_ROOT"
  chmod 700 "$VENV_ROOT"
  local venv_python="$VENV_ROOT/bin/python"
  if (( bootstrap == 1 )); then
    "$venv_python" -m pip install --disable-pip-version-check --no-input --no-cache-dir --requirement "$DIRECT_REQUIREMENTS"
    "$venv_python" -m pip check
    bootstrap_lock_antigravity "$venv_python" "$LOCK_REQUIREMENTS"
  else
    "$venv_python" -m pip install --disable-pip-version-check --no-input --no-cache-dir --no-deps --requirement "$LOCK_REQUIREMENTS"
    "$venv_python" -m pip check
    validate_freeze_antigravity "$venv_python" "$LOCK_REQUIREMENTS" "${LOCK_REQUIREMENTS}.freeze.$$"
  fi
  printf '%s\n' "EXP444_ENVIRONMENT_READY"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main_antigravity "$@"
fi
