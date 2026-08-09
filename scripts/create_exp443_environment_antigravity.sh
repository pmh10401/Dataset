#!/usr/bin/env bash
set -euo pipefail

# Exp443's environment is deliberately outside the repository.  This script
# never activates or edits a project .venv and never selects a different Python
# based on PATH ordering.  The validation helper is sourceable for tests; the
# normal execution path below always uses the fixed official root.
SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
DIRECT_REQUIREMENTS="${REPO_ROOT}/experiments/exp443_missing_normal_train_policy_comparison_antigravity/requirements_antigravity.txt"
LOCK_REQUIREMENTS="${REPO_ROOT}/experiments/exp443_missing_normal_train_policy_comparison_antigravity/requirements_lock_antigravity.txt"
VENV_ROOT="/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314"
PYTHON_BIN="/usr/local/bin/python3.14"
BOOTSTRAP_FLAG="--bootstrap-lock-antigravity"

exp443_validate_existing_venv_antigravity() {
  local venv_root="${1:-}"
  local lock_requirements="${2:-}"
  if [[ -z "$venv_root" || -z "$lock_requirements" ]]; then
    printf '%s\n' "usage: exp443_validate_existing_venv_antigravity VENV_ROOT LOCK_REQUIREMENTS" >&2
    return 2
  fi
  if [[ ! -f "$lock_requirements" ]]; then
    printf '%s\n' "tracked lock is absent; refusing existing-environment validation" >&2
    return 1
  fi
  if [[ -L "$venv_root" || ! -d "$venv_root" || ! -x "$venv_root/bin/python" ]]; then
    printf '%s\n' "existing Exp443 environment is not a valid Python venv; refusing overwrite" >&2
    return 1
  fi
  local existing_mode
  existing_mode="$(stat -f '%Lp' "$venv_root" 2>/dev/null || stat -c '%a' "$venv_root")"
  if [[ "$existing_mode" != "700" ]]; then
    printf '%s\n' "existing Exp443 venv mode is $existing_mode, expected 700; refusing overwrite" >&2
    return 1
  fi
  local venv_python="$venv_root/bin/python"
  local existing_version
  existing_version="$($venv_python -c 'import platform; print(platform.python_version())' 2>/dev/null || true)"
  if [[ "$existing_version" != "3.14.6" ]]; then
    printf '%s\n' "existing Exp443 venv has Python $existing_version; refusing overwrite" >&2
    return 1
  fi
  local existing_prefix existing_base
  existing_prefix="$($venv_python -c 'import sys; print(sys.prefix)' 2>/dev/null || true)"
  existing_base="$($venv_python -c 'import sys; print(sys.base_prefix)' 2>/dev/null || true)"
  if [[ "$existing_prefix" != "$venv_root" || "$existing_base" == "$existing_prefix" ]]; then
    printf '%s\n' "existing Exp443 environment is not an isolated venv; refusing overwrite" >&2
    return 1
  fi
  # Distribution metadata is checked with the venv interpreter before pip is
  # touched, so a package-mismatched venv also fails without mutation.
  if ! "$venv_python" - "$lock_requirements" <<'PY'
import importlib.metadata
from pathlib import Path
import sys

lock_path = Path(sys.argv[1])
expected = {}
for raw in lock_path.read_text(encoding="utf-8").splitlines():
    if not raw or raw.startswith("#"):
        continue
    name, separator, version = raw.partition("==")
    if not separator or not name or not version:
        raise SystemExit(f"invalid lock entry: {raw}")
    expected[name] = version
for name, version in expected.items():
    try:
        actual = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        raise SystemExit(f"missing locked distribution: {name}")
    if actual != version:
        raise SystemExit(f"locked distribution mismatch: {name} {actual} != {version}")
if importlib.metadata.version("pip") != "26.1.2":
    raise SystemExit("pip is not pinned to 26.1.2")
PY
  then
    printf '%s\n' "existing Exp443 environment does not match the lock metadata; refusing overwrite" >&2
    return 1
  fi
  local actual_freeze="${lock_requirements}.preflight.$$"
  if ! "$venv_python" -m pip check; then
    rm -f "$actual_freeze"
    printf '%s\n' "existing Exp443 environment failed pip check; refusing overwrite" >&2
    return 1
  fi
  if ! "$venv_python" -m pip freeze --all | LC_ALL=C sort > "$actual_freeze"; then
    rm -f "$actual_freeze"
    printf '%s\n' "existing Exp443 environment freeze failed; refusing overwrite" >&2
    return 1
  fi
  if ! cmp -s "$lock_requirements" "$actual_freeze"; then
    rm -f "$actual_freeze"
    printf '%s\n' "existing Exp443 environment does not match the byte-identical sorted lock; refusing overwrite" >&2
    return 1
  fi
  rm -f "$actual_freeze"
  return 0
}

exp443_main_antigravity() {
  local bootstrap=0
  if [[ "$#" -gt 1 || ( "$#" -eq 1 && "$1" != "$BOOTSTRAP_FLAG" ) ]]; then
    printf '%s\n' "usage: $0 [--bootstrap-lock-antigravity]" >&2
    return 2
  fi
  if [[ "$#" -eq 1 ]]; then bootstrap=1; fi

  if [[ ! -x "$PYTHON_BIN" ]]; then
    printf '%s\n' "Exp443 requires $PYTHON_BIN (Python 3.14.6)" >&2
    return 1
  fi
  local python_version
  python_version="$($PYTHON_BIN -c 'import platform; print(platform.python_version())')"
  if [[ "$python_version" != "3.14.6" ]]; then
    printf '%s\n' "Exp443 requires Python 3.14.6, got $python_version" >&2
    return 1
  fi
  if [[ "$VENV_ROOT" == "$REPO_ROOT/.venv" || "$VENV_ROOT" == "$REPO_ROOT/.venv/"* ]]; then
    printf '%s\n' "refusing to use the project .venv" >&2
    return 1
  fi
  if [[ ! -f "$DIRECT_REQUIREMENTS" ]]; then
    printf '%s\n' "direct requirements are missing: $DIRECT_REQUIREMENTS" >&2
    return 1
  fi
  local expected_direct=$'aeon==1.5.0\nnumpy==2.3.5\nscikit-learn==1.8.0\n'
  if [[ "$(<"$DIRECT_REQUIREMENTS")"$'\n' != "$expected_direct" ]]; then
    printf '%s\n' "direct requirements do not match the frozen three-line contract" >&2
    return 1
  fi
  if (( bootstrap == 1 )) && [[ -e "$LOCK_REQUIREMENTS" ]]; then
    printf '%s\n' "bootstrap is allowed only while the tracked lock is absent" >&2
    return 1
  fi
  if (( bootstrap == 0 )) && [[ ! -f "$LOCK_REQUIREMENTS" ]]; then
    printf '%s\n' "tracked lock is absent; use --bootstrap-lock-antigravity once" >&2
    return 1
  fi

  # The official CPU route is launched with every common BLAS/OpenMP control
  # set explicitly. These are process-local settings.
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1
  export BLIS_NUM_THREADS=1

  umask 077
  local venv_parent="$(dirname -- "$VENV_ROOT")"
  if [[ -e "$venv_parent" && ! -d "$venv_parent" ]]; then
    printf '%s\n' "Exp443 venv parent is not a directory; refusing overwrite" >&2
    return 1
  fi
  if [[ ! -e "$venv_parent" ]]; then
    mkdir -p -m 700 "$venv_parent"
  fi

  if [[ -e "$VENV_ROOT" ]]; then
    # Existing environments are read-only from this script's perspective.
    if (( bootstrap == 1 )); then
      printf '%s\n' "bootstrap is creation-only; an existing Exp443 environment is never mutated" >&2
      return 1
    fi
    if ! exp443_validate_existing_venv_antigravity "$VENV_ROOT" "$LOCK_REQUIREMENTS"; then
      return 1
    fi
    printf '%s\n' "EXP443_ENVIRONMENT_READY"
    return 0
  fi

  "$PYTHON_BIN" -m venv "$VENV_ROOT"
  chmod 700 "$VENV_ROOT"
  local venv_python="$VENV_ROOT/bin/python"
  "$venv_python" -m pip install --disable-pip-version-check --no-input --upgrade --no-cache-dir 'pip==26.1.2'
  if (( bootstrap == 1 )); then
    "$venv_python" -m pip install --disable-pip-version-check --no-input --no-cache-dir --requirement "$DIRECT_REQUIREMENTS"
    "$venv_python" -m pip check
    local tmp_lock="${LOCK_REQUIREMENTS}.tmp.$$"
    trap 'rm -f "$tmp_lock"' EXIT HUP INT TERM
    "$venv_python" -m pip freeze --all | LC_ALL=C sort > "$tmp_lock"
    if [[ ! -s "$tmp_lock" ]]; then
      printf '%s\n' "pip freeze produced an empty Exp443 lock" >&2
      return 1
    fi
    if ! ( set -C; : > "$LOCK_REQUIREMENTS" ) 2>/dev/null; then
      printf '%s\n' "lock appeared concurrently; refusing overwrite" >&2
      return 1
    fi
    mv "$tmp_lock" "$LOCK_REQUIREMENTS"
    trap - EXIT HUP INT TERM
  else
    "$venv_python" -m pip install --disable-pip-version-check --no-input --no-cache-dir --no-deps --requirement "$LOCK_REQUIREMENTS"
    "$venv_python" -m pip check
    local actual_freeze="${LOCK_REQUIREMENTS}.freeze.$$"
    trap 'rm -f "$actual_freeze"' EXIT HUP INT TERM
    "$venv_python" -m pip freeze --all | LC_ALL=C sort > "$actual_freeze"
    if ! cmp -s "$LOCK_REQUIREMENTS" "$actual_freeze"; then
      printf '%s\n' "installed Exp443 environment does not match the byte-identical sorted lock" >&2
      diff -u "$LOCK_REQUIREMENTS" "$actual_freeze" >&2 || true
      return 1
    fi
  fi
  printf '%s\n' "EXP443_ENVIRONMENT_READY"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  exp443_main_antigravity "$@"
fi
