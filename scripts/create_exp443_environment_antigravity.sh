#!/usr/bin/env bash
set -euo pipefail

# Exp443's environment is deliberately outside the repository.  This script
# never activates or edits a project .venv and never selects a different Python
# based on PATH ordering.
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
DIRECT_REQUIREMENTS="${REPO_ROOT}/experiments/exp443_missing_normal_train_policy_comparison_antigravity/requirements_antigravity.txt"
LOCK_REQUIREMENTS="${REPO_ROOT}/experiments/exp443_missing_normal_train_policy_comparison_antigravity/requirements_lock_antigravity.txt"
VENV_ROOT="/Users/mac/Documents/Dataset/venvs/exp443_minirocket_antigravity_py314"
PYTHON_BIN="/usr/local/bin/python3.14"
BOOTSTRAP_FLAG="--bootstrap-lock-antigravity"

if [[ "$#" -gt 1 || ( "$#" -eq 1 && "$1" != "$BOOTSTRAP_FLAG" ) ]]; then
  printf '%s\n' "usage: $0 [--bootstrap-lock-antigravity]" >&2
  exit 2
fi
BOOTSTRAP=0
if [[ "$#" -eq 1 ]]; then BOOTSTRAP=1; fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf '%s\n' "Exp443 requires $PYTHON_BIN (Python 3.14.6)" >&2
  exit 1
fi
PYTHON_VERSION="$($PYTHON_BIN -c 'import platform; print(platform.python_version())')"
if [[ "$PYTHON_VERSION" != "3.14.6" ]]; then
  printf '%s\n' "Exp443 requires Python 3.14.6, got $PYTHON_VERSION" >&2
  exit 1
fi
if [[ "$VENV_ROOT" == "$REPO_ROOT/.venv" || "$VENV_ROOT" == "$REPO_ROOT/.venv/"* ]]; then
  printf '%s\n' "refusing to use the project .venv" >&2
  exit 1
fi
if [[ ! -f "$DIRECT_REQUIREMENTS" ]]; then
  printf '%s\n' "direct requirements are missing: $DIRECT_REQUIREMENTS" >&2
  exit 1
fi
EXPECTED_DIRECT=$'aeon==1.5.0\nnumpy==2.3.5\nscikit-learn==1.8.0\n'
if [[ "$(<"$DIRECT_REQUIREMENTS")"$'\n' != "$EXPECTED_DIRECT" ]]; then
  printf '%s\n' "direct requirements do not match the frozen three-line contract" >&2
  exit 1
fi
if (( BOOTSTRAP == 1 )) && [[ -e "$LOCK_REQUIREMENTS" ]]; then
  printf '%s\n' "bootstrap is allowed only while the tracked lock is absent" >&2
  exit 1
fi
if (( BOOTSTRAP == 0 )) && [[ ! -f "$LOCK_REQUIREMENTS" ]]; then
  printf '%s\n' "tracked lock is absent; use --bootstrap-lock-antigravity once" >&2
  exit 1
fi

umask 077
VENV_PARENT="$(dirname -- "$VENV_ROOT")"
mkdir -p -m 700 "$VENV_PARENT"
if [[ -e "$VENV_ROOT" ]]; then
  if [[ ! -d "$VENV_ROOT" || ! -x "$VENV_ROOT/bin/python" ]]; then
    printf '%s\n' "existing Exp443 environment is not a valid Python venv; refusing overwrite" >&2
    exit 1
  fi
  chmod 700 "$VENV_ROOT"
  EXISTING_VERSION="$($VENV_ROOT/bin/python -c 'import platform; print(platform.python_version())' 2>/dev/null || true)"
  if [[ "$EXISTING_VERSION" != "3.14.6" ]]; then
    printf '%s\n' "existing Exp443 venv has Python $EXISTING_VERSION; refusing overwrite" >&2
    exit 1
  fi
else
  "$PYTHON_BIN" -m venv "$VENV_ROOT"
  chmod 700 "$VENV_ROOT"
fi
VENV_PYTHON="$VENV_ROOT/bin/python"

# Pin the package manager before resolving anything else.
"$VENV_PYTHON" -m pip install --disable-pip-version-check --no-input --upgrade --no-cache-dir 'pip==26.1.2'
if (( BOOTSTRAP == 1 )); then
  "$VENV_PYTHON" -m pip install --disable-pip-version-check --no-input --no-cache-dir --requirement "$DIRECT_REQUIREMENTS"
  "$VENV_PYTHON" -m pip check
  TMP_LOCK="${LOCK_REQUIREMENTS}.tmp.$$"
  trap 'rm -f "$TMP_LOCK"' EXIT HUP INT TERM
  "$VENV_PYTHON" -m pip freeze --all | LC_ALL=C sort > "$TMP_LOCK"
  if [[ ! -s "$TMP_LOCK" ]]; then
    printf '%s\n' "pip freeze produced an empty Exp443 lock" >&2
    exit 1
  fi
  # The lock is created only once.  A concurrent winner is never replaced.
  if ! ( set -C; : > "$LOCK_REQUIREMENTS" ) 2>/dev/null; then
    printf '%s\n' "lock appeared concurrently; refusing overwrite" >&2
    exit 1
  fi
  mv "$TMP_LOCK" "$LOCK_REQUIREMENTS"
  trap - EXIT HUP INT TERM
else
  "$VENV_PYTHON" -m pip install --disable-pip-version-check --no-input --no-cache-dir --no-deps --requirement "$LOCK_REQUIREMENTS"
  "$VENV_PYTHON" -m pip check
  ACTUAL_FREEZE="${LOCK_REQUIREMENTS}.freeze.$$"
  trap 'rm -f "$ACTUAL_FREEZE"' EXIT HUP INT TERM
  "$VENV_PYTHON" -m pip freeze --all | LC_ALL=C sort > "$ACTUAL_FREEZE"
  if ! cmp -s "$LOCK_REQUIREMENTS" "$ACTUAL_FREEZE"; then
    printf '%s\n' "installed Exp443 environment does not match the byte-identical sorted lock" >&2
    diff -u "$LOCK_REQUIREMENTS" "$ACTUAL_FREEZE" >&2 || true
    exit 1
  fi
fi

printf '%s\n' "EXP443_ENVIRONMENT_READY"
