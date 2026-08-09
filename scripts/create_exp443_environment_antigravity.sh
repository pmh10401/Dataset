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

# The official CPU route is launched with every common BLAS/OpenMP control set
# explicitly.  These are process-local settings; no project .venv is edited.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

umask 077
VENV_PARENT="$(dirname -- "$VENV_ROOT")"
if [[ -e "$VENV_PARENT" && ! -d "$VENV_PARENT" ]]; then
  printf '%s\n' "Exp443 venv parent is not a directory; refusing overwrite" >&2
  exit 1
fi
if [[ ! -e "$VENV_PARENT" ]]; then
  mkdir -p -m 700 "$VENV_PARENT"
fi

if [[ -e "$VENV_ROOT" ]]; then
  # A pre-existing environment is read-only from this script's perspective.
  # Validate it completely before any chmod, pip invocation, or installation.
  if (( BOOTSTRAP == 1 )); then
    printf '%s\n' "bootstrap is creation-only; an existing Exp443 environment is never mutated" >&2
    exit 1
  fi
  if [[ -L "$VENV_ROOT" || ! -d "$VENV_ROOT" || ! -x "$VENV_ROOT/bin/python" ]]; then
    printf '%s\n' "existing Exp443 environment is not a valid Python venv; refusing overwrite" >&2
    exit 1
  fi
  EXISTING_MODE="$(stat -f '%Lp' "$VENV_ROOT" 2>/dev/null || stat -c '%a' "$VENV_ROOT")"
  if [[ "$EXISTING_MODE" != "700" ]]; then
    printf '%s\n' "existing Exp443 venv mode is $EXISTING_MODE, expected 700; refusing overwrite" >&2
    exit 1
  fi
  VENV_PYTHON="$VENV_ROOT/bin/python"
  EXISTING_VERSION="$($VENV_PYTHON -c 'import platform; print(platform.python_version())' 2>/dev/null || true)"
  if [[ "$EXISTING_VERSION" != "3.14.6" ]]; then
    printf '%s\n' "existing Exp443 venv has Python $EXISTING_VERSION; refusing overwrite" >&2
    exit 1
  fi
  EXISTING_PREFIX="$($VENV_PYTHON -c 'import sys; print(sys.prefix); print(sys.base_prefix)' 2>/dev/null || true)"
  if [[ "$EXISTING_PREFIX" != "$VENV_ROOT"$'\n'* || "$EXISTING_PREFIX" == "$VENV_ROOT"$'\n'"$VENV_ROOT" ]]; then
    printf '%s\n' "existing Exp443 environment is not an isolated venv; refusing overwrite" >&2
    exit 1
  fi
  # Distribution metadata is checked with the venv interpreter before pip is
  # touched, so a package-mismatched venv also fails without mutation.
  if ! "$VENV_PYTHON" - "$LOCK_REQUIREMENTS" <<'PY'
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
    exit 1
  fi
  ACTUAL_FREEZE="${LOCK_REQUIREMENTS}.preflight.$$"
  trap 'rm -f "$ACTUAL_FREEZE"' EXIT HUP INT TERM
  "$VENV_PYTHON" -m pip check
  "$VENV_PYTHON" -m pip freeze --all | LC_ALL=C sort > "$ACTUAL_FREEZE"
  if ! cmp -s "$LOCK_REQUIREMENTS" "$ACTUAL_FREEZE"; then
    printf '%s\n' "existing Exp443 environment does not match the byte-identical sorted lock" >&2
    diff -u "$LOCK_REQUIREMENTS" "$ACTUAL_FREEZE" >&2 || true
    exit 1
  fi
  rm -f "$ACTUAL_FREEZE"
  trap - EXIT HUP INT TERM
  printf '%s\n' "EXP443_ENVIRONMENT_READY"
  exit 0
else
  "$PYTHON_BIN" -m venv "$VENV_ROOT"
  chmod 700 "$VENV_ROOT"
  VENV_PYTHON="$VENV_ROOT/bin/python"
fi

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
