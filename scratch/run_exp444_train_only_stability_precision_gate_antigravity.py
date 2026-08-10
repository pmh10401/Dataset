"""Outside-CWD-safe fixed Exp444 lifecycle launcher."""
import os
import sys
import json
from pathlib import Path

_THREAD_ENV_NAMES_ANTIGRAVITY = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "NUMBA_NUM_THREADS",
)
for _name_antigravity in _THREAD_ENV_NAMES_ANTIGRAVITY:
    os.environ[_name_antigravity] = "1"

_REPOSITORY_ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT_ANTIGRAVITY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT_ANTIGRAVITY))

from experiments.exp444_train_only_stability_precision_gate_antigravity.cli_antigravity import main_antigravity


if __name__ == "__main__":
    if sys.argv[1:] == ["thread-probe-antigravity"]:
        print(json.dumps({name: os.environ[name] for name in _THREAD_ENV_NAMES_ANTIGRAVITY}, sort_keys=True, separators=(",", ":")))
        raise SystemExit(0)
    raise SystemExit(main_antigravity(tuple(sys.argv[1:])))
