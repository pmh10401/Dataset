"""Direct launcher for the closed Exp473 command surface."""

from __future__ import annotations

import os
from pathlib import Path
import sys


REPOSITORY_ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "NUMBA_CACHE_DIR", "/private/tmp/exp473_numba_cache_antigravity"
)
if str(REPOSITORY_ROOT_ANTIGRAVITY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT_ANTIGRAVITY))

from experiments.exp473_contemporary_pchip_control_antigravity.cli_antigravity import (  # noqa: E402
    main_antigravity,
)


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
