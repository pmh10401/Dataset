"""Outside-CWD launcher for the reviewed Exp455 command boundary."""

from __future__ import annotations

from pathlib import Path
import sys


_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ANTIGRAVITY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ANTIGRAVITY))

from experiments.exp455_exp454_conservative_label_free_routing_antigravity.cli_antigravity import main_antigravity


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
