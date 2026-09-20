"""Outside-CWD-safe fixed Exp442 lifecycle launcher."""
from pathlib import Path
import sys


_REPOSITORY_ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT_ANTIGRAVITY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT_ANTIGRAVITY))

from experiments.exp442_terminal_eta_publication_repair_antigravity.cli_antigravity import (
    main_antigravity,
)


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
