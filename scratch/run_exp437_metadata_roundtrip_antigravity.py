"""Outside-CWD-safe fixed Exp437 CLI wrapper."""
from pathlib import Path
import sys

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from experiments.exp436_missing_safe_strict_consensus_antigravity.cli_antigravity import (
    main_for_profile_antigravity,
)
from experiments.exp436_missing_safe_strict_consensus_antigravity.execution_profile_antigravity import (
    EXP437_PROFILE,
)

if __name__ == "__main__":
    raise SystemExit(main_for_profile_antigravity(EXP437_PROFILE))
