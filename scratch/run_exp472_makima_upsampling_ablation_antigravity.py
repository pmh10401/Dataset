"""Fixed launcher for the isolated Exp472 Makima ablation."""

from pathlib import Path
import sys


_REPOSITORY_ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT_ANTIGRAVITY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT_ANTIGRAVITY))

from experiments.exp472_makima_upsampling_ablation_antigravity.cli_antigravity import main_antigravity


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
