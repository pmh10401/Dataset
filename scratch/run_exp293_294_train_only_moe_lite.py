#!/usr/bin/env python3
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp293_294_train_only_moe_lite.run_pipeline import main


if __name__ == "__main__":
    main()
