#!/usr/bin/env python3
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp292_standard_review_evidence_tiering.run_pipeline import main


if __name__ == "__main__":
    main()

