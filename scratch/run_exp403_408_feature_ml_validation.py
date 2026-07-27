#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import argparse


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp403_408.pipeline import main
from scratch.task13_checkpoint_reset import reset_checkpoint_state


if __name__ == "__main__":
    reset_parser = argparse.ArgumentParser(add_help=False)
    reset_parser.add_argument("--reset-checkpoint", action="store_true")
    reset_parser.add_argument("--output-dir", type=Path)
    reset_args, _unknown = reset_parser.parse_known_args()
    if reset_args.reset_checkpoint:
        if reset_args.output_dir is None:
            raise SystemExit("--reset-checkpoint requires --output-dir")
        reset_checkpoint_state(reset_args.output_dir)
    raise SystemExit(main())
