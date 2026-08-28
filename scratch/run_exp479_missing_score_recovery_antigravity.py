#!/usr/bin/env python3
"""Command line entry point for Exp479 validation and smoke execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys

ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
if str(ROOT_ANTIGRAVITY) not in sys.path:
    sys.path.insert(0, str(ROOT_ANTIGRAVITY))

from experiments.exp474_conservative_fallback_routing_antigravity.run_antigravity import (
    CHECKPOINT_DIR_ANTIGRAVITY,
)
from experiments.exp478_full_1117_official_routing_antigravity.run_antigravity import (
    DATABASE_PATH_ANTIGRAVITY,
)
from experiments.exp479_missing_score_recovery_antigravity.population_antigravity import (
    select_recovery_population_antigravity,
)
from experiments.exp479_missing_score_recovery_antigravity.run_antigravity import (
    run_official_antigravity,
    run_smoke_antigravity,
)


def main_antigravity() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--limit", type=int, choices=(1, 2))
    parser.add_argument("--official", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        connection = sqlite3.connect(
            f"file:{DATABASE_PATH_ANTIGRAVITY}?mode=ro", uri=True
        )
        try:
            audit = select_recovery_population_antigravity(
                connection, CHECKPOINT_DIR_ANTIGRAVITY
            )
        finally:
            connection.close()
        print(json.dumps({"recovery": len(audit.datasets), "status": "ready"}))
        return 0
    if args.official:
        if args.limit is not None:
            parser.error("--official and --limit cannot be used together")
        result = run_official_antigravity(resume=args.resume)
        print(
            json.dumps(
                {
                    "state": result["state"],
                    "success": result["success"],
                    "failures": result["failures"],
                    "combined_datasets": result["combined_datasets"],
                    "recovery_summary": result["recovery_summary"],
                    "combined_summary": result["combined_summary"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.limit is None:
        parser.error("--limit 1, --limit 2, or --official is required")
    print(json.dumps(run_smoke_antigravity(args.limit, resume=args.resume), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
