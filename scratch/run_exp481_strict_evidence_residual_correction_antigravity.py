#!/usr/bin/env python3
"""CLI for Exp481 validation, smoke, official run, and verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
if str(ROOT_ANTIGRAVITY) not in sys.path:
    sys.path.insert(0, str(ROOT_ANTIGRAVITY))

from experiments.exp481_strict_evidence_residual_correction_antigravity.run_antigravity import (  # noqa: E402
    run_official_antigravity,
    run_smoke_antigravity,
    run_validate_antigravity,
    verify_official_antigravity,
)


def main_antigravity() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--smoke-families", type=int)
    modes.add_argument("--official", action="store_true")
    modes.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        result = run_validate_antigravity()
    elif args.smoke_families is not None:
        result = run_smoke_antigravity(family_count=args.smoke_families)
    elif args.official:
        result = run_official_antigravity()
    else:
        result = verify_official_antigravity()
    if args.smoke_families is not None or args.official or args.verify_only:
        evaluation = result["evaluation"]
        result = {
            "state": result["state"],
            "families": result["family_count"],
            "datasets": result["dataset_count"],
            "selected_backend": result["selected_backend"],
            "fallback_reason": result["fallback_reason"],
            "family_overlap_count": evaluation["leakage"]["family_overlap_count"],
            "route_input_key_count": evaluation["leakage"]["route_input_key_count"],
            "changed_count": evaluation["summaries"]["changed"]["count"],
        }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main_antigravity()
