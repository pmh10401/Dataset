from __future__ import annotations

import argparse
import sys

STAGE_META = {
    "206": "Alternative-Source OOF Reconstruction",
    "207": "TRAIN-OOF Conditional Novelty Audit",
    "208": "Continuous Alternative-Source Fusion",
    "209": "Seven-Source Joint Normal Novelty",
    "210": "Independent Frontier Reaggregation",
}


def main():
    parser = argparse.ArgumentParser(description="Exp206-210 runner")
    parser.add_argument("--stage", choices=list(STAGE_META), required=True)
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-root", type=str)
    args = parser.parse_args()
    print(f"Running stage {args.stage}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
