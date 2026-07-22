from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def reaggregate(root: Path, experiment: int) -> list[dict[str, object]]:
    with (root / f"exp{experiment}_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with (root / f"exp{experiment}_summary.csv").open(newline="", encoding="utf-8") as handle:
        expected = {row["variant"]: row for row in csv.DictReader(handle)}
    output = []
    for variant in sorted({row["variant"] for row in rows}):
        selected = [row for row in rows if row["variant"] == variant]
        actual = {
            "variant": variant, "datasets": len(selected),
            "candidate_count": sum(int(row["candidate_count"]) for row in selected),
            "tp": sum(int(row["tp"]) for row in selected),
            "fp": sum(int(row["fp"]) for row in selected),
            "outside_all_existing_tp": sum(int(row["outside_all_existing_tp"]) for row in selected),
        }
        reference = expected[variant]
        for field in ("datasets", "candidate_count", "tp", "fp", "outside_all_existing_tp"):
            if int(actual[field]) != int(reference[field]):
                raise ValueError(f"reaggregation mismatch: {variant} {field}")
        output.append(actual)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--experiments", type=int, nargs="+", required=True); args = parser.parse_args()
    report = {}
    for experiment in args.experiments:
        roots = sorted(args.data_root.glob(f"exp{experiment}_*"))
        if len(roots) != 1: raise ValueError(f"expected one Exp{experiment} root, found {roots}")
        report[str(experiment)] = reaggregate(roots[0], experiment)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__": main()
