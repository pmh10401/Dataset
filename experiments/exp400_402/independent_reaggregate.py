from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _calculate(
    candidates: Sequence[Mapping[str, str]],
    workload_type: str,
    workload_k: int,
) -> tuple[int, int, int, float]:
    if workload_type == "global_k":
        selected = sorted(
            candidates,
            key=lambda row: (
                int(row["rank"]),
                row["dataset"],
                int(row["candidate_index"]),
            ),
        )[:workload_k]
    elif workload_type == "per_dataset_top1":
        grouped: dict[str, list[Mapping[str, str]]] = {}
        for row in candidates:
            grouped.setdefault(row["dataset"], []).append(row)
        selected = [
            min(
                rows,
                key=lambda row: (
                    int(row["rank"]),
                    int(row["candidate_index"]),
                ),
            )
            for rows in grouped.values()
            if rows
        ]
    else:
        raise ValueError(f"unknown workload type: {workload_type}")
    reviewed = len(selected)
    tp = sum(_truth(row["is_true_anomaly"]) for row in selected)
    fp = reviewed - tp
    return reviewed, tp, fp, tp / reviewed if reviewed else 0.0


def run_independent_reaggregate(output_root: Path) -> dict[str, object]:
    output_root = Path(output_root)
    candidates = _read_csv(output_root / "exp401_candidate_evaluation.csv")
    summaries = _read_csv(output_root / "exp401_precision_at_k.csv")
    grouped: dict[tuple[str, float], list[dict[str, str]]] = {}
    for row in candidates:
        grouped.setdefault((row["variant"], float(row["alpha"])), []).append(row)

    universes = {
        frozenset((row["dataset"], int(row["candidate_index"])) for row in rows)
        for rows in grouped.values()
    }
    mismatches: list[dict[str, object]] = []
    if len(universes) != 1:
        mismatches.append({"kind": "candidate_universe", "actual": len(universes)})

    for row in summaries:
        key = (row["variant"], float(row["alpha"]))
        reviewed, tp, fp, precision = _calculate(
            grouped[key],
            row["workload_type"],
            int(row["workload_k"]),
        )
        expected = {
            "reviewed": int(row["reviewed"]),
            "tp": int(row["tp_at_k"]),
            "fp": int(row["fp_at_k"]),
            "precision": float(row["precision_at_k"]),
        }
        actual = {
            "reviewed": reviewed,
            "tp": tp,
            "fp": fp,
            "precision": precision,
        }
        if (
            expected["reviewed"] != actual["reviewed"]
            or expected["tp"] != actual["tp"]
            or expected["fp"] != actual["fp"]
            or abs(expected["precision"] - actual["precision"]) > 1e-12
        ):
            mismatches.append(
                {
                    "kind": "summary_metric",
                    "variant": key[0],
                    "alpha": key[1],
                    "workload_type": row["workload_type"],
                    "workload_k": int(row["workload_k"]),
                    "expected": expected,
                    "actual": actual,
                }
            )

    report: dict[str, object] = {
        "experiment_id": "experiment_401_exp397_shadow_replay_evaluation",
        "candidate_rows": len(candidates),
        "summary_rows": len(summaries),
        "candidate_universe_count": len(next(iter(universes), frozenset())),
        "candidate_universe_variants": len(universes),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "retrospective": True,
        "prospective": False,
    }
    (output_root / "independent_consistency_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if mismatches:
        raise ValueError(f"independent reaggregation found {len(mismatches)} mismatches")
    return report
