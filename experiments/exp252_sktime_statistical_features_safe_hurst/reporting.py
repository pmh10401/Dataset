from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from experiments.exp252_sktime_statistical_features_safe_hurst.config import EXPERIMENT, SOURCE


def write_report(root: Path) -> Path:
    with (root / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    lines = [
        "# Experiment 252: sktime statistical features with safe Hurst handling",
        "",
        "Retrospective score-source qualification against frozen Exp236 lanes.",
        "The autonomous Hard alert policy was not changed.",
        "",
        "| Variant | Alpha | Hard confirm | Precision | Standard | Precision | Outside TP | Outside FP | Precision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {float(row['alpha']):.3f} | "
            f"{row['hard_confirmation_count']} | {100*float(row['hard_confirmation_precision']):.2f}% | "
            f"{row['standard_priority_count']} | {100*float(row['standard_priority_precision']):.2f}% | "
            f"{row['outside_lane_tp']} | {row['outside_lane_fp']} | "
            f"{100*float(row['outside_lane_precision']):.2f}% |"
        )
    lines.extend(["", "No variant or alpha is selected for deployment from these TEST results."])
    path = root / "RESULT_REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_manifest(root: Path, command: str) -> Path:
    filenames = (
        "audit.json", "cache_manifest.csv", "frozen_predictions.jsonl",
        "prediction_manifest.json", "results.csv", "summary.csv",
        "evaluation_manifest.json", "independent_reaggregation.json", "RESULT_REPORT.md",
    )
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
             for name in filenames if (root / name).exists()}
    payload = {
        "experiment": EXPERIMENT, "source": SOURCE, "files": files,
        "execution_command": command, "scope": "retrospective_score_source_qualification",
        "autonomous_policy_changed": False,
        "known_limitations": ["reused benchmark TEST data", "no prospective validation",
                              "no operating alpha or feature variant selected"],
    }
    path = root / "MANIFEST.json"
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return path
