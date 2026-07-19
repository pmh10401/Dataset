from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from experiments.exp249_250_official_mini_multirocket.cache_pipeline import source_root


def write_report(root: Path, experiment: int) -> Path:
    base = source_root(root, experiment)
    with (base / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    source = rows[0]["source"] if rows else "unknown"
    lines = [
        f"# Experiment {experiment}: {source}",
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
    lines.extend(
        [
            "",
            "No kernel count, alpha, variant, or source is selected for deployment from these TEST results.",
        ]
    )
    path = base / "RESULT_REPORT.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_manifest(root: Path, experiment: int, command: str) -> Path:
    base = source_root(root, experiment)
    filenames = (
        "audit.json",
        "cache_manifest.csv",
        "frozen_predictions.jsonl",
        "prediction_manifest.json",
        "results.csv",
        "summary.csv",
        "evaluation_manifest.json",
        "independent_reaggregation.json",
        "RESULT_REPORT.md",
    )
    files = {
        name: hashlib.sha256((base / name).read_bytes()).hexdigest()
        for name in filenames
        if (base / name).exists()
    }
    payload = {
        "experiment": int(experiment),
        "files": files,
        "execution_command": command,
        "scope": "retrospective_score_source_qualification",
        "autonomous_policy_changed": False,
        "known_limitations": [
            "reused benchmark TEST data",
            "no prospective validation",
            "no operating alpha or kernel count selected",
        ],
    }
    path = base / "MANIFEST.json"
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return path
