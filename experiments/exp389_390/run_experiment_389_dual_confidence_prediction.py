from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp389_390.dual_confidence_core import (
    EXPECTED_DATASETS,
    build_dual_tier_row,
    freeze_predictions,
    sha256_file,
)


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
SOURCE_ROOT = DATA_ROOT / "exp386_388_existing_evidence_alpha_sweep"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "exp389_390_dual_confidence_tiers"


def load_source_rows(root: Path) -> tuple[dict[str, dict], dict]:
    prediction_path = root / "exp358_frozen_predictions.jsonl"
    manifest_path = root / "exp358_prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(prediction_path) != manifest.get("prediction_sha256"):
        raise ValueError(f"source prediction SHA256 mismatch: {root}")
    if manifest.get("contains_test_labels") is not False:
        raise ValueError(f"source manifest contains TEST labels: {root}")
    if manifest.get("contains_evaluation_metrics") is not False:
        raise ValueError(f"source manifest contains evaluation metrics: {root}")
    rows = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_dataset = {str(row["dataset"]): row for row in rows}
    if len(by_dataset) != len(rows):
        raise ValueError(f"duplicate datasets in source: {root}")
    if len(rows) != int(manifest.get("rows", -1)):
        raise ValueError(f"source row count mismatch: {root}")
    metadata = {
        "prediction_file": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "manifest_file": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "datasets": len(rows),
    }
    return by_dataset, metadata


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_status(root: Path, status: str, done: int, total: int, errors: int = 0) -> None:
    (root / "exp389_status.json").write_text(
        json.dumps(
            {
                "id": "experiment_389_dual_confidence_prediction",
                "status": status,
                "phase": status,
                "datasets_done": done,
                "datasets_total": total,
                "progress_percent": 100.0 * done / total if total else 0.0,
                "errors": errors,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run(output_root: Path, dataset_limit: int | None = None) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    alpha005, metadata005 = load_source_rows(SOURCE_ROOT / "alpha_0p005")
    alpha010, metadata010 = load_source_rows(SOURCE_ROOT / "alpha_0p010")
    if set(alpha005) != set(alpha010):
        raise ValueError("alpha source dataset coverage mismatch")
    names = sorted(alpha005)
    if len(names) != EXPECTED_DATASETS:
        raise ValueError(f"expected {EXPECTED_DATASETS} source datasets, got {len(names)}")
    if dataset_limit is not None:
        names = names[:dataset_limit]
    write_status(output_root, "predicting", 0, len(names))
    rows = []
    for number, dataset in enumerate(names, start=1):
        rows.append(build_dual_tier_row(alpha005[dataset], alpha010[dataset]))
        if number == len(names) or number % 25 == 0:
            write_status(output_root, "predicting", number, len(names))
            print(
                f"Progress: [{number}/{len(names)}] rows={number} "
                f"dataset={dataset} errors=0",
                flush=True,
            )

    prediction_path = output_root / "exp389_frozen_predictions.jsonl"
    manifest_path = output_root / "exp389_prediction_manifest.json"
    manifest = freeze_predictions(
        rows,
        prediction_path,
        manifest_path,
        source_metadata={
            "alpha_0.005": metadata005,
            "alpha_0.010": metadata010,
        },
    )
    detail_rows = [
        {
            "dataset": row["dataset"],
            "n_test": row["n_test"],
            "hard_count": len(row["automatic_hard_indices"]),
            "highest_count": len(row["highest_confidence_indices"]),
            "extended_count": len(row["extended_confidence_indices"]),
            "general_count": len(row["general_confidence_indices"]),
            "standard_review_count": len(row["standard_review_indices"]),
            "priority_review_count": len(row["priority_review_indices"]),
            "hard_lane_preserved": True,
            "tier_partition_valid": True,
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
        }
        for row in rows
    ]
    write_csv(output_root / "exp389_detail.csv", detail_rows)
    write_csv(
        output_root / "exp389_summary.csv",
        [
            {
                "datasets": len(rows),
                "hard_count": sum(row["hard_count"] for row in detail_rows),
                "highest_count": sum(row["highest_count"] for row in detail_rows),
                "extended_count": sum(row["extended_count"] for row in detail_rows),
                "general_count": sum(row["general_count"] for row in detail_rows),
                "hard_lane_preserved": True,
                "review_lanes_preserved": True,
                "contains_test_labels": False,
                "contains_evaluation_metrics": False,
                "test_metrics_select_policy": False,
                "retrospective_counterfactual": True,
                "prospective_validated": False,
            }
        ],
    )
    write_status(output_root, "completed", len(rows), len(rows))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-limit", type=int)
    args = parser.parse_args()
    run(args.output_dir, args.dataset_limit)


if __name__ == "__main__":
    main()
