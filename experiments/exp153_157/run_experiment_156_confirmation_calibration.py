"""Exp156: Block-B/Block-C confirmation calibration sensitivity.

The candidate universe remains the frozen Exp152 candidate set.  Only the
independent confirmation policy changes.  Legacy count-cap and two TRAIN-only
conformal confirmation variants are reported together; no TEST metric selects
one of them.  Priority review remains human-assisted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import exp153_157_bridge as bridge
import exp153_157_core as core
import exp153_157_reporting as reporting

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/exp153_157_train_only_improvements"
EXPERIMENT_ID = "experiment_156_confirmation_calibration"


def _load_frozen_exp152_predictions():
    prediction_path, manifest_path = bridge.resolve_exp152_prediction_paths()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("exp152_prediction_sha256") or manifest.get("prediction_sha256")
    if not expected:
        raise ValueError("Exp152 manifest does not contain an Exp152 prediction hash")
    core.verify_frozen_file(prediction_path, expected)
    rows = reporting.read_jsonl(prediction_path)
    return prediction_path, manifest_path, rows


def _confirmation_indices(cache, variant):
    if variant["block_b_method"] == "countcap":
        block_b = set(cache["legacy_block_b_indices"].astype(int).tolist())
    else:
        p_b = core.empirical_conformal_p_values(
            cache["block_b_train_oof"], cache["block_b_test"]
        )
        block_b = set(
            np.flatnonzero(p_b <= float(variant["block_b_level"]))
            .astype(int)
            .tolist()
        )

    if variant["block_c_method"] == "countcap":
        block_c = set(cache["legacy_block_c_indices"].astype(int).tolist())
    else:
        p_c = core.empirical_conformal_p_values(
            cache["block_c_train_oof"], cache["block_c_test"]
        )
        block_c = set(
            np.flatnonzero(p_c <= float(variant["block_c_level"]))
            .astype(int)
            .tolist()
        )
    return block_b, block_c


def predict_confirmation_rows(base_rows, manifest_rows, cache_dir: Path):
    """Create label-free confirmation-policy predictions."""

    manifest_by_dataset = {row["dataset_name"]: row for row in manifest_rows}
    cache_by_dataset = {}
    confirmation_by_dataset = {}
    output = []
    for base_row in base_rows:
        dataset = str(base_row["dataset_name"])
        if dataset not in cache_by_dataset:
            cache_by_dataset[dataset] = bridge.load_cache_entry(
                cache_dir, manifest_by_dataset[dataset]
            )
            confirmation_by_dataset[dataset] = {
                variant["name"]: _confirmation_indices(
                    cache_by_dataset[dataset], variant
                )
                for variant in core.CONFIRMATION_VARIANTS
            }
        cache = cache_by_dataset[dataset]
        n_test = int(cache["metadata"]["n_test"])
        candidates = core.parse_indices(base_row.get("candidate_indices"))
        for variant in core.CONFIRMATION_VARIANTS:
            block_b, block_c = confirmation_by_dataset[dataset][variant["name"]]
            lanes = core.route_lanes(candidates, block_b, block_c, n_test)
            output.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "variant": variant["name"],
                    "candidate_source_variant": base_row.get("variant", "exp152_b2_source_v2"),
                    "dataset_name": dataset,
                    "alpha": float(base_row["alpha"]),
                    "block_b_method": variant["block_b_method"],
                    "block_b_level": variant["block_b_level"],
                    "block_c_method": variant["block_c_method"],
                    "block_c_level": variant["block_c_level"],
                    "n_train": int(cache["metadata"]["n_train"]),
                    "n_test": n_test,
                    "abstained": int(cache["metadata"]["abstain"]),
                    "candidate_indices": core.format_indices(candidates),
                    "hard_indices": core.format_indices(lanes["hard"]),
                    "standard_review_indices": core.format_indices(lanes["standard_review"]),
                    "priority_review_indices": core.format_indices(lanes["priority_review"]),
                    "candidate_count": len(candidates),
                    "contains_test_labels": False,
                    "uses_family_name": 0,
                    "uses_test_length": 0,
                    "retrospective_counterfactual": 1,
                }
            )
    return output


def run_prediction(output_dir: Path = DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    source_prediction, source_manifest, base_rows = _load_frozen_exp152_predictions()
    cache_manifest = reporting.read_rows(output_dir / "01_score_cache_manifest.csv")
    rows = predict_confirmation_rows(base_rows, cache_manifest, output_dir / "cache")
    rows.sort(key=lambda row: (row["variant"], float(row["alpha"]), row["dataset_name"]))
    prediction = output_dir / "16_exp156_predictions.jsonl"
    prediction_manifest = output_dir / "17_exp156_prediction_manifest.json"
    reporting.write_jsonl(prediction, rows)
    reporting.freeze_prediction_file(
        prediction,
        prediction_manifest,
        experiment_ids=[EXPERIMENT_ID],
        row_count=len(base_rows) * len(core.CONFIRMATION_VARIANTS),
        input_hashes={
            "exp152_prediction": core.sha256_file(source_prediction),
            "exp152_prediction_manifest": core.sha256_file(source_manifest),
            "score_cache_manifest": core.sha256_file(output_dir / "01_score_cache_manifest.csv"),
        },
    )
    return prediction


def run_evaluation(output_dir: Path = DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    return reporting.evaluate_prediction_file(
        output_dir / "16_exp156_predictions.jsonl",
        output_dir / "17_exp156_prediction_manifest.json",
        output_dir / "18_exp156_results.csv",
        output_dir / "19_exp156_summary.csv",
        output_dir / "20_exp156_summary.md",
        group_fields=("experiment_id", "variant", "alpha"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("predict", "evaluate", "both"), default="both")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if args.phase in {"predict", "both"}:
        run_prediction(output_dir)
    if args.phase in {"evaluate", "both"}:
        run_evaluation(output_dir)


if __name__ == "__main__":
    main()
