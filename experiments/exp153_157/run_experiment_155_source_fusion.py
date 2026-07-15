"""Exp155: source independence audit and continuous fusion sensitivity.

The experiment compares the current discrete 2-of-4 vote with two continuous
TRAIN-only meta-scores.  Correlation weights are estimated only from TRAIN OOF
scores.  No fusion method is selected with TEST metrics.
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
EXPERIMENT_ID = "experiment_155_source_independence_continuous_fusion"
VARIANTS = ("discrete_2of4", "top2_mean_meta", "corr_weighted_top2_meta")


def _matrices(cache):
    train_columns = []
    test_columns = []
    for source in bridge.SOURCE_NAMES:
        train = np.asarray(cache[f"{source}__train_oof"], dtype=float)
        test = np.asarray(cache[f"{source}__test"], dtype=float)
        train_z, location, scale = core.robust_standardize(train, train)
        test_z = (test - location) / scale
        train_columns.append(train_z)
        test_columns.append(test_z)
    return np.column_stack(train_columns), np.column_stack(test_columns)


def predict_dataset(dataset_name: str, cache):
    """Create label-free Exp155 candidate and lane rows."""

    metadata = cache["metadata"]
    n_test = int(metadata["n_test"])
    abstain = bool(int(metadata["abstain"]))
    legacy_b = set(cache["legacy_block_b_indices"].astype(int).tolist())
    legacy_c = set(cache["legacy_block_c_indices"].astype(int).tolist())

    audit = {
        "dataset_name": dataset_name,
        "n_train": int(metadata["n_train"]),
        "n_test": n_test,
        "abstained": int(abstain),
    }
    candidate_p_values = {}
    if not abstain:
        train_matrix, test_matrix = _matrices(cache)
        corr = core.spearman_correlation_matrix(train_matrix)
        weights, _ = core.inverse_correlation_weights(train_matrix)
        for i, source_i in enumerate(bridge.SOURCE_NAMES):
            audit[f"weight__{source_i}"] = float(weights[i])
            for j in range(i + 1, len(bridge.SOURCE_NAMES)):
                source_j = bridge.SOURCE_NAMES[j]
                audit[f"spearman__{source_i}__{source_j}"] = float(corr[i, j])

        source_local_p = {
            source: core.empirical_conformal_p_values(
                cache[f"{source}__train_oof"], cache[f"{source}__test"]
            )
            for source in bridge.SOURCE_NAMES
        }
        candidate_p_values["discrete_2of4"] = source_local_p
        top2_p, _, _ = core.meta_conformal_p_values(
            train_matrix, test_matrix, method="top2_mean"
        )
        weighted_p, _, _ = core.meta_conformal_p_values(
            train_matrix, test_matrix, method="corr_weighted_top2"
        )
        candidate_p_values["top2_mean_meta"] = top2_p
        candidate_p_values["corr_weighted_top2_meta"] = weighted_p
    else:
        audit["audit_status"] = "insufficient_calibration_n_train_lt5"

    rows = []
    for alpha in core.ALPHAS:
        for variant in VARIANTS:
            if abstain:
                candidates = set()
                evidence_count = np.zeros(n_test, dtype=int)
            elif variant == "discrete_2of4":
                candidates, evidence_count = core.agreement_indices(
                    candidate_p_values[variant],
                    alpha,
                    core.REQUIRED_SOURCE_AGREEMENT,
                )
            else:
                p_values = np.asarray(candidate_p_values[variant], dtype=float)
                candidates = set(np.flatnonzero(p_values <= alpha).astype(int).tolist())
                evidence_count = np.asarray(p_values <= alpha, dtype=int)
            lanes = core.route_lanes(candidates, legacy_b, legacy_c, n_test)
            rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "variant": variant,
                    "dataset_name": dataset_name,
                    "alpha": alpha,
                    "n_train": int(metadata["n_train"]),
                    "n_test": n_test,
                    "abstained": int(abstain),
                    "candidate_indices": core.format_indices(candidates),
                    "hard_indices": core.format_indices(lanes["hard"]),
                    "standard_review_indices": core.format_indices(lanes["standard_review"]),
                    "priority_review_indices": core.format_indices(lanes["priority_review"]),
                    "candidate_count": len(candidates),
                    "evidence_count_distribution": json.dumps(
                        {
                            str(value): int(np.sum(evidence_count == value))
                            for value in sorted(set(evidence_count.tolist()))
                        },
                        sort_keys=True,
                    ),
                    "contains_test_labels": False,
                    "uses_family_name": 0,
                    "uses_test_length": 0,
                    "retrospective_counterfactual": 1,
                }
            )
    return rows, audit


def run_prediction(output_dir: Path = DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    manifest = reporting.read_rows(output_dir / "01_score_cache_manifest.csv")
    cache_dir = output_dir / "cache"
    rows, audits = [], []
    for done, manifest_row in enumerate(manifest, 1):
        cache = bridge.load_cache_entry(cache_dir, manifest_row)
        prediction_rows, audit = predict_dataset(manifest_row["dataset_name"], cache)
        rows.extend(prediction_rows)
        audits.append(audit)
        if done % 50 == 0 or done == len(manifest):
            print(f"Exp155 prediction {done}/{len(manifest)}", flush=True)
    rows.sort(key=lambda row: (row["variant"], float(row["alpha"]), row["dataset_name"]))
    reporting.write_rows(output_dir / "10_exp155_source_independence_audit.csv", audits)
    prediction = output_dir / "11_exp155_predictions.jsonl"
    prediction_manifest = output_dir / "12_exp155_prediction_manifest.json"
    reporting.write_jsonl(prediction, rows)
    reporting.freeze_prediction_file(
        prediction,
        prediction_manifest,
        experiment_ids=[EXPERIMENT_ID],
        row_count=len(manifest) * len(core.ALPHAS) * len(VARIANTS),
        input_hashes={
            "score_cache_manifest": core.sha256_file(output_dir / "01_score_cache_manifest.csv")
        },
    )
    return prediction


def run_evaluation(output_dir: Path = DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    return reporting.evaluate_prediction_file(
        output_dir / "11_exp155_predictions.jsonl",
        output_dir / "12_exp155_prediction_manifest.json",
        output_dir / "13_exp155_results.csv",
        output_dir / "14_exp155_summary.csv",
        output_dir / "15_exp155_summary.md",
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
