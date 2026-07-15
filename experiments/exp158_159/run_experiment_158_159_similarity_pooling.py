"""Orchestrate Exp158 and Exp159 TRAIN-only similarity pooling conformal pipeline.

This module loads the frozen cache, computes robust-scaled descriptors,
selects tie-invariant nearest neighbors, runs conformal inference,
and integrates candidate indices with the baseline lanes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EXP153_DIR = REPO_ROOT / "experiments" / "exp153_157"
if str(EXP153_DIR) not in sys.path:
    sys.path.insert(0, str(EXP153_DIR))

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import exp153_157_bridge as bridge
import exp153_157_core as core
import exp153_157_reporting as reporting
import similarity_pool_core as pool_core

DEFAULT_MANIFEST_PATH = (
    REPO_ROOT
    / "outputs"
    / "exp153_157_train_only_improvements"
    / "01_score_cache_manifest.csv"
)
DEFAULT_CACHE_DIR = (
    REPO_ROOT / "outputs" / "exp153_157_train_only_improvements" / "cache"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "exp158_159_small_train_similarity_pooling"
)

MANIFEST_SHA256 = (
    "bf73796c046a6102fce53d3b6fc606afa1ed0a5161ebb6e92d94698ab56a68a0"
)
BASELINE_PRED_SHA256 = (
    "9684f6bae6e561535f190969a4c71b1f258c97d0b7cf9899b8d9147a74c13499"
)
BASELINE_MANIFEST_SHA256 = (
    "bce61baee9f08a68e1bebef5b74ec93b230ed18634a66798b2baa293cb372b48"
)


def write_output_manifest(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or "cache" in path.parts:
            continue
        if path.name in {"MANIFEST.csv", "MANIFEST.md"}:
            continue
        rows.append(
            {
                "file": str(path.relative_to(output_dir)),
                "sha256": core.sha256_file(path),
                "retrospective_or_prospective": "retrospective_counterfactual",
                "autonomous_or_human_assisted": "hard_autonomous_review_human_assisted",
            }
        )
    if not rows:
        return
    with (output_dir / "MANIFEST.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "MANIFEST.md").write_text(
        "# Exp158--159 output manifest\n\n"
        "No TEST metric selected alpha, pooling, or similarity threshold.\n\n"
        + "\n".join(f"- `{row['file']}`: `{row['sha256']}`" for row in rows)
        + "\n",
        encoding="utf-8",
    )


def run_audit(
    output_dir: Path,
    manifest_path: Path,
    cache_dir: Path,
    dataset_limit: int | None = None,
):
    print("=== Stage: audit ===", flush=True)
    # Verify manifest hash
    core.verify_frozen_file(manifest_path, MANIFEST_SHA256)

    manifest_rows = reporting.read_rows(manifest_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []

    for row in manifest_rows:
        dataset_name = row["dataset_name"]
        n_train = int(row["n_train"])
        n_test = int(row["n_test"])
        cache_file = row["cache_file"]

        descriptor_complete = 0
        status_msg = "uninitialized"

        try:
            entry = bridge.load_cache_entry(cache_dir, row)
            train_oof_by_source = {
                s: entry[f"{s}__train_oof"] for s in bridge.SOURCE_NAMES
            }
            descriptor, status_msg = pool_core.compute_dataset_descriptor(
                train_oof_by_source
            )
            if descriptor is not None:
                descriptor_complete = 1
        except Exception as e:
            descriptor_complete = 0
            status_msg = f"exception: {e}"

        audit_rows.append(
            {
                "dataset_name": dataset_name,
                "n_train": n_train,
                "n_test": n_test,
                "descriptor_complete": descriptor_complete,
                "status_msg": status_msg,
                "cache_file": cache_file,
                "cache_sha256": row["cache_sha256"],
            }
        )

    audit_csv = output_dir / "01_descriptor_manifest.csv"
    reporting.write_rows(audit_csv, audit_rows)
    print(
        f"Audit completed: {len(audit_rows)} datasets audited. Written to {audit_csv}"
    )


def run_predict(
    output_dir: Path,
    manifest_path: Path,
    cache_dir: Path,
    dataset_limit: int | None = None,
):
    print("=== Stage: predict ===", flush=True)
    core.verify_frozen_file(manifest_path, MANIFEST_SHA256)

    audit_csv = output_dir / "01_descriptor_manifest.csv"
    if not audit_csv.exists():
        print("Audit manifest not found. Auto-running audit...")
        run_audit(output_dir, manifest_path, cache_dir, dataset_limit)

    all_manifest_rows = reporting.read_rows(manifest_path)
    donor_rows, target_rows = pool_core.partition_manifest_rows(
        all_manifest_rows, dataset_limit
    )
    expected_names = [row["dataset_name"] for row in donor_rows]

    audit_rows = reporting.read_rows(audit_csv)
    audit_map = {row["dataset_name"]: row for row in audit_rows}
    if set(audit_map) != set(expected_names):
        print("Audit coverage is incomplete. Re-running the full donor audit...", flush=True)
        run_audit(output_dir, manifest_path, cache_dir, dataset_limit=None)
        audit_rows = reporting.read_rows(audit_csv)
        audit_map = {row["dataset_name"]: row for row in audit_rows}
    if set(audit_map) != set(expected_names):
        raise ValueError(
            f"audit coverage mismatch expected={len(expected_names)} actual={len(audit_map)}"
        )

    # Identify eligible donors (all datasets with descriptor_complete == 1 and n_train >= 5)
    eligible_donors = []
    donor_descriptors = []
    donor_reference_counts_by_source = {source: [] for source in bridge.SOURCE_NAMES}

    entries = {}
    for done, row in enumerate(donor_rows, 1):
        dataset_name = row["dataset_name"]
        audit_row = audit_map.get(dataset_name)
        if audit_row is None or int(audit_row["descriptor_complete"]) == 0:
            continue
        try:
            entry = bridge.load_cache_entry(cache_dir, row)
            entries[dataset_name] = entry

            train_oof_by_source = {
                s: entry[f"{s}__train_oof"] for s in bridge.SOURCE_NAMES
            }
            desc, _ = pool_core.compute_dataset_descriptor(train_oof_by_source)
            if desc is not None:
                eligible_donors.append(dataset_name)
                donor_descriptors.append(desc)
                for source in bridge.SOURCE_NAMES:
                    donor_reference_counts_by_source[source].append(
                        len(core.finite(entry[f"{source}__train_oof"]))
                    )
        except Exception as e:
            print(f"Warning: failed to load entry {dataset_name}: {e}")
        if done % 100 == 0 or done == len(donor_rows):
            print(f"Descriptor index {done}/{len(donor_rows)}", flush=True)

    if not eligible_donors:
        raise ValueError("No eligible donors found!")

    # Robust scale donor descriptors
    descriptors_matrix = np.vstack(donor_descriptors)
    scaled_matrix, medians, scales = pool_core.robust_scale_descriptors(
        descriptors_matrix
    )

    # Load baseline predictions from Exp152 for Exp159 integration
    baseline_pred_path, baseline_manifest_path = (
        bridge.resolve_exp152_prediction_paths()
    )
    core.verify_frozen_file(baseline_pred_path, BASELINE_PRED_SHA256)
    core.verify_frozen_file(baseline_manifest_path, BASELINE_MANIFEST_SHA256)

    baseline_predictions = reporting.read_jsonl(baseline_pred_path)
    baseline_map = pool_core.validate_baseline_prediction_rows(
        baseline_predictions,
        expected_dataset_names=expected_names,
        alphas=core.ALPHAS,
    )

    exp158_rows = []
    exp159_rows = []

    for done, row in enumerate(target_rows, 1):
        dataset_name = row["dataset_name"]
        n_train = int(row["n_train"])
        n_test = int(row["n_test"])

        eligible_small_train_target = 1 if (5 <= n_train < 20) else 0

        audit_row = audit_map.get(dataset_name)
        descriptor_complete = (
            1
            if (
                audit_row is not None
                and int(audit_row["descriptor_complete"]) == 1
            )
            else 0
        )

        entry = entries.get(dataset_name)
        if entry is None and eligible_small_train_target:
            try:
                entry = bridge.load_cache_entry(cache_dir, row)
            except Exception:
                entry = None

        pool_available = 0
        donor_count = 0
        pooled_reference_count_by_source = {
            source: 0 for source in bridge.SOURCE_NAMES
        }
        donor_dataset_names = ""
        neighbor_distance_min = -1.0
        neighbor_distance_boundary = -1.0

        p_values_by_source = {}
        if eligible_small_train_target and descriptor_complete and entry is not None:
            target_desc = None
            try:
                train_oof_by_source = {
                    s: entry[f"{s}__train_oof"] for s in bridge.SOURCE_NAMES
                }
                target_desc, _ = pool_core.compute_dataset_descriptor(
                    train_oof_by_source
                )
            except Exception:
                pass

            if target_desc is not None:
                # Scale target descriptor with global donor params
                target_desc_scaled = (target_desc - medians) / scales
                distances = pool_core.compute_distances(
                    target_desc_scaled, scaled_matrix
                )
                selection = pool_core.select_donors_by_source_counts(
                    dataset_name,
                    eligible_donors,
                    distances,
                    donor_reference_counts_by_source,
                    min_pool_reference=core.MIN_POOL_REFERENCE,
                )
                if selection is not None:
                    selected_donors, min_dist, boundary_dist, selected_counts = selection
                    pool_available = 1
                    donor_count = len(selected_donors)
                    donor_dataset_names = " ".join(selected_donors)
                    neighbor_distance_min = min_dist
                    neighbor_distance_boundary = boundary_dist

                    pooled_reference_count_by_source = dict(selected_counts)

                    # Compute conformal p-values
                    for s in bridge.SOURCE_NAMES:
                        target_train_oof = entry[f"{s}__train_oof"]
                        target_test = entry[f"{s}__test"]

                        # Pool donor robust standardized train oof scores
                        pooled_refs = []
                        for name in selected_donors:
                            d_entry = entries[name]
                            d_train = d_entry[f"{s}__train_oof"]
                            d_std = pool_core.compute_source_robust_z_scores(
                                d_train
                            )
                            pooled_refs.append(d_std)

                        pooled_reference_standardized = np.concatenate(
                            pooled_refs
                        )
                        actual_count = len(pooled_reference_standardized)
                        if actual_count != pooled_reference_count_by_source[s]:
                            raise ValueError(
                                f"pooled reference count mismatch dataset={dataset_name} "
                                f"source={s} expected={pooled_reference_count_by_source[s]} "
                                f"actual={actual_count}"
                            )
                        if actual_count < core.MIN_POOL_REFERENCE:
                            raise ValueError(
                                f"pooled reference below minimum dataset={dataset_name} "
                                f"source={s} count={actual_count}"
                            )
                        test_z, _, _ = core.robust_standardize(
                            target_train_oof, target_test
                        )
                        p_vals = core.empirical_conformal_p_values(
                            pooled_reference_standardized, test_z
                        )
                        p_values_by_source[s] = p_vals

        for alpha in core.ALPHAS:
            # Baseline lanes from Exp152
            baseline_row = baseline_map[(dataset_name, alpha)]
            b_hard = core.parse_indices(baseline_row["hard_indices"])
            b_standard = core.parse_indices(baseline_row["standard_review_indices"])
            b_priority = core.parse_indices(baseline_row["priority_review_indices"])

            # Legacy blocks for Exp158 routing
            if entry is not None:
                legacy_b = set(
                    entry["legacy_block_b_indices"].astype(int).tolist()
                )
                legacy_c = set(
                    entry["legacy_block_c_indices"].astype(int).tolist()
                )
            else:
                legacy_b = set()
                legacy_c = set()

            if n_train < 5:
                calibration_status = "complete_calibration_abstention"
            elif n_train >= 20:
                calibration_status = "not_target_n_train_ge20"
            elif pool_available:
                calibration_status = "similarity_pool_available"
            else:
                calibration_status = "similarity_pool_unavailable"

            # Exp158 Conformal candidates
            candidate_indices_158 = set()
            if pool_available and p_values_by_source:
                candidate_indices_158, _ = core.agreement_indices(
                    p_values_by_source, alpha, required_sources=2
                )

            # Route Exp158 candidates into lanes
            lanes_158 = core.route_lanes(
                candidate_indices_158, legacy_b, legacy_c, n_test
            )

            lanes_159 = pool_core.integrate_similarity_review_candidates(
                baseline_hard=b_hard,
                baseline_standard=b_standard,
                baseline_priority=b_priority,
                similarity_candidates=candidate_indices_158,
                n_test=n_test,
            )

            exp158_rows.append(
                {
                    "experiment_id": "experiment_158_train_score_similarity_pooling",
                    "dataset_name": dataset_name,
                    "alpha": alpha,
                    "n_train": n_train,
                    "n_test": n_test,
                    "calibration_status": calibration_status,
                    "eligible_small_train_target": eligible_small_train_target,
                    "descriptor_complete": descriptor_complete,
                    "pool_available": pool_available,
                    "donor_count": donor_count,
                    "pooled_reference_count_by_source": json.dumps(
                        pooled_reference_count_by_source, sort_keys=True
                    ),
                    "donor_dataset_names": donor_dataset_names,
                    "neighbor_distance_min": float(neighbor_distance_min),
                    "neighbor_distance_boundary": float(
                        neighbor_distance_boundary
                    ),
                    "candidate_indices": core.format_indices(
                        candidate_indices_158
                    ),
                    "hard_indices": core.format_indices(lanes_158["hard"]),
                    "standard_review_indices": core.format_indices(
                        lanes_158["standard_review"]
                    ),
                    "priority_review_indices": core.format_indices(
                        lanes_158["priority_review"]
                    ),
                    "candidate_count": len(candidate_indices_158),
                    "hard_count": len(lanes_158["hard"]),
                    "standard_review_count": len(lanes_158["standard_review"]),
                    "priority_review_count": len(lanes_158["priority_review"]),
                    "contains_test_labels": False,
                    "contains_evaluation_metrics": False,
                    "uses_family_name": 0,
                    "uses_test_length_for_policy": 0,
                    "uses_dataset_name_for_policy": 0,
                    "target_excluded_from_pool": 1,
                    "retrospective_counterfactual": 1,
                    "prospective_validated": 0,
                    "abstained": int(n_train < 5),
                }
            )

            exp159_rows.append(
                {
                    "experiment_id": "experiment_159_similarity_pool_review_integration",
                    "dataset_name": dataset_name,
                    "alpha": alpha,
                    "n_train": n_train,
                    "n_test": n_test,
                    "calibration_status": calibration_status,
                    "eligible_small_train_target": eligible_small_train_target,
                    "descriptor_complete": descriptor_complete,
                    "pool_available": pool_available,
                    "donor_count": donor_count,
                    "pooled_reference_count_by_source": json.dumps(
                        pooled_reference_count_by_source, sort_keys=True
                    ),
                    "donor_dataset_names": donor_dataset_names,
                    "neighbor_distance_min": float(neighbor_distance_min),
                    "neighbor_distance_boundary": float(
                        neighbor_distance_boundary
                    ),
                    "candidate_indices": core.format_indices(
                        lanes_159["candidate"]
                    ),
                    "hard_indices": core.format_indices(lanes_159["hard"]),
                    "standard_review_indices": core.format_indices(
                        lanes_159["standard_review"]
                    ),
                    "priority_review_indices": core.format_indices(
                        lanes_159["priority_review"]
                    ),
                    "candidate_count": len(lanes_159["candidate"]),
                    "hard_count": len(lanes_159["hard"]),
                    "standard_review_count": len(lanes_159["standard_review"]),
                    "priority_review_count": len(lanes_159["priority_review"]),
                    "contains_test_labels": False,
                    "contains_evaluation_metrics": False,
                    "uses_family_name": 0,
                    "uses_test_length_for_policy": 0,
                    "uses_dataset_name_for_policy": 0,
                    "target_excluded_from_pool": 1,
                    "retrospective_counterfactual": 1,
                    "prospective_validated": 0,
                    "abstained": int(n_train < 5),
                }
            )
        if done % 100 == 0 or done == len(target_rows):
            print(f"Prediction targets {done}/{len(target_rows)}", flush=True)

    exp158_pred_path = output_dir / "02_exp158_predictions.jsonl"
    exp158_manifest_path = output_dir / "03_exp158_prediction_manifest.json"
    reporting.write_jsonl(exp158_pred_path, exp158_rows)
    reporting.freeze_prediction_file(
        exp158_pred_path,
        exp158_manifest_path,
        experiment_ids=["experiment_158_train_score_similarity_pooling"],
        row_count=len(target_rows) * len(core.ALPHAS),
        input_hashes={
            "score_cache_manifest": core.sha256_file(manifest_path)
        },
    )

    exp159_pred_path = output_dir / "04_exp159_predictions.jsonl"
    exp159_manifest_path = output_dir / "05_exp159_prediction_manifest.json"
    reporting.write_jsonl(exp159_pred_path, exp159_rows)
    reporting.freeze_prediction_file(
        exp159_pred_path,
        exp159_manifest_path,
        experiment_ids=["experiment_159_similarity_pool_review_integration"],
        row_count=len(target_rows) * len(core.ALPHAS),
        input_hashes={
            "score_cache_manifest": core.sha256_file(manifest_path),
            "baseline_predictions": core.sha256_file(baseline_pred_path),
        },
    )
    print("Predictions and manifests written successfully.")


def run_evaluate(output_dir: Path):
    print("=== Stage: evaluate ===", flush=True)

    # Exp158 evaluation
    reporting.evaluate_prediction_file(
        output_dir / "02_exp158_predictions.jsonl",
        output_dir / "03_exp158_prediction_manifest.json",
        output_dir / "06_exp158_results.csv",
        output_dir / "07_exp158_summary.csv",
        output_dir / "08_exp158_summary.md",
        group_fields=("experiment_id", "alpha"),
    )

    # Exp159 evaluation
    reporting.evaluate_prediction_file(
        output_dir / "04_exp159_predictions.jsonl",
        output_dir / "05_exp159_prediction_manifest.json",
        output_dir / "09_exp159_results.csv",
        output_dir / "10_exp159_summary.csv",
        output_dir / "11_exp159_summary.md",
        group_fields=("experiment_id", "alpha"),
    )
    print("Evaluation completed successfully.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        action="append",
        choices=("audit", "predict", "evaluate", "manifest", "all"),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest_path)
    cache_dir = Path(args.cache_dir)

    stages = args.stage or ["all"]
    if "all" in stages:
        stages = ["audit", "predict", "evaluate", "manifest"]

    for stage in stages:
        if stage == "audit":
            run_audit(
                output_dir,
                manifest_path,
                cache_dir,
                dataset_limit=args.dataset_limit,
            )
        elif stage == "predict":
            run_predict(
                output_dir,
                manifest_path,
                cache_dir,
                dataset_limit=args.dataset_limit,
            )
        elif stage == "evaluate":
            run_evaluate(output_dir)
        elif stage == "manifest":
            print("=== Stage: manifest ===", flush=True)
            write_output_manifest(output_dir)
        else:
            raise ValueError(f"Unknown stage: {stage}")


if __name__ == "__main__":
    main()
