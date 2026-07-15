"""Exp154: small-sample pooled/hierarchical conformal sensitivity experiment.

Four policies are reported together and none is selected with TEST metrics:

1. local_baseline: Exp152-style dataset-specific conformal calibration;
2. global_pool: leave-one-dataset-out robust-z pool across every dataset;
3. group_pool: leave-one-dataset-out pool inside TRAIN-only count/length bins;
4. hybrid_fallback: local when resolvable, then group, then global pool.

Family names and TEST statistics are never used to form a pool.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import exp153_157_bridge as bridge
import exp153_157_core as core
import exp153_157_reporting as reporting

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/exp153_157_train_only_improvements"
EXPERIMENT_ID = "experiment_154_small_sample_pooled_conformal"
VARIANTS = ("local_baseline", "global_pool", "group_pool", "hybrid_fallback")


def _load_entries(output_dir: Path):
    manifest = reporting.read_rows(output_dir / "01_score_cache_manifest.csv")
    cache_dir = output_dir / "cache"
    entries = {
        row["dataset_name"]: {
            "manifest": row,
            "cache": bridge.load_cache_entry(cache_dir, row),
        }
        for row in manifest
    }
    return manifest, entries


def _prepare_pools(entries):
    target_lengths = np.asarray(
        [int(item["cache"]["metadata"]["train_target_len"]) for item in entries.values()],
        dtype=float,
    )
    length_boundaries = tuple(np.quantile(target_lengths, [0.25, 0.50, 0.75]).tolist())
    standardized: dict[str, dict[str, np.ndarray]] = {
        source: {} for source in bridge.SOURCE_NAMES
    }
    group_members: dict[str, list[str]] = defaultdict(list)
    for dataset, item in entries.items():
        metadata = item["cache"]["metadata"]
        group = core.train_only_group_key(
            int(metadata["n_train"]),
            int(metadata["train_target_len"]),
            length_boundaries,
        )
        item["group_key"] = group
        group_members[group].append(dataset)
        if int(metadata["abstain"]):
            continue
        for source in bridge.SOURCE_NAMES:
            train = item["cache"][f"{source}__train_oof"]
            standardized[source][dataset] = core.robust_standardize(train, train)[0]

    global_references = {
        source: np.concatenate(list(mapping.values())) if mapping else np.asarray([])
        for source, mapping in standardized.items()
    }
    group_references: dict[str, dict[str, np.ndarray]] = {
        source: {} for source in bridge.SOURCE_NAMES
    }
    for source, mapping in standardized.items():
        for group, datasets in group_members.items():
            pieces = [mapping[name] for name in datasets if name in mapping]
            group_references[source][group] = (
                np.concatenate(pieces) if pieces else np.asarray([], dtype=float)
            )
    return length_boundaries, standardized, global_references, group_references


def predict_dataset(
    dataset_name: str,
    entry,
    standardized,
    global_references,
    group_references,
):
    """Create label-free Exp154 prediction rows for one dataset."""

    cache = entry["cache"]
    metadata = cache["metadata"]
    n_test = int(metadata["n_test"])
    abstain = bool(int(metadata["abstain"]))
    group = entry["group_key"]
    legacy_b = set(cache["legacy_block_b_indices"].astype(int).tolist())
    legacy_c = set(cache["legacy_block_c_indices"].astype(int).tolist())

    source_p: dict[str, dict[str, np.ndarray]] = {}
    pool_sizes: dict[str, dict[str, int]] = {}
    if not abstain:
        for source in bridge.SOURCE_NAMES:
            train = np.asarray(cache[f"{source}__train_oof"], dtype=float)
            test = np.asarray(cache[f"{source}__test"], dtype=float)
            train_z = standardized[source][dataset_name]
            test_z = core.robust_standardize(train, test)[0]
            global_ref = global_references[source]
            group_ref = group_references[source][group]
            global_remaining = len(global_ref) - len(train_z)
            group_remaining = len(group_ref) - len(train_z)
            source_p[source] = {
                "local_baseline": core.empirical_conformal_p_values(train, test),
                "global_pool": core.excluded_pool_conformal_p_values(
                    global_ref, train_z, test_z
                )
                if global_remaining >= core.MIN_POOL_REFERENCE
                else np.ones(n_test),
                "group_pool": core.excluded_pool_conformal_p_values(
                    group_ref, train_z, test_z
                )
                if group_remaining >= core.MIN_POOL_REFERENCE
                else np.ones(n_test),
            }
            pool_sizes[source] = {
                "global": global_remaining,
                "group": group_remaining,
            }

    rows = []
    for alpha in core.ALPHAS:
        for variant in VARIANTS:
            p_values = {}
            calibration_sources = {}
            if not abstain:
                for source in bridge.SOURCE_NAMES:
                    if variant == "hybrid_fallback":
                        train = cache[f"{source}__train_oof"]
                        if core.local_minimum_p(train) <= alpha:
                            p_values[source] = source_p[source]["local_baseline"]
                            calibration_sources[source] = "local"
                        elif pool_sizes[source]["group"] >= core.MIN_POOL_REFERENCE:
                            p_values[source] = source_p[source]["group_pool"]
                            calibration_sources[source] = "group_pool"
                        elif pool_sizes[source]["global"] >= core.MIN_POOL_REFERENCE:
                            p_values[source] = source_p[source]["global_pool"]
                            calibration_sources[source] = "global_pool"
                        else:
                            p_values[source] = np.ones(n_test)
                            calibration_sources[source] = "unavailable"
                    else:
                        p_values[source] = source_p[source][variant]
                        calibration_sources[source] = variant
                candidates, agreement_count = core.agreement_indices(
                    p_values, alpha, core.REQUIRED_SOURCE_AGREEMENT
                )
            else:
                candidates = set()
                agreement_count = np.zeros(n_test, dtype=int)
                calibration_sources = {source: "abstain" for source in bridge.SOURCE_NAMES}
            lanes = core.route_lanes(candidates, legacy_b, legacy_c, n_test)
            rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "variant": variant,
                    "dataset_name": dataset_name,
                    "alpha": alpha,
                    "group_key": group,
                    "n_train": int(metadata["n_train"]),
                    "n_test": n_test,
                    "abstained": int(abstain),
                    "candidate_indices": core.format_indices(candidates),
                    "hard_indices": core.format_indices(lanes["hard"]),
                    "standard_review_indices": core.format_indices(lanes["standard_review"]),
                    "priority_review_indices": core.format_indices(lanes["priority_review"]),
                    "candidate_count": len(candidates),
                    "agreement_count_distribution": json.dumps(
                        {
                            str(value): int(np.sum(agreement_count == value))
                            for value in range(len(bridge.SOURCE_NAMES) + 1)
                        },
                        sort_keys=True,
                    ),
                    "calibration_source_by_model": json.dumps(
                        calibration_sources, sort_keys=True
                    ),
                    "contains_test_labels": False,
                    "uses_family_name": 0,
                    "uses_test_length": 0,
                    "retrospective_counterfactual": 1,
                }
            )
    return rows


def run_prediction(output_dir: Path = DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    manifest, entries = _load_entries(output_dir)
    boundaries, standardized, global_refs, group_refs = _prepare_pools(entries)
    rows = []
    for done, dataset in enumerate(sorted(entries), 1):
        rows.extend(
            predict_dataset(
                dataset,
                entries[dataset],
                standardized,
                global_refs,
                group_refs,
            )
        )
        if done % 50 == 0 or done == len(entries):
            print(f"Exp154 prediction {done}/{len(entries)}", flush=True)
    rows.sort(key=lambda row: (row["variant"], float(row["alpha"]), row["dataset_name"]))
    prediction = output_dir / "05_exp154_predictions.jsonl"
    prediction_manifest = output_dir / "06_exp154_prediction_manifest.json"
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
    (output_dir / "06a_exp154_pool_contract.json").write_text(
        json.dumps(
            {
                "length_quantile_boundaries": boundaries,
                "train_count_bins": ["5-19", "20-49", "50-99", "100+"],
                "minimum_pool_reference": core.MIN_POOL_REFERENCE,
                "family_used": False,
                "test_statistics_used": False,
                "variants": list(VARIANTS),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return prediction


def run_evaluation(output_dir: Path = DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    return reporting.evaluate_prediction_file(
        output_dir / "05_exp154_predictions.jsonl",
        output_dir / "06_exp154_prediction_manifest.json",
        output_dir / "07_exp154_results.csv",
        output_dir / "08_exp154_summary.csv",
        output_dir / "09_exp154_summary.md",
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
