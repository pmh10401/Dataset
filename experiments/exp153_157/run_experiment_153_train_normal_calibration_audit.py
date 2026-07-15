"""Exp153: TRAIN-normal cross-fitted calibration validity audit.

Every TRAIN-normal OOF score is treated once as a pseudo-normal run.  Its own
score is removed from the empirical reference before computing a conformal
p-value.  Block-B and Block-C are also audited with leave-one-out TRAIN-only
thresholds.  Because every pseudo-run is normal, every routed Hard alert is a
false alert by construction; TEST labels are neither loaded nor required.
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
EXPERIMENT_ID = "experiment_153_train_normal_calibration_validity_audit"


def _source_train_p_values(cache: dict[str, object]) -> dict[str, np.ndarray]:
    return {
        source: core.leave_one_out_conformal_p_values(
            cache[f"{source}__train_oof"]
        )
        for source in bridge.SOURCE_NAMES
    }


def audit_dataset(manifest_row, cache_dir: Path) -> list[dict[str, object]]:
    cache = bridge.load_cache_entry(cache_dir, manifest_row)
    metadata = cache["metadata"]
    dataset = metadata["dataset_name"]
    n_train = int(metadata["n_train"])
    abstain = bool(int(metadata["abstain"]))
    rows = []

    if abstain:
        for alpha in core.ALPHAS:
            rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "dataset_name": dataset,
                    "alpha": alpha,
                    "pseudo_runs": 0,
                    "hard_false_alerts": 0,
                    "standard_false_reviews": 0,
                    "priority_false_reviews": 0,
                    "candidate_count": 0,
                    "abstained": 1,
                    "resolution_blocked": 1,
                    "audit_scope": "train_normal_crossfitted_pseudo_runs_only",
                }
            )
        return rows

    p_values = _source_train_p_values(cache)
    block_b_flags = bridge.leave_one_out_historical_countcap_flags(
        cache["block_b_train_oof"], bridge.LEGACY_BLOCK_B_RATE
    )
    block_c_flags = bridge.leave_one_out_historical_countcap_flags(
        cache["block_c_train_oof"], bridge.LEGACY_BLOCK_C_RATE
    )
    block_b = set(np.flatnonzero(block_b_flags).astype(int).tolist())
    block_c = set(np.flatnonzero(block_c_flags).astype(int).tolist())
    minimum_p = {source: 1.0 / n_train for source in bridge.SOURCE_NAMES}

    for alpha in core.ALPHAS:
        matrix = np.vstack([p_values[source] <= alpha for source in bridge.SOURCE_NAMES])
        agreement_count = matrix.sum(axis=0)
        candidates = set(
            np.flatnonzero(agreement_count >= core.REQUIRED_SOURCE_AGREEMENT)
            .astype(int)
            .tolist()
        )
        lanes = core.route_lanes(candidates, block_b, block_c, n_train)
        resolution = sum(value <= alpha for value in minimum_p.values()) < core.REQUIRED_SOURCE_AGREEMENT
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "dataset_name": dataset,
                "alpha": alpha,
                "pseudo_runs": n_train,
                "hard_false_alerts": len(lanes["hard"]),
                "standard_false_reviews": len(lanes["standard_review"]),
                "priority_false_reviews": len(lanes["priority_review"]),
                "candidate_count": len(candidates),
                "hard_false_alert_rate": len(lanes["hard"]) / n_train,
                "hard_false_alerts_per_100_runs": len(lanes["hard"]) / n_train * 100.0,
                "standard_reviews_per_100_runs": len(lanes["standard_review"]) / n_train * 100.0,
                "priority_reviews_per_100_runs": len(lanes["priority_review"]) / n_train * 100.0,
                "abstained": 0,
                "resolution_blocked": int(resolution),
                "audit_scope": "train_normal_crossfitted_pseudo_runs_only",
            }
        )
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[float(row["alpha"])].append(row)
    summaries = []
    for alpha, subset in sorted(grouped.items()):
        pseudo_runs = int(sum(int(row["pseudo_runs"]) for row in subset))
        hard = int(sum(int(row["hard_false_alerts"]) for row in subset))
        standard = int(sum(int(row["standard_false_reviews"]) for row in subset))
        priority = int(sum(int(row["priority_false_reviews"]) for row in subset))
        summaries.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "alpha": alpha,
                "datasets": len(subset),
                "pseudo_normal_runs": pseudo_runs,
                "hard_false_alerts": hard,
                "hard_false_alert_rate": hard / max(1, pseudo_runs),
                "hard_false_alerts_per_100_runs": hard / max(1, pseudo_runs) * 100.0,
                "standard_false_reviews": standard,
                "standard_reviews_per_100_runs": standard / max(1, pseudo_runs) * 100.0,
                "priority_false_reviews": priority,
                "priority_reviews_per_100_runs": priority / max(1, pseudo_runs) * 100.0,
                "abstained_datasets": int(sum(int(row["abstained"]) for row in subset)),
                "resolution_blocked_datasets": int(sum(int(row["resolution_blocked"]) for row in subset)),
                "uses_test_labels": 0,
                "uses_test_values": 0,
            }
        )
    return summaries


def run(output_dir: Path = DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    manifest = reporting.read_rows(output_dir / "01_score_cache_manifest.csv")
    cache_dir = output_dir / "cache"
    rows = []
    for done, manifest_row in enumerate(manifest, 1):
        rows.extend(audit_dataset(manifest_row, cache_dir))
        if done % 50 == 0 or done == len(manifest):
            print(f"Exp153 audit {done}/{len(manifest)}", flush=True)
    rows.sort(key=lambda row: (float(row["alpha"]), row["dataset_name"]))
    reporting.write_rows(output_dir / "02_exp153_dataset_audit.csv", rows)
    summaries = summarize(rows)
    reporting.write_rows(output_dir / "03_exp153_summary.csv", summaries)
    (output_dir / "04_exp153_summary.md").write_text(
        "# Exp153 TRAIN-normal calibration validity audit\n\n"
        "All pseudo-runs are TRAIN-normal; therefore every Hard alert in this file "
        "is a false alert by construction. TEST labels and TEST values were not used.\n\n```json\n"
        + json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n```\n",
        encoding="utf-8",
    )
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    run(Path(args.output_dir))


if __name__ == "__main__":
    main()
