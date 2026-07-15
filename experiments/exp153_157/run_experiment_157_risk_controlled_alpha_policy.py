"""Exp157: TRAIN-only risk-controlled alpha policy sensitivity.

For each dataset and each externally defined normal false-alert target, choose
the largest pre-registered alpha whose one-sided 95% Wilson upper bound from
Exp153 is no larger than the target.  TEST labels never select alpha.  The
three risk targets are reported together and none is declared primary here.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import exp153_157_bridge as bridge
import exp153_157_core as core
import exp153_157_reporting as reporting

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/exp153_157_train_only_improvements"
EXPERIMENT_ID = "experiment_157_train_only_risk_controlled_alpha_policy"


def _load_frozen_exp152_predictions():
    prediction_path, manifest_path = bridge.resolve_exp152_prediction_paths()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("exp152_prediction_sha256") or manifest.get("prediction_sha256")
    if not expected:
        raise ValueError("Exp152 manifest does not contain an Exp152 prediction hash")
    core.verify_frozen_file(prediction_path, expected)
    return prediction_path, manifest_path, reporting.read_jsonl(prediction_path)


def predict_risk_controlled_rows(audit_rows, base_rows):
    """Create label-free Exp157 rows from TRAIN audit and frozen Exp152 rows."""

    audit_by_dataset = defaultdict(list)
    for row in audit_rows:
        audit_by_dataset[str(row["dataset_name"])].append(row)
    base_by_dataset_alpha = {
        (str(row["dataset_name"]), float(row["alpha"])): row for row in base_rows
    }
    datasets = sorted({str(row["dataset_name"]) for row in base_rows})
    output = []
    for dataset in datasets:
        dataset_audit = audit_by_dataset.get(dataset, [])
        any_base = next(
            row for (name, _), row in base_by_dataset_alpha.items() if name == dataset
        )
        n_test = int(any_base["n_test"])
        for target in core.RISK_TARGETS_PER_RUN:
            selection = core.select_largest_alpha_by_risk(
                dataset_audit,
                target_rate=target,
                confidence=core.RISK_CONFIDENCE,
            )
            selected_alpha = selection["selected_alpha"]
            if selected_alpha is None:
                candidates = hard = standard = priority = set()
            else:
                selected_row = base_by_dataset_alpha[(dataset, float(selected_alpha))]
                candidates = core.parse_indices(selected_row.get("candidate_indices"))
                hard = core.parse_indices(selected_row.get("hard_indices"))
                standard = core.parse_indices(selected_row.get("standard_review_indices"))
                priority = core.parse_indices(selected_row.get("priority_review_indices"))
                if candidates != hard | standard | priority:
                    raise ValueError(f"Exp152 candidate union mismatch dataset={dataset}")
            output.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "variant": f"risk_target_{target:.3f}_per_run",
                    "dataset_name": dataset,
                    "risk_target_per_run": target,
                    "risk_target_per_100_runs": target * 100.0,
                    "risk_confidence": core.RISK_CONFIDENCE,
                    "selected_alpha": "" if selected_alpha is None else selected_alpha,
                    "alpha": "" if selected_alpha is None else selected_alpha,
                    "risk_abstained": int(selected_alpha is None),
                    "risk_selection_details": json.dumps(selection["evaluated"], sort_keys=True),
                    "n_train": int(any_base["n_train"]),
                    "n_test": n_test,
                    "abstained": int(selected_alpha is None),
                    "candidate_indices": core.format_indices(candidates),
                    "hard_indices": core.format_indices(hard),
                    "standard_review_indices": core.format_indices(standard),
                    "priority_review_indices": core.format_indices(priority),
                    "candidate_count": len(candidates),
                    "contains_test_labels": False,
                    "alpha_selected_from": "exp153_train_normal_audit_only",
                    "uses_test_metrics_for_selection": 0,
                    "retrospective_counterfactual": 1,
                }
            )
    return output


def run_prediction(output_dir: Path = DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    audit_path = output_dir / "02_exp153_dataset_audit.csv"
    if not audit_path.exists():
        raise FileNotFoundError("run Exp153 before Exp157")
    audit_rows = reporting.read_rows(audit_path)
    source_prediction, source_manifest, base_rows = _load_frozen_exp152_predictions()
    rows = predict_risk_controlled_rows(audit_rows, base_rows)
    rows.sort(key=lambda row: (float(row["risk_target_per_run"]), row["dataset_name"]))
    prediction = output_dir / "21_exp157_predictions.jsonl"
    prediction_manifest = output_dir / "22_exp157_prediction_manifest.json"
    reporting.write_jsonl(prediction, rows)
    reporting.freeze_prediction_file(
        prediction,
        prediction_manifest,
        experiment_ids=[EXPERIMENT_ID],
        row_count=len({row["dataset_name"] for row in base_rows}) * len(core.RISK_TARGETS_PER_RUN),
        input_hashes={
            "exp153_audit": core.sha256_file(audit_path),
            "exp152_prediction": core.sha256_file(source_prediction),
            "exp152_prediction_manifest": core.sha256_file(source_manifest),
        },
    )
    return prediction


def run_evaluation(output_dir: Path = DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    return reporting.evaluate_prediction_file(
        output_dir / "21_exp157_predictions.jsonl",
        output_dir / "22_exp157_prediction_manifest.json",
        output_dir / "23_exp157_results.csv",
        output_dir / "24_exp157_summary.csv",
        output_dir / "25_exp157_summary.md",
        group_fields=("experiment_id", "variant", "risk_target_per_run"),
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
