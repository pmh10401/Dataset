from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.database import load_values_bundle
from experiments.exp226_235_hard_alert_quality.preprocessing import align_bundle
from experiments.exp278_281_train_only_recovery.contracts import atomic_bytes, candidate_rows, sha256_file, write_csv

from .model import ALPHAS, MIN_TRAIN, anomaly_scores, cross_fitted_normal_scores, empirical_pvalue, fit_classifier


def generation_audit(output_root: Path) -> tuple[Path, Path]:
    rows, datasets = candidate_rows(); detail = []
    for row in rows:
        count = len(set(row["recovery_shadow_indices"]) | set(row["recovery_priority_indices"]))
        detail.append({
            "dataset": row["dataset"], "n_train": row["n_train"], "candidate_count": count,
            "eligible": count > 0 and int(row["n_train"]) >= MIN_TRAIN,
            "synthetic_types": "spike|level_shift|drift|frequency",
            "contains_test_labels": False, "contains_evaluation_metrics": False,
        })
    root = Path(output_root); detail_path = root / "exp285_generation_audit.csv"; summary_path = root / "exp285_summary.csv"
    write_csv(detail_path, detail); write_csv(summary_path, [{
        "experiment_id": "experiment_285_synthetic_shape_generation_audit", "datasets": len(datasets),
        "candidate_datasets": sum(int(row["candidate_count"]) > 0 for row in detail),
        "eligible_datasets": sum(bool(row["eligible"]) for row in detail),
        "candidate_count": sum(int(row["candidate_count"]) for row in detail),
        "contains_test_labels": False, "contains_evaluation_metrics": False,
    }]); return detail_path, summary_path


def freeze_predictions(output_root: Path) -> tuple[Path, Path]:
    rows, datasets = candidate_rows(); frozen = []
    for row in rows:
        candidates = sorted(set(row["recovery_shadow_indices"]) | set(row["recovery_priority_indices"]))
        base = {
            "dataset": row["dataset"], "n_train": row["n_train"], "n_test": row["n_test"], "hard_indices": row["hard_indices"],
            "candidate_records": [], "abstained": False, "abstention_reason": None,
            "contains_test_labels": False, "contains_evaluation_metrics": False,
            "hard_policy_changed": False, "review_only": True, "retrospective": True,
        }
        if not candidates: frozen.append(base); continue
        if int(row["n_train"]) < MIN_TRAIN:
            base.update({"abstained": True, "abstention_reason": "n_train_below_100"}); frozen.append(base); continue
        bundle = align_bundle(load_values_bundle(str(row["dataset"]))); train = np.asarray(bundle.train_normal, dtype=float)
        test = np.asarray(bundle.test_values, dtype=float); normal_scores = cross_fitted_normal_scores(train, identity=str(row["dataset"]))
        model, bases, synthetic = fit_classifier(train, identity=f"{row['dataset']}:final")
        query_scores = anomaly_scores(model, test[candidates])
        for position, index in enumerate(candidates):
            pvalue = empirical_pvalue(normal_scores, float(query_scores[position]))
            base["candidate_records"].append({
                "candidate_index": index,
                "source_lane": "recovery_shadow" if index in row["recovery_shadow_indices"] else "recovery_priority",
                "synthetic_shape_score": float(query_scores[position]), "pvalue": pvalue,
                "variant_pass": {f"synthetic_shape_alpha_{alpha:0.2f}".replace("0.", "0"): pvalue <= alpha for alpha in ALPHAS},
            })
        base.update({"crossfit_normal_reference_count": len(normal_scores), "final_normal_bases": bases, "final_synthetic_rows": synthetic})
        frozen.append(base)
    payload = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode() for row in frozen)
    root = Path(output_root); prediction_path = root / "exp286_frozen_predictions.jsonl"; manifest_path = root / "exp286_prediction_manifest.json"
    atomic_bytes(prediction_path, payload)
    manifest = {
        "experiment_id": "experiment_286_synthetic_shape_prediction", "datasets": len(datasets), "rows": len(frozen),
        "prediction_sha256": hashlib.sha256(payload).hexdigest(), "contains_test_labels": False,
        "contains_evaluation_metrics": False, "hard_policy_changed": False, "review_only": True, "retrospective": True,
    }
    atomic_bytes(manifest_path, (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode())
    variants = tuple(next((record["variant_pass"].keys() for row in frozen for record in row["candidate_records"]), ()))
    write_csv(root / "exp286_summary.csv", [{
        "experiment_id": manifest["experiment_id"], "datasets": len(frozen),
        **{f"{variant}_count": sum(record["variant_pass"][variant] for row in frozen for record in row["candidate_records"]) for variant in variants},
        "contains_test_labels": False, "contains_evaluation_metrics": False, "hard_policy_changed": False,
    }]); return prediction_path, manifest_path


def verify_prediction(prediction_path: Path, manifest_path: Path) -> list[dict[str, object]]:
    manifest = json.loads(Path(manifest_path).read_text())
    if sha256_file(prediction_path) != manifest.get("prediction_sha256"): raise ValueError("Exp286 prediction hash mismatch")
    rows = [json.loads(line) for line in Path(prediction_path).read_text().splitlines() if line]
    if len(rows) != 1117 or any(row.get("contains_test_labels") is not False or row.get("contains_evaluation_metrics") is not False for row in rows):
        raise ValueError("Exp286 prediction boundary/coverage mismatch")
    return rows


__all__ = ["freeze_predictions", "generation_audit", "verify_prediction"]
