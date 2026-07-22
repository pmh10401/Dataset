from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.database import load_values_bundle
from experiments.exp226_235_hard_alert_quality.preprocessing import align_bundle
from experiments.exp267_273_zero_support_fn_recovery.recovery_routing import verify_recovery_prediction_hash
from experiments.exp278_281_train_only_recovery.contracts import atomic_bytes, candidate_rows, sha256_file

from .distance_engine import ALPHA, DISTANCES, MIN_TRAIN, cross_fitted_references, empirical_pvalue, fit_distance_model, score_distances


EXP272_ROOT = Path("/Users/minho/Documents/Dataset/exp267_273_zero_support_fn_recovery")


def build_distance_cache(output_root: Path) -> tuple[Path, Path]:
    rows, datasets = candidate_rows()
    verify_recovery_prediction_hash(
        EXP272_ROOT / "exp272_frozen_predictions.jsonl", EXP272_ROOT / "exp272_prediction_manifest.json",
        expected_datasets=datasets,
    )
    output = []
    for source_row in rows:
        candidates = sorted(set(source_row["recovery_shadow_indices"]) | set(source_row["recovery_priority_indices"]))
        base = {
            "dataset": source_row["dataset"], "n_train": source_row["n_train"], "n_test": source_row["n_test"],
            "hard_indices": source_row["hard_indices"],
            "candidate_records": [], "abstained": False, "abstention_reason": None,
            "contains_test_labels": False, "contains_evaluation_metrics": False,
        }
        if not candidates:
            output.append(base); continue
        if int(source_row["n_train"]) < MIN_TRAIN:
            base.update({"abstained": True, "abstention_reason": "n_train_below_20"}); output.append(base); continue
        bundle = align_bundle(load_values_bundle(str(source_row["dataset"])))
        train = np.asarray(bundle.train_normal, dtype=float); test = np.asarray(bundle.test_values, dtype=float)
        references, fallback_folds = cross_fitted_references(train)
        model = fit_distance_model(train); scores = score_distances(model, test[candidates])
        for position, index in enumerate(candidates):
            pvalues = {name: empirical_pvalue(references[name], float(scores[name][position])) for name in DISTANCES}
            base["candidate_records"].append({
                "candidate_index": index,
                "source_lane": "recovery_shadow" if index in source_row["recovery_shadow_indices"] else "recovery_priority",
                "pvalues": pvalues,
                "distances": {name: float(scores[name][position]) for name in DISTANCES},
            })
        base.update({
            "target_length": int(bundle.target_length), "crossfit_folds": 5,
            "shrinkage_fallback_folds": fallback_folds,
            "final_shrinkage_fallback": model.shrinkage_fallback,
            "final_condition_number": model.condition_number,
        })
        output.append(base)
    payload = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode() for row in output)
    root = Path(output_root); cache_path = root / "exp282_distance_cache.jsonl"; manifest_path = root / "exp282_cache_manifest.json"
    atomic_bytes(cache_path, payload)
    manifest = {
        "experiment_id": "experiment_282_prototype_distance_cache", "datasets": len(output), "rows": len(output),
        "cache_sha256": hashlib.sha256(payload).hexdigest(), "alpha": ALPHA,
        "input_sha256": {"exp272_prediction": sha256_file(EXP272_ROOT / "exp272_frozen_predictions.jsonl")},
        "contains_test_labels": False, "contains_evaluation_metrics": False, "retrospective": True,
    }
    atomic_bytes(manifest_path, (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode())
    summary = [{
        "experiment_id": manifest["experiment_id"], "datasets": len(output),
        "candidate_datasets": sum(bool(row["candidate_records"]) for row in output),
        "candidate_count": sum(len(row["candidate_records"]) for row in output),
        "abstained_datasets": sum(bool(row["abstained"]) for row in output),
        "shrinkage_fallback_datasets": sum(bool(row.get("final_shrinkage_fallback")) for row in output),
        "contains_test_labels": False, "contains_evaluation_metrics": False,
    }]
    from experiments.exp278_281_train_only_recovery.contracts import write_csv
    write_csv(root / "exp282_summary.csv", summary)
    return cache_path, manifest_path


def verify_cache(cache_path: Path, manifest_path: Path) -> list[dict[str, object]]:
    manifest = json.loads(Path(manifest_path).read_text())
    if sha256_file(cache_path) != manifest.get("cache_sha256"):
        raise ValueError("Exp282 cache hash mismatch")
    rows = [json.loads(line) for line in Path(cache_path).read_text().splitlines() if line]
    if len(rows) != 1117 or any(row.get("contains_test_labels") is not False for row in rows):
        raise ValueError("Exp282 cache boundary/coverage mismatch")
    return rows


__all__ = ["build_distance_cache", "verify_cache"]
