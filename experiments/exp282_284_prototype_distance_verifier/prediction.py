from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.exp278_281_train_only_recovery.contracts import atomic_bytes, sha256_file, write_csv

from .cache import verify_cache
from .distance_engine import ALPHA


VARIANTS = ("strict_geometry_shape_spectrum", "balanced_consensus_2of3", "sensitivity_any_mahal_2of3")


def _passes(pvalues: dict[str, float]) -> dict[str, bool]:
    diagonal = pvalues["diagonal_mahalanobis"] <= ALPHA
    shrinkage = pvalues["shrinkage_mahalanobis"] <= ALPHA
    shape = pvalues["correlation"] <= ALPHA
    spectrum = pvalues["spectral_wasserstein"] <= ALPHA
    geometry_both, geometry_any = diagonal and shrinkage, diagonal or shrinkage
    return {
        "strict_geometry_shape_spectrum": geometry_both and shape and spectrum,
        "balanced_consensus_2of3": sum((geometry_both, shape, spectrum)) >= 2,
        "sensitivity_any_mahal_2of3": sum((geometry_any, shape, spectrum)) >= 2,
    }


def freeze_predictions(output_root: Path) -> tuple[Path, Path]:
    root = Path(output_root); cache_path = root / "exp282_distance_cache.jsonl"; cache_manifest = root / "exp282_cache_manifest.json"
    rows = verify_cache(cache_path, cache_manifest); frozen = []
    for row in rows:
        records = []
        for record in row["candidate_records"]:
            records.append({
                "candidate_index": record["candidate_index"], "source_lane": record["source_lane"],
                "pvalues": record["pvalues"], "variant_pass": _passes(record["pvalues"]),
            })
        frozen.append({
            "dataset": row["dataset"], "n_train": row["n_train"], "n_test": row["n_test"],
            "hard_indices": row["hard_indices"],
            "candidate_records": records, "abstained": row["abstained"],
            "contains_test_labels": False, "contains_evaluation_metrics": False,
            "hard_policy_changed": False, "review_only": True, "retrospective": True,
        })
    payload = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode() for row in frozen)
    prediction_path, manifest_path = root / "exp283_frozen_predictions.jsonl", root / "exp283_prediction_manifest.json"
    atomic_bytes(prediction_path, payload)
    manifest = {
        "experiment_id": "experiment_283_prototype_distance_prediction", "datasets": len(frozen), "rows": len(frozen),
        "prediction_sha256": hashlib.sha256(payload).hexdigest(), "cache_sha256": sha256_file(cache_path),
        "contains_test_labels": False, "contains_evaluation_metrics": False,
        "hard_policy_changed": False, "review_only": True, "retrospective": True,
    }
    atomic_bytes(manifest_path, (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode())
    write_csv(root / "exp283_summary.csv", [{
        "experiment_id": manifest["experiment_id"], "datasets": len(frozen),
        **{f"{variant}_count": sum(bool(record["variant_pass"][variant]) for row in frozen for record in row["candidate_records"]) for variant in VARIANTS},
        "contains_test_labels": False, "contains_evaluation_metrics": False, "hard_policy_changed": False,
    }])
    return prediction_path, manifest_path


def verify_prediction(prediction_path: Path, manifest_path: Path) -> list[dict[str, object]]:
    manifest = json.loads(Path(manifest_path).read_text())
    if sha256_file(prediction_path) != manifest.get("prediction_sha256"):
        raise ValueError("Exp283 prediction hash mismatch")
    rows = [json.loads(line) for line in Path(prediction_path).read_text().splitlines() if line]
    if len(rows) != 1117 or any(row.get("contains_test_labels") is not False or row.get("contains_evaluation_metrics") is not False for row in rows):
        raise ValueError("Exp283 prediction boundary/coverage mismatch")
    return rows


__all__ = ["VARIANTS", "freeze_predictions", "verify_prediction"]
