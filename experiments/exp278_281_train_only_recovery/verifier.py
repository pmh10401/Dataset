from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .contracts import (
    EXP272_ROOT, MIN_CONTINUOUS_GROUPS, VARIANTS, atomic_bytes, candidate_rows,
    sha256_file,
)


def _read_index(path: Path, value_fields: tuple[str, ...]) -> dict[tuple[str, int], dict[str, object]]:
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], int(row["candidate_index"]))
            if key in result:
                raise ValueError(f"duplicate candidate audit row: {key}")
            converted: dict[str, object] = {}
            for field in value_fields:
                converted[field] = float(row[field]) if field not in {"continuous_group_count"} else int(row[field])
            result[key] = converted
    return result


def freeze_verifier_predictions(output_root: Path) -> tuple[Path, Path]:
    rows, datasets = candidate_rows()
    root = Path(output_root)
    risk_path, stability_path = root / "exp278_risk_resolution_dataset.csv", root / "exp279_candidate_stability.csv"
    risks = _read_index(risk_path, ("continuous_group_count", "maximum_train_risk_upper_95"))
    stability = _read_index(stability_path, ("mean_calibration_stability", "minimum_calibration_stability"))
    if set(risks) != set(stability) or len(risks) != 179:
        raise ValueError("Exp278/279 candidate universe mismatch")
    frozen = []
    for row in rows:
        candidates = sorted(set(row["recovery_shadow_indices"]) | set(row["recovery_priority_indices"]))
        records = []
        for index in candidates:
            key = (str(row["dataset"]), index)
            risk, stable = risks[key], stability[key]
            group_count = int(risk["continuous_group_count"])
            max_risk = float(risk["maximum_train_risk_upper_95"])
            mean_stability = float(stable["mean_calibration_stability"])
            records.append({
                "candidate_index": index,
                "source_lane": "recovery_shadow" if index in row["recovery_shadow_indices"] else "recovery_priority",
                "continuous_group_count": group_count,
                "maximum_train_risk_upper_95": max_risk,
                "mean_calibration_stability": mean_stability,
                "minimum_calibration_stability": float(stable["minimum_calibration_stability"]),
                "variant_pass": {
                    name: group_count >= MIN_CONTINUOUS_GROUPS and max_risk <= risk_limit and mean_stability >= stability_limit
                    for name, (risk_limit, stability_limit) in VARIANTS.items()
                },
            })
        frozen.append({
            "dataset": row["dataset"], "n_train": row["n_train"], "n_test": row["n_test"],
            "hard_indices": row["hard_indices"], "candidate_records": records,
            "contains_test_labels": False, "contains_evaluation_metrics": False,
            "hard_policy_changed": False, "review_only": True, "retrospective": True,
        })
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        for row in frozen
    )
    prediction_path, manifest_path = root / "exp281_frozen_predictions.jsonl", root / "exp281_prediction_manifest.json"
    atomic_bytes(prediction_path, payload)
    manifest = {
        "experiment_id": "experiment_281_train_only_two_stage_verifier",
        "datasets": len(datasets), "rows": len(frozen),
        "prediction_sha256": hashlib.sha256(payload).hexdigest(),
        "input_sha256": {
            "exp272_prediction": sha256_file(EXP272_ROOT / "exp272_frozen_predictions.jsonl"),
            "exp278_risk": sha256_file(risk_path), "exp279_stability": sha256_file(stability_path),
        },
        "contains_test_labels": False, "contains_evaluation_metrics": False,
        "hard_policy_changed": False, "review_only": True, "retrospective": True,
    }
    atomic_bytes(manifest_path, (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode())
    return prediction_path, manifest_path


def verify_verifier_prediction(
    prediction_path: Path, manifest_path: Path, *, expected_datasets: tuple[str, ...],
) -> list[dict[str, object]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if sha256_file(prediction_path) != manifest.get("prediction_sha256"):
        raise ValueError("Exp281 prediction hash mismatch")
    rows = [json.loads(line) for line in Path(prediction_path).read_text().splitlines() if line]
    if tuple(str(row.get("dataset")) for row in rows) != tuple(sorted(expected_datasets)):
        raise ValueError("Exp281 prediction population/order mismatch")
    for row in rows:
        if row.get("contains_test_labels") is not False or row.get("contains_evaluation_metrics") is not False:
            raise ValueError("Exp281 prediction boundary mismatch")
        if any(key in row for key in ("y_test", "tp", "fp", "fn", "precision", "recall", "f1")):
            raise ValueError("Exp281 prediction includes evaluation data")
        indices = [record["candidate_index"] for record in row["candidate_records"]]
        if indices != sorted(set(indices)) or any(index < 0 or index >= row["n_test"] for index in indices):
            raise ValueError("Exp281 candidate index contract mismatch")
    return rows


__all__ = ["freeze_verifier_predictions", "verify_verifier_prediction"]
