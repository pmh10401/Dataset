from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence


EXPECTED_DATASETS = 1117
SOURCE_ALPHAS = (0.005, 0.010)
FORBIDDEN_PREDICTION_KEYS = {
    "y_test",
    "labels",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "combined_f1",
}
PRESERVED_LANES = (
    "automatic_hard_indices",
    "standard_review_indices",
    "priority_review_indices",
    "automatic_normal_exact_indices",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _indices(row: dict, key: str, n_test: int) -> set[int]:
    values = row.get(key)
    if not isinstance(values, list):
        raise ValueError(f"{key} must be a list")
    result = set(values)
    if len(result) != len(values):
        raise ValueError(f"{key} contains duplicate indices")
    if any(type(index) is not int or index < 0 or index >= n_test for index in result):
        raise ValueError(f"{key} contains out-of-range indices")
    return result


def build_dual_tier_row(alpha005_row: dict, alpha010_row: dict) -> dict:
    dataset = alpha005_row.get("dataset")
    if not isinstance(dataset, str) or not dataset:
        raise ValueError("dataset is required")
    if alpha010_row.get("dataset") != dataset:
        raise ValueError("dataset mismatch between alpha sources")
    n_test = alpha005_row.get("n_test")
    if type(n_test) is not int or n_test < 0 or alpha010_row.get("n_test") != n_test:
        raise ValueError("n_test mismatch between alpha sources")
    for row in (alpha005_row, alpha010_row):
        if row.get("contains_test_labels") is not False:
            raise ValueError("source prediction contains TEST labels")
        if row.get("contains_evaluation_metrics") is not False:
            raise ValueError("source prediction contains evaluation metrics")

    preserved: dict[str, set[int]] = {}
    for key in PRESERVED_LANES:
        left = _indices(alpha005_row, key, n_test)
        right = _indices(alpha010_row, key, n_test)
        if left != right:
            label = "Hard lane" if key == "automatic_hard_indices" else key
            raise ValueError(f"{label} mismatch between alpha sources")
        preserved[key] = left

    hard = preserved["automatic_hard_indices"]
    highest = _indices(alpha005_row, "strong_hard_indices", n_test)
    expanded = _indices(alpha010_row, "strong_hard_indices", n_test)
    if not highest <= expanded:
        raise ValueError("alpha 0.005 strong tier is not nested in alpha 0.010")
    if not expanded <= hard:
        raise ValueError("strong confidence contains a non-Hard index")

    extended = expanded - highest
    general = hard - expanded
    row = {
        "experiment_id": "experiment_389_dual_confidence_prediction",
        "dataset": dataset,
        "n_test": n_test,
        "automatic_hard_indices": sorted(hard),
        "highest_confidence_indices": sorted(highest),
        "extended_confidence_indices": sorted(extended),
        "general_confidence_indices": sorted(general),
        "standard_review_indices": sorted(preserved["standard_review_indices"]),
        "priority_review_indices": sorted(preserved["priority_review_indices"]),
        "automatic_normal_exact_indices": sorted(
            preserved["automatic_normal_exact_indices"]
        ),
        "highest_confidence_alpha": SOURCE_ALPHAS[0],
        "extended_confidence_alpha": SOURCE_ALPHAS[1],
        "hard_lane_preserved": True,
        "review_lanes_preserved": True,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "retrospective_counterfactual": True,
        "prospective_validated": False,
    }
    validate_prediction_row(row)
    return row


def validate_prediction_row(row: dict) -> None:
    forbidden = FORBIDDEN_PREDICTION_KEYS & set(row)
    if forbidden:
        raise ValueError(f"prediction contains forbidden fields: {sorted(forbidden)}")
    if row.get("contains_test_labels") is not False:
        raise ValueError("prediction label-use flag must be false")
    if row.get("contains_evaluation_metrics") is not False:
        raise ValueError("prediction metric-use flag must be false")
    n_test = row.get("n_test")
    if type(n_test) is not int or n_test < 0:
        raise ValueError("invalid n_test")
    hard = _indices(row, "automatic_hard_indices", n_test)
    highest = _indices(row, "highest_confidence_indices", n_test)
    extended = _indices(row, "extended_confidence_indices", n_test)
    general = _indices(row, "general_confidence_indices", n_test)
    if highest & extended or highest & general or extended & general:
        raise ValueError("confidence tiers overlap")
    if highest | extended | general != hard:
        raise ValueError("confidence tiers do not preserve Hard lane")
    _indices(row, "standard_review_indices", n_test)
    _indices(row, "priority_review_indices", n_test)
    _indices(row, "automatic_normal_exact_indices", n_test)


def freeze_predictions(
    rows: Sequence[dict],
    prediction_path: Path,
    manifest_path: Path,
    *,
    source_metadata: dict | None = None,
) -> dict:
    if not rows:
        raise ValueError("cannot freeze empty predictions")
    ordered = sorted(rows, key=lambda row: row["dataset"])
    datasets = [str(row["dataset"]) for row in ordered]
    if len(set(datasets)) != len(datasets):
        raise ValueError("duplicate datasets in prediction rows")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in ordered:
            validate_prediction_row(row)
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    manifest = {
        "experiment_id": "experiment_389_dual_confidence_prediction",
        "prediction_file": prediction_path.name,
        "prediction_sha256": sha256_file(prediction_path),
        "rows": len(ordered),
        "datasets": len(ordered),
        "source_alphas": list(SOURCE_ALPHAS),
        "hard_lane_preserved": True,
        "review_lanes_preserved": True,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "test_metrics_select_policy": False,
        "retrospective_counterfactual": True,
        "prospective_validated": False,
        "source_metadata": source_metadata or {},
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_frozen_predictions(
    prediction_path: Path,
    manifest_path: Path,
) -> list[dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(prediction_path) != manifest.get("prediction_sha256"):
        raise ValueError("prediction SHA256 does not match frozen manifest")
    rows = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(manifest.get("rows", -1)):
        raise ValueError("prediction row count does not match manifest")
    if len({row.get("dataset") for row in rows}) != len(rows):
        raise ValueError("prediction contains duplicate datasets")
    for row in rows:
        validate_prediction_row(row)
    return rows


def evaluate_indices(
    indices: Iterable[int],
    labels: Sequence[int | str],
) -> dict[str, int | float]:
    truth = [1 if str(label) == "1" else 0 for label in labels]
    predicted = {int(index) for index in indices}
    if any(index < 0 or index >= len(truth) for index in predicted):
        raise ValueError("evaluation index is out of range")
    true_indices = {index for index, label in enumerate(truth) if label == 1}
    tp = len(predicted & true_indices)
    fp = len(predicted - true_indices)
    fn = len(true_indices - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "count": len(predicted),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
