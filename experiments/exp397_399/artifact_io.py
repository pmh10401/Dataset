from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from experiments.exp393_396.artifact_io import load_verified_prediction


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
EXP292_ROOT = DATA_ROOT / "exp292_standard_review_evidence_tiering"
EXP393_ROOT = DATA_ROOT / "exp393_396_normal_process_grammar"
EXP292_PREDICTION = EXP292_ROOT / "exp292_frozen_review_tiers.jsonl"
EXP292_MANIFEST = EXP292_ROOT / "exp292_prediction_manifest.json"
EXP393_PREDICTION = EXP393_ROOT / "exp393_predictions.jsonl"
EXP393_MANIFEST = EXP393_ROOT / "exp393_prediction_manifest.json"
PREDICTION_NAME = "exp397_frozen_review_ranking.jsonl"
MANIFEST_NAME = "exp397_prediction_manifest.json"

FORBIDDEN_KEYS = {
    "y_test",
    "true_label",
    "is_true_anomaly",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "combined_f1",
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_label_free(row: dict[str, Any]) -> None:
    if row.get("contains_test_labels") is not False:
        raise ValueError("prediction row must declare contains_test_labels=false")
    if row.get("contains_evaluation_metrics") is not False:
        raise ValueError(
            "prediction row must declare contains_evaluation_metrics=false"
        )
    for key in row:
        lowered = key.lower()
        if lowered in FORBIDDEN_KEYS or lowered.endswith(
            ("_tp", "_fp", "_fn", "_f1", "_precision", "_recall")
        ):
            raise ValueError(f"prediction contains forbidden field: {key}")


def load_exp292_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(EXP292_MANIFEST.read_text(encoding="utf-8"))
    if sha256_file(EXP292_PREDICTION) != manifest.get("prediction_sha256"):
        raise ValueError("Exp292 prediction SHA256 mismatch")
    if manifest.get("contains_test_labels") is not False:
        raise ValueError("Exp292 manifest contains TEST labels")
    rows = [
        json.loads(line)
        for line in EXP292_PREDICTION.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(manifest.get("rows", -1)):
        raise ValueError("Exp292 row count mismatch")
    return rows, manifest


def load_exp393_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return load_verified_prediction(EXP393_PREDICTION, EXP393_MANIFEST)


def write_ranking_artifact(
    output_root: Path,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_path = output_root / PREDICTION_NAME
    manifest_path = output_root / MANIFEST_NAME
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["variant"]),
            float(row["alpha"]),
            int(row["rank"]),
        ),
    )
    for row in ordered:
        _assert_label_free(row)
    temporary = prediction_path.with_name(
        f".{prediction_path.name}.{os.getpid()}.tmp"
    )
    with temporary.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, prediction_path)

    universes: dict[tuple[str, float], set[tuple[str, int]]] = {}
    for row in ordered:
        key = (str(row["variant"]), float(row["alpha"]))
        universes.setdefault(key, set()).add(
            (str(row["dataset"]), int(row["candidate_index"]))
        )
    unique_universes = {frozenset(value) for value in universes.values()}
    if len(unique_universes) != 1:
        raise ValueError("ranking variants changed the candidate universe")
    candidate_count = len(next(iter(unique_universes), frozenset()))
    manifest = {
        "experiment_id": "experiment_397_frozen_review_ranking",
        "prediction_file": prediction_path.name,
        "prediction_sha256": sha256_file(prediction_path),
        "rows": len(ordered),
        "candidate_universe_count": candidate_count,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "automatic_hard_changed": False,
        "priority_review_changed": False,
        "standard_review_changed": False,
        "test_metrics_select_policy": False,
        "eligible_for_deployment": False,
        "retrospective_counterfactual": True,
        "prospective_validated": False,
        **metadata,
    }
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.tmp"
    )
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return prediction_path, manifest_path


def load_verified_ranking(
    prediction_path: Path,
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("contains_test_labels") is not False:
        raise ValueError("ranking manifest contains TEST labels")
    if manifest.get("contains_evaluation_metrics") is not False:
        raise ValueError("ranking manifest contains evaluation metrics")
    if sha256_file(prediction_path) != manifest.get("prediction_sha256"):
        raise ValueError("ranking prediction SHA256 mismatch")
    rows = [
        json.loads(line)
        for line in Path(prediction_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(manifest.get("rows", -1)):
        raise ValueError("ranking prediction row count mismatch")
    for row in rows:
        _assert_label_free(row)
    return rows, manifest

