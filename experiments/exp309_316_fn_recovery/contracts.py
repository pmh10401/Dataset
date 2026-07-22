from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


FORBIDDEN_FIELDS = {
    "y_test", "labels", "test_labels", "anomaly_positions",
    "tp", "fp", "fn", "precision", "recall", "f1", "combined_f1",
}


@dataclass(frozen=True)
class ScoreArtifact:
    train_oof_scores: np.ndarray
    test_scores: np.ndarray
    diagnostics: dict[str, object] = field(default_factory=dict)
    available: bool = True
    reason: str = ""

    def __post_init__(self) -> None:
        train = np.asarray(self.train_oof_scores, dtype=np.float64).reshape(-1)
        test = np.asarray(self.test_scores, dtype=np.float64).reshape(-1)
        if self.available and (not len(train) or not np.isfinite(train).all() or not np.isfinite(test).all()):
            raise ValueError("available ScoreArtifact requires finite TRAIN OOF and TEST scores")
        object.__setattr__(self, "train_oof_scores", train)
        object.__setattr__(self, "test_scores", test)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_prediction_row(row: dict[str, object]) -> None:
    if row.get("contains_test_labels") is not False:
        raise ValueError("prediction must declare contains_test_labels=false")
    if row.get("contains_evaluation_metrics") is not False:
        raise ValueError("prediction must declare contains_evaluation_metrics=false")
    for key in row:
        lower = str(key).lower()
        if lower in FORBIDDEN_FIELDS or lower.endswith(("_tp", "_fp", "_fn", "_f1", "_precision", "_recall")):
            raise ValueError(f"forbidden prediction field: {key}")


def freeze_prediction_rows(
    rows: list[dict[str, object]], prediction_path: Path, manifest_path: Path,
    *, experiment_id: str, metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    for row in rows:
        validate_prediction_row(row)
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )
    manifest = {
        "experiment_id": experiment_id,
        "rows": len(rows),
        "prediction_sha256": hashlib.sha256(payload).hexdigest(),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        **(metadata or {}),
    }
    for path, content in (
        (Path(prediction_path), payload),
        (Path(manifest_path), (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(content)
        os.replace(temporary, path)
    return manifest


def verify_prediction_rows(prediction_path: Path, manifest_path: Path) -> list[dict[str, object]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if sha256_file(Path(prediction_path)) != manifest.get("prediction_sha256"):
        raise ValueError("prediction hash mismatch")
    rows = [
        json.loads(line)
        for line in Path(prediction_path).read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != int(manifest.get("rows", -1)):
        raise ValueError("prediction row count mismatch")
    for row in rows:
        validate_prediction_row(row)
    return rows
