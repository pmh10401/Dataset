from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from experiments.exp206_210.core import empirical_upper_pvalues
from experiments.exp226_235_hard_alert_quality.contracts import ScoreBundle
from experiments.exp226_235_hard_alert_quality.database import sha256_file
from experiments.exp237_new_score_source_qualification.config import ALPHAS, SOURCE_NAMES


FORBIDDEN_FIELDS = {
    "y_test", "labels", "test_labels", "anomaly_positions", "tp", "fp", "fn",
    "precision", "recall", "f1", "combined_f1",
}


@dataclass(frozen=True)
class SourcePrediction:
    experiment_id: str
    dataset: str
    source: str
    alpha: float
    candidate_indices: tuple[int, ...]
    n_train: int
    n_test: int
    calibration_abstained: bool
    contains_test_labels: bool = False
    contains_evaluation_metrics: bool = False
    retrospective: bool = True
    autonomous_alert: bool = False


def build_prediction_rows(scores: ScoreBundle) -> list[SourcePrediction]:
    abstained = bool(scores.provenance.get("calibration_abstained", False))
    rows: list[SourcePrediction] = []
    for source in SOURCE_NAMES:
        if abstained:
            candidates: tuple[int, ...] = ()
        else:
            pvalues = empirical_upper_pvalues(scores.train_oof[source], scores.test_scores[source])
            for alpha in ALPHAS:
                indices = tuple(np.flatnonzero(pvalues <= alpha).astype(int))
                rows.append(SourcePrediction(
                    experiment_id="experiment_237_new_score_source_qualification",
                    dataset=scores.dataset,
                    source=source,
                    alpha=float(alpha),
                    candidate_indices=indices,
                    n_train=int(scores.provenance.get("n_train", len(scores.train_oof[source]))),
                    n_test=int(scores.provenance.get("n_test", len(scores.test_scores[source]))),
                    calibration_abstained=False,
                ))
            continue
        for alpha in ALPHAS:
            rows.append(SourcePrediction(
                experiment_id="experiment_237_new_score_source_qualification",
                dataset=scores.dataset,
                source=source,
                alpha=float(alpha),
                candidate_indices=candidates,
                n_train=int(scores.provenance.get("n_train", 0)),
                n_test=int(scores.provenance.get("n_test", 0)),
                calibration_abstained=True,
            ))
    return rows


def _payload(row: SourcePrediction) -> dict[str, object]:
    payload = dataclasses.asdict(row)
    if FORBIDDEN_FIELDS & set(payload):
        raise ValueError("prediction payload contains evaluation fields")
    return payload


def _atomic_write(path: Path, data: bytes) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def freeze_predictions(
    rows: Iterable[SourcePrediction], prediction_path: Path, manifest_path: Path
) -> dict[str, object]:
    payloads = sorted(
        (_payload(row) for row in rows),
        key=lambda row: (str(row["dataset"]), str(row["source"]), float(row["alpha"])),
    )
    serialized = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in payloads
    )
    digest = hashlib.sha256(serialized).hexdigest()
    _atomic_write(Path(prediction_path), serialized)
    manifest = {
        "schema": "exp237_prediction_manifest_v1",
        "prediction_rows": len(payloads),
        "prediction_sha256": digest,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }
    _atomic_write(
        Path(manifest_path),
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
    )
    return manifest


def load_verified_predictions(prediction_path: Path, manifest_path: Path) -> list[dict[str, object]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    actual = sha256_file(Path(prediction_path))
    if actual != manifest.get("prediction_sha256"):
        raise ValueError(
            f"prediction SHA-256 mismatch: expected={manifest.get('prediction_sha256')} actual={actual}"
        )
    rows = [
        json.loads(line)
        for line in Path(prediction_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(manifest.get("prediction_rows", -1)):
        raise ValueError("prediction row count mismatch")
    for row in rows:
        if FORBIDDEN_FIELDS & set(row):
            raise ValueError("verified prediction contains evaluation fields")
    return rows
