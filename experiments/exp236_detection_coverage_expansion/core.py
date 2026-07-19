from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from experiments.exp226_235_hard_alert_quality.baseline_exp137 import BaselineLanes


EXPERIMENT_ID = "experiment_236_three_source_hard_two_source_review"
FORBIDDEN_FIELDS = {
    "y_test", "labels", "test_labels", "anomaly_positions", "tp", "fp",
    "fn", "f1", "precision", "recall", "combined_f1",
}


@dataclass(frozen=True)
class ExpandedLanes:
    hard: frozenset[int]
    standard_review: frozenset[int]
    priority_review: frozenset[int]
    new_hard: frozenset[int]
    new_standard_review: frozenset[int]


def build_expanded_lanes(
    baseline: BaselineLanes,
    *,
    minimum_two: Iterable[int],
    minimum_three: Iterable[int],
) -> ExpandedLanes:
    source_two = frozenset(int(value) for value in minimum_two)
    source_three = frozenset(int(value) for value in minimum_three)
    if not source_three <= source_two:
        raise ValueError("minimum-three candidates must be a subset of minimum-two candidates")
    baseline_union = baseline.hard | baseline.standard_review | baseline.priority_review
    if (source_two | source_three) & baseline_union:
        raise ValueError("new expansion candidates overlap an Exp137 lane")
    if any(index < 0 or index >= baseline.test_size for index in source_two | source_three):
        raise ValueError("expansion candidate index outside TEST range")
    exact_two = source_two - source_three
    hard = baseline.hard | source_three
    standard = baseline.standard_review | exact_two
    priority = baseline.priority_review
    if hard & standard or hard & priority or standard & priority:
        raise AssertionError("Exp236 lanes overlap")
    return ExpandedLanes(
        hard=frozenset(hard),
        standard_review=frozenset(standard),
        priority_review=frozenset(priority),
        new_hard=source_three,
        new_standard_review=exact_two,
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_predictions(
    rows: Iterable[dict[str, object]],
    output_root: Path,
    *,
    input_hashes: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    payloads = sorted(rows, key=lambda row: str(row["dataset"]))
    datasets = [str(row["dataset"]) for row in payloads]
    if len(datasets) != len(set(datasets)):
        raise ValueError("duplicate Exp236 prediction dataset")
    for row in payloads:
        if FORBIDDEN_FIELDS & set(row):
            raise ValueError("Exp236 prediction contains evaluation fields")
        if row.get("contains_test_labels") is not False:
            raise ValueError("Exp236 prediction label flag must be false")
        if row.get("contains_evaluation_metrics") is not False:
            raise ValueError("Exp236 prediction metric flag must be false")
    serialized = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in payloads
    )
    prediction = Path(output_root) / "exp236_frozen_predictions.jsonl"
    manifest_path = Path(output_root) / "exp236_prediction_manifest.json"
    _atomic_write(prediction, serialized)
    manifest = {
        "schema": "exp236_prediction_manifest_v1",
        "experiment_id": EXPERIMENT_ID,
        "prediction_rows": len(payloads),
        "prediction_sha256": hashlib.sha256(serialized).hexdigest(),
        "input_hashes": dict(sorted((input_hashes or {}).items())),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "retrospective_candidate_packaging": True,
        "selected_from_prior_test_review": True,
        "deployment_eligible": False,
    }
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
    )
    return prediction, manifest_path


def load_verified_predictions(prediction: Path, manifest_path: Path) -> list[dict[str, object]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    actual = _sha256(prediction)
    if actual != manifest.get("prediction_sha256"):
        raise ValueError(
            f"prediction SHA-256 mismatch: expected={manifest.get('prediction_sha256')} actual={actual}"
        )
    rows = [
        json.loads(line)
        for line in Path(prediction).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(manifest.get("prediction_rows", -1)):
        raise ValueError("Exp236 prediction row count mismatch")
    return rows


def sha256_file(path: Path) -> str:
    return _sha256(path)
