from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .prediction import validate_prediction_row


EXPERIMENT_ID = "experiment_290_train_raw_exact_normal_guard"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def freeze_prediction_rows(
    rows: list[dict[str, object]], prediction_path: Path, manifest_path: Path
) -> dict[str, object]:
    for row in rows:
        validate_prediction_row(row)
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )
    prediction_sha256 = hashlib.sha256(payload).hexdigest()
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "prediction_sha256": prediction_sha256,
        "rows": len(rows),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "match_definition": "sqlite_values_blob_byte_exact",
        "retrospective_counterfactual": True,
        "upstream_end_to_end_strict_train_only": False,
    }
    _atomic_write(prediction_path, payload)
    _atomic_write(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest


def verify_prediction_rows(
    prediction_path: Path, manifest_path: Path
) -> list[dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(prediction_path) != manifest.get("prediction_sha256"):
        raise ValueError("prediction hash mismatch")
    rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != int(manifest.get("rows", -1)):
        raise ValueError("prediction row count mismatch")
    for row in rows:
        validate_prediction_row(row)
    return rows


__all__ = ["EXPERIMENT_ID", "freeze_prediction_rows", "sha256_file", "verify_prediction_rows"]
