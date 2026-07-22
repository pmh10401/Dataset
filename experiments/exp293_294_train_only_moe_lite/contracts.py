from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


FORBIDDEN = {"y_test", "labels", "test_labels", "anomaly_positions", "tp", "fp", "fn", "precision", "recall", "f1", "combined_f1"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(row: dict[str, object]) -> None:
    if row.get("contains_test_labels") is not False or row.get("contains_evaluation_metrics") is not False:
        raise ValueError("forbidden labels or metrics in prediction")
    for key in row:
        lower = key.lower()
        if lower in FORBIDDEN or lower.endswith(("_tp", "_fp", "_fn", "_f1", "_precision", "_recall")):
            raise ValueError(f"forbidden prediction field: {key}")


def freeze_rows(rows: list[dict[str, object]], prediction_path: Path, manifest_path: Path, *, experiment_id: str, metadata: dict[str, object] | None = None) -> dict[str, object]:
    for row in rows:
        _validate(row)
    payload = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows)
    manifest = {"experiment_id": experiment_id, "rows": len(rows), "prediction_sha256": hashlib.sha256(payload).hexdigest(), "contains_test_labels": False, "contains_evaluation_metrics": False, **(metadata or {})}
    for path, content in ((prediction_path, payload), (manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(content)
        os.replace(temporary, path)
    return manifest


def verify_rows(prediction_path: Path, manifest_path: Path) -> list[dict[str, object]]:
    manifest = json.loads(manifest_path.read_text())
    if sha256_file(prediction_path) != manifest.get("prediction_sha256"):
        raise ValueError("prediction hash mismatch")
    rows = [json.loads(line) for line in prediction_path.read_text().splitlines() if line]
    if len(rows) != int(manifest.get("rows", -1)):
        raise ValueError("prediction row count mismatch")
    for row in rows:
        _validate(row)
    return rows
