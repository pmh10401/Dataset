from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


FORBIDDEN_FIELDS = {
    "y_test", "labels", "test_labels", "anomaly_positions",
    "tp", "fp", "fn", "precision", "recall", "f1", "combined_f1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_row(row: dict[str, object]) -> None:
    if row.get("contains_test_labels") is not False:
        raise ValueError("frozen row must declare contains_test_labels=false")
    if row.get("contains_evaluation_metrics") is not False:
        raise ValueError("frozen row must declare contains_evaluation_metrics=false")
    for key in row:
        lowered = str(key).lower()
        if lowered in FORBIDDEN_FIELDS or lowered.endswith(
            ("_tp", "_fp", "_fn", "_f1", "_precision", "_recall")
        ):
            raise ValueError(f"forbidden frozen field: {key}")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def freeze_jsonl(
    rows: list[dict[str, object]], rows_path: Path, manifest_path: Path, *,
    experiment_id: str, metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    for row in rows:
        validate_frozen_row(row)
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )
    manifest = {
        "experiment_id": str(experiment_id),
        "rows": len(rows),
        "prediction_sha256": hashlib.sha256(payload).hexdigest(),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        **(metadata or {}),
    }
    _atomic_write(Path(rows_path), payload)
    _atomic_write(
        Path(manifest_path),
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest


def verify_frozen_jsonl(rows_path: Path, manifest_path: Path) -> list[dict[str, object]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if sha256_file(Path(rows_path)) != manifest.get("prediction_sha256"):
        raise ValueError("frozen JSONL hash mismatch")
    rows = [
        json.loads(line)
        for line in Path(rows_path).read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != int(manifest.get("rows", -1)):
        raise ValueError("frozen JSONL row count mismatch")
    for row in rows:
        validate_frozen_row(row)
    return rows
