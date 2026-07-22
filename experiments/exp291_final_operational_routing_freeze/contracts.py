from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


EXPERIMENT_ID = "experiment_291_final_operational_routing_freeze"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_rows(
    rows: list[dict[str, object]],
    prediction_path: Path,
    manifest_path: Path,
    *,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    for row in rows:
        if row.get("contains_test_labels") is not False or row.get("contains_evaluation_metrics") is not False:
            raise ValueError("routing prediction must be label and metric free")
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "prediction_sha256": hashlib.sha256(payload).hexdigest(),
        "rows": len(rows),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "retrospective_integration": True,
        **(metadata or {}),
    }
    for path, data in (
        (prediction_path, payload),
        (manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
    return manifest


def verify_rows(prediction_path: Path, manifest_path: Path) -> list[dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(prediction_path) != manifest.get("prediction_sha256"):
        raise ValueError("routing prediction hash mismatch")
    rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != int(manifest.get("rows", -1)):
        raise ValueError("routing prediction row count mismatch")
    if any(row.get("contains_test_labels") is not False or row.get("contains_evaluation_metrics") is not False for row in rows):
        raise ValueError("routing prediction contains forbidden fields")
    return rows


__all__ = ["EXPERIMENT_ID", "freeze_rows", "sha256_file", "verify_rows"]
