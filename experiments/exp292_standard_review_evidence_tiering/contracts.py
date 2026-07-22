from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .config import EXPERIMENT_ID


FORBIDDEN_PREDICTION_KEYS = {
    "y_test", "true_label", "true_anomaly_count", "tp", "fp", "fn", "f1",
    "precision", "recall", "combined_f1", "hard_tp", "hard_fp", "hard_f1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_label_free(row: dict[str, object]) -> None:
    if row.get("contains_test_labels") is not False or row.get("contains_evaluation_metrics") is not False:
        raise ValueError("prediction contains forbidden labels or metrics")
    for key in row:
        lowered = key.lower()
        if lowered in FORBIDDEN_PREDICTION_KEYS or lowered.endswith(("_tp", "_fp", "_fn", "_f1", "_precision", "_recall")):
            raise ValueError(f"prediction contains forbidden field: {key}")


def freeze_prediction_rows(
    rows: list[dict[str, object]], prediction_path: Path, manifest_path: Path,
    *, metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    for row in rows:
        _assert_label_free(row)
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "rows": len(rows),
        "prediction_sha256": hashlib.sha256(payload).hexdigest(),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "review_only": True,
        "retrospective": True,
        **(metadata or {}),
    }
    for path, content in (
        (prediction_path, payload),
        (manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(content)
        os.replace(temporary, path)
    return manifest


def verify_prediction_rows(prediction_path: Path, manifest_path: Path) -> list[dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(prediction_path) != manifest.get("prediction_sha256"):
        raise ValueError("prediction hash mismatch")
    rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != int(manifest.get("rows", -1)):
        raise ValueError("prediction row count mismatch")
    for row in rows:
        _assert_label_free(row)
    return rows

