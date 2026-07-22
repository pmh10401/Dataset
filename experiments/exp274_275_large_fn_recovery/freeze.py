from __future__ import annotations

import hashlib
import csv
from io import StringIO
import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path

from experiments.exp267_273_zero_support_fn_recovery.recovery_routing import (
    verify_recovery_prediction_hash,
)

from .config import EXP239_ROOT, EXP254_ROOT, EXP272_ROOT, PRIMARY_ALPHA
from .policy import build_prediction_row


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(raw, path)
    finally:
        Path(raw).unlink(missing_ok=True)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _verify_external_prediction(root: Path) -> tuple[Path, Path, dict[str, object]]:
    prediction, manifest_path = root / "frozen_predictions.jsonl", root / "prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _sha256(prediction) != manifest.get("prediction_sha256"):
        raise ValueError(f"external prediction hash mismatch: {root.name}")
    if manifest.get("contains_test_labels") is not False:
        raise ValueError(f"external prediction contains labels: {root.name}")
    if int(manifest.get("datasets", 0)) != 1117:
        raise ValueError(f"external prediction population mismatch: {root.name}")
    return prediction, manifest_path, manifest


def _candidate_map(
    rows: Iterable[Mapping[str, object]], *, variant: str,
) -> dict[str, set[int]]:
    selected: dict[str, set[int]] = {}
    for row in rows:
        if float(row.get("alpha", -1.0)) != PRIMARY_ALPHA or row.get("variant") != variant:
            continue
        dataset = str(row["dataset"])
        values = row.get("candidate_indices")
        if dataset in selected or not isinstance(values, list) or any(type(value) is not int for value in values):
            raise ValueError(f"invalid or duplicate frozen source row: {dataset}")
        selected[dataset] = set(values)
    return selected


def freeze_predictions(
    *, output_root: Path, expected_datasets: int = 1117, dataset_limit: int | None = None,
) -> tuple[Path, Path]:
    exp272_prediction = EXP272_ROOT / "exp272_frozen_predictions.jsonl"
    exp272_manifest = EXP272_ROOT / "exp272_prediction_manifest.json"
    exp272_rows = _read_jsonl(exp272_prediction)
    full_datasets = tuple(sorted(str(row["dataset"]) for row in exp272_rows))
    if len(full_datasets) != 1117 or len(full_datasets) != len(set(full_datasets)):
        raise ValueError("Exp272 population mismatch")
    verify_recovery_prediction_hash(exp272_prediction, exp272_manifest, expected_datasets=full_datasets)
    datasets = full_datasets[:dataset_limit] if dataset_limit is not None else full_datasets
    if len(datasets) != expected_datasets:
        raise ValueError("requested output population mismatch")

    wave_path, wave_manifest_path, _ = _verify_external_prediction(EXP239_ROOT)
    matrix_path, matrix_manifest_path, _ = _verify_external_prediction(EXP254_ROOT)
    wavelet = _candidate_map(_read_jsonl(wave_path), variant="default")
    matrix = _candidate_map(_read_jsonl(matrix_path), variant="matrix_profile_16")
    if set(wavelet) != set(full_datasets) or set(matrix) != set(full_datasets):
        raise ValueError("external candidate coverage mismatch")

    input_hashes = {
        "exp272_prediction": _sha256(exp272_prediction),
        "exp272_manifest": _sha256(exp272_manifest),
        "exp239_prediction": _sha256(wave_path),
        "exp239_manifest": _sha256(wave_manifest_path),
        "exp254_prediction": _sha256(matrix_path),
        "exp254_manifest": _sha256(matrix_manifest_path),
    }
    by_dataset = {str(row["dataset"]): row for row in exp272_rows}
    frozen = [
        build_prediction_row(
            by_dataset[dataset], wavelet=wavelet[dataset], matrix_profile=matrix[dataset],
            input_sha256=input_hashes,
        )
        for dataset in datasets
    ]
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        for row in frozen
    )
    prediction_path = Path(output_root) / "exp274_frozen_predictions.jsonl"
    manifest_path = Path(output_root) / "exp274_prediction_manifest.json"
    _atomic_bytes(prediction_path, payload)
    manifest = {
        "experiment_id": "experiment_274_large_train_multisource_fn_recovery",
        "datasets": len(frozen), "rows": len(frozen),
        "prediction_sha256": hashlib.sha256(payload).hexdigest(),
        "contains_test_labels": False, "contains_evaluation_metrics": False,
        "hard_policy_changed": False, "review_only": True,
        "retrospective": True, "input_sha256": input_hashes,
    }
    _atomic_bytes(manifest_path, (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode())
    summary_buffer = StringIO(newline="")
    summary = {
        "datasets": len(frozen),
        "eligible_datasets": sum(bool(row["large_train_eligible"]) for row in frozen),
        "forecast_wavelet_matrix_count": sum(len(row["forecast_wavelet_matrix_indices"]) for row in frozen),
        "any_source_wavelet_matrix_count": sum(len(row["any_source_wavelet_matrix_indices"]) for row in frozen),
        "shadow_either_structure_count": sum(len(row["shadow_either_structure_indices"]) for row in frozen),
        "shadow_both_structures_count": sum(len(row["shadow_both_structures_indices"]) for row in frozen),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "hard_policy_changed": False,
    }
    writer = csv.DictWriter(summary_buffer, fieldnames=list(summary))
    writer.writeheader(); writer.writerow(summary)
    _atomic_bytes(Path(output_root) / "exp274_summary.csv", summary_buffer.getvalue().encode())
    return prediction_path, manifest_path


def verify_prediction(
    prediction_path: Path, manifest_path: Path, *, expected_datasets: Iterable[str],
) -> list[dict[str, object]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if _sha256(prediction_path) != manifest.get("prediction_sha256"):
        raise ValueError("prediction hash mismatch")
    if manifest.get("contains_test_labels") is not False or manifest.get("contains_evaluation_metrics") is not False:
        raise ValueError("prediction metric boundary mismatch")
    rows = _read_jsonl(prediction_path)
    expected = tuple(sorted(str(value) for value in expected_datasets))
    names = tuple(str(row.get("dataset")) for row in rows)
    if names != expected or len(rows) != int(manifest.get("rows", -1)):
        raise ValueError("prediction population mismatch")
    for row in rows:
        if row.get("contains_test_labels") is not False or row.get("contains_evaluation_metrics") is not False:
            raise ValueError("prediction row metric boundary mismatch")
        n_test = int(row["n_test"])
        for key, values in row.items():
            if key.endswith("_indices"):
                if not isinstance(values, list) or values != sorted(set(values)):
                    raise ValueError(f"invalid prediction indices: {key}")
                if any(type(index) is not int or index < 0 or index >= n_test for index in values):
                    raise ValueError(f"prediction index outside range: {key}")
    return rows


__all__ = ["freeze_predictions", "verify_prediction"]
