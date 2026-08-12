"""Canonical, no-clobber artifacts for values-only anomaly scores."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping

from .config import ALPHAS


SCORE_FIELDS = frozenset(
    {
        "dataset",
        "method",
        "n_train",
        "n_test",
        "status",
        "reason",
        "test_indices",
        "train_oof_scores",
        "test_scores",
        "alerts_by_alpha",
        "runtime_seconds",
        "contains_test_labels",
        "contains_evaluation_metrics",
    }
)
FORBIDDEN_EVALUATION_FIELDS = frozenset(
    {
        "label",
        "labels",
        "ground_truth",
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "f1",
        "auroc",
        "average_precision",
    }
)


def canonical_json_bytes(payload: object) -> bytes:
    """Encode strict canonical JSON with one terminal newline."""
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("payload is not strict JSON") from error
    return (encoded + "\n").encode("utf-8")


def validate_score_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate the exact label-free score-shard schema."""
    keys = set(payload)
    forbidden = keys & FORBIDDEN_EVALUATION_FIELDS
    if forbidden or keys - SCORE_FIELDS:
        raise ValueError(f"score shard contains forbidden fields: {sorted(keys - SCORE_FIELDS | forbidden)}")
    if keys != SCORE_FIELDS:
        raise ValueError(f"score shard fields are not exact: {sorted(SCORE_FIELDS - keys)}")
    dataset, method = payload["dataset"], payload["method"]
    if type(dataset) is not str or not dataset or type(method) is not str or not method:
        raise ValueError("score shard dataset and method must be non-empty strings")
    n_train, n_test = payload["n_train"], payload["n_test"]
    if type(n_train) is not int or n_train < 1 or type(n_test) is not int or n_test < 1:
        raise ValueError("score shard counts must be positive integers")
    test_indices = _integer_list(payload["test_indices"], "test_indices")
    if test_indices != list(range(n_test)):
        raise ValueError("score shard TEST indices must be contiguous from zero")
    runtime = payload["runtime_seconds"]
    if type(runtime) not in (int, float) or not math.isfinite(runtime) or runtime < 0:
        raise ValueError("score shard runtime must be finite and non-negative")
    if payload["contains_test_labels"] is not False:
        raise ValueError("score shard must not contain TEST labels")
    if payload["contains_evaluation_metrics"] is not False:
        raise ValueError("score shard must not contain evaluation metrics")
    alerts = payload["alerts_by_alpha"]
    expected_alphas = {_alpha_key(alpha) for alpha in ALPHAS}
    if not isinstance(alerts, Mapping) or set(alerts) != expected_alphas:
        raise ValueError("score shard alpha coverage is not exact")
    parsed_alerts = {
        key: _integer_list(value, f"alerts_by_alpha[{key}]")
        for key, value in alerts.items()
    }
    if any(
        values != sorted(set(values)) or any(index not in test_indices for index in values)
        for values in parsed_alerts.values()
    ):
        raise ValueError("score shard alerts are invalid")
    oof = _finite_number_list(payload["train_oof_scores"], "train_oof_scores")
    test = _finite_number_list(payload["test_scores"], "test_scores")
    status, reason = payload["status"], payload["reason"]
    if status == "complete":
        if reason is not None or len(oof) != n_train or len(test) != n_test:
            raise ValueError("complete score shard is inconsistent")
    elif status == "abstain":
        if type(reason) is not str or not reason or oof or test:
            raise ValueError("abstaining score shard is inconsistent")
        if any(parsed_alerts.values()):
            raise ValueError("abstaining score shard cannot emit alerts")
    else:
        raise ValueError("score shard status must be complete or abstain")
    return dict(payload)


def write_score_shard_create_only(
    path: Path, payload: Mapping[str, object]
) -> str:
    """Create one canonical score shard or reuse byte-identical content."""
    validated = validate_score_payload(payload)
    encoded = canonical_json_bytes(validated)
    write_create_only_bytes(path, encoded)
    return hashlib.sha256(encoded).hexdigest()


def write_create_only_bytes(path: Path, payload: bytes) -> None:
    """Atomically publish bytes without replacing a pre-existing artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ValueError(f"existing artifact mismatch: {destination}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != payload:
                raise ValueError(f"existing artifact mismatch: {destination}")
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def read_strict_json(path: Path) -> dict[str, object]:
    """Read a UTF-8 JSON object while rejecting duplicate keys and NaN."""
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _finite_number_list(value: object, name: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    if any(type(item) not in (int, float) or not math.isfinite(item) for item in value):
        raise ValueError(f"{name} must contain only finite numbers")
    return [float(item) for item in value]


def _integer_list(value: object, name: str) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int or item < 0 for item in value):
        raise ValueError(f"{name} must contain only non-negative integers")
    return list(value)


def _alpha_key(alpha: float) -> str:
    return format(float(alpha), ".12g")


__all__ = [
    "canonical_json_bytes",
    "read_strict_json",
    "sha256_file",
    "validate_score_payload",
    "write_create_only_bytes",
    "write_score_shard_create_only",
]
