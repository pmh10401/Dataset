"""Label-free Exp403/404 prediction artifacts and their tamper boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .artifact_io import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    verify_sha256,
    write_csv_atomic,
    write_json_atomic,
    write_jsonl_atomic,
)
from .classical_committee import grade_frozen_hard, score_classical_committee
from .contracts import (
    ALPHA_KEYS,
    ALPHAS,
    DatasetInput,
    FrozenConfig,
    PredictionRow,
    _is_forbidden_key,
    train_eligibility,
    validate_prediction_row,
)


FEATURE_MANIFEST_NAME = "exp403_feature_manifest.csv"
HARD_CONFIDENCE_NAME = "exp404_hard_confidence.jsonl"
PREDICTION_MANIFEST_NAME = "exp404_prediction_manifest.json"
_CONFIDENCE_GRADES = frozenset({"high", "medium", "recheck", "abstain"})


def find_forbidden_fields(value: object, path: str = "prediction") -> tuple[str, ...]:
    """Return every label or evaluation field found in a prediction payload."""
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if _is_forbidden_key(key):
                found.append(child_path)
            found.extend(find_forbidden_fields(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(find_forbidden_fields(child, f"{path}[{index}]"))
    return tuple(found)


def predict_dataset(bundle: DatasetInput, config: FrozenConfig) -> PredictionRow:
    """Produce one label-free confidence row for the first frozen Hard candidate.

    The caller supplies a single dataset bundle and never supplies TEST labels.
    Multi-candidate publication is performed by calling this boundary per
    candidate bundle, then passing the resulting rows to ``write_prediction``.
    """
    _validate_config(config)
    if int(bundle.target_length) != int(config.target_length):
        raise ValueError("DatasetInput target_length must match FrozenConfig")
    test_values = np.asarray(bundle.test_values, dtype=float)
    if test_values.ndim != 2 or not len(test_values) or not np.isfinite(test_values).all():
        raise ValueError("test_values must be a non-empty finite matrix")
    hard_indices = tuple(sorted({int(index) for index in bundle.hard_indices}))
    if not hard_indices:
        raise ValueError("prediction requires one frozen Hard candidate")
    candidate_index = hard_indices[0]
    if candidate_index < 0 or candidate_index >= len(test_values):
        raise ValueError("frozen Hard candidate index outside test_values")

    eligibility = train_eligibility(len(bundle.train_normal))
    committee = score_classical_committee(
        bundle.train_normal,
        test_values,
        eligibility,
        seed=config.seed,
    )
    graded = grade_frozen_hard(frozen_hard=hard_indices, committee=committee)
    if not committee.available_models:
        return PredictionRow(
            dataset=str(bundle.dataset),
            candidate_index=candidate_index,
            scores_by_alpha={key: 1.0 for key in ALPHA_KEYS},
            confidence_by_alpha={key: "abstain" for key in ALPHA_KEYS},
            eligibility=eligibility.level,
            abstention_reason="insufficient_train_normal",
        )
    return PredictionRow(
        dataset=str(bundle.dataset),
        candidate_index=candidate_index,
        scores_by_alpha={
            key: float(graded.scores_by_alpha[key][candidate_index]) for key in ALPHA_KEYS
        },
        confidence_by_alpha={
            key: str(graded.confidence_by_alpha[key][candidate_index]) for key in ALPHA_KEYS
        },
        eligibility=eligibility.level,
    )


def write_prediction(
    rows: Sequence[PredictionRow | Mapping[str, object]],
    output: Path,
    *,
    config: FrozenConfig = FrozenConfig(),
    input_hashes: Mapping[str, str] | None = None,
    baseline_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Atomically freeze sorted Exp403/404 prediction artifacts and hashes."""
    _validate_config(config)
    output = Path(output)
    ordered = _validated_rows(rows)
    serialized_rows = [_row_mapping(row) for row in ordered]
    feature_rows = [
        {
            "dataset": row["dataset"],
            "candidate_index": row["candidate_index"],
            "feature_sha256": sha256_bytes(
                canonical_json_bytes(
                    {
                        "scores_by_alpha": row["scores_by_alpha"],
                        "eligibility": row["eligibility"],
                    }
                )
            ),
        }
        for row in serialized_rows
    ]
    feature_path = output / FEATURE_MANIFEST_NAME
    prediction_path = output / HARD_CONFIDENCE_NAME
    feature_sha256 = write_csv_atomic(
        feature_path,
        fieldnames=("dataset", "candidate_index", "feature_sha256"),
        rows=feature_rows,
    )
    prediction_sha256 = write_jsonl_atomic(prediction_path, serialized_rows)
    input_payload = _sorted_hashes(input_hashes)
    baseline_payload = _sorted_hashes(baseline_hashes)
    artifact_hashes = {
        "config": sha256_bytes(canonical_json_bytes(asdict(config))),
        "input": sha256_bytes(canonical_json_bytes(input_payload)),
        "feature": feature_sha256,
        "prediction": prediction_sha256,
        "code": sha256_file(Path(__file__)),
        "baseline": sha256_bytes(canonical_json_bytes(baseline_payload)),
    }
    manifest: dict[str, object] = {
        "experiment_ids": [403, 404],
        "feature_manifest_file": FEATURE_MANIFEST_NAME,
        "hard_confidence_file": HARD_CONFIDENCE_NAME,
        "config": asdict(config),
        "input_hashes": input_payload,
        "baseline_hashes": baseline_payload,
        "artifact_hashes": artifact_hashes,
        "rows": len(serialized_rows),
        "datasets": len({row["dataset"] for row in serialized_rows}),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }
    if find_forbidden_fields(manifest):
        raise ValueError("prediction manifest contains forbidden evaluation fields")
    write_json_atomic(output / PREDICTION_MANIFEST_NAME, manifest)
    return manifest


def verify_prediction(output: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Load only a complete, hash-bound label-free prediction publication."""
    output = Path(output)
    manifest_path = output / PREDICTION_MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read prediction manifest") from error
    if not isinstance(manifest, dict):
        raise ValueError("prediction manifest must be an object")
    if find_forbidden_fields(manifest):
        raise ValueError("prediction manifest contains forbidden evaluation fields")
    if manifest.get("contains_test_labels") is not False or manifest.get("contains_evaluation_metrics") is not False:
        raise ValueError("prediction manifest must declare label-free output")
    if manifest.get("experiment_ids") != [403, 404]:
        raise ValueError("prediction manifest experiment identity mismatch")
    if manifest.get("feature_manifest_file") != FEATURE_MANIFEST_NAME or manifest.get("hard_confidence_file") != HARD_CONFIDENCE_NAME:
        raise ValueError("prediction manifest artifact names mismatch")
    config = _config_from_manifest(manifest)
    expected_hashes = manifest.get("artifact_hashes")
    if not isinstance(expected_hashes, Mapping) or set(expected_hashes) != {
        "config", "input", "feature", "prediction", "code", "baseline"
    }:
        raise ValueError("prediction manifest hash contract mismatch")
    _verify_manifest_hashes(output, config, manifest, expected_hashes)
    rows = _read_rows(output / HARD_CONFIDENCE_NAME)
    if rows != sorted(rows, key=lambda row: (str(row["dataset"]), int(row["candidate_index"]))):
        raise ValueError("prediction rows are not canonically sorted")
    if len(rows) != manifest.get("rows") or len({row["dataset"] for row in rows}) != manifest.get("datasets"):
        raise ValueError("prediction manifest coverage mismatch")
    return rows, manifest


def _validate_config(config: FrozenConfig) -> None:
    if tuple(config.alphas) != ALPHAS:
        raise ValueError("FrozenConfig alphas must match the frozen alpha contract")
    if int(config.seed) <= 0 or int(config.target_length) <= 0:
        raise ValueError("FrozenConfig seed and target_length must be positive")


def _validated_rows(rows: Sequence[PredictionRow | Mapping[str, object]]) -> list[dict[str, object]]:
    normalized = [_row_mapping(row) for row in rows]
    ordered = sorted(normalized, key=lambda row: (str(row["dataset"]), int(row["candidate_index"])))
    identities = [(row["dataset"], row["candidate_index"]) for row in ordered]
    if len(identities) != len(set(identities)):
        raise ValueError("prediction contains duplicate dataset/candidate rows")
    return ordered


def _row_mapping(row: PredictionRow | Mapping[str, object]) -> dict[str, object]:
    if is_dataclass(row):
        row = asdict(row)
    if not isinstance(row, Mapping):
        raise ValueError("prediction row must be a mapping or PredictionRow")
    normalized = validate_prediction_row(dict(row))
    if not isinstance(normalized.get("dataset"), str) or not normalized["dataset"]:
        raise ValueError("prediction dataset must be a non-empty string")
    if isinstance(normalized.get("candidate_index"), bool) or not isinstance(normalized.get("candidate_index"), int) or normalized["candidate_index"] < 0:
        raise ValueError("prediction candidate_index must be a non-negative integer")
    confidence = normalized.get("confidence_by_alpha")
    if set(confidence or {}) != set(ALPHA_KEYS) or not all(
        grade in _CONFIDENCE_GRADES for grade in (confidence or {}).values()
    ):
        raise ValueError("confidence_by_alpha must contain four valid confidence grades")
    if not isinstance(normalized.get("eligibility"), str) or not normalized["eligibility"]:
        raise ValueError("prediction eligibility is required")
    if not isinstance(normalized.get("abstention_reason"), str):
        raise ValueError("prediction abstention_reason must be a string")
    return normalized


def _sorted_hashes(values: Mapping[str, str] | None) -> dict[str, str]:
    result = {str(key): str(value) for key, value in (values or {}).items()}
    if find_forbidden_fields(result, "hashes"):
        raise ValueError("hash provenance contains forbidden evaluation fields")
    return dict(sorted(result.items()))


def _config_from_manifest(manifest: Mapping[str, object]) -> FrozenConfig:
    config_payload = manifest.get("config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("prediction manifest config is invalid")
    try:
        config = FrozenConfig(
            seed=int(config_payload["seed"]),
            alphas=tuple(float(alpha) for alpha in config_payload["alphas"]),
            target_length=int(config_payload["target_length"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("prediction manifest config is invalid") from error
    _validate_config(config)
    return config


def _verify_manifest_hashes(
    output: Path,
    config: FrozenConfig,
    manifest: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    input_payload = _sorted_hashes(_hash_mapping(manifest.get("input_hashes"), "input_hashes"))
    baseline_payload = _sorted_hashes(_hash_mapping(manifest.get("baseline_hashes"), "baseline_hashes"))
    feature_sha256 = verify_sha256(
        output / FEATURE_MANIFEST_NAME, str(expected.get("feature"))
    )
    prediction_sha256 = verify_sha256(
        output / HARD_CONFIDENCE_NAME, str(expected.get("prediction"))
    )
    observed = {
        "config": sha256_bytes(canonical_json_bytes(asdict(config))),
        "input": sha256_bytes(canonical_json_bytes(input_payload)),
        "feature": feature_sha256,
        "prediction": prediction_sha256,
        "code": sha256_file(Path(__file__)),
        "baseline": sha256_bytes(canonical_json_bytes(baseline_payload)),
    }
    for name, digest in observed.items():
        if expected.get(name) != digest:
            raise ValueError(f"prediction {name} SHA256 mismatch")


def _hash_mapping(value: object, name: str) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"prediction manifest {name} is invalid")
    return {str(key): str(item) for key, item in value.items()}


def _read_rows(path: Path) -> list[dict[str, object]]:
    try:
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read prediction rows") from error
    return [_row_mapping(value) for value in values]
