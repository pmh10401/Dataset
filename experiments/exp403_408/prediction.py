"""Label-free Exp403/404 prediction artifacts and their tamper boundary."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import re
from types import MappingProxyType

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
    EXPECTED_DATASETS,
    EXPECTED_HARD,
    DatasetInput,
    FrozenConfig,
    PredictionRow,
    _is_forbidden_key,
    train_eligibility,
    validate_prediction_row,
)
from .foundation import FrozenLaneUniverse


FEATURE_MANIFEST_NAME = "exp403_feature_manifest.csv"
HARD_CONFIDENCE_NAME = "exp404_hard_confidence.jsonl"
PREDICTION_MANIFEST_NAME = "exp404_prediction_manifest.json"
_CONFIDENCE_GRADES = frozenset({"high", "medium", "recheck", "abstain"})


_ALLOWED_DECLARATION_FIELDS = frozenset(
    {"contains_test_labels", "contains_evaluation_metrics"}
)
_FORBIDDEN_PAYLOAD_TOKENS = frozenset(
    {
        "accuracy",
        "auc",
        "evaluation",
        "f1",
        "fn",
        "fp",
        "label",
        "labels",
        "loss",
        "metric",
        "metrics",
        "precision",
        "recall",
        "recommendation",
        "recommendations",
        "tp",
    }
)
_ROW_FIELDS = frozenset(
    {
        "dataset",
        "candidate_index",
        "scores_by_alpha",
        "contains_test_labels",
        "contains_evaluation_metrics",
        "confidence_by_alpha",
        "eligibility",
        "abstention_reason",
    }
)
_PROVENANCE_CATEGORIES = (
    "feature_artifacts",
    "input_artifacts",
    "baseline_artifacts",
    "config_artifacts",
    "code_artifacts",
)
_EXPECTED_PROVENANCE_PROOF = object()


class ExpectedProvenance:
    """Immutable source-artifact contract supplied independently of a manifest."""

    __slots__ = ("_payload",)

    def __init__(
        self,
        payload: Mapping[str, Mapping[str, Mapping[str, str]]],
        *,
        _proof: object,
    ) -> None:
        if _proof is not _EXPECTED_PROVENANCE_PROOF:
            raise TypeError("ExpectedProvenance instances must come from from_paths")
        if set(payload) != set(_PROVENANCE_CATEGORIES):
            raise ValueError("expected provenance contract mismatch")
        self._payload = MappingProxyType({
            category: MappingProxyType({
                str(name): MappingProxyType({
                    "path": str(details["path"]),
                    "sha256": str(details["sha256"]),
                })
                for name, details in artifacts.items()
            })
            for category, artifacts in payload.items()
        })

    @classmethod
    def from_paths(
        cls,
        *,
        feature_artifacts: Mapping[str, Path] | None = None,
        input_artifacts: Mapping[str, Path] | None = None,
        baseline_artifacts: Mapping[str, Path] | None = None,
        config_artifacts: Mapping[str, Path] | None = None,
        code_artifacts: Mapping[str, Path] | None = None,
    ) -> "ExpectedProvenance":
        return cls(
            _freeze_provenance(
                feature_artifacts=feature_artifacts or {},
                input_artifacts=input_artifacts or {},
                baseline_artifacts=baseline_artifacts or {},
                config_artifacts=config_artifacts or {},
                code_artifacts=code_artifacts or {},
            ),
            _proof=_EXPECTED_PROVENANCE_PROOF,
        )

    def manifest_payload(self) -> dict[str, dict[str, dict[str, str]]]:
        return {
            category: {
                name: dict(details) for name, details in artifacts.items()
            }
            for category, artifacts in self._payload.items()
        }


def find_forbidden_fields(value: object, path: str = "prediction") -> tuple[str, ...]:
    """Return every label, metric, or recommendation field in a payload."""
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if _is_forbidden_payload_key(key):
                found.append(child_path)
            found.extend(find_forbidden_fields(child, child_path))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(find_forbidden_fields(child, f"{path}[{index}]"))
    return tuple(found)


def predict_dataset(bundle: DatasetInput, config: FrozenConfig) -> tuple[PredictionRow, ...]:
    """Produce one label-free confidence row for every frozen Hard candidate."""
    _validate_config(config)
    if int(bundle.target_length) != int(config.target_length):
        raise ValueError("DatasetInput target_length must match FrozenConfig")
    test_values = np.asarray(bundle.test_values, dtype=float)
    if test_values.ndim != 2 or not len(test_values) or not np.isfinite(test_values).all():
        raise ValueError("test_values must be a non-empty finite matrix")
    hard_indices = tuple(sorted({int(index) for index in bundle.hard_indices}))
    if not hard_indices:
        raise ValueError("prediction requires one frozen Hard candidate")
    if any(index < 0 or index >= len(test_values) for index in hard_indices):
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
        return tuple(
            PredictionRow(
                dataset=str(bundle.dataset),
                candidate_index=candidate_index,
                scores_by_alpha={key: 1.0 for key in ALPHA_KEYS},
                confidence_by_alpha={key: "abstain" for key in ALPHA_KEYS},
                eligibility=eligibility.level,
                abstention_reason="insufficient_train_normal",
            )
            for candidate_index in hard_indices
        )
    return tuple(
        PredictionRow(
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
        for candidate_index in hard_indices
    )


def write_prediction(
    rows: Sequence[PredictionRow | Mapping[str, object]],
    output: Path,
    *,
    config: FrozenConfig,
    frozen_lane_universe: FrozenLaneUniverse,
    expected_provenance: ExpectedProvenance,
) -> dict[str, object]:
    """Atomically freeze sorted Exp403/404 prediction artifacts and hashes."""
    _validate_config(config)
    output = Path(output)
    ordered = _validated_rows(rows)
    serialized_rows = [_row_mapping(row) for row in ordered]
    frozen_lanes = _verified_hard_lanes(frozen_lane_universe)
    _validate_lane_coverage(serialized_rows, frozen_lanes)
    provenance = _expected_provenance_payload(expected_provenance)
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
    artifact_hashes = {
        "config": _config_provenance_sha256(config, provenance["config_artifacts"]),
        "input": _provenance_sha256(provenance["input_artifacts"]),
        "feature": _provenance_sha256(provenance["feature_artifacts"]),
        "prediction": prediction_sha256,
        "code": _provenance_sha256(provenance["code_artifacts"]),
        "baseline": _provenance_sha256(provenance["baseline_artifacts"]),
        "feature_manifest": feature_sha256,
    }
    manifest: dict[str, object] = {
        "experiment_ids": [403, 404],
        "feature_manifest_file": FEATURE_MANIFEST_NAME,
        "hard_confidence_file": HARD_CONFIDENCE_NAME,
        "config": asdict(config),
        "frozen_hard_lanes": frozen_lanes,
        "frozen_hard_lanes_sha256": sha256_bytes(canonical_json_bytes(frozen_lanes)),
        "provenance": provenance,
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


def verify_prediction(
    output: Path,
    *,
    frozen_lane_universe: FrozenLaneUniverse,
    expected_provenance: ExpectedProvenance,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Load only a complete, hash-bound label-free prediction publication."""
    output = Path(output)
    manifest_path = output / PREDICTION_MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read prediction manifest") from error
    if not isinstance(manifest, dict):
        raise ValueError("prediction manifest must be an object")
    _validate_manifest_fields(manifest)
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
        "config", "input", "feature", "prediction", "code", "baseline", "feature_manifest"
    }:
        raise ValueError("prediction manifest hash contract mismatch")
    frozen_lanes = _verified_hard_lanes(frozen_lane_universe)
    if manifest.get("frozen_hard_lanes") != frozen_lanes:
        raise ValueError("prediction frozen Hard lane universe mismatch")
    if manifest.get("frozen_hard_lanes_sha256") != sha256_bytes(canonical_json_bytes(frozen_lanes)):
        raise ValueError("prediction frozen Hard lane SHA256 mismatch")
    provenance = _expected_provenance_payload(expected_provenance)
    if manifest.get("provenance") != provenance:
        raise ValueError("prediction expected provenance mismatch")
    _verify_manifest_hashes(output, config, provenance, expected_hashes)
    rows = _read_rows(output / HARD_CONFIDENCE_NAME)
    if rows != sorted(rows, key=lambda row: (str(row["dataset"]), int(row["candidate_index"]))):
        raise ValueError("prediction rows are not canonically sorted")
    if len(rows) != manifest.get("rows") or len({row["dataset"] for row in rows}) != manifest.get("datasets"):
        raise ValueError("prediction manifest coverage mismatch")
    _validate_lane_coverage(rows, frozen_lanes)
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
    if find_forbidden_fields(row):
        raise ValueError("forbidden prediction field")
    if set(row) != _ROW_FIELDS:
        raise ValueError("prediction row has unsupported fields")
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
    provenance: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    provenance = _verify_provenance(provenance)
    feature_sha256 = verify_sha256(
        output / FEATURE_MANIFEST_NAME, str(expected.get("feature_manifest"))
    )
    prediction_sha256 = verify_sha256(
        output / HARD_CONFIDENCE_NAME, str(expected.get("prediction"))
    )
    observed = {
        "config": _config_provenance_sha256(config, provenance["config_artifacts"]),
        "input": _provenance_sha256(provenance["input_artifacts"]),
        "feature": _provenance_sha256(provenance["feature_artifacts"]),
        "prediction": prediction_sha256,
        "code": _provenance_sha256(provenance["code_artifacts"]),
        "baseline": _provenance_sha256(provenance["baseline_artifacts"]),
        "feature_manifest": feature_sha256,
    }
    for name, digest in observed.items():
        if expected.get(name) != digest:
            raise ValueError(f"prediction {name} SHA256 mismatch")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"prediction manifest {name} is invalid")
    return {str(key): item for key, item in value.items()}


def _read_rows(path: Path) -> list[dict[str, object]]:
    try:
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read prediction rows") from error
    return [_row_mapping(value) for value in values]


def _is_forbidden_payload_key(key: object) -> bool:
    lowered = str(key).lower()
    if lowered in _ALLOWED_DECLARATION_FIELDS:
        return False
    if _is_forbidden_key(key):
        return True
    return bool(set(re.findall(r"[a-z0-9]+", lowered)) & _FORBIDDEN_PAYLOAD_TOKENS)


def _normalize_frozen_hard_lanes(
    lanes: Mapping[str, Collection[int]],
) -> dict[str, list[int]]:
    normalized: dict[str, list[int]] = {}
    for dataset, values in lanes.items():
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("frozen Hard lane dataset is invalid")
        try:
            indices = sorted({int(index) for index in values})
        except TypeError as error:
            raise ValueError("frozen Hard lane indices are invalid") from error
        if any(index < 0 for index in indices):
            raise ValueError("frozen Hard lane index is invalid")
        normalized[dataset] = indices
    return dict(sorted(normalized.items()))


def _verified_hard_lanes(universe: FrozenLaneUniverse) -> dict[str, list[int]]:
    if not isinstance(universe, FrozenLaneUniverse):
        raise TypeError("frozen_lane_universe must be a verified FrozenLaneUniverse")
    lanes = _normalize_frozen_hard_lanes(universe.hard_lanes())
    if len(lanes) != EXPECTED_DATASETS:
        raise ValueError("frozen Hard lane dataset coverage mismatch")
    if sum(len(indices) for indices in lanes.values()) != EXPECTED_HARD:
        raise ValueError("frozen Hard lane total mismatch")
    return lanes


def _validate_lane_coverage(
    rows: Sequence[Mapping[str, object]], lanes: Mapping[str, Sequence[int]]
) -> None:
    actual = {(str(row["dataset"]), int(row["candidate_index"])) for row in rows}
    expected = {(dataset, index) for dataset, indices in lanes.items() for index in indices}
    if actual != expected:
        raise ValueError("frozen Hard lane coverage mismatch")


def _freeze_provenance(**sources: Mapping[str, Path]) -> dict[str, dict[str, dict[str, str]]]:
    frozen = {
        category: _freeze_provenance_category(category, sources.get(category, {}))
        for category in _PROVENANCE_CATEGORIES
    }
    if "prediction_boundary" in frozen["code_artifacts"]:
        raise ValueError("code_artifacts reserves prediction_boundary provenance")
    frozen["code_artifacts"]["prediction_boundary"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256_file(Path(__file__)),
    }
    frozen["code_artifacts"] = dict(sorted(frozen["code_artifacts"].items()))
    return frozen


def _expected_provenance_payload(
    expected: ExpectedProvenance,
) -> dict[str, dict[str, dict[str, str]]]:
    if not isinstance(expected, ExpectedProvenance):
        raise TypeError("expected_provenance must be an ExpectedProvenance")
    return expected.manifest_payload()


def _freeze_provenance_category(
    category: str, artifacts: Mapping[str, Path]
) -> dict[str, dict[str, str]]:
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError(f"{category} provenance must be non-empty")
    frozen: dict[str, dict[str, str]] = {}
    for name, raw_path in artifacts.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{category} provenance name is invalid")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise ValueError(f"{category} provenance artifact is missing")
        frozen[name] = {"path": str(path), "sha256": sha256_file(path)}
    return dict(sorted(frozen.items()))


def _verify_provenance(value: Mapping[str, object]) -> dict[str, dict[str, dict[str, str]]]:
    if set(value) != set(_PROVENANCE_CATEGORIES):
        raise ValueError("prediction provenance contract mismatch")
    verified: dict[str, dict[str, dict[str, str]]] = {}
    for category in _PROVENANCE_CATEGORIES:
        entries = _mapping(value[category], category)
        if not entries:
            raise ValueError(f"{category} provenance must be non-empty")
        checked: dict[str, dict[str, str]] = {}
        for name, entry in entries.items():
            details = _mapping(entry, f"{category}.{name}")
            if set(details) != {"path", "sha256"} or not isinstance(details["path"], str) or not isinstance(details["sha256"], str):
                raise ValueError(f"{category} provenance entry is invalid")
            path = Path(details["path"])
            checked[name] = {"path": str(path), "sha256": verify_sha256(path, details["sha256"])}
        verified[category] = dict(sorted(checked.items()))
    return verified


def _validate_manifest_fields(manifest: Mapping[str, object]) -> None:
    required = {
        "experiment_ids",
        "feature_manifest_file",
        "hard_confidence_file",
        "config",
        "frozen_hard_lanes",
        "frozen_hard_lanes_sha256",
        "provenance",
        "artifact_hashes",
        "rows",
        "datasets",
        "contains_test_labels",
        "contains_evaluation_metrics",
    }
    if set(manifest) != required:
        raise ValueError("prediction manifest has unsupported fields")


def _config_provenance_sha256(
    config: FrozenConfig, config_artifacts: Mapping[str, object]
) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {"config": asdict(config), "config_artifacts": config_artifacts}
        )
    )


def _provenance_sha256(provenance: Mapping[str, object]) -> str:
    return sha256_bytes(canonical_json_bytes(provenance))
