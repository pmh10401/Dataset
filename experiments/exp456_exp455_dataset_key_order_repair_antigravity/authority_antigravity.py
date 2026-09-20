"""Read Exp454 once and expose immutable, label-free routing inputs for Exp456."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

from experiments.exp448_exp447_text_label_boundary_repair_antigravity.sources_antigravity import (
    capture_exp446_publication_antigravity,
    capture_hard_predictions_antigravity,
)
from experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.config_antigravity import (
    EVALUATION_LEAVES_ANTIGRAVITY,
    EXP454_EVALUATION_ROOT_ANTIGRAVITY,
)

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY
from .contracts_antigravity import (
    ConfusionAntigravity,
    DatasetRoutingAuthorityAntigravity,
    FrozenRoutingAuthorityAntigravity,
)


_TARGET_METHODS_ANTIGRAVITY = {
    "ECG5000": "hydra_local_gap",
    "ChlorineConcentration": "multirocket_local_gap",
}
_PREDICTION_KEYS_ANTIGRAVITY = frozenset({
    "sequence", "candidate_name", "dataset", "family", "method", "indices", "target_length",
})


def _strict_json_antigravity(payload: bytes, name: str) -> object:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Exp454 {name} is not JSON") from error
    return value


def _strict_jsonl_antigravity(payload: bytes, name: str) -> tuple[object, ...]:
    if not payload.endswith(b"\n"):
        raise ValueError(f"Exp454 {name} is not JSONL")
    rows = []
    for line in payload.splitlines():
        rows.append(_strict_json_antigravity(line + b"\n", name))
    return tuple(rows)


def _indices_antigravity(value: object, field: str) -> frozenset[int]:
    if type(value) is not list or any(type(index) is not int or index < 0 for index in value):
        raise ValueError(f"Exp454 {field} indices are invalid")
    if len(set(value)) != len(value):
        raise ValueError(f"Exp454 {field} indices are duplicated")
    return frozenset(value)


def _confusion_antigravity(value: object, field: str) -> ConfusionAntigravity:
    if type(value) is not dict or set(value) != {"tp", "fp", "fn", "tn"}:
        raise ValueError(f"Exp454 {field} confusion is invalid")
    try:
        return ConfusionAntigravity(value["tp"], value["fp"], value["fn"], value["tn"])
    except ValueError as error:
        raise ValueError(f"Exp454 {field} confusion is invalid") from error


def _capture_exp454_leaves_antigravity() -> dict[str, bytes]:
    root = Path(EXP454_EVALUATION_ROOT_ANTIGRAVITY)
    if not root.is_dir() or {entry.name for entry in root.iterdir()} != set(EVALUATION_LEAVES_ANTIGRAVITY):
        raise ValueError("Exp454 12-leaf root differs")
    try:
        return {name: (root / name).read_bytes() for name in EVALUATION_LEAVES_ANTIGRAVITY}
    except OSError as error:
        raise ValueError("Exp454 12-leaf root is unreadable") from error


def _validate_leaves_antigravity(config: object, leaves: Mapping[str, bytes]) -> tuple[tuple[str, str], ...]:
    if set(leaves) != set(EVALUATION_LEAVES_ANTIGRAVITY) or any(type(payload) is not bytes for payload in leaves.values()):
        raise ValueError("Exp454 12-leaf inventory differs")
    digests = {name: sha256(payload).hexdigest() for name, payload in leaves.items()}
    configured = getattr(config, "exp454_sha256", None)
    if (
        type(configured) is not tuple
        or any(type(item) is not tuple or len(item) != 2 for item in configured)
        or {item[0] for item in configured} != {"artifact_map", "evaluation", "decision"}
    ):
        raise ValueError("Exp455 frozen Exp454 hashes differ")
    configured_map = dict(configured)
    if (
        digests["artifact_sha256_antigravity.json"] != configured_map.get("artifact_map")
        or digests["evaluation_antigravity.json"] != configured_map.get("evaluation")
        or digests["decision_antigravity.json"] != configured_map.get("decision")
    ):
        raise ValueError("Exp454 configured leaf hash differs")
    artifact_map = _strict_json_antigravity(leaves["artifact_sha256_antigravity.json"], "artifact map")
    if type(artifact_map) is not dict or set(artifact_map) != set(EVALUATION_LEAVES_ANTIGRAVITY[:-1]):
        raise ValueError("Exp454 artifact map differs")
    for name in EVALUATION_LEAVES_ANTIGRAVITY[:-1]:
        if type(artifact_map[name]) is not str or artifact_map[name] != digests[name]:
            raise ValueError("Exp454 leaf hash differs")
    return tuple(sorted(digests.items()))


def _capture_routing_authority_with_dependencies_antigravity(
    config: object,
    identity: object,
    hard: object,
    leaves: Mapping[str, bytes],
) -> FrozenRoutingAuthorityAntigravity:
    """Private synthetic seam; production uses only pinned Exp446/Exp404 sources."""
    leaf_sha256 = _validate_leaves_antigravity(config, leaves)
    stable_coordinates = getattr(identity, "stable_coordinates", None)
    hard_coordinates = getattr(hard, "coordinates", None)
    if type(stable_coordinates) is not tuple or type(hard_coordinates) is not frozenset:
        raise ValueError("Exp446 or Exp404 source authority is invalid")
    stable_by_dataset: dict[str, set[int]] = {}
    hard_by_dataset: dict[str, set[int]] = {}
    for source, destination, name in (
        (stable_coordinates, stable_by_dataset, "Exp446"),
        (hard_coordinates, hard_by_dataset, "Exp404"),
    ):
        for coordinate in source:
            if type(coordinate) is not tuple or len(coordinate) != 2 or type(coordinate[0]) is not str or type(coordinate[1]) is not int or coordinate[1] < 0:
                raise ValueError(f"{name} coordinate is invalid")
            destination.setdefault(coordinate[0], set()).add(coordinate[1])

    evidence = _strict_json_antigravity(leaves["evaluation_antigravity.json"], "evaluation")
    predictions = _strict_jsonl_antigravity(leaves["frozen_predictions_antigravity.jsonl"], "frozen predictions")
    if type(evidence) is not dict or type(evidence.get("dataset_evidence")) is not list or len(evidence["dataset_evidence"]) != 48:
        raise ValueError("Exp454 dataset evidence differs")
    if len(predictions) != 96:
        raise ValueError("Exp454 frozen prediction count differs")
    candidates: dict[tuple[str, str], frozenset[int]] = {}
    for sequence, prediction in enumerate(predictions):
        if type(prediction) is not dict or set(prediction) != _PREDICTION_KEYS_ANTIGRAVITY:
            raise ValueError("Exp454 frozen prediction row differs")
        dataset, family, method = prediction.get("dataset"), prediction.get("family"), prediction.get("method")
        if (
            prediction.get("sequence") != sequence
            or type(dataset) is not str or type(family) is not str or type(method) is not str
            or prediction.get("candidate_name") != f"{dataset}__{method}"
            or type(prediction.get("target_length")) is not int or prediction["target_length"] < 0
        ):
            raise ValueError("Exp454 frozen prediction row differs")
        key = (dataset, method)
        if key in candidates:
            raise ValueError("Exp454 frozen prediction row is duplicated")
        candidates[key] = _indices_antigravity(prediction.get("indices"), "candidate")

    rows = []
    family_counts = {family: 0 for family in _TARGET_METHODS_ANTIGRAVITY}
    for entry in evidence["dataset_evidence"]:
        if type(entry) is not dict or set(entry) != {"dataset", "family", "target_length", "baseline_confusion", "candidates"}:
            raise ValueError("Exp454 dataset evidence row differs")
        dataset, family, target_length = entry["dataset"], entry["family"], entry["target_length"]
        if type(dataset) is not str or type(family) is not str or type(target_length) is not int or target_length < 0 or type(entry["candidates"]) is not dict:
            raise ValueError("Exp454 dataset evidence row differs")
        method = _TARGET_METHODS_ANTIGRAVITY.get(family)
        candidate_indices = candidate_confusion = None
        if method is not None:
            family_counts[family] += 1
            candidate_entry = entry["candidates"].get(method)
            if type(candidate_entry) is not dict or set(candidate_entry) != {"indices", "confusion"}:
                raise ValueError("Exp454 target candidate differs")
            candidate_indices = _indices_antigravity(candidate_entry["indices"], "target candidate")
            if candidates.get((dataset, method)) != candidate_indices:
                raise ValueError("Exp454 target candidate coordinate differs")
            candidate_confusion = _confusion_antigravity(candidate_entry["confusion"], "target candidate")
        baseline = frozenset(hard_by_dataset.get(dataset, set()) | stable_by_dataset.get(dataset, set()))
        rows.append(DatasetRoutingAuthorityAntigravity(
            dataset, family, target_length, method, baseline, candidate_indices,
            _confusion_antigravity(entry["baseline_confusion"], "baseline"), candidate_confusion,
        ))
    if family_counts != {"ECG5000": 5, "ChlorineConcentration": 3}:
        raise ValueError("Exp454 frozen target population differs")
    return FrozenRoutingAuthorityAntigravity(tuple(rows), leaf_sha256, 0)


def capture_routing_authority_antigravity() -> FrozenRoutingAuthorityAntigravity:
    """Capture only frozen prediction coordinates; this function never opens TEST labels."""
    identity = capture_exp446_publication_antigravity()
    if getattr(identity, "label_access_count", 0) != 0:
        raise ValueError("Exp446 source authority queried labels")
    hard = capture_hard_predictions_antigravity(identity)
    return _capture_routing_authority_with_dependencies_antigravity(
        FROZEN_CONFIG_ANTIGRAVITY, identity, hard, _capture_exp454_leaves_antigravity(),
    )


__all__ = ["capture_routing_authority_antigravity"]
