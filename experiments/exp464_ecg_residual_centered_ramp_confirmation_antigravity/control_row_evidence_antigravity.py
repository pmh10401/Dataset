"""Tracked Exp463 control-row evidence for Exp464 authority validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

from .config_antigravity import FROZEN_CONFIG_ANTIGRAVITY


_EVIDENCE_PATH_ANTIGRAVITY = (
    Path(__file__).resolve().parent / "exp463_control_row_evidence_antigravity.jsonl"
)
_CONTROL_METHOD_MAP_ANTIGRAVITY = MappingProxyType(
    {"M0": "C0", "M2": "C1", "M4": "C2", "M6": "C7"}
)
_CONTROL_METHOD_ORDER_ANTIGRAVITY = ("C0", "C1", "C2", "C7")
_CHANGE_TYPES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)


def _canonical_json_bytes_antigravity(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"


def _strict_json_antigravity(payload: bytes) -> object:
    def no_duplicates_antigravity(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant_antigravity(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=no_duplicates_antigravity,
        parse_constant=reject_constant_antigravity,
    )
    if payload != _canonical_json_bytes_antigravity(value):
        raise ValueError("control-row evidence must be canonical JSONL")
    return value


def _strict_jsonl_antigravity(path: Path) -> tuple[object, ...]:
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != FROZEN_CONFIG_ANTIGRAVITY.predecessor_control_row_evidence_sha256:
        raise ValueError("Exp463 control-row evidence SHA-256 drifted from the frozen contract")
    if not payload.endswith(b"\n"):
        raise ValueError("control-row evidence JSONL must end with newline")
    return tuple(_strict_json_antigravity(line + b"\n") for line in payload.splitlines())


def _sha64_antigravity(value: object, name: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase 64-hex digest")
    return value


def _float_antigravity(value: object, name: str) -> float:
    if type(value) is not float or not isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return value


def _score_vector_antigravity(value: object, name: str) -> tuple[float, ...]:
    if type(value) is not list or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return tuple(_float_antigravity(item, f"{name}[]") for item in value)


def _primary_scores_antigravity(value: object) -> Mapping[str, tuple[float, ...]]:
    if not isinstance(value, Mapping) or set(value) != set(_CHANGE_TYPES_ANTIGRAVITY):
        raise ValueError("primary_scores must contain the exact frozen change types")
    return MappingProxyType(
        {
            change_type: _score_vector_antigravity(
                value[change_type],
                f"primary_scores[{change_type}]",
            )
            for change_type in _CHANGE_TYPES_ANTIGRAVITY
        }
    )


def _control_row_digest_antigravity(
    *,
    method: str,
    seed: int,
    fold_index: int,
    normal_scores: tuple[float, ...],
    primary_scores: Mapping[str, tuple[float, ...]],
) -> str:
    payload = {
        "method": method,
        "seed": seed,
        "fold_index": fold_index,
        "normal_scores": list(normal_scores),
        "primary_scores": {
            change_type: list(primary_scores[change_type])
            for change_type in _CHANGE_TYPES_ANTIGRAVITY
        },
    }
    return sha256(_canonical_json_bytes_antigravity(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class Exp463ControlRowEvidenceAntigravity:
    metadata: Mapping[str, object]
    rows: tuple[Mapping[str, object], ...]
    control_row_digests: Mapping[str, str]


def load_control_row_evidence_antigravity() -> Exp463ControlRowEvidenceAntigravity:
    values = _strict_jsonl_antigravity(_EVIDENCE_PATH_ANTIGRAVITY)
    if len(values) != 41:
        raise ValueError("Exp463 control-row evidence must contain one metadata row and forty control rows")
    metadata = values[0]
    if not isinstance(metadata, Mapping):
        raise ValueError("control-row evidence metadata must be an object")
    expected_meta_keys = (
        "control_method_map",
        "control_row_count",
        "kind",
        "predecessor_experiment",
        "reviewed_cli_commit",
        "source_artifact_map_sha256",
        "source_artifact_map_value",
        "source_leaf_name",
        "source_leaf_sha256",
        "source_row_count",
        "task5_report_sha256",
        "terminal_closeout_commit",
    )
    if tuple(metadata) != expected_meta_keys:
        raise ValueError("control-row evidence metadata keys drifted")
    if metadata["kind"] != "exp463_control_row_provenance":
        raise ValueError("control-row evidence metadata kind is invalid")
    if metadata["predecessor_experiment"] != FROZEN_CONFIG_ANTIGRAVITY.predecessor_experiment:
        raise ValueError("control-row evidence predecessor experiment drifted")
    if metadata["terminal_closeout_commit"] != FROZEN_CONFIG_ANTIGRAVITY.predecessor_terminal_closeout_commit:
        raise ValueError("control-row evidence terminal commit drifted")
    if metadata["reviewed_cli_commit"] != "923cc4ad6dd6737f84f4ea2c4f621eb652ac54b4":
        raise ValueError("control-row evidence reviewed CLI commit drifted")
    if metadata["task5_report_sha256"] != FROZEN_CONFIG_ANTIGRAVITY.predecessor_task5_report_sha256:
        raise ValueError("control-row evidence task5 report SHA drifted")
    if metadata["source_leaf_name"] != "method_fold_metrics_antigravity.jsonl":
        raise ValueError("control-row evidence source leaf name drifted")
    _sha64_antigravity(metadata["source_leaf_sha256"], "source_leaf_sha256")
    if metadata["source_leaf_sha256"] != FROZEN_CONFIG_ANTIGRAVITY.predecessor_method_fold_metrics_sha256:
        raise ValueError("control-row evidence source leaf SHA drifted")
    _sha64_antigravity(metadata["source_artifact_map_sha256"], "source_artifact_map_sha256")
    if metadata["source_artifact_map_sha256"] != FROZEN_CONFIG_ANTIGRAVITY.predecessor_artifact_map_sha256:
        raise ValueError("control-row evidence artifact-map leaf SHA drifted")
    _sha64_antigravity(metadata["source_artifact_map_value"], "source_artifact_map_value")
    if metadata["source_artifact_map_value"] != FROZEN_CONFIG_ANTIGRAVITY.predecessor_artifact_map_sha256:
        raise ValueError("control-row evidence artifact-map value drifted")
    if metadata["control_method_map"] != dict(_CONTROL_METHOD_MAP_ANTIGRAVITY):
        raise ValueError("control-row evidence method map drifted")
    if metadata["control_row_count"] != 40 or metadata["source_row_count"] != 70:
        raise ValueError("control-row evidence counts drifted")

    expected_identities = tuple(
        (method, seed, fold_index)
        for method in _CONTROL_METHOD_ORDER_ANTIGRAVITY
        for seed in FROZEN_CONFIG_ANTIGRAVITY.hydra_seeds
        for fold_index in range(FROZEN_CONFIG_ANTIGRAVITY.folds)
    )
    rows: list[Mapping[str, object]] = []
    digests: dict[str, str] = {}
    for value, identity in zip(values[1:], expected_identities, strict=True):
        if not isinstance(value, Mapping):
            raise ValueError("control-row evidence rows must be objects")
        expected_keys = (
            "failure_reasons",
            "fold_index",
            "kind",
            "method",
            "normal_alerts",
            "normal_scores",
            "normal_trials",
            "normal_wilson_upper",
            "passed",
            "per_type_auc",
            "pooled_auc",
            "primary_scores",
            "scores_finite",
            "seed",
            "source_method",
        )
        if tuple(value) != expected_keys:
            raise ValueError("control-row evidence row keys drifted")
        if value["kind"] != "exp463_control_row":
            raise ValueError("control-row evidence row kind is invalid")
        method, seed, fold_index = identity
        if value["method"] != method or value["seed"] != seed or value["fold_index"] != fold_index:
            raise ValueError("control-row evidence row order drifted")
        source_method = value["source_method"]
        if type(source_method) is not str or _CONTROL_METHOD_MAP_ANTIGRAVITY.get(source_method) != method:
            raise ValueError("control-row evidence source method drifted")
        if type(value["normal_trials"]) is not int or value["normal_trials"] < 1:
            raise ValueError("control-row evidence normal_trials is invalid")
        if type(value["normal_alerts"]) is not int or value["normal_alerts"] < 0:
            raise ValueError("control-row evidence normal_alerts is invalid")
        _float_antigravity(value["normal_wilson_upper"], "normal_wilson_upper")
        if type(value["passed"]) is not bool or type(value["scores_finite"]) is not bool:
            raise ValueError("control-row evidence booleans drifted")
        if type(value["failure_reasons"]) is not list or any(
            type(reason) is not str or not reason for reason in value["failure_reasons"]
        ):
            raise ValueError("control-row evidence failure_reasons drifted")
        normal_scores = _score_vector_antigravity(value["normal_scores"], "normal_scores")
        primary_scores = _primary_scores_antigravity(value["primary_scores"])
        if len(normal_scores) != value["normal_trials"]:
            raise ValueError("control-row evidence normal length drifted")
        if any(len(primary_scores[change_type]) != len(normal_scores) for change_type in _CHANGE_TYPES_ANTIGRAVITY):
            raise ValueError("control-row evidence primary length drifted")
        key = f"{method}:{seed}:{fold_index}"
        digests[key] = _control_row_digest_antigravity(
            method=method,
            seed=seed,
            fold_index=fold_index,
            normal_scores=normal_scores,
            primary_scores=primary_scores,
        )
        rows.append(value)
    return Exp463ControlRowEvidenceAntigravity(
        metadata=MappingProxyType(dict(metadata)),
        rows=tuple(rows),
        control_row_digests=MappingProxyType(digests),
    )


__all__ = (
    "Exp463ControlRowEvidenceAntigravity",
    "load_control_row_evidence_antigravity",
)
