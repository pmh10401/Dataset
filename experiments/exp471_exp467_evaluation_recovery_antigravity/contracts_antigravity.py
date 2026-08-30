"""Strict, deeply immutable generic artifact contracts for Exp471."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
import json
import math
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import Literal

from .config_antigravity import (
    CONDITION_ORDER_ANTIGRAVITY,
    EXP467_IMPORTED_TOTAL_ANTIGRAVITY,
    EXP467_NATIVE_TOTAL_ANTIGRAVITY,
    EXP467_SOURCE_COMMIT_ANTIGRAVITY,
    EXP467_SOURCE_CONTEXT_SHA256_ANTIGRAVITY,
    EXP467_TOTAL_SUCCESSES_ANTIGRAVITY,
    METHOD_ORDER_ANTIGRAVITY,
)


_SHA256_ANTIGRAVITY = re.compile(r"^[0-9a-f]{64}$")


def _sha256_antigravity(value: object, field_name: str) -> str:
    if type(value) is not str or _SHA256_ANTIGRAVITY.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


def _nonempty_name_antigravity(value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _positive_count_antigravity(value: object, field_name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_count_antigravity(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")
    return value


def _contained_relative_path_antigravity(value: object, field_name: str) -> str:
    path = _nonempty_name_antigravity(value, field_name)
    raw_parts = path.split("/")
    if (
        path.startswith("/")
        or "\\" in path
        or "\x00" in path
        or any(part in {"", ".", ".."} for part in raw_parts)
        or PurePosixPath(path).is_absolute()
    ):
        raise ValueError(f"{field_name} must be a contained relative path")
    return path


def _json_plain_antigravity(value: object, active: frozenset[int]) -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite float")
        return value

    if is_dataclass(value) and not isinstance(value, type):
        identity = id(value)
        if identity in active:
            raise ValueError("cyclic dataclass")
        nested_active = active | {identity}
        return {
            field.name: _json_plain_antigravity(
                getattr(value, field.name), nested_active
            )
            for field in fields(value)
        }

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError("cyclic mapping")
        nested_active = active | {identity}
        result: dict[str, object] = {}
        for key, nested in value.items():
            if type(key) is not str:
                raise ValueError("mapping key is not a string")
            if key in result:
                raise ValueError("mapping contains a duplicate key")
            result[key] = _json_plain_antigravity(nested, nested_active)
        return result

    if isinstance(value, (tuple, list)):
        identity = id(value)
        if identity in active:
            raise ValueError("cyclic sequence")
        nested_active = active | {identity}
        return [
            _json_plain_antigravity(nested, nested_active) for nested in value
        ]

    raise ValueError("unsupported canonical JSON value")


def canonical_json_bytes_exp471_antigravity(value: object) -> bytes:
    """Encode one recursively plain, finite, deterministic UTF-8 JSON record."""

    try:
        plain = _json_plain_antigravity(value, frozenset())
        return (
            json.dumps(
                plain,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Exp471 artifact data must be finite canonical JSON"
        ) from error


def _freeze_json_antigravity(value: object, field_name: str) -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        for key, nested in value.items():
            if type(key) is not str:
                raise ValueError(f"{field_name} mapping keys must be strings")
            if key in copied:
                raise ValueError(f"{field_name} contains a duplicate key")
            copied[key] = _freeze_json_antigravity(
                nested, f"{field_name}.{key}"
            )
        return MappingProxyType(dict(sorted(copied.items())))
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_json_antigravity(nested, field_name) for nested in value
        )
    raise ValueError(f"{field_name} must contain only finite canonical JSON values")


def _frozen_digest_mapping_antigravity(
    value: object,
    field_name: str,
    *,
    paths: bool = False,
    exact_keys: tuple[str, ...] | None = None,
) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field_name} must be a non-empty digest mapping")
    copied: dict[str, str] = {}
    for key, digest in value.items():
        name = (
            _contained_relative_path_antigravity(key, field_name)
            if paths
            else _nonempty_name_antigravity(key, field_name)
        )
        if name in copied:
            raise ValueError(f"{field_name} contains a duplicate identity")
        copied[name] = _sha256_antigravity(digest, f"{field_name}.{name}")
    if exact_keys is not None and tuple(sorted(copied)) != tuple(sorted(exact_keys)):
        raise ValueError(f"{field_name} keys differ from the fixed condition order")
    return MappingProxyType(dict(sorted(copied.items())))


def _fixed_method_order_antigravity(value: object) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError("method_order must be a sequence")
    methods = tuple(value)
    if methods != METHOD_ORDER_ANTIGRAVITY:
        raise ValueError("method_order differs from the fixed Exp471 method order")
    return methods


def _dataset_order_antigravity(value: object) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError("dataset_order must be a non-empty sequence")
    datasets = tuple(
        _nonempty_name_antigravity(dataset, "dataset_order dataset")
        for dataset in value
    )
    if len(datasets) != len(set(datasets)):
        raise ValueError("dataset_order contains a duplicate dataset")
    return datasets


def _finite_scores_antigravity(value: object, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{field_name} must be a finite score sequence")
    scores: list[float] = []
    for score in value:
        if type(score) not in (int, float) or not math.isfinite(float(score)):
            raise ValueError(f"{field_name} must contain only finite scores")
        scores.append(float(score))
    return tuple(scores)


def _test_indices_antigravity(value: object, test_count: int) -> tuple[int, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError("test_indices must be an integer sequence")
    indices = tuple(value)
    if (
        len(indices) != test_count
        or any(type(index) is not int or index < 0 for index in indices)
        or len(set(indices)) != test_count
    ):
        raise ValueError(
            "test_count and test_indices must preserve exact unique TEST identity"
        )
    return indices


@dataclass(frozen=True, slots=True)
class Exp471SourceAuthorityAntigravity:
    source_commit: str
    source_context_sha256: str
    inventory_sha256: str
    catalog_sha256: str
    database_sha256: str
    source_file_sha256: Mapping[str, str]
    source_tree_sha256: str
    condition_manifest_sha256: Mapping[str, str]
    total_successes: int
    imported_total: int
    native_total: int
    authority_sha256: str

    def __post_init__(self) -> None:
        if self.source_commit != EXP467_SOURCE_COMMIT_ANTIGRAVITY:
            raise ValueError("source_commit differs from the fixed Exp467 commit")
        if self.source_context_sha256 != EXP467_SOURCE_CONTEXT_SHA256_ANTIGRAVITY:
            raise ValueError(
                "source_context_sha256 differs from the fixed Exp467 context"
            )
        _sha256_antigravity(self.source_context_sha256, "source_context_sha256")
        for name in ("inventory_sha256", "catalog_sha256", "database_sha256"):
            _sha256_antigravity(getattr(self, name), name)
        source_files = _frozen_digest_mapping_antigravity(
            self.source_file_sha256,
            "source_file_sha256",
            paths=True,
        )
        source_tree_sha256 = _sha256_antigravity(
            self.source_tree_sha256, "source_tree_sha256"
        )
        observed_tree_sha256 = sha256(
            canonical_json_bytes_exp471_antigravity(source_files)
        ).hexdigest()
        if source_tree_sha256 != observed_tree_sha256:
            raise ValueError("source_tree_sha256 differs from the canonical source map")
        condition_manifests = _frozen_digest_mapping_antigravity(
            self.condition_manifest_sha256,
            "condition_manifest_sha256",
            exact_keys=CONDITION_ORDER_ANTIGRAVITY,
        )
        if (
            type(self.total_successes) is not int
            or type(self.imported_total) is not int
            or type(self.native_total) is not int
            or (
                self.total_successes,
                self.imported_total,
                self.native_total,
            )
            != (
                EXP467_TOTAL_SUCCESSES_ANTIGRAVITY,
                EXP467_IMPORTED_TOTAL_ANTIGRAVITY,
                EXP467_NATIVE_TOTAL_ANTIGRAVITY,
            )
            or self.total_successes != self.imported_total + self.native_total
        ):
            raise ValueError("Exp471 source population differs from fixed Exp467 authority")
        _sha256_antigravity(self.authority_sha256, "authority_sha256")
        object.__setattr__(self, "source_file_sha256", source_files)
        object.__setattr__(
            self, "condition_manifest_sha256", condition_manifests
        )


@dataclass(frozen=True, slots=True)
class Exp471ScoreShardAntigravity:
    source_authority_sha256: str
    cache_context_sha256: str
    method: str
    dataset: str
    status: Literal["scored", "abstained"]
    test_count: int
    test_indices: tuple[int, ...]
    raw_scores: tuple[float, ...]
    micro_scores: tuple[float, ...]
    abstention_reason: str | None
    source_manifest_sha256: str
    payload_sha256: str
    envelope_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "source_authority_sha256",
            "cache_context_sha256",
            "source_manifest_sha256",
            "payload_sha256",
            "envelope_sha256",
        ):
            _sha256_antigravity(getattr(self, name), name)
        if self.method not in METHOD_ORDER_ANTIGRAVITY:
            raise ValueError("method is outside the fixed Exp471 method order")
        _nonempty_name_antigravity(self.dataset, "dataset")
        if self.status not in {"scored", "abstained"}:
            raise ValueError("status must be scored or abstained")
        count = _positive_count_antigravity(self.test_count, "test_count")
        test_indices = _test_indices_antigravity(self.test_indices, count)
        raw_scores = _finite_scores_antigravity(self.raw_scores, "raw_scores")
        micro_scores = _finite_scores_antigravity(
            self.micro_scores, "micro_scores"
        )
        if self.status == "scored":
            if self.abstention_reason is not None:
                raise ValueError("scored shard must not contain an abstention reason")
            if len(raw_scores) != count or len(micro_scores) != count:
                raise ValueError("test_count differs from the score vector lengths")
        else:
            if self.abstention_reason != "insufficient_train_rows":
                raise ValueError(
                    "abstained shard reason must be insufficient_train_rows"
                )
            if raw_scores or micro_scores:
                raise ValueError("abstained shard score vectors must be empty")
        object.__setattr__(self, "test_indices", test_indices)
        object.__setattr__(self, "raw_scores", raw_scores)
        object.__setattr__(self, "micro_scores", micro_scores)


@dataclass(frozen=True, slots=True)
class Exp471ScoreCacheManifestAntigravity:
    source_authority_sha256: str
    cache_context_sha256: str
    method_order: tuple[str, ...]
    dataset_order: tuple[str, ...]
    shard_sha256: Mapping[str, str]
    total_shards: int
    scored_shards: int
    abstained_shards: int
    failed_shards: int
    manifest_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "source_authority_sha256",
            "cache_context_sha256",
            "manifest_sha256",
        ):
            _sha256_antigravity(getattr(self, name), name)
        methods = _fixed_method_order_antigravity(self.method_order)
        datasets = _dataset_order_antigravity(self.dataset_order)
        shards = _frozen_digest_mapping_antigravity(
            self.shard_sha256, "shard_sha256"
        )
        total_shards = _positive_count_antigravity(
            self.total_shards, "total_shards"
        )
        expected_total = len(methods) * len(datasets)
        if total_shards != expected_total or len(shards) != expected_total:
            raise ValueError(
                "total_shards must equal complete unique method/dataset coverage"
            )
        scored_shards = _nonnegative_count_antigravity(
            self.scored_shards, "scored_shards"
        )
        abstained_shards = _nonnegative_count_antigravity(
            self.abstained_shards, "abstained_shards"
        )
        failed_shards = _nonnegative_count_antigravity(
            self.failed_shards, "failed_shards"
        )
        if scored_shards + abstained_shards + failed_shards != total_shards:
            raise ValueError(
                "score-cache status partition must equal total_shards"
            )
        object.__setattr__(self, "method_order", methods)
        object.__setattr__(self, "dataset_order", datasets)
        object.__setattr__(self, "shard_sha256", shards)


@dataclass(frozen=True, slots=True)
class Exp471EvaluationArtifactAntigravity:
    source_authority_sha256: str
    cache_manifest_sha256: str
    code_closure_sha256: str
    method_order: tuple[str, ...]
    dataset_order: tuple[str, ...]
    payload: Mapping[str, object]
    evaluation_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "source_authority_sha256",
            "cache_manifest_sha256",
            "code_closure_sha256",
            "evaluation_sha256",
        ):
            _sha256_antigravity(getattr(self, name), name)
        methods = _fixed_method_order_antigravity(self.method_order)
        datasets = _dataset_order_antigravity(self.dataset_order)
        payload = _freeze_json_antigravity(self.payload, "payload")
        if not isinstance(payload, Mapping) or not payload:
            raise ValueError("payload must be a non-empty finite canonical JSON mapping")
        object.__setattr__(self, "method_order", methods)
        object.__setattr__(self, "dataset_order", datasets)
        object.__setattr__(self, "payload", payload)


@dataclass(frozen=True, slots=True)
class Exp471ReportAuthorityAntigravity:
    source_authority_sha256: str
    cache_manifest_sha256: str
    evaluation_sha256: str
    report_file_sha256: Mapping[str, str]
    report_manifest_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "source_authority_sha256",
            "cache_manifest_sha256",
            "evaluation_sha256",
            "report_manifest_sha256",
        ):
            _sha256_antigravity(getattr(self, name), name)
        report_files = _frozen_digest_mapping_antigravity(
            self.report_file_sha256,
            "report_file_sha256",
            paths=True,
        )
        object.__setattr__(self, "report_file_sha256", report_files)


__all__ = [
    "Exp471EvaluationArtifactAntigravity",
    "Exp471ReportAuthorityAntigravity",
    "Exp471ScoreCacheManifestAntigravity",
    "Exp471ScoreShardAntigravity",
    "Exp471SourceAuthorityAntigravity",
    "canonical_json_bytes_exp471_antigravity",
]
