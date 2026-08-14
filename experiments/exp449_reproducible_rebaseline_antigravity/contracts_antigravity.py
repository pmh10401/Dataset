"""Deeply immutable, fail-closed provenance contracts for Exp449."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Any

from paper.equipment_anomaly_detection.contracts_paper import FrozenTask8BackendDecision

from .config_antigravity import FROZEN_EXP449_CONFIG


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _nonnegative_count(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _finite_float(value: object, name: str) -> float:
    if type(value) not in (int, float) or not isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _freeze_json(value: object, name: str = "value") -> object:
    """Copy a strict JSON-shaped value into recursively immutable containers."""

    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not isfinite(value):
            raise ValueError(f"{name} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        for key, nested in value.items():
            if type(key) is not str:
                raise ValueError(f"{name} mapping keys must be strings")
            copied[key] = _freeze_json(nested, f"{name}.{key}")
        return MappingProxyType(dict(sorted(copied.items())))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(nested, name) for nested in value)
    raise ValueError(f"{name} must contain only canonical JSON values")


def _json_plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _json_plain(nested) for key, nested in value.items()}
    if isinstance(value, tuple):
        return [_json_plain(nested) for nested in value]
    return value


def _canonical_sha256(value: object, name: str) -> str:
    frozen = _freeze_json(value, name)
    encoded = json.dumps(
        _json_plain(frozen),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _frozen_mapping_of_sha256(value: object, name: str, *, nonempty: bool = True) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    copied = {key: _sha256(nested, f"{name}.{key}") for key, nested in value.items() if type(key) is str}
    if len(copied) != len(value) or (nonempty and not copied):
        raise ValueError(f"{name} must be a nonempty string-to-SHA-256 mapping")
    return MappingProxyType(dict(sorted(copied.items())))


@dataclass(frozen=True, slots=True)
class Exp449Context:
    """Identity and verified inputs that bind every Exp449 publication."""

    experiment_id: str
    mode: str
    run_root: Path
    database_sha256: str
    catalog_sha256: str
    code_closure: Mapping[str, str]
    lineage_sha256: str
    codemap_sha256: str
    interpreter: str
    environment: Mapping[str, object]
    backend_record: Mapping[str, object]
    seeds_and_constants: Mapping[str, object]
    historical_parity_status: str

    def __post_init__(self) -> None:
        if self.experiment_id != "Exp449" or self.mode not in {"smoke", "official"}:
            raise ValueError("Exp449 context identity is invalid")
        root = Path(self.run_root).resolve(strict=False)
        if not root.is_absolute():
            raise ValueError("Exp449 run root must resolve to an absolute path")
        if type(self.interpreter) is not str or not self.interpreter:
            raise ValueError("Exp449 interpreter identity is invalid")
        if self.historical_parity_status != "not_used":
            raise ValueError("Exp449 historical parity status must be not_used")
        object.__setattr__(self, "run_root", root)
        for field_name in ("database_sha256", "catalog_sha256", "lineage_sha256", "codemap_sha256"):
            object.__setattr__(self, field_name, _sha256(getattr(self, field_name), field_name))
        object.__setattr__(self, "code_closure", _frozen_mapping_of_sha256(self.code_closure, "code_closure"))
        environment = _freeze_json(self.environment, "environment")
        constants = _freeze_json(self.seeds_and_constants, "seeds_and_constants")
        if not isinstance(environment, Mapping) or not environment:
            raise ValueError("Exp449 environment must be a nonempty mapping")
        if not isinstance(constants, Mapping) or not constants:
            raise ValueError("Exp449 seeds and constants must be a nonempty mapping")
        if not isinstance(self.backend_record, Mapping):
            raise ValueError("Exp449 backend record must be a mapping")
        try:
            backend = FrozenTask8BackendDecision.from_artifact_record(dict(self.backend_record))
        except (TypeError, ValueError) as error:
            raise ValueError("Exp449 backend record is invalid") from error
        object.__setattr__(self, "environment", environment)
        object.__setattr__(self, "backend_record", _freeze_json(backend.artifact_record(), "backend_record"))
        object.__setattr__(self, "seeds_and_constants", constants)

    @classmethod
    def from_verified_inputs(
        cls,
        *,
        run_root: Path,
        mode: str,
        catalog: Mapping[str, object],
        acceleration: FrozenTask8BackendDecision | object,
        acceptance_path: Path | None = None,
        database_sha256: str | None = None,
        code_closure: Mapping[str, str] | None = None,
        lineage_sha256: str | None = None,
        codemap_sha256: str | None = None,
        interpreter: str | None = None,
        environment: Mapping[str, object] | None = None,
        seeds_and_constants: Mapping[str, object] | None = None,
    ) -> "Exp449Context":
        """Construct a context only from verified, non-historical inputs.

        The historical-authority refusal intentionally occurs before validation or
        inspection of every other input.
        """

        if acceptance_path is not None:
            raise ValueError("Exp449 forbids historical authority")
        if not isinstance(catalog, Mapping) or not catalog:
            raise ValueError("Exp449 catalog is invalid")
        if not isinstance(acceleration, FrozenTask8BackendDecision):
            raise ValueError("Exp449 acceleration must be a FrozenTask8BackendDecision")
        required = {
            "database_sha256": database_sha256,
            "code_closure": code_closure,
            "lineage_sha256": lineage_sha256,
            "codemap_sha256": codemap_sha256,
            "interpreter": interpreter,
            "environment": environment,
            "seeds_and_constants": seeds_and_constants,
        }
        if any(value is None for value in required.values()):
            raise ValueError("Exp449 verified inputs are incomplete")
        return cls(
            experiment_id=FROZEN_EXP449_CONFIG.experiment_id,
            mode=mode,
            run_root=Path(run_root).resolve(strict=False),
            database_sha256=database_sha256,
            catalog_sha256=_canonical_sha256(catalog, "catalog"),
            code_closure=code_closure,
            lineage_sha256=lineage_sha256,
            codemap_sha256=codemap_sha256,
            interpreter=interpreter,
            environment=environment,
            backend_record=acceleration.artifact_record(),
            seeds_and_constants=seeds_and_constants,
            historical_parity_status="not_used",
        )


@dataclass(frozen=True, slots=True)
class Exp449ShardDescriptor:
    """One create-only Exp449 typed-shard identity."""

    path: str
    size: int
    sha256: str
    schema: str

    def __post_init__(self) -> None:
        if type(self.path) is not str or not self.path or "antigravity" not in self.path:
            raise ValueError("Exp449 shard path is invalid")
        relative = PurePosixPath(self.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Exp449 shard path must be a confined relative path")
        if type(self.schema) is not str or not self.schema.startswith("exp449_"):
            raise ValueError("Exp449 shard schema is invalid")
        object.__setattr__(self, "size", _nonnegative_count(self.size, "shard size"))
        object.__setattr__(self, "sha256", _sha256(self.sha256, "shard sha256"))


@dataclass(frozen=True, slots=True)
class Exp449ProposalManifest:
    """The typed-shard inventory and proposal-digest closure."""

    schema_version: str
    context_sha256: str
    catalog_sha256: str
    shards: Mapping[str, Exp449ShardDescriptor]
    policy_coordinate_digests: Mapping[str, str]
    candidate_c_score_digests: Mapping[str, str]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not str or not self.schema_version.startswith("exp449_"):
            raise ValueError("Exp449 manifest schema is invalid")
        object.__setattr__(self, "context_sha256", _sha256(self.context_sha256, "context sha256"))
        object.__setattr__(self, "catalog_sha256", _sha256(self.catalog_sha256, "catalog sha256"))
        if not isinstance(self.shards, Mapping) or not self.shards:
            raise ValueError("Exp449 shards must be a nonempty mapping")
        descriptors: dict[str, Exp449ShardDescriptor] = {}
        for dataset, descriptor in self.shards.items():
            if type(dataset) is not str or not dataset or not isinstance(descriptor, Exp449ShardDescriptor):
                raise ValueError("Exp449 shard descriptor is invalid")
            descriptors[dataset] = descriptor
        object.__setattr__(self, "shards", MappingProxyType(dict(sorted(descriptors.items()))))
        object.__setattr__(
            self,
            "policy_coordinate_digests",
            _frozen_mapping_of_sha256(self.policy_coordinate_digests, "policy_coordinate_digests"),
        )
        object.__setattr__(
            self,
            "candidate_c_score_digests",
            _frozen_mapping_of_sha256(self.candidate_c_score_digests, "candidate_c_score_digests"),
        )


@dataclass(frozen=True, slots=True)
class Exp449TerminalStatus:
    """The only terminal outcomes allowed after at most two dataset-local waves."""

    status: str
    completed_dataset_count: int
    failed_dataset_count: int
    pending_dataset_count: int
    wave_count: int

    def __post_init__(self) -> None:
        if self.status not in {"dataset_local_complete", "proposal_score_incomplete"}:
            raise ValueError("Exp449 terminal status is invalid")
        for field_name in (
            "completed_dataset_count",
            "failed_dataset_count",
            "pending_dataset_count",
            "wave_count",
        ):
            object.__setattr__(self, field_name, _nonnegative_count(getattr(self, field_name), field_name))
        if self.wave_count > FROZEN_EXP449_CONFIG.maximum_waves:
            raise ValueError("Exp449 terminal status exceeds the maximum wave count")
        incomplete = self.failed_dataset_count > 0 or self.pending_dataset_count > 0
        if (self.status == "dataset_local_complete") == incomplete:
            raise ValueError("Exp449 terminal status does not match dataset completion")


@dataclass(frozen=True, slots=True)
class SmokeComparison:
    """The immutable result of comparing independently verified smoke roots."""

    smoke_a_manifest_sha256: str
    smoke_b_manifest_sha256: str
    compared_dataset_count: int
    coordinate_identity: bool
    maximum_score_delta: float
    passed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "smoke_a_manifest_sha256", _sha256(self.smoke_a_manifest_sha256, "smoke A manifest sha256"))
        object.__setattr__(self, "smoke_b_manifest_sha256", _sha256(self.smoke_b_manifest_sha256, "smoke B manifest sha256"))
        object.__setattr__(self, "compared_dataset_count", _nonnegative_count(self.compared_dataset_count, "compared dataset count"))
        if type(self.coordinate_identity) is not bool or type(self.passed) is not bool:
            raise ValueError("Exp449 smoke comparison flags are invalid")
        delta = _finite_float(self.maximum_score_delta, "maximum score delta")
        if delta < 0.0:
            raise ValueError("maximum score delta must be nonnegative")
        if self.passed != (
            self.coordinate_identity and delta <= FROZEN_EXP449_CONFIG.cpu_repeat_tolerance
        ):
            raise ValueError("Exp449 smoke comparison pass state is inconsistent")
        object.__setattr__(self, "maximum_score_delta", delta)
