"""Typed, label-free Exp451 score, backend, and migration contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType

from paper.equipment_anomaly_detection.contracts_paper import LowFrequencyResult, RawDataset
from paper.equipment_anomaly_detection.hard_route_paper import HardRouteResult
from paper.equipment_anomaly_detection.routing_paper import PolicyBundle
from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import HardSourceBundle
from paper.equipment_anomaly_detection.sources_paper.stability_paper import StabilityPartition


_APPROVED_CONDITIONS = {
    "A": ("linear", "linear"),
    "B": ("polyphase_pchip", "linear"),
    "C": ("linear", "area"),
    "D": ("polyphase_pchip", "area"),
}

_APPROVED_EXECUTION_ORIGINS = {
    "imported_exp450_mps",
    "exp451_native_cpu_pool_fallback",
}

_PRODUCTION_EXPECTED_MIGRATION_COUNTS = {
    "A": 1117,
    "B": 1117,
    "C": 1000,
    "D": 1000,
}


def _sha256_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_json_bytes(value: object) -> bytes:
    """Encode finite Exp451 artifact data in one stable JSON representation."""
    try:
        return json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError("Exp451 artifact data must be canonical finite JSON") from error


@dataclass(frozen=True, slots=True)
class Exp451Condition:
    condition_id: str
    length_mode: str
    image_mode: str

    def __post_init__(self) -> None:
        if _APPROVED_CONDITIONS.get(self.condition_id) != (
            self.length_mode,
            self.image_mode,
        ):
            raise ValueError("Exp451 condition differs from the approved factorial")


@dataclass(frozen=True, slots=True)
class Exp451BackendDecision:
    tensor_backend: str
    torch_version: str
    mps_available: bool
    mlx_available: bool
    logical_cpu_count: int
    cpu_workers: int
    inner_numerical_threads: int = 1

    def __post_init__(self) -> None:
        if self.tensor_backend != ("mps" if self.mps_available else "cpu"):
            raise ValueError("Exp451 tensor backend must use MPS when available")
        if type(self.torch_version) is not str or not self.torch_version:
            raise ValueError("Exp451 torch version is invalid")
        if type(self.mps_available) is not bool or type(self.mlx_available) is not bool:
            raise ValueError("Exp451 backend availability must be boolean")
        if type(self.logical_cpu_count) is not int or self.logical_cpu_count < 1:
            raise ValueError("Exp451 logical CPU count is invalid")
        if self.cpu_workers != max(1, self.logical_cpu_count - 1):
            raise ValueError("Exp451 CPU worker policy is invalid")
        if self.inner_numerical_threads != 1:
            raise ValueError("Exp451 inner numerical threads must remain one")

    def artifact_record(self) -> dict[str, object]:
        return {
            "tensor_backend": self.tensor_backend,
            "torch_version": self.torch_version,
            "mps_available": self.mps_available,
            "mlx_available": self.mlx_available,
            "logical_cpu_count": self.logical_cpu_count,
            "cpu_workers": self.cpu_workers,
            "inner_numerical_threads": self.inner_numerical_threads,
        }


@dataclass(frozen=True, slots=True)
class Exp451RunContext:
    """The complete root-bound identity required to run or reuse an Exp451 result."""

    mode: str
    run_root: Path
    database_sha256: str
    catalog_sha256: str
    code_closure: Mapping[str, str]
    backend: Exp451BackendDecision
    config_sha256: str

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "full"}:
            raise ValueError("Exp451 run mode is invalid")
        root = Path(self.run_root)
        if not root.is_absolute():
            raise ValueError("Exp451 run root must be absolute")
        for digest in (self.database_sha256, self.catalog_sha256, self.config_sha256):
            if not _sha256_digest(digest):
                raise ValueError("Exp451 run context digest is invalid")
        closure = dict(self.code_closure)
        if not closure or any(
            type(path) is not str or not path or not _sha256_digest(digest)
            for path, digest in closure.items()
        ):
            raise ValueError("Exp451 code closure is invalid")
        if not isinstance(self.backend, Exp451BackendDecision):
            raise ValueError("Exp451 run backend is invalid")
        object.__setattr__(self, "run_root", root)
        object.__setattr__(self, "code_closure", MappingProxyType(closure))

    def artifact_record(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "run_root": str(self.run_root),
            "database_sha256": self.database_sha256,
            "catalog_sha256": self.catalog_sha256,
            "code_closure": dict(sorted(self.code_closure.items())),
            "backend": self.backend.artifact_record(),
            "config_sha256": self.config_sha256,
        }

    @property
    def sha256(self) -> str:
        return sha256(canonical_json_bytes(self.artifact_record())).hexdigest()


@dataclass(frozen=True, slots=True)
class Exp451DatasetConditionResult:
    dataset: str
    condition_id: str
    sources: HardSourceBundle
    route: HardRouteResult


@dataclass(frozen=True, slots=True)
class Exp451ConditionScore:
    condition: Exp451Condition
    raw: Mapping[str, RawDataset]
    sources: Mapping[str, HardSourceBundle]
    routes: Mapping[str, HardRouteResult]
    low_frequency: LowFrequencyResult
    stability: StabilityPartition
    policies: PolicyBundle
    backend: Exp451BackendDecision
    stage_seconds: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class Exp451MigrationEntry:
    dataset: str
    condition_id: str
    source_relative_path: str
    source_envelope_sha256: str
    payload_sha256: str
    target_relative_path: str
    target_envelope_sha256: str
    execution_origin: str = "imported_exp450_mps"

    def __post_init__(self) -> None:
        if type(self.dataset) is not str or not self.dataset:
            raise ValueError("Exp451 migration entry dataset is invalid")
        if self.condition_id not in _APPROVED_CONDITIONS:
            raise ValueError("Exp451 migration entry condition_id is invalid")
        if type(self.source_relative_path) is not str or not self.source_relative_path:
            raise ValueError("Exp451 migration entry source_relative_path is invalid")
        if not _sha256_digest(self.source_envelope_sha256):
            raise ValueError("Exp451 migration entry source_envelope_sha256 digest is invalid")
        if not _sha256_digest(self.payload_sha256):
            raise ValueError("Exp451 migration entry payload_sha256 digest is invalid")
        if type(self.target_relative_path) is not str or not self.target_relative_path:
            raise ValueError("Exp451 migration entry target_relative_path is invalid")
        if not _sha256_digest(self.target_envelope_sha256):
            raise ValueError("Exp451 migration entry target_envelope_sha256 digest is invalid")
        if (
            type(self.execution_origin) is not str
            or self.execution_origin not in _APPROVED_EXECUTION_ORIGINS
        ):
            raise ValueError("Exp451 migration entry execution_origin is invalid")

    def artifact_record(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "condition_id": self.condition_id,
            "source_relative_path": self.source_relative_path,
            "source_envelope_sha256": self.source_envelope_sha256,
            "payload_sha256": self.payload_sha256,
            "target_relative_path": self.target_relative_path,
            "target_envelope_sha256": self.target_envelope_sha256,
            "execution_origin": self.execution_origin,
        }


@dataclass(frozen=True, slots=True)
class Exp451MigrationManifest:
    source_context_sha256: str
    entries: tuple[Exp451MigrationEntry, ...] = ()
    target_context_sha256: str = ""
    _expected_counts: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        if not _sha256_digest(self.source_context_sha256):
            raise ValueError("Exp451 migration manifest source_context_sha256 digest is invalid")
        if self.target_context_sha256 and not _sha256_digest(self.target_context_sha256):
            raise ValueError("Exp451 migration manifest target_context_sha256 digest is invalid")
        entries = tuple(self.entries)
        coordinates: set[tuple[str, str]] = set()
        counts: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}
        for entry in entries:
            if not isinstance(entry, Exp451MigrationEntry):
                raise ValueError("Exp451 migration manifest entries must be Exp451MigrationEntry instances")
            coordinate = (entry.dataset, entry.condition_id)
            if coordinate in coordinates:
                raise ValueError(f"Exp451 migration entries must have unique coordinates: duplicate {coordinate}")
            coordinates.add(coordinate)
            counts[entry.condition_id] += 1

        expected = (
            dict(self._expected_counts)
            if self._expected_counts is not None
            else _PRODUCTION_EXPECTED_MIGRATION_COUNTS
        )
        if counts != expected:
            raise ValueError(
                f"Exp451 migration manifest counts {counts} do not match expected {expected}"
            )

        object.__setattr__(self, "entries", entries)

    @property
    def total_count(self) -> int:
        return len(self.entries)

    @property
    def counts_by_condition(self) -> dict[str, int]:
        counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        for entry in self.entries:
            counts[entry.condition_id] += 1
        return counts

    @property
    def entry_map(self) -> dict[tuple[str, str], Exp451MigrationEntry]:
        return {(entry.dataset, entry.condition_id): entry for entry in self.entries}

    def artifact_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "source_context_sha256": self.source_context_sha256,
            "total_entries": len(self.entries),
            "counts_by_condition": self.counts_by_condition,
            "entries": [entry.artifact_record() for entry in self.entries],
        }
        if self.target_context_sha256:
            record["target_context_sha256"] = self.target_context_sha256
        return record

    @property
    def sha256(self) -> str:
        return sha256(canonical_json_bytes(self.artifact_record())).hexdigest()


__all__ = [
    "Exp451BackendDecision",
    "Exp451Condition",
    "Exp451ConditionScore",
    "Exp451DatasetConditionResult",
    "Exp451MigrationEntry",
    "Exp451MigrationManifest",
    "Exp451RunContext",
    "canonical_json_bytes",
]
