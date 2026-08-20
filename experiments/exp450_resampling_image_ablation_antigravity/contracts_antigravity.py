"""Typed, label-free Exp450 score and backend contracts."""

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


def _sha256_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_json_bytes(value: object) -> bytes:
    """Encode finite Exp450 artifact data in one stable JSON representation."""
    try:
        return json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError("Exp450 artifact data must be canonical finite JSON") from error


@dataclass(frozen=True, slots=True)
class Exp450Condition:
    condition_id: str
    length_mode: str
    image_mode: str

    def __post_init__(self) -> None:
        if _APPROVED_CONDITIONS.get(self.condition_id) != (
            self.length_mode,
            self.image_mode,
        ):
            raise ValueError("Exp450 condition differs from the approved factorial")


@dataclass(frozen=True, slots=True)
class Exp450BackendDecision:
    tensor_backend: str
    torch_version: str
    mps_available: bool
    mlx_available: bool
    logical_cpu_count: int
    cpu_workers: int
    inner_numerical_threads: int = 1

    def __post_init__(self) -> None:
        if self.tensor_backend != ("mps" if self.mps_available else "cpu"):
            raise ValueError("Exp450 tensor backend must use MPS when available")
        if type(self.torch_version) is not str or not self.torch_version:
            raise ValueError("Exp450 torch version is invalid")
        if type(self.mps_available) is not bool or type(self.mlx_available) is not bool:
            raise ValueError("Exp450 backend availability must be boolean")
        if type(self.logical_cpu_count) is not int or self.logical_cpu_count < 1:
            raise ValueError("Exp450 logical CPU count is invalid")
        if self.cpu_workers != max(1, self.logical_cpu_count - 1):
            raise ValueError("Exp450 CPU worker policy is invalid")
        if self.inner_numerical_threads != 1:
            raise ValueError("Exp450 inner numerical threads must remain one")

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
class Exp450RunContext:
    """The complete root-bound identity required to reuse an Exp450 result."""

    mode: str
    run_root: Path
    database_sha256: str
    catalog_sha256: str
    code_closure: Mapping[str, str]
    backend: Exp450BackendDecision
    config_sha256: str

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "full"}:
            raise ValueError("Exp450 run mode is invalid")
        root = Path(self.run_root)
        if not root.is_absolute():
            raise ValueError("Exp450 run root must be absolute")
        for digest in (self.database_sha256, self.catalog_sha256, self.config_sha256):
            if not _sha256_digest(digest):
                raise ValueError("Exp450 run context digest is invalid")
        closure = dict(self.code_closure)
        if not closure or any(
            type(path) is not str or not path or not _sha256_digest(digest)
            for path, digest in closure.items()
        ):
            raise ValueError("Exp450 code closure is invalid")
        if not isinstance(self.backend, Exp450BackendDecision):
            raise ValueError("Exp450 run backend is invalid")
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
class Exp450DatasetConditionResult:
    dataset: str
    condition_id: str
    sources: HardSourceBundle
    route: HardRouteResult


@dataclass(frozen=True, slots=True)
class Exp450ConditionScore:
    condition: Exp450Condition
    raw: Mapping[str, RawDataset]
    sources: Mapping[str, HardSourceBundle]
    routes: Mapping[str, HardRouteResult]
    low_frequency: LowFrequencyResult
    stability: StabilityPartition
    policies: PolicyBundle
    backend: Exp450BackendDecision
    stage_seconds: Mapping[str, float]


__all__ = [
    "Exp450BackendDecision",
    "Exp450Condition",
    "Exp450ConditionScore",
    "Exp450DatasetConditionResult",
    "Exp450RunContext",
    "canonical_json_bytes",
]
