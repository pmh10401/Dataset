"""Typed, label-free Exp467 score, backend, and migration contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType

from paper.equipment_anomaly_detection.contracts_paper import LowFrequencyResult, RawDataset
from paper.equipment_anomaly_detection.routing_paper import PolicyBundle
from paper.equipment_anomaly_detection.sources_paper.stability_paper import StabilityPartition

from .hard_route_adapter_antigravity import HardRouteResult
from .hard_sources_adapter_antigravity import HardSourceBundle


_APPROVED_CONDITIONS = {
    "A": ("linear", "linear"),
    "B": ("polyphase_pchip", "linear"),
    "C": ("linear", "area"),
    "D": ("polyphase_pchip", "area"),
}

_APPROVED_MIGRATION_EXECUTION_ORIGIN = "imported_exp450_mps"

_PRODUCTION_EXPECTED_MIGRATION_COUNTS = {
    "A": 1117,
    "B": 1117,
    "C": 1000,
    "D": 1000,
}

_MLX_ASSESSMENT_KEYS_ANTIGRAVITY = {
    "schema",
    "tensor_eligibility",
    "exact_frozen_algorithm_expressible",
    "exact_expression_status",
    "exact_expression_reason",
    "mlx_available",
    "mlx_version",
    "mlx_metal_available",
    "mlx_metal_version",
    "mlx_device",
    "transfer_compilation_overhead_status",
    "transfer_compilation_overhead_result",
    "parity_requirement",
    "parity_status",
    "parity_result",
    "speed_benchmark_requirement",
    "speed_benchmark_status",
    "speed_benchmark_result",
    "canonical_cpu_mps_fallback",
    "fallback_reason",
}

_NUMBA_CALIBRATION_KEYS_ANTIGRAVITY = {
    "schema",
    "selected_backend",
    "fast_path_dtype",
    "fastmath",
    "parallel",
    "nogil",
    "python",
    "python_implementation",
    "python_executable",
    "numpy",
    "numba",
    "llvmlite",
    "machine",
    "processor",
}


def _numba_calibration_record_antigravity() -> dict[str, object]:
    from paper.equipment_anomaly_detection.numba_calibration_antigravity import (
        numba_calibration_backend_antigravity,
    )

    return numba_calibration_backend_antigravity()


def _unavailable_mlx_assessment_antigravity() -> dict[str, object]:
    """Return a complete fixture/default assessment with no invented results."""
    return {
        "schema": "exp467-mlx-assessment-antigravity.v1",
        "tensor_eligibility": "partial_tensor_substages_only",
        "exact_frozen_algorithm_expressible": False,
        "exact_expression_status": "gate_failed",
        "exact_expression_reason": "frozen_scipy_sklearn_aeon_and_pytorch_score_path_has_no_validated_exact_mlx_expression",
        "mlx_available": False,
        "mlx_version": "unavailable",
        "mlx_metal_available": False,
        "mlx_metal_version": "unavailable",
        "mlx_device": "unavailable",
        "transfer_compilation_overhead_status": "not_run_gate_1_failed",
        "transfer_compilation_overhead_result": "not_run_without_exact_mlx_expression",
        "parity_requirement": "same_input_and_seed_finite_outputs_identical_candidates_at_every_preregistered_alpha_scores_within_tolerance",
        "parity_status": "not_run_gate_1_failed",
        "parity_result": "not_run_without_exact_mlx_expression",
        "speed_benchmark_requirement": "mlx_at_least_1.1x_faster_including_transfer_compilation_and_synchronization",
        "speed_benchmark_status": "not_run_gate_1_failed",
        "speed_benchmark_result": "not_run_without_exact_mlx_expression",
        "canonical_cpu_mps_fallback": "frozen_scipy_sklearn_aeon_cpu_plus_pytorch_mps_with_native_cpu_pool_fallback",
        "fallback_reason": "mlx_gate_1_failed_exact_frozen_algorithm_not_expressible_and_validated",
    }


def _sha256_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_safe_relative_path(value: object) -> bool:
    if type(value) is not str or not value.strip():
        return False
    if value.startswith(("/", "\\")):
        return False
    parts = value.replace("\\", "/").split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return False
    path = Path(value)
    return not path.is_absolute()


def canonical_json_bytes(value: object) -> bytes:
    """Encode finite Exp467 artifact data in one stable JSON representation."""
    try:
        return json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError("Exp467 artifact data must be canonical finite JSON") from error


@dataclass(frozen=True, slots=True)
class Exp467Condition:
    condition_id: str
    length_mode: str
    image_mode: str

    def __post_init__(self) -> None:
        if _APPROVED_CONDITIONS.get(self.condition_id) != (
            self.length_mode,
            self.image_mode,
        ):
            raise ValueError("Exp467 condition differs from the approved factorial")


@dataclass(frozen=True, slots=True)
class Exp467BackendDecision:
    tensor_backend: str
    torch_version: str
    mps_available: bool
    mlx_available: bool
    logical_cpu_count: int
    cpu_workers: int
    inner_numerical_threads: int = 1
    tensor_eligible: bool = True
    transfer_compilation_overhead: str = "bounded_device_transfer_and_cpu_pool_fallback"
    deterministic_numerical_parity: str = (
        "finite_scores_identical_candidate_indices_within_tolerance"
    )
    canonical_cpu_fallback: str = "pytorch_cpu_native_adaptive_avg_pool"
    benchmark_requirement: str = "mlx_or_mps_speedup_ge_1_1x_with_device_transfer"
    mlx_assessment: Mapping[str, object] = field(
        default_factory=_unavailable_mlx_assessment_antigravity
    )
    numba_calibration: Mapping[str, object] = field(
        default_factory=_numba_calibration_record_antigravity
    )

    def __post_init__(self) -> None:
        if self.tensor_backend != ("mps" if self.mps_available else "cpu"):
            raise ValueError("Exp467 tensor backend must use MPS when available")
        if type(self.torch_version) is not str or not self.torch_version:
            raise ValueError("Exp467 torch version is invalid")
        if type(self.mps_available) is not bool or type(self.mlx_available) is not bool:
            raise ValueError("Exp467 backend availability must be boolean")
        if type(self.logical_cpu_count) is not int or self.logical_cpu_count < 1:
            raise ValueError("Exp467 logical CPU count is invalid")
        if self.cpu_workers != max(1, self.logical_cpu_count - 1):
            raise ValueError("Exp467 CPU worker policy is invalid")
        if self.inner_numerical_threads != 1:
            raise ValueError("Exp467 inner numerical threads must remain one")
        if type(self.tensor_eligible) is not bool:
            raise ValueError("Exp467 tensor eligibility must be boolean")
        if not self.transfer_compilation_overhead or type(self.transfer_compilation_overhead) is not str:
            raise ValueError("Exp467 transfer compilation overhead must be a non-empty string")
        if not self.deterministic_numerical_parity or type(self.deterministic_numerical_parity) is not str:
            raise ValueError("Exp467 deterministic numerical parity must be a non-empty string")
        if not self.canonical_cpu_fallback or type(self.canonical_cpu_fallback) is not str:
            raise ValueError("Exp467 canonical CPU fallback must be a non-empty string")
        if not self.benchmark_requirement or type(self.benchmark_requirement) is not str:
            raise ValueError("Exp467 benchmark requirement must be a non-empty string")
        assessment = dict(self.mlx_assessment)
        if set(assessment) != _MLX_ASSESSMENT_KEYS_ANTIGRAVITY:
            raise ValueError("Exp467 MLX assessment is incomplete")
        if (
            assessment["schema"] != "exp467-mlx-assessment-antigravity.v1"
            or assessment["tensor_eligibility"] != "partial_tensor_substages_only"
            or assessment["exact_frozen_algorithm_expressible"] is not False
            or assessment["exact_expression_status"] != "gate_failed"
            or assessment["mlx_available"] is not self.mlx_available
            or type(assessment["mlx_metal_available"]) is not bool
            or assessment["transfer_compilation_overhead_status"] != "not_run_gate_1_failed"
            or assessment["transfer_compilation_overhead_result"] != "not_run_without_exact_mlx_expression"
            or assessment["parity_status"] != "not_run_gate_1_failed"
            or assessment["parity_result"] != "not_run_without_exact_mlx_expression"
            or assessment["speed_benchmark_status"] != "not_run_gate_1_failed"
            or assessment["speed_benchmark_result"] != "not_run_without_exact_mlx_expression"
        ):
            raise ValueError("Exp467 MLX assessment gate result differs")
        string_fields = _MLX_ASSESSMENT_KEYS_ANTIGRAVITY - {
            "exact_frozen_algorithm_expressible", "mlx_available", "mlx_metal_available"
        }
        if any(
            type(assessment[name]) is not str or not assessment[name]
            for name in string_fields
        ):
            raise ValueError("Exp467 MLX assessment text is incomplete")
        if self.mlx_available and (
            assessment["mlx_version"] == "unavailable"
            or assessment["mlx_device"] == "unavailable"
        ):
            raise ValueError("Exp467 available MLX assessment lacks version or device")
        if assessment["mlx_metal_available"] and (
            assessment["mlx_metal_version"] == "unavailable"
            or assessment["mlx_device"] == "unavailable"
        ):
            raise ValueError("Exp467 available mlx-metal assessment lacks version or device")
        numba_record = dict(self.numba_calibration)
        if set(numba_record) != _NUMBA_CALIBRATION_KEYS_ANTIGRAVITY:
            raise ValueError("Exp467 Numba calibration provenance is incomplete")
        if (
            numba_record["schema"]
            != "exp467-numba-calibration-backend-antigravity.v1"
            or numba_record["selected_backend"] != "numba"
            or numba_record["fast_path_dtype"] != "float64"
            or numba_record["fastmath"] is not False
            or numba_record["parallel"] is not False
            or numba_record["nogil"] is not True
        ):
            raise ValueError("Exp467 Numba calibration contract differs")
        text_fields = _NUMBA_CALIBRATION_KEYS_ANTIGRAVITY - {
            "fastmath", "parallel", "nogil"
        }
        if any(
            type(numba_record[name]) is not str or not numba_record[name]
            for name in text_fields
        ):
            raise ValueError("Exp467 Numba calibration provenance text is incomplete")
        object.__setattr__(self, "mlx_assessment", MappingProxyType(assessment))
        object.__setattr__(
            self, "numba_calibration", MappingProxyType(numba_record)
        )

    @property
    def tensor_eligibility(self) -> bool:
        return self.tensor_eligible

    @property
    def parity_numerical_requirements(self) -> str:
        return self.deterministic_numerical_parity

    @property
    def benchmark_required(self) -> str:
        return self.benchmark_requirement

    def artifact_record(self) -> dict[str, object]:
        return {
            "tensor_backend": self.tensor_backend,
            "torch_version": self.torch_version,
            "mps_available": self.mps_available,
            "mlx_available": self.mlx_available,
            "logical_cpu_count": self.logical_cpu_count,
            "cpu_workers": self.cpu_workers,
            "inner_numerical_threads": self.inner_numerical_threads,
            "tensor_eligible": self.tensor_eligible,
            "transfer_compilation_overhead": self.transfer_compilation_overhead,
            "deterministic_numerical_parity": self.deterministic_numerical_parity,
            "canonical_cpu_fallback": self.canonical_cpu_fallback,
            "benchmark_requirement": self.benchmark_requirement,
            "mlx_assessment": dict(self.mlx_assessment),
            "numba_calibration": dict(self.numba_calibration),
        }


@dataclass(frozen=True, slots=True)
class Exp467RunContext:
    """The complete root-bound identity required to run or reuse an Exp467 result."""

    mode: str
    run_root: Path
    database_sha256: str
    catalog_sha256: str
    code_closure: Mapping[str, str]
    backend: Exp467BackendDecision
    config_sha256: str

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "full"}:
            raise ValueError("Exp467 run mode is invalid")
        root = Path(self.run_root)
        if not root.is_absolute():
            raise ValueError("Exp467 run root must be absolute")
        for digest in (self.database_sha256, self.catalog_sha256, self.config_sha256):
            if not _sha256_digest(digest):
                raise ValueError("Exp467 run context digest is invalid")
        closure = dict(self.code_closure)
        if not closure or any(
            type(path) is not str or not path or not _sha256_digest(digest)
            for path, digest in closure.items()
        ):
            raise ValueError("Exp467 code closure is invalid")
        if not isinstance(self.backend, Exp467BackendDecision):
            raise ValueError("Exp467 run backend is invalid")
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
class Exp467DatasetConditionResult:
    dataset: str
    condition_id: str
    sources: HardSourceBundle
    route: HardRouteResult


@dataclass(frozen=True, slots=True)
class Exp467ConditionScore:
    condition: Exp467Condition
    raw: Mapping[str, RawDataset]
    sources: Mapping[str, HardSourceBundle]
    routes: Mapping[str, HardRouteResult]
    low_frequency: LowFrequencyResult
    stability: StabilityPartition
    policies: PolicyBundle
    backend: Exp467BackendDecision
    stage_seconds: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class Exp467MigrationEntry:
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
            raise ValueError("Exp467 migration entry dataset is invalid")
        if self.condition_id not in _APPROVED_CONDITIONS:
            raise ValueError("Exp467 migration entry condition_id is invalid")
        if not _is_safe_relative_path(self.source_relative_path):
            raise ValueError(
                "Exp467 migration entry source_relative_path must be a safe non-empty relative path without dot or traversal components"
            )
        if not _sha256_digest(self.source_envelope_sha256):
            raise ValueError("Exp467 migration entry source_envelope_sha256 digest is invalid")
        if not _sha256_digest(self.payload_sha256):
            raise ValueError("Exp467 migration entry payload_sha256 digest is invalid")
        if not _is_safe_relative_path(self.target_relative_path):
            raise ValueError(
                "Exp467 migration entry target_relative_path must be a safe non-empty relative path without dot or traversal components"
            )
        if not _sha256_digest(self.target_envelope_sha256):
            raise ValueError("Exp467 migration entry target_envelope_sha256 digest is invalid")
        if self.execution_origin != _APPROVED_MIGRATION_EXECUTION_ORIGIN:
            raise ValueError(
                f"Exp467 migration entry execution_origin must be exactly {_APPROVED_MIGRATION_EXECUTION_ORIGIN!r}"
            )

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
class Exp467MigrationManifest:
    source_context_sha256: str
    entries: tuple[Exp467MigrationEntry, ...] = ()
    target_context_sha256: str = ""

    def __post_init__(self) -> None:
        if not _sha256_digest(self.source_context_sha256):
            raise ValueError("Exp467 migration manifest source_context_sha256 digest is invalid")
        if self.target_context_sha256 and not _sha256_digest(self.target_context_sha256):
            raise ValueError("Exp467 migration manifest target_context_sha256 digest is invalid")
        entries = tuple(self.entries)
        coordinates: set[tuple[str, str]] = set()
        counts: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0}
        for entry in entries:
            if not isinstance(entry, Exp467MigrationEntry):
                raise ValueError("Exp467 migration manifest entries must be Exp467MigrationEntry instances")
            coordinate = (entry.dataset, entry.condition_id)
            if coordinate in coordinates:
                raise ValueError(f"Exp467 migration entries must have unique coordinates: duplicate {coordinate}")
            coordinates.add(coordinate)
            counts[entry.condition_id] += 1

        if counts != _PRODUCTION_EXPECTED_MIGRATION_COUNTS:
            raise ValueError(
                f"Exp467 migration manifest counts {counts} do not match required {_PRODUCTION_EXPECTED_MIGRATION_COUNTS}"
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
    def entry_map(self) -> dict[tuple[str, str], Exp467MigrationEntry]:
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
    "Exp467BackendDecision",
    "Exp467Condition",
    "Exp467ConditionScore",
    "Exp467DatasetConditionResult",
    "Exp467MigrationEntry",
    "Exp467MigrationManifest",
    "Exp467RunContext",
    "canonical_json_bytes",
]
