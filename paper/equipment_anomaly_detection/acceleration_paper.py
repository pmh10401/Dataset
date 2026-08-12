"""Task 9 CPU/MLX/MPS evidence gate for the frozen proposal scorer."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Iterable

import numpy as np


ACCELERATION_SCHEMA = "paper-task9-acceleration.v1"


@dataclass(frozen=True, slots=True)
class WorkerRecord:
    logical_cpu_count: int
    dataset_workers: int
    deep_method_workers: int
    outer_parallelism: str

    @classmethod
    def from_logical_cpu_count(cls, logical_cpu_count: int | None = None) -> "WorkerRecord":
        count = os.cpu_count() if logical_cpu_count is None else logical_cpu_count
        if type(count) is not int or count < 1:
            raise ValueError("logical CPU count is invalid")
        return cls(count, min(max(1, count - 1), 7), 1, "sequential_until_safe_outer_worker_benchmark")


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _backends() -> list[dict[str, object]]:
    torch_version = _version("torch")
    torch_mps_available = False
    torch_mps_built = False
    if torch_version is not None:
        try:
            import torch
            torch_mps_available = bool(torch.backends.mps.is_available())
            torch_mps_built = bool(torch.backends.mps.is_built())
        except Exception:
            pass
    mlx_version = _version("mlx")
    mlx_device: str | None = None
    if mlx_version is not None:
        try:
            import mlx.core as mx
            mlx_device = str(mx.default_device())
        except Exception:
            pass
    return [
        {"family": "pytorch", "available": torch_version is not None, "version": torch_version, "device": "mps" if torch_mps_available else None},
        {"family": "mlx", "available": mlx_version is not None and mlx_device is not None, "version": mlx_version, "mlx_metal_version": _version("mlx-metal"), "device": mlx_device},
        {"family": "mps", "available": torch_mps_available, "version": torch_version, "built": torch_mps_built, "device": "mps" if torch_mps_available else None},
    ]


def current_runtime_identity() -> dict[str, str]:
    """Capture interpreter provenance in the exact process that observes backends."""
    return {
        "executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "python_build": ".".join(platform.python_build()),
        "python_implementation": platform.python_implementation(),
        "prefix": str(Path(sys.prefix).resolve()),
        "base_prefix": str(Path(sys.base_prefix).resolve()),
        "platform": platform.platform(),
    }


def _canonical_sha256(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n").hexdigest()


def decide_dense_mlx_gate(
    cpu_values: object,
    mlx_values: object,
    cpu_candidates: Iterable[int],
    mlx_candidates: Iterable[int],
    *,
    cpu_seconds: float,
    mlx_seconds: float,
    tolerance: float,
) -> dict[str, object]:
    """Apply the exact MLX parity/speed gate to a measured dense implementation."""
    cpu = np.asarray(cpu_values, dtype=np.float64)
    mlx = np.asarray(mlx_values, dtype=np.float64)
    if cpu.shape != mlx.shape or cpu.ndim != 1 or not np.isfinite(cpu).all() or not np.isfinite(mlx).all() or not np.isfinite(tolerance) or tolerance < 0.0:
        return {"selected_backend": "cpu", "reason": "nonfinite_or_shape_mismatch", "speedup": None}
    if tuple(cpu_candidates) != tuple(mlx_candidates):
        return {"selected_backend": "cpu", "reason": "candidate_indices_differ", "speedup": None}
    if not np.allclose(cpu, mlx, rtol=0.0, atol=float(tolerance)):
        return {"selected_backend": "cpu", "reason": "numerical_tolerance_exceeded", "speedup": None}
    if not np.isfinite(cpu_seconds) or not np.isfinite(mlx_seconds) or cpu_seconds <= 0.0 or mlx_seconds <= 0.0:
        return {"selected_backend": "cpu", "reason": "benchmark_timing_invalid", "speedup": None}
    speedup = float(cpu_seconds / mlx_seconds)
    if speedup < 1.1:
        return {"selected_backend": "cpu", "reason": "speedup_below_1_1x", "speedup": speedup}
    return {"selected_backend": "mlx", "reason": "parity_and_speed_gate_passed", "speedup": speedup}


def build_acceleration_manifest(context_sha256: str, *, logical_cpu_count: int | None = None) -> dict[str, object]:
    """Record the actual backend inventory and CPU decision without pretending an MLX rewrite exists."""
    if type(context_sha256) is not str or len(context_sha256) != 64:
        raise ValueError("acceleration context SHA256 is invalid")
    workers = WorkerRecord.from_logical_cpu_count(logical_cpu_count)
    stages = [
        {"stage": "hard_sources", "tensor_eligible": False, "decision": "cpu", "reason": "official_third_party_transformers_cpu_comparability"},
        {"stage": "minirocket_confirmation", "tensor_eligible": False, "decision": "cpu", "reason": "official_third_party_transformer_cpu_comparability"},
        {"stage": "hard_route", "tensor_eligible": False, "decision": "cpu", "reason": "frozen_rule_not_tensor_stage"},
        {"stage": "low_frequency", "tensor_eligible": True, "decision": "cpu", "reason": "implementation_unavailable"},
        {"stage": "stability", "tensor_eligible": True, "decision": "cpu", "reason": "implementation_unavailable"},
        {"stage": "proposal_routes", "tensor_eligible": True, "decision": "cpu", "reason": "implementation_unavailable"},
    ]
    base = {
        "schema": ACCELERATION_SCHEMA,
        "context_sha256": context_sha256,
        "runtime": current_runtime_identity(),
        "selected_backend": "cpu",
        "workers": asdict(workers),
        "backends": _backends(),
        "stages": stages,
        "benchmark_requirement": "same_input_seed_finite_outputs_identical_candidates_at_all_alphas_numeric_tolerance_and_at_least_1_1x_including_transfer_and_synchronization",
        "fallback_reason": "no_exact_mlx_dense_implementation_has_been_benchmarked",
    }
    return {**base, "assessment_sha256": _canonical_sha256(base)}


def verify_acceleration_manifest(value: object, context_sha256: str, *, current_runtime: dict[str, str] | None = None) -> dict[str, object]:
    required = {"schema", "context_sha256", "runtime", "selected_backend", "workers", "backends", "stages", "benchmark_requirement", "fallback_reason", "assessment_sha256"}
    if not isinstance(value, dict) or set(value) != required or value.get("schema") != ACCELERATION_SCHEMA or value.get("context_sha256") != context_sha256:
        raise ValueError("acceleration artifact is invalid")
    base = {key: child for key, child in value.items() if key != "assessment_sha256"}
    if not isinstance(value["assessment_sha256"], str) or value["assessment_sha256"] != _canonical_sha256(base):
        raise ValueError("acceleration artifact assessment hash differs")
    runtime = value["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != set(current_runtime_identity()) or any(type(item) is not str or not item for item in runtime.values()):
        raise ValueError("acceleration artifact runtime identity is invalid")
    observed = current_runtime_identity() if current_runtime is None else current_runtime
    if observed != runtime:
        raise ValueError("interpreter_identity_mismatch")
    return value


__all__ = ["ACCELERATION_SCHEMA", "WorkerRecord", "build_acceleration_manifest", "current_runtime_identity", "decide_dense_mlx_gate", "verify_acceleration_manifest"]
