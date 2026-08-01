"""Deterministic, database-free CPU/MPS core-forward parity evidence."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import struct
from time import perf_counter

import numpy as np
import torch

from experiments.exp417_causal_multihorizon_shadow.model import CausalMultiHorizonCNN
from experiments.exp417_causal_multihorizon_shadow.preprocessing import build_horizons

from .capability import DeviceCapability
from .config import Exp422Config
from .contracts import BatchKey, CoreParityReport, StageResult, TensorDelta, TimingEvidence


_LENGTHS = (64, 128, 256, 512)


@dataclass(frozen=True)
class CoreStageResult:
    """All required core rows plus one conservative stage decision."""

    capability: DeviceCapability
    reports: tuple[CoreParityReport, ...]
    stage: StageResult


def _sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<f4")
    return sha256(array.tobytes(order="C")).hexdigest()


def _ordered_float32(values: np.ndarray) -> np.ndarray:
    signed = np.ascontiguousarray(values, dtype=np.float32).view(np.int32).astype(np.int64)
    return np.where(signed < 0, 0x80000000 - signed, signed)


def compare_tensor(cpu: np.ndarray, mps: np.ndarray, *, name: str) -> TensorDelta:
    """Produce fail-closed float32 delta evidence for one named output tensor."""
    left, right = np.asarray(cpu, dtype=np.float32), np.asarray(mps, dtype=np.float32)
    left_flat, right_flat = left.reshape(-1), right.reshape(-1)
    same_shape = left.shape == right.shape
    finite = same_shape and bool(np.isfinite(left_flat).all() and np.isfinite(right_flat).all())
    if not finite:
        failing = 0
        if same_shape:
            invalid = np.flatnonzero(~np.isfinite(left_flat) | ~np.isfinite(right_flat))
            failing = int(invalid[0]) if invalid.size else 0
        return TensorDelta(name, False, 0.0, 0.0, 0.0, 0, _sha256(left), _sha256(right), failing)
    absolute = np.abs(left_flat.astype(np.float64) - right_flat.astype(np.float64))
    normalized = absolute / np.maximum(np.abs(left_flat.astype(np.float64)), 1e-12)
    ulp = np.abs(_ordered_float32(left_flat) - _ordered_float32(right_flat))
    failing_indices = np.flatnonzero(normalized > 1e-4)
    return TensorDelta(
        name,
        True,
        float(absolute.max(initial=0.0)),
        float(absolute.mean()) if absolute.size else 0.0,
        float(normalized.max(initial=0.0)),
        int(ulp.max(initial=0)),
        _sha256(left),
        _sha256(right),
        int(failing_indices[0]) if failing_indices.size else None,
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def tie_aware_spearman(cpu: np.ndarray, mps: np.ndarray) -> float:
    """Return a finite tie-aware Spearman coefficient, or zero when undefined."""
    left, right = np.asarray(cpu).reshape(-1), np.asarray(mps).reshape(-1)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        return 0.0
    if np.array_equal(left, right):
        return 1.0
    a, b = _rankdata(left), _rankdata(right)
    if a.size < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return float(np.clip(value, -1.0, 1.0)) if np.isfinite(value) else 0.0


def candidate_coordinate_digest(scores: np.ndarray, alpha: float) -> str:
    """Hash stable top-alpha score coordinates only; labels never enter this rule."""
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if not np.isfinite(values).all() or not isinstance(alpha, float) or not 0.0 < alpha <= 1.0:
        raise ValueError("candidate scores must be finite and alpha must be in (0, 1]")
    count = max(1, int(np.ceil(alpha * values.size))) if values.size else 0
    coordinates = np.argsort(values, kind="stable")[:count]
    digest = sha256(b"exp422.candidate.coordinates.v1\x00")
    digest.update(struct.pack("<d", alpha))
    for coordinate in coordinates:
        digest.update(int(coordinate).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def _synthetic_input(batch_size: int, length: int, seed: int, *, max_batch_bytes: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    values = generator.standard_normal((batch_size, 2, length), dtype=np.float32)
    if not np.isfinite(values).all():
        raise RuntimeError("synthetic input must be finite")
    if values.nbytes > max_batch_bytes:
        raise MemoryError("fixed_batch_memory_limit")
    return values


def _timed_forward(model: CausalMultiHorizonCNN, values: np.ndarray, device: torch.device) -> tuple[dict[int, np.ndarray], TimingEvidence]:
    transfer_start = perf_counter()
    inputs = torch.from_numpy(np.array(values, copy=True, order="C")).to(device)
    transfer_in = perf_counter() - transfer_start
    compute_start = perf_counter()
    with torch.no_grad():
        outputs = model(inputs)
    compute = perf_counter() - compute_start
    sync_start = perf_counter()
    if device.type == "mps":
        torch.mps.synchronize()
    synchronize = perf_counter() - sync_start
    output_start = perf_counter()
    packed = {horizon: output.detach().cpu().numpy().astype(np.float32, copy=False) for horizon, output in outputs.items()}
    transfer = transfer_in + (perf_counter() - output_start)
    if any(not np.isfinite(item).all() for item in packed.values()):
        raise ValueError("nonfinite_output")
    return packed, TimingEvidence(device.type, transfer, compute, synchronize, transfer + compute + synchronize)


def _unavailable_report(key: BatchKey, name: str, cpu_timing: TimingEvidence) -> CoreParityReport:
    unavailable = compare_tensor(np.array([0.0], dtype=np.float32), np.array([np.nan], dtype=np.float32), name=name)
    return CoreParityReport(key, (unavailable,), cpu_timing, TimingEvidence("mps", 0.0, 0.0, 0.0, 0.0), False, 0.0)


def run_core_parity(config: Exp422Config, capability: DeviceCapability) -> CoreStageResult:
    """Evaluate exact synthetic batches CPU-first, then MPS, recording every row."""
    if type(config) is not Exp422Config:
        raise ValueError("config must be an Exp422Config")
    config.validate()
    if not isinstance(capability, DeviceCapability):
        raise ValueError("capability must be a DeviceCapability")
    torch.set_num_threads(1)
    reports: list[CoreParityReport] = []
    failure_reason: str | None = None
    for batch_size in config.candidate_batch_sizes:
        for seed, perturbation_seed in zip(config.model_seeds, config.perturbation_seeds):
            for length in _LENGTHS:
                key = BatchKey(batch_size, seed, perturbation_seed)
                name = f"length_{length}_prediction"
                try:
                    values = _synthetic_input(
                        batch_size, length, seed, max_batch_bytes=config.max_batch_bytes
                    )
                    torch.manual_seed(seed)
                    cpu_model = CausalMultiHorizonCNN(build_horizons(length)).eval().to("cpu")
                    cpu_outputs, cpu_timing = _timed_forward(cpu_model, values, torch.device("cpu"))
                except MemoryError:
                    reports.append(_unavailable_report(key, name, TimingEvidence("cpu", 0.0, 0.0, 0.0, 0.0)))
                    failure_reason = "fixed_batch_memory_limit"
                    continue
                if not capability.mps_available:
                    reports.append(_unavailable_report(key, name, cpu_timing))
                    failure_reason = "mps_unavailable"
                    continue
                try:
                    mps_model = CausalMultiHorizonCNN(build_horizons(length)).eval().to("cpu")
                    mps_model.load_state_dict(cpu_model.state_dict())
                    mps_model.to("mps")
                    mps_outputs, mps_timing = _timed_forward(mps_model, values, torch.device("mps"))
                    deltas = tuple(compare_tensor(cpu_outputs[h], mps_outputs[h], name=f"length_{length}_horizon_{h}") for h in sorted(cpu_outputs))
                    cpu_flat = np.concatenate([cpu_outputs[h].reshape(-1) for h in sorted(cpu_outputs)])
                    mps_flat = np.concatenate([mps_outputs[h].reshape(-1) for h in sorted(mps_outputs)])
                    alpha = float(config.exp417_config.candidate_alpha)
                    coordinates_match = candidate_coordinate_digest(cpu_flat, alpha) == candidate_coordinate_digest(mps_flat, alpha)
                    report = CoreParityReport(key, deltas, cpu_timing, mps_timing, coordinates_match, tie_aware_spearman(cpu_flat, mps_flat))
                    reports.append(report)
                    if not report.passes:
                        failure_reason = "core_parity_gate_failed"
                except Exception as exc:
                    reports.append(_unavailable_report(key, name, cpu_timing))
                    failure_reason = "nonfinite_output" if "nonfinite" in str(exc).lower() else "mps_runtime_error"
    stage = StageResult.passed() if not failure_reason else StageResult.failed(failure_reason)
    return CoreStageResult(capability, tuple(reports), stage)


__all__ = ("CoreStageResult", "candidate_coordinate_digest", "compare_tensor", "run_core_parity", "tie_aware_spearman")
