"""Optional PyTorch/MPS implementation of Exp436's scalar dense RMS score."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import platform
from statistics import median
from time import perf_counter
from typing import Callable, Sequence

import numpy as np

from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import BackendBatchEvidence, BackendBenchmarkEvidence
from .features_antigravity import score_channel_cpu


class MpsUnavailable(RuntimeError):
    def __init__(self, reason: str, **provenance: object) -> None:
        self.reason, self.provenance = reason, provenance
        super().__init__(reason)


def _vector(value: object, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1 or array.size < 1 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be one finite float32 vector")
    return np.ascontiguousarray(array, dtype=np.float32)


def _matrix(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1] or not np.isfinite(array).all():
        raise ValueError("references must be a non-empty finite float32 matrix")
    return np.ascontiguousarray(array, dtype=np.float32)


def _is_mps_device(value: object) -> bool:
    """Return true only for MPS/Metal provenance, never a CPU fallback."""

    device = str(value).strip().lower()
    return bool(device) and "cpu" not in device and ("mps" in device or "metal" in device)


@dataclass(frozen=True)
class MpsFixture:
    query: np.ndarray
    references: np.ndarray
    threshold: float
    k: int = FROZEN_CONFIG.neighbor_count

    def __post_init__(self) -> None:
        query, references = _vector(self.query, "query"), _matrix(self.references)
        if query.shape[0] != references.shape[1] or references.shape[0] < FROZEN_CONFIG.neighbor_count:
            raise ValueError("scalar fixture requires compatible values and frozen reference count")
        if type(self.k) is not int or self.k != FROZEN_CONFIG.neighbor_count:
            raise ValueError("k must equal the frozen neighbor count")
        threshold_value = np.asarray(self.threshold, dtype="<f4")
        if threshold_value.ndim != 0 or not np.isfinite(threshold_value):
            raise ValueError("threshold is required and must be finite")
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "references", references)
        object.__setattr__(self, "threshold", float(threshold_value))


def _digest_fixtures(fixtures: Sequence[MpsFixture]) -> str:
    digest = sha256()
    for item in fixtures:
        for array in (item.query, item.references):
            little = np.asarray(array, dtype="<f4", order="C")
            digest.update(str(little.shape).encode("ascii")); digest.update(b"\0"); digest.update(little.tobytes()); digest.update(b"\0")
        threshold=np.asarray(item.threshold,dtype="<f4")
        digest.update(str(threshold.shape).encode("ascii")); digest.update(b"\0"); digest.update(threshold.tobytes()); digest.update(b"\0"); digest.update(str(item.k).encode("ascii")); digest.update(b"\0")
    return digest.hexdigest()


def _cpu_score_suite(fixtures: Sequence[MpsFixture]) -> tuple[float, ...]:
    """The exact CPU dense-score suite, excluding all candidate/hash work."""

    return tuple(float(score_channel_cpu(item.query, item.references, item.k)) for item in fixtures)


def _candidate_coordinates(fixtures: Sequence[MpsFixture], scores: Sequence[float]) -> tuple[tuple[int, int], ...]:
    coordinates = tuple((index, 0) for index, (item, score) in enumerate(zip(fixtures, scores, strict=True)) if score > item.threshold)
    return coordinates


def _fixture_output_sha(scores: Sequence[float], coordinates: tuple[tuple[int, int], ...]) -> str:
    payload = json.dumps({"scores": scores, "candidate_coordinates": coordinates}, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


class MpsEngine:
    """Lazy PyTorch engine: all MPS work, copy, and synchronization are in score."""

    backend = "mps"

    def __init__(self) -> None:
        try:
            import torch  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - dependent on host install
            raise MpsUnavailable("pytorch_framework_unavailable") from exc
        self.torch = torch
        self.pytorch_version = str(getattr(torch, "__version__", "unknown"))
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        built = getattr(mps_backend, "is_built", None)
        available = getattr(mps_backend, "is_available", None)
        self.metal_available = bool(callable(built) and built() and callable(available) and available())
        self.device = "mps"
        self.device_info = json.dumps({"mps_built": bool(callable(built) and built()), "mps_available": self.metal_available}, separators=(",", ":"))
        if not self.metal_available:
            raise MpsUnavailable("mps_unavailable", pytorch_version=self.pytorch_version, device=self.device,
                                 device_info=self.device_info, metal_available=False)

    def score(self, query: object, references: object, k: int, *, batch_size: int = 16) -> float:
        query_array, reference_array = _vector(query, "query"), _matrix(references)
        if query_array.shape[0] != reference_array.shape[1] or reference_array.shape[0] < FROZEN_CONFIG.neighbor_count:
            raise ValueError("invalid scalar dense score shape or frozen reference count")
        if type(k) is not int or k != FROZEN_CONFIG.neighbor_count:
            raise ValueError("k must equal the frozen neighbor count")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive exact integer")
        torch = self.torch
        with torch.no_grad():
            query_device = torch.as_tensor(query_array, dtype=torch.float32, device="mps")
            nearest = None
            for start in range(0, reference_array.shape[0], batch_size):
                refs = torch.as_tensor(reference_array[start:start + batch_size], dtype=torch.float32, device="mps")
                distances = torch.sqrt(torch.mean((refs - query_device) ** 2, dim=1))
                local = torch.sort(distances).values[:min(k, int(refs.shape[0]))]
                nearest = local if nearest is None else torch.sort(torch.cat((nearest, local))).values[:k]
            assert nearest is not None
            result = nearest[k // 2] if k % 2 else (nearest[k // 2 - 1] + nearest[k // 2]) / 2.0
            output = result.detach().to("cpu").numpy()
            torch.mps.synchronize()
        score = float(np.asarray(output, dtype=np.float32))
        if not np.isfinite(score):
            raise ValueError("MPS scalar score is nonfinite")
        return score


def _normalise(fixtures: object) -> tuple[MpsFixture, ...]:
    if isinstance(fixtures, MpsFixture): return (fixtures,)
    if isinstance(fixtures, (str, bytes)) or not isinstance(fixtures, Sequence):
        raise ValueError("fixtures must be a non-empty sequence of scalar fixtures")
    prepared = tuple(fixtures)
    if not prepared or any(not isinstance(item, MpsFixture) for item in prepared):
        raise ValueError("fixtures must be a non-empty sequence of MpsFixture")
    return prepared


def _timed(timer: Callable[[str, int, Callable[[], object], Callable[[], float]], float] | None,
           kind: str, batch_size: int, call: Callable[[], object], clock: Callable[[], float]) -> tuple[float, object]:
    value: object | None = None
    def invoke() -> object:
        nonlocal value
        value = call(); return value
    if timer is None:
        start = float(clock()); invoke(); elapsed = float(clock()) - start
    else: elapsed = float(timer(kind, batch_size, invoke, clock))
    if not np.isfinite(elapsed) or elapsed <= 0: raise ValueError("benchmark timing must be finite and positive")
    return elapsed, value


def _device(engine: object, fixtures: Sequence[MpsFixture], batch_size: int) -> tuple[float, ...]:
    method = getattr(engine, "score", None)
    if not callable(method): raise ValueError("MPS engine must expose score")
    return tuple(float(method(item.query, item.references, item.k, batch_size=batch_size)) for item in fixtures)


def _validate_device_suite(raw: object, fixtures: Sequence[MpsFixture], cpu_scores: Sequence[float],
                           cpu_coordinates: tuple[tuple[int, int], ...]) -> tuple[float, float, bool]:
    output = np.asarray(raw, dtype=np.float32)
    expected = np.asarray(cpu_scores, dtype=np.float32)
    if output.shape != expected.shape or not np.isfinite(output).all():
        raise ValueError("nonfinite_or_invalid_device_output")
    coordinates = _candidate_coordinates(fixtures, output.tolist())
    absolute = float(np.max(np.abs(output - expected)))
    normalized = float(np.max(np.abs(output - expected) / np.maximum(np.abs(expected), np.float32(FROZEN_CONFIG.numeric_floor))))
    return absolute, normalized, coordinates == cpu_coordinates


def _batch(batch_size: int, fixtures: Sequence[MpsFixture], input_sha: str, output_sha: str, cpu_scores: tuple[float, ...],
           cpu_coordinates: tuple[tuple[int, int], ...], engine: object, clock: Callable[[], float],
           timer: Callable[[str, int, Callable[[], object], Callable[[], float]], float] | None) -> BackendBatchEvidence:
    warmups, pairs = 0, []
    absolute_max = normalized_max = 0.0
    identity_match = True
    try:
        for _ in range(FROZEN_CONFIG.backend_warmup_runs):
            _cpu_score_suite(fixtures)
            absolute, normalized, identity = _validate_device_suite(
                _device(engine, fixtures, batch_size), fixtures, cpu_scores, cpu_coordinates
            )
            absolute_max, normalized_max = max(absolute_max, absolute), max(normalized_max, normalized)
            identity_match = identity_match and identity
            warmups += 1
        for _ in range(FROZEN_CONFIG.backend_timed_runs):
            cpu_time, _ = _timed(timer, "cpu", batch_size, lambda: _cpu_score_suite(fixtures), clock)
            device_time, raw = _timed(timer, "device", batch_size, lambda: _device(engine, fixtures, batch_size), clock)
            absolute, normalized, identity = _validate_device_suite(raw, fixtures, cpu_scores, cpu_coordinates)
            pairs.append((cpu_time, device_time))
            absolute_max, normalized_max = max(absolute_max, absolute), max(normalized_max, normalized)
            identity_match = identity_match and identity
        cpu_median, device_median = float(median(pair[0] for pair in pairs)), float(median(pair[1] for pair in pairs))
        speedup = cpu_median / device_median
        parity = identity_match and absolute_max <= FROZEN_CONFIG.score_tolerance and normalized_max <= FROZEN_CONFIG.score_tolerance
        speed = speedup >= FROZEN_CONFIG.backend_min_speedup
        reason = "passed" if parity and speed else (
            "candidate_identity_mismatch" if not identity_match else
            "absolute_error_exceeded" if absolute_max > FROZEN_CONFIG.score_tolerance else
            "normalized_error_exceeded" if normalized_max > FROZEN_CONFIG.score_tolerance else
            "speedup_below_threshold"
        )
        return BackendBatchEvidence("mps", batch_size, "completed", warmups, tuple(pairs), input_sha, output_sha, True, identity_match,
                                    absolute_max, normalized_max, cpu_median, device_median, speedup, parity, speed, parity and speed, reason)
    except Exception as exc:
        return BackendBatchEvidence("mps", batch_size, "runtime_failed", warmups, tuple(pairs), input_sha, output_sha,
                                    None, None, None, None, None, None, None, None, None, False, f"mps_runtime_failed:{type(exc).__name__}")


def benchmark_mps_adoption(fixtures: object, smoke_seed: int = FROZEN_CONFIG.backend_smoke_seed,
                           clock: Callable[[], float] = perf_counter, *, engine: object | None = None,
                           timer: Callable[[str, int, Callable[[], object], Callable[[], float]], float] | None = None) -> BackendBenchmarkEvidence:
    """Return only MPS-local evidence. Selection belongs to backend_antigravity."""
    FROZEN_CONFIG.validate()
    if smoke_seed != FROZEN_CONFIG.backend_smoke_seed: raise ValueError("MPS smoke seed must equal frozen seed 43601")
    prepared = _normalise(fixtures)
    if engine is None:
        try: engine = MpsEngine()
        except MpsUnavailable as exc: return BackendBenchmarkEvidence.unavailable("mps", exc.reason, **exc.provenance)
    provenance = {name: getattr(engine, name, None) for name in ("pytorch_version", "device", "device_info", "metal_available")}
    if (getattr(engine, "backend", "mps") != "mps" or provenance["metal_available"] is not True
            or not _is_mps_device(provenance["device"])
            or any(not isinstance(provenance[name], str) or not provenance[name] for name in ("pytorch_version", "device", "device_info"))):
        return BackendBenchmarkEvidence.unavailable("mps", "mps_engine_provenance_incomplete", **provenance)
    input_sha = _digest_fixtures(prepared)
    cpu_scores = _cpu_score_suite(prepared)
    coordinates = _candidate_coordinates(prepared, cpu_scores)
    output_sha = _fixture_output_sha(cpu_scores, coordinates)
    batches = tuple(_batch(size, prepared, input_sha, output_sha, cpu_scores, coordinates, engine, clock, timer) for size in FROZEN_CONFIG.backend_batch_sizes)
    passing = [item for item in batches if item.passed]
    chosen = min(passing, key=lambda item: (item.device_median_seconds, item.batch_size)) if passing else None
    return BackendBenchmarkEvidence("mps", True, batches, platform.python_version(), np.__version__, str(provenance["device"]), str(provenance["device_info"]),
                                    True, FROZEN_CONFIG.backend_protocol, None if chosen is None else chosen.batch_size,
                                    None if chosen is None else chosen.device_median_seconds, "no_passing_batch" if chosen is None else None,
                                    smoke_seed, None, None, str(provenance["pytorch_version"]))


__all__ = ["MpsEngine", "MpsFixture", "MpsUnavailable", "benchmark_mps_adoption"]
