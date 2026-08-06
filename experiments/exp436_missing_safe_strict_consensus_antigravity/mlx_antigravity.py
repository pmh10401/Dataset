"""Optional MLX implementation of Exp436's scalar dense RMS score only.

The module intentionally knows nothing about MPS selection or consensus
scoring.  It can supply one backend-local readiness record; the neutral
dispatcher owns adoption and production routing.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import metadata
import json
import platform
from statistics import median
from time import perf_counter
from typing import Callable, Sequence

import numpy as np

from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import BackendBatchEvidence, BackendBenchmarkEvidence
from .features_antigravity import score_channel_cpu


class MlxUnavailable(RuntimeError):
    """Optional MLX discovery failed before a truthful benchmark could start."""

    def __init__(self, reason: str, **provenance: object) -> None:
        self.reason = reason
        self.provenance = provenance
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


@dataclass(frozen=True)
class MlxFixture:
    """One production-shaped scalar call and its CPU-only candidate threshold."""

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


def _digest_fixtures(fixtures: Sequence[MlxFixture]) -> str:
    digest = sha256()
    for item in fixtures:
        for array in (item.query, item.references):
            little = np.asarray(array, dtype="<f4", order="C")
            digest.update(str(little.shape).encode("ascii"))
            digest.update(b"\0")
            digest.update(little.tobytes())
            digest.update(b"\0")
        threshold = np.asarray(item.threshold, dtype="<f4")
        digest.update(str(threshold.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(threshold.tobytes())
        digest.update(b"\0")
        digest.update(str(item.k).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _cpu_score_suite(fixtures: Sequence[MlxFixture]) -> tuple[float, ...]:
    """The exact CPU score operation, with no routing or serialization work."""

    return tuple(float(score_channel_cpu(item.query, item.references, item.k)) for item in fixtures)


def _candidate_coordinates(fixtures: Sequence[MlxFixture], scores: Sequence[float]) -> tuple[tuple[int, int], ...]:
    coordinates = tuple((index, 0) for index, (item, score) in enumerate(zip(fixtures, scores, strict=True)) if score > item.threshold)
    return coordinates


def _fixture_output_sha(scores: Sequence[float], coordinates: tuple[tuple[int, int], ...]) -> str:
    payload = json.dumps({"scores": scores, "candidate_coordinates": coordinates}, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()


class MlxEngine:
    """Lazy MLX engine that creates device tensors inside every scalar call."""

    backend = "mlx"

    def __init__(self) -> None:
        try:
            import mlx.core as mx  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - dependent on host install
            raise MlxUnavailable("mlx_framework_unavailable") from exc
        self.mx = mx
        self.mlx_version = _version("mlx", getattr(mx, "__version__", None))
        if self.mlx_version == "unknown":
            raise MlxUnavailable("mlx_version_unavailable")
        try:
            self.mlx_metal_version = str(metadata.version("mlx-metal"))
        except metadata.PackageNotFoundError as exc:
            raise MlxUnavailable("mlx_metal_version_unavailable", mlx_version=self.mlx_version) from exc
        try:
            self.gpu = getattr(mx, "gpu", None)
            if self.gpu is None or not callable(getattr(mx, "stream", None)):
                raise MlxUnavailable("mlx_gpu_stream_unavailable", mlx_version=self.mlx_version,
                                     mlx_metal_version=self.mlx_metal_version)
            self.device = str(self.gpu)
            if not _is_metal_device(self.device):
                raise MlxUnavailable("mlx_gpu_device_unavailable", mlx_version=self.mlx_version,
                                     mlx_metal_version=self.mlx_metal_version, device=self.device)
            metal = getattr(mx, "metal", None)
            checker = getattr(metal, "is_available", None)
            self.metal_available = bool(checker()) if callable(checker) else False
            info = getattr(mx, "device_info", None) or getattr(metal, "device_info", None)
            if not callable(info):
                raise MlxUnavailable("mlx_device_info_unavailable", mlx_version=self.mlx_version,
                                     mlx_metal_version=self.mlx_metal_version, device=self.device,
                                     metal_available=self.metal_available)
            self.device_info = _canonical_info(info())
        except Exception as exc:  # pragma: no cover - backend-specific probing
            raise MlxUnavailable("mlx_device_probe_failed", mlx_version=self.mlx_version) from exc
        if not self.metal_available:
            raise MlxUnavailable("mlx_metal_unavailable", mlx_version=self.mlx_version,
                                 mlx_metal_version=self.mlx_metal_version, device=self.device,
                                 device_info=self.device_info, metal_available=False)

    def score(self, query: object, references: object, k: int, *, batch_size: int = 16) -> float:
        query_array, reference_array = _vector(query, "query"), _matrix(references)
        if query_array.shape[0] != reference_array.shape[1] or reference_array.shape[0] < FROZEN_CONFIG.neighbor_count:
            raise ValueError("invalid scalar dense score shape or frozen reference count")
        if type(k) is not int or k != FROZEN_CONFIG.neighbor_count:
            raise ValueError("k must equal the frozen neighbor count")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive exact integer")
        mx = self.mx
        synchronize = getattr(mx, "synchronize", None)
        if not callable(synchronize):
            raise MlxUnavailable("mlx_synchronization_unavailable")
        with mx.stream(self.gpu):
            query_device = mx.array(query_array, dtype=mx.float32)
            nearest = None
            for start in range(0, reference_array.shape[0], batch_size):
                refs = mx.array(reference_array[start:start + batch_size], dtype=mx.float32)
                distances = mx.sqrt(mx.mean(mx.square(refs - query_device), axis=1))
                local = mx.sort(distances)[:min(k, refs.shape[0])]
                nearest = local if nearest is None else mx.sort(mx.concatenate((nearest, local)))[:k]
            assert nearest is not None
            result = nearest[k // 2] if k % 2 else (nearest[k // 2 - 1] + nearest[k // 2]) / np.float32(2.0)
            mx.eval(result)
            synchronize(self.gpu)
        score = float(np.asarray(result, dtype=np.float32))
        if not np.isfinite(score):
            raise ValueError("MLX scalar score is nonfinite")
        return score


def _version(distribution: str, fallback: object) -> str:
    try:
        return str(metadata.version(distribution))
    except metadata.PackageNotFoundError:
        return str(fallback) if fallback is not None else "unknown"


def _canonical_info(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _is_metal_device(value: object) -> bool:
    text = str(value).strip().lower()
    return bool(text) and "cpu" not in text and ("gpu" in text or "metal" in text)


def _normalise(fixtures: object) -> tuple[MlxFixture, ...]:
    if isinstance(fixtures, MlxFixture):
        return (fixtures,)
    if isinstance(fixtures, (str, bytes)) or not isinstance(fixtures, Sequence):
        raise ValueError("fixtures must be a non-empty sequence of scalar fixtures")
    prepared = tuple(fixtures)
    if not prepared or any(not isinstance(item, MlxFixture) for item in prepared):
        raise ValueError("fixtures must be a non-empty sequence of MlxFixture")
    return prepared


def _provenance(engine: object) -> dict[str, object]:
    return {name: getattr(engine, name, None) for name in (
        "mlx_version", "mlx_metal_version", "device", "device_info", "metal_available",
    )}


def _timed(timer: Callable[[str, int, Callable[[], object], Callable[[], float]], float] | None,
           kind: str, batch_size: int, call: Callable[[], object], clock: Callable[[], float]) -> tuple[float, object]:
    value: object | None = None
    def invoke() -> object:
        nonlocal value
        value = call()
        return value
    if timer is None:
        start = float(clock()); invoke(); elapsed = float(clock()) - start
    else:
        elapsed = float(timer(kind, batch_size, invoke, clock))
    if not np.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("benchmark timing must be finite and positive")
    return elapsed, value


def _suite_device(engine: object, fixtures: Sequence[MlxFixture], batch_size: int) -> tuple[float, ...]:
    method = getattr(engine, "score", None)
    if not callable(method):
        raise ValueError("MLX engine must expose score")
    return tuple(float(method(item.query, item.references, item.k, batch_size=batch_size)) for item in fixtures)


def _validate_device_suite(raw: object, fixtures: Sequence[MlxFixture], cpu_scores: Sequence[float],
                           cpu_coordinates: tuple[tuple[int, int], ...]) -> tuple[float, float, bool]:
    output = np.asarray(raw, dtype=np.float32)
    expected = np.asarray(cpu_scores, dtype=np.float32)
    if output.shape != expected.shape or not np.isfinite(output).all():
        raise ValueError("nonfinite_or_invalid_device_output")
    coordinates = _candidate_coordinates(fixtures, output.tolist())
    absolute = float(np.max(np.abs(output - expected)))
    normalized = float(np.max(np.abs(output - expected) / np.maximum(np.abs(expected), np.float32(FROZEN_CONFIG.numeric_floor))))
    return absolute, normalized, coordinates == cpu_coordinates


def _runtime_batch(backend: str, batch_size: int, fixtures: Sequence[MlxFixture], input_sha: str, output_sha: str,
                   cpu_scores: tuple[float, ...], cpu_coordinates: tuple[tuple[int, int], ...], engine: object,
                   clock: Callable[[], float], timer: Callable[[str, int, Callable[[], object], Callable[[], float]], float] | None) -> BackendBatchEvidence:
    warmups = 0
    pairs: list[tuple[float, float]] = []
    absolute_max = normalized_max = 0.0
    identity_match = True
    try:
        for _ in range(FROZEN_CONFIG.backend_warmup_runs):
            _cpu_score_suite(fixtures)
            absolute, normalized, identity = _validate_device_suite(
                _suite_device(engine, fixtures, batch_size), fixtures, cpu_scores, cpu_coordinates
            )
            absolute_max, normalized_max = max(absolute_max, absolute), max(normalized_max, normalized)
            identity_match = identity_match and identity
            warmups += 1
        for _ in range(FROZEN_CONFIG.backend_timed_runs):
            cpu_time, _ = _timed(timer, "cpu", batch_size, lambda: _cpu_score_suite(fixtures), clock)
            device_time, raw = _timed(timer, "device", batch_size, lambda: _suite_device(engine, fixtures, batch_size), clock)
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
        return BackendBatchEvidence(backend, batch_size, "completed", warmups, tuple(pairs), input_sha, output_sha,
                                    True, identity_match, absolute_max, normalized_max, cpu_median, device_median, speedup,
                                    parity, speed, parity and speed, reason)
    except Exception as exc:
        return BackendBatchEvidence(backend, batch_size, "runtime_failed", warmups, tuple(pairs), input_sha, output_sha,
                                    None, None, None, None, None, None, None, None, None, False,
                                    f"mlx_runtime_failed:{type(exc).__name__}")


def benchmark_mlx_adoption(fixtures: object, smoke_seed: int = FROZEN_CONFIG.backend_smoke_seed,
                           clock: Callable[[], float] = perf_counter, *, engine: object | None = None,
                           timer: Callable[[str, int, Callable[[], object], Callable[[], float]], float] | None = None) -> BackendBenchmarkEvidence:
    """Benchmark only the frozen scalar dense operation; never select a backend."""

    FROZEN_CONFIG.validate()
    if smoke_seed != FROZEN_CONFIG.backend_smoke_seed:
        raise ValueError("MLX smoke seed must equal frozen seed 43601")
    prepared = _normalise(fixtures)
    if engine is None:
        try:
            engine = MlxEngine()
        except MlxUnavailable as exc:
            return BackendBenchmarkEvidence.unavailable("mlx", exc.reason, **exc.provenance)
    provenance = _provenance(engine)
    if getattr(engine, "backend", "mlx") != "mlx" or provenance["metal_available"] is not True or not _is_metal_device(provenance["device"]) or any(not isinstance(provenance[name], str) or not provenance[name] for name in ("mlx_version", "mlx_metal_version", "device", "device_info")):
        return BackendBenchmarkEvidence.unavailable("mlx", "mlx_engine_provenance_incomplete", **provenance)
    input_sha = _digest_fixtures(prepared)
    cpu_scores = _cpu_score_suite(prepared)
    cpu_coordinates = _candidate_coordinates(prepared, cpu_scores)
    output_sha = _fixture_output_sha(cpu_scores, cpu_coordinates)
    batches = tuple(_runtime_batch("mlx", batch_size, prepared, input_sha, output_sha, cpu_scores, cpu_coordinates, engine, clock, timer)
                    for batch_size in FROZEN_CONFIG.backend_batch_sizes)
    passing = [item for item in batches if item.passed]
    chosen = min(passing, key=lambda item: (item.device_median_seconds, item.batch_size)) if passing else None
    return BackendBenchmarkEvidence("mlx", True, batches, platform.python_version(), np.__version__, str(provenance["device"]),
                                    str(provenance["device_info"]), True, FROZEN_CONFIG.backend_protocol,
                                    None if chosen is None else chosen.batch_size,
                                    None if chosen is None else chosen.device_median_seconds,
                                    "no_passing_batch" if chosen is None else None, smoke_seed,
                                    str(provenance["mlx_version"]), str(provenance["mlx_metal_version"]), None)


__all__ = ["MlxEngine", "MlxFixture", "MlxUnavailable", "benchmark_mlx_adoption"]
