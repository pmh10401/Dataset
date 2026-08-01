"""Bounded, normal-TRAIN-only Exp421 CPU/MPS diagnostic benchmark.

This module is deliberately small and defensive.  It opens one verified
read-only SQLite connection, asks the frozen Exp417 input boundary for normal
TRAIN values, and evaluates at most a bounded set of exact-length channels.
The bounded device sample is measured in exact batch sizes; the complete
candidate-eligible length stratum is still used for the frozen fit/OOF step,
with its count and waveform digest recorded in provenance.  There is no TEST
loader in this module, no multiprocessing, and no path that can write into
the official Exp419 run.  The returned object is the validated Exp421
manifest; raw series, labels, and evaluation coordinates never enter the
artifact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from experiments.exp417_causal_multihorizon_shadow.contracts import (
    ProcessedGroup,
    RawTrainDataset,
)
from experiments.exp417_causal_multihorizon_shadow.inputs import (
    load_raw_train_values_from_connection,
    verified_read_only_connection,
)
from experiments.exp417_causal_multihorizon_shadow.model import (
    fit_final_seed_models,
    train_oof_models,
)
from experiments.exp417_causal_multihorizon_shadow.preprocessing import (
    build_horizons,
    preprocess_length_groups,
)
from experiments.exp417_causal_multihorizon_shadow.scoring import (
    score_horizon,
    seed_decision,
    consensus_candidate,
)

from .batch_inference import FrozenModelAdapter
from .calibration import build_calibration_cache
from .config import Exp421Config, FROZEN_CONFIG, frozen_config_sha256
from .contracts import (
    BackendMeasurement,
    BatchKey,
    Exp421Manifest,
    MpsAssessment,
    ParityReport,
)
from .manifest import (
    _source_manifest_metadata,
    build_manifest,
    write_manifest,
)
from .perturbations import build_cached_perturbations
from .policy import (
    BackendPolicyBundle,
    MpsCapability,
    assess_mps_capability,
    capability_provenance,
    choose_backend,
)


# These bounds are intentionally module constants.  The first Exp421 config
# was frozen before the benchmark resource knobs were added to the planning
# document; keeping the knobs additive here avoids changing the scientific
# config digest while still enforcing the preregistered limits.
MAX_BENCHMARK_SERIES = 64
MAX_BATCH_BYTES = 256 * 1024 * 1024
_MEASURED_REPEATS = 1
_WARMUP_REPEATS = 1


class BenchmarkBoundaryError(ValueError):
    """Raised when a benchmark request crosses a frozen data/artifact bound."""


@dataclass(frozen=True)
class _TrainSample:
    dataset: str
    group: ProcessedGroup
    indices: tuple[int, ...]
    channels: np.ndarray
    raw_by_index: Mapping[int, np.ndarray]
    max_series: int = MAX_BENCHMARK_SERIES
    max_batch_bytes: int = MAX_BATCH_BYTES

    def __post_init__(self) -> None:
        if not isinstance(self.group, ProcessedGroup):
            raise BenchmarkBoundaryError("sample group is invalid")
        if not isinstance(self.dataset, str) or not self.dataset.strip():
            raise BenchmarkBoundaryError("sample dataset is invalid")
        if not self.indices or tuple(sorted(self.indices)) != self.indices:
            raise BenchmarkBoundaryError("sample indices must be sorted and non-empty")
        if type(self.max_series) is not int or self.max_series < 1:
            raise BenchmarkBoundaryError("sample max_series is invalid")
        if type(self.max_batch_bytes) is not int or self.max_batch_bytes < 1:
            raise BenchmarkBoundaryError("sample max_batch_bytes is invalid")
        if len(self.indices) > self.max_series:
            raise BenchmarkBoundaryError("sample exceeds the bounded series limit")
        if not isinstance(self.channels, np.ndarray) or self.channels.dtype != np.dtype("float32"):
            raise BenchmarkBoundaryError("sample channels must be float32")
        if self.channels.shape != (len(self.indices), 2, self.group.group.length):
            raise BenchmarkBoundaryError("sample channels must preserve one exact length")
        if self.channels.nbytes > self.max_batch_bytes:
            raise BenchmarkBoundaryError("sample channels exceed the bounded memory limit")
        if not np.isfinite(self.channels).all():
            raise BenchmarkBoundaryError("sample channels must be finite")
        for index in self.indices:
            values = self.raw_by_index.get(index)
            if not isinstance(values, np.ndarray) or values.shape != (self.group.group.length,):
                raise BenchmarkBoundaryError("sample raw values do not preserve exact length")
            if not np.isfinite(values).all():
                raise BenchmarkBoundaryError("sample raw values must be finite")
        immutable = np.frombuffer(np.ascontiguousarray(self.channels).tobytes(), dtype=np.float32)
        immutable = immutable.reshape(self.channels.shape)
        immutable.setflags(write=False)
        object.__setattr__(self, "channels", immutable)
        object.__setattr__(self, "raw_by_index", MappingProxyType(dict(self.raw_by_index)))


def _config_bound(config: Exp421Config, name: str, default: int) -> int:
    """Read an additive resource bound without widening Exp421's config digest."""
    value = getattr(config, name, default)
    if type(value) is not int or value < 1:
        raise BenchmarkBoundaryError(f"{name} must be a positive integer")
    return value


def _validate_config(config: Exp421Config) -> tuple[int, int]:
    if type(config) is not Exp421Config:
        raise BenchmarkBoundaryError("benchmark requires the frozen Exp421 config")
    config.validate()
    return (
        _config_bound(config, "max_benchmark_series", MAX_BENCHMARK_SERIES),
        _config_bound(config, "max_batch_bytes", MAX_BATCH_BYTES),
    )


def _path_text(path: Path | str, name: str) -> Path:
    value = Path(path).expanduser()
    try:
        resolved = value.resolve(strict=False)
    except OSError as exc:
        raise BenchmarkBoundaryError(f"{name} cannot be resolved") from exc
    lowered = str(resolved).lower()
    components = [component for component in lowered.replace("\\", "/").split("/") if component]
    if any(
        component == "exp419"
        or component.startswith("exp419_")
        or component.startswith("exp419-")
        or component in {
            "exp419_deterministic_cpu_parallel.checkpoint",
            "exp419_prediction",
            "exp419_checkpoint",
        }
        for component in components
    ):
        raise BenchmarkBoundaryError(f"{name} must not target official Exp419 artifacts")
    return resolved


def _fresh_output_root(path: Path | str) -> Path:
    root = _path_text(path, "output_root")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError("output_root must be a fresh or empty directory")
    return root


def _no_evaluation_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkBoundaryError("dataset names must be non-empty strings")
    name = value.strip()
    lowered = name.lower()
    if any(token in lowered for token in ("label", "ground_truth", "location")):
        raise BenchmarkBoundaryError("dataset names cannot carry evaluation fields")
    return name


def _database_names(connection: Any) -> tuple[str, ...]:
    """Read only dataset names; series values still come exclusively from the TRAIN loader."""
    try:
        rows = connection.execute("SELECT name FROM datasets ORDER BY name").fetchall()
    except Exception as exc:
        raise BenchmarkBoundaryError("unable to enumerate the TRAIN population") from exc
    names = tuple(_no_evaluation_name(row[0]) for row in rows if isinstance(row, (tuple, list)) and len(row) == 1)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise BenchmarkBoundaryError("dataset population must be sorted and unique")
    return names


def _requested_dataset_names(connection: Any, dataset_names: Sequence[str] | None) -> tuple[str, ...]:
    if dataset_names is None:
        return _database_names(connection)
    if isinstance(dataset_names, (str, bytes)):
        raise BenchmarkBoundaryError("dataset_names must be a sequence, not a string")
    names = tuple(_no_evaluation_name(value) for value in dataset_names)
    if not names or len(names) != len(set(names)):
        raise BenchmarkBoundaryError("explicit dataset names must be non-empty and unique")
    return tuple(sorted(names))


def _raw_series_arrays(dataset: RawTrainDataset) -> dict[int, np.ndarray]:
    return {
        item.instance_index: np.asarray(item.values, dtype=np.float64)
        for item in dataset.series
    }


def _select_train_sample(
    connection: Any,
    *,
    database_sha256: str,
    dataset_names: Sequence[str] | None,
    max_series: int,
    max_batch_bytes: int,
) -> _TrainSample | None:
    """Load normal TRAIN rows once and retain a finite exact-length sample."""
    names = _requested_dataset_names(connection, dataset_names)
    loaded = 0
    candidates: list[_TrainSample] = []
    for dataset in names:
        if loaded >= max_series:
            break
        # This is the only values loader called by the benchmark.  It embeds
        # the frozen normal-TRAIN query in Exp417's module.
        raw = load_raw_train_values_from_connection(
            connection,
            dataset,
            database_sha256=database_sha256,
            datasets=names,
        )
        groups = preprocess_length_groups(raw)
        for group in groups:
            if group.status != "candidate_eligible" or not group.channels:
                continue
            finite_indices = tuple(
                item.instance_index
                for item in group.group.series
                if item.instance_index in group.channels
                and np.isfinite(group.channels[item.instance_index]).all()
            )
            if not finite_indices:
                continue
            take = finite_indices[: max_series - loaded]
            channels = np.stack([group.channels[index] for index in take]).astype(
                np.float32, copy=True, order="C"
            )
            if channels.nbytes > max_batch_bytes:
                row_bytes = int(channels.strides[0])
                if row_bytes > max_batch_bytes:
                    raise BenchmarkBoundaryError("one exact TRAIN series exceeds the batch memory bound")
                allowed = max(1, min(len(take), max_batch_bytes // max(1, row_bytes)))
                take = take[:allowed]
                channels = channels[:allowed].copy()
            sample = _TrainSample(
                dataset=dataset,
                group=group,
                indices=tuple(take),
                channels=channels,
                raw_by_index=_raw_series_arrays(raw),
                max_series=max_series,
                max_batch_bytes=max_batch_bytes,
            )
            candidates.append(sample)
            loaded += len(take)
            if loaded >= max_series:
                break
    if not candidates:
        return None
    # Use the first lexicographic exact-length stratum.  Keeping one length
    # makes every candidate batch comparable and avoids implicit padding.
    candidates.sort(key=lambda item: (item.group.group.length, item.dataset, item.indices))
    first = candidates[0]
    return first


def _coordinate_digest(
    indices: Sequence[int], *, dataset: str = "", length: int | None = None
) -> str:
    digest = sha256()
    digest.update(b"exp421.train.coordinates.v1\x00")
    digest.update(dataset.encode("utf-8"))
    digest.update(b"\x00")
    if length is not None:
        digest.update(int(length).to_bytes(8, "little", signed=False))
    for index in indices:
        digest.update(int(index).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def _device_sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _timed_predict(adapter: FrozenModelAdapter, channels: np.ndarray) -> tuple[Mapping[int, np.ndarray], tuple[float, float, float, float]]:
    """Run one adapter network call while timing transfer, compute, and sync."""
    if not isinstance(adapter, FrozenModelAdapter):
        raise BenchmarkBoundaryError("adapter is invalid")
    if channels.dtype != np.dtype("float32") or channels.ndim != 3:
        raise BenchmarkBoundaryError("benchmark channels must be float32 [N,2,L]")
    if channels.nbytes > MAX_BATCH_BYTES:
        raise BenchmarkBoundaryError("benchmark batch exceeds memory bound")
    values = np.array(channels, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(values).all():
        raise BenchmarkBoundaryError("benchmark channels must be finite")
    device = adapter.device
    network = adapter._network
    network.eval()
    transfer_start = perf_counter()
    inputs = torch.from_numpy(values).to(device)
    transfer_in = perf_counter() - transfer_start
    if inputs.device != device:
        raise BenchmarkBoundaryError("input tensor silently changed device")
    compute_start = perf_counter()
    with torch.no_grad():
        predictions = network(inputs)
        if tuple(predictions) != adapter.model.horizons:
            raise BenchmarkBoundaryError("network horizons differ from frozen model")
        packed = torch.cat([predictions[horizon].reshape(-1) for horizon in adapter.model.horizons])
    compute_seconds = perf_counter() - compute_start
    sync_start = perf_counter()
    _device_sync(device)
    synchronize_seconds = perf_counter() - sync_start
    output_start = perf_counter()
    packed_cpu = packed.detach().cpu().numpy()
    if not np.isfinite(packed_cpu).all():
        raise BenchmarkBoundaryError("model outputs must be finite")
    residuals: dict[int, np.ndarray] = {}
    offset = 0
    for horizon in adapter.model.horizons:
        expected = values.shape[0] * (values.shape[2] - horizon)
        prediction = packed_cpu[offset : offset + expected].reshape(values.shape[0], values.shape[2] - horizon)
        offset += expected
        residual = np.abs(prediction - values[:, 0, horizon:]).astype(np.float64, copy=False)
        if not np.isfinite(residual).all():
            raise BenchmarkBoundaryError("residuals must be finite")
        frozen = np.array(residual, dtype=np.float64, order="C", copy=True)
        frozen.setflags(write=False)
        residuals[horizon] = frozen
    if offset != packed_cpu.size:
        raise BenchmarkBoundaryError("network output size differs from frozen horizons")
    transfer_seconds = transfer_in + perf_counter() - output_start
    total_seconds = transfer_seconds + compute_seconds + synchronize_seconds
    return MappingProxyType(residuals), (
        max(transfer_seconds, 0.0),
        max(compute_seconds, 0.0),
        max(synchronize_seconds, 0.0),
        max(total_seconds, 1e-12),
    )


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Return deterministic average ranks without importing another experiment.

    Exp421 keeps its source trust root self-contained.  This is the same
    tie-aware, stable-sort rule used by the reviewed parity helper: equal
    values receive the mean of their occupied one-based ranks.
    """
    array = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Compute tie-aware Spearman correlation for parity evidence."""
    if np.asarray(left).size != np.asarray(right).size:
        return 0.0
    left_rank = _rankdata(np.asarray(left))
    right_rank = _rankdata(np.asarray(right))
    if len(left_rank) < 2 or float(np.std(left_rank)) <= 1e-12 or float(np.std(right_rank)) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(left_rank, right_rank)[0, 1])
    return value if np.isfinite(value) else 0.0


def _candidate_indices(
    residuals_by_seed: Sequence[Mapping[int, np.ndarray]],
    indices: Sequence[int],
    config: Exp421Config,
    *,
    oof_by_seed: Mapping[int, Mapping[int, Mapping[int, np.ndarray]]] | None = None,
) -> tuple[np.ndarray, Mapping[str, str], str]:
    """Apply the exact Exp417 score, seed, and consensus candidate rule."""
    if not residuals_by_seed or len(residuals_by_seed) != len(config.frozen_model_seeds):
        raise BenchmarkBoundaryError("candidate evidence requires every frozen model seed")
    horizons = tuple(sorted(residuals_by_seed[0]))
    if len(horizons) != 3 or any(tuple(sorted(item)) != horizons for item in residuals_by_seed):
        raise BenchmarkBoundaryError("candidate evidence must cover all frozen horizons")
    if any(residuals[horizon].shape[0] != len(indices) for residuals in residuals_by_seed for horizon in horizons):
        raise BenchmarkBoundaryError("candidate residual rows do not bind to TRAIN coordinates")
    if oof_by_seed is not None:
        if set(oof_by_seed) != set(config.frozen_model_seeds):
            raise BenchmarkBoundaryError("OOF evidence must contain every frozen model seed")
        for seed in config.frozen_model_seeds:
            if any(int(index) not in oof_by_seed[seed] for index in indices):
                raise BenchmarkBoundaryError("query coordinates must belong to the OOF identity universe")
    evidence_hash = sha256(b"exp421.oof.v1\x00")
    calibration_hash = sha256(b"exp421.calibration.v1\x00")
    for seed_position, residuals in enumerate(residuals_by_seed):
        seed = config.frozen_model_seeds[seed_position]
        for horizon in horizons:
            if oof_by_seed is None:
                values = tuple(residuals[horizon][row] for row in range(residuals[horizon].shape[0]))
            else:
                values = tuple(oof_by_seed[seed][index][horizon] for index in sorted(oof_by_seed[seed]))
                for index in sorted(oof_by_seed[seed]):
                    evidence_hash.update(int(index).to_bytes(8, "little", signed=False))
            for item in values:
                evidence_hash.update(np.ascontiguousarray(item, dtype="<f8").tobytes())
            try:
                cache = build_calibration_cache(values, horizon=horizon)
                calibration_hash.update(cache.digest.encode("ascii"))
            except ValueError:
                calibration_hash.update(b"calibration_unavailable\x00")
    selected: set[int] = set()
    qualification_hash = sha256(b"exp421.qualification.v1\x00")
    for row, index in enumerate(indices):
        decisions = []
        for seed_position, residuals in enumerate(residuals_by_seed):
            scores = {}
            for horizon in horizons:
                # The candidate is scored against the complete normal TRAIN
                # OOF universe when a frozen bundle is present.  The
                # no-bundle fallback is retained only for isolated unit
                # fixtures and removes the query row to avoid self-calibration.
                if oof_by_seed is None:
                    oof = tuple(residuals[horizon][other] for other in range(len(indices)) if other != row)
                else:
                    oof = tuple(
                        oof_by_seed[config.frozen_model_seeds[seed_position]][index][horizon]
                        for index in sorted(oof_by_seed[config.frozen_model_seeds[seed_position]])
                    )
                if len(oof) < 2:
                    scores = {}
                    break
                try:
                    scored = score_horizon(oof, residuals[horizon][row], horizon=horizon)
                    scores[horizon] = scored
                except ValueError:
                    scores = {}
                    break
            if len(scores) != len(horizons):
                decisions = []
                break
            decisions.append(seed_decision(scores, seed=config.frozen_model_seeds[seed_position], config=config.exp417_config))
        accepted = bool(decisions) and consensus_candidate(decisions)
        qualification_hash.update(int(index).to_bytes(8, "little", signed=False))
        qualification_hash.update(b"\x01" if accepted else b"\x00")
        if accepted:
            selected.add(index)
    values = np.asarray(sorted(selected), dtype=np.int64)
    values.setflags(write=False)
    return values, MappingProxyType({
        "oof_sha256": evidence_hash.hexdigest(),
        "calibration_sha256": calibration_hash.hexdigest(),
        "qualification_sha256": qualification_hash.hexdigest(),
    }), _coordinate_digest(values)


def _perturbation_provenance(sample: _TrainSample, config: Exp421Config) -> str:
    """Exercise the frozen TRAIN perturbation definitions without serializing arrays."""
    digest = sha256(b"exp421.train.perturbation-provenance.v1\x00")
    scale = max(float(sample.group.raw_scale), 1e-6)
    repaired_by_index = {
        item.instance_index: np.asarray(item.values, dtype=np.float64)
        for item in sample.group.group.series
    }
    for row, index in enumerate(sample.indices):
        series = repaired_by_index[index]
        # PerturbationCacheKey intentionally requires a TRAIN namespace.  The
        # hash-derived identifier contains no private dataset name in output.
        identifier = "train_" + sha256(sample.dataset.encode("utf-8")).hexdigest()[:16]
        for seed in config.frozen_perturbation_seeds:
            cached = build_cached_perturbations(identifier, index, series, scale, seed)
            digest.update(cached.digest.encode("ascii"))
        if row >= MAX_BENCHMARK_SERIES - 1:
            break
    return digest.hexdigest()


def _fit_population_provenance(sample: _TrainSample) -> tuple[int, str]:
    """Hash the complete eligible TRAIN stratum used by fit and OOF.

    Device prediction is bounded by ``sample.indices``.  Model fitting and
    OOF calibration intentionally use the full exact-length eligible stratum
    so the frozen Exp417 candidate rule is not changed by the device cap.
    This digest makes that distinction auditable without serializing values.
    """
    digest = sha256(b"exp421.train.fit-population.v1\x00")
    series = tuple(sorted(sample.group.group.series, key=lambda item: item.instance_index))
    for item in series:
        values = np.asarray(item.values, dtype="<f8")
        digest.update(int(item.instance_index).to_bytes(8, "little", signed=False))
        digest.update(int(values.size).to_bytes(8, "little", signed=False))
        digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    return len(series), digest.hexdigest()


def _synthetic_qualification_parity(
    sample: _TrainSample,
    *,
    cpu_adapters: Mapping[int, FrozenModelAdapter],
    mps_adapters: Mapping[int, FrozenModelAdapter],
    oof_by_seed: Mapping[int, Mapping[int, Mapping[int, np.ndarray]]],
    config: Exp421Config,
) -> tuple[str, bool, str]:
    """Run a bounded TRAIN perturbation qualification parity check.

    One deterministic TRAIN identity, every frozen perturbation type, and
    every perturbation seed are enough to exercise the qualification decision
    boundary.  MPS is never selected from this check alone: the resulting
    digest and match bit are an additional gate alongside prediction parity.
    """
    digest = sha256(b"exp421.synthetic-qualification.v1\x00")
    compare_mps = bool(mps_adapters)
    all_match = True
    nonfinite_output = False
    cpu_failure = not bool(cpu_adapters)
    qualification_failure = False
    index = sample.indices[0]
    series = next(item.values for item in sample.group.group.series if item.instance_index == index)
    identifier = "train_" + sha256(sample.dataset.encode("utf-8")).hexdigest()[:16]
    raw_scale = max(float(sample.group.raw_scale), 1e-6)
    difference_scale = max(float(sample.group.difference_scale), 1e-6)
    for perturbation_seed in config.frozen_perturbation_seeds:
        cached = build_cached_perturbations(identifier, index, np.asarray(series, dtype=np.float64), raw_scale, perturbation_seed)
        for perturbation_name in cached:
            raw = np.asarray(cached[perturbation_name], dtype=np.float64)
            differences = np.diff(raw, prepend=raw[:1])
            channels = np.stack(
                (
                    (raw - float(sample.group.raw_center)) / raw_scale,
                    (differences - float(sample.group.difference_center)) / difference_scale,
                ),
                axis=0,
            ).astype(np.float32, copy=False)[None, ...]
            if not np.isfinite(channels).all():
                all_match = False
                cpu_failure = True
                nonfinite_output = True
                digest.update(perturbation_name.encode("ascii") + b"\x00nonfinite")
                continue
            residuals_by_backend: dict[str, dict[int, Mapping[int, np.ndarray]]] = {}
            for backend, adapters in (("cpu", cpu_adapters), ("mps", mps_adapters)):
                if not adapters:
                    continue
                try:
                    residuals_by_backend[backend] = {
                        seed: adapters[seed].predict_batch(channels)
                        for seed in config.frozen_model_seeds
                    }
                except Exception as exc:
                    digest.update(backend.encode("ascii") + b"\x00prediction_error\x00")
                    all_match = False
                    error_text = str(exc).lower()
                    nonfinite_output = nonfinite_output or "finite" in error_text
                    if backend == "cpu":
                        cpu_failure = True
                    else:
                        qualification_failure = True
                    continue
            decisions_by_backend: dict[str, bool] = {}
            flattened: dict[str, np.ndarray] = {}
            for backend, residuals_by_seed in residuals_by_backend.items():
                decisions = []
                flat_parts = []
                try:
                    for seed in config.frozen_model_seeds:
                        scores = {
                            horizon: score_horizon(
                                tuple(oof_by_seed[seed][oof_index][horizon] for oof_index in sorted(oof_by_seed[seed])),
                                residuals_by_seed[seed][horizon][0],
                                horizon=horizon,
                            )
                            for horizon in build_horizons(sample.group.group.length)
                        }
                        decisions.append(seed_decision(scores, seed=seed, config=config.exp417_config))
                        flat_parts.extend(
                            residuals_by_seed[seed][horizon].reshape(-1)
                            for horizon in build_horizons(sample.group.group.length)
                        )
                except (KeyError, TypeError, ValueError) as exc:
                    decisions = []
                    all_match = False
                    qualification_failure = True
                    nonfinite_output = nonfinite_output or "finite" in str(exc).lower()
                    if backend == "cpu":
                        cpu_failure = True
                decisions_by_backend[backend] = bool(decisions) and consensus_candidate(decisions)
                flattened[backend] = np.concatenate(flat_parts) if flat_parts else np.asarray([], dtype=np.float64)
                digest.update(backend.encode("ascii"))
                digest.update(perturbation_seed.to_bytes(8, "little", signed=False))
                digest.update(perturbation_name.encode("ascii") + b"\x00")
                digest.update(b"\x01" if decisions_by_backend[backend] else b"\x00")
            if compare_mps and "mps" in decisions_by_backend:
                cpu_flat = flattened.get("cpu", np.asarray([], dtype=np.float64))
                mps_flat = flattened["mps"]
                denominator = np.maximum(np.abs(cpu_flat), 1e-12)
                error = float(np.max(np.abs(cpu_flat - mps_flat) / denominator)) if cpu_flat.size else 0.0
                rank = _spearman(cpu_flat, mps_flat)
                if (
                    decisions_by_backend.get("cpu") != decisions_by_backend.get("mps")
                    or error > config.max_normalized_error
                    or rank < config.min_spearman
                ):
                    all_match = False
                digest.update(np.asarray([error, rank], dtype="<f8").tobytes())
    if cpu_failure:
        status = "nonfinite_output" if nonfinite_output else "qualification_mismatch"
    elif not compare_mps:
        status = "mps_unavailable"
    elif nonfinite_output:
        status = "nonfinite_output"
    elif all_match and not qualification_failure:
        status = "compared_pass"
    else:
        status = "qualification_mismatch"
    return digest.hexdigest(), all_match, status


def _unavailable_assessment(assessment: MpsAssessment, capability: Any) -> MpsAssessment:
    """Convert an already measured row to an explicit runtime CPU fallback."""
    return MpsAssessment(
        batch_key=assessment.batch_key,
        cpu_measurement=assessment.cpu_measurement,
        mps_measurement=None,
        parity_report=None,
        speedup=None,
        uses_mps=False,
        fallback_reason="mps_unavailable",
        mps_available=False,
        device=str(capability.device_name),
    )


def _nonfinite_mps_assessment(
    *,
    batch_size: int,
    cpu_measurement: BackendMeasurement,
    cpu_candidates: np.ndarray,
    device: str,
) -> MpsAssessment:
    """Retain an available MPS row when its output finite gate fails."""
    key = BatchKey(batch_size, FROZEN_CONFIG.frozen_model_seeds[0], FROZEN_CONFIG.frozen_perturbation_seeds[0])
    mps_measurement = BackendMeasurement(
        batch_size=batch_size,
        backend="mps",
        transfer_seconds=0.0,
        compute_seconds=1e-12,
        synchronize_seconds=0.0,
        total_seconds=1e-12,
        finite=False,
        candidate_coordinates_sha256=cpu_measurement.candidate_coordinates_sha256,
    )
    empty = np.asarray([], dtype=np.int64)
    empty.setflags(write=False)
    parity = ParityReport(
        batch_key=key,
        cpu_candidate_indices_by_alpha={FROZEN_CONFIG.exp417_config.candidate_alpha: cpu_candidates},
        mps_candidate_indices_by_alpha={FROZEN_CONFIG.exp417_config.candidate_alpha: empty},
        max_normalized_error=1.0,
        spearman=0.0,
    )
    return MpsAssessment(
        batch_key=key,
        cpu_measurement=cpu_measurement,
        mps_measurement=mps_measurement,
        parity_report=parity,
        speedup=cpu_measurement.total_seconds / 1e-12,
        uses_mps=False,
        fallback_reason="nonfinite_output",
        mps_available=True,
        device=device,
    )


def _measurements_for_sample(
    sample: _TrainSample,
    *,
    capability: Any,
    config: Exp421Config,
    max_batch_bytes: int,
) -> tuple[dict[int, MpsAssessment], np.ndarray, Mapping[str, object]]:
    """Measure exact batch rows sequentially and derive fail-closed assessments."""
    selected_count = len(sample.indices)
    if selected_count < 1:
        raise BenchmarkBoundaryError("sample is empty")
    if selected_count < max(config.candidate_batch_sizes):
        raise BenchmarkBoundaryError(
            "benchmark sample must contain every exact preregistered batch size"
        )
    assessments: dict[int, MpsAssessment] = {}
    # Fit the frozen final states and OOF residuals on the full eligible
    # normal-TRAIN stratum before applying the bounded prediction cap.  The
    # cap therefore limits device work, not the qualification population.
    fitted_models = fit_final_seed_models(sample.group, device=torch.device("cpu"))
    model_by_seed = {model.seed: model for model in fitted_models}
    if set(model_by_seed) != set(config.frozen_model_seeds):
        raise BenchmarkBoundaryError("frozen final fit omitted a required model seed")
    oof_bundle = train_oof_models(sample.group, device=torch.device("cpu"))
    oof_by_seed: dict[int, Mapping[int, Mapping[int, np.ndarray]]] = {
        int(seed): {
            int(index): {int(horizon): values for horizon, values in by_horizon.items()}
            for index, by_horizon in by_index.items()
        }
        for seed, by_index in oof_bundle.oof_residuals.items()
    }
    evidence_hashes = {
        "model_state_sha256": sha256(
            b"exp421.model-state.v1\x00" + b"".join(
                model_by_seed[seed].model_sha256.encode("ascii")
                for seed in config.frozen_model_seeds
            )
        ).hexdigest(),
        "model_stop_sha256": sha256(
            json.dumps(
                [(seed, model_by_seed[seed].stop_epoch) for seed in config.frozen_model_seeds],
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest(),
        "loss_history_sha256": sha256(
            json.dumps(
                [
                    (seed, model_by_seed[seed].training_losses, model_by_seed[seed].validation_losses)
                    for seed in config.frozen_model_seeds
                ],
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest(),
    }
    perturbation_digest = _perturbation_provenance(sample, config)
    evidence_hashes["perturbation_sha256"] = perturbation_digest
    fit_population_count, fit_population_digest = _fit_population_provenance(sample)
    evidence_accumulators = {
        name: sha256(value.encode("ascii"))
        for name, value in evidence_hashes.items()
    }
    cpu_adapters = {
        seed: FrozenModelAdapter.from_trained_seed(model, device=torch.device("cpu"))
        for seed, model in model_by_seed.items()
    }
    mps_adapters: dict[int, FrozenModelAdapter] = {}
    if capability.mps_available and selected_count >= max(config.candidate_batch_sizes):
        # Device materialization is itself part of the fail-closed boundary:
        # a PyTorch build may report MPS availability while refusing to move a
        # frozen network to the device on this host.  In that case retain a
        # CPU assessment and let the manifest record runtime MPS unavailability.
        try:
            mps_adapters = {
                seed: FrozenModelAdapter.from_trained_seed(model, device=torch.device("mps"))
                for seed, model in model_by_seed.items()
            }
        except Exception:
            mps_adapters = {}
    synthetic_qualification_digest, synthetic_qualification_match, synthetic_qualification_status = _synthetic_qualification_parity(
        sample,
        cpu_adapters=cpu_adapters,
        mps_adapters=mps_adapters,
        oof_by_seed=oof_by_seed,
        config=config,
    )
    coordinate_indices = tuple(sample.indices)
    cached_union: set[int] = set()
    for batch_size in config.candidate_batch_sizes:
        # Every frozen batch size has an exact row count; the boundary above
        # rejects a sample too small for the largest preregistered batch.
        indices = coordinate_indices[:batch_size]
        rows = tuple(sample.indices.index(index) for index in indices)
        batch = np.ascontiguousarray(sample.channels[list(rows)], dtype=np.float32)
        if batch.nbytes > max_batch_bytes:
            raise BenchmarkBoundaryError("batch exceeds the configured memory limit")
        cpu_outputs: list[Mapping[int, np.ndarray]] = []
        cpu_timing = [0.0, 0.0, 0.0]
        for seed in config.frozen_model_seeds:
            adapter = cpu_adapters[seed]
            adapter.predict_batch(batch)  # warm-up is outside measured timing
        for _repeat in range(_MEASURED_REPEATS):
            cpu_outputs.clear()
            cpu_timing = [0.0, 0.0, 0.0]
            for seed in config.frozen_model_seeds:
                output, timing = _timed_predict(cpu_adapters[seed], batch)
                cpu_outputs.append(output)
                for position in range(3):
                    cpu_timing[position] += timing[position]
        cpu_candidates, cpu_evidence, _ = _candidate_indices(
            cpu_outputs, indices, config, oof_by_seed=oof_by_seed
        )
        for name in ("oof_sha256", "calibration_sha256", "qualification_sha256"):
            evidence_accumulators[name] = sha256(
                evidence_accumulators.get(name, sha256()).digest() + cpu_evidence[name].encode("ascii")
            )
        cached_union.update(int(index) for index in cpu_candidates.tolist())
        if not mps_adapters:
            candidate_digest = _coordinate_digest(
                cpu_candidates.tolist(), dataset=sample.dataset, length=sample.group.group.length
            )
            cpu_measurement = BackendMeasurement(
                batch_size=batch_size,
                backend="cpu",
                transfer_seconds=cpu_timing[0],
                compute_seconds=cpu_timing[1],
                synchronize_seconds=cpu_timing[2],
                total_seconds=sum(cpu_timing),
                finite=True,
                candidate_coordinates_sha256=candidate_digest,
            )
            assessment = MpsAssessment(
                batch_key=BatchKey(batch_size, config.frozen_model_seeds[0], config.frozen_perturbation_seeds[0]),
                cpu_measurement=cpu_measurement,
                mps_measurement=None,
                parity_report=None,
                speedup=None,
                uses_mps=False,
                fallback_reason="mps_unavailable",
                mps_available=False,
                device=str(capability.device_name),
            )
            assessments[batch_size] = assessment
            continue
        try:
            for seed in config.frozen_model_seeds:
                mps_adapters[seed].predict_batch(batch)  # warm-up is outside measured timing
            mps_outputs: list[Mapping[int, np.ndarray]] = []
            mps_timing = [0.0, 0.0, 0.0]
            for seed in config.frozen_model_seeds:
                output, timing = _timed_predict(mps_adapters[seed], batch)
                mps_outputs.append(output)
                for position in range(3):
                    mps_timing[position] += timing[position]
        except Exception as exc:
            # A runtime allocation, synchronization, or finite-output failure
            # is a CPU fallback, never an implicit MPS adoption.
            candidate_digest = _coordinate_digest(
                cpu_candidates.tolist(), dataset=sample.dataset, length=sample.group.group.length
            )
            cpu_measurement = BackendMeasurement(
                batch_size=batch_size,
                backend="cpu",
                transfer_seconds=cpu_timing[0],
                compute_seconds=cpu_timing[1],
                synchronize_seconds=cpu_timing[2],
                total_seconds=sum(cpu_timing),
                finite=True,
                candidate_coordinates_sha256=candidate_digest,
            )
            if "finite" in str(exc).lower() or "nonfinite" in str(exc).lower():
                assessments[batch_size] = _nonfinite_mps_assessment(
                    batch_size=batch_size,
                    cpu_measurement=cpu_measurement,
                    cpu_candidates=cpu_candidates,
                    device=str(capability.device_name),
                )
                mps_adapters = {}
                continue
            mps_adapters = {}
            for previous_size, previous in tuple(assessments.items()):
                assessments[previous_size] = _unavailable_assessment(previous, capability)
            assessments[batch_size] = MpsAssessment(
                batch_key=BatchKey(batch_size, config.frozen_model_seeds[0], config.frozen_perturbation_seeds[0]),
                cpu_measurement=cpu_measurement,
                mps_measurement=None,
                parity_report=None,
                speedup=None,
                uses_mps=False,
                fallback_reason="mps_unavailable",
                mps_available=False,
                device=str(capability.device_name),
            )
            continue
        mps_candidates, mps_evidence, _ = _candidate_indices(
            mps_outputs, indices, config, oof_by_seed=oof_by_seed
        )
        for name in ("oof_sha256", "calibration_sha256", "qualification_sha256"):
            evidence_accumulators[name] = sha256(
                evidence_accumulators.get(name, sha256()).digest() + mps_evidence[name].encode("ascii")
            )
        cpu_candidate_digest = _coordinate_digest(
            cpu_candidates.tolist(), dataset=sample.dataset, length=sample.group.group.length
        )
        mps_candidate_digest = _coordinate_digest(
            mps_candidates.tolist(), dataset=sample.dataset, length=sample.group.group.length
        )
        cpu_measurement = BackendMeasurement(
            batch_size=batch_size,
            backend="cpu",
            transfer_seconds=cpu_timing[0],
            compute_seconds=cpu_timing[1],
            synchronize_seconds=cpu_timing[2],
            total_seconds=sum(cpu_timing),
            finite=True,
            candidate_coordinates_sha256=cpu_candidate_digest,
        )
        mps_measurement = BackendMeasurement(
            batch_size=batch_size,
            backend="mps",
            transfer_seconds=mps_timing[0],
            compute_seconds=mps_timing[1],
            synchronize_seconds=mps_timing[2],
            total_seconds=sum(mps_timing),
            finite=True,
            candidate_coordinates_sha256=mps_candidate_digest,
        )
        cpu_flat = np.concatenate([cpu_outputs[seed][horizon].reshape(-1) for seed in range(len(cpu_outputs)) for horizon in cpu_outputs[seed]])
        mps_flat = np.concatenate([mps_outputs[seed][horizon].reshape(-1) for seed in range(len(mps_outputs)) for horizon in mps_outputs[seed]])
        denominator = np.maximum(np.abs(cpu_flat), 1e-12)
        error = float(np.max(np.abs(cpu_flat - mps_flat) / denominator)) if cpu_flat.size else 0.0
        parity = ParityReport(
            batch_key=BatchKey(batch_size, config.frozen_model_seeds[0], config.frozen_perturbation_seeds[0]),
            cpu_candidate_indices_by_alpha={config.exp417_config.candidate_alpha: cpu_candidates},
            mps_candidate_indices_by_alpha={config.exp417_config.candidate_alpha: mps_candidates},
            max_normalized_error=error,
            spearman=_spearman(cpu_flat, mps_flat),
            qualification_match=synthetic_qualification_match,
        )
        speedup = cpu_measurement.total_seconds / max(mps_measurement.total_seconds, 1e-12)
        passing = batch_size != 1 and parity.passes and speedup >= config.min_speedup
        if not parity.candidate_indices_match:
            fallback_reason = "candidate_coordinates_mismatch"
        elif not parity.qualification_match:
            fallback_reason = "qualification_mismatch"
        elif parity.max_normalized_error > config.max_normalized_error:
            fallback_reason = "normalized_error_above_1e-4"
        elif parity.spearman < config.min_spearman:
            fallback_reason = "spearman_below_0.999"
        elif speedup < config.min_speedup:
            fallback_reason = "speedup_below_1.1"
        elif batch_size == 1:
            fallback_reason = "batch_size_one_cpu_mandatory"
        else:
            fallback_reason = None
        assessment = MpsAssessment(
            batch_key=parity.batch_key,
            cpu_measurement=cpu_measurement,
            mps_measurement=mps_measurement,
            parity_report=parity,
            speedup=speedup,
            uses_mps=passing,
            fallback_reason=None if passing else fallback_reason,
            mps_available=True,
            device=str(capability.device_name),
        )
        assessments[batch_size] = assessment
    final_evidence = dict(evidence_hashes)
    for name, accumulator in evidence_accumulators.items():
        if name not in final_evidence:
            final_evidence[name] = accumulator.hexdigest()
    final_evidence["fit_population_count"] = fit_population_count
    final_evidence["fit_population_sha256"] = fit_population_digest
    final_evidence["synthetic_qualification_sha256"] = synthetic_qualification_digest
    final_evidence["synthetic_qualification_status"] = synthetic_qualification_status
    return assessments, np.asarray(sorted(cached_union), dtype=np.int64), MappingProxyType(final_evidence)


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return "0" * 40
    value = completed.stdout.strip().lower()
    return value if len(value) in {40, 64} and all(char in "0123456789abcdef" for char in value) else "0" * 40


def _runtime_label() -> str:
    return f"Python {platform.python_version()} / PyTorch {torch.__version__}"


def _effective_capability(capability: MpsCapability, assessments: Mapping[int, MpsAssessment]) -> MpsCapability:
    """Bind provenance to the runtime path that actually produced evidence."""
    runtime_available = (
        assessments.get(1).mps_available
        if 1 in assessments
        else bool(assessments) and all(item.mps_available for item in assessments.values())
    )
    if runtime_available == capability.mps_available:
        return capability
    # PyTorch can report a built/available backend while model materialization
    # fails on the current host.  The diagnostic must record the effective
    # fail-closed runtime state, not claim that an unmeasured MPS path passed.
    return MpsCapability(
        torch_version=capability.torch_version,
        mps_built=capability.mps_built,
        mps_available=runtime_available,
        device_name=capability.device_name,
        mlx_version=capability.mlx_version,
        mlx_metal_version=capability.mlx_metal_version,
        mlx_availability=capability.mlx_availability,
        mlx_eligibility_reason=capability.mlx_eligibility_reason,
    )


def run_train_only_benchmark(
    database: Path,
    output_root: Path,
    *,
    config: Exp421Config = FROZEN_CONFIG,
    dataset_names: Sequence[str] | None = None,
) -> Exp421Manifest:
    """Run a bounded normal-TRAIN diagnostic and emit its validated manifest."""
    max_series, max_batch_bytes = _validate_config(config)
    benchmark_started_at = datetime.now(timezone.utc).isoformat()
    database_path = _path_text(database, "database")
    attempt_root = _fresh_output_root(output_root)
    expected_db_sha = config.exp417_config.database_sha256
    capability = assess_mps_capability()
    if capability.mps_available:
        torch.set_num_threads(1)
    # Exactly one verified read-only connection spans selection and loading.
    with verified_read_only_connection(database_path, expected_db_sha) as connection:
        sample = _select_train_sample(
            connection,
            database_sha256=expected_db_sha,
            dataset_names=dataset_names,
            max_series=max_series,
            max_batch_bytes=max_batch_bytes,
        )
    if sample is None:
        raise BenchmarkBoundaryError("no finite exact-length normal TRAIN sample is available")
    measurement_result = _measurements_for_sample(
        sample,
        capability=capability,
        config=config,
        max_batch_bytes=max_batch_bytes,
    )
    if not isinstance(measurement_result, tuple) or len(measurement_result) != 3:
        raise BenchmarkBoundaryError("benchmark measurement evidence is incomplete")
    else:
        assessments, cached_indices, evidence = measurement_result
    required_hashes = (
        "fit_population_sha256", "model_state_sha256", "model_stop_sha256",
        "loss_history_sha256", "oof_sha256", "calibration_sha256",
        "perturbation_sha256", "qualification_sha256", "synthetic_qualification_sha256",
    )
    if (
        not isinstance(evidence, Mapping)
        or type(evidence.get("fit_population_count")) is not int
        or evidence["fit_population_count"] < config.exp417_config.min_train_candidate
        or any(
            not isinstance(evidence.get(name), str)
            or len(evidence[name]) != 64
            or evidence[name] == "0" * 64
            for name in required_hashes
        )
        or evidence.get("synthetic_qualification_status") not in {
            "mps_unavailable", "compared_pass", "qualification_mismatch", "nonfinite_output"
        }
    ):
        raise BenchmarkBoundaryError("benchmark provenance evidence is incomplete")
    benchmark_finished_at = datetime.now(timezone.utc).isoformat()
    capability = _effective_capability(capability, assessments)
    policy_bundle: BackendPolicyBundle = choose_backend(assessments, config)
    _, source_digests, _ = _source_manifest_metadata()
    provenance = capability_provenance(
        capability,
        {
            "git_commit": _git_commit(),
            "database_sha256": expected_db_sha,
            "config_sha256": frozen_config_sha256(config),
            "device": str(capability.device_name),
            "runtime": _runtime_label(),
            "source": "exp421-train-only-benchmark",
            "fresh_no_clobber_attempt": True,
            "train_only": True,
            "benchmark_started_at": benchmark_started_at,
            "benchmark_finished_at": benchmark_finished_at,
            "process_id": os.getpid(),
            "host": f"{platform.system()} {platform.machine()}",
            **dict(evidence),
        },
    )
    state = "passed" if all(
        reason in {"eligible", "batch_size_one_cpu_mandatory"}
        for reason in policy_bundle.reasons.values()
    ) else "failed"
    manifest = build_manifest(
        policy_bundle=policy_bundle,
        cached_candidate_indices=cached_indices,
        provenance=provenance,
        source_digests=source_digests,
        attempt_root=attempt_root,
        state=state,
    )
    return write_manifest(attempt_root / "manifest.json", manifest)


__all__ = [
    "BenchmarkBoundaryError",
    "MAX_BATCH_BYTES",
    "MAX_BENCHMARK_SERIES",
    "run_train_only_benchmark",
]
