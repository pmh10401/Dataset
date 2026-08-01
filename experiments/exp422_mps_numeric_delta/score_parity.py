"""Bounded normal-TRAIN score parity for the Exp422 numeric-delta audit."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter
from types import MappingProxyType

import numpy as np
import torch

from experiments.exp417_causal_multihorizon_shadow.contracts import ProcessedGroup
from experiments.exp417_causal_multihorizon_shadow.inputs import (
    load_raw_train_values_from_connection,
    verified_read_only_connection,
)
from experiments.exp417_causal_multihorizon_shadow.model import (
    fit_final_seed_models,
    train_oof_models,
)
from experiments.exp417_causal_multihorizon_shadow.preprocessing import preprocess_length_groups
from experiments.exp417_causal_multihorizon_shadow.scoring import (
    consensus_candidate,
    seed_decision,
)
from experiments.exp421_mps_batched_acceleration.batch_inference import FrozenModelAdapter
from experiments.exp421_mps_batched_acceleration.calibration import (
    CalibrationCache,
    build_calibration_cache,
    score_query_with_cache,
)
from experiments.exp421_mps_batched_acceleration.perturbations import build_cached_perturbations

from .capability import DeviceCapability
from .config import FROZEN_CONFIG, Exp422Config
from .contracts import StageResult, TensorDelta, TimingEvidence
from .core_parity import compare_tensor, tie_aware_spearman


_FORBIDDEN_DATASET_TOKENS = frozenset(
    {
        "test",
        "eval",
        "evaluation",
        "label",
        "labels",
        "location",
        "locations",
        "metric",
        "metrics",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def validate_dataset_name(value: object) -> str:
    """Return one non-evaluation dataset identifier or fail closed."""
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError("dataset name must be a nonempty exact identifier")
    tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9]+", value)}
    if tokens & _FORBIDDEN_DATASET_TOKENS:
        raise ValueError("dataset name contains a forbidden evaluation token")
    return value


def decide_stage_two(core_stage: StageResult) -> str:
    """Permit score parity only after an explicit successful core stage."""
    if not isinstance(core_stage, StageResult):
        raise ValueError("core_stage must be a StageResult")
    return "run" if core_stage.state == "passed" else "skipped"


def _readonly(values: np.ndarray, dtype: np.dtype | str) -> np.ndarray:
    copied = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(copied.tobytes(order="C"), dtype=copied.dtype).reshape(copied.shape)


def _raw_waveform_sha256(
    dataset: str, length: int, raw_by_index: Mapping[int, np.ndarray]
) -> str:
    digest = sha256(b"exp422.normal-train.raw-waveform.v1\x00")
    digest.update(dataset.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(length.to_bytes(8, "little", signed=False))
    digest.update(len(raw_by_index).to_bytes(8, "little", signed=False))
    for index in sorted(raw_by_index):
        values = np.ascontiguousarray(raw_by_index[index], dtype="<f8")
        digest.update(index.to_bytes(8, "little", signed=False))
        digest.update(values.size.to_bytes(8, "little", signed=False))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainSample:
    """One exact-length normal-TRAIN fit population and bounded device sample."""

    dataset: str
    length: int
    train_count: int
    raw_waveform_sha256: str
    bounded_series_count: int
    group: ProcessedGroup
    indices: tuple[int, ...]
    channels: np.ndarray
    raw_by_index: Mapping[int, np.ndarray]
    max_series: int
    max_batch_bytes: int

    def __post_init__(self) -> None:
        validate_dataset_name(self.dataset)
        if type(self.length) is not int or self.length < 1:
            raise ValueError("sample length must be a positive integer")
        if not isinstance(self.group, ProcessedGroup) or self.group.group.length != self.length:
            raise ValueError("sample group must preserve the exact length")
        if self.group.status != "candidate_eligible":
            raise ValueError("sample group must be candidate eligible")
        if type(self.train_count) is not int or self.train_count != len(self.group.group.series):
            raise ValueError("train_count must bind the complete exact-length TRAIN group")
        if not isinstance(self.raw_waveform_sha256, str) or _SHA256_RE.fullmatch(
            self.raw_waveform_sha256
        ) is None:
            raise ValueError("raw_waveform_sha256 must be a lowercase SHA-256 digest")
        if type(self.max_series) is not int or self.max_series < 1:
            raise ValueError("max_series must be a positive integer")
        if type(self.max_batch_bytes) is not int or self.max_batch_bytes < 1:
            raise ValueError("max_batch_bytes must be a positive integer")
        indices = tuple(self.indices)
        if not indices or indices != tuple(sorted(indices)) or len(indices) != len(set(indices)):
            raise ValueError("sample indices must be sorted, unique, and nonempty")
        if type(self.bounded_series_count) is not int or self.bounded_series_count != len(indices):
            raise ValueError("bounded_series_count must match sample indices")
        if self.bounded_series_count > self.max_series:
            raise ValueError("sample exceeds the bounded series limit")
        if (
            not isinstance(self.channels, np.ndarray)
            or self.channels.dtype != np.dtype("float32")
            or self.channels.shape != (len(indices), 2, self.length)
        ):
            raise ValueError("sample channels must be exact float32 [N,2,L]")
        if self.channels.nbytes > self.max_batch_bytes:
            raise ValueError("sample channels exceed the bounded memory limit")
        if not np.isfinite(self.channels).all():
            raise ValueError("sample channels must be finite")
        if not isinstance(self.raw_by_index, Mapping) or tuple(sorted(self.raw_by_index)) != indices:
            raise ValueError("sample raw values must bind the exact bounded indices")
        raw: dict[int, np.ndarray] = {}
        for index in indices:
            values = self.raw_by_index[index]
            if (
                not isinstance(values, np.ndarray)
                or values.shape != (self.length,)
                or not np.issubdtype(values.dtype, np.number)
                or not np.isfinite(values).all()
            ):
                raise ValueError("sample raw series must be finite and exact length")
            raw[index] = _readonly(values, "<f8")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "channels", _readonly(self.channels, "<f4"))
        object.__setattr__(self, "raw_by_index", MappingProxyType(raw))


@dataclass(frozen=True)
class ScoreBatchReport:
    """Numeric score-path evidence for one Exp422-owned batch policy."""

    batch_size: int
    chunk_sizes: tuple[int, ...]
    tensor_deltas: tuple[TensorDelta, ...]
    cpu_forward_sha256: str
    mps_forward_sha256: str
    cpu_residual_sha256: str
    mps_residual_sha256: str
    cpu_score_sha256: str
    mps_score_sha256: str
    cpu_candidate_coordinates_sha256: str
    mps_candidate_coordinates_sha256: str
    candidate_indices_match: bool
    spearman: float
    qualification_match: bool
    cpu_timing: TimingEvidence
    mps_timing: TimingEvidence
    max_normalized_error: float
    status: str = "passed"
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.batch_size) is not int or self.batch_size not in FROZEN_CONFIG.candidate_batch_sizes:
            raise ValueError("batch_size must be preregistered for Exp422")
        if self.status not in {"passed", "failed", "skipped"}:
            raise ValueError("score batch status must be passed, failed, or skipped")
        if self.status == "passed" and (
            not self.chunk_sizes
            or any(type(size) is not int or not 1 <= size <= self.batch_size for size in self.chunk_sizes)
        ):
            raise ValueError("chunk_sizes must be positive and bounded by the batch policy")
        if self.status == "passed" and (
            not self.tensor_deltas or any(not isinstance(item, TensorDelta) for item in self.tensor_deltas)
        ):
            raise ValueError("tensor_deltas must contain parity evidence")
        for name in (
            "cpu_forward_sha256",
            "mps_forward_sha256",
            "cpu_residual_sha256",
            "mps_residual_sha256",
            "cpu_score_sha256",
            "mps_score_sha256",
            "cpu_candidate_coordinates_sha256",
            "mps_candidate_coordinates_sha256",
        ):
            if not isinstance(getattr(self, name), str) or _SHA256_RE.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if type(self.candidate_indices_match) is not bool or type(self.qualification_match) is not bool:
            raise ValueError("parity flags must be bools")
        if self.status == "passed" and self.fallback_reason is not None:
            raise ValueError("passed score batch cannot have a fallback reason")
        if self.status != "passed" and (
            not isinstance(self.fallback_reason, str) or not self.fallback_reason.strip()
        ):
            raise ValueError("failed or skipped score batch requires a fallback reason")
        if not np.isfinite(self.spearman) or not -1.0 <= self.spearman <= 1.0:
            raise ValueError("spearman must be finite and between -1 and 1")
        if not np.isfinite(self.max_normalized_error) or self.max_normalized_error < 0.0:
            raise ValueError("max_normalized_error must be finite and nonnegative")
        if self.cpu_timing.backend != "cpu" or self.mps_timing.backend != "mps":
            raise ValueError("score timing backends must be CPU then MPS")

    @property
    def speedup(self) -> float:
        return self.cpu_timing.total_seconds / max(self.mps_timing.total_seconds, 1e-12)

    @property
    def passes(self) -> bool:
        return (
            self.status == "passed"
            and self.candidate_indices_match
            and self.qualification_match
            and all(delta.passes for delta in self.tensor_deltas)
            and self.max_normalized_error <= FROZEN_CONFIG.max_normalized_error
            and self.spearman >= FROZEN_CONFIG.min_spearman
            and self.speedup >= FROZEN_CONFIG.min_speedup
        )


@dataclass(frozen=True)
class ScoreStageResult:
    """Complete bounded score-stage evidence or an explicit skip."""

    sample: TrainSample
    reports: tuple[ScoreBatchReport, ...]
    stage: StageResult
    model_state_sha256: str | None = None
    model_stop_sha256: str | None = None
    loss_history_sha256: str | None = None
    oof_residual_sha256: str | None = None
    calibration_sha256: str | None = None
    perturbation_sha256: str | None = None
    cpu_qualification_sha256: str | None = None
    mps_qualification_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample, TrainSample) or not isinstance(self.stage, StageResult):
            raise ValueError("score result requires a TrainSample and StageResult")
        if any(not isinstance(report, ScoreBatchReport) for report in self.reports):
            raise ValueError("reports must contain ScoreBatchReport records")
        expected_batches = FROZEN_CONFIG.candidate_batch_sizes
        actual_batches = tuple(report.batch_size for report in self.reports)
        if self.stage.state != "skipped" and actual_batches != expected_batches:
            raise ValueError("score result requires exact Exp422 batch coverage")
        digests = (
            self.model_state_sha256,
            self.model_stop_sha256,
            self.loss_history_sha256,
            self.oof_residual_sha256,
            self.calibration_sha256,
            self.perturbation_sha256,
            self.cpu_qualification_sha256,
            self.mps_qualification_sha256,
        )
        if self.stage.state == "skipped":
            if self.reports or any(value is not None for value in digests):
                raise ValueError("skipped score stage cannot retain computed evidence")
        elif self.stage.state == "passed" and any(
            not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None for value in digests
        ):
            raise ValueError("completed score stage requires every SHA-256 evidence digest")
        else:
            for value in digests:
                if value is not None and (
                    not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
                ):
                    raise ValueError("score evidence digests must be lowercase SHA-256 values")


def _dataset_names(connection: object) -> tuple[str, ...]:
    try:
        rows = connection.execute("SELECT name FROM datasets ORDER BY name").fetchall()  # type: ignore[attr-defined]
    except Exception as exc:
        raise ValueError("unable to enumerate normal TRAIN datasets") from exc
    names = tuple(validate_dataset_name(row[0]) for row in rows if len(row) == 1)
    if not names or names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise ValueError("dataset population must be sorted, unique, and nonempty")
    return names


def _verify_quick_check(connection: object) -> None:
    try:
        row = connection.execute("PRAGMA quick_check").fetchone()  # type: ignore[attr-defined]
    except Exception as exc:
        raise ValueError("SQLite quick_check failed") from exc
    if row != ("ok",):
        raise ValueError("SQLite quick_check did not return ok")


def select_train_sample(database: Path, config: Exp422Config) -> TrainSample:
    """Select the first deterministic finite exact-length normal-TRAIN group."""
    if type(config) is not Exp422Config:
        raise ValueError("config must be an Exp422Config")
    config.validate()
    database_path = Path(database)
    expected_sha256 = config.exp417_config.database_sha256
    with verified_read_only_connection(database_path, expected_sha256) as connection:
        _verify_quick_check(connection)
        datasets = _dataset_names(connection)
        for dataset in datasets:
            raw = load_raw_train_values_from_connection(
                connection,
                dataset,
                database_sha256=expected_sha256,
                datasets=datasets,
            )
            raw_arrays = {
                item.instance_index: np.asarray(item.values, dtype=np.float64)
                for item in raw.series
            }
            if any(not np.isfinite(values).all() for values in raw_arrays.values()):
                raise ValueError("normal TRAIN series must be finite")
            for group in preprocess_length_groups(raw, config=config.exp417_config):
                if group.status != "candidate_eligible":
                    continue
                group_indices = tuple(item.instance_index for item in group.group.series)
                group_raw = {index: raw_arrays[index] for index in group_indices}
                row_bytes = 2 * group.group.length * np.dtype("float32").itemsize
                if row_bytes > config.max_batch_bytes:
                    raise ValueError("one exact TRAIN series exceeds the batch memory bound")
                bounded_count = min(
                    len(group_indices),
                    config.max_series,
                    config.max_batch_bytes // row_bytes,
                )
                indices = group_indices[:bounded_count]
                channels = np.stack([group.channels[index] for index in indices]).astype(
                    np.float32, copy=True, order="C"
                )
                return TrainSample(
                    dataset=dataset,
                    length=group.group.length,
                    train_count=len(group_indices),
                    raw_waveform_sha256=_raw_waveform_sha256(
                        dataset, group.group.length, group_raw
                    ),
                    bounded_series_count=len(indices),
                    group=group,
                    indices=indices,
                    channels=channels,
                    raw_by_index={index: group_raw[index] for index in indices},
                    max_series=config.max_series,
                    max_batch_bytes=config.max_batch_bytes,
                )
    raise ValueError("no finite exact-length normal TRAIN sample is available")


def _hash_arrays(
    tag: bytes,
    values: Mapping[int, Mapping[int, np.ndarray]],
    *,
    dtype: np.dtype | str = "<f8",
) -> str:
    digest = sha256(tag + b"\x00")
    for seed in sorted(values):
        digest.update(seed.to_bytes(8, "little", signed=False))
        for horizon in sorted(values[seed]):
            array = np.ascontiguousarray(values[seed][horizon], dtype=dtype)
            digest.update(horizon.to_bytes(8, "little", signed=False))
            digest.update(len(array.shape).to_bytes(8, "little", signed=False))
            for size in array.shape:
                digest.update(size.to_bytes(8, "little", signed=False))
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _oof_digest(oof: Mapping[int, Mapping[int, Mapping[int, np.ndarray]]]) -> str:
    digest = sha256(b"exp422.normal-train.oof-residual.v1\x00")
    for seed in sorted(oof):
        digest.update(seed.to_bytes(8, "little", signed=False))
        for index in sorted(oof[seed]):
            digest.update(index.to_bytes(8, "little", signed=False))
            for horizon in sorted(oof[seed][index]):
                values = np.ascontiguousarray(oof[seed][index][horizon], dtype="<f8")
                if not np.isfinite(values).all():
                    raise ValueError("OOF residuals must be finite")
                digest.update(horizon.to_bytes(8, "little", signed=False))
                digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _build_calibration(
    oof: Mapping[int, Mapping[int, Mapping[int, np.ndarray]]],
    config: Exp422Config,
) -> tuple[Mapping[int, Mapping[int, CalibrationCache]], str]:
    caches: dict[int, Mapping[int, CalibrationCache]] = {}
    digest = sha256(b"exp422.normal-train.calibration.v1\x00")
    for seed in config.model_seeds:
        by_horizon: dict[int, CalibrationCache] = {}
        indices = tuple(sorted(oof[seed]))
        horizons = tuple(sorted(oof[seed][indices[0]]))
        for horizon in horizons:
            cache = build_calibration_cache(
                tuple(oof[seed][index][horizon] for index in indices), horizon=horizon
            )
            by_horizon[horizon] = cache
            digest.update(seed.to_bytes(8, "little", signed=False))
            digest.update(cache.digest.encode("ascii"))
        caches[seed] = MappingProxyType(by_horizon)
    return MappingProxyType(caches), digest.hexdigest()


def _timed_adapter_predict(
    adapter: FrozenModelAdapter, channels: np.ndarray
) -> tuple[Mapping[int, np.ndarray], Mapping[int, np.ndarray], TimingEvidence]:
    """Time the low-level frozen adapter including transfers and MPS synchronization."""
    values = np.array(channels, dtype=np.float32, copy=True, order="C")
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("score channels must be finite float32 [N,2,L]")
    device = adapter.device
    network = adapter._network
    network.eval()
    transfer_start = perf_counter()
    inputs = torch.from_numpy(values).to(device)
    transfer_in = perf_counter() - transfer_start
    if inputs.device != device:
        raise ValueError("score input silently changed device")
    compute_start = perf_counter()
    with torch.no_grad():
        predictions = network(inputs)
        packed = torch.cat([predictions[horizon].reshape(-1) for horizon in adapter.model.horizons])
    compute = perf_counter() - compute_start
    sync_start = perf_counter()
    if device.type == "mps":
        torch.mps.synchronize()
    synchronize = perf_counter() - sync_start
    output_start = perf_counter()
    packed_cpu = packed.detach().cpu().numpy()
    transfer = transfer_in + perf_counter() - output_start
    if not np.isfinite(packed_cpu).all():
        raise ValueError("score forward outputs must be finite")
    predictions_by_horizon: dict[int, np.ndarray] = {}
    residuals: dict[int, np.ndarray] = {}
    offset = 0
    for horizon in adapter.model.horizons:
        count = values.shape[0] * (values.shape[2] - horizon)
        prediction = packed_cpu[offset : offset + count].reshape(values.shape[0], values.shape[2] - horizon)
        offset += count
        predictions_by_horizon[horizon] = _readonly(prediction, "<f4")
        residual = np.abs(prediction - values[:, 0, horizon:]).astype(np.float64, copy=False)
        if not np.isfinite(residual).all():
            raise ValueError("score residuals must be finite")
        residuals[horizon] = _readonly(residual, "<f8")
    if offset != packed_cpu.size:
        raise ValueError("score forward output shape is invalid")
    return (
        MappingProxyType(predictions_by_horizon),
        MappingProxyType(residuals),
        TimingEvidence(device.type, transfer, compute, synchronize, transfer + compute + synchronize),
    )


def _predict_for_policy(
    adapters: Mapping[int, FrozenModelAdapter],
    channels: np.ndarray,
    *,
    batch_size: int,
    max_batch_bytes: int,
) -> tuple[
    Mapping[int, Mapping[int, np.ndarray]],
    Mapping[int, Mapping[int, np.ndarray]],
    TimingEvidence,
    tuple[int, ...],
]:
    """Predict every bounded row in deterministic chunks without padding or replay."""
    if channels.nbytes > max_batch_bytes:
        raise MemoryError("fixed_batch_memory_limit")
    chunks = tuple(min(batch_size, len(channels) - start) for start in range(0, len(channels), batch_size))
    backend = next(iter(adapters.values())).device.type
    totals = [0.0, 0.0, 0.0]
    predictions_by_seed: dict[int, Mapping[int, np.ndarray]] = {}
    residuals_by_seed: dict[int, Mapping[int, np.ndarray]] = {}
    for seed in sorted(adapters):
        prediction_pieces: dict[int, list[np.ndarray]] = {}
        residual_pieces: dict[int, list[np.ndarray]] = {}
        for start in range(0, len(channels), batch_size):
            predicted, residuals, timing = _timed_adapter_predict(
                adapters[seed], channels[start : start + batch_size]
            )
            for horizon, values in predicted.items():
                prediction_pieces.setdefault(horizon, []).append(values)
            for horizon, values in residuals.items():
                residual_pieces.setdefault(horizon, []).append(values)
            totals[0] += timing.transfer_seconds
            totals[1] += timing.compute_seconds
            totals[2] += timing.synchronize_seconds
        predictions_by_seed[seed] = MappingProxyType(
            {
                horizon: _readonly(np.concatenate(parts, axis=0), "<f4")
                for horizon, parts in sorted(prediction_pieces.items())
            }
        )
        residuals_by_seed[seed] = MappingProxyType(
            {
                horizon: _readonly(np.concatenate(parts, axis=0), "<f8")
                for horizon, parts in sorted(residual_pieces.items())
            }
        )
    timing = TimingEvidence(backend, totals[0], totals[1], totals[2], sum(totals))
    return MappingProxyType(predictions_by_seed), MappingProxyType(residuals_by_seed), timing, chunks


def _coordinate_digest(dataset: str, length: int, indices: tuple[int, ...]) -> str:
    digest = sha256(b"exp422.normal-train.candidate-coordinates.v1\x00")
    digest.update(dataset.encode("utf-8") + b"\x00")
    digest.update(length.to_bytes(8, "little", signed=False))
    for index in indices:
        digest.update(index.to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def _score_outputs(
    outputs: Mapping[int, Mapping[int, np.ndarray]],
    caches: Mapping[int, Mapping[int, CalibrationCache]],
    indices: tuple[int, ...],
    config: Exp422Config,
) -> tuple[np.ndarray, tuple[int, ...], str]:
    scored_values: list[float] = []
    selected: list[int] = []
    digest = sha256(b"exp422.normal-train.horizon-score.v1\x00")
    for row, index in enumerate(indices):
        decisions = []
        for seed in config.model_seeds:
            scores = {
                horizon: score_query_with_cache(caches[seed][horizon], outputs[seed][horizon][row])
                for horizon in sorted(caches[seed])
            }
            decision = seed_decision(scores, seed=seed, config=config.exp417_config)
            decisions.append(decision)
            for horizon in sorted(scores):
                score = scores[horizon]
                scored_values.extend((score.score, score.p_value, score.persistence))
                digest.update(np.asarray([score.score, score.p_value, score.persistence], dtype="<f8").tobytes())
        accepted = consensus_candidate(decisions)
        digest.update(index.to_bytes(8, "little", signed=False))
        digest.update(b"\x01" if accepted else b"\x00")
        if accepted:
            selected.append(index)
    values = np.asarray(scored_values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("horizon scores must be finite")
    return values, tuple(selected), digest.hexdigest()


class _ScoreBackendError(RuntimeError):
    """A backend-specific score failure that must become a structured row."""


def _qualification_evidence(
    sample: TrainSample,
    cpu_adapters: Mapping[int, FrozenModelAdapter],
    mps_adapters: Mapping[int, FrozenModelAdapter],
    caches: Mapping[int, Mapping[int, CalibrationCache]],
    config: Exp422Config,
    *,
    batch_size: int,
) -> tuple[str, str, str, bool]:
    if type(batch_size) is not int or batch_size not in config.candidate_batch_sizes:
        raise ValueError("qualification batch size must be preregistered")
    perturbation_names = (
        "level_shift",
        "gradual_drift",
        "amplitude_scaling",
        "period_change",
        "secondary_spike",
    )
    perturb_digest = sha256(b"exp422.normal-train.perturbation.v1\x00")
    arrays: list[np.ndarray] = []
    for row in range(batch_size):
        index = sample.indices[row % len(sample.indices)]
        seed = config.perturbation_seeds[row % len(config.perturbation_seeds)]
        repaired = next(
            item.values for item in sample.group.group.series if item.instance_index == index
        )
        cached = build_cached_perturbations(
            "TRAIN_" + sha256(sample.dataset.encode("utf-8")).hexdigest()[:16],
            index,
            np.asarray(repaired, dtype=np.float64),
            max(float(sample.group.raw_scale), 1e-6),
            seed,
        )
        perturb_digest.update(cached.digest.encode("ascii"))
        arrays.append(np.asarray(cached[perturbation_names[row % len(perturbation_names)]], dtype=np.float64))
    raw_batch = np.stack(arrays)
    differences = np.diff(raw_batch, axis=1, prepend=raw_batch[:, :1])
    channels = np.stack(
        (
            (raw_batch - sample.group.raw_center) / sample.group.raw_scale,
            (differences - sample.group.difference_center) / sample.group.difference_scale,
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    if not np.isfinite(channels).all() or channels.nbytes > config.max_batch_bytes:
        raise ValueError("qualification channels must be finite and bounded")
    try:
        _cpu_forward, cpu, _cpu_timing, _ = _predict_for_policy(
            cpu_adapters, channels, batch_size=batch_size, max_batch_bytes=config.max_batch_bytes
        )
    except (MemoryError, OSError, RuntimeError, ValueError) as exc:
        raise _ScoreBackendError("cpu_qualification_runtime_error") from exc
    try:
        _mps_forward, mps, _mps_timing, _ = _predict_for_policy(
            mps_adapters, channels, batch_size=batch_size, max_batch_bytes=config.max_batch_bytes
        )
    except (MemoryError, OSError, RuntimeError, ValueError) as exc:
        raise _ScoreBackendError("mps_qualification_runtime_error") from exc
    coordinates = tuple(range(len(channels)))
    cpu_scores, cpu_selected, cpu_digest = _score_outputs(cpu, caches, coordinates, config)
    mps_scores, mps_selected, mps_digest = _score_outputs(mps, caches, coordinates, config)
    delta = compare_tensor(cpu_scores, mps_scores, name="synthetic_qualification_scores")
    matches = (
        cpu_selected == mps_selected
        and cpu_digest == mps_digest
        and delta.passes
        and tie_aware_spearman(cpu_scores, mps_scores) >= config.min_spearman
    )
    return perturb_digest.hexdigest(), cpu_digest, mps_digest, matches


def _empty_score_report(
    batch_size: int,
    reason: str,
    *,
    cpu_timing: TimingEvidence | None = None,
) -> ScoreBatchReport:
    """Record a policy that could not execute at its exact registered size."""
    zero_cpu = cpu_timing or TimingEvidence("cpu", 0.0, 0.0, 0.0, 0.0)
    zero_mps = TimingEvidence("mps", 0.0, 0.0, 0.0, 0.0)
    digest = sha256(f"exp422.failed-policy.{batch_size}.{reason}".encode("ascii")).hexdigest()
    return ScoreBatchReport(
        batch_size=batch_size,
        chunk_sizes=(),
        tensor_deltas=(),
        cpu_forward_sha256=digest,
        mps_forward_sha256=digest,
        cpu_residual_sha256=digest,
        mps_residual_sha256=digest,
        cpu_score_sha256=digest,
        mps_score_sha256=digest,
        cpu_candidate_coordinates_sha256=digest,
        mps_candidate_coordinates_sha256=digest,
        candidate_indices_match=False,
        spearman=0.0,
        qualification_match=False,
        cpu_timing=zero_cpu,
        mps_timing=zero_mps,
        max_normalized_error=1.0,
        status="failed",
        fallback_reason=reason,
    )


def run_score_parity(
    sample: TrainSample, capability: DeviceCapability, config: Exp422Config
) -> ScoreStageResult:
    """Fit CPU reference once and compare bounded CPU/MPS score paths."""
    if not isinstance(sample, TrainSample):
        raise ValueError("sample must be a TrainSample")
    if not isinstance(capability, DeviceCapability):
        raise ValueError("capability must be a DeviceCapability")
    if type(config) is not Exp422Config:
        raise ValueError("config must be an Exp422Config")
    config.validate()
    if sample.max_series != config.max_series or sample.max_batch_bytes != config.max_batch_bytes:
        raise ValueError("sample resource bounds must bind to Exp422 config")
    if not capability.mps_available:
        return ScoreStageResult(sample, (), StageResult.skipped("mps_unavailable"))
    torch.set_num_threads(1)
    models = fit_final_seed_models(
        sample.group, device=torch.device("cpu"), config=config.exp417_config
    )
    model_by_seed = {model.seed: model for model in models}
    if set(model_by_seed) != set(config.model_seeds):
        raise ValueError("final CPU fit omitted a frozen model seed")
    oof_bundle = train_oof_models(
        sample.group, device=torch.device("cpu"), config=config.exp417_config
    )
    oof = oof_bundle.oof_residuals
    if set(oof) != set(config.model_seeds):
        raise ValueError("CPU OOF fit omitted a frozen model seed")
    caches, calibration_sha256 = _build_calibration(oof, config)
    oof_sha256 = _oof_digest(oof)
    model_state_sha256 = sha256(
        b"exp422.cpu-model-state.v1\x00"
        + b"".join(model_by_seed[seed].model_sha256.encode("ascii") for seed in config.model_seeds)
    ).hexdigest()
    model_stop_sha256 = sha256(
        json.dumps([(seed, model_by_seed[seed].stop_epoch) for seed in config.model_seeds], separators=(",", ":")).encode("ascii")
    ).hexdigest()
    loss_history_sha256 = sha256(
        json.dumps(
            [
                (seed, model_by_seed[seed].training_losses, model_by_seed[seed].validation_losses)
                for seed in config.model_seeds
            ],
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    cpu_adapters = MappingProxyType(
        {
            seed: FrozenModelAdapter.from_trained_seed(model_by_seed[seed], device=torch.device("cpu"))
            for seed in config.model_seeds
        }
    )
    try:
        mps_adapters = MappingProxyType(
            {
                seed: FrozenModelAdapter.from_trained_seed(model_by_seed[seed], device=torch.device("mps"))
                for seed in config.model_seeds
            }
        )
    except (MemoryError, OSError, RuntimeError, ValueError):
        return ScoreStageResult(sample, (), StageResult.skipped("mps_runtime_error"))
    reports: list[ScoreBatchReport] = []
    perturbation_digests: list[str] = []
    cpu_qualification_digests: list[str] = []
    mps_qualification_digests: list[str] = []
    for batch_size in config.candidate_batch_sizes:
        if sample.bounded_series_count < batch_size:
            reports.append(
                _empty_score_report(batch_size, "fixed_batch_sample_too_small")
            )
            continue
        batch_channels = sample.channels[:batch_size]
        batch_indices = sample.indices[:batch_size]
        try:
            cpu_forward, cpu_outputs, cpu_timing, cpu_chunks = _predict_for_policy(
                cpu_adapters,
                batch_channels,
                batch_size=batch_size,
                max_batch_bytes=config.max_batch_bytes,
            )
        except (MemoryError, OSError, RuntimeError, ValueError):
            reports.append(_empty_score_report(batch_size, "cpu_runtime_error"))
            continue
        if cpu_chunks != (batch_size,):
            reports.append(_empty_score_report(batch_size, "cpu_exact_batch_policy_error", cpu_timing=cpu_timing))
            continue
        try:
            mps_forward, mps_outputs, mps_timing, mps_chunks = _predict_for_policy(
                mps_adapters,
                batch_channels,
                batch_size=batch_size,
                max_batch_bytes=config.max_batch_bytes,
            )
        except (MemoryError, OSError, RuntimeError, ValueError):
            reports.append(_empty_score_report(batch_size, "mps_runtime_error", cpu_timing=cpu_timing))
            continue
        if mps_chunks != (batch_size,):
            reports.append(_empty_score_report(batch_size, "mps_exact_batch_policy_error", cpu_timing=cpu_timing))
            continue
        try:
            perturbation_sha256, cpu_qualification_sha256, mps_qualification_sha256, qualification_match = _qualification_evidence(
                sample,
                cpu_adapters,
                mps_adapters,
                caches,
                config,
                batch_size=batch_size,
            )
        except _ScoreBackendError as exc:
            reports.append(_empty_score_report(batch_size, str(exc), cpu_timing=cpu_timing))
            continue
        except (MemoryError, OSError, RuntimeError, ValueError):
            reports.append(_empty_score_report(batch_size, "qualification_runtime_error", cpu_timing=cpu_timing))
            continue
        perturbation_digests.append(perturbation_sha256)
        cpu_qualification_digests.append(cpu_qualification_sha256)
        mps_qualification_digests.append(mps_qualification_sha256)
        deltas = []
        for seed in config.model_seeds:
            for horizon in sorted(cpu_forward[seed]):
                deltas.append(
                    compare_tensor(
                        cpu_forward[seed][horizon],
                        mps_forward[seed][horizon],
                        name=f"batch_{batch_size}_seed_{seed}_horizon_{horizon}_forward",
                    )
                )
                deltas.append(
                    compare_tensor(
                        cpu_outputs[seed][horizon],
                        mps_outputs[seed][horizon],
                        name=f"batch_{batch_size}_seed_{seed}_horizon_{horizon}_residual",
                    )
                )
        cpu_scores, cpu_candidates, cpu_score_sha256 = _score_outputs(
            cpu_outputs, caches, batch_indices, config
        )
        mps_scores, mps_candidates, mps_score_sha256 = _score_outputs(
            mps_outputs, caches, batch_indices, config
        )
        deltas.append(
            compare_tensor(cpu_scores, mps_scores, name=f"batch_{batch_size}_horizon_scores")
        )
        reports.append(
            ScoreBatchReport(
                batch_size=batch_size,
                chunk_sizes=cpu_chunks,
                tensor_deltas=tuple(deltas),
                cpu_forward_sha256=_hash_arrays(
                    b"exp422.forward-output.v1", cpu_forward, dtype="<f4"
                ),
                mps_forward_sha256=_hash_arrays(
                    b"exp422.forward-output.v1", mps_forward, dtype="<f4"
                ),
                cpu_residual_sha256=_hash_arrays(
                    b"exp422.residual-output.v1", cpu_outputs, dtype="<f8"
                ),
                mps_residual_sha256=_hash_arrays(
                    b"exp422.residual-output.v1", mps_outputs, dtype="<f8"
                ),
                cpu_score_sha256=cpu_score_sha256,
                mps_score_sha256=mps_score_sha256,
                cpu_candidate_coordinates_sha256=_coordinate_digest(
                    sample.dataset, sample.length, cpu_candidates
                ),
                mps_candidate_coordinates_sha256=_coordinate_digest(
                    sample.dataset, sample.length, mps_candidates
                ),
                candidate_indices_match=cpu_candidates == mps_candidates,
                spearman=tie_aware_spearman(cpu_scores, mps_scores),
                qualification_match=qualification_match
                and cpu_qualification_sha256 == mps_qualification_sha256,
                cpu_timing=cpu_timing,
                mps_timing=mps_timing,
                max_normalized_error=max(delta.max_normalized_error for delta in deltas),
            )
        )
    stage = StageResult.passed() if all(report.passes for report in reports) else StageResult.failed("score_parity_gate_failed")
    digest_qual_cpu = sha256("".join(cpu_qualification_digests).encode("ascii")).hexdigest()
    digest_qual_mps = sha256("".join(mps_qualification_digests).encode("ascii")).hexdigest()
    digest_perturb = sha256("".join(perturbation_digests).encode("ascii")).hexdigest()
    return ScoreStageResult(
        sample=sample,
        reports=tuple(reports),
        stage=stage,
        model_state_sha256=model_state_sha256,
        model_stop_sha256=model_stop_sha256,
        loss_history_sha256=loss_history_sha256,
        oof_residual_sha256=oof_sha256,
        calibration_sha256=calibration_sha256,
        perturbation_sha256=digest_perturb,
        cpu_qualification_sha256=digest_qual_cpu,
        mps_qualification_sha256=digest_qual_mps,
    )


__all__ = (
    "ScoreBatchReport",
    "ScoreStageResult",
    "TrainSample",
    "decide_stage_two",
    "run_score_parity",
    "select_train_sample",
    "validate_dataset_name",
)
