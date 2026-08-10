"""Closed CPU/MLX/Numba parity and speed gate for Exp444.

Task 4 owns the scientific scoring rule.  This module deliberately injects a
different implementation only at Exp440's dense RMS-distance seam; all
calibration, thresholds, masks, candidate construction, and review lanes are
still produced by :mod:`stability_antigravity`.

MLX is treated as unavailable when the installed Apple device cannot execute
the required float64 operation.  Falling back is intentional: silently using
float32 would change the frozen algorithm.  The real benchmark is a single
process and runs the complete stability-view stage for fixed inputs, never a
synthetic distance loop alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import platform
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np

from experiments.exp439_scale_aware_reconstruction_repair_antigravity.scoring_antigravity import (
    score_channel_cpu_antigravity,
)

from .config_antigravity import FROZEN_CONFIG, cpu_worker_count_antigravity
from .contracts_antigravity import BackendDecisionAntigravity, CandidateInputAntigravity
from .stability_antigravity import score_stability_candidate_antigravity


ACCELERATION_TOLERANCE_ANTIGRAVITY = FROZEN_CONFIG.acceleration_tolerance
ACCELERATION_SPEEDUP_MIN_ANTIGRAVITY = FROZEN_CONFIG.acceleration_speedup_min
# These are part of the preregistered benchmark selection.  No unmeasured
# warmup is permitted: the first (and only) full-stage repetition includes
# Numba/MLX compilation, transfer, synchronization, and result materializing
# costs.  Recording zero warmups makes the gate conservative and reproducible
# for this expensive frozen Task 4 stage.
BENCHMARK_WARMUPS_ANTIGRAVITY = 0
BENCHMARK_REPETITIONS_ANTIGRAVITY = 1
BENCHMARK_SELECTION_POLICY_ANTIGRAVITY = "size-ordered lower median; first coordinate on ties"


class BackendUnavailableAntigravity(ValueError):
    """Raised when an optional backend cannot satisfy the frozen contract."""


@dataclass(frozen=True)
class BackendRunAntigravity:
    """One complete-stage benchmark/parity result.

    ``seconds`` is the elapsed time for the fixed warmup+timed repetitions.
    For an unavailable backend it is zero and ``error`` carries the reason.
    ``lanes`` contains ``(dataset, instance_index, lane)`` for each selected
    candidate in preregistered input order.
    """

    backend: str
    seconds: float
    lanes: tuple[tuple[str, int, str], ...] = ()
    max_diff: float | None = 0.0
    finite: bool = True
    synchronized: bool = True
    masks_match: bool = True
    candidate_presence_match: bool = True
    survival_counts_match: bool = True
    lane_coordinates_match: bool = True
    available: bool = True
    device: str | None = "CPU"
    version: str | None = ""
    compile_warmup_included: bool = True
    transfer_included: bool = True
    materialization_included: bool = True
    process_count: int = 1
    warmups: int = BENCHMARK_WARMUPS_ANTIGRAVITY
    repetitions: int = BENCHMARK_REPETITIONS_ANTIGRAVITY
    inner_threads: int = 1
    selection: tuple[Mapping[str, object], ...] = ()
    error: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.backend not in {"cpu", "mlx", "numba"}:
            raise ValueError("backend is invalid")
        if type(self.seconds) not in (int, float) or not math.isfinite(float(self.seconds)) or self.seconds < 0.0:
            raise ValueError("benchmark seconds must be finite and nonnegative")
        lanes = tuple(self.lanes)
        if any(
            not isinstance(item, Sequence)
            or len(item) != 3
            or type(item[0]) is not str
            or type(item[1]) is not int
            or type(item[2]) is not str
            for item in lanes
        ):
            raise ValueError("benchmark lanes are invalid")
        if self.max_diff is not None and (
            type(self.max_diff) not in (int, float) or not math.isfinite(float(self.max_diff)) or self.max_diff < 0.0
        ):
            raise ValueError("max_diff must be finite, nonnegative, or None")
        for name in (
            "finite",
            "synchronized",
            "masks_match",
            "candidate_presence_match",
            "survival_counts_match",
            "lane_coordinates_match",
            "available",
            "compile_warmup_included",
            "transfer_included",
            "materialization_included",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool")
        for name in ("process_count", "warmups", "repetitions", "inner_threads"):
            value = getattr(self, name)
            minimum = 0 if name == "warmups" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be a valid exact integer")
        if self.backend == "mlx" and self.available and self.process_count != 1:
            raise ValueError("MLX benchmark must run in exactly one process")
        if self.error is not None and (type(self.error) is not str or not self.error):
            raise ValueError("backend error must be a non-empty string")
        copied_selection = tuple(dict(item) for item in self.selection)
        copied_details = dict(self.details)
        object.__setattr__(self, "lanes", lanes)
        object.__setattr__(self, "selection", copied_selection)
        object.__setattr__(self, "details", copied_details)
        object.__setattr__(self, "seconds", float(self.seconds))
        if self.max_diff is not None:
            object.__setattr__(self, "max_diff", float(self.max_diff))

    @property
    def max_abs_scalar_diff(self) -> float | None:
        return self.max_diff

    @property
    def max_abs_error(self) -> float | None:
        return self.max_diff

    @property
    def synchronization_included(self) -> bool:
        return self.synchronized

    @property
    def result_materialization_included(self) -> bool:
        return self.materialization_included

    @property
    def transfer_device_included(self) -> bool:
        return self.transfer_included

    @property
    def passed(self) -> bool:
        """Whether this run can be considered by the accelerated selector."""
        return bool(
            self.available
            and self.finite
            and self.synchronized
            and self.masks_match
            and self.candidate_presence_match
            and self.survival_counts_match
            and self.lane_coordinates_match
            and self.compile_warmup_included
            and self.transfer_included
            and self.materialization_included
            and self.process_count == 1
            and self.max_diff is not None
            and self.max_diff <= ACCELERATION_TOLERANCE_ANTIGRAVITY
            and self.device is not None
            and bool(self.device)
            and self.version is not None
            and bool(self.version)
            and self.seconds > 0.0
        )

    def to_payload(self, *, cpu_seconds: float | None = None) -> dict[str, object]:
        speed_ratio = None
        if cpu_seconds is not None and self.seconds > 0.0 and math.isfinite(float(cpu_seconds)):
            speed_ratio = float(cpu_seconds) / self.seconds
        return {
            "backend": self.backend,
            "available": self.available,
            "error": self.error,
            "seconds": self.seconds,
            "speed_ratio_vs_cpu": speed_ratio,
            "warmups": self.warmups,
            "repetitions": self.repetitions,
            "compile_warmup_included": self.compile_warmup_included,
            "transfer_included": self.transfer_included,
            "synchronization_included": self.synchronized,
            "result_materialization_included": self.materialization_included,
            "finite_outputs": self.finite,
            "max_abs_scalar_diff": self.max_diff,
            "masks_match": self.masks_match,
            "candidate_presence_match": self.candidate_presence_match,
            "survival_counts_match": self.survival_counts_match,
            "lane_coordinates_match": self.lane_coordinates_match,
            "lanes": [list(item) for item in self.lanes],
            "device": self.device,
            "version": self.version,
            "process_count": self.process_count,
            "inner_threads": self.inner_threads,
            "selection": [dict(item) for item in self.selection],
            "passed_parity": self.passed,
            "details": dict(self.details),
        }

    @classmethod
    def unavailable(
        cls,
        backend: str,
        reason: str,
        *,
        version: str | None = None,
        device: str | None = None,
        selection: tuple[Mapping[str, object], ...] = (),
    ) -> "BackendRunAntigravity":
        return cls(
            backend=backend,
            seconds=0.0,
            lanes=(),
            max_diff=None,
            finite=False,
            synchronized=False,
            masks_match=False,
            candidate_presence_match=False,
            survival_counts_match=False,
            lane_coordinates_match=False,
            available=False,
            device=device,
            version=version,
            compile_warmup_included=False,
            transfer_included=False,
            materialization_included=False,
            selection=selection,
            error=reason,
        )


def _read_field(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _run_backend_name(value: object) -> str:
    name = _read_field(value, "backend", _read_field(value, "name", ""))
    return name if type(name) is str else ""


def _run_seconds(value: object) -> float:
    raw = _read_field(value, "seconds", _read_field(value, "elapsed_seconds", None))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("nan")


def _run_passes(value: object, *, cpu_seconds: float, cpu_lanes: object) -> tuple[bool, str]:
    backend = _run_backend_name(value)
    reasons: list[str] = []
    available = _read_field(value, "available", True)
    if available is not True:
        reasons.append("unavailable")
    if _read_field(value, "finite", _read_field(value, "finite_outputs", True)) is not True:
        reasons.append("nonfinite_output")
    if _read_field(value, "synchronized", _read_field(value, "synchronization_included", _read_field(value, "sync_included", True))) is not True:
        reasons.append("synchronization_missing")
    if _read_field(value, "transfer_included", _read_field(value, "transfer_device_included", _read_field(value, "device_transfer_included", True))) is not True:
        reasons.append("device_transfer_missing")
    if _read_field(value, "compile_warmup_included", True) is not True:
        reasons.append("compile_warmup_policy_missing")
    if _read_field(value, "materialization_included", _read_field(value, "result_materialization_included", _read_field(value, "materialization", True))) is not True:
        reasons.append("result_materialization_missing")
    for field_name, aliases, label in (
        ("masks_match", ("mask_match",), "mask"),
        ("candidate_presence_match", ("candidate_presence",), "candidate_presence"),
        ("survival_counts_match", ("survival_counts",), "survival_count"),
        ("lane_coordinates_match", ("lane_match", "lane_identity_match"), "lane"),
    ):
        flag = _read_field(value, field_name, None)
        if flag is None:
            for alias in aliases:
                flag = _read_field(value, alias, None)
                if flag is not None:
                    break
        if flag is False:
            reasons.append(f"{label}_mismatch")
    process_count = _read_field(value, "process_count", 1)
    if backend == "mlx" and process_count != 1:
        reasons.append("mlx_process_pool_forbidden")
    raw_diff = _read_field(
        value,
        "max_diff",
        _read_field(
            value,
            "max_abs_scalar_diff",
            _read_field(value, "max_abs_diff", _read_field(value, "max_abs_error", None)),
        ),
    )
    try:
        max_diff = float(raw_diff)
    except (TypeError, ValueError):
        max_diff = float("inf")
    if not math.isfinite(max_diff) or max_diff > ACCELERATION_TOLERANCE_ANTIGRAVITY:
        reasons.append("scalar_tolerance_exceeded")
    seconds = _run_seconds(value)
    if not math.isfinite(seconds) or seconds <= 0.0:
        reasons.append("invalid_timing")
    speed_ratio = cpu_seconds / seconds if math.isfinite(cpu_seconds) and seconds > 0.0 else float("nan")
    if backend != "cpu" and (not math.isfinite(speed_ratio) or speed_ratio < ACCELERATION_SPEEDUP_MIN_ANTIGRAVITY):
        reasons.append("speed_ratio_below_1.1")
    lanes = _read_field(value, "lanes", ())
    if backend != "cpu" and lanes != cpu_lanes:
        reasons.append("lane_coordinates_mismatch")
    version = _read_field(value, "version", None)
    device = _read_field(value, "device", None)
    if backend != "cpu" and (not version or not device):
        reasons.append("device_provenance_missing")
    return not reasons, ";".join(reasons)


def choose_backend_antigravity(
    cpu: object,
    mlx: object | None = None,
    numba: object | None = None,
) -> BackendDecisionAntigravity:
    """Select the fastest parity-passing backend, or retain CPU.

    The function accepts :class:`BackendRunAntigravity` and deliberately also
    accepts mapping/duck-typed evidence so review tests can exercise the gate
    without importing optional packages.
    """
    cpu_seconds = _run_seconds(cpu)
    cpu_lanes = _read_field(cpu, "lanes", ())
    candidates = tuple(item for item in (mlx, numba) if item is not None)
    evidence: list[dict[str, object]] = []
    passing: list[tuple[float, str, object]] = []
    fallback_reasons: list[str] = []
    for item in candidates:
        name = _run_backend_name(item)
        if not name:
            name = "backend"
        passed, reason = _run_passes(item, cpu_seconds=cpu_seconds, cpu_lanes=cpu_lanes)
        seconds = _run_seconds(item)
        ratio = cpu_seconds / seconds if math.isfinite(cpu_seconds) and seconds > 0 else None
        if isinstance(item, BackendRunAntigravity):
            payload = item.to_payload(cpu_seconds=cpu_seconds)
        elif isinstance(item, Mapping):
            payload = dict(item)
        else:
            payload = {
                key: _read_field(item, key)
                for key in (
                    "backend", "available", "seconds", "lanes", "max_diff", "finite",
                    "synchronized", "masks_match", "candidate_presence_match",
                    "survival_counts_match", "lane_coordinates_match", "device", "version",
                )
            }
        payload.update({"speed_ratio_vs_cpu": ratio, "passed_gate": passed, "gate_reason": reason or None})
        evidence.append(payload)
        if passed:
            passing.append((float(ratio), name, item))
        else:
            fallback_reasons.append(f"{name}:{reason or 'gate_failed'}")
    if passing:
        # Fastest wins; lexical tie break is deterministic and prefers MLX.
        _, selected_name, _ = max(passing, key=lambda value: (value[0], "z" if value[1] == "mlx" else value[1]))
        return BackendDecisionAntigravity(selected_name, 1, tuple(evidence), None)
    reason = ";".join(fallback_reasons) if fallback_reasons else "no_accelerated_backend_evidence"
    return BackendDecisionAntigravity("cpu", cpu_worker_count_antigravity(FROZEN_CONFIG.logical_cpu_count), tuple(evidence), reason)


def _dense_inputs_antigravity(query: object, references: object, k: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        vector = np.asarray(query, dtype=np.float32)
        matrix = np.asarray(references, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("channel score input is invalid") from error
    if (
        vector.ndim != 1
        or matrix.ndim != 2
        or vector.size == 0
        or matrix.shape[1] != vector.size
        or type(k) is not int
        or k != FROZEN_CONFIG.nearest_k
        or matrix.shape[0] < k
        or not np.isfinite(vector).all()
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("Exp444 acceleration requires finite dense input and k=3")
    return np.ascontiguousarray(vector), np.ascontiguousarray(matrix)


def score_distance_cpu_antigravity(query: object, references: object, k: int) -> float:
    """The unchanged Exp440 CPU reference implementation."""
    return score_channel_cpu_antigravity(query, references, k)


def _load_mlx_antigravity() -> tuple[Any, object, str]:
    try:
        import mlx.core as mx
    except Exception as error:  # pragma: no cover - depends on environment
        raise BackendUnavailableAntigravity(f"mlx_import_unavailable:{type(error).__name__}") from error
    try:
        device = mx.default_device()
        device_type = getattr(device, "type", None)
        if device_type != mx.DeviceType.gpu:
            raise BackendUnavailableAntigravity("mlx_gpu_device_unavailable")
        device_label = str(device)
        # Probe the exact dtype/operator path once.  MLX 0.32 on this Apple
        # device raises here because float64 is unsupported on Metal; that is
        # an honest fail-closed result, not permission to switch to float32.
        with mx.stream(device):
            probe = mx.array(np.asarray([0.0], dtype=np.float64), dtype=mx.float64)
            probe_result = mx.sqrt(mx.mean(probe * probe))
            mx.eval(probe_result)
            _ = probe_result.item()
    except BackendUnavailableAntigravity:
        raise
    except Exception as error:
        raise BackendUnavailableAntigravity(
            f"mlx_float64_unsupported_on_device:{type(error).__name__}"
        ) from error
    version = str(getattr(mx, "__version__", ""))
    if not version:
        raise BackendUnavailableAntigravity("mlx_version_unavailable")
    return mx, device, device_label


def score_distance_mlx_antigravity(query: object, references: object, k: int) -> float:
    """Compute the frozen float64 RMS/nearest-3/median score on Metal."""
    vector, matrix = _dense_inputs_antigravity(query, references, k)
    mx, device, _ = _load_mlx_antigravity()
    try:
        with mx.stream(device):
            query_mx = mx.array(vector.astype(np.float64, copy=False), dtype=mx.float64)
            references_mx = mx.array(matrix.astype(np.float64, copy=False), dtype=mx.float64)
            delta = references_mx - query_mx
            distances = mx.sqrt(mx.mean(delta * delta, axis=1))
            nearest = mx.partition(distances, k - 1)[:k]
            ordered = mx.sort(nearest)
            result = ordered[k // 2]
            # Synchronize all lazy work before materialization/return.  The
            # caller's wall-clock interval therefore includes this barrier.
            mx.eval(distances, nearest, ordered, result)
            scalar = float(result.item())
    except Exception as error:
        if isinstance(error, BackendUnavailableAntigravity):
            raise
        raise BackendUnavailableAntigravity(
            f"mlx_float64_distance_failed:{type(error).__name__}"
        ) from error
    if not math.isfinite(scalar):
        raise ValueError("MLX channel score is nonfinite")
    return scalar


_NUMBA_KERNEL_ANTIGRAVITY: Callable[..., float] | None = None


def _load_numba_kernel_antigravity() -> Callable[..., float]:
    global _NUMBA_KERNEL_ANTIGRAVITY
    if _NUMBA_KERNEL_ANTIGRAVITY is not None:
        return _NUMBA_KERNEL_ANTIGRAVITY
    try:
        import numba
    except Exception as error:  # pragma: no cover - depends on environment
        raise BackendUnavailableAntigravity(f"numba_import_unavailable:{type(error).__name__}") from error

    @numba.njit(cache=False, nogil=True)
    def _kernel(query: np.ndarray, references: np.ndarray) -> float:
        # Keep all arithmetic explicitly float64 and preserve row/column order
        # of the NumPy CPU reference.  The insertion of the three smallest
        # distances is deterministic and avoids a backend-specific median.
        nearest0 = np.inf
        nearest1 = np.inf
        nearest2 = np.inf
        width = query.shape[0]
        for row in range(references.shape[0]):
            total = 0.0
            for column in range(width):
                difference = float(references[row, column]) - float(query[column])
                total += difference * difference
            distance = math.sqrt(total / float(width))
            if distance < nearest0:
                nearest2 = nearest1
                nearest1 = nearest0
                nearest0 = distance
            elif distance < nearest1:
                nearest2 = nearest1
                nearest1 = distance
            elif distance < nearest2:
                nearest2 = distance
        if nearest2 == np.inf:
            return np.nan
        # The insertion loop keeps nearest0 <= nearest1 <= nearest2, so the
        # middle slot is the deterministic median-of-three value.
        return nearest1

    _NUMBA_KERNEL_ANTIGRAVITY = _kernel
    return _kernel


def score_distance_numba_antigravity(query: object, references: object, k: int) -> float:
    """Numba float64 dense distance with deterministic median-of-three."""
    vector, matrix = _dense_inputs_antigravity(query, references, k)
    kernel = _load_numba_kernel_antigravity()
    scalar = float(kernel(vector, matrix))
    if not math.isfinite(scalar):
        raise ValueError("Numba channel score is nonfinite")
    return scalar


def _selection_inputs_antigravity(
    inputs: tuple[CandidateInputAntigravity, ...],
) -> tuple[tuple[CandidateInputAntigravity, ...], tuple[Mapping[str, object], ...]]:
    if not inputs or any(not isinstance(item, CandidateInputAntigravity) for item in inputs):
        raise ValueError("benchmark inputs must be non-empty CandidateInputAntigravity values")
    ordered = tuple(sorted(inputs, key=lambda item: (item.train_values.shape[0], item.dataset, item.instance_index)))
    sizes = tuple(int(item.train_values.shape[0]) for item in ordered)
    if 5 not in sizes or 160 not in sizes:
        raise ValueError("fixed benchmark requires exact group sizes 5 and 160")
    small = next(item for item in ordered if item.train_values.shape[0] == 5)
    large = next(item for item in ordered if item.train_values.shape[0] == 160)
    median_position = (len(ordered) - 1) // 2
    median = ordered[median_position]
    # If the lower median has no duplicate exact-size peer (possible for a
    # caller fixture), the selected item is still the deterministic median in
    # the total ordering; real Task 3 inputs have an exact median-size group.
    selected = (small, median, large)
    selection = tuple(
        {
            "order": order,
            "dataset": item.dataset,
            "instance_index": item.instance_index,
            "train_group_size": int(item.train_values.shape[0]),
            "raw_length": item.raw_length,
            "role": role,
            "selection_policy": BENCHMARK_SELECTION_POLICY_ANTIGRAVITY,
        }
        for order, (item, role) in enumerate(zip(selected, ("group_size_5", "median_available", "group_size_160")), start=1)
    )
    return selected, selection


def fixed_benchmark_selection_antigravity(
    inputs: tuple[CandidateInputAntigravity, ...],
) -> tuple[CandidateInputAntigravity, ...]:
    """Return the fixed size-5/median/160 candidates used by the real gate."""
    return _selection_inputs_antigravity(tuple(inputs))[0]


def _json_float_scalars_antigravity(value: object, path: str = "") -> list[tuple[str, float]]:
    if isinstance(value, Mapping):
        result: list[tuple[str, float]] = []
        for key in sorted(value):
            result.extend(_json_float_scalars_antigravity(value[key], f"{path}.{key}"))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for index, item in enumerate(value):
            result.extend(_json_float_scalars_antigravity(item, f"{path}[{index}]"))
        return result
    if type(value) is float:
        return [(path, value)]
    return []


def _stable_payload_antigravity(result: object) -> dict[str, object]:
    if not hasattr(result, "to_payload"):
        raise ValueError("stability result lacks canonical payload")
    payload = result.to_payload()
    if not isinstance(payload, Mapping):
        raise ValueError("stability result payload is invalid")
    return dict(payload)


def _stage_result_lanes_antigravity(results: Sequence[object]) -> tuple[tuple[str, int, str], ...]:
    lanes: list[tuple[str, int, str]] = []
    for result in results:
        coordinate = getattr(result, "coordinate", None)
        lane = getattr(result, "lane", None)
        if not isinstance(coordinate, tuple) or len(coordinate) != 2 or type(coordinate[0]) is not str or type(coordinate[1]) is not int or type(lane) is not str:
            raise ValueError("stability result lane identity is invalid")
        lanes.append((coordinate[0], coordinate[1], lane))
    return tuple(lanes)


def _compare_stage_results_antigravity(
    cpu_payloads: Sequence[Mapping[str, object]],
    backend_payloads: Sequence[Mapping[str, object]],
) -> tuple[bool, bool, bool, bool, bool, float | None]:
    if len(cpu_payloads) != len(backend_payloads):
        return (False, False, False, False, False, None)
    scalar_diffs: list[float] = []
    for cpu_payload, backend_payload in zip(cpu_payloads, backend_payloads):
        cpu_scalars = dict(_json_float_scalars_antigravity(cpu_payload))
        backend_scalars = dict(_json_float_scalars_antigravity(backend_payload))
        for path in set(cpu_scalars) & set(backend_scalars):
            left, right = cpu_scalars[path], backend_scalars[path]
            if not math.isfinite(left) or not math.isfinite(right):
                continue
            scalar_diffs.append(abs(left - right))
    max_diff = max(scalar_diffs) if scalar_diffs else 0.0
    # Compare the four semantic gates independently.  Scalar score/threshold
    # fields are intentionally allowed to differ within 1e-5; a changed
    # non-original depth's ``exceeds`` flag therefore must not masquerade as a
    # mask or lane mismatch.  Candidate presence is reconstructed exactly from
    # Exp440's public rule: a provisional candidate exists when at least two
    # depth scores strictly exceed their thresholds.
    def views(payload: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
        raw = payload.get("views", ())
        return tuple(item for item in raw if isinstance(item, Mapping))

    cpu_views = tuple(views(payload) for payload in cpu_payloads)
    backend_views = tuple(views(payload) for payload in backend_payloads)
    masks_match = True
    candidate_presence_match = True
    for left_group, right_group in zip(cpu_views, backend_views):
        if len(left_group) != len(right_group):
            masks_match = candidate_presence_match = False
            continue
        for left, right in zip(left_group, right_group):
            if (left.get("mask"), left.get("retained_train_indices")) != (
                right.get("mask"), right.get("retained_train_indices")
            ):
                masks_match = False

            def candidate_present(view: Mapping[str, object]) -> bool:
                explicit = view.get("candidate_present")
                if type(explicit) is bool:
                    return explicit
                scores = view.get("depth_scores", ())
                return sum(1 for score in scores if isinstance(score, Mapping) and score.get("exceeds") is True) >= FROZEN_CONFIG.minimum_consensus_depths

            if candidate_present(left) != candidate_present(right):
                candidate_presence_match = False

    survival_counts_match = tuple(
        (payload.get("survival_count"), payload.get("required_survival_count")) for payload in cpu_payloads
    ) == tuple(
        (payload.get("survival_count"), payload.get("required_survival_count")) for payload in backend_payloads
    )
    lane_coordinates_match = tuple(
        (payload.get("candidate", {}).get("dataset"), payload.get("candidate", {}).get("instance_index"), payload.get("lane"))
        for payload in cpu_payloads
    ) == tuple(
        (payload.get("candidate", {}).get("dataset"), payload.get("candidate", {}).get("instance_index"), payload.get("lane"))
        for payload in backend_payloads
    )
    finite = all(math.isfinite(item) for item in scalar_diffs) and all(
        math.isfinite(float(value)) for payload in tuple(cpu_payloads) + tuple(backend_payloads) for _, value in _json_float_scalars_antigravity(payload)
    )
    return finite, masks_match, candidate_presence_match, survival_counts_match, lane_coordinates_match, max_diff


def _run_stage_antigravity(
    inputs: tuple[CandidateInputAntigravity, ...],
    distance: Callable[[object, object, int], float],
    *,
    backend: str,
    selection: tuple[Mapping[str, object], ...],
    device: str | None,
    version: str | None,
    numba_threads: int = 1,
) -> tuple[BackendRunAntigravity, tuple[dict[str, object], ...]]:
    payloads: tuple[dict[str, object], ...] = ()
    results: tuple[object, ...] = ()
    started = time.perf_counter()
    try:
        latest: tuple[object, ...] = ()
        for _ in range(BENCHMARK_WARMUPS_ANTIGRAVITY + BENCHMARK_REPETITIONS_ANTIGRAVITY):
            latest = tuple(score_stability_candidate_antigravity(item, distance=distance) for item in inputs)
        elapsed = time.perf_counter() - started
        results = latest
        payloads = tuple(_stable_payload_antigravity(item) for item in results)
        lanes = _stage_result_lanes_antigravity(results)
        finite = all(
            math.isfinite(float(value))
            for payload in payloads
            for _, value in _json_float_scalars_antigravity(payload)
        )
        run = BackendRunAntigravity(
            backend=backend,
            seconds=elapsed,
            lanes=lanes,
            max_diff=0.0,
            finite=finite,
            synchronized=True,
            masks_match=True,
            candidate_presence_match=True,
            survival_counts_match=True,
            lane_coordinates_match=True,
            available=True,
            device=device,
            version=version,
            compile_warmup_included=True,
            transfer_included=True,
            materialization_included=True,
            process_count=1,
            warmups=BENCHMARK_WARMUPS_ANTIGRAVITY,
            repetitions=BENCHMARK_REPETITIONS_ANTIGRAVITY,
            inner_threads=numba_threads,
            selection=selection,
            details={"platform": platform.platform()},
        )
        return run, payloads
    except BackendUnavailableAntigravity:
        raise
    except Exception as error:
        raise ValueError(f"{backend}_whole_stage_failed:{type(error).__name__}:{error}") from error


def benchmark_backends_antigravity(
    inputs: tuple[CandidateInputAntigravity, ...],
) -> BackendDecisionAntigravity:
    """Run fixed whole-view CPU/MLX/Numba parity and speed gates once.

    The function intentionally performs no multiprocessing.  It returns a
    complete immutable decision payload even when optional backends fail.
    """
    selected, selection = _selection_inputs_antigravity(tuple(inputs))
    cpu_version = platform.python_version()
    cpu_run, cpu_payloads = _run_stage_antigravity(
        selected,
        score_distance_cpu_antigravity,
        backend="cpu",
        selection=selection,
        device="CPU",
        version=cpu_version,
        numba_threads=1,
    )
    cpu_payload = cpu_run.to_payload()
    evidence: list[Mapping[str, object]] = [
        {
            "benchmark_selection": [dict(item) for item in selection],
            "backend": "cpu",
            "reference": cpu_payload,
            "stage": "whole_candidate_view_scoring",
            "labels_accessed": 0,
        }
    ]
    accelerated: list[BackendRunAntigravity] = []

    try:
        mx, device, device_label = _load_mlx_antigravity()
        mlx_version = str(getattr(mx, "__version__", FROZEN_CONFIG.mlx_version))
        mlx_run, mlx_payloads = _run_stage_antigravity(
            selected,
            score_distance_mlx_antigravity,
            backend="mlx",
            selection=selection,
            device=device_label,
            version=mlx_version,
            numba_threads=1,
        )
        finite, masks, presence, survival, lanes, max_diff = _compare_stage_results_antigravity(cpu_payloads, mlx_payloads)
        mlx_run = BackendRunAntigravity(
            **{
                **mlx_run.__dict__,
                "finite": finite,
                "masks_match": masks,
                "candidate_presence_match": presence,
                "survival_counts_match": survival,
                "lane_coordinates_match": lanes,
                "max_diff": max_diff,
            }
        )
    except BackendUnavailableAntigravity as error:
        mlx_run = BackendRunAntigravity.unavailable(
            "mlx",
            str(error),
            version=FROZEN_CONFIG.mlx_version,
            device="Apple GPU",
            selection=selection,
        )
    except Exception as error:
        mlx_run = BackendRunAntigravity.unavailable(
            "mlx",
            f"mlx_benchmark_failed:{type(error).__name__}:{error}",
            version=FROZEN_CONFIG.mlx_version,
            device="Apple GPU",
            selection=selection,
        )
    accelerated.append(mlx_run)
    evidence.append({"benchmark_selection": [dict(item) for item in selection], "backend": "mlx", "run": mlx_run.to_payload(cpu_seconds=cpu_run.seconds), "labels_accessed": 0})

    try:
        import numba

        previous_threads = int(numba.get_num_threads())
        numba.set_num_threads(1)
        try:
            numba_version = str(getattr(numba, "__version__", FROZEN_CONFIG.numba_version))
            numba_run, numba_payloads = _run_stage_antigravity(
                selected,
                score_distance_numba_antigravity,
                backend="numba",
                selection=selection,
                device="CPU",
                version=numba_version,
                numba_threads=1,
            )
            finite, masks, presence, survival, lanes, max_diff = _compare_stage_results_antigravity(cpu_payloads, numba_payloads)
            numba_run = BackendRunAntigravity(
                **{
                    **numba_run.__dict__,
                    "finite": finite,
                    "masks_match": masks,
                    "candidate_presence_match": presence,
                    "survival_counts_match": survival,
                    "lane_coordinates_match": lanes,
                    "max_diff": max_diff,
                }
            )
        finally:
            numba.set_num_threads(previous_threads)
    except BackendUnavailableAntigravity as error:
        numba_run = BackendRunAntigravity.unavailable("numba", str(error), version=FROZEN_CONFIG.numba_version, device="CPU", selection=selection)
    except Exception as error:
        numba_run = BackendRunAntigravity.unavailable(
            "numba",
            f"numba_benchmark_failed:{type(error).__name__}:{error}",
            version=FROZEN_CONFIG.numba_version,
            device="CPU",
            selection=selection,
        )
    accelerated.append(numba_run)
    evidence.append({"benchmark_selection": [dict(item) for item in selection], "backend": "numba", "run": numba_run.to_payload(cpu_seconds=cpu_run.seconds), "labels_accessed": 0})

    decision = choose_backend_antigravity(cpu=cpu_run, mlx=mlx_run, numba=numba_run)
    # ``BackendDecisionAntigravity`` is immutable; preserve the complete fixed
    # selection and labels-zero evidence alongside the gate rows.
    combined = tuple(evidence) + tuple(decision.evidence)
    return BackendDecisionAntigravity(decision.backend, decision.workers, combined, decision.fallback_reason)


__all__ = [
    "ACCELERATION_TOLERANCE_ANTIGRAVITY",
    "ACCELERATION_SPEEDUP_MIN_ANTIGRAVITY",
    "BENCHMARK_WARMUPS_ANTIGRAVITY",
    "BENCHMARK_REPETITIONS_ANTIGRAVITY",
    "BENCHMARK_SELECTION_POLICY_ANTIGRAVITY",
    "BackendUnavailableAntigravity",
    "BackendRunAntigravity",
    "BackendDecisionAntigravity",
    "choose_backend_antigravity",
    "fixed_benchmark_selection_antigravity",
    "score_distance_cpu_antigravity",
    "score_distance_mlx_antigravity",
    "score_distance_numba_antigravity",
    "benchmark_backends_antigravity",
]
