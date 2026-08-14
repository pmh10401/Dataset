"""Standalone full-TEST Exp440 low-frequency evidence and Exp441 selection.

This is the values-only scientific closure for the frozen Exp440 feature,
calibration, score, and strict two-depth decision, followed by Exp441's
Hard-empty one-per-dataset budget-30 rule.  It intentionally has no runtime
dependency on ``experiments.exp440...`` or on historical artifacts.

This Task 6 source performs no full-dataset acceleration assessment.  Task 9
must establish exact CPU/device numerical and candidate parity plus an
end-to-end 1.1x speed gate before any non-CPU backend can be authorized.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from time import monotonic
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np

from ..contracts_paper import (
    LowFrequencyCandidate,
    LowFrequencyDepthEvidence,
    LowFrequencyResult,
    LowFrequencyRow,
    RawDataset,
)
from ..data_paper import ProgressCallback, emit_progress_paper


# Copied frozen numerical closure and constants.  The Exp439 entries are the
# exact helper origin used by Exp440, not a replacement feature definition.
PROVENANCE_PAPER = {
    "experiments/exp439_scale_aware_reconstruction_repair_antigravity/aspe_antigravity.py": (
        "amplitude_sensitive_permutation_entropy_antigravity",
    ),
    "experiments/exp439_scale_aware_reconstruction_repair_antigravity/config_antigravity.py": (
        "FROZEN_CONFIG", "Exp439FrozenConfig",
    ),
    "experiments/exp439_scale_aware_reconstruction_repair_antigravity/features_antigravity.py": (
        "fft_split_antigravity", "lower_median_depth_antigravity",
        "maximum_depth_antigravity", "preferred_depth_antigravity",
    ),
    "experiments/exp439_scale_aware_reconstruction_repair_antigravity/scoring_antigravity.py": (
        "score_channel_cpu_antigravity",
    ),
    "experiments/exp440_train_only_low_frequency_precision_gate_antigravity/features_antigravity.py": (
        "adjacent_depths_antigravity", "fit_group_calibration_antigravity",
    ),
    "experiments/exp440_train_only_low_frequency_precision_gate_antigravity/config_antigravity.py": (
        "FROZEN_CONFIG", "Exp440FrozenConfig",
    ),
    "experiments/exp440_train_only_low_frequency_precision_gate_antigravity/inputs_antigravity.py": (
        "partition_train_rows_antigravity", "test_row_abstention_reason_antigravity",
    ),
    "experiments/exp440_train_only_low_frequency_precision_gate_antigravity/scoring_antigravity.py": (
        "score_test_row_antigravity",
    ),
    "experiments/exp441_hard_empty_single_shot_recovery_antigravity/selector_antigravity.py": (
        "select_hard_empty_single_shot_antigravity",
    ),
    "experiments/exp441_hard_empty_single_shot_recovery_antigravity/contracts_antigravity.py": (
        "CandidateEvidenceAntigravity",
    ),
}

MINIMUM_RAW_LENGTH_PAPER = 16
MAXIMUM_DEPTH_PAPER = 5
EMBEDDING_DIMENSION_PAPER = 3
DELAYS_PAPER = (1, 2, 4)
NEAREST_K_PAPER = 3
QUANTILE_PAPER = .99
MINIMUM_TRAIN_ROWS_PAPER = 4
MAXIMUM_TRAIN_ROWS_PAPER = 256
STABILITY_RATIO_MAX_PAPER = 2.0
GLOBAL_CANDIDATE_BUDGET_PAPER = 30
_FLAT_EPSILON_PAPER = 1e-12


def _finite_vector_paper(values: object, *, minimum_length: int = 1) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("feature input is invalid") from error
    if vector.ndim != 1 or vector.size < minimum_length or not np.isfinite(vector).all():
        raise ValueError("feature input must be a finite vector")
    return np.ascontiguousarray(vector)


def maximum_depth_paper(raw_length: int) -> int:
    if type(raw_length) is not int or raw_length < MINIMUM_RAW_LENGTH_PAPER:
        raise ValueError("raw length is invalid")
    return min(MAXIMUM_DEPTH_PAPER, max(1, math.floor(math.log2(raw_length)) - 3))


def fft_split_paper(values: object, depth: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact Exp439 float64 FFT / float32 component conversion used by Exp440."""
    original = _finite_vector_paper(values, minimum_length=MINIMUM_RAW_LENGTH_PAPER)
    maximum = maximum_depth_paper(int(original.size))
    if type(depth) is not int or depth < 1 or depth > maximum:
        raise ValueError("decomposition depth is invalid for raw length")
    original64 = original.astype(np.float64, copy=False)
    spectrum = np.fft.rfft(original64)
    cutoff_bin = max(1, math.floor((original.size // 2) / (2**depth)))
    low_spectrum = np.zeros_like(spectrum)
    low_spectrum[:cutoff_bin + 1] = spectrum[:cutoff_bin + 1]
    low64 = np.fft.irfft(low_spectrum, n=original.size)
    high64 = original64 - low64
    low, high = np.ascontiguousarray(low64, dtype=np.float32), np.ascontiguousarray(high64, dtype=np.float32)
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("decomposition output is nonfinite")
    peak = max(1.0, float(np.max(np.abs(original))), float(np.max(np.abs(low))), float(np.max(np.abs(high))))
    tolerance = max(1e-5, float(np.spacing(np.float32(peak))))
    if float(np.max(np.abs(original - (low + high)))) > tolerance:
        raise ValueError("decomposition reconstruction tolerance failed")
    low.setflags(write=False)
    high.setflags(write=False)
    return low, high


def _aspe_paper(values: object, embedding_dimension: int, delay: int) -> float:
    """Exact scalar behavior of Exp439's ASPE primitive used for depth choice."""
    if type(embedding_dimension) is not int or embedding_dimension < 2 or type(delay) is not int or delay < 1:
        raise ValueError("ASPE parameters are invalid")
    vector = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
        raise ValueError("ASPE input must be a finite nonempty vector")
    window_count = vector.size - (embedding_dimension - 1) * delay
    if window_count < 1:
        raise ValueError("ASPE input is too short for the embedding")
    minimum = float(np.min(vector))
    peak_to_peak = float(np.max(vector) - minimum)
    if peak_to_peak <= _FLAT_EPSILON_PAPER:
        return 0.0
    normalized = (vector - minimum) / peak_to_peak
    weighted_patterns: dict[tuple[int, ...], float] = {}
    total_weight = 0.0
    offsets = np.arange(embedding_dimension, dtype=np.int64) * delay
    for start in range(window_count):
        window = normalized[start + offsets]
        mean = float(np.mean(window, dtype=np.float64))
        standard_deviation = float(np.std(window, ddof=1, dtype=np.float64))
        weight = 0.0 if mean <= _FLAT_EPSILON_PAPER and standard_deviation <= _FLAT_EPSILON_PAPER else standard_deviation / mean
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("ASPE window weight is nonfinite")
        pattern = tuple(int(index) for index in np.argsort(window, kind="stable"))
        weighted_patterns[pattern] = weighted_patterns.get(pattern, 0.0) + weight
        total_weight += weight
    if not math.isfinite(total_weight):
        raise ValueError("ASPE total weight is nonfinite")
    if total_weight <= _FLAT_EPSILON_PAPER:
        return 0.0
    entropy = -sum((weight / total_weight) * math.log2(weight / total_weight) for weight in weighted_patterns.values() if weight > 0.0)
    result = entropy / math.log2(math.factorial(embedding_dimension))
    if not math.isfinite(result):
        raise ValueError("ASPE entropy is nonfinite")
    return 0.0 if abs(result) <= _FLAT_EPSILON_PAPER else float(result)


def _preferred_depth_paper(values: object) -> int:
    vector = _finite_vector_paper(values, minimum_length=MINIMUM_RAW_LENGTH_PAPER)
    scores = []
    for depth in range(1, maximum_depth_paper(int(vector.size)) + 1):
        _, high = fft_split_paper(vector, depth)
        scores.append(float(np.median(np.asarray([_aspe_paper(high, EMBEDDING_DIMENSION_PAPER, delay) for delay in DELAYS_PAPER], dtype=np.float64))))
    for index in range(1, len(scores)):
        if scores[index] >= scores[index - 1] - 1e-6:
            return index
    return len(scores)


def _lower_median_depth_paper(depths: Iterable[int]) -> int:
    values = tuple(depths)
    if not values or any(type(value) is not int or value < 1 or value > MAXIMUM_DEPTH_PAPER for value in values):
        raise ValueError("preferred depths are invalid")
    return tuple(sorted(values))[(len(values) - 1) // 2]


def adjacent_depths_paper(preferred_depth: int, maximum_depth: int) -> tuple[int, ...]:
    if type(preferred_depth) is not int or type(maximum_depth) is not int or preferred_depth < 1 or maximum_depth < 1 or preferred_depth > maximum_depth:
        raise ValueError("depth boundary is invalid")
    result = tuple(value for value in (preferred_depth - 1, preferred_depth, preferred_depth + 1) if 1 <= value <= maximum_depth)
    return result if len(result) >= 2 else ()


def score_channel_cpu_paper(query: object, references: object, k: int) -> float:
    vector, matrix = np.asarray(query, dtype=np.float32), np.asarray(references, dtype=np.float32)
    if vector.ndim != 1 or matrix.ndim != 2 or matrix.shape[1] != vector.size or type(k) is not int or k < 1 or matrix.shape[0] < k or not np.isfinite(vector).all() or not np.isfinite(matrix).all():
        raise ValueError("channel score input is invalid")
    delta = matrix.astype(np.float64) - vector.astype(np.float64)
    nearest = np.partition(np.sqrt(np.mean(delta * delta, axis=1)), k - 1)[:k]
    result = float(np.median(nearest))
    if not math.isfinite(result):
        raise ValueError("channel score is nonfinite")
    return result


@dataclass(frozen=True, slots=True)
class _DepthCalibrationPaper:
    depth: int
    oof_scores: tuple[float, ...]
    threshold: float
    delete_one_thresholds: tuple[float, ...]
    robust_scale: float
    stability_ratio: float
    stable: bool


@dataclass(frozen=True, slots=True)
class _DepthModelPaper:
    calibration: _DepthCalibrationPaper
    references: np.ndarray
    center: float
    scale: float

    def __post_init__(self) -> None:
        matrix = np.ascontiguousarray(self.references, dtype=np.float32).copy()
        if matrix.ndim != 2 or not np.isfinite(matrix).all() or not math.isfinite(self.center) or not math.isfinite(self.scale) or self.scale <= 1e-12:
            raise ValueError("depth model is invalid")
        matrix.setflags(write=False)
        object.__setattr__(self, "references", matrix)


@dataclass(frozen=True, slots=True)
class FittedGroupPaper:
    raw_length: int
    preferred_depth: int
    calibrations: tuple[_DepthCalibrationPaper, ...]
    models: tuple[_DepthModelPaper, ...]

    @property
    def eligible(self) -> bool:
        return len(self.models) >= 2


def _low_matrix_paper(matrix: np.ndarray, depth: int) -> np.ndarray:
    return np.ascontiguousarray(np.stack([fft_split_paper(row, depth)[0] for row in matrix]), dtype=np.float32)


def _fit_scalar_paper(matrix: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = matrix.astype(np.float64)
    center = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - center)))
    if not math.isfinite(center) or not math.isfinite(scale) or scale <= 1e-12:
        raise ValueError("zero robust scale")
    return np.ascontiguousarray((values - center) / scale, dtype=np.float32), center, scale


def fit_group_calibration_paper(matrix: object) -> FittedGroupPaper:
    """Copy Exp440's adjacent-depth, fold-local OOF calibration exactly."""
    train = np.ascontiguousarray(matrix, dtype=np.float32)
    if train.ndim != 2 or train.shape[0] < MINIMUM_TRAIN_ROWS_PAPER or not np.isfinite(train).all():
        raise ValueError("finite TRAIN group with at least four rows is required")
    maximum = maximum_depth_paper(int(train.shape[1]))
    preferred = _lower_median_depth_paper(_preferred_depth_paper(row) for row in train)
    available = adjacent_depths_paper(preferred, maximum)
    calibrations: list[_DepthCalibrationPaper] = []
    models: list[_DepthModelPaper] = []
    for depth in available:
        low = _low_matrix_paper(train, depth)
        oof: list[float] = []
        for held in range(train.shape[0]):
            references = np.delete(low, held, axis=0)
            standardized, center, scale = _fit_scalar_paper(references)
            query = np.ascontiguousarray((low[held].astype(np.float64) - center) / scale, dtype=np.float32)
            oof.append(float(score_channel_cpu_paper(query, standardized, NEAREST_K_PAPER)))
        scores = np.asarray(oof, dtype=np.float64)
        threshold = float(np.quantile(scores, QUANTILE_PAPER, method="higher"))
        deletes = tuple(float(np.quantile(np.delete(scores, index), QUANTILE_PAPER, method="higher")) for index in range(len(scores)))
        ratio = max(deletes) / max(min(deletes), 1e-12)
        try:
            standardized, center, scale = _fit_scalar_paper(low)
        except ValueError:
            standardized, center, scale = None, 0.0, 0.0
        stable = bool(scale > 1e-12 and math.isfinite(ratio) and ratio <= STABILITY_RATIO_MAX_PAPER)
        calibration = _DepthCalibrationPaper(depth, tuple(oof), threshold, deletes, scale, ratio, stable)
        calibrations.append(calibration)
        if stable and standardized is not None:
            models.append(_DepthModelPaper(calibration, standardized, center, scale))
    return FittedGroupPaper(int(train.shape[1]), preferred, tuple(calibrations), tuple(models))


def strict_two_depth_pass(scores: np.ndarray, thresholds: np.ndarray) -> bool:
    """Frozen strict decision: equality is not a depth hit."""
    return bool(np.count_nonzero(np.asarray(scores) > np.asarray(thresholds)) >= 2)


def score_test_row_paper(dataset: str, instance_index: int, values: object, group: FittedGroupPaper) -> tuple[tuple[LowFrequencyDepthEvidence, ...], LowFrequencyCandidate | None]:
    vector = np.ascontiguousarray(values, dtype=np.float32)
    if type(dataset) is not str or not dataset or type(instance_index) is not int or instance_index < 0 or vector.ndim != 1 or vector.size != group.raw_length or not np.isfinite(vector).all():
        raise ValueError("TEST row is invalid")
    evidence: list[LowFrequencyDepthEvidence] = []
    for model in group.models:
        low = fft_split_paper(vector, model.calibration.depth)[0]
        query = np.ascontiguousarray((low.astype(np.float64) - model.center) / model.scale, dtype=np.float32)
        score = score_channel_cpu_paper(query, model.references, NEAREST_K_PAPER)
        threshold = model.calibration.threshold
        rank = (1 + sum(value >= score for value in model.calibration.oof_scores)) / (len(model.calibration.oof_scores) + 1)
        evidence.append(LowFrequencyDepthEvidence(model.calibration.depth, score, threshold, score > threshold, score / max(threshold, 1e-12), rank))
    scores = tuple(evidence)
    agreeing = tuple(item for item in scores if item.strict_pass)
    if len(agreeing) < 2:
        return scores, None
    ratios = sorted((item.excess_ratio for item in agreeing), reverse=True)
    ranks = sorted(item.empirical_rank for item in agreeing)
    return scores, LowFrequencyCandidate(dataset, instance_index, int(vector.size), tuple(sorted(item.depth for item in agreeing)), ratios[1], ranks[1], scores)


def _validated_hard_lanes_paper(hard: Mapping[str, frozenset[int]]) -> Mapping[str, frozenset[int]]:
    if not isinstance(hard, Mapping):
        raise ValueError("frozen Hard lanes are invalid")
    result: dict[str, frozenset[int]] = {}
    for dataset, indices in hard.items():
        if type(dataset) is not str or not dataset or not isinstance(indices, frozenset) or any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("frozen Hard lanes are invalid")
        result[dataset] = indices
    return MappingProxyType(result)


def select_source_candidates(candidates: Iterable[LowFrequencyCandidate], hard: Mapping[str, frozenset[int]], *, budget: int = GLOBAL_CANDIDATE_BUDGET_PAPER) -> tuple[LowFrequencyCandidate, ...]:
    """Copy Exp441's Hard-empty local winner then deterministic global top-30."""
    if budget != GLOBAL_CANDIDATE_BUDGET_PAPER:
        raise ValueError("low-frequency global budget is frozen at 30")
    values = tuple(candidates)
    if any(not isinstance(candidate, LowFrequencyCandidate) for candidate in values) or len({candidate.coordinate for candidate in values}) != len(values):
        raise ValueError("low-frequency candidates are invalid")
    lanes = _validated_hard_lanes_paper(hard)
    if any(candidate.dataset not in lanes for candidate in values):
        raise ValueError("frozen Hard lane is unavailable for a candidate dataset")
    provisional = tuple(sorted(values, key=lambda candidate: candidate.coordinate))
    eligible = tuple(candidate for candidate in provisional if not lanes[candidate.dataset] and candidate.coordinate not in {(dataset, index) for dataset, indices in lanes.items() for index in indices})
    grouped: dict[str, list[LowFrequencyCandidate]] = defaultdict(list)
    for candidate in eligible:
        grouped[candidate.dataset].append(candidate)
    local_winners = tuple(min(grouped[dataset], key=lambda candidate: candidate.dataset_key) for dataset in sorted(grouped))
    return tuple(sorted(local_winners, key=lambda candidate: candidate.global_key)[:GLOBAL_CANDIDATE_BUDGET_PAPER])


def _validate_values_paper(values: RawDataset) -> None:
    if not isinstance(values, RawDataset) or type(values.dataset) is not str or not values.dataset:
        raise ValueError("raw dataset is invalid")
    for indices, rows, split in ((values.train_indices, values.train_rows, "TRAIN"), (values.test_indices, values.test_rows, "TEST")):
        if len(indices) != len(rows) or tuple(indices) != tuple(range(len(indices))):
            raise ValueError(f"{split} values must retain contiguous identity")
        if any(np.asarray(row).ndim != 1 or not len(np.asarray(row)) for row in rows):
            raise ValueError(f"{split} rows are invalid")


def _score_one_dataset_paper(
    values: RawDataset,
    *,
    progress: ProgressCallback | None = None,
    parent_completed: int = 0,
    parent_total: int = 1,
    started: float | None = None,
) -> tuple[tuple[LowFrequencyRow, ...], tuple[LowFrequencyCandidate, ...], int]:
    """Replay all TEST rows for one dataset before the cross-dataset selection."""
    _validate_values_paper(values)
    finite_groups: dict[int, list[np.ndarray]] = defaultdict(list)
    excluded = 0
    for row in values.train_rows:
        vector = np.asarray(row, dtype=np.float32)
        if np.isfinite(vector).all():
            finite_groups[int(vector.size)].append(vector)
        else:
            excluded += 1
    models: dict[int, FittedGroupPaper] = {}
    for length, rows in finite_groups.items():
        emit_progress_paper(progress, {
            "stage": "task6", "phase": "tick",
            "completed": parent_completed, "total": parent_total,
            "dataset": values.dataset,
            "detail": f"group_len={len(rows)}",
            "elapsed_s": float(monotonic() - (started if started is not None else monotonic())),
        })
        if MINIMUM_TRAIN_ROWS_PAPER <= len(rows) <= MAXIMUM_TRAIN_ROWS_PAPER:
            try:
                group = fit_group_calibration_paper(np.stack(rows))
            except ValueError:
                continue
            if group.eligible:
                models[length] = group
    records: list[LowFrequencyRow] = []
    candidates: list[LowFrequencyCandidate] = []
    for instance_index, row in zip(values.test_indices, values.test_rows, strict=True):
        vector = np.asarray(row, dtype=np.float32)
        if not np.isfinite(vector).all():
            records.append(LowFrequencyRow(values.dataset, instance_index, int(vector.size), (), False, None, None, False, True, "nonfinite_test"))
            continue
        group = models.get(int(vector.size))
        if group is None:
            records.append(LowFrequencyRow(values.dataset, instance_index, int(vector.size), (), False, None, None, False, True, "no_eligible_exact_length_group"))
            continue
        depth_evidence, candidate = score_test_row_paper(values.dataset, instance_index, vector, group)
        if candidate is None:
            records.append(LowFrequencyRow(values.dataset, instance_index, int(vector.size), depth_evidence, False, None, None, False, False, None))
        else:
            records.append(LowFrequencyRow(values.dataset, instance_index, int(vector.size), depth_evidence, True, candidate.second_excess_ratio, candidate.second_empirical_rank, True, False, None))
            candidates.append(candidate)
    return tuple(records), tuple(candidates), excluded


def score_low_frequency(
    values: RawDataset | Mapping[str, RawDataset],
    hard: Mapping[str, frozenset[int]],
    *,
    progress: ProgressCallback | None = None,
) -> LowFrequencyResult:
    """Replay one or many datasets and expose the exact global Hard-empty top 30.

    Passing one ``RawDataset`` preserves the primary single-dataset input
    boundary.  A mapping is the batch form required to produce the scientific
    cross-dataset global budget without a hidden stateful accumulator.
    """
    if isinstance(values, RawDataset):
        datasets = {values.dataset: values}
    elif isinstance(values, Mapping):
        datasets = dict(values)
        if not datasets or any(type(dataset) is not str or not dataset or not isinstance(dataset_values, RawDataset) or dataset_values.dataset != dataset for dataset, dataset_values in datasets.items()):
            raise ValueError("raw dataset mapping is invalid")
    else:
        raise ValueError("raw dataset is invalid")
    lanes = _validated_hard_lanes_paper(hard)
    if any(dataset not in lanes for dataset in datasets):
        raise ValueError("frozen Hard lane is unavailable for dataset")
    rows: list[LowFrequencyRow] = []
    candidates: list[LowFrequencyCandidate] = []
    excluded = 0
    started = monotonic()
    total = len(datasets)
    finished = 0
    emit_progress_paper(progress, {"stage": "task6", "phase": "start", "completed": 0, "total": total, "elapsed_s": 0.0})
    try:
        for dataset in sorted(datasets):
            dataset_rows, dataset_candidates, dataset_excluded = _score_one_dataset_paper(
                datasets[dataset], progress=progress, parent_completed=finished, parent_total=total, started=started,
            )
            finished += 1
            emit_progress_paper(progress, {
                "stage": "task6", "phase": "tick",
                "completed": finished, "total": total,
                "dataset": dataset, "elapsed_s": float(monotonic() - started),
            })
            rows.extend(dataset_rows)
            candidates.extend(dataset_candidates)
            excluded += dataset_excluded
        emit_progress_paper(progress, {"stage": "task6", "phase": "end", "completed": total, "total": total, "elapsed_s": float(monotonic() - started)})
    except Exception:
        emit_progress_paper(progress, {"stage": "task6", "phase": "error", "completed": finished, "total": total, "elapsed_s": float(monotonic() - started)})
        raise
    provisional = tuple(candidates)
    scoped_lanes = {dataset: lanes[dataset] for dataset in sorted(datasets)}
    return LowFrequencyResult(tuple(sorted(datasets)), tuple(rows), provisional, select_source_candidates(provisional, scoped_lanes), excluded, scoped_lanes)


__all__ = [
    "FittedGroupPaper", "GLOBAL_CANDIDATE_BUDGET_PAPER", "PROVENANCE_PAPER",
    "adjacent_depths_paper", "fit_group_calibration_paper", "score_low_frequency",
    "score_test_row_paper", "select_source_candidates", "strict_two_depth_pass",
]
