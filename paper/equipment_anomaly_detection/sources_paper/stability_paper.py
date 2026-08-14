"""Standalone deterministic TRAIN-omission replay for selected paper candidates.

The numerical low-frequency calibration remains in ``low_frequency_paper``;
this module copies Exp444's pure mask, retained-TRAIN, survival, and partition
closure without importing the legacy Exp444 package at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
import math
from time import monotonic
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from ..contracts_paper import LowFrequencyCandidate, LowFrequencyDepthEvidence, LowFrequencyResult, RawDataset
from ..data_paper import ProgressCallback, emit_progress_paper
from .low_frequency_paper import fit_group_calibration_paper, score_test_row_paper


PROVENANCE_PAPER = {
    "experiments/exp444_train_only_stability_precision_gate_antigravity/sources_antigravity.py": {
        "symbols": ("_prepare_candidate_input_antigravity",),
        "source_sha256": "f6b6f61b6517c46b3e63ef008d79f77f488348ba1e63b5980519aef8c9216e6a",
    },
    "experiments/exp444_train_only_stability_precision_gate_antigravity/masks_antigravity.py": {
        "symbols": (
            "canonical_mask_identity_sha256_antigravity", "omission_count_antigravity",
            "required_survival_count_antigravity", "generate_omission_masks_antigravity",
        ),
        "source_sha256": "b44c3d13ddca570f1d6747b39f52285a04d18d017d46c32dcd6709d9ecc94daa",
    },
    "experiments/exp444_train_only_stability_precision_gate_antigravity/stability_antigravity.py": {
        "symbols": (
            "classify_view_antigravity", "aggregate_stability_views_antigravity",
            "score_stability_candidate_antigravity",
        ),
        "source_sha256": "aaa14d720613f39815945aac9526d86ad275577f46c792d8879cd5e16131ef4c",
    },
}

_MASK_SEED_PAPER = 44420260810
_MAXIMUM_VIEWS_PAPER = 16
_MINIMUM_TRAIN_ROWS_PAPER = 4
_MINIMUM_CONSENSUS_DEPTHS_PAPER = 2


def _exact_indices(values: Iterable[int], name: str, *, nonempty: bool = False) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be integer indices")
    try:
        result = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be iterable") from error
    if any(type(value) is not int or value < 0 for value in result) or result != tuple(sorted(set(result))) or (nonempty and not result):
        raise ValueError(f"{name} must be sorted unique nonnegative indices")
    return result


def _coordinate(value: object, name: str = "coordinate") -> tuple[str, int]:
    if not isinstance(value, tuple) or len(value) != 2 or type(value[0]) is not str or not value[0] or type(value[1]) is not int or value[1] < 0:
        raise ValueError(f"{name} is invalid")
    return value[0], value[1]


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def omission_count_paper(train_count: int) -> int:
    if type(train_count) is not int or train_count < 1:
        raise ValueError("train_count must be a positive exact integer")
    return max(1, train_count // 10)


def required_survival_count_paper(view_count: int) -> int:
    """Exact frozen Exp444 ``ceil(7 * views / 8)`` boundary."""
    if type(view_count) is not int or view_count < 1:
        raise ValueError("view_count must be a positive exact integer")
    return (7 * view_count + 7) // 8


def canonical_mask_identity_sha256_paper(
    dataset: str, raw_length: int, train_indices: Sequence[int], excluded_train_indices: Sequence[int],
) -> str:
    if type(dataset) is not str or not dataset or type(raw_length) is not int or raw_length < 1:
        raise ValueError("mask identity is invalid")
    train = _exact_indices(train_indices, "train_indices", nonempty=True)
    excluded = _exact_indices(excluded_train_indices, "excluded_train_indices", nonempty=True)
    if not set(excluded) <= set(train) or len(train) - len(excluded) < _MINIMUM_TRAIN_ROWS_PAPER:
        raise ValueError("omission mask cannot retain the frozen minimum TRAIN rows")
    return sha256(_canonical_bytes({
        "dataset": dataset, "excluded_train_indices": list(excluded), "mask_seed": _MASK_SEED_PAPER,
        "raw_length": raw_length, "train_indices": list(train),
    })).hexdigest()


@dataclass(frozen=True, slots=True)
class OmissionMaskPaper:
    view_index: int
    excluded_train_indices: tuple[int, ...]
    identity_sha256: str

    def __post_init__(self) -> None:
        if type(self.view_index) is not int or self.view_index < 0:
            raise ValueError("view_index is invalid")
        excluded = _exact_indices(self.excluded_train_indices, "excluded_train_indices", nonempty=True)
        if type(self.identity_sha256) is not str or len(self.identity_sha256) != 64 or any(value not in "0123456789abcdef" for value in self.identity_sha256):
            raise ValueError("mask identity SHA256 is invalid")
        object.__setattr__(self, "excluded_train_indices", excluded)


def _counter_mask(dataset: str, raw_length: int, train: tuple[int, ...], omit_count: int, counter: int) -> tuple[int, ...]:
    payload = _canonical_bytes({
        "dataset": dataset, "mask_seed": _MASK_SEED_PAPER, "raw_length": raw_length,
        "train_indices": list(train), "view_counter": counter,
    })
    ranked = sorted(train, key=lambda index: sha256(payload + b"\x00" + str(index).encode("ascii")).digest())
    return tuple(sorted(ranked[:omit_count]))


def generate_omission_masks_paper(dataset: str, raw_length: int, train_indices: Sequence[int]) -> tuple[OmissionMaskPaper, ...]:
    """Exact Exp444 deterministic all-small-group / bounded-large-group masks."""
    if type(dataset) is not str or not dataset or type(raw_length) is not int or raw_length < 1:
        raise ValueError("mask inputs are invalid")
    train = _exact_indices(train_indices, "train_indices", nonempty=True)
    omit_count = omission_count_paper(len(train))
    if len(train) - omit_count < _MINIMUM_TRAIN_ROWS_PAPER:
        raise ValueError("omission would retain fewer than four TRAIN rows")
    possible_count = math.comb(len(train), omit_count)
    target = min(_MAXIMUM_VIEWS_PAPER, possible_count)
    candidates: dict[tuple[int, ...], str] = {}
    if possible_count <= _MAXIMUM_VIEWS_PAPER:
        for excluded in combinations(train, omit_count):
            candidates[excluded] = canonical_mask_identity_sha256_paper(dataset, raw_length, train, excluded)
    else:
        counter = 0
        while len(candidates) < target and counter < target * 128:
            excluded = _counter_mask(dataset, raw_length, train, omit_count, counter)
            candidates.setdefault(excluded, canonical_mask_identity_sha256_paper(dataset, raw_length, train, excluded))
            counter += 1
        if len(candidates) != target:
            raise RuntimeError("deterministic omission mask generation did not reach target")
    ordered = sorted(candidates.items(), key=lambda item: (item[1], item[0]))
    return tuple(OmissionMaskPaper(index, excluded, identity) for index, (excluded, identity) in enumerate(ordered))


@dataclass(frozen=True, slots=True)
class StabilityViewPaper:
    """One retained-TRAIN refit with raw score evidence and frozen survival result."""

    mask: OmissionMaskPaper
    retained_train_indices: tuple[int, ...]
    depth_evidence: tuple[LowFrequencyDepthEvidence, ...]
    agreeing_depths: tuple[int, ...]
    survived: bool
    missing_reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.mask, OmissionMaskPaper):
            raise ValueError("view mask is invalid")
        retained = _exact_indices(self.retained_train_indices, "retained_train_indices", nonempty=True)
        if set(retained) & set(self.mask.excluded_train_indices):
            raise ValueError("retained and excluded TRAIN indices overlap")
        evidence = tuple(self.depth_evidence)
        if any(not isinstance(item, LowFrequencyDepthEvidence) for item in evidence) or tuple(item.depth for item in evidence) != tuple(sorted({item.depth for item in evidence})):
            raise ValueError("view raw evidence is invalid")
        agreeing = _exact_indices(self.agreeing_depths, "agreeing_depths")
        if any(depth not in tuple(item.depth for item in evidence) for depth in agreeing):
            raise ValueError("agreeing depth is absent from raw evidence")
        if type(self.survived) is not bool or self.missing_reason is not None and (type(self.missing_reason) is not str or not self.missing_reason):
            raise ValueError("view status is invalid")
        if self.missing_reason is not None:
            if agreeing or self.survived:
                raise ValueError("missing candidate view cannot survive")
        else:
            expected = len(agreeing) >= _MINIMUM_CONSENSUS_DEPTHS_PAPER and all(
                next(item for item in evidence if item.depth == depth).strict_pass for depth in agreeing
            )
            if self.survived != expected:
                raise ValueError("view survival does not match raw evidence")
        object.__setattr__(self, "retained_train_indices", retained)
        object.__setattr__(self, "depth_evidence", evidence)
        object.__setattr__(self, "agreeing_depths", agreeing)

    @property
    def view_index(self) -> int:
        return self.mask.view_index


@dataclass(frozen=True, slots=True)
class CandidateStabilityPaper:
    coordinate: tuple[str, int]
    views: tuple[StabilityViewPaper, ...]
    survival_count: int
    required_survival_count: int
    lane: str

    def __post_init__(self) -> None:
        coordinate = _coordinate(self.coordinate)
        views = tuple(self.views)
        if not views or any(not isinstance(view, StabilityViewPaper) for view in views) or tuple(view.view_index for view in views) != tuple(range(len(views))):
            raise ValueError("candidate stability views are invalid")
        if type(self.survival_count) is not int or type(self.required_survival_count) is not int or self.required_survival_count < 1 or self.required_survival_count > len(views):
            raise ValueError("candidate stability counts are invalid")
        survival = sum(view.survived for view in views)
        if self.survival_count != survival:
            raise ValueError("candidate survival count does not match views")
        required = required_survival_count_paper(len(views))
        if self.required_survival_count != required:
            raise ValueError("candidate required survival threshold does not match frozen rule")
        expected_lane = "stable_review" if survival >= required else "unstable_review"
        if self.lane != expected_lane:
            raise ValueError("candidate stability lane is invalid")
        object.__setattr__(self, "coordinate", coordinate)
        object.__setattr__(self, "views", views)


def aggregate_stability_views(coordinate: tuple[str, int], views: Sequence[StabilityViewPaper]) -> CandidateStabilityPaper:
    coordinate = _coordinate(coordinate)
    ordered = tuple(views)
    if not ordered or any(not isinstance(view, StabilityViewPaper) for view in ordered):
        raise ValueError("stability aggregate views are invalid")
    if tuple(view.view_index for view in ordered) != tuple(range(len(ordered))):
        raise ValueError("stability view indices must be a contiguous sorted sequence")
    survival = sum(view.survived for view in ordered)
    required = required_survival_count_paper(len(ordered))
    return CandidateStabilityPaper(coordinate, ordered, survival, required, "stable_review" if survival >= required else "unstable_review")


@dataclass(frozen=True, slots=True)
class StabilityPartition:
    selected_coordinates: tuple[tuple[str, int], ...]
    stable_coordinates: tuple[tuple[str, int], ...]
    unstable_coordinates: tuple[tuple[str, int], ...]
    by_coordinate: Mapping[tuple[str, int], CandidateStabilityPaper]

    def __post_init__(self) -> None:
        selected = tuple(_coordinate(item, "selected coordinate") for item in self.selected_coordinates)
        stable = tuple(_coordinate(item, "stable coordinate") for item in self.stable_coordinates)
        unstable = tuple(_coordinate(item, "unstable coordinate") for item in self.unstable_coordinates)
        evidence = dict(self.by_coordinate)
        if len(selected) != len(set(selected)) or len(stable) != len(set(stable)) or len(unstable) != len(set(unstable)):
            raise ValueError("stability coordinates contain duplicates")
        if set(stable) & set(unstable):
            raise ValueError("stable and unstable coordinates overlap")
        if set(stable) | set(unstable) != set(selected):
            raise ValueError("stability partition omits or adds selected coordinates")
        if set(evidence) != set(selected) or any(not isinstance(value, CandidateStabilityPaper) or value.coordinate != coordinate for coordinate, value in evidence.items()):
            raise ValueError("stability evidence identities are mismatched")
        if {coordinate for coordinate, value in evidence.items() if value.lane == "stable_review"} != set(stable) or {coordinate for coordinate, value in evidence.items() if value.lane == "unstable_review"} != set(unstable):
            raise ValueError("stability lanes do not match per-view evidence")
        object.__setattr__(self, "selected_coordinates", selected)
        object.__setattr__(self, "stable_coordinates", stable)
        object.__setattr__(self, "unstable_coordinates", unstable)
        object.__setattr__(self, "by_coordinate", MappingProxyType(dict(evidence)))


Replay = Callable[[str, int, np.ndarray, np.ndarray], tuple[tuple[LowFrequencyDepthEvidence, ...], LowFrequencyCandidate | None]]


def _default_replay(dataset: str, index: int, test: np.ndarray, retained: np.ndarray) -> tuple[tuple[LowFrequencyDepthEvidence, ...], LowFrequencyCandidate | None]:
    return score_test_row_paper(dataset, index, test, fit_group_calibration_paper(retained))


def _candidate_views(
    candidate: LowFrequencyCandidate,
    values: RawDataset,
    replay: Replay,
    *,
    progress: ProgressCallback | None = None,
    parent_completed: int = 0,
    parent_total: int = 1,
    started: float | None = None,
) -> tuple[StabilityViewPaper, ...]:
    if values.dataset != candidate.dataset or candidate.instance_index >= len(values.test_rows) or values.test_indices[candidate.instance_index] != candidate.instance_index:
        raise ValueError("raw candidate TEST identity is unavailable")
    if values.test_rows[candidate.instance_index].size != candidate.raw_length:
        raise ValueError("raw candidate length differs from selected coordinate")
    raw_indices = _exact_indices(values.train_indices, "raw train_indices", nonempty=True)
    if len(values.train_rows) != len(raw_indices):
        raise ValueError("raw TRAIN rows do not match indices")
    # This is the exact source reconstruction boundary: rows of a different
    # raw length and whole-row nonfinite values never enter this candidate's
    # omission population, but retain their original database identities.
    eligible = tuple(
        (index, np.asarray(row, dtype="<f4"))
        for index, row in zip(raw_indices, values.train_rows, strict=True)
        if np.asarray(row).ndim == 1
        and np.asarray(row).size == candidate.raw_length
        and np.isfinite(np.asarray(row)).all()
    )
    train_indices = tuple(index for index, _ in eligible)
    if len(train_indices) < _MINIMUM_TRAIN_ROWS_PAPER:
        raise ValueError("candidate exact-length TRAIN group is too small")
    masks = generate_omission_masks_paper(candidate.dataset, candidate.raw_length, train_indices)
    train = np.ascontiguousarray(np.stack(tuple(row for _, row in eligible)), dtype="<f4")
    views: list[StabilityViewPaper] = []
    for mask in masks:
        excluded = set(mask.excluded_train_indices)
        retained_positions = tuple(position for position, index in enumerate(train_indices) if index not in excluded)
        retained_indices = tuple(train_indices[position] for position in retained_positions)
        if len(retained_indices) < _MINIMUM_TRAIN_ROWS_PAPER:
            raise ValueError("omission view retains fewer than four TRAIN rows")
        retained = np.ascontiguousarray(train[list(retained_positions)], dtype="<f4")
        # Exp444 failure semantics are fail-closed: scorer/calibration failures
        # propagate rather than becoming a synthetic survived/failed default.
        evidence, replayed = replay(candidate.dataset, candidate.instance_index, values.test_rows[candidate.instance_index], retained)
        copied_evidence = tuple(evidence)
        if any(not isinstance(item, LowFrequencyDepthEvidence) for item in copied_evidence):
            raise ValueError("omission view returned invalid low-frequency evidence")
        if replayed is None:
            views.append(StabilityViewPaper(mask, retained_indices, copied_evidence, (), False, "candidate_absent"))
        elif replayed.coordinate != candidate.coordinate or replayed.raw_length != candidate.raw_length:
            raise ValueError("omission view candidate coordinate changed")
        else:
            agreeing = tuple(sorted(item.depth for item in copied_evidence if item.strict_pass and item.depth in candidate.depths))
            views.append(StabilityViewPaper(mask, retained_indices, copied_evidence, agreeing, len(agreeing) >= _MINIMUM_CONSENSUS_DEPTHS_PAPER, None))
        emit_progress_paper(progress, {
            "stage": "task7", "phase": "tick",
            "completed": parent_completed, "total": parent_total,
            "dataset": candidate.dataset,
            "detail": f"view_index={mask.view_index}",
            "elapsed_s": float(monotonic() - (started if started is not None else monotonic())),
        })
    return tuple(views)


def classify_stability(
    low_frequency: LowFrequencyResult,
    raw_datasets: Mapping[str, RawDataset],
    *,
    replay: Replay | None = None,
    progress: ProgressCallback | None = None,
) -> StabilityPartition:
    """Refit selected candidates only, once per deterministic retained-TRAIN view."""
    if not isinstance(low_frequency, LowFrequencyResult) or not isinstance(raw_datasets, Mapping):
        raise ValueError("stability inputs are invalid")
    selected = tuple(low_frequency.selected_candidates)
    if len(selected) > 30 or len({candidate.coordinate for candidate in selected}) != len(selected) or len({candidate.dataset for candidate in selected}) != len(selected):
        raise ValueError("selected low-frequency candidates violate the frozen budget")
    raw = dict(raw_datasets)
    if any(not isinstance(raw.get(candidate.dataset), RawDataset) for candidate in selected):
        raise ValueError("selected candidates require raw normal TRAIN groups")
    scorer = _default_replay if replay is None else replay
    if not callable(scorer):
        raise ValueError("replay must be callable")
    started = monotonic()
    total = len(selected)
    finished = 0
    emit_progress_paper(progress, {"stage": "task7", "phase": "start", "completed": 0, "total": total, "elapsed_s": 0.0})
    try:
        results = []
        for candidate in selected:
            emit_progress_paper(progress, {
                "stage": "task7", "phase": "tick",
                "completed": finished, "total": total,
                "dataset": candidate.dataset,
                "detail": f"instance_index={candidate.instance_index}",
                "elapsed_s": float(monotonic() - started),
            })
            views = _candidate_views(
                candidate, raw[candidate.dataset], scorer,
                progress=progress, parent_completed=finished, parent_total=total, started=started,
            )
            results.append(aggregate_stability_views(candidate.coordinate, views))
            finished += 1
            emit_progress_paper(progress, {
                "stage": "task7", "phase": "tick",
                "completed": finished, "total": total,
                "dataset": candidate.dataset,
                "detail": f"instance_index={candidate.instance_index}",
                "elapsed_s": float(monotonic() - started),
            })
        emit_progress_paper(progress, {"stage": "task7", "phase": "end", "completed": total, "total": total, "elapsed_s": float(monotonic() - started)})
    except Exception:
        emit_progress_paper(progress, {"stage": "task7", "phase": "error", "completed": finished, "total": total, "elapsed_s": float(monotonic() - started)})
        raise
    evidence = {result.coordinate: result for result in results}
    return StabilityPartition(
        tuple(candidate.coordinate for candidate in selected),
        tuple(result.coordinate for result in results if result.lane == "stable_review"),
        tuple(result.coordinate for result in results if result.lane == "unstable_review"),
        evidence,
    )


__all__ = [
    "CandidateStabilityPaper", "OmissionMaskPaper", "PROVENANCE_PAPER", "StabilityPartition",
    "StabilityViewPaper", "aggregate_stability_views", "canonical_mask_identity_sha256_paper",
    "classify_stability", "generate_omission_masks_paper", "omission_count_paper", "required_survival_count_paper",
]
