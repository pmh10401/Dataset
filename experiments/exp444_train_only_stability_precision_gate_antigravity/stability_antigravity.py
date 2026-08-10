"""Exact Exp440 replay and deterministic TRAIN-only Exp444 stability views."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable, Iterable, Sequence

import numpy as np

from experiments.exp439_scale_aware_reconstruction_repair_antigravity.scoring_antigravity import (
    score_channel_cpu_antigravity,
)
from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.contracts_antigravity import (
    DepthScoreAntigravity,
    ProvisionalCandidateAntigravity,
)
from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.features_antigravity import (
    FittedGroupAntigravity,
    fit_group_calibration_antigravity,
)
from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.scoring_antigravity import (
    score_test_row_antigravity,
)
from experiments.exp441_hard_empty_single_shot_recovery_antigravity.contracts_antigravity import (
    CandidateEvidenceAntigravity,
)

from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import (
    CandidateInputAntigravity,
    CandidateStabilityResultAntigravity,
    DepthViewScoreAntigravity,
    SourceCandidateAntigravity,
    StabilityViewAntigravity,
)
from .masks_antigravity import (
    generate_omission_masks_antigravity,
    required_survival_count_antigravity,
)


Distance = Callable[[object, object, int], float]


def _copy_depth_score_antigravity(score: object) -> DepthViewScoreAntigravity:
    """Copy unchanged Exp440 score evidence into the Exp444 contract."""
    values = {}
    for name in ("depth", "score", "threshold", "exceeds", "excess_ratio", "empirical_rank"):
        if not hasattr(score, name):
            raise ValueError(f"Exp440 score is missing {name}")
        values[name] = getattr(score, name)
    try:
        return DepthViewScoreAntigravity(**values)
    except (TypeError, ValueError) as error:
        raise ValueError("Exp440 score evidence is invalid or nonfinite") from error


def _copy_scores_antigravity(scores: Iterable[object]) -> tuple[DepthViewScoreAntigravity, ...]:
    result = tuple(_copy_depth_score_antigravity(score) for score in scores)
    if result and tuple(item.depth for item in result) != tuple(sorted({item.depth for item in result})):
        raise ValueError("Exp440 score depths are not sorted and unique")
    return result


def _copy_candidate_antigravity(candidate: object) -> ProvisionalCandidateAntigravity:
    if not isinstance(candidate, ProvisionalCandidateAntigravity):
        raise ValueError("Exp440 candidate evidence has the wrong type")
    # Reconstructing the public contract copies all nested score evidence and
    # prevents any caller-owned tuple from leaking into an Exp444 result.
    return ProvisionalCandidateAntigravity(
        candidate.dataset,
        candidate.instance_index,
        candidate.raw_length,
        tuple(candidate.depths),
        candidate.agreement_count,
        candidate.second_excess_ratio,
        candidate.second_empirical_rank,
        tuple(_copy_depth_score_antigravity(score) for score in candidate.depth_scores),
    )


@dataclass(frozen=True)
class OriginalReplayAntigravity:
    """Immutable evidence that one full TRAIN group reproduces Exp442.

    The source candidate's rank is provenance from the frozen Exp442 selected
    list; Exp440's scorer supplies the coordinate, raw length, agreeing depths,
    scalar scores, thresholds, and rank-like empirical evidence for the replay.
    """

    source: SourceCandidateAntigravity
    depth_scores: tuple[DepthViewScoreAntigravity, ...]
    replayed_depths: tuple[int, ...]
    replayed_candidate: ProvisionalCandidateAntigravity
    matches_source: bool = True
    replayed_rank: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceCandidateAntigravity):
            raise ValueError("original replay source is invalid")
        scores = tuple(self.depth_scores)
        if not scores or any(not isinstance(item, DepthViewScoreAntigravity) for item in scores):
            raise ValueError("original replay scores are invalid")
        depths = tuple(self.replayed_depths)
        if depths != tuple(sorted(set(depths))) or any(type(item) is not int or item < 1 for item in depths):
            raise ValueError("original replay depths are invalid")
        if not isinstance(self.replayed_candidate, ProvisionalCandidateAntigravity):
            raise ValueError("original replay candidate is invalid")
        if type(self.matches_source) is not bool:
            raise ValueError("matches_source must be bool")
        if self.replayed_rank is not None and (type(self.replayed_rank) is not int or self.replayed_rank < 1):
            raise ValueError("replayed_rank must be a positive exact integer or None")
        object.__setattr__(self, "depth_scores", scores)
        object.__setattr__(self, "replayed_depths", depths)

    @property
    def coordinate(self) -> tuple[str, int]:
        return self.source.coordinate

    @property
    def candidate(self) -> SourceCandidateAntigravity:
        """The frozen source candidate (convenient audit alias)."""
        return self.source

    @property
    def original_depths(self) -> tuple[int, ...]:
        return self.source.original_depths

    @property
    def original_rank(self) -> int:
        return self.source.original_rank

    @property
    def candidate_depths(self) -> tuple[int, ...]:
        return self.replayed_depths

    @property
    def depths(self) -> tuple[int, ...]:
        return self.replayed_depths

    @property
    def score_evidence(self) -> tuple[DepthViewScoreAntigravity, ...]:
        return self.depth_scores

    @property
    def thresholds(self) -> tuple[float, ...]:
        return tuple(item.threshold for item in self.depth_scores)

    @property
    def candidate_present(self) -> bool:
        return True


def classify_view_antigravity(
    original_depths: Sequence[int],
    view_scores: Sequence[object],
) -> bool:
    """Return whether at least two *original* depths exceed view thresholds."""
    if isinstance(original_depths, (str, bytes)):
        raise ValueError("original_depths must be a sequence")
    original = tuple(original_depths)
    if any(type(item) is not int or item < 1 for item in original):
        raise ValueError("original_depths must contain positive exact integers")
    if len(original) < FROZEN_CONFIG.minimum_consensus_depths:
        return False
    agreeing: set[int] = set()
    for score in view_scores:
        if not hasattr(score, "depth") or not hasattr(score, "exceeds"):
            raise ValueError("view score evidence is invalid")
        depth = getattr(score, "depth")
        exceeds = getattr(score, "exceeds")
        if type(depth) is not int or depth < 1 or type(exceeds) is not bool:
            raise ValueError("view score evidence is invalid")
        if exceeds and depth in original:
            agreeing.add(depth)
    return len(agreeing) >= FROZEN_CONFIG.minimum_consensus_depths


def replay_original_candidate_antigravity(
    candidate_input: CandidateInputAntigravity,
    distance: Distance = score_channel_cpu_antigravity,
) -> OriginalReplayAntigravity:
    """Replay the unchanged Exp440 rule on a complete candidate TRAIN group.

    A missing candidate, changed coordinate, changed raw length, changed
    agreeing depths, nonfinite scalar, or invalid score evidence is a hard
    control failure.  There is deliberately no fallback or threshold repair.
    """
    if not isinstance(candidate_input, CandidateInputAntigravity):
        raise ValueError("candidate input is invalid")
    if not callable(distance):
        raise ValueError("distance must be callable")
    source = candidate_input.candidate
    try:
        fitted: FittedGroupAntigravity = fit_group_calibration_antigravity(
            candidate_input.train_values,
            distance=distance,
        )
    except Exception as error:
        raise ValueError("original Exp442 calibration is unavailable") from error
    try:
        raw_scores, replayed = score_test_row_antigravity(
            source.dataset,
            source.instance_index,
            candidate_input.test_values,
            fitted,
            distance=distance,
        )
    except Exception as error:
        raise ValueError("original Exp442 scoring failed") from error
    scores = _copy_scores_antigravity(raw_scores)
    if replayed is None:
        raise ValueError("original Exp442 candidate is absent")
    replayed_copy = _copy_candidate_antigravity(replayed)
    if replayed_copy.coordinate != source.coordinate:
        raise ValueError("original replay candidate coordinate changed")
    if replayed_copy.raw_length != source.raw_length:
        raise ValueError("original replay candidate raw length changed")
    if replayed_copy.depths != source.original_depths:
        raise ValueError("original replay agreeing depths changed")
    if tuple(score.depth for score in scores if score.exceeds) != replayed_copy.depths:
        raise ValueError("original replay score threshold semantics changed")
    # Exp442's rank is the frozen selected-list rank.  It is not re-ranked by a
    # single-candidate replay; retaining it here makes that provenance explicit
    # and lets the caller compare all thirty source ranks exactly.
    return OriginalReplayAntigravity(source, scores, replayed_copy.depths, replayed_copy, True)


def _frozen_exp442_global_key_antigravity(
    replay: OriginalReplayAntigravity,
) -> tuple[float, float, str, int]:
    """Return the exact public Exp441 key inherited by Exp442 selection."""
    candidate = replay.replayed_candidate
    evidence = CandidateEvidenceAntigravity(
        candidate.dataset,
        candidate.instance_index,
        candidate.raw_length,
        candidate.depths,
        candidate.agreement_count,
        candidate.second_excess_ratio,
        candidate.second_empirical_rank,
    )
    return evidence.global_key


def verify_original_replay_ranks_antigravity(
    replays: Sequence[OriginalReplayAntigravity],
) -> tuple[OriginalReplayAntigravity, ...]:
    """Independently recompute and verify all thirty frozen Exp442 ranks."""
    values = tuple(replays)
    if len(values) != FROZEN_CONFIG.source_candidate_count or any(
        not isinstance(item, OriginalReplayAntigravity) for item in values
    ):
        raise ValueError("aggregate replay requires exactly 30 candidates")
    coordinates = tuple(item.coordinate for item in values)
    if len(coordinates) != len(set(coordinates)) or len({item.source.dataset for item in values}) != len(values):
        raise ValueError("aggregate replay coordinates are duplicated")
    expected_ranks = tuple(range(1, FROZEN_CONFIG.source_candidate_count + 1))
    if tuple(sorted(item.original_rank for item in values)) != expected_ranks:
        raise ValueError("aggregate replay source ranks are not 1-30")
    ranked = tuple(sorted(values, key=_frozen_exp442_global_key_antigravity))
    with_ranks = tuple(replace(item, replayed_rank=rank) for rank, item in enumerate(ranked, start=1))
    if any(item.original_rank != item.replayed_rank for item in with_ranks):
        raise ValueError("aggregate replay rank differs from frozen Exp442 ranking")
    source_order = tuple(item.coordinate for item in sorted(values, key=lambda item: item.original_rank))
    replay_order = tuple(item.coordinate for item in with_ranks)
    if replay_order != source_order:
        raise ValueError("aggregate replay rank coordinates differ from frozen Exp442 ranking")
    return with_ranks


def replay_original_candidates_antigravity(
    candidate_inputs: Sequence[CandidateInputAntigravity],
    distance: Distance = score_channel_cpu_antigravity,
) -> tuple[OriginalReplayAntigravity, ...]:
    """Replay and independently rank-verify the exact 30-candidate control."""
    inputs = tuple(candidate_inputs)
    if len(inputs) != FROZEN_CONFIG.source_candidate_count or any(
        not isinstance(item, CandidateInputAntigravity) for item in inputs
    ):
        raise ValueError("original control requires exactly 30 candidate inputs")
    return verify_original_replay_ranks_antigravity(
        tuple(replay_original_candidate_antigravity(item, distance=distance) for item in inputs)
    )


def _view_from_scores_antigravity(
    mask: object,
    retained_indices: tuple[int, ...],
    scores: tuple[DepthViewScoreAntigravity, ...],
    original_depths: tuple[int, ...],
    candidate_present: bool,
) -> StabilityViewAntigravity:
    agreeing = (
        tuple(sorted({score.depth for score in scores if score.exceeds and score.depth in original_depths}))
        if candidate_present
        else ()
    )
    survived = bool(candidate_present and len(agreeing) >= FROZEN_CONFIG.minimum_consensus_depths)
    return StabilityViewAntigravity(mask, retained_indices, scores, agreeing, survived)


def aggregate_stability_views_antigravity(
    source: SourceCandidateAntigravity,
    views: Sequence[StabilityViewAntigravity],
) -> CandidateStabilityResultAntigravity:
    """Apply the frozen 0.875 review-lane boundary to immutable views."""
    if not isinstance(source, SourceCandidateAntigravity):
        raise ValueError("stability aggregate source is invalid")
    ordered_views = tuple(views)
    if not ordered_views or any(not isinstance(item, StabilityViewAntigravity) for item in ordered_views):
        raise ValueError("stability aggregate views are invalid")
    survival_count = sum(item.survived for item in ordered_views)
    required = required_survival_count_antigravity(len(ordered_views))
    lane = "stable_review" if survival_count >= required else "unstable_review"
    return CandidateStabilityResultAntigravity(source, ordered_views, survival_count, required, lane)


def score_stability_candidate_antigravity(
    candidate_input: CandidateInputAntigravity,
    distance: Distance = score_channel_cpu_antigravity,
) -> CandidateStabilityResultAntigravity:
    """Replay Exp442, then recompute every deterministic TRAIN omission view."""
    control = replay_original_candidate_antigravity(candidate_input, distance=distance)
    source = candidate_input.candidate
    masks = generate_omission_masks_antigravity(source.dataset, source.raw_length, candidate_input.train_indices)
    views: list[StabilityViewAntigravity] = []
    for mask in masks:
        excluded = set(mask.excluded_train_indices)
        retained_positions = tuple(
            position for position, index in enumerate(candidate_input.train_indices) if index not in excluded
        )
        retained_indices = tuple(candidate_input.train_indices[position] for position in retained_positions)
        if len(retained_indices) < FROZEN_CONFIG.minimum_train_rows:
            raise ValueError("omission view retains fewer than four TRAIN rows")
        retained_matrix = np.ascontiguousarray(candidate_input.train_values[list(retained_positions)], dtype="<f4")
        try:
            fitted = fit_group_calibration_antigravity(retained_matrix, distance=distance)
        except Exception as error:
            raise ValueError(f"omission view {mask.view_index} calibration is unavailable") from error
        try:
            raw_scores, candidate = score_test_row_antigravity(
                source.dataset,
                source.instance_index,
                candidate_input.test_values,
                fitted,
                distance=distance,
            )
        except Exception as error:
            raise ValueError(f"omission view {mask.view_index} scoring failed") from error
        scores = _copy_scores_antigravity(raw_scores)
        if candidate is not None:
            copied = _copy_candidate_antigravity(candidate)
            if copied.coordinate != source.coordinate or copied.raw_length != source.raw_length:
                raise ValueError(f"omission view {mask.view_index} candidate coordinate changed")
        views.append(
            _view_from_scores_antigravity(
                mask,
                retained_indices,
                scores,
                control.original_depths,
                candidate is not None,
            )
        )
    return aggregate_stability_views_antigravity(
        source,
        tuple(sorted(views, key=lambda item: item.view_index)),
    )


__all__ = [
    "OriginalReplayAntigravity",
    "aggregate_stability_views_antigravity",
    "classify_view_antigravity",
    "replay_original_candidate_antigravity",
    "replay_original_candidates_antigravity",
    "score_stability_candidate_antigravity",
    "verify_original_replay_ranks_antigravity",
]
