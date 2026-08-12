"""Exact, label-free Exp74d→253 guarded Hard-route replay.

Only ``minimum_source_expansion`` consumes Task 4 OOF evidence.  The legacy
route selectors consume the separate raw-computable full-fit views from
``legacy_route_sources_paper``; neither function accepts artifact paths or
historical prediction coordinates.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .calibration_paper import upper_tail_pvalues
from .contracts_paper import RawDataset
from .sources_paper.hard_sources_paper import HardSourceBundle
from .sources_paper.legacy_route_sources_paper import ConfirmationBlocks, FullFitScoreView, LegacyRouteEvidence, build_legacy_route_evidence
from .sources_paper.minirocket_paper import MiniRocketResult


MAX_TOP_PAPER = 12
CANONICAL_SMOKE_DATASET_COUNT_PAPER = 10
EXP93_WEIGHTS_PAPER = {"rocket": .36, "exp55": .22, "exp56": .22, "exp84_fg": .12, "exp84_cap3": .08}
EXP234_SOURCES_PAPER = ("rocket_exp40", "exp55_best", "exp56_best", "exp84", "canonical_normal_state")
PROVENANCE_PAPER = {
    "run_experiment_60_62_rocket_imaging_selector_variants.py": ("exp69_base_selection", "rank_ensemble_indices", "rank_ensemble_guarded_indices", "large_data_budget"),
    "run_experiment_89_74d_with_exp84_candidate.py": ("choose_rows_for_dataset",),
    "run_experiment_90_zero_f1_repair_selector.py": ("choose_rows_for_dataset",),
    "run_experiment_93_nonpos_candidate_reranker.py": ("bundle_order", "exp84_order", "score_candidates", "sorted_candidates", "weak_base_indices"),
    "run_experiment_119a_exp93_rank_order_validation.py": ("run_one",),
    "run_experiment_133_block_b_confidence_tiers.py": ("run_one",),
    "run_experiment_134_block_b_review_tail_guard.py": ("run_one",),
    "run_experiment_135_block_c_review_confirmation.py": ("run_one",),
    "run_experiment_137_operational_triage.py": ("route_tiers",),
    "experiments/exp226_235_hard_alert_quality/exp234_precision_first_rescue.py": ("research_rescue_candidates",),
    "experiments/exp236_detection_coverage_expansion/core.py": ("build_expanded_lanes",),
    "experiments/exp290_train_raw_exact_normal_guard/prediction.py": ("partition_hard_indices",),
}


@dataclass(frozen=True, slots=True)
class RouteLanes:
    hard: frozenset[int]
    standard: frozenset[int]
    priority: frozenset[int]

    def __post_init__(self) -> None:
        for lane in (self.hard, self.standard, self.priority):
            if not isinstance(lane, frozenset) or any(type(index) is not int or index < 0 for index in lane):
                raise ValueError("route lanes must be non-negative frozensets")
        if self.hard & self.standard or self.hard & self.priority or self.standard & self.priority:
            raise ValueError("route lanes must be disjoint")


@dataclass(frozen=True, slots=True)
class SourceExpansion:
    minimum_two: frozenset[int]
    minimum_three: frozenset[int]

    def __post_init__(self) -> None:
        if not self.minimum_three <= self.minimum_two or any(type(index) is not int or index < 0 for index in self.minimum_two):
            raise ValueError("source expansion lanes are invalid")


@dataclass(frozen=True, slots=True)
class HardRouteResult:
    exp74d: RouteLanes
    exp89: frozenset[int]
    exp90: frozenset[int]
    exp93: frozenset[int]
    exp137: RouteLanes
    exp234: SourceExpansion
    exp236: RouteLanes
    promoted: frozenset[int]
    exp253: RouteLanes
    exact_normal_removed: frozenset[int]
    automatic_hard: frozenset[int]
    n_test: int

    def __post_init__(self) -> None:
        if type(self.n_test) is not int or self.n_test < 1:
            raise ValueError("hard route result requires a positive TEST size")
        for lane in (self.exp74d, self.exp137, self.exp236, self.exp253):
            if not isinstance(lane, RouteLanes):
                raise ValueError("hard route stages require route lanes")
        if not isinstance(self.exp234, SourceExpansion):
            raise ValueError("hard route requires source expansion")
        values = (self.exp89, self.exp90, self.exp93, self.promoted, self.exact_normal_removed, self.automatic_hard)
        if any(not isinstance(value, frozenset) or any(type(index) is not int or index < 0 for index in value) for value in values):
            raise ValueError("hard route result indices are invalid")
        if self.promoted - self.exp236.standard or self.exp253.hard != self.exp236.hard | self.promoted or self.exp253.standard != self.exp236.standard - self.promoted:
            raise ValueError("hard route promotion lanes are inconsistent")
        if self.exact_normal_removed - self.exp253.hard or self.automatic_hard != self.exp253.hard - self.exact_normal_removed:
            raise ValueError("exact normal guard lanes are inconsistent")
        all_indices = set().union(self.exp74d.hard, self.exp74d.standard, self.exp74d.priority, self.exp89, self.exp90, self.exp93, self.exp137.hard, self.exp137.standard, self.exp137.priority, self.exp234.minimum_two, self.exp234.minimum_three, self.exp236.hard, self.exp236.standard, self.exp236.priority, self.promoted, self.exp253.hard, self.exp253.standard, self.exp253.priority, self.exact_normal_removed, self.automatic_hard)
        if any(index >= self.n_test for index in all_indices):
            raise ValueError("hard route index exceeds TEST length")

    @property
    def exp236_hard(self) -> frozenset[int]:
        return self.exp236.hard

    @property
    def exp253_hard(self) -> frozenset[int]:
        return self.exp253.hard


def _cap(indices: set[int], score: np.ndarray, count: int) -> set[int]:
    if len(indices) <= max(1, int(count)):
        return set(indices)
    return set(sorted(indices, key=lambda index: float(score[index]), reverse=True)[:max(1, int(count))])


def _at_least(required: int, sets: tuple[set[int], ...]) -> set[int]:
    counts: Counter[int] = Counter()
    for values in sets:
        counts.update(values)
    return {index for index, count in counts.items() if count >= required}


def _exp93_order(view: FullFitScoreView, count: int = MAX_TOP_PAPER) -> list[int]:
    """Copy Exp93's stable descending primitive exactly."""
    return [int(index) for index in np.argsort(-np.asarray(view.test_score), kind="stable")[:max(1, min(int(count), len(view.test_score)))]]


def _ranked_primary(evidence: LegacyRouteEvidence, count: int) -> set[int]:
    n = evidence.n_test
    combined = np.zeros(n, dtype=float)
    for name, weight in (("rocket_exp40", .50), ("exp55_best", .25), ("exp56_best", .25)):
        order = np.argsort(-np.asarray(evidence.primary[name].test_score))
        rank = np.empty(n, dtype=float)
        rank[order] = np.arange(1, n + 1, dtype=float)
        combined += weight * rank
    return set(np.argsort(combined)[:max(1, min(int(count), n))].astype(int).tolist())


def replay_exp74d_89_90_93(evidence: LegacyRouteEvidence) -> tuple[RouteLanes, frozenset[int], frozenset[int], frozenset[int]]:
    """Replay frozen selectors through Exp93 with their original tie behaviors."""
    rocket, exp55, exp56 = (evidence.primary[name] for name in ("rocket_exp40", "exp55_best", "exp56_best"))
    base_candidates = _at_least(2, (set(rocket.candidate), set(exp55.candidate), set(exp56.candidate)))
    base = _cap(base_candidates, rocket.test_score, max(1, math.ceil(.02 * evidence.n_test)))
    if not base:
        base = _cap(set(rocket.candidate), rocket.test_score, max(1, math.ceil(.01 * evidence.n_test)))
    budget1 = max(2, min(5, math.ceil(.01 * evidence.n_test)))
    budget2 = max(3, min(8, math.ceil(.02 * evidence.n_test)))
    large = evidence.n_train >= 100 and evidence.n_test >= 100
    primary = _ranked_primary(evidence, budget2) & _cap(set(range(evidence.n_test)), rocket.test_score, max(budget2, 2 * budget1))
    review_pool = _ranked_primary(evidence, budget2) - primary
    review = _cap(review_pool, rocket.test_score, budget1)
    exp74d = RouteLanes(frozenset(base | primary if large else base), frozenset(review if large else set()), frozenset())

    exp84_fg, exp84_cap3 = evidence.exp84_family_guard, evidence.exp84_count_cap_3pct
    four = _at_least(2, (set(rocket.candidate), set(exp55.candidate), set(exp56.candidate), set(exp84_fg.candidate)))
    primary89 = _cap(four & _cap(set(range(evidence.n_test)), rocket.test_score, max(budget2, 2 * budget1)), rocket.test_score, budget2)
    top1 = exp84_fg.order[0] if exp84_fg.order else None
    noalert = {top1} if not exp74d.hard and top1 is not None and exp84_fg.train_exceed_rate <= .015 and exp84_fg.top1_threshold_margin >= 0.0 and exp84_fg.top1_top2_margin >= 0.0 else set()
    exp89 = frozenset((set(exp74d.hard) | primary89 if large else set(exp74d.hard)) | noalert)

    repair: set[int] = set()
    if not exp89 and evidence.n_train > 10:
        if top1 is not None and exp84_fg.train_exceed_rate <= .015 and exp84_fg.top1_threshold_margin >= 0.0:
            repair.add(top1)
        elif rocket.order and rocket.train_exceed_rate <= .02:
            repair.add(rocket.order[0])
    exp90 = frozenset(set(exp89) | repair)

    orders = {"rocket": _exp93_order(rocket), "exp55": _exp93_order(exp55), "exp56": _exp93_order(exp56), "exp84_fg": list(exp84_fg.order[:MAX_TOP_PAPER]) if exp84_fg.train_exceed_rate <= .015 else [], "exp84_cap3": list(exp84_cap3.order[:MAX_TOP_PAPER]) if exp84_cap3.train_exceed_rate <= .02 else []}
    maps = {name: {index: rank for rank, index in enumerate(order, 1)} for name, order in orders.items()}
    pool = set().union(*(set(order[:8]) for order in orders.values())) | set(exp90)
    scored: dict[int, tuple[float, int, int, float]] = {}
    for index in pool:
        ranks = [mapping[index] for mapping in maps.values() if index in mapping]
        support, best, rank_sum = len(ranks), min(ranks, default=MAX_TOP_PAPER + 1), float(sum(ranks) if ranks else 999.0)
        weighted = sum(EXP93_WEIGHTS_PAPER[name] * ((MAX_TOP_PAPER + 1 - maps[name][index]) / MAX_TOP_PAPER) for name in maps if index in maps[name]) + .08 * max(0, support - 1)
        scored[index] = (weighted, support, best, rank_sum)
    ranked = sorted(scored, key=lambda index: (scored[index][0], scored[index][1], -scored[index][2], -scored[index][3], -index), reverse=True)
    candidate = next((index for index in ranked if index not in exp90), None)
    weak = {index for index in exp90 if scored.get(index, (0., 0, 99, 999.))[1] <= 1 and scored.get(index, (0., 0, 99, 999.))[2] > 5}
    base_best = max((scored.get(index, (0., 0, 99, 999.))[0] for index in exp90), default=0.)
    if evidence.n_train > 10 and len(exp90) <= 1 and weak and candidate is not None:
        score, support, best, _ = scored[candidate]
        exp93 = frozenset((set(exp90) - weak) | {candidate}) if support >= 2 and best <= 3 and score - base_best >= .04 else exp90
    else:
        exp93 = exp90
    return exp74d, exp89, exp90, frozenset(exp93)


def replay_exp133_134_135_137(exp93: frozenset[int], blocks: ConfirmationBlocks) -> RouteLanes:
    high, standard = set(exp93) & set(blocks.block_b_015.candidate), set(exp93) - set(blocks.block_b_015.candidate)
    candidates = (set(blocks.block_a.candidate) & set(blocks.block_b_01.candidate)) - set(exp93)
    review = set()
    if len(blocks.block_b_01.train_score) > 10 and candidates:
        review = {max(candidates, key=lambda index: (float(blocks.block_b_01.test_score[index]), -index))}
    confirmed = review & set(blocks.block_c.candidate)
    priority = confirmed if not high and bool(standard) else set()
    return RouteLanes(frozenset(high), frozenset(standard), frozenset(priority))


def minimum_source_expansion(exp137: RouteLanes, sources: HardSourceBundle, *, alpha: float) -> SourceExpansion:
    if tuple(sources.source_names) != tuple(sources.by_name):
        raise ValueError("HardSourceBundle source ordering differs")
    hits: dict[str, set[int]] = {}
    for name in EXP234_SOURCES_PAPER:
        result = sources.by_name[name]
        if result.eligible:
            hits[name] = set(np.flatnonzero(upper_tail_pvalues(result.train_oof_score, result.test_score) <= float(alpha)).astype(int).tolist())
        else:
            hits[name] = set()
    excluded = set(exp137.hard) | set(exp137.standard) | set(exp137.priority)
    counts: Counter[int] = Counter()
    for values in hits.values():
        counts.update(values)
    return SourceExpansion(frozenset(index for index, count in counts.items() if count >= 2 and index not in excluded), frozenset(index for index, count in counts.items() if count >= 3 and index not in excluded))


def union_three_source_hard(exp137: RouteLanes, exp234: SourceExpansion) -> RouteLanes:
    if not exp234.minimum_three <= exp234.minimum_two:
        raise ValueError("minimum-three candidates must be a subset of minimum-two candidates")
    baseline = set(exp137.hard) | set(exp137.standard) | set(exp137.priority)
    if baseline & set(exp234.minimum_two):
        raise ValueError("new expansion candidates overlap an Exp137 lane")
    return RouteLanes(frozenset(set(exp137.hard) | set(exp234.minimum_three)), frozenset(set(exp137.standard) | (set(exp234.minimum_two) - set(exp234.minimum_three))), exp137.priority)


def exact_normal_guard(hard: frozenset[int], train_blobs: tuple[bytes, ...], test_blobs: tuple[bytes, ...]) -> tuple[frozenset[int], frozenset[int]]:
    normal = frozenset(train_blobs)
    removed = frozenset(index for index in hard if test_blobs[index] in normal)
    return frozenset(set(hard) - set(removed)), removed


def replay_hard_route(values: RawDataset, sources: HardSourceBundle, mini: MiniRocketResult) -> HardRouteResult:
    evidence = build_legacy_route_evidence(values)
    exp74d, exp89, exp90, exp93 = replay_exp74d_89_90_93(evidence)
    exp137 = replay_exp133_134_135_137(exp93, evidence.confirmation_blocks)
    exp234 = minimum_source_expansion(exp137, sources, alpha=.01)
    exp236 = union_three_source_hard(exp137, exp234)
    promoted = frozenset(set(exp236.standard) & (set(mini.k1024) | set(mini.k4096)))
    exp253 = RouteLanes(frozenset(set(exp236.hard) | set(promoted)), frozenset(set(exp236.standard) - set(promoted)), exp236.priority)
    automatic, removed = exact_normal_guard(exp253.hard, values.train_blobs, values.test_blobs)
    return HardRouteResult(exp74d, exp89, exp90, exp93, exp137, exp234, exp236, promoted, exp253, removed, automatic, len(values.test_rows))


@dataclass(frozen=True, slots=True)
class ParityReport:
    exact: bool
    differing: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FrozenSmokeReport:
    path: Path
    sha256: str
    datasets: tuple[str, ...]


def freeze_canonical_smoke(new: Mapping[str, frozenset[int]], output_path: Path) -> FrozenSmokeReport:
    """Create-only canonical new-route bytes before a separate oracle comparison."""
    if not isinstance(new, Mapping) or len(new) != CANONICAL_SMOKE_DATASET_COUNT_PAPER:
        raise ValueError("canonical smoke requires exactly ten dataset route outputs")
    normalized: dict[str, list[int]] = {}
    for dataset, indices in new.items():
        if not isinstance(dataset, str) or not dataset or not isinstance(indices, frozenset) or any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("canonical smoke route outputs are invalid")
        normalized[dataset] = sorted(indices)
    payload = (json.dumps({"schema": "paper-route-smoke.v1", "routes": normalized}, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
    return FrozenSmokeReport(path, sha256(payload).hexdigest(), tuple(sorted(normalized)))


def compare_parity(new: Mapping[str, frozenset[int]], oracle_path: Path) -> ParityReport:
    """Parity-only seam; route/evidence construction cannot receive artifact paths."""
    rows = [json.loads(line) for line in Path(oracle_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    oracle = {str(row["dataset"]): frozenset(map(int, row["indices"])) for row in rows}
    differing = tuple(sorted(name for name, indices in new.items() if oracle.get(name) != frozenset(indices)))
    return ParityReport(not differing and set(new) == set(oracle), differing)


__all__ = ["CANONICAL_SMOKE_DATASET_COUNT_PAPER", "EXP234_SOURCES_PAPER", "FrozenSmokeReport", "HardRouteResult", "MAX_TOP_PAPER", "ParityReport", "RouteLanes", "SourceExpansion", "compare_parity", "exact_normal_guard", "freeze_canonical_smoke", "minimum_source_expansion", "replay_exp133_134_135_137", "replay_exp74d_89_90_93", "replay_hard_route", "union_three_source_hard"]
