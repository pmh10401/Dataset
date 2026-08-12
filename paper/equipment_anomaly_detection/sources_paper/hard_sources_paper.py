"""Six-source label-free raw Hard score orchestration.

Historical source provenance is declared by ``lineage_lock_paper.json`` and
carried in every result.  This runtime imports only paper modules and public
third-party dependencies; it neither opens labels nor loads prediction/CSV
artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.model_selection import KFold

from ..calibration_paper import upper_tail_pvalues
from ..contracts_paper import LineageNode, RawDataset
from ..lineage_paper import LINEAGE_NODES
from .canonical_paper import canonical_features_paper, crossfit_scores_paper
from .imaging_paper import EXP55_CONFIG_PAPER, EXP56_CONFIG_PAPER, imaging_score_pair_paper
from .knn_paper import density_knn_score_pair_paper, scale_feature_pair_paper
from .multirocket_hydra_paper import exp84_score_pair_paper
from .preprocessing_paper import PAPER_SEED, align_series_lengths_paper, has_only_finite_values_paper, source_metadata_paper, z_normalize_paper
from .rocket_paper import block_b_feature_pair_paper, rocket_feature_pair_paper


RAW_PRIMARY_SOURCES = ("rocket_exp40", "exp55_best", "exp56_best", "exp84")
EXPECTED_HARD_SOURCES = RAW_PRIMARY_SOURCES + ("rocket_block_b", "canonical_normal_state")
SOURCE_ALPHA_PAPER = {"rocket_exp40": .03, "exp55_best": .03, "exp56_best": .02, "exp84": .02, "rocket_block_b": .015, "canonical_normal_state": .02}
_LINEAGE_BY_SOURCE = {"rocket_exp40": "rocket_exp40", "exp55_best": "exp55_best", "exp56_best": "exp56_best", "exp84": "exp84_family_guard", "rocket_block_b": "rocket_block_b", "canonical_normal_state": "canonical_normal_state"}


@dataclass(frozen=True, slots=True)
class HardSourceResult:
    name: str
    origin: LineageNode
    train_oof_score: np.ndarray
    test_score: np.ndarray
    test_pvalue: np.ndarray
    threshold: float
    test_native_pass: np.ndarray
    eligible: bool
    eligibility_reason: str


@dataclass(frozen=True, slots=True)
class HardSourceBundle:
    source_names: tuple[str, ...]
    by_name: dict[str, HardSourceResult]


def fold_plan_paper(n_train: int):
    if n_train < 2: return []
    if n_train < 20: return [(np.delete(np.arange(n_train), index), np.asarray([index])) for index in range(n_train)]
    return [(fit.astype(int), query.astype(int)) for fit, query in KFold(n_splits=5, shuffle=True, random_state=PAPER_SEED).split(np.arange(n_train))]


def _historical_pair_paper(name: str, fit: np.ndarray, query: np.ndarray, target_len: int) -> tuple[np.ndarray, np.ndarray]:
    if name == "rocket_exp40":
        fit_z, query_z = z_normalize_paper(fit), z_normalize_paper(query)
        fit_features, query_features = rocket_feature_pair_paper(fit_z, query_z, fit_z.shape[1], 256)
        fit_features, query_features = scale_feature_pair_paper(fit_features, query_features)
        return density_knn_score_pair_paper(fit_features, query_features, 3, "local_gap")
    if name == "exp55_best": return imaging_score_pair_paper(fit, query, EXP55_CONFIG_PAPER)
    if name == "exp56_best": return imaging_score_pair_paper(fit, query, EXP56_CONFIG_PAPER)
    if name == "exp84": return exp84_score_pair_paper(z_normalize_paper(fit), z_normalize_paper(query))
    raise ValueError(f"unknown historical source {name}")


def crossfit_source_paper(name: str, train: np.ndarray, test: np.ndarray, target_len: int) -> tuple[np.ndarray, np.ndarray]:
    oof = np.full(len(train), np.nan, dtype=float)
    for fit_indices, query_indices in fold_plan_paper(len(train)):
        _, scores = _historical_pair_paper(name, train[fit_indices], train[query_indices], target_len)
        oof[query_indices] = scores
    _, test_scores = _historical_pair_paper(name, train, test, target_len)
    if not np.all(np.isfinite(oof)) or not np.all(np.isfinite(test_scores)): raise ValueError("nonfinite_score")
    return oof, np.asarray(test_scores, dtype=float)


def crossfit_confirmation_scores_paper(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    oof = np.full(len(train), np.nan, dtype=float)
    for fit_indices, query_indices in fold_plan_paper(len(train)):
        fit_features, query_features = block_b_feature_pair_paper(z_normalize_paper(train[fit_indices]), z_normalize_paper(train[query_indices]))
        _, scores = density_knn_score_pair_paper(fit_features, query_features, 3, "local_gap"); oof[query_indices] = scores
    fit_features, test_features = block_b_feature_pair_paper(z_normalize_paper(train), z_normalize_paper(test)); _, test_scores = density_knn_score_pair_paper(fit_features, test_features, 3, "local_gap")
    if not np.all(np.isfinite(oof)) or not np.all(np.isfinite(test_scores)): raise ValueError("nonfinite_score")
    return oof, test_scores


def _canonical_source_paper(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return crossfit_scores_paper(canonical_features_paper(train), canonical_features_paper(test), backend="knn_local_gap")


def _origin_paper(name: str) -> LineageNode:
    target = _LINEAGE_BY_SOURCE[name]
    return next(node for node in LINEAGE_NODES if node.node_id == target)


def _abstain_paper(name: str, test_count: int, reason: str) -> HardSourceResult:
    scores = np.empty(0, dtype=float); pvalue = np.ones(test_count, dtype=float)
    return HardSourceResult(name, _origin_paper(name), scores, np.empty(0, dtype=float), pvalue, SOURCE_ALPHA_PAPER[name], np.zeros(test_count, dtype=bool), False, reason)


def _completed_paper(name: str, train_oof: np.ndarray, test_score: np.ndarray) -> HardSourceResult:
    pvalue = upper_tail_pvalues(train_oof, test_score); threshold = SOURCE_ALPHA_PAPER[name]
    return HardSourceResult(name, _origin_paper(name), np.asarray(train_oof, dtype=float), np.asarray(test_score, dtype=float), pvalue, threshold, pvalue <= threshold, True, "eligible")


def score_hard_sources(values: RawDataset) -> HardSourceBundle:
    """Score the frozen six sources from raw values only, with explicit abstention."""
    if len(values.train_rows) < 5:
        return HardSourceBundle(EXPECTED_HARD_SOURCES, {name: _abstain_paper(name, len(values.test_rows), "unsupported_n_train_lt_5") for name in EXPECTED_HARD_SOURCES})
    if not has_only_finite_values_paper(values):
        return HardSourceBundle(EXPECTED_HARD_SOURCES, {name: _abstain_paper(name, len(values.test_rows), "nonfinite_input") for name in EXPECTED_HARD_SOURCES})
    metadata = source_metadata_paper(values); train = align_series_lengths_paper(values.train_rows, metadata.train_target_len); test = align_series_lengths_paper(values.test_rows, metadata.train_target_len); results = {}
    for name in RAW_PRIMARY_SOURCES:
        try: results[name] = _completed_paper(name, *crossfit_source_paper(name, train, test, metadata.train_target_len))
        except RuntimeError as exc:
            if str(exc).startswith("dependency_unavailable:"): results[name] = _abstain_paper(name, len(test), str(exc))
            else: raise
    try: results["rocket_block_b"] = _completed_paper("rocket_block_b", *crossfit_confirmation_scores_paper(train, test))
    except RuntimeError as exc: results["rocket_block_b"] = _abstain_paper("rocket_block_b", len(test), str(exc))
    try: results["canonical_normal_state"] = _completed_paper("canonical_normal_state", *_canonical_source_paper(train, test))
    except RuntimeError as exc:
        if str(exc).startswith("dependency_unavailable:"): results["canonical_normal_state"] = _abstain_paper("canonical_normal_state", len(test), str(exc))
        else: raise
    return HardSourceBundle(EXPECTED_HARD_SOURCES, results)


__all__ = ["EXPECTED_HARD_SOURCES", "HardSourceBundle", "HardSourceResult", "RAW_PRIMARY_SOURCES", "crossfit_confirmation_scores_paper", "crossfit_source_paper", "score_hard_sources"]
