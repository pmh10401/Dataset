"""Raw-computable full-fit evidence required by the frozen Exp74d→137 route.

This is deliberately distinct from Task 4's OOF evidence.  Exp40 follows the
historical database metadata-length path; Exp55/56/84 use the historical
TRAIN+TEST actual-length median.  The latter is label-free transductive value
preprocessing and is surfaced on ``LegacyRouteEvidence`` for run provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from ..contracts_paper import RawDataset
from .imaging_paper import EXP55_CONFIG_PAPER, EXP56_CONFIG_PAPER, imaging_score_pair_paper
from .knn_paper import density_knn_score_pair_paper, scale_feature_pair_paper
from .multirocket_hydra_paper import exp84_score_pair_paper
from .preprocessing_paper import align_series_lengths_paper, z_normalize_paper
from .rocket_paper import make_kernels_paper, reference_train_paper, rocket_feature_pair_paper, rocket_transform_paper


WEAK_FAMILIES_PAPER = frozenset(("ChlorineConcentration", "Computers", "EthanolLevel", "HandOutlines", "LargeKitchenAppliances", "MiddlePhalanxOutlineCorrect", "MoteStrain", "RefrigerationDevices", "ScreenType", "Yoga"))
PROVENANCE_PAPER = {
    "run_experiment_40_original_score_normalization_sweep.py": ("parse_family", "count_cap_threshold", "score_pair_for_config"),
    "run_original_improvement_experiment.py": ("target_len_for_record", "rate_for_threshold"),
    "run_balanced_improvement_experiment.py": ("WEAK_FAMILIES",),
    "run_experiment_60_62_rocket_imaging_selector_variants.py": ("prediction_bundle",),
    "run_experiment_131_rocket_block_b_calibration.py": ("fixed_reference", "score_block_b"),
    "run_experiment_132_block_b_review_integration.py": ("fixed_reference", "block_scores"),
    "run_experiment_135_block_c_review_confirmation.py": ("block_c_candidates",),
}


def parse_family_paper(dataset_name: str) -> str:
    if dataset_name == "CornellWhaleChallenge":
        return "CornellWhaleChallenge"
    return dataset_name.rsplit("_normal_", 1)[0] if "_normal_" in dataset_name else dataset_name


def count_cap_threshold_paper(train_scores: np.ndarray, rate: float) -> tuple[float, float, int]:
    values = np.asarray(train_scores, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        values = np.asarray([0.0], dtype=np.float64)
    cap = max(0, min(int(np.floor(float(rate) * len(values))), len(values) - 1))
    return float(np.sort(values)[len(values) - cap - 1]), cap / max(1, len(values)), cap


@dataclass(frozen=True, slots=True)
class FullFitScoreView:
    name: str
    train_score: np.ndarray
    test_score: np.ndarray
    rate: float

    @property
    def threshold(self) -> float:
        return count_cap_threshold_paper(self.train_score, self.rate)[0]

    @property
    def candidate(self) -> frozenset[int]:
        return frozenset(np.flatnonzero(np.asarray(self.test_score) > self.threshold).astype(int).tolist())

    @property
    def train_exceed_rate(self) -> float:
        score = np.asarray(self.train_score)
        return float(np.mean(score > self.threshold)) if len(score) else 0.0

    @property
    def order(self) -> tuple[int, ...]:
        return tuple(np.argsort(np.asarray(self.test_score))[::-1].astype(int).tolist())

    @property
    def top1_threshold_margin(self) -> float:
        return float(self.test_score[self.order[0]] - self.threshold) if self.order else float("-inf")

    @property
    def top1_top2_margin(self) -> float:
        return float(self.test_score[self.order[0]] - self.test_score[self.order[1]]) if len(self.order) > 1 else float("inf")


@dataclass(frozen=True, slots=True)
class ConfirmationBlocks:
    block_a: FullFitScoreView
    block_b: FullFitScoreView
    block_c: FullFitScoreView


@dataclass(frozen=True, slots=True)
class LegacyRouteEvidence:
    dataset: str
    family: str
    n_train: int
    n_test: int
    test_median_length: int
    primary: dict[str, FullFitScoreView]
    exp84: FullFitScoreView
    confirmation_blocks: ConfirmationBlocks
    exp40_target_length: int | None = None
    transductive_target_length: int | None = None
    transductive_value_preprocessing: bool = True

    @property
    def exp84_family_guard(self) -> FullFitScoreView:
        rate = .03 if self.family in WEAK_FAMILIES_PAPER or self.test_median_length >= 512 else .02
        return FullFitScoreView("exp84_family_guard", self.exp84.train_score, self.exp84.test_score, rate)

    @property
    def exp84_count_cap_3pct(self) -> FullFitScoreView:
        return FullFitScoreView("exp84_count_cap_3pct", self.exp84.train_score, self.exp84.test_score, .03)


def _fullfit_primary(name: str, train: np.ndarray, test: np.ndarray, target: int) -> FullFitScoreView:
    if name == "rocket_exp40":
        train_z, test_z = z_normalize_paper(train), z_normalize_paper(test)
        features, query = rocket_feature_pair_paper(train_z, test_z, target, 256)
        features, query = scale_feature_pair_paper(features, query)
        train_score, test_score = density_knn_score_pair_paper(features, query, 3, "local_gap")
    elif name == "exp55_best":
        train_score, test_score = imaging_score_pair_paper(train, test, EXP55_CONFIG_PAPER)
    elif name == "exp56_best":
        train_score, test_score = imaging_score_pair_paper(train, test, EXP56_CONFIG_PAPER)
    elif name == "exp84":
        train_score, test_score = exp84_score_pair_paper(z_normalize_paper(train), z_normalize_paper(test))
    else:
        raise ValueError(f"unknown full-fit primary source: {name}")
    rate = .03 if name in {"rocket_exp40", "exp55_best"} else .02
    return FullFitScoreView(name, np.asarray(train_score, dtype=float), np.asarray(test_score, dtype=float), rate)


def _confirmation_blocks(train: np.ndarray, test: np.ndarray) -> ConfirmationBlocks:
    train_z, test_z = z_normalize_paper(train), z_normalize_paper(test)
    reference = reference_train_paper(train_z, train_z.shape[1], seed_offset=130)
    kernels512 = make_kernels_paper(train_z.shape[1], num_kernels=512)
    train_features, test_features = rocket_transform_paper(reference, kernels512), rocket_transform_paper(test_z, kernels512)
    a_train, a_test = density_knn_score_pair_paper(train_features[:, :512], test_features[:, :512], 3, "local_gap")
    b_train, b_test = density_knn_score_pair_paper(train_features[:, 512:], test_features[:, 512:], 3, "local_gap")
    kernels768 = make_kernels_paper(train_z.shape[1], num_kernels=768)[512:768]
    c_train, c_test = density_knn_score_pair_paper(rocket_transform_paper(reference, kernels768), rocket_transform_paper(test_z, kernels768), 3, "local_gap")
    return ConfirmationBlocks(FullFitScoreView("block_a", a_train, a_test, .015), FullFitScoreView("block_b", b_train, b_test, .015), FullFitScoreView("block_c", c_train, c_test, .01))


def build_legacy_route_evidence(values: RawDataset) -> LegacyRouteEvidence:
    if len(values.train_rows) < 2 or not values.test_rows:
        raise ValueError("legacy full-fit route requires at least two TRAIN and one TEST row")
    if any(not np.isfinite(np.asarray(row, dtype=float)).all() for row in (*values.train_rows, *values.test_rows)):
        raise ValueError("legacy full-fit route requires finite raw values")
    exp40_target = max(8, int(values.metadata_length))
    all_lengths = [len(row) for row in (*values.train_rows, *values.test_rows)]
    transductive_target = max(8, int(round(float(np.median(all_lengths)))))
    exp40_train, exp40_test = align_series_lengths_paper(values.train_rows, exp40_target), align_series_lengths_paper(values.test_rows, exp40_target)
    route_train, route_test = align_series_lengths_paper(values.train_rows, transductive_target), align_series_lengths_paper(values.test_rows, transductive_target)
    primary = {"rocket_exp40": _fullfit_primary("rocket_exp40", exp40_train, exp40_test, exp40_target)}
    for name in ("exp55_best", "exp56_best", "exp84"):
        primary[name] = _fullfit_primary(name, route_train, route_test, transductive_target)
    return LegacyRouteEvidence(values.dataset, parse_family_paper(values.dataset), len(values.train_rows), len(values.test_rows), int(round(float(np.median([len(row) for row in values.test_rows])))), primary, primary["exp84"], _confirmation_blocks(exp40_train, exp40_test), exp40_target, transductive_target, True)


__all__ = ["ConfirmationBlocks", "FullFitScoreView", "LegacyRouteEvidence", "WEAK_FAMILIES_PAPER", "build_legacy_route_evidence", "count_cap_threshold_paper", "parse_family_paper"]
