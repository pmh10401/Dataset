"""Dataset-local A--D hard-score integration for the Exp450 ablation.

Condition A calls the frozen Exp449-compatible paper closures directly.  The
remaining conditions retain those closures' constants, fold plans, repair,
and selector algebra, replacing only series alignment and Exp55/56 image
construction with the approved factor pair.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from paper.equipment_anomaly_detection.calibration_paper import upper_tail_pvalues
from paper.equipment_anomaly_detection.contracts_paper import RawDataset
from paper.equipment_anomaly_detection.hard_route_paper import (
    HardRouteResult,
    RouteLanes,
    empty_hard_route_result_paper,
    exact_normal_guard,
    minimum_source_expansion,
    replay_exp133_134_135_137,
    replay_exp74d_89_90_93,
    replay_hard_route,
    union_three_source_hard,
)
from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import (
    EXPECTED_HARD_SOURCES,
    RAW_PRIMARY_SOURCES,
    HardSourceBundle,
    _abstain_paper,
    _canonical_source_paper,
    _completed_paper,
    crossfit_confirmation_scores_paper,
    fold_plan_paper,
    score_hard_sources,
)
from paper.equipment_anomaly_detection.sources_paper.imaging_paper import (
    EXP55_CONFIG_PAPER,
    EXP56_CONFIG_PAPER,
)
from paper.equipment_anomaly_detection.sources_paper.knn_paper import (
    density_knn_score_pair_paper,
    scale_feature_pair_paper,
)
from paper.equipment_anomaly_detection.sources_paper.legacy_route_sources_paper import (
    ConfirmationBlocks,
    FullFitScoreView,
    LegacyRouteEvidence,
    _repair_rows_for_legacy_fullfit_paper,
    parse_family_paper,
)
from paper.equipment_anomaly_detection.sources_paper.minirocket_paper import (
    MINIMUM_TRAIN_PAPER,
    SEEDS_PAPER,
    MiniRocketConfig,
    MiniRocketKernelResult,
    MiniRocketResult,
    _repair_rows_for_minirocket_paper,
    score_configuration_paper,
    score_minirocket_confirmations,
)
from paper.equipment_anomaly_detection.sources_paper.multirocket_hydra_paper import (
    exp84_score_pair_paper,
)
from paper.equipment_anomaly_detection.sources_paper.preprocessing_paper import (
    DependencyUnavailablePaper,
    has_only_finite_values_paper,
    source_metadata_paper,
    z_normalize_paper,
)
from paper.equipment_anomaly_detection.sources_paper.rocket_paper import (
    make_kernels_paper,
    reference_train_paper,
    rocket_feature_pair_paper,
    rocket_transform_paper,
)

from .contracts_antigravity import (
    Exp450BackendDecision,
    Exp450Condition,
    Exp450DatasetConditionResult,
)
from .imaging_antigravity import imaging_score_pair_exp450_antigravity
from .transforms_antigravity import align_rows_exp450_antigravity


AlignmentRowsAntigravity = Callable[[Sequence[np.ndarray], int], np.ndarray]


def _align_rows_selected_antigravity(
    rows: Sequence[np.ndarray],
    target_len: int,
    condition: Exp450Condition,
    align_rows_override: AlignmentRowsAntigravity | None,
) -> np.ndarray:
    """Select the frozen Exp450 alignment unless an isolated caller supplies one."""
    if align_rows_override is None:
        return align_rows_exp450_antigravity(
            rows, target_len, mode=condition.length_mode
        )
    return np.asarray(align_rows_override(rows, target_len), dtype=np.float32)


def _historical_pair_exp450_antigravity(
    name: str,
    fit: np.ndarray,
    query: np.ndarray,
    target_len: int,
    *,
    condition: Exp450Condition,
    backend: Exp450BackendDecision,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one retained primary source with only the declared image factor."""
    if name == "rocket_exp40":
        fit_z, query_z = z_normalize_paper(fit), z_normalize_paper(query)
        fit_features, query_features = rocket_feature_pair_paper(
            fit_z, query_z, fit_z.shape[1], 256
        )
        fit_features, query_features = scale_feature_pair_paper(
            fit_features, query_features
        )
        return density_knn_score_pair_paper(
            fit_features, query_features, 3, "local_gap"
        )
    if name == "exp55_best":
        return imaging_score_pair_exp450_antigravity(
            fit,
            query,
            EXP55_CONFIG_PAPER,
            image_mode=condition.image_mode,
            device=backend.tensor_backend,
        )
    if name == "exp56_best":
        return imaging_score_pair_exp450_antigravity(
            fit,
            query,
            EXP56_CONFIG_PAPER,
            image_mode=condition.image_mode,
            device=backend.tensor_backend,
        )
    if name == "exp84":
        return exp84_score_pair_paper(z_normalize_paper(fit), z_normalize_paper(query))
    raise ValueError(f"unknown historical source {name}")


def _crossfit_source_exp450_antigravity(
    name: str,
    train: np.ndarray,
    test: np.ndarray,
    target_len: int,
    *,
    condition: Exp450Condition,
    backend: Exp450BackendDecision,
) -> tuple[np.ndarray, np.ndarray]:
    oof = np.full(len(train), np.nan, dtype=float)
    for fit_indices, query_indices in fold_plan_paper(len(train)):
        _, scores = _historical_pair_exp450_antigravity(
            name,
            train[fit_indices],
            train[query_indices],
            target_len,
            condition=condition,
            backend=backend,
        )
        oof[query_indices] = scores
    _, test_scores = _historical_pair_exp450_antigravity(
        name, train, test, target_len, condition=condition, backend=backend
    )
    if not np.all(np.isfinite(oof)) or not np.all(np.isfinite(test_scores)):
        raise ValueError("nonfinite_score")
    return oof, np.asarray(test_scores, dtype=float)


def score_hard_sources_exp450_antigravity(
    values: RawDataset,
    condition: Exp450Condition,
    backend: Exp450BackendDecision,
    *,
    align_rows_override: AlignmentRowsAntigravity | None = None,
) -> HardSourceBundle:
    """Score six frozen Hard sources with the approved Exp450 factor pair."""
    if condition.condition_id == "A":
        return score_hard_sources(values)
    if len(values.train_rows) < 5:
        return HardSourceBundle(
            EXPECTED_HARD_SOURCES,
            {
                name: _abstain_paper(
                    name, len(values.test_rows), "unsupported_n_train_lt_5"
                )
                for name in EXPECTED_HARD_SOURCES
            },
        )
    if not has_only_finite_values_paper(values):
        return HardSourceBundle(
            EXPECTED_HARD_SOURCES,
            {
                name: _abstain_paper(name, len(values.test_rows), "nonfinite_input")
                for name in EXPECTED_HARD_SOURCES
            },
        )
    metadata = source_metadata_paper(values)
    train = _align_rows_selected_antigravity(
        values.train_rows,
        metadata.train_target_len,
        condition,
        align_rows_override,
    )
    test = _align_rows_selected_antigravity(
        values.test_rows,
        metadata.train_target_len,
        condition,
        align_rows_override,
    )
    results = {}
    for name in RAW_PRIMARY_SOURCES:
        try:
            results[name] = _completed_paper(
                name,
                *_crossfit_source_exp450_antigravity(
                    name,
                    train,
                    test,
                    metadata.train_target_len,
                    condition=condition,
                    backend=backend,
                ),
            )
        except DependencyUnavailablePaper as exc:
            results[name] = _abstain_paper(name, len(test), str(exc))
    try:
        results["rocket_block_b"] = _completed_paper(
            "rocket_block_b", *crossfit_confirmation_scores_paper(train, test)
        )
    except DependencyUnavailablePaper as exc:
        results["rocket_block_b"] = _abstain_paper(
            "rocket_block_b", len(test), str(exc)
        )
    results["canonical_normal_state"] = _completed_paper(
        "canonical_normal_state", *_canonical_source_paper(train, test)
    )
    return HardSourceBundle(EXPECTED_HARD_SOURCES, results)


def score_minirocket_exp450_antigravity(
    values: RawDataset,
    condition: Exp450Condition,
    backend: Exp450BackendDecision,
    *,
    align_rows_override: AlignmentRowsAntigravity | None = None,
) -> MiniRocketResult:
    """Score frozen MiniROCKET confirmations with only Exp450 alignment changed."""
    del backend
    if condition.condition_id == "A":
        return score_minirocket_confirmations(values)
    if len(values.train_rows) < MINIMUM_TRAIN_PAPER:
        empty = {
            requested: MiniRocketKernelResult(requested, dimension, {}, frozenset())
            for requested, dimension in ((1024, 1008), (4096, 4032))
        }
        return MiniRocketResult(frozenset(), frozenset(), empty)
    target = int(np.clip(np.median([len(row) for row in values.train_rows]), 8, 2048))
    train_rows = _repair_rows_for_minirocket_paper(
        values.train_rows, dataset=values.dataset, split="TRAIN"
    )
    test_rows = _repair_rows_for_minirocket_paper(
        values.test_rows, dataset=values.dataset, split="TEST"
    )
    train = z_normalize_paper(
        _align_rows_selected_antigravity(
            train_rows, target, condition, align_rows_override
        )
    )
    test = z_normalize_paper(
        _align_rows_selected_antigravity(
            test_rows, target, condition, align_rows_override
        )
    )
    by_kernel: dict[int, MiniRocketKernelResult] = {}
    for requested, expected in ((1024, 1008), (4096, 4032)):
        scores = {
            seed: score_configuration_paper(
                train,
                test,
                config=MiniRocketConfig(),
                requested_kernels=requested,
                seed=seed,
                expected_dimension=expected,
            )
            for seed in SEEDS_PAPER
        }
        candidates = [
            frozenset(
                np.flatnonzero(
                    upper_tail_pvalues(score.train_oof, score.test_score) <= 0.005
                )
                .astype(int)
                .tolist()
            )
            for score in scores.values()
        ]
        by_kernel[requested] = MiniRocketKernelResult(
            requested, expected, scores, frozenset.intersection(*candidates)
        )
    return MiniRocketResult(
        by_kernel[1024].candidate, by_kernel[4096].candidate, by_kernel
    )


score_minirocket_confirmations_exp450_antigravity = score_minirocket_exp450_antigravity


def _fullfit_primary_exp450_antigravity(
    name: str,
    train: np.ndarray,
    test: np.ndarray,
    target: int,
    *,
    condition: Exp450Condition,
    backend: Exp450BackendDecision,
) -> FullFitScoreView:
    if name == "rocket_exp40":
        train_z, test_z = z_normalize_paper(train), z_normalize_paper(test)
        features, query = rocket_feature_pair_paper(train_z, test_z, target, 256)
        train_score, test_score = density_knn_score_pair_paper(
            features, query, 3, "local_gap"
        )
    elif name == "exp55_best":
        train_score, test_score = imaging_score_pair_exp450_antigravity(
            train,
            test,
            EXP55_CONFIG_PAPER,
            image_mode=condition.image_mode,
            device=backend.tensor_backend,
        )
    elif name == "exp56_best":
        train_score, test_score = imaging_score_pair_exp450_antigravity(
            train,
            test,
            EXP56_CONFIG_PAPER,
            image_mode=condition.image_mode,
            device=backend.tensor_backend,
        )
    elif name == "exp84":
        train_score, test_score = exp84_score_pair_paper(
            z_normalize_paper(train), z_normalize_paper(test)
        )
    else:
        raise ValueError(f"unknown full-fit primary source: {name}")
    return FullFitScoreView(
        name, np.asarray(train_score, dtype=float), np.asarray(test_score, dtype=float), 0.015
    )


def _confirmation_blocks_exp450_antigravity(
    train: np.ndarray, test: np.ndarray
) -> ConfirmationBlocks:
    train_z, test_z = z_normalize_paper(train), z_normalize_paper(test)
    reference = reference_train_paper(train_z, train_z.shape[1], seed_offset=130)
    kernels512 = make_kernels_paper(train_z.shape[1], num_kernels=512)
    train_features, test_features = (
        rocket_transform_paper(reference, kernels512),
        rocket_transform_paper(test_z, kernels512),
    )
    a_train, a_test = density_knn_score_pair_paper(
        train_features[:, :512], test_features[:, :512], 3, "local_gap"
    )
    b_train, b_test = density_knn_score_pair_paper(
        train_features[:, 512:], test_features[:, 512:], 3, "local_gap"
    )
    kernels768 = make_kernels_paper(train_z.shape[1], num_kernels=768)[512:768]
    c_train, c_test = density_knn_score_pair_paper(
        rocket_transform_paper(reference, kernels768),
        rocket_transform_paper(test_z, kernels768),
        3,
        "local_gap",
    )
    return ConfirmationBlocks(
        FullFitScoreView("block_a", a_train, a_test, 0.015),
        FullFitScoreView("block_b", b_train, b_test, 0.015),
        FullFitScoreView("block_b", b_train, b_test, 0.01),
        FullFitScoreView("block_c", c_train, c_test, 0.01),
    )


def build_legacy_route_evidence_exp450_antigravity(
    values: RawDataset,
    condition: Exp450Condition,
    backend: Exp450BackendDecision,
    *,
    align_rows_override: AlignmentRowsAntigravity | None = None,
) -> LegacyRouteEvidence:
    """Build frozen route evidence while applying only the approved factors."""
    if condition.condition_id == "A":
        from paper.equipment_anomaly_detection.sources_paper.legacy_route_sources_paper import (
            build_legacy_route_evidence,
        )

        return build_legacy_route_evidence(values)
    if len(values.train_rows) < 2 or not values.test_rows:
        raise ValueError("legacy full-fit route requires at least two TRAIN and one TEST row")
    train_work = _repair_rows_for_legacy_fullfit_paper(
        values.train_rows, dataset=values.dataset, split="TRAIN"
    )
    test_work = _repair_rows_for_legacy_fullfit_paper(
        values.test_rows, dataset=values.dataset, split="TEST"
    )
    exp40_target = max(8, int(values.metadata_length))
    all_lengths = [len(row) for row in (*train_work, *test_work)]
    transductive_target = min(
        max(8, int(round(float(np.median(all_lengths))))), 2048
    )
    exp40_train = _align_rows_selected_antigravity(
        train_work, exp40_target, condition, align_rows_override
    )
    exp40_test = _align_rows_selected_antigravity(
        test_work, exp40_target, condition, align_rows_override
    )
    route_train = _align_rows_selected_antigravity(
        train_work, transductive_target, condition, align_rows_override
    )
    route_test = _align_rows_selected_antigravity(
        test_work, transductive_target, condition, align_rows_override
    )
    primary = {
        "rocket_exp40": _fullfit_primary_exp450_antigravity(
            "rocket_exp40",
            exp40_train,
            exp40_test,
            exp40_target,
            condition=condition,
            backend=backend,
        )
    }
    for name in ("exp55_best", "exp56_best"):
        primary[name] = _fullfit_primary_exp450_antigravity(
            name,
            route_train,
            route_test,
            transductive_target,
            condition=condition,
            backend=backend,
        )
    exp84 = _fullfit_primary_exp450_antigravity(
        "exp84",
        route_train,
        route_test,
        transductive_target,
        condition=condition,
        backend=backend,
    )
    return LegacyRouteEvidence(
        values.dataset,
        parse_family_paper(values.dataset),
        len(values.train_rows),
        len(values.test_rows),
        int(round(float(np.median([len(row) for row in values.test_rows])))),
        primary,
        exp84,
        _confirmation_blocks_exp450_antigravity(exp40_train, exp40_test),
        exp40_target,
        transductive_target,
        True,
    )


def replay_hard_route_exp450_antigravity(
    values: RawDataset,
    sources: HardSourceBundle,
    mini: MiniRocketResult,
    condition: Exp450Condition,
    backend: Exp450BackendDecision,
    *,
    align_rows_override: AlignmentRowsAntigravity | None = None,
) -> HardRouteResult:
    """Replay the exact frozen selectors over condition-local score evidence."""
    if condition.condition_id == "A":
        return replay_hard_route(values, sources, mini)
    if len(values.train_rows) < 2 or not values.test_rows:
        return empty_hard_route_result_paper(len(values.test_rows))
    evidence = build_legacy_route_evidence_exp450_antigravity(
        values,
        condition,
        backend,
        align_rows_override=align_rows_override,
    )
    exp74d, exp89, exp90, exp93 = replay_exp74d_89_90_93(evidence)
    exp137 = replay_exp133_134_135_137(exp93, evidence.confirmation_blocks)
    exp234 = minimum_source_expansion(exp137, sources, alpha=0.01)
    exp236 = union_three_source_hard(exp137, exp234)
    promoted = frozenset(
        set(exp236.standard) & (set(mini.k1024) | set(mini.k4096))
    )
    exp253 = RouteLanes(
        frozenset(set(exp236.hard) | set(promoted)),
        frozenset(set(exp236.standard) - set(promoted)),
        exp236.priority,
    )
    automatic, removed = exact_normal_guard(
        exp253.hard, values.train_blobs, values.test_blobs
    )
    return HardRouteResult(
        exp74d,
        exp89,
        exp90,
        exp93,
        exp137,
        exp234,
        exp236,
        promoted,
        exp253,
        removed,
        automatic,
        len(values.test_rows),
    )


def score_dataset_condition_exp450_antigravity(
    values: RawDataset,
    condition: Exp450Condition,
    backend: Exp450BackendDecision,
) -> Exp450DatasetConditionResult:
    """Produce one typed, values-only result for one dataset and condition."""
    return score_dataset_condition_with_alignment_exp450_antigravity(
        values, condition, backend, align_rows_override=None
    )


def score_dataset_condition_with_alignment_exp450_antigravity(
    values: RawDataset,
    condition: Exp450Condition,
    backend: Exp450BackendDecision,
    *,
    align_rows_override: AlignmentRowsAntigravity | None,
) -> Exp450DatasetConditionResult:
    """Produce one condition result with an optional isolated row aligner."""
    if condition.condition_id == "A":
        sources = score_hard_sources(values)
        mini = score_minirocket_confirmations(values)
        route = replay_hard_route(values, sources, mini)
    else:
        sources = score_hard_sources_exp450_antigravity(
            values,
            condition,
            backend,
            align_rows_override=align_rows_override,
        )
        mini = score_minirocket_confirmations_exp450_antigravity(
            values,
            condition,
            backend,
            align_rows_override=align_rows_override,
        )
        route = replay_hard_route_exp450_antigravity(
            values,
            sources,
            mini,
            condition,
            backend,
            align_rows_override=align_rows_override,
        )
    return Exp450DatasetConditionResult(values.dataset, condition.condition_id, sources, route)


__all__ = [
    "build_legacy_route_evidence_exp450_antigravity",
    "replay_hard_route_exp450_antigravity",
    "score_dataset_condition_exp450_antigravity",
    "score_dataset_condition_with_alignment_exp450_antigravity",
    "score_hard_sources_exp450_antigravity",
    "score_minirocket_exp450_antigravity",
]
