from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.kernel_approximation import RBFSampler

from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle
from experiments.exp239_248_diverse_score_sources.foundation import deterministic_folds
from experiments.exp267_273_zero_support_fn_recovery.config import RFF_MIN_TRAIN, SEED
from experiments.exp267_273_zero_support_fn_recovery.contracts import SourceScore


RFF_COMPONENTS = 256
RFF_RANDOM_SEED = SEED
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0)
CONTEXT_REQUESTS = (8, 16, 32)
RESIDUAL_AGGREGATES = ("median", "p95", "maximum", "longest_run_above_train_p95")
_NUMERICAL_EPSILON = 1e-12


def forecast_horizon(target_length: int) -> int:
    """Return the frozen forecast horizon derived only from TRAIN target length."""
    length = int(target_length)
    if length < 1:
        raise ValueError("forecast target length must be positive")
    return max(1, min(8, length // 16))


def forecast_context_lengths(target_length: int) -> tuple[int, ...]:
    """Clip frozen contexts so each retained context has a forecast origin."""
    length = int(target_length)
    maximum_context = length - forecast_horizon(length)
    if maximum_context < 1:
        return ()
    return tuple(sorted({min(context, maximum_context) for context in CONTEXT_REQUESTS}))


def _validate_rows(rows: np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{name} rows must have shape (batch, time)")
    if values.shape[1] != int(expected_length):
        raise ValueError(f"{name} rows must match the train target length")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} rows must be finite")
    return values


def _forecast_windows(
    row: np.ndarray, *, context_length: int, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(row, dtype=float).reshape(-1)
    count = len(values) - int(context_length) - int(horizon) + 1
    if count < 1:
        raise ValueError("forecast context and horizon leave no prediction origins")
    contexts = np.lib.stride_tricks.sliding_window_view(values, int(context_length))[
        :count
    ]
    targets = values[int(context_length) + int(horizon) - 1 :]
    if contexts.shape != (count, int(context_length)) or targets.shape != (count,):
        raise ValueError("unexpected forecast window shape")
    return contexts, targets


def _longest_run(mask: np.ndarray) -> float:
    longest = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return float(longest)


def _robust_location_scale(values: Iterable[float]) -> tuple[float, float]:
    reference = np.asarray(list(values), dtype=float)
    if not len(reference) or not np.isfinite(reference).all():
        raise ValueError("robust reference must be finite and non-empty")
    location = float(np.median(reference))
    scale = 1.4826 * float(np.median(np.abs(reference - location)))
    if not np.isfinite(scale) or scale <= _NUMERICAL_EPSILON:
        q75, q25 = np.quantile(reference, (0.75, 0.25))
        scale = float((q75 - q25) / 1.349) if q75 > q25 else 0.0
    if not np.isfinite(scale) or scale <= _NUMERICAL_EPSILON:
        scale = float(np.std(reference))
    if not np.isfinite(scale) or scale <= _NUMERICAL_EPSILON:
        scale = 1.0
    return location, scale


def _robust_standardize_features(
    reference: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fit = np.asarray(reference, dtype=float)
    query = np.asarray(values, dtype=float)
    if fit.ndim != 2 or query.ndim != 2 or fit.shape[1] != query.shape[1]:
        raise ValueError("residual aggregate matrices must have matching columns")
    if not np.isfinite(fit).all() or not np.isfinite(query).all():
        raise ValueError("residual aggregate matrices must be finite")
    centers = np.empty(fit.shape[1], dtype=float)
    scales = np.empty(fit.shape[1], dtype=float)
    for index in range(fit.shape[1]):
        centers[index], scales[index] = _robust_location_scale(fit[:, index])
    standardized = (query - centers) / scales
    if not np.isfinite(standardized).all():
        raise ValueError("non-finite robust standardized residual aggregate")
    return standardized, centers, scales


@dataclass(frozen=True)
class _ContextStatistics:
    context_length: int
    sampler: RBFSampler
    gram: np.ndarray
    response: np.ndarray


@dataclass(frozen=True)
class _FoldGroupStatistics:
    indices: np.ndarray
    contexts: tuple[_ContextStatistics, ...]


@dataclass(frozen=True)
class _CoefficientContext:
    context_length: int
    sampler: RBFSampler
    coefficients: np.ndarray


@dataclass(frozen=True)
class _CoefficientFamily:
    horizon: int
    contexts: tuple[_CoefficientContext, ...]


def _make_sampler(context_length: int) -> RBFSampler:
    sampler = RBFSampler(
        gamma=1.0 / float(context_length),
        n_components=RFF_COMPONENTS,
        random_state=RFF_RANDOM_SEED,
    )
    sampler.fit(np.zeros((1, context_length), dtype=float))
    return sampler


def _build_fold_group_statistics(
    rows: np.ndarray,
    folds: tuple[np.ndarray, ...],
    *,
    contexts: tuple[int, ...],
    horizon: int,
) -> tuple[_FoldGroupStatistics, ...]:
    """Stream every TRAIN row once into its global fold's additive statistics."""
    groups: list[_FoldGroupStatistics] = []
    for indices in folds:
        entries: list[_ContextStatistics] = []
        for context_length in contexts:
            sampler = _make_sampler(context_length)
            gram = np.zeros((RFF_COMPONENTS, RFF_COMPONENTS), dtype=float)
            response = np.zeros(RFF_COMPONENTS, dtype=float)
            for row in rows[indices]:
                inputs, targets = _forecast_windows(
                    row, context_length=context_length, horizon=horizon
                )
                features = sampler.transform(inputs)
                gram += features.T @ features
                response += features.T @ targets
            if not np.isfinite(gram).all() or not np.isfinite(response).all():
                raise ValueError("non-finite RFF ridge sufficient statistics")
            entries.append(_ContextStatistics(context_length, sampler, gram, response))
        groups.append(_FoldGroupStatistics(np.asarray(indices, dtype=int), tuple(entries)))
    return tuple(groups)


def _combine_fold_statistics(
    groups: tuple[_FoldGroupStatistics, ...], excluded: frozenset[int]
) -> tuple[_ContextStatistics, ...]:
    included = [group for index, group in enumerate(groups) if index not in excluded]
    if not included:
        raise ValueError("at least one TRAIN fold group must remain fitted")
    output: list[_ContextStatistics] = []
    for context_index, template in enumerate(included[0].contexts):
        gram = np.add.reduce([group.contexts[context_index].gram for group in included])
        response = np.add.reduce(
            [group.contexts[context_index].response for group in included]
        )
        output.append(
            _ContextStatistics(template.context_length, template.sampler, gram, response)
        )
    return tuple(output)


def _coefficient_family(
    statistics: tuple[_ContextStatistics, ...], *, horizon: int
) -> _CoefficientFamily:
    """Reuse one symmetric eigendecomposition for all four frozen alphas."""
    contexts: list[_CoefficientContext] = []
    alpha = np.asarray(RIDGE_ALPHAS, dtype=float)
    for item in statistics:
        values, vectors = np.linalg.eigh(item.gram)
        values = np.maximum(values, 0.0)
        projected = vectors.T @ item.response
        coefficients = vectors @ (projected[:, None] / (values[:, None] + alpha))
        if not np.isfinite(coefficients).all():
            raise ValueError("non-finite RFF ridge coefficients")
        contexts.append(
            _CoefficientContext(item.context_length, item.sampler, coefficients)
        )
    return _CoefficientFamily(int(horizon), tuple(contexts))


class _FamilyCache:
    def __init__(
        self, groups: tuple[_FoldGroupStatistics, ...], *, horizon: int
    ) -> None:
        self.groups = groups
        self.horizon = int(horizon)
        self._items: dict[tuple[int, ...], _CoefficientFamily] = {}

    def get(self, excluded: Iterable[int]) -> _CoefficientFamily:
        key = tuple(sorted(set(int(value) for value in excluded)))
        if key not in self._items:
            statistics = _combine_fold_statistics(self.groups, frozenset(key))
            self._items[key] = _coefficient_family(statistics, horizon=self.horizon)
        return self._items[key]

    @property
    def eigendecompositions(self) -> int:
        return len(self._items) * len(self.groups[0].contexts)


def _all_alpha_loss(
    family: _CoefficientFamily, rows: np.ndarray
) -> tuple[np.ndarray, int]:
    """Transform each validation window once and score all alpha vectors together."""
    total = np.zeros(len(RIDGE_ALPHAS), dtype=float)
    count = 0
    for row in np.asarray(rows, dtype=float):
        for context in family.contexts:
            inputs, targets = _forecast_windows(
                row,
                context_length=context.context_length,
                horizon=family.horizon,
            )
            prediction = context.sampler.transform(inputs) @ context.coefficients
            residual = targets[:, None] - prediction
            total += np.square(residual).sum(axis=0)
            count += len(targets)
    if count < 1 or not np.isfinite(total).all():
        raise ValueError("non-finite TRAIN cross-fit ridge loss")
    return total, count


def _select_alpha(loss_total: np.ndarray, count: int) -> tuple[float, dict[float, float]]:
    if count < 1:
        raise ValueError("ridge selection requires validation residuals")
    losses = np.asarray(loss_total, dtype=float) / float(count)
    if not np.isfinite(losses).all():
        raise ValueError("non-finite TRAIN cross-fit ridge loss")
    by_alpha = {alpha: float(losses[index]) for index, alpha in enumerate(RIDGE_ALPHAS)}
    selected = min(RIDGE_ALPHAS, key=lambda alpha: (by_alpha[alpha], alpha))
    return float(selected), by_alpha


def _selected_alpha_index(alpha: float) -> int:
    for index, candidate in enumerate(RIDGE_ALPHAS):
        if float(candidate) == float(alpha):
            return index
    raise ValueError(f"unknown frozen ridge alpha: {alpha}")


def _residual_threshold(
    family: _CoefficientFamily, alpha_index: int, row_groups: Sequence[np.ndarray]
) -> float:
    pieces: list[np.ndarray] = []
    for rows in row_groups:
        for row in np.asarray(rows, dtype=float):
            for context in family.contexts:
                inputs, targets = _forecast_windows(
                    row,
                    context_length=context.context_length,
                    horizon=family.horizon,
                )
                prediction = context.sampler.transform(inputs) @ context.coefficients[:, alpha_index]
                pieces.append(np.abs(targets - prediction))
    if not pieces:
        raise ValueError("TRAIN residual p95 requires fitted TRAIN rows")
    threshold = float(np.quantile(np.concatenate(pieces), 0.95))
    if not np.isfinite(threshold):
        raise ValueError("non-finite TRAIN forecast residual p95")
    return threshold


def _trace_features(
    family: _CoefficientFamily,
    alpha_index: int,
    threshold: float,
    rows: np.ndarray,
) -> np.ndarray:
    values = np.asarray(rows, dtype=float)
    if not len(values):
        return np.empty((0, len(RESIDUAL_AGGREGATES)), dtype=float)
    output: list[tuple[float, float, float, float]] = []
    for row in values:
        pieces = []
        for context in family.contexts:
            inputs, targets = _forecast_windows(
                row,
                context_length=context.context_length,
                horizon=family.horizon,
            )
            prediction = context.sampler.transform(inputs) @ context.coefficients[:, alpha_index]
            pieces.append(np.abs(targets - prediction))
        residuals = np.concatenate(pieces)
        if not len(residuals) or not np.isfinite(residuals).all():
            raise ValueError("non-finite forecast residuals")
        output.append(
            (
                float(np.median(residuals)),
                float(np.quantile(residuals, 0.95)),
                float(np.max(residuals)),
                _longest_run(residuals > threshold),
            )
        )
    features = np.asarray(output, dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("non-finite forecast residual aggregate")
    return features


def _rows_for_groups(
    rows: np.ndarray, folds: tuple[np.ndarray, ...], excluded: Iterable[int]
) -> tuple[np.ndarray, ...]:
    excluded_indices = frozenset(int(value) for value in excluded)
    return tuple(
        rows[indices] for index, indices in enumerate(folds) if index not in excluded_indices
    )


def _global_alpha_selection(
    cache: _FamilyCache, rows: np.ndarray, folds: tuple[np.ndarray, ...]
) -> tuple[float, dict[float, float], int]:
    loss_total = np.zeros(len(RIDGE_ALPHAS), dtype=float)
    count = 0
    for fold_index, indices in enumerate(folds):
        total, current_count = _all_alpha_loss(cache.get((fold_index,)), rows[indices])
        loss_total += total
        count += current_count
    alpha, losses = _select_alpha(loss_total, count)
    return alpha, losses, len(folds)


def select_ridge_alpha(rows: np.ndarray, seed: int) -> tuple[float, dict[float, float]]:
    """Select a frozen alpha with a deterministic global TRAIN-row partition."""
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("ridge selection requires at least two TRAIN rows")
    horizon = forecast_horizon(values.shape[1])
    contexts = forecast_context_lengths(values.shape[1])
    if not contexts:
        raise ValueError("ridge selection requires at least one forecast context")
    folds = deterministic_folds(len(values), seed=int(seed))
    groups = _build_fold_group_statistics(
        values, folds, contexts=contexts, horizon=horizon
    )
    alpha, losses, _ = _global_alpha_selection(_FamilyCache(groups, horizon=horizon), values, folds)
    return alpha, losses


class RFFNormalForecastResidual:
    """TRAIN-normal RFF ridge forecasting residual score for Exp270-A."""

    name = "rff_normal_forecast_residual"
    minimum_train = RFF_MIN_TRAIN

    def score(
        self, bundle: DatasetBundle, seed: int, artifact_dir: Path
    ) -> SourceScore:
        del artifact_dir
        raw_train = np.asarray(bundle.train_normal, dtype=float)
        if raw_train.ndim != 2:
            raise ValueError("TRAIN rows must have shape (batch, time)")
        target_length = int(raw_train.shape[1])
        train_rows = _validate_rows(raw_train, expected_length=target_length, name="TRAIN")
        test_rows = _validate_rows(
            bundle.test_values, expected_length=target_length, name="TEST"
        )
        contexts = forecast_context_lengths(target_length)
        horizon = forecast_horizon(target_length)
        common_provenance = {
            "seed": int(seed),
            "rff_random_seed": RFF_RANDOM_SEED,
            "n_train": int(len(train_rows)),
            "n_test": int(len(test_rows)),
            "minimum_train": self.minimum_train,
            "target_length": target_length,
            "target_length_from_train_only": True,
            "context_requests": CONTEXT_REQUESTS,
            "context_lengths": contexts,
            "forecast_horizon": horizon,
            "rff_components": RFF_COMPONENTS,
            "rff_gamma_rule": "1/context_length",
            "ridge_alphas": RIDGE_ALPHAS,
            "residual_aggregates": RESIDUAL_AGGREGATES,
            "residual_run_threshold": "TRAIN residual p95",
            "standardization": "TRAIN fit median/MAD with IQR/std/unit fallbacks",
            "aggregate_weights": (1.0, 1.0, 1.0, 1.0),
            "contains_test_labels": False,
        }
        if not contexts:
            return SourceScore(
                source=self.name,
                variant="primary",
                train_oof=np.asarray([], dtype=float),
                test_scores=np.asarray([], dtype=float),
                eligible=False,
                abstained=True,
                abstention_reason="target_length_below_2",
                provenance={
                    **common_provenance,
                    "cross_fitted": False,
                    "self_reference_violations": 0,
                    "alpha_selection_query_fit_violations": 0,
                },
            )
        if len(train_rows) < self.minimum_train:
            return SourceScore(
                source=self.name,
                variant="primary",
                train_oof=np.asarray([], dtype=float),
                test_scores=np.asarray([], dtype=float),
                eligible=False,
                abstained=True,
                abstention_reason=f"n_train_below_{self.minimum_train}",
                provenance={
                    **common_provenance,
                    "cross_fitted": False,
                    "self_reference_violations": 0,
                    "alpha_selection_query_fit_violations": 0,
                },
            )

        folds = deterministic_folds(len(train_rows), seed=int(seed))
        groups = _build_fold_group_statistics(
            train_rows, folds, contexts=contexts, horizon=horizon
        )
        cache = _FamilyCache(groups, horizon=horizon)

        selected_outer_alpha: list[float] = []
        selected_outer_losses: list[dict[float, float]] = []
        nested_violation_count = 0
        all_alpha_validation_calls = 0
        for outer_index in range(len(folds)):
            total = np.zeros(len(RIDGE_ALPHAS), dtype=float)
            count = 0
            for query_index in range(len(folds)):
                if query_index == outer_index:
                    continue
                family = cache.get((outer_index, query_index))
                loss, current_count = _all_alpha_loss(
                    family, train_rows[folds[query_index]]
                )
                total += loss
                count += current_count
                all_alpha_validation_calls += 1
                fit_indices = np.concatenate(
                    [
                        folds[index]
                        for index in range(len(folds))
                        if index not in (outer_index, query_index)
                    ]
                )
                nested_violation_count += int(
                    len(np.intersect1d(fit_indices, folds[query_index]))
                )
            alpha, losses = _select_alpha(total, count)
            selected_outer_alpha.append(alpha)
            selected_outer_losses.append(losses)

        selected_test_alpha, test_alpha_losses, final_alpha_calls = _global_alpha_selection(
            cache, train_rows, folds
        )
        all_alpha_validation_calls += final_alpha_calls

        train_oof = np.empty(len(train_rows), dtype=float)
        outer_overlap_counts: list[int] = []
        for outer_index, indices in enumerate(folds):
            alpha_index = _selected_alpha_index(selected_outer_alpha[outer_index])
            reference_pieces: list[np.ndarray] = []
            for inner_query in range(len(folds)):
                if inner_query == outer_index:
                    continue
                inner_family = cache.get((outer_index, inner_query))
                inner_fit_rows = _rows_for_groups(
                    train_rows, folds, (outer_index, inner_query)
                )
                threshold = _residual_threshold(
                    inner_family, alpha_index, inner_fit_rows
                )
                reference_pieces.append(
                    _trace_features(
                        inner_family,
                        alpha_index,
                        threshold,
                        train_rows[folds[inner_query]],
                    )
                )
            reference = np.vstack(reference_pieces)
            outer_family = cache.get((outer_index,))
            outer_fit_rows = _rows_for_groups(train_rows, folds, (outer_index,))
            outer_threshold = _residual_threshold(
                outer_family, alpha_index, outer_fit_rows
            )
            query_features = _trace_features(
                outer_family,
                alpha_index,
                outer_threshold,
                train_rows[indices],
            )
            standardized, _, _ = _robust_standardize_features(reference, query_features)
            train_oof[indices] = standardized.sum(axis=1)
            outer_fit_indices = np.concatenate(
                [folds[index] for index in range(len(folds)) if index != outer_index]
            )
            outer_overlap_counts.append(
                int(len(np.intersect1d(outer_fit_indices, indices)))
            )

        final_alpha_index = _selected_alpha_index(selected_test_alpha)
        final_reference_pieces: list[np.ndarray] = []
        for query_index, indices in enumerate(folds):
            family = cache.get((query_index,))
            fit_rows = _rows_for_groups(train_rows, folds, (query_index,))
            threshold = _residual_threshold(family, final_alpha_index, fit_rows)
            final_reference_pieces.append(
                _trace_features(
                    family,
                    final_alpha_index,
                    threshold,
                    train_rows[indices],
                )
            )
        final_reference = np.vstack(final_reference_pieces)
        full_family = cache.get(())
        full_train_threshold = _residual_threshold(
            full_family, final_alpha_index, (train_rows,)
        )
        test_features = _trace_features(
            full_family, final_alpha_index, full_train_threshold, test_rows
        )
        standardized_test, centers, scales = _robust_standardize_features(
            final_reference, test_features
        )
        test_scores = standardized_test.sum(axis=1)
        if not np.isfinite(train_oof).all() or not np.isfinite(test_scores).all():
            raise ValueError("non-finite RFF normal forecast source scores")

        return SourceScore(
            source=self.name,
            variant="primary",
            train_oof=train_oof,
            test_scores=test_scores,
            eligible=True,
            abstained=False,
            abstention_reason=None,
            provenance={
                **common_provenance,
                "cross_fitted": True,
                "self_reference_violations": int(sum(outer_overlap_counts)),
                "outer_fit_query_overlap_counts": tuple(outer_overlap_counts),
                "alpha_selection_query_fit_violations": nested_violation_count,
                "selected_alpha_oof": tuple(selected_outer_alpha),
                "selected_alpha_oof_train_losses": tuple(selected_outer_losses),
                "selected_alpha_test": selected_test_alpha,
                "selected_alpha_test_train_losses": test_alpha_losses,
                "full_train_residual_p95": full_train_threshold,
                "full_train_aggregate_centers": tuple(float(value) for value in centers),
                "full_train_aggregate_scales": tuple(float(value) for value in scales),
                "global_fold_count": len(folds),
                "fold_group_statistics_builds": len(groups),
                "fold_group_context_statistics_builds": len(groups) * len(contexts),
                "statistics_train_row_transforms": len(train_rows) * len(contexts),
                "alpha_candidate_count": len(RIDGE_ALPHAS),
                "ridge_eigendecompositions": cache.eigendecompositions,
                "all_alpha_validation_calls": all_alpha_validation_calls,
            },
        )
