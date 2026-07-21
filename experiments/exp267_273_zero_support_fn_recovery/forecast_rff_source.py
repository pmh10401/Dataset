from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.kernel_approximation import RBFSampler

from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle
from experiments.exp239_248_diverse_score_sources.foundation import crossfit_plan
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
    """Clip the frozen contexts so every retained context has one forecast origin."""
    length = int(target_length)
    horizon = forecast_horizon(length)
    maximum_context = length - horizon
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
class _ContextModel:
    context_length: int
    sampler: RBFSampler
    coefficients: np.ndarray


@dataclass(frozen=True)
class _ForecastModel:
    horizon: int
    contexts: tuple[_ContextModel, ...]
    train_residual_p95: float

    def absolute_residuals(self, rows: np.ndarray) -> tuple[np.ndarray, ...]:
        values = np.asarray(rows, dtype=float)
        per_trace: list[np.ndarray] = []
        for row in values:
            pieces = []
            for context in self.contexts:
                inputs, targets = _forecast_windows(
                    row,
                    context_length=context.context_length,
                    horizon=self.horizon,
                )
                prediction = context.sampler.transform(inputs) @ context.coefficients
                pieces.append(np.abs(targets - prediction))
            residuals = np.concatenate(pieces)
            if not len(residuals) or not np.isfinite(residuals).all():
                raise ValueError("non-finite forecast residuals")
            per_trace.append(residuals)
        return tuple(per_trace)

    def trace_features(self, rows: np.ndarray) -> np.ndarray:
        residuals = self.absolute_residuals(rows)
        features = np.asarray(
            [
                (
                    float(np.median(trace)),
                    float(np.quantile(trace, 0.95)),
                    float(np.max(trace)),
                    _longest_run(trace > self.train_residual_p95),
                )
                for trace in residuals
            ],
            dtype=float,
        )
        if features.ndim != 2 or features.shape[1] != len(RESIDUAL_AGGREGATES):
            raise ValueError("unexpected forecast residual aggregate shape")
        if not np.isfinite(features).all():
            raise ValueError("non-finite forecast residual aggregate")
        return features


def _statistics_for_rows(
    rows: np.ndarray, *, contexts: tuple[int, ...], horizon: int
) -> tuple[_ContextStatistics, ...]:
    values = np.asarray(rows, dtype=float)
    statistics: list[_ContextStatistics] = []
    for context_length in contexts:
        sampler = RBFSampler(
            gamma=1.0 / float(context_length),
            n_components=RFF_COMPONENTS,
            random_state=RFF_RANDOM_SEED,
        )
        sampler.fit(np.zeros((1, context_length), dtype=float))
        gram = np.zeros((RFF_COMPONENTS, RFF_COMPONENTS), dtype=float)
        response = np.zeros(RFF_COMPONENTS, dtype=float)
        for row in values:
            inputs, targets = _forecast_windows(
                row, context_length=context_length, horizon=horizon
            )
            features = sampler.transform(inputs)
            gram += features.T @ features
            response += features.T @ targets
        if not np.isfinite(gram).all() or not np.isfinite(response).all():
            raise ValueError("non-finite RFF ridge sufficient statistics")
        statistics.append(
            _ContextStatistics(context_length, sampler, gram, response)
        )
    return tuple(statistics)


def _model_from_statistics(
    statistics: tuple[_ContextStatistics, ...], *, horizon: int, alpha: float
) -> _ForecastModel:
    context_models: list[_ContextModel] = []
    for item in statistics:
        system = item.gram + float(alpha) * np.eye(item.gram.shape[0])
        try:
            coefficients = np.linalg.solve(system, item.response)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.lstsq(system, item.response, rcond=None)[0]
        if not np.isfinite(coefficients).all():
            raise ValueError("non-finite RFF ridge coefficients")
        context_models.append(
            _ContextModel(item.context_length, item.sampler, coefficients)
        )
    provisional = _ForecastModel(int(horizon), tuple(context_models), 0.0)
    return provisional


def _fit_model(rows: np.ndarray, *, horizon: int, contexts: tuple[int, ...], alpha: float) -> _ForecastModel:
    statistics = _statistics_for_rows(rows, contexts=contexts, horizon=horizon)
    provisional = _model_from_statistics(statistics, horizon=horizon, alpha=alpha)
    train_residuals = np.concatenate(provisional.absolute_residuals(rows))
    threshold = float(np.quantile(train_residuals, 0.95))
    if not np.isfinite(threshold):
        raise ValueError("non-finite TRAIN forecast residual p95")
    return _ForecastModel(provisional.horizon, provisional.contexts, threshold)


def select_ridge_alpha(rows: np.ndarray, seed: int) -> tuple[float, dict[float, float]]:
    """Select a ridge alpha using only deterministic TRAIN-row cross-fitting."""
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("ridge selection requires at least two TRAIN rows")
    horizon = forecast_horizon(values.shape[1])
    contexts = forecast_context_lengths(values.shape[1])
    if not contexts:
        raise ValueError("ridge selection requires at least one forecast context")
    losses: dict[float, list[float]] = {alpha: [] for alpha in RIDGE_ALPHAS}
    for fit, query in crossfit_plan(len(values), seed=int(seed)):
        statistics = _statistics_for_rows(values[fit], contexts=contexts, horizon=horizon)
        for alpha in RIDGE_ALPHAS:
            model = _model_from_statistics(statistics, horizon=horizon, alpha=alpha)
            residuals = np.concatenate(model.absolute_residuals(values[query]))
            losses[alpha].append(float(np.mean(np.square(residuals))))
    mean_losses = {alpha: float(np.mean(losses[alpha])) for alpha in RIDGE_ALPHAS}
    if not all(np.isfinite(value) for value in mean_losses.values()):
        raise ValueError("non-finite TRAIN cross-fit ridge loss")
    selected = min(RIDGE_ALPHAS, key=lambda alpha: (mean_losses[alpha], alpha))
    return float(selected), mean_losses


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
        # The actual TRAIN matrix, rather than bundle metadata, owns this policy.
        target_length = int(raw_train.shape[1])
        train_rows = _validate_rows(
            raw_train, expected_length=target_length, name="TRAIN"
        )
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

        train_oof = np.empty(len(train_rows), dtype=float)
        selected_oof_alphas: list[float] = []
        outer_overlap_counts: list[int] = []
        alpha_query_fit_violations = 0
        for fold_index, (fit, query) in enumerate(crossfit_plan(len(train_rows), seed=int(seed))):
            overlap = int(len(np.intersect1d(fit, query)))
            outer_overlap_counts.append(overlap)
            nested_seed = int(seed) + fold_index + 1
            selected_alpha, _ = select_ridge_alpha(train_rows[fit], nested_seed)
            selected_oof_alphas.append(selected_alpha)
            for nested_fit, nested_query in crossfit_plan(len(fit), seed=nested_seed):
                alpha_query_fit_violations += int(
                    len(np.intersect1d(fit[nested_fit], fit[nested_query]))
                )
            model = _fit_model(
                train_rows[fit],
                horizon=horizon,
                contexts=contexts,
                alpha=selected_alpha,
            )
            fit_features = model.trace_features(train_rows[fit])
            query_features = model.trace_features(train_rows[query])
            standardized, _, _ = _robust_standardize_features(
                fit_features, query_features
            )
            train_oof[query] = standardized.sum(axis=1)

        selected_test_alpha, test_alpha_losses = select_ridge_alpha(train_rows, int(seed))
        full_model = _fit_model(
            train_rows,
            horizon=horizon,
            contexts=contexts,
            alpha=selected_test_alpha,
        )
        full_train_features = full_model.trace_features(train_rows)
        test_features = full_model.trace_features(test_rows)
        standardized_test, centers, scales = _robust_standardize_features(
            full_train_features, test_features
        )
        test_scores = standardized_test.sum(axis=1)
        if not np.isfinite(train_oof).all() or not np.isfinite(test_scores).all():
            raise ValueError("non-finite RFF normal forecast source scores")
        self_reference_violations = int(sum(outer_overlap_counts))
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
                "self_reference_violations": self_reference_violations,
                "outer_fit_query_overlap_counts": tuple(outer_overlap_counts),
                "alpha_selection_query_fit_violations": alpha_query_fit_violations,
                "selected_alpha_oof": tuple(selected_oof_alphas),
                "selected_alpha_test": selected_test_alpha,
                "selected_alpha_test_train_losses": test_alpha_losses,
                "full_train_residual_p95": full_model.train_residual_p95,
                "full_train_aggregate_centers": tuple(float(value) for value in centers),
                "full_train_aggregate_scales": tuple(float(value) for value in scales),
            },
        )
