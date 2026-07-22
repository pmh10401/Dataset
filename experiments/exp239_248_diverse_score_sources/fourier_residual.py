from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle, ScoreResult
from experiments.exp239_248_diverse_score_sources.foundation import crossfit_plan
from experiments.exp239_248_diverse_score_sources.robust_distance import RobustDistanceModel
from experiments.exp239_248_diverse_score_sources.source_base import abstained_result


RIDGE_GRID = (1e-4, 1e-3, 1e-2, 1e-1)


def _harmonics(reference: np.ndarray, count: int = 8) -> np.ndarray:
    energy = np.mean(np.abs(np.fft.rfft(reference, axis=1)) ** 2, axis=0)
    candidates = np.arange(1, min(17, len(energy)))
    ranked = candidates[np.argsort(energy[candidates])[::-1]]
    return np.sort(ranked[: min(count, len(ranked))]).astype(int)


def _design(length: int, harmonics: np.ndarray) -> np.ndarray:
    time = np.arange(length, dtype=float) / max(1, length)
    columns = [np.ones(length)]
    for frequency in harmonics:
        columns.extend([np.sin(2 * np.pi * frequency * time), np.cos(2 * np.pi * frequency * time)])
    return np.column_stack(columns)


def _fit_rows(rows: np.ndarray, design: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    penalty = np.eye(design.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    projector = np.linalg.solve(design.T @ design + penalty, design.T)
    coefficients = np.asarray(rows, dtype=float) @ projector.T
    reconstructed = coefficients @ design.T
    residual = np.mean((np.asarray(rows, dtype=float) - reconstructed) ** 2, axis=1)
    return coefficients, residual


def _select_ridge(reference: np.ndarray, harmonics: np.ndarray) -> float:
    design = _design(reference.shape[1], harmonics)
    losses = {}
    for ridge in RIDGE_GRID:
        _, residual = _fit_rows(reference, design, ridge)
        coefficient_norm = np.mean(_fit_rows(reference, design, ridge)[0] ** 2, axis=1)
        losses[ridge] = float(np.median(residual) + 1e-4 * np.median(coefficient_norm))
    return min(RIDGE_GRID, key=lambda value: (losses[value], value))


class _FourierModel:
    def __init__(self, reference: np.ndarray):
        self.harmonics = _harmonics(reference)
        self.ridge = _select_ridge(reference, self.harmonics)
        self.design = _design(reference.shape[1], self.harmonics)
        coefficients, residual = _fit_rows(reference, self.design, self.ridge)
        self.coefficient_model = RobustDistanceModel(coefficients)
        coefficient_score = self.coefficient_model.score(coefficients)
        self.residual_center = float(np.median(residual))
        self.residual_scale = float(1.4826 * np.median(np.abs(residual - self.residual_center)) + 1e-8)
        self.coefficient_center = float(np.median(coefficient_score))
        self.coefficient_scale = float(1.4826 * np.median(np.abs(coefficient_score - self.coefficient_center)) + 1e-8)

    def score(self, rows: np.ndarray) -> np.ndarray:
        coefficients, residual = _fit_rows(rows, self.design, self.ridge)
        coefficient_score = self.coefficient_model.score(coefficients)
        residual_z = np.maximum(0.0, (residual - self.residual_center) / self.residual_scale)
        coefficient_z = np.maximum(0.0, (coefficient_score - self.coefficient_center) / self.coefficient_scale)
        return 0.5 * residual_z + 0.5 * coefficient_z


class SmoothFourierNormalResidual:
    name = "smooth_fourier_normal_residual"
    minimum_train = 10

    def score(self, bundle: DatasetBundle, seed: int, artifact_dir: Path) -> ScoreResult:
        if len(bundle.train_normal) < self.minimum_train:
            return abstained_result(self.name, bundle, self.minimum_train)
        train_oof = np.empty(len(bundle.train_normal), dtype=float)
        for fit, query in crossfit_plan(len(bundle.train_normal), seed=seed):
            train_oof[query] = _FourierModel(bundle.train_normal[fit]).score(bundle.train_normal[query])
        model = _FourierModel(bundle.train_normal)
        test_scores = model.score(bundle.test_values)
        return ScoreResult(
            self.name,
            train_oof,
            test_scores,
            False,
            None,
            {
                "seed": seed,
                "harmonics": model.harmonics.tolist(),
                "retained_harmonics": len(model.harmonics),
                "ridge": model.ridge,
                "ridge_grid": RIDGE_GRID,
                "uses_test_frequency_selection": False,
                "cross_fitted": True,
                "self_reference_violations": 0,
                "contains_test_labels": False,
            },
        )
