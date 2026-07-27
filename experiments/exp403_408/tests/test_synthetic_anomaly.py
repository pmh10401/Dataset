from __future__ import annotations

import hashlib
import unittest

import numpy as np
import torch

from experiments.exp403_408.contracts import ALPHAS, FrozenConfig, LearnedSourceScores, SEED
from experiments.exp403_408.synthetic_anomaly import (
    SYNTHETIC_PERTURBATION_TYPES,
    SyntheticAnomalyConfig,
    generate_synthetic_anomalies,
    score_synthetic_models,
    synthetic_shadow_predictions,
)


def train8(length: int = 64) -> np.ndarray:
    grid = np.linspace(0.0, 2.0 * np.pi, length, endpoint=False)
    rows = np.asarray(
        [
            np.sin(grid * (1.0 + (index % 3)))
            + 0.15 * np.cos(grid * (2.0 + (index % 2)))
            + 0.01 * index
            for index in range(8)
        ],
        dtype=float,
    )
    return (rows - rows.mean(axis=1, keepdims=True)) / rows.std(axis=1, keepdims=True)


def train50(length: int = 64) -> np.ndarray:
    return np.vstack([train8(length) for _ in range(7)])[:50]


def test2(length: int = 64) -> np.ndarray:
    grid = np.linspace(0.0, 2.0 * np.pi, length, endpoint=False)
    rows = np.asarray(
        [
            np.sin(grid * 5.0) + 2.5 * (np.arange(length) == 20),
            np.cos(grid * 3.0) + np.linspace(0.0, 2.0, length),
        ],
        dtype=float,
    )
    return (rows - rows.mean(axis=1, keepdims=True)) / rows.std(axis=1, keepdims=True)


def config(seed: int = SEED) -> SyntheticAnomalyConfig:
    return SyntheticAnomalyConfig(
        frozen=FrozenConfig(seed=seed, target_length=64),
        examples_per_type_per_run=1,
        mlp_epochs=12,
    )


def score_hash(seed: int = SEED) -> str:
    scores = score_synthetic_models(train50(), test2(), config(seed))
    payload = repr(
        [
            (score.model_name, tuple(score.raw_test_scores), sorted(score.scores_by_alpha.items()))
            for score in scores
        ]
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


class SyntheticAnomalyTests(unittest.TestCase):
    def test_every_frozen_perturbation_is_generated(self) -> None:
        generated = generate_synthetic_anomalies(train8(), config(), SEED)

        self.assertEqual(set(generated.by_type), set(SYNTHETIC_PERTURBATION_TYPES))
        self.assertTrue(all(len(values) == len(train8()) for values in generated.by_type.values()))
        self.assertEqual(generated.values.shape, (len(train8()) * 7, 64))
        self.assertTrue(np.isfinite(generated.values).all())

    def test_perturbation_scale_depends_only_on_train(self) -> None:
        first = generate_synthetic_anomalies(train8(), config(), SEED)
        second = generate_synthetic_anomalies(train8(), config(), SEED)

        self.assertEqual(first.sha256, second.sha256)
        self.assertTrue(np.array_equal(first.values, second.values))

    def test_n_train_49_abstains_from_all_synthetic_models(self) -> None:
        result = score_synthetic_models(train50()[:49], test2(), config())

        self.assertEqual(len(result), 3)
        self.assertTrue(all(not score.available for score in result))
        self.assertTrue(all(score.abstention_reason == "n_train_below_50" for score in result))

    def test_reports_all_three_frozen_classifiers_separately(self) -> None:
        scores = score_synthetic_models(train50(), test2(), config())

        self.assertEqual(
            tuple(score.model_name for score in scores),
            ("logistic_regression", "hist_gradient_boosting", "small_mlp"),
        )
        self.assertTrue(all(score.available for score in scores))
        self.assertTrue(all(set(score.scores_by_alpha) == {str(alpha) for alpha in ALPHAS} for score in scores))
        self.assertTrue(all(len(score.raw_test_scores) == len(test2()) for score in scores))

    def test_same_seed_produces_deterministic_model_hash(self) -> None:
        self.assertEqual(score_hash(SEED), score_hash(SEED))

    def test_mlp_fit_restores_process_determinism_setting(self) -> None:
        before = torch.are_deterministic_algorithms_enabled()

        score_synthetic_models(train50(), test2(), config())

        self.assertEqual(torch.are_deterministic_algorithms_enabled(), before)

    def test_an_unrelated_test_query_cannot_change_existing_model_scores(self) -> None:
        baseline = score_synthetic_models(train50(), test2(), config())
        extended = score_synthetic_models(
            train50(), np.vstack((test2(), train8()[:1])), config()
        )

        for baseline_model, extended_model in zip(baseline, extended, strict=True):
            self.assertEqual(baseline_model.model_name, extended_model.model_name)
            self.assertEqual(
                baseline_model.raw_test_scores,
                extended_model.raw_test_scores[: len(test2())],
            )
            for alpha in ALPHAS:
                key = str(alpha)
                self.assertEqual(
                    baseline_model.scores_by_alpha[key],
                    extended_model.scores_by_alpha[key][: len(test2())],
                )

    def test_shadow_predictions_exclude_hard_and_mark_synthetic_provenance(self) -> None:
        scores = LearnedSourceScores(
            scores_by_alpha={str(alpha): (0.001, 0.001) for alpha in ALPHAS},
            raw_test_scores=(0.9, 0.1),
            available=True,
            model_name="logistic_regression",
        )
        frozen_hard = frozenset({0})

        candidates = synthetic_shadow_predictions("toy", scores, frozen_hard)

        self.assertEqual(frozen_hard, frozenset({0}))
        self.assertEqual(set(candidates), {str(alpha) for alpha in ALPHAS})
        self.assertTrue(any(candidates.values()))
        self.assertTrue(all(
            prediction.candidate_index != 0
            and prediction.dataset == "toy"
            and prediction.source == "synthetic_anomaly_shadow"
            for values in candidates.values()
            for prediction in values
        ))


if __name__ == "__main__":
    unittest.main()
