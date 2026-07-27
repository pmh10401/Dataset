from __future__ import annotations

import hashlib
import unittest

import numpy as np
import torch

from experiments.exp403_408.contracts import (
    ALPHAS,
    FrozenConfig,
    LearnedSourceScores,
    SEED,
)
from experiments.exp403_408.tri_view_ssl import (
    _require_mps_parity,
    build_tri_views,
    score_tri_view,
    shadow_predictions,
)


def train_rows(count: int, length: int = 64) -> np.ndarray:
    grid = np.linspace(0.0, 2.0 * np.pi, length, endpoint=False)
    rows = np.asarray(
        [np.sin(grid * (1.0 + (index % 3))) + 0.03 * index for index in range(count)],
        dtype=float,
    )
    return (rows - rows.mean(axis=1, keepdims=True)) / rows.std(axis=1, keepdims=True)


def test_rows(length: int = 64) -> np.ndarray:
    grid = np.linspace(0.0, 2.0 * np.pi, length, endpoint=False)
    rows = np.asarray(
        [np.sin(grid * 4.0), np.cos(grid * 2.0)], dtype=float
    )
    return (rows - rows.mean(axis=1, keepdims=True)) / rows.std(axis=1, keepdims=True)


def config(seed: int = SEED) -> FrozenConfig:
    return FrozenConfig(seed=seed, target_length=64)


def run_hash(seed: int = SEED) -> str:
    result = score_tri_view(train_rows(50), test_rows(), config(seed), device="cpu")
    payload = repr(sorted((key, tuple(values)) for key, values in result.scores_by_alpha.items()))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def learned_scores(*, raw_test_scores: tuple[float, ...]) -> LearnedSourceScores:
    return LearnedSourceScores(
        scores_by_alpha={str(alpha): (0.5, 0.5) for alpha in ALPHAS},
        raw_test_scores=raw_test_scores,
        available=True,
    )


class TriViewSSLTests(unittest.TestCase):
    def test_three_views_have_fixed_width_and_are_finite(self) -> None:
        views = build_tri_views(train_rows(8))

        self.assertEqual(set(views), {"time", "frequency", "residual"})
        self.assertEqual({value.shape for value in views.values()}, {(8, 64)})
        self.assertTrue(all(np.isfinite(value).all() for value in views.values()))

    def test_n_train_49_abstains_from_learned_model(self) -> None:
        result = score_tri_view(train_rows(49), test_rows(), config(), device="cpu")

        self.assertFalse(result.available)
        self.assertEqual(result.abstention_reason, "n_train_below_50")
        self.assertEqual(result.scores_by_alpha, {})

    def test_same_seed_produces_same_cpu_prediction_hash(self) -> None:
        self.assertEqual(run_hash(seed=SEED), run_hash(seed=SEED))

    def test_an_unrelated_test_query_cannot_change_an_existing_score(self) -> None:
        train = train_rows(50)
        baseline = score_tri_view(train, test_rows(), config(), device="cpu")
        extended_test = np.vstack((test_rows(), train_rows(1)))
        extended = score_tri_view(train, extended_test, config(), device="cpu")

        for alpha in ALPHAS:
            key = str(alpha)
            self.assertEqual(
                baseline.scores_by_alpha[key], extended.scores_by_alpha[key][:2]
            )

    def test_shadow_predictions_exclude_frozen_hard_and_carry_provenance(self) -> None:
        result = score_tri_view(train_rows(50), test_rows(), config(), device="cpu")
        frozen_hard = frozenset({0})

        candidates = shadow_predictions("toy", result, frozen_hard)

        self.assertEqual(frozen_hard, frozenset({0}))
        self.assertEqual(set(candidates), {str(alpha) for alpha in ALPHAS})
        self.assertTrue(any(candidates.values()))
        self.assertTrue(all(
            prediction.candidate_index != 0
            and prediction.dataset == "toy"
            and prediction.source == "self_supervised_shadow"
            for values in candidates.values()
            for prediction in values
        ))

    def test_mps_rejects_quantized_p_value_match_when_raw_ranks_differ(self) -> None:
        cpu = learned_scores(raw_test_scores=(0.9, 0.8))
        mps = learned_scores(raw_test_scores=(0.8, 0.9))

        with self.assertRaisesRegex(ValueError, "raw candidate rank"):
            _require_mps_parity(cpu, mps)

    def test_mps_rejects_quantized_p_value_match_when_raw_delta_exceeds_tolerance(self) -> None:
        cpu = learned_scores(raw_test_scores=(0.9, 0.8))
        mps = learned_scores(raw_test_scores=(0.90002, 0.8))

        with self.assertRaisesRegex(ValueError, "raw score tolerance"):
            _require_mps_parity(cpu, mps)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_requires_cpu_score_and_rank_parity(self) -> None:
        cpu = score_tri_view(train_rows(50), test_rows(), config(), device="cpu")
        mps = score_tri_view(train_rows(50), test_rows(), config(), device="mps")

        self.assertEqual(len(cpu.raw_test_scores), len(test_rows()))
        self.assertTrue(np.isfinite(cpu.raw_test_scores).all())
        self.assertTrue(np.array_equal(
            np.argsort(-np.asarray(cpu.raw_test_scores), kind="stable"),
            np.argsort(-np.asarray(mps.raw_test_scores), kind="stable"),
        ))
        self.assertLessEqual(
            np.max(np.abs(np.asarray(cpu.raw_test_scores) - np.asarray(mps.raw_test_scores))),
            1e-5,
        )


if __name__ == "__main__":
    unittest.main()
