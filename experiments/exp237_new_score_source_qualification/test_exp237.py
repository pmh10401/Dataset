from __future__ import annotations

import inspect
import json
import tempfile
import unittest
import dataclasses
from pathlib import Path

import numpy as np

from experiments.exp237_new_score_source_qualification.core import (
    ALPHAS,
    SOURCE_NAMES,
    build_score_bundle,
    deterministic_fold_plan,
)
from experiments.exp237_new_score_source_qualification.prediction import (
    build_prediction_rows,
    freeze_predictions,
    load_verified_predictions,
)
from experiments.exp237_new_score_source_qualification.reporting import (
    evaluate_rows,
    summarize_rows,
)
from experiments.exp237_new_score_source_qualification.run_experiment_237 import (
    cache_file_stem,
    validate_prediction_coverage,
)


def _series(n_rows: int, length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 4.0 * np.pi, length)
    return np.vstack(
        [np.sin(x + 0.05 * i) + 0.03 * rng.normal(size=length) for i in range(n_rows)]
    ).astype(np.float32)


class FoldPlanTests(unittest.TestCase):
    def test_fold_plan_is_deterministic_disjoint_and_complete(self) -> None:
        first = deterministic_fold_plan(11, seed=20260719)
        second = deterministic_fold_plan(11, seed=20260719)
        self.assertEqual(len(first), 5)
        seen: list[int] = []
        for (fit_a, query_a), (fit_b, query_b) in zip(first, second):
            np.testing.assert_array_equal(fit_a, fit_b)
            np.testing.assert_array_equal(query_a, query_b)
            self.assertFalse(set(fit_a) & set(query_a))
            seen.extend(int(value) for value in query_a)
        self.assertEqual(sorted(seen), list(range(11)))


class ScoreSourceTests(unittest.TestCase):
    def test_all_five_sources_produce_finite_oof_and_test_scores(self) -> None:
        train = _series(12, 64, 1)
        test = _series(5, 64, 2)
        test[0, 30:34] += 4.0
        bundle = build_score_bundle("synthetic", train, test, seed=20260719)
        self.assertEqual(set(bundle.train_oof), set(SOURCE_NAMES))
        self.assertEqual(set(bundle.test_scores), set(SOURCE_NAMES))
        for source in SOURCE_NAMES:
            self.assertEqual(bundle.train_oof[source].shape, (12,))
            self.assertEqual(bundle.test_scores[source].shape, (5,))
            self.assertTrue(np.isfinite(bundle.train_oof[source]).all())
            self.assertTrue(np.isfinite(bundle.test_scores[source]).all())

    def test_score_bundle_is_deterministic(self) -> None:
        train = _series(10, 48, 3)
        test = _series(4, 48, 4)
        first = build_score_bundle("synthetic", train, test, seed=20260719)
        second = build_score_bundle("synthetic", train, test, seed=20260719)
        for source in SOURCE_NAMES:
            np.testing.assert_allclose(first.train_oof[source], second.train_oof[source])
            np.testing.assert_allclose(first.test_scores[source], second.test_scores[source])

    def test_less_than_five_train_rows_abstains(self) -> None:
        bundle = build_score_bundle(
            "small", _series(4, 32, 5), _series(3, 32, 6), seed=20260719
        )
        self.assertTrue(bundle.provenance["calibration_abstained"])
        self.assertEqual(bundle.provenance["abstention_reason"], "n_train_lt_5")
        for source in SOURCE_NAMES:
            self.assertEqual(len(bundle.train_oof[source]), 0)
            self.assertEqual(len(bundle.test_scores[source]), 0)


class PredictionContractTests(unittest.TestCase):
    def test_prediction_api_does_not_accept_labels(self) -> None:
        parameters = inspect.signature(build_prediction_rows).parameters
        self.assertNotIn("y_test", parameters)
        self.assertNotIn("labels", parameters)

    def test_prediction_is_label_free_and_hash_verified(self) -> None:
        bundle = build_score_bundle(
            "synthetic", _series(10, 48, 7), _series(4, 48, 8), seed=20260719
        )
        rows = build_prediction_rows(bundle)
        self.assertEqual(len(rows), len(SOURCE_NAMES) * len(ALPHAS))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_path = root / "prediction.jsonl"
            manifest_path = root / "manifest.json"
            freeze_predictions(rows, prediction_path, manifest_path)
            payloads = load_verified_predictions(prediction_path, manifest_path)
            forbidden = {"y_test", "labels", "tp", "fp", "f1", "precision", "recall"}
            for payload in payloads:
                self.assertFalse(forbidden & set(payload))
                self.assertFalse(payload["contains_test_labels"])
                self.assertFalse(payload["contains_evaluation_metrics"])
            prediction_path.write_text(
                prediction_path.read_text(encoding="utf-8") + json.dumps({"tamper": True}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_verified_predictions(prediction_path, manifest_path)

    def test_evaluation_is_separate_and_summarizes_source_alpha(self) -> None:
        bundle = build_score_bundle(
            "synthetic", _series(10, 48, 9), _series(4, 48, 10), seed=20260719
        )
        predictions = [dataclasses.asdict(row) for row in build_prediction_rows(bundle)]
        evaluated = evaluate_rows(predictions, lambda _: np.asarray([1, 0, 1, 0], dtype=np.int8))
        summaries = summarize_rows(evaluated)
        self.assertEqual(len(evaluated), len(SOURCE_NAMES) * len(ALPHAS))
        self.assertEqual(len(summaries), len(SOURCE_NAMES) * len(ALPHAS))
        self.assertTrue(all(row["metric_scope"] == "retrospective_research_only" for row in evaluated))
        self.assertTrue(all(row["policy_selected_from_test_metrics"] == 0 for row in summaries))

    def test_cache_stem_and_prediction_coverage_are_deterministic(self) -> None:
        self.assertEqual(cache_file_stem("a/b"), cache_file_stem("a/b"))
        self.assertNotEqual(cache_file_stem("a/b"), cache_file_stem("a_b"))
        rows = [
            {
                "dataset": "d1",
                "source": source,
                "alpha": alpha,
                "candidate_indices": [],
                "n_test": 3,
            }
            for source in SOURCE_NAMES
            for alpha in ALPHAS
        ]
        validate_prediction_coverage(rows, ("d1",))
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_prediction_coverage(rows[:-1], ("d1",))


if __name__ == "__main__":
    unittest.main()
