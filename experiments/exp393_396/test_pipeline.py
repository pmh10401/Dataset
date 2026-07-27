import json
import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.exp393_396.artifact_io import (
    load_verified_prediction,
    write_prediction_artifact,
)
from experiments.exp393_396.contracts import GrammarConfig
from experiments.exp393_396.reporting import audit_prediction_row
from experiments.exp393_396.run_experiment_393_396_pipeline import (
    evaluate_candidate_set,
    integrate_review,
    predict_dataset,
    run_audit,
    run_evaluate,
)


class PredictionPipelineTests(unittest.TestCase):
    def test_prediction_row_contains_no_labels_or_metrics(self):
        train = [
            np.sin(np.linspace(0.0, np.pi, 64) + offset)
            for offset in np.linspace(0.0, 0.4, 5)
        ]
        test = [np.sin(np.linspace(0.0, np.pi, 64) + 0.1)]

        row = predict_dataset(train, test, GrammarConfig(state_count=3))

        forbidden = {
            "y_test",
            "labels",
            "hard_tp",
            "hard_fp",
            "f1",
            "combined_f1",
        }
        self.assertTrue(forbidden.isdisjoint(row))
        self.assertFalse(row["contains_test_labels"])
        self.assertFalse(row["contains_evaluation_metrics"])

    def test_prediction_abstains_when_train_count_is_below_five(self):
        row = predict_dataset(
            [np.arange(16.0) + offset for offset in range(4)],
            [np.arange(16.0)],
            GrammarConfig(state_count=3),
        )

        self.assertTrue(row["abstained"])
        self.assertEqual(row["abstention_reason"], "n_train_lt_5")
        self.assertEqual(row["candidate_indices_by_alpha"], {})

    def test_prediction_artifact_rejects_hash_tampering(self):
        rows = [
            {
                "dataset": "A",
                "n_test": 1,
                "contains_test_labels": False,
                "contains_evaluation_metrics": False,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction, manifest = write_prediction_artifact(
                root,
                rows,
                GrammarConfig(),
            )
            prediction.write_text(
                prediction.read_text(encoding="utf-8") + "{}\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_verified_prediction(prediction, manifest)

    def test_prediction_artifact_sorts_datasets(self):
        rows = [
            {"dataset": "B", "n_test": 1},
            {"dataset": "A", "n_test": 1},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction, manifest = write_prediction_artifact(
                root,
                rows,
                GrammarConfig(),
            )
            loaded, _ = load_verified_prediction(prediction, manifest)
            parsed_manifest = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertEqual([row["dataset"] for row in loaded], ["A", "B"])
        self.assertEqual(parsed_manifest["rows"], 2)


class CalibrationAuditTests(unittest.TestCase):
    def test_audit_reports_train_count_abstention(self):
        row = audit_prediction_row(
            {
                "dataset": "tiny",
                "n_train_normal": 4,
                "n_test": 10,
                "abstained": True,
                "abstention_reason": "n_train_lt_5",
                "train_oof_scores": {},
            },
            GrammarConfig(),
        )

        self.assertTrue(row["abstained"])
        self.assertEqual(row["abstention_reason"], "n_train_lt_5")

    def test_audit_rejects_modified_prediction(self):
        rows = [{"dataset": "A", "n_train_normal": 4, "n_test": 1}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction, _ = write_prediction_artifact(root, rows, GrammarConfig())
            prediction.write_text(
                prediction.read_text(encoding="utf-8") + "{}\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "SHA256"):
                run_audit(root)


class LockedEvaluationTests(unittest.TestCase):
    def test_candidate_metrics_separate_hard_fn_review_overlap_and_new_fp(self):
        result, details = evaluate_candidate_set(
            candidates={1, 2, 3, 4},
            hard={0},
            standard={2},
            priority={4},
            labels=["1", "1", "1", "0", "0"],
            n_test=5,
        )

        self.assertEqual(result["candidate_count"], 4)
        self.assertEqual(result["recovered_hard_fn"], 2)
        self.assertEqual(result["added_fp"], 2)
        self.assertEqual(result["review_overlap_count"], 2)
        self.assertEqual(result["review_recovered_fn"], 1)
        self.assertEqual(len(details), 4)

    def test_evaluation_source_verifies_prediction_before_loading_labels(self):
        source = inspect.getsource(run_evaluate)

        self.assertLess(
            source.index("load_verified_prediction"),
            source.index("load_test_labels"),
        )


class ReviewIntegrationTests(unittest.TestCase):
    def test_new_candidates_enter_standard_review_only(self):
        result = integrate_review(
            baseline_hard={0},
            baseline_standard={1},
            baseline_priority={2},
            npgs_candidates={3},
        )

        self.assertEqual(result.hard, frozenset({0}))
        self.assertEqual(result.priority, frozenset({2}))
        self.assertEqual(result.standard, frozenset({1, 3}))

    def test_existing_lane_precedence_removes_all_overlap(self):
        result = integrate_review(
            baseline_hard={0},
            baseline_standard={1},
            baseline_priority={2},
            npgs_candidates={0, 1, 2, 3},
        )

        self.assertFalse(result.hard & result.standard)
        self.assertFalse(result.hard & result.priority)
        self.assertFalse(result.standard & result.priority)
        self.assertEqual(result.added_standard, frozenset({3}))


if __name__ == "__main__":
    unittest.main()
