"""TDD coverage for the hash-gated Exp435 retrospective evaluator."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.exp435_train_only_consensus_recovery_antigravity.evaluation_antigravity import (
    BootstrapResult,
    ConfidenceInterval,
    ConfusionCounts,
    DatasetComparison,
    SuccessDecision,
    decide_exp435_success,
    paired_dataset_bootstrap,
)
from experiments.exp435_train_only_consensus_recovery_antigravity.reporting_antigravity import (
    PublicationExpectations,
    publish_exp435,
)
from experiments.exp435_train_only_consensus_recovery_antigravity.tests.test_reporting_antigravity import (
    _runtime,
)


class Exp435EvaluationTests(unittest.TestCase):
    def _rows(self) -> tuple[DatasetComparison, ...]:
        # D0 is rescued, D1 keeps a true positive and adds no FP, and D2 is a
        # strong-baseline regression.  These rows make the paired nature of
        # the bootstrap visible without involving TEST labels.
        return (
            DatasetComparison(
                dataset="D0",
                test_size=4,
                baseline=ConfusionCounts(tp=0, fp=0, fn=1, tn=3),
                combined=ConfusionCounts(tp=1, fp=0, fn=0, tn=3),
            ),
            DatasetComparison(
                dataset="D1",
                test_size=4,
                baseline=ConfusionCounts(tp=1, fp=0, fn=0, tn=3),
                combined=ConfusionCounts(tp=1, fp=0, fn=0, tn=3),
            ),
            DatasetComparison(
                dataset="D2",
                test_size=4,
                baseline=ConfusionCounts(tp=4, fp=0, fn=0, tn=0),
                combined=ConfusionCounts(tp=3, fp=1, fn=0, tn=0),
            ),
        )

    def test_hand_counts_rescued_and_regression_properties(self) -> None:
        rows = self._rows()
        self.assertEqual(rows[0].baseline.f1, 0.0)
        self.assertEqual(rows[0].combined.f1, 1.0)
        self.assertTrue(rows[0].rescued)
        self.assertTrue(rows[2].regressed)
        self.assertTrue(rows[2].strong_baseline)

    def test_bootstrap_is_paired_deterministic_and_frozen(self) -> None:
        first = paired_dataset_bootstrap(self._rows())
        second = paired_dataset_bootstrap(self._rows())
        self.assertIsInstance(first, BootstrapResult)
        self.assertEqual(first.repetitions, 20_000)
        self.assertEqual(first.seed, 43_520_260_806)
        self.assertEqual(first.to_payload(), second.to_payload())
        self.assertEqual(first.micro_f1_delta_ci, first.micro_f1_ci)
        self.assertEqual(first.macro_f1_delta_ci, first.macro_f1_ci)
        self.assertLessEqual(first.micro_f1_ci.lower, first.micro_f1_ci.upper)
        self.assertLessEqual(first.macro_f1_ci.lower, first.macro_f1_ci.upper)

    def test_decision_requires_every_gate(self) -> None:
        rows = self._rows()
        bootstrap = paired_dataset_bootstrap(rows, repetitions=16, seed=123)
        decision = decide_exp435_success(
            dataset_count=1117,
            candidate_count=3,
            rescued_datasets=tuple("R" + str(index) for index in range(10)),
            added_fp=0,
            micro_f1_delta=0.1,
            macro_f1_delta=0.1,
            micro_f1_ci=ConfidenceInterval(0.01, 0.2),
            macro_f1_ci=ConfidenceInterval(0.01, 0.2),
            prediction_label_access_count=0,
        )
        self.assertIsInstance(decision, SuccessDecision)
        self.assertTrue(decision.passed)
        failed = decide_exp435_success(
            dataset_count=1117,
            candidate_count=3,
            rescued_datasets=tuple("R" + str(index) for index in range(10)),
            added_fp=31,
            micro_f1_delta=0.1,
            macro_f1_delta=0.1,
            micro_f1_ci=ConfidenceInterval(0.01, 0.2),
            macro_f1_ci=ConfidenceInterval(0.01, 0.2),
            prediction_label_access_count=0,
        )
        self.assertFalse(failed.passed)
        self.assertIn("added_false_positive_gate", failed.reasons)

    def test_nonpositive_point_estimates_are_explicit_failures(self) -> None:
        common = dict(
            dataset_count=1117,
            candidate_count=0,
            rescued_datasets=tuple("D" + str(i) for i in range(10)),
            added_fp=0,
            micro_f1_delta=0.0,
            macro_f1_delta=-0.1,
            micro_f1_ci=ConfidenceInterval(0.01, 0.2),
            macro_f1_ci=ConfidenceInterval(0.01, 0.2),
            prediction_label_access_count=0,
        )
        decision = decide_exp435_success(**common)
        self.assertFalse(decision.passed)
        self.assertIn("micro_f1_point_gate", decision.reasons)
        self.assertIn("macro_f1_point_gate", decision.reasons)

    def test_prediction_hash_failure_precedes_test_label_loader(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            store = _runtime(root / "output", root / "checkpoint")
            publication = publish_exp435(
                store,
                root / "output",
                _expectations=PublicationExpectations(total=1),
            )
            prediction = publication.publication / "prediction_antigravity.jsonl"
            prediction.write_bytes(b'{"dataset":"tampered","instance_index":0}\n')
            calls: list[str] = []
            with self.assertRaisesRegex(ValueError, "prediction SHA256"):
                # Import locally to keep the test's high-level API obvious.
                from experiments.exp435_train_only_consensus_recovery_antigravity.evaluation_antigravity import evaluate_frozen_prediction

                evaluate_frozen_prediction(
                    publication=publication,
                    hard_coordinates={"Demo": ()},
                    label_loader=lambda _dataset: calls.append("opened"),
                    completed_store=store,
                    expected_datasets=1,
                )
            self.assertEqual(calls, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
