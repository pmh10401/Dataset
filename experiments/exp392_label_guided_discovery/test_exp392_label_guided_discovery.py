from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from experiments.exp392_label_guided_discovery.discovery_core import (
    CombinationRule,
    benjamini_hochberg,
    candidate_indices_for_rule,
    enumerate_combination_rules,
    evaluate_candidate_set,
)
from experiments.exp392_label_guided_discovery.run_experiment_392_label_guided_score_discovery_audit import (
    build_prepare_manifest,
    build_prepare_row,
    load_prepared_rows,
)


class CombinationEnumerationTests(unittest.TestCase):
    def test_enumerates_every_pair_and_triple_without_duplicates(self):
        rules = enumerate_combination_rules(("a", "b", "c"), alpha=0.005)
        names = [rule.rule_id for rule in rules]

        self.assertEqual(len(names), len(set(names)))
        self.assertIn("intersection__a__b__alpha_0p005", names)
        self.assertIn("union__a__b__alpha_0p005", names)
        self.assertIn("atleast2__a__b__c__alpha_0p005", names)
        self.assertEqual(len(rules), 9)

    def test_candidate_rule_never_reintroduces_excluded_indices(self):
        rule = CombinationRule("intersection", ("a", "b"), 0.005, 2)

        result = candidate_indices_for_rule(
            rule,
            {"a": {1, 2}, "b": {2, 3}},
            excluded={2},
            n_test=5,
        )

        self.assertEqual(result, set())

    def test_atleast_two_rule_counts_votes(self):
        rule = CombinationRule("atleast2", ("a", "b", "c"), 0.005, 2)

        result = candidate_indices_for_rule(
            rule,
            {"a": {1, 2}, "b": {2, 3}, "c": {2, 3, 4}},
            excluded=set(),
            n_test=5,
        )

        self.assertEqual(result, {2, 3})

    def test_rejects_out_of_range_source_index(self):
        rule = CombinationRule("union", ("a", "b"), 0.005, 1)

        with self.assertRaisesRegex(ValueError, "out-of-range"):
            candidate_indices_for_rule(
                rule,
                {"a": {1, 5}, "b": set()},
                excluded=set(),
                n_test=5,
            )


class CandidateEvaluationTests(unittest.TestCase):
    def test_reports_fn_recovery_fp_and_review_overlap(self):
        result = evaluate_candidate_set(
            candidate_indices={1, 2, 4},
            hard_indices={0},
            review_indices={1, 2},
            labels=("0", "1", "0", "1", "1"),
        )

        self.assertEqual(result["baseline_fn"], 3)
        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual(result["recovered_fn"], 2)
        self.assertEqual(result["added_fp"], 1)
        self.assertEqual(result["candidate_precision"], 2 / 3)
        self.assertEqual(result["review_overlap_count"], 2)
        self.assertEqual(result["review_recovered_fn"], 1)
        self.assertEqual(result["review_overlap_fp"], 1)
        self.assertEqual(result["novel_recovered_fn"], 1)


class MultipleComparisonTests(unittest.TestCase):
    def test_benjamini_hochberg_preserves_order_and_is_monotone(self):
        adjusted = benjamini_hochberg([0.04, 0.001, 0.03, 0.2])

        self.assertEqual(len(adjusted), 4)
        self.assertAlmostEqual(adjusted[0], 0.05333333333333334)
        self.assertAlmostEqual(adjusted[1], 0.004)
        self.assertAlmostEqual(adjusted[2], 0.05333333333333334)
        self.assertAlmostEqual(adjusted[3], 0.2)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted))


class PreparePhaseTests(unittest.TestCase):
    def setUp(self):
        self.hard_row = {
            "dataset": "Demo_normal_0",
            "n_test": 5,
            "automatic_hard_indices": [1],
            "standard_review_indices": [2],
            "priority_review_indices": [3],
            "automatic_normal_exact_indices": [0],
            "quality_conflict_indices": [4],
        }
        self.evidence_by_alpha = {
            0.005: {
                "Demo_normal_0": {
                    "evidence": {
                        "ewma": {
                            "available": True,
                            "indices": [2, 4],
                            "resolution_blocked": False,
                        },
                        "frequency": {
                            "available": False,
                            "indices": [],
                            "resolution_blocked": True,
                        },
                    }
                }
            }
        }
        self.coverage_row = {"n_train_normal": "20"}

    def test_prepare_row_contains_no_labels_or_metrics(self):
        row = build_prepare_row(
            self.hard_row,
            self.evidence_by_alpha,
            self.coverage_row,
            alphas=(0.005,),
            sources=("ewma", "frequency"),
        )

        forbidden = {
            "y_test",
            "label",
            "is_anomaly",
            "tp",
            "fp",
            "fn",
            "f1",
        }
        self.assertTrue(forbidden.isdisjoint(row))
        self.assertFalse(row["contains_test_labels"])
        self.assertFalse(row["contains_evaluation_metrics"])
        self.assertEqual(row["evidence_by_alpha"]["0p005"]["ewma"]["indices"], [2, 4])

    def test_prepare_manifest_disallows_policy_selection(self):
        manifest = build_prepare_manifest(
            rows=[{"dataset": "Demo_normal_0"}],
            prediction_sha256="abc",
            input_records=[],
            code_sha256="def",
        )

        self.assertFalse(manifest["test_metrics_select_policy"])
        self.assertFalse(manifest["eligible_for_deployment"])
        self.assertFalse(manifest["contains_test_labels"])
        self.assertFalse(manifest["contains_evaluation_metrics"])

    def test_prepared_rows_reject_tampered_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matrix = root / "exp392_frozen_candidate_matrix.jsonl"
            matrix.write_text('{"dataset":"Demo_normal_0"}\n', encoding="utf-8")
            manifest = build_prepare_manifest(
                rows=[{"dataset": "Demo_normal_0"}],
                prediction_sha256="not-the-real-hash",
                input_records=[],
                code_sha256="def",
            )
            (root / "exp392_prepare_manifest.json").write_text(
                __import__("json").dumps(manifest),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_prepared_rows(root)


if __name__ == "__main__":
    unittest.main()
