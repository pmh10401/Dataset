from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from experiments.exp392_label_guided_discovery.discovery_core import (
    CombinationRule,
    assign_research_label,
    benjamini_hochberg,
    bootstrap_intervals,
    candidate_indices_for_rule,
    concentration_diagnostics,
    enumerate_combination_rules,
    evaluate_candidate_set,
)
from experiments.exp392_label_guided_discovery.run_experiment_392_label_guided_score_discovery_audit import (
    build_observation_catalog,
    build_prepare_manifest,
    build_prepare_row,
    candidate_signature_sha256,
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


class EvaluationCatalogTests(unittest.TestCase):
    def test_catalog_contains_all_sources_pairs_and_triples(self):
        catalog = build_observation_catalog()

        self.assertEqual(len(catalog), 1998)
        ids = [row["observation_id"] for row in catalog]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            sum(row["observation_type"] == "source" for row in catalog),
            54,
        )
        self.assertEqual(
            sum(row["observation_type"] == "combination" for row in catalog),
            1944,
        )
        self.assertIn("source__ewma__alpha_0p005", ids)
        self.assertIn(
            "atleast2__ewma__frequency__topology__alpha_0p005",
            ids,
        )

    def test_candidate_signature_is_order_independent_and_dataset_aware(self):
        first = candidate_signature_sha256(
            {"B": {2, 1}, "A": {0}},
        )
        second = candidate_signature_sha256(
            {"A": {0}, "B": {1, 2}},
        )
        different = candidate_signature_sha256(
            {"A": {0, 1}, "B": {2}},
        )

        self.assertEqual(first, second)
        self.assertNotEqual(first, different)


class StabilityTests(unittest.TestCase):
    def test_bootstrap_is_reproducible(self):
        recovered = np.array([[1, 0, 2], [0, 1, 0]], dtype=np.int32)
        added_fp = np.array([[0, 1, 1], [1, 0, 1]], dtype=np.int32)

        first = bootstrap_intervals(
            recovered,
            added_fp,
            seed=20260727,
            iterations=200,
        )
        second = bootstrap_intervals(
            recovered,
            added_fp,
            seed=20260727,
            iterations=200,
        )

        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_concentration_flags_single_lineage_signal(self):
        recovered = np.array([[3, 2, 0], [1, 1, 1]], dtype=np.int32)

        result = concentration_diagnostics(
            recovered,
            lineages=("A", "A", "B"),
        )

        self.assertEqual(result["tp_contributing_lineages"].tolist(), [1, 2])
        self.assertEqual(result["top_lineage_tp_share"].tolist(), [1.0, 2 / 3])
        self.assertEqual(result["concentrated_signal"].tolist(), [True, False])

    def test_research_label_uses_only_preregistered_descriptive_thresholds(self):
        self.assertEqual(
            assign_research_label(
                recovered_fn=0,
                candidate_precision=1.0,
                tp_contributing_lineages=10,
                top_lineage_tp_share=0.1,
                precision_ci_low=0.9,
            ),
            "unsupported",
        )
        self.assertEqual(
            assign_research_label(
                recovered_fn=12,
                candidate_precision=0.6,
                tp_contributing_lineages=6,
                top_lineage_tp_share=0.5,
                precision_ci_low=0.4,
            ),
            "preregistration_candidate",
        )
        self.assertEqual(
            assign_research_label(
                recovered_fn=8,
                candidate_precision=0.6,
                tp_contributing_lineages=4,
                top_lineage_tp_share=0.5,
                precision_ci_low=0.4,
            ),
            "research_signal",
        )


class QueueRegistrationTests(unittest.TestCase):
    def test_exp392_has_explicit_queue_entry(self):
        from run_rank_experiments_sequential import get_experiment

        experiment = get_experiment(
            "experiment_392_label_guided_score_discovery_audit"
        )

        self.assertEqual(
            experiment["script"].name,
            "run_experiment_392_label_guided_score_discovery_audit.py",
        )
        self.assertEqual(
            experiment["detail_csv"].name,
            "exp392_individual_source_summary.csv",
        )
        self.assertEqual(
            experiment["summary_csv"].name,
            "exp392_pair_triple_summary.csv",
        )

    def test_runner_summarizes_audit_without_fake_f1(self):
        from run_rank_experiments_sequential import summarize_csv

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.csv"
            path.write_text(
                "observation_id,datasets_total,recovered_fn\n"
                "source_a,1117,10\n",
                encoding="utf-8",
            )

            summary = summarize_csv(path)

        self.assertEqual(summary["datasets"], 1117)
        self.assertEqual(summary["configs"], {})


if __name__ == "__main__":
    unittest.main()
