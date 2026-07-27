from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from experiments.exp391_fn_reachability.fn_reachability_core import (
    ALPHAS,
    VOTE_THRESHOLDS,
    aggregate_recovery_rows,
    build_vote_candidates,
    classify_baseline,
    evaluate_recovery,
    load_verified_jsonl,
)
from experiments.exp391_fn_reachability.run_experiment_391_fn_reachability_audit import (
    render_report,
)


class BaselineClassificationTests(unittest.TestCase):
    def test_separates_no_hard_wrong_hard_and_detected_datasets(self):
        labels = ["0", "1", "0", "1"]

        self.assertEqual(classify_baseline([], labels), "zero_no_hard")
        self.assertEqual(classify_baseline([0, 2], labels), "zero_wrong_hard")
        self.assertEqual(classify_baseline([1], labels), "hard_has_tp")


class CandidateVoteTests(unittest.TestCase):
    def test_excludes_existing_hard_and_exact_normal_indices(self):
        evidence = {
            "frequency": {"available": True, "indices": [0, 1, 3]},
            "ewma": {"available": True, "indices": [1, 2, 3]},
            "temporal": {"available": False, "indices": []},
        }

        candidates = build_vote_candidates(
            evidence,
            vote_threshold=2,
            n_test=5,
            excluded_indices={1, 2},
        )

        self.assertEqual(candidates, {3})

    def test_rejects_out_of_range_evidence_indices(self):
        evidence = {"frequency": {"available": True, "indices": [4]}}

        with self.assertRaisesRegex(ValueError, "out-of-range"):
            build_vote_candidates(
                evidence,
                vote_threshold=1,
                n_test=4,
                excluded_indices=set(),
            )


class RecoveryEvaluationTests(unittest.TestCase):
    def test_reports_recovered_fn_added_fp_and_review_overlap(self):
        result = evaluate_recovery(
            candidate_indices={1, 2, 4},
            hard_indices={0},
            review_indices={1, 2, 3},
            labels=["0", "1", "0", "1", "1"],
        )

        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual(result["recovered_fn"], 2)
        self.assertEqual(result["added_fp"], 1)
        self.assertEqual(result["candidate_precision"], 2 / 3)
        self.assertEqual(result["review_overlap_count"], 2)
        self.assertEqual(result["review_recovered_fn"], 1)
        self.assertEqual(result["review_overlap_fp"], 1)
        self.assertEqual(result["review_overlap_precision"], 0.5)
        self.assertEqual(result["novel_recovered_fn"], 1)


class FrozenInputTests(unittest.TestCase):
    def test_uses_preregistered_alpha_and_vote_grids(self):
        self.assertEqual(ALPHAS, (0.005, 0.01, 0.02, 0.05, 0.1, 0.2))
        self.assertEqual(VOTE_THRESHOLDS, (1, 2, 3))

    def test_rejects_jsonl_when_manifest_hash_does_not_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "prediction.jsonl"
            manifest = root / "manifest.json"
            prediction.write_text('{"dataset":"A"}\n', encoding="utf-8")
            manifest.write_text(
                json.dumps(
                    {
                        "prediction_sha256": "0" * 64,
                        "rows": 1,
                        "contains_test_labels": False,
                        "contains_evaluation_metrics": False,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_verified_jsonl(prediction, manifest)


class RecoveryAggregationTests(unittest.TestCase):
    def test_aggregates_fixed_rule_without_selecting_it(self):
        rows = [
            {
                "dataset": "A",
                "alpha": 0.01,
                "vote_threshold": 2,
                "population": "zero_f1",
                "baseline_fn": 3,
                "candidate_count": 2,
                "recovered_fn": 1,
                "added_fp": 1,
                "review_overlap_count": 1,
                "review_recovered_fn": 1,
                "review_overlap_fp": 0,
                "novel_recovered_fn": 0,
            },
            {
                "dataset": "B",
                "alpha": 0.01,
                "vote_threshold": 2,
                "population": "zero_f1",
                "baseline_fn": 2,
                "candidate_count": 1,
                "recovered_fn": 1,
                "added_fp": 0,
                "review_overlap_count": 0,
                "review_recovered_fn": 0,
                "review_overlap_fp": 0,
                "novel_recovered_fn": 1,
            },
        ]

        summary = aggregate_recovery_rows(rows)

        self.assertEqual(summary["datasets"], 2)
        self.assertEqual(summary["baseline_fn"], 5)
        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["recovered_fn"], 2)
        self.assertEqual(summary["added_fp"], 1)
        self.assertEqual(summary["candidate_precision"], 2 / 3)
        self.assertEqual(summary["fn_recovery_rate"], 2 / 5)
        self.assertEqual(summary["review_overlap_precision"], 1.0)
        self.assertFalse(summary["selected_for_deployment"])
        self.assertFalse(summary["test_metrics_select_policy"])


class ReportTests(unittest.TestCase):
    def test_report_uses_fixed_diagnostic_rows_without_selecting_best(self):
        baseline = {
            "hard_count": 10,
            "hard_tp": 7,
            "hard_fp": 3,
            "hard_fn": 5,
            "zero_f1_datasets": 2,
            "zero_no_hard_datasets": 1,
            "zero_wrong_hard_datasets": 1,
        }
        dataset_rows = [
            {"n_test": 150, "hard_fn": 4, "baseline_category": "zero_no_hard"},
            {"n_test": 20, "hard_fn": 1, "baseline_category": "hard_has_tp"},
        ]
        rule_summary = [
            {
                "population": "zero_f1_test_ge_100",
                "alpha": 0.005,
                "vote_threshold": votes,
                "candidate_count": 4 - votes,
                "recovered_fn": 1,
                "added_fp": 3 - votes,
                "candidate_precision": 1 / (4 - votes),
            }
            for votes in (1, 2, 3)
        ]

        report = render_report(
            baseline,
            dataset_rows,
            rule_summary,
            source_summary=[],
        )

        self.assertIn("사전 고정 진단표", report)
        self.assertIn("큰 데이터셋 FN 비중", report)
        self.assertNotIn("가장 높은", report)
        self.assertNotIn("최적", report)


class QueueRegistrationTests(unittest.TestCase):
    def test_exp391_has_an_explicit_queue_entry(self):
        from run_rank_experiments_sequential import get_experiment

        experiment = get_experiment("experiment_391_fn_zero_reachability_audit")

        self.assertEqual(
            experiment["script"].name,
            "run_experiment_391_fn_reachability_audit.py",
        )
        self.assertEqual(experiment["summary_csv"].name, "exp391_rule_summary.csv")

    def test_runner_creates_all_output_directories_before_opening_logs(self):
        from run_rank_experiments_sequential import ensure_output_directories

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = {
                "detail_csv": root / "nested" / "detail.csv",
                "summary_csv": root / "nested" / "summary.csv",
                "stdout_log": root / "nested" / "logs" / "stdout.log",
            }

            ensure_output_directories(experiment)

            self.assertTrue(experiment["detail_csv"].parent.is_dir())
            self.assertTrue(experiment["summary_csv"].parent.is_dir())
            self.assertTrue(experiment["stdout_log"].parent.is_dir())


if __name__ == "__main__":
    unittest.main()
