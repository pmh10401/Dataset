from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.exp266_fn_recovery_shadow.train_risk import (
    CALIBRATION_LIMITATION,
    TrainRiskContractError,
    audit_dataset_train_risk,
    audit_train_risk,
    loo_tail_pvalues,
    seed_intersection_flags,
    validate_query_excluded_provenance,
    wilson_upper_bound,
    write_train_risk_audit,
)


def policy_arrays(n_train: int = 200) -> dict[str, np.ndarray]:
    baseline = np.arange(n_train, dtype=float)
    reverse = baseline[::-1].copy()
    return {
        "kernel_1024": np.stack((baseline, baseline)),
        "kernel_4096": np.stack((baseline, baseline)),
        "statistical": baseline.copy(),
        "structural": baseline.copy(),
        "change_point": reverse.copy(),
        "reconstruction": np.stack((reverse, reverse)),
    }


class TrainRiskTests(unittest.TestCase):
    def test_loo_pvalue_excludes_query(self) -> None:
        scores = np.array([1.0, 2.0, 10.0])

        self.assertEqual(loo_tail_pvalues(scores).tolist(), [1.0, 2 / 3, 1 / 3])

    def test_loo_pvalue_uses_only_finite_reference_scores(self) -> None:
        scores = np.array([1.0, np.nan, 10.0, np.inf])

        self.assertEqual(loo_tail_pvalues(scores).tolist(), [1.0, 1.0, 0.5, 1.0])

    def test_n_train_below_five_abstains(self) -> None:
        arrays = {
            "kernel_1024": np.ones((2, 4)),
            "kernel_4096": np.ones((2, 4)),
            "statistical": np.ones(4),
            "structural": np.ones(4),
            "change_point": np.ones(4),
        }

        row = audit_dataset_train_risk("demo", n_train=4, source_arrays=arrays)

        self.assertTrue(row.abstained)
        self.assertEqual(row.normal_runs, 0)
        self.assertEqual(row.abstention_reason, "n_train_below_5")

    def test_seed_flags_require_every_seed(self) -> None:
        scores = np.array(
            [
                [0.0, 1.0, 2.0],
                [2.0, 1.0, 0.0],
            ]
        )

        flags = seed_intersection_flags(scores, alpha=1 / 3)

        self.assertEqual(flags.tolist(), [False, False, False])

    def test_combined_policy_counts_evidence_groups_once(self) -> None:
        row = audit_dataset_train_risk(
            "demo", n_train=200, source_arrays=policy_arrays()
        )

        self.assertFalse(row.abstained)
        self.assertEqual(row.source_count, 4)
        self.assertTrue(row.provenance_verified)
        self.assertTrue(row.policy_risk_validated)
        self.assertTrue(row.query_excluded_source_fitting)
        self.assertTrue(row.reconstruction_available_but_provenance_unverified)
        self.assertEqual(row.recovery_shadow_false_reviews, 1)
        self.assertEqual(row.recovery_priority_false_reviews, 0)
        self.assertEqual(row.combined_false_reviews, 1)
        self.assertEqual(row.normal_runs, 200)
        self.assertEqual(row.combined_false_review_rate, 0.005)
        self.assertFalse(row.combined_meets_risk_target)
        self.assertTrue(row.five_source_sensitivity_calculated)
        self.assertEqual(row.five_source_sensitivity_source_count, 5)
        self.assertFalse(row.five_source_sensitivity_provenance_verified)
        self.assertFalse(row.five_source_sensitivity_policy_risk_validated)
        self.assertEqual(row.five_source_sensitivity_combined_false_reviews, 2)
        self.assertIsNone(row.five_source_sensitivity_risk_target_pass)
        self.assertFalse(row.five_source_sensitivity_deployment_eligible)
        self.assertIn(
            "blocked until", row.five_source_sensitivity_approval_blocker
        )

    def test_mismatched_train_array_lengths_fail_explicitly(self) -> None:
        arrays = policy_arrays()
        arrays["structural"] = np.ones(199)

        with self.assertRaisesRegex(TrainRiskContractError, "TRAIN array length mismatch"):
            audit_dataset_train_risk("demo", n_train=200, source_arrays=arrays)

    def test_reconstruction_is_unavailable_below_eligibility_threshold(self) -> None:
        arrays = policy_arrays(n_train=99)
        arrays.pop("reconstruction")

        row = audit_dataset_train_risk("demo", n_train=99, source_arrays=arrays)

        self.assertEqual(row.source_count, 4)
        self.assertFalse(row.reconstruction_available_but_provenance_unverified)
        self.assertTrue(row.five_source_sensitivity_calculated)
        self.assertEqual(row.five_source_sensitivity_source_count, 5)
        self.assertEqual(
            row.five_source_sensitivity_combined_false_reviews,
            row.combined_false_reviews,
        )
        self.assertIsNotNone(row.five_source_sensitivity_combined_wilson_upper_95)
        self.assertFalse(row.abstained)

    def test_query_excluded_provenance_is_mandatory(self) -> None:
        valid = {
            "kernel": "five_fold_transformer_crossfit",
            "statistical": "five_fold_oof",
            "structural": "five_fold_oof",
            "change_point": "five_fold_oof",
        }
        validate_query_excluded_provenance(valid)

        with self.assertRaisesRegex(TrainRiskContractError, "query-excluded"):
            validate_query_excluded_provenance({**valid, "structural": "in_sample"})

    def test_unverified_reconstruction_cannot_enter_certified_policy(self) -> None:
        self_attested = {
            "kernel": "five_fold_transformer_crossfit",
            "statistical": "five_fold_oof",
            "structural": "five_fold_oof",
            "change_point": "five_fold_oof",
            "reconstruction": "five_fold_oof",
        }

        with self.assertRaisesRegex(
            TrainRiskContractError, "certified source provenance must contain exactly"
        ):
            validate_query_excluded_provenance(self_attested)

    def test_one_sided_wilson_upper_bound_controls_the_risk_gate(self) -> None:
        expected_zero_of_100 = 1.6448536269514722**2 / (
            100 + 1.6448536269514722**2
        )

        self.assertAlmostEqual(wilson_upper_bound(0, 100), expected_zero_of_100)
        self.assertGreater(wilson_upper_bound(2, 200), 0.01)

    def test_csv_records_frozen_contract_and_mixed_fold_limitation(self) -> None:
        row = audit_dataset_train_risk(
            "demo", n_train=200, source_arrays=policy_arrays()
        )
        with TemporaryDirectory() as directory:
            path = write_train_risk_audit([row], Path(directory))

            with path.open(newline="", encoding="utf-8") as handle:
                written = list(csv.DictReader(handle))

        self.assertEqual(path.name, "02_train_normal_risk_audit.csv")
        self.assertEqual(written[0]["alpha"], "0.005")
        self.assertEqual(written[0]["seed"], "20260721")
        self.assertEqual(written[0]["risk_target"], "0.01")
        self.assertEqual(written[0]["calibration_limitation"], CALIBRATION_LIMITATION)
        self.assertEqual(written[0]["source_count"], "4")
        self.assertEqual(
            written[0]["certified_sources"],
            "kernel|statistical|structural|change_point",
        )
        self.assertEqual(written[0]["provenance_verified"], "True")
        self.assertEqual(written[0]["policy_risk_validated"], "True")
        self.assertEqual(
            written[0]["reconstruction_available_but_provenance_unverified"],
            "True",
        )
        self.assertEqual(
            written[0]["five_source_sensitivity_provenance_verified"], "False"
        )
        self.assertEqual(
            written[0]["five_source_sensitivity_policy_risk_validated"], "False"
        )
        self.assertEqual(written[0]["five_source_sensitivity_risk_target_pass"], "")
        self.assertEqual(
            written[0]["five_source_sensitivity_deployment_eligible"], "False"
        )
        self.assertIn(
            "blocked until",
            written[0]["five_source_sensitivity_approval_blocker"],
        )
        self.assertEqual(written[0]["contains_test_values"], "False")
        self.assertEqual(written[0]["contains_test_labels"], "False")
        self.assertEqual(written[0]["contains_evaluation_metrics"], "False")

    def test_audits_complete_frozen_population_from_train_caches(self) -> None:
        with TemporaryDirectory() as directory:
            rows = audit_train_risk(output_root=Path(directory))

            self.assertEqual(len(rows), 1117)
            self.assertEqual(len({row.dataset for row in rows}), 1117)
            self.assertTrue(
                (Path(directory) / "02_train_normal_risk_audit.csv").is_file()
            )
            self.assertTrue(all(row.normal_runs in {0, row.n_train} for row in rows))
            self.assertTrue(all(row.source_count == 4 for row in rows))
            sensitivity = [
                row
                for row in rows
                if row.reconstruction_available_but_provenance_unverified
            ]
            self.assertEqual(len(sensitivity), 251)
            eligible = [row for row in rows if not row.abstained]
            self.assertTrue(
                all(row.five_source_sensitivity_calculated for row in eligible)
            )
            self.assertTrue(
                all(
                    row.five_source_sensitivity_calculated
                    and not row.five_source_sensitivity_provenance_verified
                    and not row.five_source_sensitivity_policy_risk_validated
                    and row.five_source_sensitivity_risk_target_pass is None
                    for row in sensitivity
                )
            )


if __name__ == "__main__":
    unittest.main()
