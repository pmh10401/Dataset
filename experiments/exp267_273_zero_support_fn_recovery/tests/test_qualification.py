from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.exp267_273_zero_support_fn_recovery.config import (
    CORRELATION_COLLAPSE_ABS_SPEARMAN,
    CORRELATION_MIN_ALIGNED_TRAIN_OOF,
)
from experiments.exp267_273_zero_support_fn_recovery.qualification import (
    MembershipHashError,
    collapse_correlated_families,
    collapse_votes,
    evaluate_frozen_qualification,
    qualify_sources,
)
from experiments.exp267_273_zero_support_fn_recovery.train_risk import (
    audit_train_risk,
    leave_one_out_upper_pvalues,
    wilson_upper,
)


class TrainRiskTests(unittest.TestCase):
    def test_wilson_upper_bound_matches_known_case(self) -> None:
        self.assertAlmostEqual(wilson_upper(0, 100, confidence=0.95), 0.0263427, places=6)

    def test_query_exclusion_and_tie_formula(self) -> None:
        pvalues = leave_one_out_upper_pvalues(np.asarray([2.0, 2.0, 1.0]))
        np.testing.assert_allclose(pvalues, [2.0 / 3.0, 2.0 / 3.0, 1.0])
        rows = audit_train_risk([
            {"dataset": "d", "source": "scattering", "variant": "primary", "n_train": 3,
             "train_oof": np.asarray([2.0, 2.0, 1.0]), "eligible": True, "abstained": False},
        ], alphas=(2.0 / 3.0,))
        self.assertEqual(rows[0]["normal_candidate_count"], 2)
        self.assertEqual(rows[0]["calibration_reference"], "query_excluded_train_oof")

    def test_train_risk_rejects_test_or_evaluation_inputs(self) -> None:
        with self.assertRaises(ValueError):
            audit_train_risk([{
                "dataset": "d", "source": "scattering", "variant": "primary", "n_train": 3,
                "train_oof": np.asarray([1.0, 2.0, 3.0]), "test_scores": np.asarray([9.0]),
            }])


class QualificationTests(unittest.TestCase):
    def test_frozen_correlation_contract(self) -> None:
        self.assertEqual(CORRELATION_COLLAPSE_ABS_SPEARMAN, 0.90)
        self.assertEqual(CORRELATION_MIN_ALIGNED_TRAIN_OOF, 20)

    def test_variants_of_one_family_cannot_double_vote(self) -> None:
        votes = collapse_votes({"tcn_seed_median": {3}, "tcn_seed_intersection": {3}})
        self.assertEqual(votes[3], {"normal_forecast"})

    def test_high_correlation_cross_family_collapses_connected_component(self) -> None:
        scores = {
            "scattering": np.arange(20, dtype=float),
            "signature": np.arange(20, dtype=float) * 2.0,
        }
        collapsed = collapse_correlated_families(scores)
        self.assertEqual(len(collapsed["groups"]), 1)
        self.assertTrue(collapsed["groups"][0]["independence_verified"])

    def test_low_overlap_marks_independence_unverified(self) -> None:
        collapsed = collapse_correlated_families({
            "scattering": np.arange(19, dtype=float),
            "signature": np.arange(19, dtype=float),
        })
        self.assertEqual(len(collapsed["groups"]), 2)
        self.assertTrue(all(not item["independence_verified"] for item in collapsed["groups"]))

    def test_dataset_keyed_correlation_uses_only_shared_train_datasets(self) -> None:
        collapsed = collapse_correlated_families({
            "scattering": {"shared": np.arange(20, dtype=float), "left_only": np.arange(100, dtype=float)},
            "signature": {"shared": np.arange(20, dtype=float), "right_only": np.arange(100, dtype=float)},
        })
        pair = collapsed["pairs"][0]
        self.assertEqual(pair["aligned_count"], 20)

    def test_membership_is_label_invariant_and_blocks_same_family_double_vote(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            prediction, manifest = _write_prediction_fixture(root)
            risk = _passing_risk_rows()
            oof = {
                "tcn_seed_median": np.arange(24, dtype=float),
                "tcn_seed_intersection": np.arange(24, dtype=float),
            }
            first = qualify_sources(
                prediction, manifest, output_dir=root, risk_rows=risk, train_oof_by_variant=oof,
                expected_datasets=("d",), verify_predictions=False,
            )
            with patch(
                "experiments.exp267_273_zero_support_fn_recovery.qualification._load_test_truth",
                return_value={"d": {0, 1}},
            ):
                second = qualify_sources(
                    prediction, manifest, output_dir=root / "second", risk_rows=risk,
                    train_oof_by_variant=oof, expected_datasets=("d",), verify_predictions=False,
                )
            self.assertEqual(first["membership_sha256"], second["membership_sha256"])
            self.assertIn("config_hashes", first)
            self.assertIn("input_hashes", first)
            self.assertIn("run_contract_sha256", first)
            self.assertEqual(len(first["qualified_groups"]), 1)
            self.assertEqual(first["qualified_groups"][0]["members"], ["tcn_seed_intersection", "tcn_seed_median"])

    def test_membership_rejects_incomplete_coverage_and_hash_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            prediction, manifest = _write_prediction_fixture(root)
            with self.assertRaises(ValueError):
                qualify_sources(
                    prediction, manifest, output_dir=root, risk_rows=_passing_risk_rows(),
                    train_oof_by_variant={"tcn_seed_median": np.arange(24.0)},
                    expected_datasets=("d", "missing"), verify_predictions=False,
                )
            membership = qualify_sources(
                prediction, manifest, output_dir=root, risk_rows=_passing_risk_rows(),
                train_oof_by_variant={
                    "tcn_seed_median": np.arange(24.0),
                    "tcn_seed_intersection": np.arange(24.0),
                },
                expected_datasets=("d",), verify_predictions=False,
            )
            membership_path = root / "exp272_source_membership.json"
            membership_path.write_text(json.dumps({**membership, "qualified_groups": []}), encoding="utf-8")
            with self.assertRaises(MembershipHashError):
                evaluate_frozen_qualification(
                    prediction, manifest, membership_path, expected_datasets=("d",),
                    existing_lanes={"d": {"hard": set(), "review": set()}},
                    test_truth={"d": {0}}, verify_predictions=False,
                )

    def test_post_freeze_metrics_keep_autonomous_and_review_separate(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            prediction, manifest = _write_prediction_fixture(root)
            qualify_sources(
                prediction, manifest, output_dir=root, risk_rows=_passing_risk_rows(),
                train_oof_by_variant={
                    "tcn_seed_median": np.arange(24.0),
                    "tcn_seed_intersection": np.arange(24.0),
                },
                expected_datasets=("d",), verify_predictions=False,
            )
            result = evaluate_frozen_qualification(
                prediction, manifest, root / "exp272_source_membership.json",
                expected_datasets=("d",), existing_lanes={"d": {"hard": {1}, "review": {2}}},
                test_truth={"d": {0, 1, 2}}, verify_predictions=False,
            )
            self.assertEqual(result["autonomous"]["hard_tp"], 1)
            self.assertEqual(result["human_assisted_review"]["review_tp"], 1)
            self.assertNotIn("combined_f1", result["autonomous"])
            self.assertNotIn("hard_tp", result["human_assisted_review"])


def _write_prediction_fixture(root: Path) -> tuple[Path, Path]:
    median = {
        "dataset": "d", "source": "tcn_forecast", "variant": "seed_median", "alpha": 0.005,
        "candidate_indices": [0], "evidence_family": "normal_forecast",
        "evidence_identity": "tcn_forecast", "contains_test_labels": False,
    }
    intersection = {**median, "variant": "seed_intersection"}
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in (median, intersection)
    )
    prediction = root / "prediction.jsonl"
    prediction.write_bytes(payload)
    manifest = root / "prediction_manifest.json"
    manifest.write_text(json.dumps({"prediction_sha256": hashlib.sha256(payload).hexdigest()}), encoding="utf-8")
    return prediction, manifest


def _passing_risk_rows() -> list[dict[str, object]]:
    return [{
        "dataset": "d", "source": "tcn_forecast", "variant": "seed_median", "alpha": 0.005,
        "eligible": True, "abstained": False, "coverage_complete": True,
        "normal_candidate_count": 0, "normal_candidate_rate": 0.0,
        "wilson_upper_95": 0.001,
    }, {
        "dataset": "d", "source": "tcn_forecast", "variant": "seed_intersection", "alpha": 0.005,
        "eligible": True, "abstained": False, "coverage_complete": True,
        "normal_candidate_count": 0, "normal_candidate_rate": 0.0,
        "wilson_upper_95": 0.001,
    }]


if __name__ == "__main__":
    unittest.main()
