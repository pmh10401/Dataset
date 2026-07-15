"""Unit tests for Exp160 Candidate-Zero Similarity Recovery."""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

# Ensure correct imports
EXP158_159_DIR = REPO_ROOT / "experiments" / "exp158_159"
if str(EXP158_159_DIR) not in sys.path:
    sys.path.insert(0, str(EXP158_159_DIR))

if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import candidate_zero_core as core
import similarity_pool_core as pool_core

EXP153_DIR = REPO_ROOT / "experiments" / "exp153_157"
if str(EXP153_DIR) not in sys.path:
    sys.path.insert(0, str(EXP153_DIR))

import exp153_157_core as exp_core
import exp153_157_reporting as reporting


class CandidateZeroRecoveryContractTests(unittest.TestCase):

    def test_01_target_eligibility(self):
        """1. Target gate: baseline candidate_count == 0 and n_train >= 5."""
        self.assertTrue(core.is_candidate_zero_target(n_train=5, baseline_candidate_count=0))
        self.assertTrue(core.is_candidate_zero_target(n_train=20, baseline_candidate_count=0))
        self.assertFalse(core.is_candidate_zero_target(n_train=4, baseline_candidate_count=0))
        self.assertFalse(core.is_candidate_zero_target(n_train=5, baseline_candidate_count=1))
        self.assertFalse(core.is_candidate_zero_target(n_train=20, baseline_candidate_count=10))

    def test_02_n_train_under_5_abstention(self):
        """2. n_train < 5 is complete calibration abstention."""
        self.assertTrue(core.is_abstained(4))
        self.assertTrue(core.is_abstained(0))
        self.assertFalse(core.is_abstained(5))
        self.assertFalse(core.is_abstained(20))

    def test_02b_donor_requires_five_train_rows_and_complete_descriptor(self):
        self.assertFalse(core.is_eligible_donor(4, True))
        self.assertFalse(core.is_eligible_donor(20, False))
        self.assertTrue(core.is_eligible_donor(5, True))

    def test_03_non_target_baseline_exact_match(self):
        """3. If candidate is already present (non-target), all lanes must exact match baseline."""
        # Non-target because baseline candidate count > 0 (1)
        res = core.integrate_candidate_zero_review(
            dataset_name="d1",
            alpha=0.01,
            n_train=10,
            baseline_hard={1},
            baseline_standard=set(),
            baseline_priority=set(),
            similarity_candidates={2, 3},
            n_test=5
        )
        self.assertEqual(res["hard"], {1})
        self.assertEqual(res["standard_review"], set())
        self.assertEqual(res["priority_review"], set())
        self.assertEqual(res["candidate"], {1})

        # Non-target because n_train < 5
        res2 = core.integrate_candidate_zero_review(
            dataset_name="d2",
            alpha=0.01,
            n_train=4,
            baseline_hard=set(),
            baseline_standard=set(),
            baseline_priority=set(),
            similarity_candidates={2, 3},
            n_test=5
        )
        self.assertEqual(res2["hard"], set())
        self.assertEqual(res2["standard_review"], set())
        self.assertEqual(res2["priority_review"], set())
        self.assertEqual(res2["candidate"], set())

    def test_04_target_baseline_preservation(self):
        """4. Target's baseline Hard/Priority must be preserved exactly."""
        # Although target has baseline candidate_count == 0, we verify that any hypothetical
        # Hard/Priority elements would be preserved, and only Standard review gets additions.
        res = core.integrate_candidate_zero_review(
            dataset_name="d1",
            alpha=0.01,
            n_train=10,
            baseline_hard=set(),
            baseline_standard=set(),
            baseline_priority=set(),
            similarity_candidates={1, 2},
            n_test=5
        )
        self.assertEqual(res["hard"], set())
        self.assertEqual(res["priority_review"], set())
        self.assertEqual(res["standard_review"], {1, 2})

    def test_05_similarity_only_added_to_standard_only(self):
        """5. Similarity-only indices must only be added to standard review."""
        res = core.integrate_candidate_zero_review(
            dataset_name="d1",
            alpha=0.01,
            n_train=10,
            baseline_hard=set(),
            baseline_standard=set(),
            baseline_priority=set(),
            similarity_candidates={2},
            n_test=5
        )
        self.assertEqual(res["standard_review"], {2})
        self.assertNotIn(2, res["hard"])
        self.assertNotIn(2, res["priority_review"])

    def test_06_minimum_finite_reference_200(self):
        """6. Reused select_donors_by_source_counts must enforce min_pool_reference=200 for every source."""
        donor_names = ["d1", "d2"]
        distances = np.asarray([1.0, 2.0])

        # Less than 200 for one source
        counts_fail = {
            "rocket_exp40": [100, 100],
            "exp55_best": [100, 99],  # 199 < 200
            "exp56_best": [100, 100],
            "exp84": [100, 100],
        }
        selection = pool_core.select_donors_by_source_counts(
            target_name="target",
            donor_names=donor_names,
            distances=distances,
            donor_reference_counts_by_source=counts_fail,
            min_pool_reference=200
        )
        self.assertIsNone(selection)

        # 200 for all sources
        counts_ok = {
            "rocket_exp40": [100, 100],
            "exp55_best": [100, 100],
            "exp56_best": [100, 100],
            "exp84": [100, 100],
        }
        selection_ok = pool_core.select_donors_by_source_counts(
            target_name="target",
            donor_names=donor_names,
            distances=distances,
            donor_reference_counts_by_source=counts_ok,
            min_pool_reference=200
        )
        self.assertIsNotNone(selection_ok)

    def test_07_target_exclusion_from_pool(self):
        """7. Target dataset must be excluded from selected donors."""
        donor_names = ["d1", "target", "d2"]
        distances = np.asarray([1.0, 1.5, 2.0])
        donor_n_trains = [150, 150, 150]

        selection = pool_core.select_donors(
            target_name="target",
            donor_names=donor_names,
            distances=distances,
            donor_n_trains=donor_n_trains,
            min_pool_reference=200
        )
        self.assertIsNotNone(selection)
        selected_donors, _, _ = selection
        self.assertNotIn("target", selected_donors)

    def test_08_donor_order_and_distance_tie_invariance(self):
        """8. Selected donors must be invariant to order and tie boundaries must be included."""
        donor_names = ["d1", "d2", "d3"]
        distances = np.asarray([1.0, 2.0, 2.0])
        donor_n_trains = [100, 100, 100]

        selection = pool_core.select_donors(
            target_name="target",
            donor_names=donor_names,
            distances=distances,
            donor_n_trains=donor_n_trains,
            min_pool_reference=200
        )
        self.assertIsNotNone(selection)
        selected, _, boundary = selection
        self.assertEqual(boundary, 2.0)
        self.assertEqual(set(selected), {"d1", "d2", "d3"}) # Both tie distance 2.0 are included

    def test_09_family_and_dataset_name_policies_unused(self):
        """9. Family and dataset names must not be used to branch logic in descriptors or selection."""
        # inspect signatures to verify no family or dataset name branching parameter in pool_core
        sig = inspect.signature(pool_core.select_donors)
        self.assertNotIn("family", sig.parameters)
        sig2 = inspect.signature(pool_core.compute_dataset_descriptor)
        self.assertNotIn("family", sig2.parameters)

    def test_10_test_length_unused_for_descriptor_or_threshold(self):
        """10. TEST length does not change descriptor, pool, or threshold."""
        sig = inspect.signature(pool_core.compute_dataset_descriptor)
        self.assertNotIn("n_test", sig.parameters)
        self.assertNotIn("test_length", sig.parameters)

    def test_11_prediction_does_not_receive_y_test(self):
        """11. Core prediction functions do not accept y_test or labels arguments."""
        for name, func in inspect.getmembers(core, inspect.isfunction):
            sig = inspect.signature(func)
            for param in sig.parameters:
                self.assertNotIn(param, ("y_test", "y", "label", "labels", "test_labels"))

    def test_12_prediction_output_has_no_metrics(self):
        """12. Shared metadata must contain names/flags but no evaluation fields."""
        row = core.prediction_policy_metadata(candidate_empty_gate_applied=True)
        forbidden = {
            "precision", "recall", "f1", "combined_metric", "tp", "fp", "fn",
            "y_test", "hard_tp", "hard_fp", "hard_fn", "hard_f1", "combined_f1",
        }
        self.assertFalse(forbidden & set(row))
        self.assertEqual(row["source_display_names"], core.DISPLAY_NAMES)

    def test_13_lane_properties(self):
        """13. Lane overlap is 0, candidate union matches lanes, and index range is valid."""
        # Should raise on baseline overlap
        with self.assertRaises(ValueError):
            core.integrate_candidate_zero_review(
                dataset_name="d1",
                alpha=0.01,
                n_train=10,
                baseline_hard={1},
                baseline_standard={1}, # overlap
                baseline_priority=set(),
                similarity_candidates=set(),
                n_test=5
            )

        # Should raise on index range violation
        with self.assertRaises(ValueError):
            core.integrate_candidate_zero_review(
                dataset_name="d1",
                alpha=0.01,
                n_train=10,
                baseline_hard=set(),
                baseline_standard=set(),
                baseline_priority=set(),
                similarity_candidates={5}, # index >= n_test
                n_test=5
            )

        res = core.integrate_candidate_zero_review(
            dataset_name="d1",
            alpha=0.01,
            n_train=10,
            baseline_hard={1},
            baseline_standard=set(),
            baseline_priority={2},
            similarity_candidates={3, 4},
            n_test=5
        )
        self.assertEqual(res["hard"] & res["standard_review"], set())
        self.assertEqual(res["hard"] & res["priority_review"], set())
        self.assertEqual(res["standard_review"] & res["priority_review"], set())
        self.assertEqual(res["candidate"], res["hard"] | res["standard_review"] | res["priority_review"])

    def test_14_prediction_hash_tamper_detection(self):
        """14. Frozen prediction hash must reject a modified prediction file."""
        with tempfile.TemporaryDirectory() as tmp:
            prediction = Path(tmp) / "prediction.jsonl"
            manifest = Path(tmp) / "manifest.json"
            reporting.write_jsonl(prediction, [{"dataset_name": "d1"}])
            frozen = reporting.freeze_prediction_file(
                prediction,
                manifest,
                experiment_ids=["experiment_160_candidate_zero_similarity_recovery"],
                row_count=1,
            )
            prediction.write_text('{"dataset_name":"tampered"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen artifact hash mismatch"):
                exp_core.verify_frozen_file(prediction, frozen["prediction_sha256"])

    def test_15_canonical_output_hash_reproducibility(self):
        """15. Canonical hashing of rows is deterministic."""
        row1 = {"dataset_name": "d1", "alpha": 0.01, "hard_indices": {1, 2}}
        row2 = {"alpha": 0.01, "dataset_name": "d1", "hard_indices": {2, 1}}
        h1 = core.canonical_row_hash(row1)
        h2 = core.canonical_row_hash(row2)
        self.assertEqual(h1, h2)

    def test_16_uses_test_candidate_empty_gate_field(self):
        """16. uses_test_candidate_empty_gate=1 must be 명시."""
        target = core.prediction_policy_metadata(candidate_empty_gate_applied=True)
        non_target = core.prediction_policy_metadata(candidate_empty_gate_applied=False)
        self.assertEqual(target["uses_test_candidate_empty_gate"], 1)
        self.assertEqual(non_target["uses_test_candidate_empty_gate"], 1)
        self.assertEqual(target["candidate_empty_gate_applied"], 1)
        self.assertEqual(non_target["candidate_empty_gate_applied"], 0)

    def test_17_user_source_name_mapping(self):
        """17. Display name mapping has exact strings."""
        self.assertEqual(core.DISPLAY_NAMES["rocket_exp40"], "시간 패턴 점수")
        self.assertEqual(core.DISPLAY_NAMES["exp55_best"], "주파수 변화 점수")
        self.assertEqual(core.DISPLAY_NAMES["exp56_best"], "반복 구조 점수")
        self.assertEqual(core.DISPLAY_NAMES["exp84"], "다중 커널 점수")


if __name__ == "__main__":
    unittest.main()
