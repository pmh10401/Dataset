from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.exp438_frequency_complexity_evidence_antigravity.evaluation_antigravity import (
    EVALUATION_FILES_ANTIGRAVITY,
    _evaluation_artifacts_antigravity,
    _evaluate_frozen_coordinates_antigravity,
    _write_evaluation_artifacts_antigravity,
    evaluate_after_gates_antigravity,
)


class Exp438EvaluationAntigravityTests(unittest.TestCase):
    def test_any_prefreeze_gate_failure_keeps_label_access_zero(self) -> None:
        calls = []

        def labels():
            calls.append("labels")
            return {}

        with self.assertRaisesRegex(ValueError, "synthetic gate"):
            evaluate_after_gates_antigravity(
                gates=(lambda: None, lambda: (_ for _ in ()).throw(ValueError("synthetic gate"))),
                coordinate_loader=lambda: ({}, {}),
                label_loader=labels,
                bootstrap_samples=20,
            )
        self.assertEqual(calls, [])

    def test_channels_are_evaluated_separately_with_paired_bootstrap(self) -> None:
        baseline = {"A": (0,), "B": (0,)}
        channels = {
            "high_frequency": {"A": (1,), "B": ()},
            "low_frequency": {"A": (), "B": (1,)},
            "amplitude_complexity": {"A": (2,), "B": (1,)},
        }
        labels = {
            "A": {0: 1, 1: 1, 2: 0},
            "B": {0: 0, 1: 1, 2: 0},
        }
        result = _evaluate_frozen_coordinates_antigravity(
            channels,
            baseline,
            labels,
            bootstrap_samples=200,
            bootstrap_seed=43820260807,
        )
        high = result.channels["high_frequency"]
        self.assertEqual(high.confusion.to_payload(), {"tp": 2, "fp": 1, "fn": 1, "tn": 2})
        self.assertEqual(high.rescued_datasets, ("A",))
        self.assertEqual(high.missed_datasets, ("B",))
        self.assertEqual(high.regressed_datasets, ())
        self.assertEqual(high.bootstrap.samples, 200)
        self.assertEqual(high.full_candidate_count, 1)
        self.assertEqual(high.outside_lane_candidate_count, 1)
        self.assertEqual(high.dataset_rows[0].dataset, "A")
        self.assertTrue(high.dataset_rows[0].rescued)
        self.assertEqual(tuple(result.channels), (
            "high_frequency", "low_frequency", "amplitude_complexity"
        ))

    def test_full_and_outside_lane_candidate_counts_are_distinct(self) -> None:
        channels = {
            name: {"A": (1,)}
            for name in ("high_frequency", "low_frequency", "amplitude_complexity")
        }
        result = _evaluate_frozen_coordinates_antigravity(
            channels,
            {"A": (0,)},
            {"A": {0: 0, 1: 1}},
            bootstrap_samples=20,
            bootstrap_seed=1,
            full_candidate_counts={
                "high_frequency": 7,
                "low_frequency": 8,
                "amplitude_complexity": 9,
            },
        )
        self.assertEqual(result.channels["high_frequency"].full_candidate_count, 7)
        self.assertEqual(result.channels["high_frequency"].outside_lane_candidate_count, 1)

    def test_outside_lane_candidates_cannot_overlap_frozen_hard(self) -> None:
        channels = {
            name: {"A": (0,)}
            for name in ("high_frequency", "low_frequency", "amplitude_complexity")
        }
        with self.assertRaisesRegex(ValueError, "overlaps frozen Hard"):
            _evaluate_frozen_coordinates_antigravity(
                channels,
                {"A": (0,)},
                {"A": {0: 1}},
                bootstrap_samples=20,
                bootstrap_seed=1,
            )

    def test_evaluation_artifacts_are_complete_hashed_and_do_not_change_sources(self) -> None:
        channels = {
            name: {"A": (1,)}
            for name in ("high_frequency", "low_frequency", "amplitude_complexity")
        }
        result = _evaluate_frozen_coordinates_antigravity(
            channels,
            {"A": (0,)},
            {"A": {0: 0, 1: 1}},
            bootstrap_samples=20,
            bootstrap_seed=1,
        )
        payloads = _evaluation_artifacts_antigravity(result, prediction_sha256="a" * 64)
        self.assertEqual(set(payloads), EVALUATION_FILES_ANTIGRAVITY)
        hashes = json.loads(payloads["artifact_sha256_antigravity.json"])
        self.assertEqual(hashes["prediction_sha256"], "a" * 64)
        self.assertNotIn("artifact_sha256_antigravity.json", hashes["artifact_sha256"])
        for name, digest in hashes["artifact_sha256"].items():
            self.assertEqual(sha256(payloads[name]).hexdigest(), digest)

        with TemporaryDirectory() as temporary:
            parent = Path(temporary)
            source = parent / "prediction_antigravity.jsonl"
            source.write_bytes(b"frozen\n")
            before = source.read_bytes()
            target = parent / "evaluation_antigravity"
            with patch(
                "experiments.exp438_frequency_complexity_evidence_antigravity.evaluation_antigravity._exclusive_rename_antigravity",
                os.rename,
            ):
                _write_evaluation_artifacts_antigravity(
                    target,
                    result,
                    prediction_sha256="a" * 64,
                )
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual({path.name for path in target.iterdir()}, EVALUATION_FILES_ANTIGRAVITY)

    def test_unavailable_independence_cannot_pass_integration_gate(self) -> None:
        result = _evaluate_frozen_coordinates_antigravity(
            {name: {"A": ()} for name in ("high_frequency", "low_frequency", "amplitude_complexity")},
            {"A": ()},
            {"A": {0: 0}},
            bootstrap_samples=20,
            bootstrap_seed=1,
            train_oof_spearman=None,
        )
        self.assertFalse(result.integration_evidence_passed)

    def test_direct_and_integration_gates_can_pass_but_remain_separate(self) -> None:
        datasets = tuple(f"D{index:02d}" for index in range(10))
        labels = {dataset: {0: 1, 1: 1, 2: 1, 3: 0} for dataset in datasets}
        baseline = {dataset: () for dataset in datasets}
        channels = {
            channel: {dataset: (0, 1, 2) for dataset in datasets}
            for channel in ("high_frequency", "low_frequency", "amplitude_complexity")
        }
        correlations = {
            channel: {reference: 0.1 for reference in ("prototype", "gradient", "acf")}
            for channel in channels
        }
        aligned = {
            channel: {reference: 20 for reference in ("prototype", "gradient", "acf")}
            for channel in channels
        }
        result = _evaluate_frozen_coordinates_antigravity(
            channels,
            baseline,
            labels,
            bootstrap_samples=50,
            bootstrap_seed=1,
            train_oof_spearman=correlations,
            train_oof_aligned_counts=aligned,
        )
        self.assertTrue(result.direct_improvement_passed)
        self.assertTrue(result.integration_evidence_passed)

        underpowered = _evaluate_frozen_coordinates_antigravity(
            channels,
            baseline,
            labels,
            bootstrap_samples=50,
            bootstrap_seed=1,
            train_oof_spearman=correlations,
            train_oof_aligned_counts={
                channel: {reference: 19 for reference in ("prototype", "gradient", "acf")}
                for channel in channels
            },
        )
        self.assertTrue(underpowered.direct_improvement_passed)
        self.assertFalse(underpowered.integration_evidence_passed)


if __name__ == "__main__":
    unittest.main()
