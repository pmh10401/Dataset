"""Label-free contracts for the standalone Exp440 low-frequency replay."""
from __future__ import annotations

import unittest

import numpy as np

from paper.equipment_anomaly_detection.contracts_paper import (
    LowFrequencyCandidate,
    LowFrequencyDepthEvidence,
    RawDataset,
)
from paper.equipment_anomaly_detection.sources_paper.low_frequency_paper import (
    fit_group_calibration_paper,
    select_source_candidates,
    score_test_row_paper,
    score_low_frequency,
    strict_two_depth_pass,
)


def _values(dataset: str = "fixture") -> RawDataset:
    x = np.arange(64, dtype=np.float32)
    train = tuple((np.sin(x / (index + 2)) + index * 0.03).astype(np.float32) for index in range(5))
    test = (
        (9.0 * np.sin(x / 4)).astype(np.float32),
        np.full(64, np.nan, dtype=np.float32),
        np.arange(32, dtype=np.float32),
    )
    return RawDataset(dataset, 64, tuple(range(5)), (b"x",) * 5, train, (0, 1, 2), (b"y",) * 3, test)


class LowFrequencyTests(unittest.TestCase):
    def test_equal_threshold_is_not_a_hit(self) -> None:
        self.assertFalse(strict_two_depth_pass(np.asarray([3.0, 4.0]), np.asarray([3.0, 4.0])))

    def test_every_test_row_has_a_deterministic_evidence_record(self) -> None:
        result = score_low_frequency(_values(), {"fixture": frozenset()})
        self.assertEqual(tuple(row.instance_index for row in result.rows), (0, 1, 2))
        self.assertFalse(result.rows[0].evidence_missing)
        self.assertTrue(result.rows[1].evidence_missing)
        self.assertEqual(result.rows[1].missing_reason, "nonfinite_test")
        self.assertTrue(result.rows[2].evidence_missing)
        self.assertEqual(result.rows[2].missing_reason, "no_eligible_exact_length_group")

    def test_exact_length_and_minimum_four_finite_training_rows_are_required(self) -> None:
        values = _values()
        shortened = RawDataset(
            values.dataset, values.metadata_length, tuple(range(3)),
            values.train_blobs[:3], values.train_rows[:3], values.test_indices,
            values.test_blobs, values.test_rows,
        )
        result = score_low_frequency(shortened, {"fixture": frozenset()})
        self.assertTrue(all(row.evidence_missing for row in result.rows))
        self.assertTrue(all(row.missing_reason != "nonfinite_test" for row in result.rows[::2]))

    def test_nonfinite_train_row_is_excluded_without_poisoning_its_length_group(self) -> None:
        values = _values()
        train = values.train_rows[:4] + (np.full(64, np.nan, dtype=np.float32),)
        with_nonfinite_train = RawDataset(
            values.dataset, values.metadata_length, values.train_indices,
            values.train_blobs, train, (0,), (b"y",), values.test_rows[:1],
        )
        result = score_low_frequency(with_nonfinite_train, {"fixture": frozenset()})
        self.assertEqual(result.excluded_train_rows, 1)
        self.assertFalse(result.rows[0].evidence_missing)

    def test_hard_empty_one_per_dataset_global_30_and_exact_ties(self) -> None:
        candidates = tuple(
            LowFrequencyCandidate(
                dataset=f"D{index:02d}", instance_index=2,
                raw_length=64, depths=(1, 2), second_excess_ratio=2.0,
                second_empirical_rank=.2,
            )
            for index in range(31)
        ) + (
            LowFrequencyCandidate("D00", 1, 64, (1, 2), 3.0, .2),
            LowFrequencyCandidate("blocked", 0, 64, (1, 2), 99.0, .01),
        )
        hard = {**{f"D{index:02d}": frozenset() for index in range(31)}, "blocked": frozenset({0})}
        selected = select_source_candidates(candidates, hard)
        self.assertEqual(len(selected), 30)
        self.assertEqual(len({row.dataset for row in selected}), 30)
        self.assertEqual(selected[0].coordinate, ("D00", 1))
        self.assertEqual(selected[-1].dataset, "D29")
        self.assertTrue(all(not hard[row.dataset] for row in selected))

    def test_contracts_are_immutable_and_reject_inconsistent_depth_evidence(self) -> None:
        candidate = LowFrequencyCandidate("D", 0, 64, (1, 2), 1.1, .5)
        with self.assertRaises((AttributeError, TypeError)):
            candidate.dataset = "changed"  # type: ignore[misc]
        with self.assertRaises(ValueError):
            LowFrequencyCandidate("D", 0, 64, (2, 1), 1.1, .5)
        with self.assertRaises(ValueError):
            LowFrequencyDepthEvidence(1, -1.0, 1.0, False, 0.0, .5)
        evidence = (
            LowFrequencyDepthEvidence(1, 2.0, 1.0, True, 2.0, .2),
            LowFrequencyDepthEvidence(2, 2.0, 1.0, True, 2.0, .2),
        )
        with self.assertRaises(ValueError):
            LowFrequencyCandidate("D", 0, 64, (1, 2), 1.1, .5, evidence)

    def test_mapping_input_exposes_a_cross_dataset_global_selection(self) -> None:
        values = {"A": _values("A"), "B": _values("B")}
        result = score_low_frequency(values, {"A": frozenset(), "B": frozenset()})
        self.assertEqual(tuple(candidate.dataset for candidate in result.selected_candidates), ("A", "B"))
        self.assertEqual(len(result.selected_candidates), 2)
        with self.assertRaises(TypeError):
            result.hard_lanes["A"] = frozenset({0})  # type: ignore[index]

    def test_paper_module_has_no_legacy_exp440_runtime_import(self) -> None:
        import sys

        sys.modules.pop("experiments.exp440_train_only_low_frequency_precision_gate_antigravity.scoring_antigravity", None)
        score_low_frequency(_values(), {"fixture": frozenset()})
        self.assertNotIn("experiments.exp440_train_only_low_frequency_precision_gate_antigravity.scoring_antigravity", sys.modules)

    def test_executable_exp440_and_exp441_origins_match_nonvacuously(self) -> None:
        """Origin oracle and paper closure use separate execution paths."""
        from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.features_antigravity import fit_group_calibration_antigravity
        from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.scoring_antigravity import score_test_row_antigravity
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity.contracts_antigravity import CandidateEvidenceAntigravity, FrozenSourceSnapshotAntigravity
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity.selector_antigravity import select_hard_empty_single_shot_antigravity

        values = _values()
        matrix = np.stack(values.train_rows)
        origin_group, paper_group = fit_group_calibration_antigravity(matrix), fit_group_calibration_paper(matrix)
        self.assertTrue(origin_group.eligible and paper_group.eligible)
        self.assertEqual(tuple(item.calibration.depth for item in origin_group.models), tuple(item.calibration.depth for item in paper_group.models))
        for origin, paper in zip(origin_group.models, paper_group.models, strict=True):
            np.testing.assert_allclose(origin.references, paper.references, rtol=0.0, atol=0.0)
            self.assertEqual(origin.calibration.oof_scores, paper.calibration.oof_scores)
            self.assertEqual(origin.calibration.threshold, paper.calibration.threshold)

        origin_scores, origin_candidate = score_test_row_antigravity(values.dataset, 0, values.test_rows[0], origin_group)
        paper_scores, paper_candidate = score_test_row_paper(values.dataset, 0, values.test_rows[0], paper_group)
        self.assertIsNotNone(origin_candidate)
        self.assertIsNotNone(paper_candidate)
        self.assertEqual(tuple((item.depth, item.score, item.threshold, item.exceeds, item.empirical_rank) for item in origin_scores), tuple((item.depth, item.score, item.threshold, item.strict_pass, item.empirical_rank) for item in paper_scores))
        self.assertEqual((origin_candidate.depths, origin_candidate.second_excess_ratio, origin_candidate.second_empirical_rank), (paper_candidate.depths, paper_candidate.second_excess_ratio, paper_candidate.second_empirical_rank))

        candidates = tuple(LowFrequencyCandidate(f"P{index:02d}", 1, 64, (1, 2), 2.0, .2) for index in range(31))
        hard = {**{candidate.dataset: frozenset() for candidate in candidates}, "blocked": frozenset({0})}
        paper_selected = select_source_candidates(candidates + (LowFrequencyCandidate("blocked", 0, 64, (1, 2), 9.0, .1),), hard)
        origin_candidates = tuple(CandidateEvidenceAntigravity(item.dataset, item.instance_index, item.raw_length, item.depths, len(item.depths), item.second_excess_ratio, item.second_empirical_rank) for item in candidates) + (CandidateEvidenceAntigravity("blocked", 0, 64, (1, 2), 2, 9.0, .1),)
        datasets = tuple(sorted(hard))
        snapshot = FrozenSourceSnapshotAntigravity(
            datasets, {dataset: 2 for dataset in datasets}, {dataset: "0" * 64 for dataset in datasets},
            frozenset({("blocked", 0)}), origin_candidates, *("0" * 64,) * 9,
        )
        origin_selected = select_hard_empty_single_shot_antigravity(snapshot).selected
        self.assertEqual(tuple(item.coordinate for item in paper_selected), tuple(item.coordinate for item in origin_selected))


if __name__ == "__main__":
    unittest.main()
