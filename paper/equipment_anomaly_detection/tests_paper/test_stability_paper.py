"""RED/GREEN contracts for selected-candidate stability and proposal routes."""
from __future__ import annotations

import ast
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from paper.equipment_anomaly_detection.contracts_paper import (
    LowFrequencyCandidate,
    LowFrequencyDepthEvidence,
    LowFrequencyResult,
    LowFrequencyRow,
    RawDataset,
)
from paper.equipment_anomaly_detection.sources_paper.low_frequency_paper import (
    fit_group_calibration_paper,
    score_test_row_paper,
)


def _raw(dataset: str, *, offset: float = 0.0, train_count: int = 8) -> RawDataset:
    x = np.arange(32, dtype=np.float32)
    rng = np.random.default_rng(44 + int(offset))
    train = tuple(
        (np.sin(x / (2.0 + index * .1)) + offset + index * .05 + rng.normal(0.0, .1, x.size)).astype("<f4")
        for index in range(train_count)
    )
    test = (np.mean(np.stack(train), axis=0) + np.float32(8.0)).astype("<f4")
    return RawDataset(
        dataset=dataset, metadata_length=32,
        train_indices=tuple(range(len(train))), train_blobs=tuple(bytes([index]) for index in range(len(train))), train_rows=train,
        test_indices=(0, 1), test_blobs=(b"test", b"hard"), test_rows=(test, train[0]),
    )


def _selected(values: RawDataset) -> LowFrequencyCandidate:
    fitted = fit_group_calibration_paper(np.stack(values.train_rows))
    _scores, candidate = score_test_row_paper(values.dataset, 0, values.test_rows[0], fitted)
    if candidate is None:
        raise AssertionError("fixture must be a nonvacuous low-frequency candidate")
    return candidate


def _result(*values: RawDataset, hard_lanes: dict[str, frozenset[int]] | None = None) -> LowFrequencyResult:
    rows_list: list[LowFrequencyRow] = []
    candidates_list: list[LowFrequencyCandidate] = []
    for values_item in values:
        fitted = fit_group_calibration_paper(np.stack(values_item.train_rows))
        for index, test in zip(values_item.test_indices, values_item.test_rows, strict=True):
            evidence, candidate = score_test_row_paper(values_item.dataset, index, test, fitted)
            if candidate is None:
                rows_list.append(LowFrequencyRow(values_item.dataset, index, int(test.size), evidence, False, None, None, False, False, None))
            else:
                rows_list.append(LowFrequencyRow(values_item.dataset, index, candidate.raw_length, candidate.depth_evidence, True, candidate.second_excess_ratio, candidate.second_empirical_rank, True, False, None))
                candidates_list.append(candidate)
    rows = tuple(rows_list)
    candidates = tuple(candidates_list)
    datasets = tuple(sorted(candidate.dataset for candidate in candidates))
    lanes = {dataset: frozenset() for dataset in datasets} if hard_lanes is None else hard_lanes
    selected = tuple(sorted(
        (candidate for candidate in candidates if not lanes[candidate.dataset]),
        key=lambda candidate: candidate.global_key,
    )[:30])
    return LowFrequencyResult(
        datasets=datasets, rows=tuple(sorted(rows, key=lambda row: (row.dataset, row.instance_index))),
        provisional_candidates=tuple(sorted(candidates, key=lambda candidate: candidate.global_key)),
        selected_candidates=selected,
        excluded_train_rows=0, hard_lanes=lanes,
    )


def _partition_for(*candidates: LowFrequencyCandidate):
    from paper.equipment_anomaly_detection.sources_paper.stability_paper import (
        OmissionMaskPaper,
        StabilityPartition,
        StabilityViewPaper,
        aggregate_stability_views,
    )

    results = {}
    for candidate in candidates:
        view = StabilityViewPaper(
            OmissionMaskPaper(0, (99,), "9" * 64), (0, 1, 2, 3), candidate.depth_evidence,
            candidate.depths, True, None,
        )
        results[candidate.coordinate] = aggregate_stability_views(candidate.coordinate, (view,))
    coordinates = tuple(candidate.coordinate for candidate in candidates)
    return StabilityPartition(coordinates, coordinates, (), results)


class StabilityPaperTests(unittest.TestCase):
    def test_masks_required_counts_and_survival_boundary_match_independent_origin(self) -> None:
        """Paper masks/thresholds and 14/16 boundary equal an independently imported origin."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity.masks_antigravity import (
            generate_omission_masks_antigravity,
            required_survival_count_antigravity,
        )
        from paper.equipment_anomaly_detection.sources_paper.stability_paper import (
            OmissionMaskPaper,
            StabilityViewPaper,
            aggregate_stability_views,
            generate_omission_masks_paper,
            required_survival_count_paper,
        )

        origin = generate_omission_masks_antigravity("Fixture_normal_1", 32, tuple(range(8)))
        paper = generate_omission_masks_paper("Fixture_normal_1", 32, tuple(range(8)))
        self.assertEqual(
            tuple((item.view_index, item.excluded_train_indices, item.identity_sha256) for item in paper),
            tuple((item.view_index, item.excluded_train_indices, item.identity_sha256) for item in origin),
        )
        for count in (1, 2, 7, 8, 15, 16, 17):
            self.assertEqual(required_survival_count_paper(count), required_survival_count_antigravity(count))
        self.assertEqual(required_survival_count_paper(16), 14)
        candidate = _selected(_raw("Stable" ))
        scores = candidate.depth_evidence
        stable_views = tuple(
            StabilityViewPaper(
                OmissionMaskPaper(index, (index,), f"{index + 1:064x}"),
                (100, 101, 102, 103), scores, candidate.depths if index < 14 else (), index < 14,
                None if index < 14 else "candidate_absent",
            )
            for index in range(16)
        )
        stable = aggregate_stability_views(("Stable", 0), stable_views)
        unstable = aggregate_stability_views(("Unstable", 0), tuple(
            StabilityViewPaper(
                item.mask, item.retained_train_indices, item.depth_evidence,
                candidate.depths if index < 13 else (), index < 13,
                None if index < 13 else "candidate_absent",
            )
            for index, item in enumerate(stable_views)
        ))
        self.assertEqual((stable.survival_count, stable.required_survival_count, stable.lane), (14, 14, "stable_review"))
        self.assertEqual((unstable.survival_count, unstable.required_survival_count, unstable.lane), (13, 14, "unstable_review"))

    def test_classification_refits_only_retained_train_and_partitions_selected_coordinates(self) -> None:
        from paper.equipment_anomaly_detection.sources_paper.stability_paper import classify_stability

        first, second = _raw("Alpha_normal_1"), _raw("Beta_normal_1", offset=1.0)
        result = _result(first, second)
        observed: list[tuple[int, ...]] = []

        def replay(dataset: str, index: int, test: np.ndarray, retained: np.ndarray):
            observed.append(tuple(int(value) for value in retained[:, 0]))
            fitted = fit_group_calibration_paper(retained)
            return score_test_row_paper(dataset, index, test, fitted)

        partition = classify_stability(result, {first.dataset: first, second.dataset: second}, replay=replay)
        selected = {candidate.coordinate for candidate in result.selected_candidates}
        self.assertEqual(set(partition.stable_coordinates) | set(partition.unstable_coordinates), selected)
        self.assertFalse(set(partition.stable_coordinates) & set(partition.unstable_coordinates))
        self.assertEqual(len(partition.by_coordinate), len(selected))
        self.assertTrue(observed)
        # Every view loses exactly one of the eight TRAIN rows; no full TRAIN fit is allowed here.
        self.assertTrue(all(len(retained) == 7 for retained in observed))

    def test_policy_order_coordinates_strength_and_score_routes_are_exact(self) -> None:
        from paper.equipment_anomaly_detection.routing_paper import (
            POLICY_ORDER,
            EvidenceSurface,
            build_policies,
        )
        from paper.equipment_anomaly_detection.sources_paper.stability_paper import (
            OmissionMaskPaper,
            StabilityPartition,
            StabilityViewPaper,
            aggregate_stability_views,
        )

        first, second = _raw("Alpha_normal_1", train_count=20), _raw("Beta_normal_1", offset=1.0, train_count=20)
        low = _result(first, second, hard_lanes={first.dataset: frozenset(), second.dataset: frozenset({1})})
        selected = tuple(candidate.coordinate for candidate in low.selected_candidates)
        self.assertEqual(len(selected), 1)
        partition = _partition_for(low.selected_candidates[0])
        hard = EvidenceSurface(
            alerts={first.dataset: frozenset(), second.dataset: frozenset({1})},
            strength={first.dataset: np.asarray([.01, .2]), second.dataset: np.asarray([.2, .9])},
            evidence_missing={first.dataset: np.asarray([False, False]), second.dataset: np.asarray([False, False])},
        )
        policies = build_policies(hard, low, partition, {first.dataset: first, second.dataset: second})
        self.assertEqual(policies.order, POLICY_ORDER)
        self.assertEqual(tuple(policy.name for policy in policies.policies), POLICY_ORDER)
        coordinates = [policy.coordinates for policy in policies.policies]
        hard_coordinate = (second.dataset, 1)
        self.assertEqual(coordinates[0], frozenset({hard_coordinate}))
        self.assertEqual(coordinates[1], frozenset({hard_coordinate, *selected}))
        self.assertEqual(coordinates[2], frozenset({hard_coordinate, selected[0]}))
        self.assertEqual(coordinates[3], frozenset({hard_coordinate}))
        for policy in policies.policies:
            for dataset, score in policy.scores.items():
                np.testing.assert_array_equal(score >= .5, np.asarray([index in policy.alerts[dataset] for index in range(score.size)]))
        second_row = next(row for row in low.rows if row.dataset == second.dataset and row.instance_index == 0)
        self.assertEqual(
            policies.by_name["hard_only"].strength[second.dataset][0],
            max(.2, sorted((1.0 - item.empirical_rank for item in second_row.depth_evidence), reverse=True)[1]),
        )
        stable_candidate = next(candidate for candidate in low.selected_candidates if candidate.coordinate == selected[0])
        self.assertEqual(
            policies.by_name["hard_plus_stable"].strength[stable_candidate.dataset][stable_candidate.instance_index],
            max(.01, 1.0 - stable_candidate.second_empirical_rank),
        )

    def test_mixed_length_and_nonfinite_train_rows_match_origin_filtered_masks(self) -> None:
        """Only finite TRAIN rows of the candidate's exact raw length enter Exp444 views."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity.contracts_antigravity import (
            SourceCandidateAntigravity,
        )
        from experiments.exp444_train_only_stability_precision_gate_antigravity.masks_antigravity import (
            generate_omission_masks_antigravity,
        )
        from experiments.exp444_train_only_stability_precision_gate_antigravity.sources_antigravity import (
            _prepare_candidate_input_antigravity,
        )
        from paper.equipment_anomaly_detection.sources_paper.stability_paper import classify_stability

        base = _raw("Mixed_normal_1")
        eligible = (base.train_rows[0], base.train_rows[1], base.train_rows[2], base.train_rows[3], base.train_rows[4], base.train_rows[5])
        nonfinite = base.train_rows[6].copy()
        nonfinite[0] = np.nan
        train_rows = (eligible[0], np.arange(24, dtype=np.float32), eligible[1], nonfinite, eligible[2], eligible[3], eligible[4], eligible[5])
        values = RawDataset(
            dataset=base.dataset, metadata_length=32,
            train_indices=tuple(range(8)), train_blobs=tuple(bytes([index]) for index in range(8)), train_rows=train_rows,
            test_indices=base.test_indices, test_blobs=base.test_blobs, test_rows=base.test_rows,
        )
        fitted = fit_group_calibration_paper(np.stack(eligible))
        _scores, candidate = score_test_row_paper(values.dataset, 0, values.test_rows[0], fitted)
        self.assertIsNotNone(candidate)
        assert candidate is not None
        row = LowFrequencyRow(values.dataset, 0, candidate.raw_length, candidate.depth_evidence, True, candidate.second_excess_ratio, candidate.second_empirical_rank, True, False, None)
        low = LowFrequencyResult((values.dataset,), (row,), (candidate,), (candidate,), 1, {values.dataset: frozenset()})
        # This executes Exp444's independent source closure rather than
        # reconstructing the eligible rows from paper output.
        origin_input = _prepare_candidate_input_antigravity(
            SourceCandidateAntigravity(candidate.dataset, candidate.instance_index, candidate.raw_length, candidate.depths, 1),
            tuple((index, np.asarray(row, dtype="<f4").tobytes()) for index, row in zip(values.train_indices, values.train_rows, strict=True)),
            ((candidate.instance_index, np.asarray(values.test_rows[candidate.instance_index], dtype="<f4").tobytes()),),
        )
        origin_masks = generate_omission_masks_antigravity(values.dataset, candidate.raw_length, origin_input.train_indices)

        paper = classify_stability(low, {values.dataset: values}).by_coordinate[candidate.coordinate]
        self.assertEqual(
            tuple(view.retained_train_indices for view in paper.views),
            tuple(tuple(index for index in origin_input.train_indices if index not in mask.excluded_train_indices) for mask in origin_masks),
        )
        self.assertEqual(
            tuple((view.mask.excluded_train_indices, view.mask.identity_sha256) for view in paper.views),
            tuple((mask.excluded_train_indices, mask.identity_sha256) for mask in origin_masks),
        )

    def test_source_strength_covers_scoreable_nonselected_rows_and_hard_identity_is_bound(self) -> None:
        """Selection only controls alerts; every finite Task 6 row keeps continuous evidence."""
        from paper.equipment_anomaly_detection.routing_paper import EvidenceSurface, build_policies

        values = _raw("Strength_normal_1")
        candidate = _selected(values)
        values = RawDataset(
            dataset=values.dataset, metadata_length=values.metadata_length,
            train_indices=values.train_indices, train_blobs=values.train_blobs, train_rows=values.train_rows,
            test_indices=(0, 1, 2), test_blobs=(b"test", b"normal", b"unscoreable"),
            test_rows=(values.test_rows[0], values.test_rows[1], np.arange(12, dtype="<f4")),
        )
        nonselected_evidence = tuple(
            LowFrequencyDepthEvidence(depth, 1.0, 1.0, False, 1.0, rank)
            for depth, rank in ((1, .2), (2, .4))
        )
        rows = (
            LowFrequencyRow(values.dataset, 0, candidate.raw_length, candidate.depth_evidence, True, candidate.second_excess_ratio, candidate.second_empirical_rank, True, False, None),
            LowFrequencyRow(values.dataset, 1, candidate.raw_length, nonselected_evidence, False, None, None, False, False, None),
            LowFrequencyRow(values.dataset, 2, 12, (), False, None, None, False, True, "no_eligible_exact_length_group"),
        )
        low = LowFrequencyResult((values.dataset,), rows, (candidate,), (candidate,), 0, {values.dataset: frozenset()})
        partition = _partition_for(candidate)
        hard = EvidenceSurface({values.dataset: frozenset()}, {values.dataset: np.asarray([0.0, 0.0, 0.0])}, {values.dataset: np.asarray([False, False, True])})
        policies = build_policies(hard, low, partition, {values.dataset: values})
        for policy in policies.policies:
            self.assertGreater(policy.strength[values.dataset][1], 0.0)
            self.assertGreater(policy.scores[values.dataset][1], 0.0)
            self.assertLess(policy.scores[values.dataset][1], .5)
            self.assertFalse(policy.alerts[values.dataset] & frozenset({1}))
            self.assertEqual((policy.strength[values.dataset][2], policy.scores[values.dataset][2]), (0.0, .25))
            self.assertFalse(policy.alerts[values.dataset] & frozenset({2}))

        mismatched_hard = EvidenceSurface({values.dataset: frozenset({1})}, {values.dataset: np.asarray([0.0, .2, 0.0])}, {values.dataset: np.asarray([False, False, True])})
        with self.assertRaisesRegex(ValueError, "Hard"):
            build_policies(mismatched_hard, low, partition, {values.dataset: values})

    def test_selected_only_replay_has_stable_unstable_and_missing_view_nonvacuity(self) -> None:
        """Candidate absence is recorded per view and cannot silently become survival."""
        from paper.equipment_anomaly_detection.sources_paper.stability_paper import classify_stability

        first, second = _raw("Alpha_normal_1", train_count=20), _raw("Beta_normal_1", offset=1.0, train_count=20)
        low = _result(first, second)
        selected_by_dataset = {candidate.dataset: candidate for candidate in low.selected_candidates}
        calls = {first.dataset: 0, second.dataset: 0}

        def replay(dataset: str, index: int, test: np.ndarray, retained: np.ndarray):
            calls[dataset] += 1
            candidate = selected_by_dataset[dataset]
            # This controlled oracle makes one selected candidate stable while
            # the other has exactly three explicit unscoreable views: 13/16.
            if dataset == second.dataset and calls[dataset] <= 3:
                return candidate.depth_evidence, None
            return candidate.depth_evidence, candidate

        partition = classify_stability(low, {first.dataset: first, second.dataset: second}, replay=replay)
        self.assertEqual(calls, {first.dataset: 16, second.dataset: 16})
        self.assertEqual(set(partition.stable_coordinates), {(first.dataset, 0)})
        self.assertEqual(set(partition.unstable_coordinates), {(second.dataset, 0)})
        unstable = partition.by_coordinate[(second.dataset, 0)]
        self.assertEqual((unstable.survival_count, unstable.required_survival_count, unstable.lane), (13, 14, "unstable_review"))
        self.assertEqual(sum(view.missing_reason == "candidate_absent" for view in unstable.views), 3)
        self.assertTrue(all(not view.survived for view in unstable.views[:3]))

    def test_independent_origin_controls_cover_stable_unstable_and_missing_views(self) -> None:
        """Origin output is executed from native contracts, never rebuilt from paper views."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity.contracts_antigravity import (
            CandidateInputAntigravity,
            SourceCandidateAntigravity,
        )
        from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.contracts_antigravity import (
            DepthScoreAntigravity,
            ProvisionalCandidateAntigravity,
        )
        from experiments.exp444_train_only_stability_precision_gate_antigravity import stability_antigravity as origin
        from paper.equipment_anomaly_detection.sources_paper.stability_paper import classify_stability

        first, second = _raw("OriginStable_normal_1", train_count=20), _raw("OriginUnstable_normal_1", offset=1.0, train_count=20)
        low = _result(first, second)
        paper_candidates = {candidate.dataset: candidate for candidate in low.selected_candidates}
        native_scores = tuple(DepthScoreAntigravity(depth, 2.0, 1.0, True, 2.0, .5) for depth in (1, 2))
        native_calls = {first.dataset: 0, second.dataset: 0}

        def native_candidate(dataset: str) -> ProvisionalCandidateAntigravity:
            return ProvisionalCandidateAntigravity(dataset, 0, 32, (1, 2), 2, 2.0, .5)

        def native_score(dataset: str, index: int, test: np.ndarray, fitted: object, **_kwargs: object):
            self.assertEqual((index, test.size), (0, 32))
            native_calls[dataset] += 1
            return native_scores, None if dataset == second.dataset and native_calls[dataset] <= 3 else native_candidate(dataset)

        def native_input(values: RawDataset, rank: int) -> CandidateInputAntigravity:
            source = SourceCandidateAntigravity(values.dataset, 0, 32, (1, 2), rank)
            return CandidateInputAntigravity(source, values.train_indices, np.stack(values.train_rows), values.test_rows[0])

        with patch.object(origin, "replay_original_candidate_antigravity", side_effect=lambda item, distance: SimpleNamespace(original_depths=(1, 2))), patch.object(origin, "fit_group_calibration_antigravity", return_value=object()), patch.object(origin, "score_test_row_antigravity", side_effect=native_score):
            native = {
                first.dataset: origin.score_stability_candidate_antigravity(native_input(first, 1)),
                second.dataset: origin.score_stability_candidate_antigravity(native_input(second, 2)),
            }
        self.assertEqual(native_calls, {first.dataset: 16, second.dataset: 16})
        self.assertEqual((native[first.dataset].survival_count, native[first.dataset].required_survival_count, native[first.dataset].lane), (16, 14, "stable_review"))
        self.assertEqual((native[second.dataset].survival_count, native[second.dataset].required_survival_count, native[second.dataset].lane), (13, 14, "unstable_review"))
        self.assertEqual(sum(not view.survived and not view.agreeing_depths for view in native[second.dataset].views), 3)

        paper_calls = {first.dataset: 0, second.dataset: 0}
        def paper_score(dataset: str, index: int, test: np.ndarray, retained: np.ndarray):
            paper_calls[dataset] += 1
            candidate = paper_candidates[dataset]
            return candidate.depth_evidence, None if dataset == second.dataset and paper_calls[dataset] <= 3 else candidate
        paper = classify_stability(low, {first.dataset: first, second.dataset: second}, replay=paper_score)
        self.assertEqual(paper_calls, native_calls)
        for dataset, native_result in native.items():
            paper_result = paper.by_coordinate[(dataset, 0)]
            self.assertEqual(
                (paper_result.survival_count, paper_result.required_survival_count, paper_result.lane),
                (native_result.survival_count, native_result.required_survival_count, native_result.lane),
            )
            self.assertEqual(
                tuple((view.mask.excluded_train_indices, view.retained_train_indices, view.survived) for view in paper_result.views),
                tuple((view.mask.excluded_train_indices, view.retained_train_indices, view.survived) for view in native_result.views),
            )
        self.assertEqual(tuple(view.missing_reason for view in paper.by_coordinate[(second.dataset, 0)].views[:3]), ("candidate_absent",) * 3)

    def test_contracts_are_deeply_immutable_and_reject_overlap_omission_nonfinite_and_score_mismatch(self) -> None:
        from paper.equipment_anomaly_detection.routing_paper import EvidenceSurface, PolicyBundle, PolicyRoute
        from paper.equipment_anomaly_detection.sources_paper.stability_paper import StabilityPartition

        strength = np.asarray([.2])
        surface = EvidenceSurface({"A": frozenset({0})}, {"A": strength}, {"A": np.asarray([False])})
        strength[0] = .8
        self.assertEqual(surface.strength["A"].tolist(), [.2])
        with self.assertRaises(ValueError):
            surface.strength["A"][0] = .1
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            EvidenceSurface({"A": frozenset()}, {"A": np.asarray([np.nan])}, {"A": np.asarray([False])})
        with self.assertRaisesRegex(ValueError, "overlap"):
            StabilityPartition((("A", 0),), (("A", 0),), (("A", 0),), {})
        route = PolicyRoute("hard_only", {"A": frozenset({0})}, {"A": np.asarray([.2])}, {"A": np.asarray([.6])})
        with self.assertRaisesRegex(ValueError, "order"):
            PolicyBundle((route,), ("wrong",))
        with self.assertRaisesRegex(ValueError, "route"):
            PolicyRoute("hard_only", {"A": frozenset({0})}, {"A": np.asarray([.2])}, {"A": np.asarray([.1])})

    def test_runtime_isolation_from_exp444_and_origin_provenance_is_bound(self) -> None:
        import paper.equipment_anomaly_detection.sources_paper.stability_paper as stability

        source = Path(stability.__file__).read_text(encoding="utf-8")
        self.assertNotIn("experiments.exp444_train_only_stability_precision_gate_antigravity", source)
        isolated = subprocess.check_output(
            [sys.executable, "-c", "import sys; import paper.equipment_anomaly_detection.sources_paper.stability_paper; print(any(name.startswith('experiments.exp444_train_only_stability_precision_gate_antigravity') for name in sys.modules))"],
            text=True,
        ).strip()
        self.assertEqual(isolated, "False")
        self.assertEqual(
            stability.PROVENANCE_PAPER["experiments/exp444_train_only_stability_precision_gate_antigravity/stability_antigravity.py"]["source_sha256"],
            sha256(Path("experiments/exp444_train_only_stability_precision_gate_antigravity/stability_antigravity.py").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            stability.PROVENANCE_PAPER["experiments/exp444_train_only_stability_precision_gate_antigravity/sources_antigravity.py"]["source_sha256"],
            sha256(Path("experiments/exp444_train_only_stability_precision_gate_antigravity/sources_antigravity.py").read_bytes()).hexdigest(),
        )

    def test_executable_origin_replay_matches_paper_views_and_partition_nonvacuously(self) -> None:
        """Origin and paper paths independently replay the same selected row and masks."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity.contracts_antigravity import (
            CandidateInputAntigravity,
            SourceCandidateAntigravity,
        )
        from experiments.exp444_train_only_stability_precision_gate_antigravity.stability_antigravity import (
            score_stability_candidate_antigravity,
        )
        from paper.equipment_anomaly_detection.sources_paper.stability_paper import classify_stability

        values = _raw("Parity_normal_1")
        low = _result(values)
        candidate = low.selected_candidates[0]
        origin_input = CandidateInputAntigravity(
            SourceCandidateAntigravity(candidate.dataset, candidate.instance_index, candidate.raw_length, candidate.depths, 1),
            values.train_indices, np.stack(values.train_rows), values.test_rows[candidate.instance_index],
        )
        origin = score_stability_candidate_antigravity(origin_input)
        paper = classify_stability(low, {values.dataset: values}).by_coordinate[candidate.coordinate]
        self.assertEqual((paper.survival_count, paper.required_survival_count, paper.lane), (origin.survival_count, origin.required_survival_count, origin.lane))
        self.assertEqual(len(paper.views), len(origin.views))
        self.assertTrue(any(view.survived for view in paper.views))
        for paper_view, origin_view in zip(paper.views, origin.views, strict=True):
            self.assertEqual(
                (paper_view.mask.view_index, paper_view.mask.excluded_train_indices, paper_view.mask.identity_sha256, paper_view.retained_train_indices, paper_view.survived),
                (origin_view.mask.view_index, origin_view.mask.excluded_train_indices, origin_view.mask.identity_sha256, origin_view.retained_train_indices, origin_view.survived),
            )
            self.assertEqual(
                tuple((item.depth, item.score, item.threshold, item.strict_pass) for item in paper_view.depth_evidence),
                tuple((item.depth, item.score, item.threshold, item.exceeds) for item in origin_view.depth_scores),
            )


if __name__ == "__main__":
    unittest.main()
