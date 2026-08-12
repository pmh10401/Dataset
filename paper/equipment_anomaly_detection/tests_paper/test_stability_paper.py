"""RED/GREEN contracts for selected-candidate stability and proposal routes."""
from __future__ import annotations

import ast
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
import unittest

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
        test_indices=(0, 1), test_blobs=(b"test", b"hard"), test_rows=(test, test + np.float32(.01)),
    )


def _selected(values: RawDataset) -> LowFrequencyCandidate:
    fitted = fit_group_calibration_paper(np.stack(values.train_rows))
    _scores, candidate = score_test_row_paper(values.dataset, 0, values.test_rows[0], fitted)
    if candidate is None:
        raise AssertionError("fixture must be a nonvacuous low-frequency candidate")
    return candidate


def _result(*values: RawDataset) -> LowFrequencyResult:
    candidates = tuple(_selected(item) for item in values)
    rows = tuple(
        LowFrequencyRow(
            candidate.dataset, candidate.instance_index, candidate.raw_length,
            candidate.depth_evidence, True, candidate.second_excess_ratio,
            candidate.second_empirical_rank, True, False, None,
        )
        for candidate in candidates
    )
    datasets = tuple(sorted(candidate.dataset for candidate in candidates))
    return LowFrequencyResult(
        datasets=datasets, rows=tuple(sorted(rows, key=lambda row: (row.dataset, row.instance_index))),
        provisional_candidates=tuple(sorted(candidates, key=lambda candidate: candidate.global_key)),
        selected_candidates=tuple(sorted(candidates, key=lambda candidate: candidate.global_key)),
        excluded_train_rows=0, hard_lanes={dataset: frozenset() for dataset in datasets},
    )


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
        low = _result(first, second)
        selected = tuple(candidate.coordinate for candidate in low.selected_candidates)
        def classified(candidate: LowFrequencyCandidate, stable: bool):
            view = StabilityViewPaper(
                OmissionMaskPaper(0, (0,), "1" * 64), tuple(range(1, 8)), candidate.depth_evidence,
                candidate.depths if stable else (), stable, None if stable else "candidate_absent",
            )
            return aggregate_stability_views(candidate.coordinate, (view,))

        classified_by_coordinate = {
            selected[0]: classified(next(candidate for candidate in low.selected_candidates if candidate.coordinate == selected[0]), True),
            selected[1]: classified(next(candidate for candidate in low.selected_candidates if candidate.coordinate == selected[1]), False),
        }
        partition = StabilityPartition(selected, (selected[0],), (selected[1],), classified_by_coordinate)
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
        self.assertEqual(coordinates[3], frozenset({hard_coordinate, selected[1]}))
        for policy in policies.policies:
            for dataset, score in policy.scores.items():
                np.testing.assert_array_equal(score >= .5, np.asarray([index in policy.alerts[dataset] for index in range(score.size)]))
        second_candidate = next(candidate for candidate in low.selected_candidates if candidate.dataset == second.dataset)
        self.assertEqual(
            policies.by_name["hard_only"].strength[second.dataset][0],
            max(.2, 1.0 - second_candidate.second_empirical_rank),
        )
        stable_candidate = next(candidate for candidate in low.selected_candidates if candidate.coordinate == selected[0])
        self.assertEqual(
            policies.by_name["hard_plus_stable"].strength[stable_candidate.dataset][stable_candidate.instance_index],
            max(.01, 1.0 - stable_candidate.second_empirical_rank),
        )

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

        # The executable origin accepts the same raw per-view evidence and
        # independently applies 14/16 to a stable lane and a lane with explicit
        # candidate-absent views; it has no paper partition dependency.
        from experiments.exp444_train_only_stability_precision_gate_antigravity.contracts_antigravity import (
            DepthViewScoreAntigravity,
            OmissionMaskAntigravity,
            SourceCandidateAntigravity,
            StabilityViewAntigravity,
        )
        from experiments.exp444_train_only_stability_precision_gate_antigravity.stability_antigravity import (
            aggregate_stability_views_antigravity,
        )

        def origin_result(dataset: str):
            candidate = selected_by_dataset[dataset]
            paper_result = partition.by_coordinate[candidate.coordinate]
            source = SourceCandidateAntigravity(dataset, candidate.instance_index, candidate.raw_length, candidate.depths, 1)
            views = tuple(
                StabilityViewAntigravity(
                    OmissionMaskAntigravity(view.view_index, view.mask.excluded_train_indices, view.mask.identity_sha256),
                    view.retained_train_indices,
                    tuple(DepthViewScoreAntigravity(item.depth, item.score, item.threshold, item.strict_pass, item.excess_ratio, item.empirical_rank) for item in view.depth_evidence),
                    view.agreeing_depths, view.survived,
                )
                for view in paper_result.views
            )
            return aggregate_stability_views_antigravity(source, views)

        self.assertEqual(
            (origin_result(first.dataset).survival_count, origin_result(first.dataset).required_survival_count, origin_result(first.dataset).lane),
            (16, 14, "stable_review"),
        )
        self.assertEqual(
            (origin_result(second.dataset).survival_count, origin_result(second.dataset).required_survival_count, origin_result(second.dataset).lane),
            (13, 14, "unstable_review"),
        )

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
