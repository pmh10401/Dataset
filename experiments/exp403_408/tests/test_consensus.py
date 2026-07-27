from __future__ import annotations

import unittest

import numpy as np

from experiments.exp403_408.contracts import ALPHAS, ShadowSourcePrediction
from experiments.exp403_408.foundation import load_frozen_lanes
from experiments.exp403_408.consensus import (
    VARIANTS,
    ConsensusSource,
    build_consensus,
    build_independence_components,
    expected_source_marker,
)


def source(
    name: str,
    lineage: str,
    kind: str,
    oof: object,
    *,
    dataset: str = "toy",
    candidates: tuple[int, ...] = (),
    marker: str | None = None,
) -> ConsensusSource:
    scores = {str(alpha): 0.001 for alpha in ALPHAS}
    predictions = {
        str(alpha): tuple(
            ShadowSourcePrediction(
                dataset=dataset,
                candidate_index=index,
                scores_by_alpha=scores,
                source=(
                    expected_source_marker(kind, name)
                    if marker is None
                    else marker
                ),
            )
            for index in candidates
        )
        for alpha in ALPHAS
    }
    return ConsensusSource(
        name=name,
        lineage=lineage,
        kind=kind,
        train_oof_scores=oof,
        shadow_predictions=predictions,
    )


def two_rocket_capacities() -> tuple[ConsensusSource, ...]:
    return (
        source("rocket256", "temporal_random_kernel", "existing", np.arange(20.0)),
        source(
            "rocket512",
            "temporal_random_kernel",
            "existing",
            scores_below_correlation_boundary(),
        ),
    )


def correlated_sources(correlation: float) -> tuple[ConsensusSource, ...]:
    scores = (
        scores_at_or_above_correlation_boundary()
        if float(correlation) >= 0.90
        else scores_below_correlation_boundary()
    )
    return (
        source("normal_density", "normal_space_distance", "existing", np.arange(20.0)),
        source("change_point_combined", "change_delta", "existing", scores),
    )


def sources_with_19_rows() -> tuple[ConsensusSource, ...]:
    return (
        source("tri_view", "tri_view", "tri_view", np.arange(19.0)),
        source("synthetic", "synthetic", "synthetic", np.arange(19.0) ** 2),
    )


class ConsensusTests(unittest.TestCase):
    def test_same_lineage_collapses_even_with_low_observed_correlation(self) -> None:
        sources = two_rocket_capacities()
        components = build_independence_components(sources)

        self.assertLess(observed_spearman(sources), 0.90)
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0].source_names, ("rocket256", "rocket512"))

    def test_distinct_sources_at_or_above_point_90_collapse(self) -> None:
        sources = correlated_sources(0.90)
        components = build_independence_components(sources)

        self.assertGreaterEqual(observed_spearman(sources), 0.90)
        self.assertEqual(len(components), 1)
        self.assertEqual(
            components[0].source_names,
            ("change_point_combined", "normal_density"),
        )

    def test_distinct_sources_below_point_90_remain_independent(self) -> None:
        sources = correlated_sources(0.899)
        components = build_independence_components(sources)

        self.assertLess(observed_spearman(sources), 0.90)
        self.assertEqual(len(components), 2)
        self.assertTrue(all(component.status == "independent" for component in components))

    def test_unverified_independence_does_not_add_vote(self) -> None:
        components = build_independence_components(sources_with_19_rows())

        self.assertEqual(components[1].status, "independence_unverified")

    def test_rejects_mismatched_frozen_shadow_source_marker(self) -> None:
        relabeled = source(
            "tri_view",
            "tri_view",
            "tri_view",
            np.arange(20.0),
            candidates=(10_002,),
            marker="synthetic_anomaly_shadow",
        )

        with self.assertRaisesRegex(ValueError, "source marker"):
            build_independence_components((relabeled,))

    def test_rejects_unknown_kind_name_marker_registration(self) -> None:
        relabeled = source(
            "synthetic",
            "synthetic",
            "tri_view",
            np.arange(20.0),
            marker="self_supervised_shadow",
        )

        with self.assertRaisesRegex(ValueError, "registered"):
            build_independence_components((relabeled,))

    def test_consensus_never_promotes_to_hard_and_preserves_review_lanes(self) -> None:
        universe = load_frozen_lanes()
        dataset = next(iter(universe.hard_lanes()))
        lane = universe.lane_for(dataset)
        sources = independent_sources(dataset)

        result = build_consensus(universe, sources)

        self.assertEqual(result.hard_indices, lane["hard"])
        self.assertEqual(result.standard_indices, lane["standard"])
        self.assertEqual(result.priority_indices, lane["priority"])
        self.assertEqual(tuple(result.candidates_by_variant), VARIANTS)
        self.assertTrue(
            all(
                candidate.candidate_index not in result.hard_indices
                for candidates in result.candidates_by_variant.values()
                for candidates_at_alpha in candidates.values()
                for candidate in candidates_at_alpha
            )
        )

    def test_five_frozen_variants_use_only_verified_independent_votes(self) -> None:
        universe = load_frozen_lanes()
        dataset = next(iter(universe.hard_lanes()))

        result = build_consensus(universe, independent_sources(dataset))

        expected = {
            "tri_view_only": (10_001, 10_002),
            "synthetic_only": (10_002, 10_003),
            "tri_view_and_synthetic": (10_002,),
            "one_new_plus_existing_independent": (10_002,),
            "two_new_plus_existing_independent": (10_002,),
        }
        for variant, candidate_indices in expected.items():
            self.assertEqual(
                tuple(candidate.candidate_index for candidate in result.candidates_by_variant[variant]["0.05"]),
                candidate_indices,
            )

        one_new_candidate = result.candidates_by_variant[
            "one_new_plus_existing_independent"
        ]["0.05"][0]
        self.assertEqual(
            one_new_candidate.supporting_components,
            (("normal_density",), ("synthetic",), ("tri_view",)),
        )
        self.assertEqual(
            one_new_candidate.supporting_source_markers,
            (
                ("normal_density",),
                ("synthetic_anomaly_shadow",),
                ("self_supervised_shadow",),
            ),
        )

        unverified = build_consensus(universe, sources_with_19_rows_for(dataset))
        self.assertEqual(
            unverified.candidates_by_variant["tri_view_and_synthetic"]["0.05"], ()
        )


def independent_sources(dataset: str) -> tuple[ConsensusSource, ...]:
    return (
        source("tri_view", "tri_view", "tri_view", np.arange(20.0), dataset=dataset, candidates=(10_001, 10_002)),
        source("synthetic", "synthetic", "synthetic", np.roll(np.arange(20.0), 5), dataset=dataset, candidates=(10_002, 10_003)),
        source("normal_density", "normal_space_distance", "existing", np.asarray([value * 3 % 20 for value in range(20)], dtype=float), dataset=dataset, candidates=(10_002, 10_004)),
    )


def sources_with_19_rows_for(dataset: str) -> tuple[ConsensusSource, ...]:
    return tuple(
        source(
            evidence.name,
            evidence.lineage,
            evidence.kind,
            evidence.train_oof_scores,
            dataset=dataset,
            candidates=(10_002,),
        )
        for evidence in sources_with_19_rows()
    )


def scores_at_or_above_correlation_boundary() -> np.ndarray:
    values = np.arange(20.0)
    values[[0, 8]] = values[[8, 0]]
    values[[1, 2]] = values[[2, 1]]
    values[[3, 4]] = values[[4, 3]]
    return values


def scores_below_correlation_boundary() -> np.ndarray:
    values = scores_at_or_above_correlation_boundary()
    values[[5, 6]] = values[[6, 5]]
    return values


def observed_spearman(sources: tuple[ConsensusSource, ...]) -> float:
    first = np.asarray(sources[0].train_oof_scores, dtype=float)
    second = np.asarray(sources[1].train_oof_scores, dtype=float)
    return float(np.corrcoef(first, second)[0, 1])


if __name__ == "__main__":
    unittest.main()
