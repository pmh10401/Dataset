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
)


def source(
    name: str,
    lineage: str,
    kind: str,
    oof: object,
    *,
    dataset: str = "toy",
    candidates: tuple[int, ...] = (),
) -> ConsensusSource:
    scores = {str(alpha): 0.001 for alpha in ALPHAS}
    predictions = {
        str(alpha): tuple(
            ShadowSourcePrediction(
                dataset=dataset,
                candidate_index=index,
                scores_by_alpha=scores,
                source=name,
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
        source("rocket_256", "rocket", "existing", np.arange(20.0)),
        source(
            "rocket_512",
            "rocket",
            "existing",
            np.asarray([value * 3 % 20 for value in range(20)], dtype=float),
        ),
    )


def correlated_sources(correlation: float) -> tuple[ConsensusSource, ...]:
    del correlation
    scores = np.arange(20.0)
    return (
        source("distance", "distance", "existing", scores),
        source("spectral", "spectral", "existing", scores),
    )


def sources_with_19_rows() -> tuple[ConsensusSource, ...]:
    return (
        source("tri_view", "tri_view", "tri_view", np.arange(19.0)),
        source("synthetic", "synthetic", "synthetic", np.arange(19.0) ** 2),
    )


class ConsensusTests(unittest.TestCase):
    def test_same_lineage_collapses_even_with_low_observed_correlation(self) -> None:
        components = build_independence_components(two_rocket_capacities())

        self.assertEqual(len(components), 1)
        self.assertEqual(components[0].source_names, ("rocket_256", "rocket_512"))

    def test_high_train_oof_correlation_collapses_distinct_sources(self) -> None:
        components = build_independence_components(correlated_sources(0.95))

        self.assertEqual(len(components), 1)
        self.assertEqual(components[0].source_names, ("distance", "spectral"))

    def test_unverified_independence_does_not_add_vote(self) -> None:
        components = build_independence_components(sources_with_19_rows())

        self.assertEqual(components[1].status, "independence_unverified")

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
            (("distance",), ("synthetic",), ("tri_view",)),
        )

        unverified = build_consensus(universe, sources_with_19_rows_for(dataset))
        self.assertEqual(
            unverified.candidates_by_variant["tri_view_and_synthetic"]["0.05"], ()
        )


def independent_sources(dataset: str) -> tuple[ConsensusSource, ...]:
    return (
        source("tri_view", "tri_view", "tri_view", np.arange(20.0), dataset=dataset, candidates=(10_001, 10_002)),
        source("synthetic", "synthetic", "synthetic", np.roll(np.arange(20.0), 5), dataset=dataset, candidates=(10_002, 10_003)),
        source("distance", "distance", "existing", np.asarray([value * 3 % 20 for value in range(20)], dtype=float), dataset=dataset, candidates=(10_002, 10_004)),
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


if __name__ == "__main__":
    unittest.main()
