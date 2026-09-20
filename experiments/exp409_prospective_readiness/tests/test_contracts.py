from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from experiments.exp409_prospective_readiness.config import (
    Exp409FrozenConfig,
    FROZEN_CONFIG,
)
from experiments.exp409_prospective_readiness.contracts import (
    DatasetReadinessResult,
    ReadinessBatch,
    ShadowCandidate,
    SourceStatus,
    validate_batch,
    validate_candidate,
)


def candidate(index: int, *, dataset: str = "sample") -> ShadowCandidate:
    return ShadowCandidate(
        dataset=dataset,
        candidate_index=index,
        variant=FROZEN_CONFIG.variant,
        alpha=FROZEN_CONFIG.alpha,
        supporting_components=(
            ("tri_view",),
            ("synthetic_a",),
            ("existing_a",),
        ),
        supporting_source_markers=(
            ("tri_view",),
            ("synthetic_a",),
            ("existing_a",),
        ),
        disposition="review_only",
    )


def statuses() -> tuple[SourceStatus, ...]:
    return tuple(
        SourceStatus(source=f"source-{index:02d}", status="available", reason="")
        for index in range(13)
    )


def batch_with_candidates(rows: tuple[ShadowCandidate, ...]) -> ReadinessBatch:
    datasets: dict[str, list[ShadowCandidate]] = {}
    for row in rows:
        datasets.setdefault(row.dataset, []).append(row)
    results = tuple(
        DatasetReadinessResult(
            dataset=dataset,
            train_size=50,
            test_size=200,
            hard_indices=(),
            standard_indices=(),
            priority_indices=(),
            source_statuses=statuses(),
            candidates=tuple(candidates),
            source_output_sha256={
                status.source: "a" * 64 for status in statuses()
            },
            status="ready",
        )
        for dataset, candidates in sorted(datasets.items())
    )
    return ReadinessBatch(dataset_results=results)


class FrozenConfigTests(unittest.TestCase):
    def test_frozen_values_are_exact(self) -> None:
        self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp409")
        self.assertEqual(
            FROZEN_CONFIG.variant, "two_new_plus_existing_independent"
        )
        self.assertEqual(FROZEN_CONFIG.alpha, "0.005")
        self.assertEqual(FROZEN_CONFIG.review_budget, 100)
        self.assertEqual(FROZEN_CONFIG.performance_evaluation, "not_evaluated")
        self.assertFalse(FROZEN_CONFIG.deployment_eligible)

    def test_config_is_immutable(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            FROZEN_CONFIG.alpha = "0.01"  # type: ignore[misc]

    def test_alternate_research_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen"):
            Exp409FrozenConfig(alpha="0.01")


class CandidateContractTests(unittest.TestCase):
    def test_hard_overlap_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Hard"):
            validate_candidate(candidate(3), n_test=10, hard_indices={3})

    def test_candidate_outside_test_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "range"):
            validate_candidate(candidate(10), n_test=10, hard_indices=set())

    def test_duplicate_batch_candidate_is_rejected(self) -> None:
        batch = batch_with_candidates((candidate(2), candidate(2)))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_batch(batch, review_budget=100)

    def test_budget_100_passes(self) -> None:
        rows = tuple(candidate(index) for index in range(100))
        validate_batch(batch_with_candidates(rows), review_budget=100)

    def test_budget_101_is_rejected(self) -> None:
        rows = tuple(candidate(index) for index in range(101))
        batch = batch_with_candidates(rows)
        with self.assertRaisesRegex(ValueError, "budget"):
            validate_batch(batch, review_budget=100)


if __name__ == "__main__":
    unittest.main()
