from __future__ import annotations

import inspect
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp403_408.consensus import ConsensusCandidate, ConsensusSource
from experiments.exp403_408.contracts import DatasetInput
from experiments.exp403_408.feature_registry import FEATURE_SOURCES
from experiments.exp403_408.synthetic_anomaly import MODEL_NAMES
from experiments.exp409_prospective_readiness.config import FROZEN_CONFIG
from experiments.exp409_prospective_readiness.contracts import SourceStatus
from experiments.exp409_prospective_readiness.runner import (
    _assert_deterministic,
    run_dataset,
    select_exp409_smoke_datasets,
    validate_lane_parity,
)
from experiments.exp409_prospective_readiness.sources import SourceBatch


_DIGEST = "a" * 64


def _names() -> tuple[str, ...]:
    return tuple(
        sorted(
            {source.name for source in FEATURE_SOURCES}
            | {"tri_view"}
            | {f"synthetic_{name}" for name in MODEL_NAMES}
        )
    )


def _bundle() -> DatasetInput:
    return DatasetInput(
        dataset="sample",
        train_normal=np.arange(200, dtype=float).reshape(50, 4),
        test_values=np.arange(20, dtype=float).reshape(5, 4),
        target_length=4,
        hard_indices=frozenset({4}),
        standard_indices=frozenset({2}),
        priority_indices=frozenset({3}),
    )


def _source_batch(*, abstained: bool = False) -> SourceBatch:
    status = "abstained" if abstained else "available"
    reason = "source_abstained" if abstained else ""
    return SourceBatch(
        sources=(
            ()
            if abstained
            else (
                ConsensusSource(
                    name="rocket256",
                    lineage="temporal_random_kernel",
                    kind="existing",
                    train_oof_scores=tuple(float(value) for value in range(50)),
                    dataset="sample",
                ),
            )
        ),
        statuses=tuple(SourceStatus(name, status, reason) for name in _names()),
        output_sha256={name: _DIGEST for name in _names()},
    )


class RunnerTests(unittest.TestCase):
    @patch("experiments.exp409_prospective_readiness.runner.build_consensus")
    @patch("experiments.exp409_prospective_readiness.runner.build_source_batch")
    @patch("experiments.exp409_prospective_readiness.runner.load_dataset_input")
    @patch("experiments.exp409_prospective_readiness.runner.load_frozen_lanes")
    def test_run_dataset_exposes_only_frozen_variant_and_alpha(
        self, lanes, loader, source_builder, consensus_builder
    ) -> None:
        loader.return_value = _bundle()
        lanes.return_value = object()
        source_builder.return_value = _source_batch()
        row = ConsensusCandidate(
            dataset="sample",
            candidate_index=1,
            variant=FROZEN_CONFIG.variant,
            alpha=FROZEN_CONFIG.alpha,
            supporting_components=(("a",), ("b",), ("c",)),
            supporting_source_markers=(("a",), ("b",), ("c",)),
        )
        consensus_builder.return_value = SimpleNamespace(
            hard_indices=frozenset({4}),
            standard_indices=frozenset({2}),
            priority_indices=frozenset({3}),
            candidates_by_variant={
                FROZEN_CONFIG.variant: {FROZEN_CONFIG.alpha: (row,)}
            },
        )

        result = run_dataset("sample")

        self.assertTrue(
            all(
                candidate.variant == FROZEN_CONFIG.variant
                and candidate.alpha == FROZEN_CONFIG.alpha
                for candidate in result.candidates
            )
        )
        self.assertEqual(result.hard_indices, (4,))
        self.assertEqual(result.standard_indices, (2,))
        self.assertEqual(result.priority_indices, (3,))

    @patch("experiments.exp409_prospective_readiness.runner.build_consensus")
    @patch("experiments.exp409_prospective_readiness.runner.build_source_batch")
    @patch("experiments.exp409_prospective_readiness.runner.load_dataset_input")
    @patch("experiments.exp409_prospective_readiness.runner.load_frozen_lanes")
    def test_every_source_abstention_skips_consensus(
        self, lanes, loader, source_builder, consensus_builder
    ) -> None:
        loader.return_value = _bundle()
        lanes.return_value = object()
        source_builder.return_value = _source_batch(abstained=True)

        result = run_dataset("sample")

        self.assertEqual(result.status, "abstained")
        self.assertEqual(result.candidates, ())
        consensus_builder.assert_not_called()

    def test_lane_mutation_is_rejected(self) -> None:
        original = {
            "hard": (1,),
            "standard": (2,),
            "priority": (3,),
        }
        changed = {
            "hard": (1, 2),
            "standard": (),
            "priority": (3,),
        }
        with self.assertRaisesRegex(ValueError, "lane parity"):
            validate_lane_parity(original, changed)

    @patch(
        "experiments.exp409_prospective_readiness.runner._train_normal_counts",
        return_value={"a": 4, "b": 10, "c": 25, "d": 60, "e": 80},
    )
    @patch(
        "experiments.exp409_prospective_readiness.runner.list_datasets_by_train_normal_count",
        return_value=["a", "b", "c", "d", "e"],
    )
    def test_smoke_covers_all_train_tiers(self, datasets, counts) -> None:
        selected, coverage = select_exp409_smoke_datasets(limit=4)

        self.assertEqual(set(coverage), {
            "abstain",
            "robust_distance",
            "classical_committee",
            "learned_methods",
        })
        self.assertTrue(all(count >= 1 for count in coverage.values()))
        self.assertEqual(len(selected), 4)

    def test_selector_does_not_accept_test_order_or_labels(self) -> None:
        signature = inspect.signature(select_exp409_smoke_datasets)
        self.assertNotIn("test_labels", signature.parameters)
        self.assertNotIn("test_order", signature.parameters)

    def test_determinism_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "determinism gate"):
            _assert_deterministic({"candidate": "a"}, {"candidate": "b"})


if __name__ == "__main__":
    unittest.main()
