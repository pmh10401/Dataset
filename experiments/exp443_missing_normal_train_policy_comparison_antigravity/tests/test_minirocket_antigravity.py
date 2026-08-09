"""Behavior tests for the Exp443 MiniROCKET adapter.

The scorer seam is patched only at the unchanged Exp249 call boundary.  This
keeps these tests deterministic and still asserts the exact frozen call
contract that the production adapter must use.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp443_missing_normal_train_policy_comparison_antigravity.contracts_antigravity import RawSeriesAntigravity
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.preprocessing_antigravity import (
    POLICY_A_ANTIGRAVITY,
    prepare_dataset_antigravity,
)


def _raw(values: list[float], index: int, *, split: str = "TRAIN") -> RawSeriesAntigravity:
    array = np.asarray(values, dtype="<f4")
    missing = tuple(int(value) for value in np.flatnonzero(~np.isfinite(array)))
    return RawSeriesAntigravity(
        "Demo",
        split,
        index,
        sha256(array.tobytes(order="C")).hexdigest(),
        array,
        missing,
    )


@dataclass(frozen=True)
class _RawDataset:
    dataset: str
    train: tuple[RawSeriesAntigravity, ...]
    test: tuple[RawSeriesAntigravity, ...]


def _prepared():
    raw = _RawDataset(
        "Demo",
        tuple(_raw([float(index + step) for step in range(8)], index) for index in range(6)),
        (_raw([10.0 + step for step in range(8)], 0, split="TEST"), _raw([20.0 + step for step in range(8)], 1, split="TEST")),
    )
    return prepare_dataset_antigravity(raw, POLICY_A_ANTIGRAVITY)


class MiniRocketAdapterAntigravityTests(unittest.TestCase):
    def test_score_prepared_dataset_makes_exact_four_frozen_calls(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_score(train_rows, test_rows, **kwargs):
            config = kwargs["config"]
            calls.append(
                {
                    "train": np.asarray(train_rows),
                    "test": np.asarray(test_rows),
                    "config": config,
                    **{key: kwargs[key] for key in ("requested_kernels", "seed", "expected_dimension", "max_train_reference", "query_chunk_size")},
                }
            )
            return SimpleNamespace(
                train_oof=np.linspace(0.0, 1.0, len(train_rows), dtype="<f8"),
                test_scores=np.linspace(0.0, 1.0, len(test_rows), dtype="<f8"),
                feature_dimension=kwargs["expected_dimension"],
            )

        with patch(
            "experiments.exp443_missing_normal_train_policy_comparison_antigravity.minirocket_antigravity.score_configuration",
            side_effect=fake_score,
        ):
            from experiments.exp443_missing_normal_train_policy_comparison_antigravity.minirocket_antigravity import (
                score_prepared_dataset_antigravity,
            )

            result = score_prepared_dataset_antigravity(_prepared())

        self.assertEqual(len(calls), 4)
        self.assertEqual(
            {(call["requested_kernels"], call["seed"]) for call in calls},
            {(1024, 20260721), (1024, 20261721), (4096, 20260721), (4096, 20261721)},
        )
        for call in calls:
            config = call["config"]
            self.assertEqual(config.experiment, 249)
            self.assertEqual(config.source, "official_minirocket")
            self.assertEqual(config.transformer, "minirocket")
            self.assertEqual(config.requested_kernels, (1024, 4096))
            self.assertEqual(config.max_dilations_per_kernel, 32)
            self.assertEqual(call["expected_dimension"], {1024: 1008, 4096: 4032}[call["requested_kernels"]])
            self.assertEqual(call["max_train_reference"], 1000)
            self.assertEqual(call["query_chunk_size"], 128)
            self.assertEqual(call["train"].shape, (6, 8))
            self.assertEqual(call["test"].shape, (2, 8))
        self.assertEqual(result.kernel_scores.__len__(), 4)
        self.assertEqual(result.candidates_by_kernel, {1024: (), 4096: ()})
        self.assertEqual(result.status, "scored")

    def test_seed_intersection_is_per_kernel_and_uses_exact_alpha(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.contracts_antigravity import KernelSeedScoresAntigravity
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.minirocket_antigravity import (
            seed_intersection_candidates_antigravity,
        )

        reference = np.arange(1000, dtype="<f8")
        first = KernelSeedScoresAntigravity(1024, 20260721, reference, np.asarray([2000.0, 20.0, 3000.0], dtype="<f8"), (0, 2))
        second = KernelSeedScoresAntigravity(1024, 20261721, reference, np.asarray([2000.0, 500.0, 3000.0], dtype="<f8"), (0, 2))
        other_kernel = KernelSeedScoresAntigravity(4096, 20260721, reference, np.asarray([2000.0, 3000.0, 4000.0], dtype="<f8"), (0, 1, 2))
        self.assertEqual(seed_intersection_candidates_antigravity((first, second)), (0, 2))
        with self.assertRaises(ValueError):
            # A changed kernel cannot be combined with a 1024 score.
            seed_intersection_candidates_antigravity((first, other_kernel))
        with self.assertRaises(ValueError):
            # Duplicate/changed seed identity is rejected before routing.
            seed_intersection_candidates_antigravity((first, first))
        with self.assertRaises(ValueError):
            seed_intersection_candidates_antigravity((other_kernel,))
        with self.assertRaises(ValueError):
            seed_intersection_candidates_antigravity((first, second), alpha=0.01)

        with self.assertRaises(ValueError):
            KernelSeedScoresAntigravity(512, 20260721, reference, np.ones(3, dtype="<f8"), ())
        with self.assertRaises(ValueError):
            KernelSeedScoresAntigravity(1024, 1, reference, np.ones(3, dtype="<f8"), ())

    def test_score_rejects_nonfinite_wrong_length_and_wrong_dimension_outputs(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.minirocket_antigravity import score_prepared_dataset_antigravity

        cases = (
            (SimpleNamespace(train_oof=np.ones(6), test_scores=np.asarray([1.0, np.nan]), feature_dimension=1008), "finite"),
            (SimpleNamespace(train_oof=np.ones(5), test_scores=np.ones(2), feature_dimension=1008), "length"),
            (SimpleNamespace(train_oof=np.ones(6), test_scores=np.ones(2), feature_dimension=7), "dimension"),
        )
        for output, message in cases:
            with self.subTest(message=message), patch(
                "experiments.exp443_missing_normal_train_policy_comparison_antigravity.minirocket_antigravity.score_configuration",
                return_value=output,
            ), self.assertRaisesRegex(ValueError, message):
                score_prepared_dataset_antigravity(_prepared())


if __name__ == "__main__":
    unittest.main()
