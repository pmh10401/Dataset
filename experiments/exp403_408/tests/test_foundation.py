from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.contracts import DatasetBundle
from experiments.exp403_408.artifact_io import (
    sha256_file,
    verify_sha256,
    write_jsonl_atomic,
    write_npz_atomic,
)
from experiments.exp403_408.foundation import (
    crossfit_plan,
    load_dataset_input,
    load_frozen_lanes,
    validate_frozen_universe,
)


def fake_values_loader(dataset: str, _path: Path) -> DatasetBundle:
    return DatasetBundle(
        dataset=dataset,
        train_normal=(
            np.array([0.0, np.nan, 2.0], dtype=np.float32),
            np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        ),
        test_values=(np.array([100.0] * 20, dtype=np.float32),),
    )


def frozen_rows(*, hard_count: int) -> list[dict[str, object]]:
    return [
        {
            "dataset": "toy",
            "n_test": max(hard_count, 806, 31),
            "automatic_hard_indices": list(range(hard_count)),
            "standard_review_indices": list(range(806)),
            "priority_review_indices": list(range(31)),
        }
    ]


class FoundationTests(unittest.TestCase):
    def test_dataset_loader_preserves_preprocessing_order(self) -> None:
        bundle = load_dataset_input("toy", loader=fake_values_loader)

        self.assertEqual(
            bundle.preprocessing_order,
            ("missing_repair", "train_length_alignment", "normalization"),
        )
        self.assertEqual(bundle.target_length, 8)
        self.assertTrue(np.isfinite(bundle.train_normal).all())
        self.assertTrue(np.isfinite(bundle.test_values).all())
        np.testing.assert_allclose(bundle.train_normal.mean(axis=1), 0.0, atol=1e-6)

    def test_crossfit_never_scores_query_from_its_fit_fold(self) -> None:
        plan = crossfit_plan(20, seed=20260727)

        seen: set[int] = set()
        for fit, query in plan:
            self.assertFalse(set(fit) & set(query))
            seen.update(query)
        self.assertEqual(seen, set(range(20)))
        self.assertEqual(plan, crossfit_plan(20, seed=20260727))

    def test_dataset_loader_rejects_unverified_lane_injection(self) -> None:
        raw_lanes = {
            "toy": {"hard": frozenset({1}), "standard": frozenset({2}), "priority": frozenset({3})}
        }

        with self.assertRaisesRegex(TypeError, "FrozenLaneUniverse"):
            load_dataset_input(
                "toy", loader=fake_values_loader, frozen_lane_universe=raw_lanes
            )

    def test_dataset_loader_attaches_verified_frozen_lane_universe(self) -> None:
        universe = load_frozen_lanes()
        dataset = "ACSF1_normal_0"

        bundle = load_dataset_input(
            dataset, loader=fake_values_loader, frozen_lane_universe=universe
        )
        expected = universe.lane_for(dataset)

        self.assertEqual(bundle.hard_indices, expected["hard"])
        self.assertEqual(bundle.standard_indices, expected["standard"])
        self.assertEqual(bundle.priority_indices, expected["priority"])

    def test_frozen_lane_totals_must_match(self) -> None:
        with self.assertRaisesRegex(ValueError, "Hard"):
            validate_frozen_universe(frozen_rows(hard_count=2217))

    def test_atomic_jsonl_and_npz_round_trip_with_hash_verification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl_path = root / "predictions.jsonl"
            npz_path = root / "features.npz"

            write_jsonl_atomic(jsonl_path, [{"dataset": "toy", "score": 1.0}])
            write_npz_atomic(npz_path, {"train": np.array([[1.0, 2.0]])})

            self.assertEqual(verify_sha256(jsonl_path, sha256_file(jsonl_path)), sha256_file(jsonl_path))
            self.assertEqual(verify_sha256(npz_path, sha256_file(npz_path)), sha256_file(npz_path))
            with np.load(npz_path) as artifact:
                np.testing.assert_array_equal(artifact["train"], [[1.0, 2.0]])

            with self.assertRaisesRegex(ValueError, "SHA256"):
                verify_sha256(jsonl_path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
