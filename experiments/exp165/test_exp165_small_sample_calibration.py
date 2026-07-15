from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from experiments.exp165.similarity_calibration_core import (
    agreement_flags,
    build_descriptor,
    leave_one_out_location_scale,
    select_donors,
    similarity_pooled_flags,
)
from experiments.exp165.run_experiment_165_small_sample_calibration import (
    EXPERIMENT_ID,
    valid_existing_target_cache,
)


class Exp165CoreTests(unittest.TestCase):
    def test_leave_one_out_location_scale_excludes_query(self):
        values = np.asarray([1.0, 2.0, 100.0])

        location, scale = leave_one_out_location_scale(values, query_index=2)

        self.assertEqual(location, 1.5)
        self.assertGreater(scale, 0.0)
        self.assertLess((values[2] - location) / scale, 200.0)

    def test_descriptor_has_seven_statistics_per_source(self):
        scores = {
            "spectral": np.asarray([0.0, 1.0, 2.0, 3.0]),
            "shapelet": np.asarray([2.0, 2.5, 3.0, 4.0]),
            "interval": np.asarray([10.0, 20.0, 30.0, 40.0]),
        }

        descriptor = build_descriptor(scores)

        self.assertEqual(descriptor.shape, (21,))
        self.assertTrue(np.isfinite(descriptor).all())

    def test_donor_selection_is_deterministic_and_keeps_boundary_ties(self):
        target = np.asarray([0.0, 0.0])
        donor_descriptors = {
            "a": np.asarray([1.0, 0.0]),
            "b": np.asarray([-1.0, 0.0]),
            "c": np.asarray([2.0, 0.0]),
        }
        donor_counts = {
            "a": {"spectral": 60, "shapelet": 60, "interval": 60},
            "b": {"spectral": 60, "shapelet": 60, "interval": 60},
            "c": {"spectral": 100, "shapelet": 100, "interval": 100},
        }

        selected = select_donors(target, donor_descriptors, donor_counts, minimum_reference=100)

        self.assertEqual(selected, ["a", "b"])

    def test_donor_selection_rejects_insufficient_reference(self):
        with self.assertRaisesRegex(ValueError, "minimum pooled reference"):
            select_donors(
                np.asarray([0.0]),
                {"a": np.asarray([1.0])},
                {"a": {"spectral": 50, "shapelet": 50, "interval": 50}},
                minimum_reference=200,
            )

    def test_similarity_flags_use_strict_threshold_and_all_alphas(self):
        query_scores = {
            "spectral": 3.0,
            "shapelet": 2.0,
            "interval": 1.0,
        }
        references = {
            "spectral": np.asarray([0.0, 1.0, 2.0]),
            "shapelet": np.asarray([0.0, 1.0, 2.0]),
            "interval": np.asarray([0.0, 1.0, 2.0]),
        }

        thresholds, flags = similarity_pooled_flags(query_scores, references, (0.005, 0.05))

        self.assertTrue(flags[0.005]["spectral"])
        self.assertFalse(flags[0.005]["shapelet"])
        self.assertFalse(flags[0.005]["interval"])
        self.assertEqual(thresholds[0.005]["shapelet"], 2.0)

    def test_agreement_flags_are_nested(self):
        flags = {"spectral": True, "shapelet": True, "interval": False}

        self.assertTrue(agreement_flags(flags, required_sources=2))
        self.assertFalse(agreement_flags(flags, required_sources=3))

    def test_valid_existing_cache_recovers_after_parent_interruption(self):
        row = {"dataset": "Demo", "n_train": "8", "file_hash": "source-hash"}
        metadata = {
            "experiment_id": EXPERIMENT_ID,
            "dataset": "Demo",
            "n_train": 8,
            "config_hash": "config-hash",
            "source_cache_sha256": "source-hash",
            "contains_test_values": False,
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Demo.npz"
            np.savez_compressed(path, metadata_json=np.asarray(json.dumps(metadata)))

            self.assertTrue(valid_existing_target_cache(path, row, "config-hash"))
            self.assertFalse(valid_existing_target_cache(path, row, "other-config"))


if __name__ == "__main__":
    unittest.main()
