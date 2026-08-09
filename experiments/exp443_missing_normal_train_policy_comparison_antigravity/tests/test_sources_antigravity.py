"""Strict RED tests for Exp443's inherited source parser and identity proof."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.exp443_missing_normal_train_policy_comparison_antigravity import sources_antigravity as sources


class Exp443SourcesAntigravityTests(unittest.TestCase):
    def test_duplicate_json_keys_are_rejected_without_last_value_wins(self) -> None:
        with self.assertRaises(ValueError):
            sources.parse_json_object_antigravity(b'{"dataset":"A","dataset":"B"}', "fixture")

    def test_duplicate_csv_dataset_and_missing_or_extra_cache_artifacts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "cache_manifest_antigravity.csv"
            manifest.write_text("dataset,cache_file,metadata_sha256,cache_sha256\nA,cache/a.npz," + "a" * 64 + "," + "b" * 64 + "\nA,cache/b.npz," + "c" * 64 + "," + "d" * 64 + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                sources.parse_cache_manifest_antigravity(manifest, root)
            manifest.write_text("dataset,cache_file,metadata_sha256,cache_sha256\nA,cache/a.npz," + "a" * 64 + "," + "b" * 64 + "\n", encoding="utf-8")
            (root / "cache").mkdir()
            with self.assertRaises(ValueError):
                sources.parse_cache_manifest_antigravity(manifest, root)
            np.savez(root / "cache" / "a.npz", values=np.asarray([1.0], dtype="<f4"))
            (root / "cache" / "extra.npz").write_bytes((root / "cache" / "a.npz").read_bytes())
            with self.assertRaises(ValueError):
                sources.parse_cache_manifest_antigravity(manifest, root)

    def test_malformed_and_out_of_range_coordinates_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sources.validate_coordinate_antigravity("A", -1, 3)
        with self.assertRaises(ValueError):
            sources.validate_coordinate_antigravity("A", 3, 3)
        with self.assertRaises(ValueError):
            sources.validate_coordinate_antigravity("", 0, 3)
        with self.assertRaises(ValueError):
            sources.validate_coordinate_antigravity("A", True, 3)

    def test_npz_loader_disallows_pickle_and_requires_expected_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache_antigravity.npz"
            np.savez(path, forbidden=np.asarray([object()], dtype=object))
            with self.assertRaises(ValueError):
                sources.load_pinned_npz_antigravity(path, path.read_bytes())

    def test_real_frozen_sources_reproduce_population_lanes_and_zero_labels(self) -> None:
        snapshot = sources.verify_frozen_sources_antigravity()
        self.assertEqual(len(snapshot.datasets), 1117)
        self.assertEqual(snapshot.datasets, tuple(sorted(snapshot.datasets)))
        self.assertEqual(len(snapshot.affected_datasets), 19)
        self.assertEqual(snapshot.affected_rows, 164)
        self.assertEqual(snapshot.nonfinite_scalars, 983)
        self.assertEqual(snapshot.all_missing_rows, 0)
        self.assertEqual(snapshot.automatic_hard_count, 2218)
        self.assertEqual(snapshot.exact_normal_count, 35)
        self.assertEqual(snapshot.standard_count, 806)
        self.assertEqual(snapshot.priority_count, 31)
        self.assertEqual(snapshot.label_access_count, 0)
        self.assertEqual(snapshot.hard_coordinates_exp291, snapshot.hard_coordinates_exp404)
        self.assertEqual(snapshot.routing_algebra, (2116, 137, 35, 2218))


if __name__ == "__main__":
    unittest.main()
