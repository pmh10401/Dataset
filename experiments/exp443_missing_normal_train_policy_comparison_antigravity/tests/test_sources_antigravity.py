"""Strict RED tests for Exp443's inherited source parser and identity proof."""
from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.exp443_missing_normal_train_policy_comparison_antigravity import sources_antigravity as sources


class Exp443SourcesAntigravityTests(unittest.TestCase):
    def _valid_cache_fixture(self, root: Path, datasets: tuple[str, ...] = ("A",)) -> tuple[Path, list[dict[str, str]]]:
        """Create a schema-valid full 14-column manifest and pinned artifacts."""
        cache = root / "cache"
        cache.mkdir(parents=True)
        rows: list[dict[str, str]] = []
        for dataset in datasets:
            relative = f"cache/{dataset}.npz"
            npz_path = root / relative
            arrays = {
                "kernels_1024__train_by_seed": np.ones((2, 1), dtype="<f8"),
                "kernels_1024__test_by_seed": np.ones((2, 1), dtype="<f8"),
                "kernels_4096__train_by_seed": np.ones((2, 1), dtype="<f8"),
                "kernels_4096__test_by_seed": np.ones((2, 1), dtype="<f8"),
            }
            np.savez(npz_path, **arrays)
            cache_sha256 = hashlib.sha256(npz_path.read_bytes()).hexdigest()
            metadata = {
                "abstained": False, "abstention_reason": None, "cache_file": relative,
                "cache_sha256": cache_sha256, "config": {"fixture": True},
                "config_hash": sources.EXP249_CACHE_CONFIG_HASH_ANTIGRAVITY,
                "contains_test_labels": False, "dataset": dataset, "experiment": 249,
                "feature_dimensions": {"1024": 1, "4096": 1}, "n_test": 1,
                "n_train": 1, "source": "official_minirocket", "target_length": 1,
            }
            metadata_path = npz_path.with_suffix(".json")
            metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
            rows.append({
                "abstained": "False", "abstention_reason": "", "cache_file": relative,
                "cache_sha256": cache_sha256, "config_hash": sources.EXP249_CACHE_CONFIG_HASH_ANTIGRAVITY,
                "contains_test_labels": "False", "dataset": dataset, "experiment": "249",
                "feature_dimensions": '{"1024": 1, "4096": 1}', "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                "n_test": "1", "n_train": "1", "source": "official_minirocket", "target_length": "1",
            })
        manifest = root / "cache_manifest_antigravity.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return manifest, rows

    def _rewrite_manifest(self, manifest: Path, rows: list[dict[str, str]]) -> None:
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_duplicate_json_keys_are_rejected_without_last_value_wins(self) -> None:
        with self.assertRaises(ValueError):
            sources.parse_json_object_antigravity(b'{"dataset":"A","dataset":"B"}', "fixture")

    def test_cache_manifest_reaches_each_independent_integrity_branch(self) -> None:
        mutations = ("duplicate", "missing_npz", "missing_metadata", "extra", "wrong_npz_hash", "wrong_metadata_hash", "wrong_manifest_hash")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, rows = self._valid_cache_fixture(root)
                if mutation == "duplicate":
                    self._rewrite_manifest(manifest, rows + [dict(rows[0])])
                    with self.assertRaises(ValueError):
                        sources.parse_cache_manifest_antigravity(manifest, root, expected_dataset_count=2)
                elif mutation == "missing_npz":
                    (root / rows[0]["cache_file"]).unlink()
                    with self.assertRaises(ValueError):
                        sources.parse_cache_manifest_antigravity(manifest, root, expected_dataset_count=1)
                elif mutation == "missing_metadata":
                    (root / rows[0]["cache_file"]).with_suffix(".json").unlink()
                    with self.assertRaises(ValueError):
                        sources.parse_cache_manifest_antigravity(manifest, root, expected_dataset_count=1)
                elif mutation == "extra":
                    (root / "cache" / "extra.npz").write_bytes((root / rows[0]["cache_file"]).read_bytes())
                    with self.assertRaises(ValueError):
                        sources.parse_cache_manifest_antigravity(manifest, root, expected_dataset_count=1)
                elif mutation == "wrong_npz_hash":
                    changed = dict(rows[0]); changed["cache_sha256"] = "0" * 64
                    self._rewrite_manifest(manifest, [changed])
                    with self.assertRaises(ValueError):
                        sources.parse_cache_manifest_antigravity(manifest, root, expected_dataset_count=1)
                elif mutation == "wrong_metadata_hash":
                    changed = dict(rows[0]); changed["metadata_sha256"] = "0" * 64
                    self._rewrite_manifest(manifest, [changed])
                    with self.assertRaises(ValueError):
                        sources.parse_cache_manifest_antigravity(manifest, root, expected_dataset_count=1)
                else:
                    with self.assertRaises(ValueError):
                        sources.parse_cache_manifest_antigravity(manifest, root, expected_dataset_count=1, expected_sha256="0" * 64)

    def test_source_pre_open_swap_to_fifo_and_symlink_is_fail_closed(self) -> None:
        for replacement_kind in ("fifo", "symlink"):
            with self.subTest(replacement_kind=replacement_kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                source = root / "source_antigravity.json"
                source.write_bytes(b'{"ok":true}')
                expected = hashlib.sha256(source.read_bytes()).hexdigest()
                backup = root / "source_backup_antigravity.json"

                def swap(_path: Path, *, kind: str = replacement_kind) -> None:
                    source.rename(backup)
                    if kind == "fifo":
                        os.mkfifo(source)
                    else:
                        source.symlink_to(backup)

                previous = sources._SOURCE_PRE_OPEN_HOOK_ANTIGRAVITY
                sources._SOURCE_PRE_OPEN_HOOK_ANTIGRAVITY = swap
                try:
                    with self.assertRaises(ValueError):
                        sources._read_regular_bytes_antigravity(source, expected)
                finally:
                    sources._SOURCE_PRE_OPEN_HOOK_ANTIGRAVITY = previous
                    if source.is_symlink() or source.exists():
                        source.unlink()
                    backup.rename(source)

    def test_cache_arrays_and_snapshot_are_deeply_immutable_and_typed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, rows = self._valid_cache_fixture(root)
            parsed = sources.parse_cache_manifest_antigravity(manifest, root, expected_dataset_count=1)
            arrays = parsed["A"]["arrays"]
            self.assertFalse(arrays["kernels_1024__train_by_seed"].flags.writeable)
            with self.assertRaises(ValueError):
                arrays["kernels_1024__train_by_seed"].setflags(write=True)

        snapshot = sources.verify_frozen_sources_antigravity()
        dataset = snapshot.datasets[0]
        array_name = "kernels_1024__train_by_seed"
        with self.assertRaises(TypeError):
            snapshot.cache_inventory[dataset]["arrays"][array_name] = np.zeros((2, 1))
        with self.assertRaises(ValueError):
            snapshot.cache_inventory[dataset]["arrays"][array_name].setflags(write=True)
        with self.assertRaises(ValueError):
            replace(snapshot, datasets=("forged",))
        with self.assertRaises(ValueError):
            replace(snapshot, affected_rows=True)
        for forged in (np.zeros((1, 1), dtype="<f8"), np.zeros((2, 1), dtype="<f4"), np.full_like(snapshot.cache_inventory[dataset]["arrays"][array_name], np.nan)):
            cache = dict(snapshot.cache_inventory)
            entry = dict(cache[dataset])
            nested = dict(entry["arrays"])
            nested[array_name] = forged
            entry["arrays"] = nested
            cache[dataset] = entry
            with self.assertRaises(ValueError):
                replace(snapshot, cache_inventory=cache)

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
