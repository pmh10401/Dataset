from __future__ import annotations

import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.exp381_385.coverage_audit_core import (
    REQUIRED_ARRAYS,
    alpha_resolution,
    audit_exp162_cache,
    train_size_bin,
)


class CoverageContractTests(unittest.TestCase):
    def test_alpha_resolution_uses_finite_sample_minimum(self) -> None:
        self.assertEqual(alpha_resolution(4, 0.05), "abstained_n_train_lt5")
        self.assertEqual(alpha_resolution(19, 0.05), "attainable")
        self.assertEqual(alpha_resolution(19, 0.02), "resolution_blocked")
        self.assertEqual(alpha_resolution(199, 0.005), "attainable")

    def test_train_size_bins_are_stable(self) -> None:
        expected = {
            1: "1-4",
            4: "1-4",
            5: "5-19",
            19: "5-19",
            20: "20-49",
            49: "20-49",
            50: "50-99",
            99: "50-99",
            100: "100-199",
            199: "100-199",
            200: "ge200",
        }
        for n_train, expected_bin in expected.items():
            with self.subTest(n_train=n_train):
                self.assertEqual(train_size_bin(n_train), expected_bin)


class CacheAuditTests(unittest.TestCase):
    def _write_cache(
        self,
        root: Path,
        *,
        missing_array: str | None = None,
        nonfinite_array: str | None = None,
        wrong_train_length: bool = False,
    ) -> tuple[Path, str]:
        path = root / "dataset_a_cache.npz"
        arrays: dict[str, np.ndarray] = {}
        for name in REQUIRED_ARRAYS:
            length = 3 if name.endswith("_train") else 2
            if wrong_train_length and name == "spectral_train":
                length = 4
            values = np.arange(length, dtype=np.float64)
            if nonfinite_array == name:
                values[0] = np.nan
            if missing_array != name:
                arrays[name] = values
        np.savez(path, **arrays)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return path, digest

    def _write_manifest(
        self,
        root: Path,
        cache_path: Path,
        digest: str,
        *,
        duplicate: bool = False,
    ) -> Path:
        manifest = root / "manifest.csv"
        rows = [
            {
                "dataset": "dataset_a",
                "file_path": str(cache_path),
                "file_hash": digest,
                "status": "completed",
            }
        ]
        if duplicate:
            rows.append(dict(rows[0]))
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return manifest

    def _audit(
        self,
        root: Path,
        *,
        missing_array: str | None = None,
        nonfinite_array: str | None = None,
        wrong_train_length: bool = False,
        wrong_hash: bool = False,
        duplicate: bool = False,
        contracts: dict[str, tuple[int, int]] | None = None,
    ):
        cache_path, digest = self._write_cache(
            root,
            missing_array=missing_array,
            nonfinite_array=nonfinite_array,
            wrong_train_length=wrong_train_length,
        )
        if wrong_hash:
            digest = "0" * 64
        manifest = self._write_manifest(root, cache_path, digest, duplicate=duplicate)
        return audit_exp162_cache(
            manifest,
            contracts or {"dataset_a": (3, 2)},
            expected_dataset_count=1,
            repository_root=root,
        )

    def test_valid_cache_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._audit(Path(tmp))
        self.assertTrue(result.complete)
        self.assertEqual(result.valid_count, 1)

    def test_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._audit(Path(tmp), wrong_hash=True)
        self.assertFalse(result.complete)
        self.assertIn("file_hash_mismatch", result.rows[0].errors)

    def test_missing_array_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._audit(Path(tmp), missing_array="interval_test")
        self.assertFalse(result.complete)
        self.assertIn("missing_array:interval_test", result.rows[0].errors)

    def test_nonfinite_array_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._audit(Path(tmp), nonfinite_array="shapelet_train")
        self.assertFalse(result.complete)
        self.assertIn("nonfinite_array:shapelet_train", result.rows[0].errors)

    def test_length_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self._audit(Path(tmp), wrong_train_length=True)
        self.assertFalse(result.complete)
        self.assertIn("length_mismatch:spectral_train", result.rows[0].errors)

    def test_duplicate_and_missing_dataset_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self._audit(
                root,
                duplicate=True,
                contracts={"dataset_a": (3, 2), "dataset_b": (3, 2)},
            )
        self.assertFalse(result.complete)
        self.assertEqual(result.duplicate_datasets, ("dataset_a",))
        self.assertEqual(result.missing_datasets, ("dataset_b",))


if __name__ == "__main__":
    unittest.main()
