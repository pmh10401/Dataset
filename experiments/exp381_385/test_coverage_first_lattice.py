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
from experiments.exp381_385.alternative_score_core import (
    ALTERNATIVE_SOURCES,
    calibrate_local_empirical,
    evaluate_rule,
    generate_rule_specs,
)
from experiments.exp381_385.source_dependence_core import (
    collapse_correlated_sources,
    rankdata_average,
    spearman_correlation,
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


class AlternativeScoreTests(unittest.TestCase):
    def test_local_tail_is_strict_and_resolution_aware(self) -> None:
        blocked = calibrate_local_empirical(
            np.arange(5, dtype=float),
            np.array([100.0]),
            0.05,
        )
        self.assertEqual(blocked.status, "resolution_blocked")
        self.assertEqual(blocked.indices, ())

        result = calibrate_local_empirical(
            np.arange(19, dtype=float),
            np.array([18.0, 19.0]),
            0.05,
        )
        self.assertEqual(result.status, "attainable")
        self.assertEqual(result.indices, (1,))
        self.assertEqual(result.threshold, 18.0)

    def test_local_tail_abstains_below_five(self) -> None:
        result = calibrate_local_empirical(
            np.arange(4, dtype=float),
            np.array([100.0]),
            0.05,
        )
        self.assertEqual(result.status, "calibration_abstained")
        self.assertEqual(result.indices, ())

    def test_three_sources_generate_all_twelve_rules(self) -> None:
        rules = generate_rule_specs(ALTERNATIVE_SOURCES)
        self.assertEqual(len(rules), 12)
        self.assertEqual(len({rule.rule_id for rule in rules}), 12)

    def test_missing_source_is_not_a_negative_vote(self) -> None:
        rule = next(
            rule
            for rule in generate_rule_specs(ALTERNATIVE_SOURCES)
            if len(rule.sources) == 2 and rule.k == 1
        )
        result = evaluate_rule(
            rule,
            source_indices={rule.sources[0]: (3,), rule.sources[1]: ()},
            source_available={rule.sources[0]: True, rule.sources[1]: False},
        )
        self.assertEqual(result.indices, (3,))
        self.assertFalse(result.resolution_blocked)
        self.assertFalse(result.fully_observable)


class SourceDependenceTests(unittest.TestCase):
    def test_rankdata_uses_average_tie_ranks(self) -> None:
        ranks = rankdata_average(np.array([10.0, 20.0, 20.0, 40.0]))
        np.testing.assert_allclose(ranks, [1.0, 2.5, 2.5, 4.0])

    def test_spearman_handles_ties_and_constant_arrays(self) -> None:
        value = spearman_correlation(
            np.array([1.0, 2.0, 2.0, 4.0]),
            np.array([10.0, 20.0, 20.0, 40.0]),
        )
        self.assertAlmostEqual(value or 0.0, 1.0)
        self.assertIsNone(
            spearman_correlation(np.ones(4), np.arange(4, dtype=float))
        )

    def test_family_collapse_is_deterministic_and_transitive(self) -> None:
        sources = ("a", "b", "c", "d")
        correlations = {
            ("a", "b"): 0.90,
            ("b", "c"): 0.91,
            ("a", "c"): 0.20,
            ("c", "d"): 0.10,
        }
        families = collapse_correlated_sources(sources, correlations, 0.85)
        self.assertEqual(families, (("a", "b", "c"), ("d",)))

    def test_missing_pair_does_not_collapse_sources(self) -> None:
        families = collapse_correlated_sources(("a", "b"), {}, 0.70)
        self.assertEqual(families, (("a",), ("b",)))


if __name__ == "__main__":
    unittest.main()
