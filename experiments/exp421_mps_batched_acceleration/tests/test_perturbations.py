"""TRAIN-only immutable perturbation-cache tests for Exp421."""

from __future__ import annotations

from dataclasses import fields
import unittest

import numpy as np

from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG as EXP417_CONFIG
from experiments.exp417_causal_multihorizon_shadow.qualification import (
    build_fixed_perturbations,
)
from experiments.exp421_mps_batched_acceleration.perturbations import (
    CachedPerturbations,
    PerturbationCacheKey,
    build_cached_perturbations,
    clear_perturbation_cache,
)


def _series() -> np.ndarray:
    return np.linspace(-2.0, 3.0, 128, dtype=np.float64)


class PerturbationCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_perturbation_cache()

    def tearDown(self) -> None:
        clear_perturbation_cache()

    def test_cached_arrays_match_the_frozen_builder_for_every_seed(self) -> None:
        """Catches any changed Exp417 perturbation definition or float32 conversion."""
        series = _series()
        for seed in EXP417_CONFIG.perturbation_seeds:
            with self.subTest(seed=seed):
                expected = build_fixed_perturbations(series, 1.75, seed=seed)
                actual = build_cached_perturbations("TRAIN_fixture", 7, series, 1.75, seed)
                self.assertIsInstance(actual, CachedPerturbations)
                self.assertEqual(tuple(actual), tuple(expected))
                for perturbation in expected:
                    self.assertEqual(actual[perturbation].dtype, np.dtype("float64"))
                    self.assertEqual(actual[perturbation].tobytes(), expected[perturbation].tobytes())
                    self.assertFalse(actual[perturbation].flags.writeable)

    def test_primary_arrays_are_seed_independent_and_secondary_spike_is_deterministic(self) -> None:
        """Catches accidental per-seed regeneration of primary arrays or spikes."""
        series = _series()
        cached = tuple(
            build_cached_perturbations("TRAIN_fixture", 7, series, 1.75, seed)
            for seed in EXP417_CONFIG.perturbation_seeds
        )
        for name in ("level_shift", "gradual_drift", "amplitude_scaling", "period_change"):
            self.assertTrue(all(item[name].tobytes() == cached[0][name].tobytes() for item in cached[1:]))
        repeat = build_cached_perturbations("TRAIN_fixture", 7, series, 1.75, EXP417_CONFIG.perturbation_seeds[1])
        self.assertEqual(repeat["secondary_spike"].tobytes(), cached[1]["secondary_spike"].tobytes())
        self.assertGreater(len({item["secondary_spike"].tobytes() for item in cached}), 1)

    def test_cache_reuses_exact_key_and_source_mutation_cannot_change_cached_bytes(self) -> None:
        """Catches source aliasing or a cache key that omits exact series identity fields."""
        series = _series()
        first = build_cached_perturbations("TRAIN_fixture", 7, series, 1.75, 41711)
        original = first["level_shift"].tobytes()
        series[:] = 99.0
        repeated = build_cached_perturbations("TRAIN_fixture", 7, series, 1.75, 41711)
        self.assertIs(first, repeated)
        self.assertEqual(repeated["level_shift"].tobytes(), original)
        missed_dataset = build_cached_perturbations("TRAIN_other", 7, _series(), 1.75, 41711)
        missed_instance = build_cached_perturbations("TRAIN_fixture", 8, _series(), 1.75, 41711)
        missed_length = build_cached_perturbations("TRAIN_fixture", 7, _series()[:-1], 1.75, 41711)
        self.assertIsNot(missed_dataset, first)
        self.assertIsNot(missed_instance, first)
        self.assertIsNot(missed_length, first)

    def test_keys_digest_and_fields_exclude_prohibited_evaluation_names(self) -> None:
        """Catches TEST, label, location, or metric material in cache keys or evidence."""
        cached = build_cached_perturbations("TRAIN_fixture", 7, _series(), 1.75, 41711)
        self.assertEqual(len(cached.cache_keys), 5)
        self.assertEqual({key.perturbation for key in cached.cache_keys}, set(cached))
        self.assertTrue(all(key.dataset == "TRAIN_fixture" for key in cached.cache_keys))
        self.assertEqual(len(cached.digest), 64)
        self.assertTrue(all(character in "0123456789abcdef" for character in cached.digest))
        prohibited = ("label", "test", "location", "metric")
        for record_type in (CachedPerturbations, PerturbationCacheKey):
            self.assertFalse(any(token in item.name.lower() for item in fields(record_type) for token in prohibited))
        for invalid_dataset in ("TEST_fixture", "train_label", "train_location", "train_metric"):
            with self.subTest(dataset=invalid_dataset):
                with self.assertRaisesRegex(ValueError, "TRAIN"):
                    build_cached_perturbations(invalid_dataset, 7, _series(), 1.75, 41711)

    def test_cache_rejects_malformed_keys_and_a_digest_that_does_not_bind_arrays(self) -> None:
        """Catches injectable malformed identity evidence or unbound cache bytes."""
        cached = build_cached_perturbations("TRAIN_fixture", 7, _series(), 1.75, 41711)
        with self.assertRaisesRegex(ValueError, "TRAIN"):
            PerturbationCacheKey("TEST_fixture", 7, 128, "level_shift", 41711)
        with self.assertRaisesRegex(ValueError, "length"):
            PerturbationCacheKey("TRAIN_fixture", 7, 0, "level_shift", 41711)
        with self.assertRaisesRegex(ValueError, "digest"):
            CachedPerturbations(
                arrays=cached.arrays,
                cache_keys=cached.cache_keys,
                digest="0" * 64,
            )


if __name__ == "__main__":
    unittest.main()
