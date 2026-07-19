from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.baseline_exp137 import (
    load_exp137_baseline,
    verify_exp137_exact_replay,
)
from experiments.exp226_235_hard_alert_quality.contracts import ScoreBundle
from experiments.exp226_235_hard_alert_quality.score_cache import (
    FrozenScoreRepository,
    load_verified_score_bundle,
    write_score_bundle,
)


class ImmutableScoreCacheTests(unittest.TestCase):
    def _bundle(self) -> ScoreBundle:
        return ScoreBundle(
            dataset="demo",
            train_oof={"temporal_random_kernel": np.asarray([0.1, 0.2])},
            test_scores={"temporal_random_kernel": np.asarray([0.3])},
            provenance={"seed": 7, "config": "frozen"},
        )

    def test_cache_rejects_modified_npz(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "demo.npz"
            manifest = root / "demo.json"
            write_score_bundle(cache, manifest, self._bundle())
            cache.write_bytes(cache.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_verified_score_bundle(cache, manifest)

    def test_cache_round_trip_preserves_vectors_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "demo.npz"
            manifest = root / "demo.json"
            write_score_bundle(cache, manifest, self._bundle())
            actual = load_verified_score_bundle(cache, manifest)
            self.assertEqual(actual.dataset, "demo")
            np.testing.assert_allclose(
                actual.train_oof["temporal_random_kernel"], [0.1, 0.2]
            )
            self.assertEqual(actual.provenance["config"], "frozen")


class Exp137BaselineTests(unittest.TestCase):
    def test_exp137_reference_totals_are_exact(self) -> None:
        report = verify_exp137_exact_replay(load_exp137_baseline())
        self.assertEqual(report.datasets, 1117)
        self.assertEqual(report.hard_alerts, 2005)
        self.assertEqual(report.hard_tp, 1691)
        self.assertEqual(report.hard_fp, 314)
        self.assertEqual(report.lane_overlap_datasets, 0)
        self.assertEqual(report.index_error_datasets, 0)


class FrozenSourceIntegrationTests(unittest.TestCase):
    def test_full_source_cache_loads_operational_names(self) -> None:
        repository = FrozenScoreRepository.from_frozen_manifests()
        bundle = repository.load("ACSF1_normal_0")
        self.assertIn("temporal_random_kernel", bundle.train_oof)
        self.assertIn("frequency_image", bundle.train_oof)
        self.assertIn("texture_shape", bundle.train_oof)
        self.assertEqual(len(bundle.train_oof["temporal_random_kernel"]), 16)
        self.assertEqual(len(bundle.test_scores["temporal_random_kernel"]), 50)

    def test_canonical_availability_is_explicit(self) -> None:
        repository = FrozenScoreRepository.from_frozen_manifests()
        available = repository.load("Adiac_normal_12")
        unavailable = repository.load("ACSF1_normal_0")
        self.assertIn("canonical_normal_state", available.train_oof)
        self.assertNotIn("canonical_normal_state", unavailable.train_oof)
        self.assertFalse(unavailable.provenance["canonical_available"])


if __name__ == "__main__":
    unittest.main()
