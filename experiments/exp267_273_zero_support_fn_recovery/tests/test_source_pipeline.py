from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.exp239_248_diverse_score_sources.contracts import DatasetBundle
from experiments.exp267_273_zero_support_fn_recovery.contracts import SourceScore


class _FakeSource:
    name = "fake_scattering"
    minimum_train = 5

    def score(self, bundle: DatasetBundle, seed: int, artifact_dir: Path) -> SourceScore:
        del seed, artifact_dir
        return SourceScore(
            source=self.name,
            variant="primary",
            train_oof=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
            test_scores=np.asarray([4.0, 3.0, 2.0]),
            provenance={"contains_test_labels": False},
        )


class _AbstainingSource:
    name = "fake_scattering"
    minimum_train = 5

    def score(self, bundle: DatasetBundle, seed: int, artifact_dir: Path) -> SourceScore:
        del bundle, seed, artifact_dir
        return SourceScore(
            source=self.name,
            variant="primary",
            train_oof=np.asarray([], dtype=float),
            test_scores=np.asarray([], dtype=float),
            eligible=False,
            abstained=True,
            abstention_reason="n_train_below_5",
            provenance={"contains_test_labels": False},
        )


def _bundle(dataset: str = "d1") -> DatasetBundle:
    return DatasetBundle(
        dataset=dataset,
        train_normal=np.arange(5 * 16, dtype=np.float32).reshape(5, 16),
        test_values=np.arange(3 * 16, dtype=np.float32).reshape(3, 16),
        target_length=16,
    )


class SourceRegistryTests(unittest.TestCase):
    def test_tcn_variants_count_as_one_evidence_family(self) -> None:
        from experiments.exp267_273_zero_support_fn_recovery.source_registry import registration

        self.assertEqual(registration("tcn_seed_median").family, "normal_forecast")
        self.assertEqual(registration("tcn_seed_intersection").family, "normal_forecast")
        self.assertEqual(
            registration("tcn_seed_median").evidence_identity,
            registration("tcn_seed_intersection").evidence_identity,
        )


class SourcePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _build(
        self,
        *,
        reset: bool = False,
        dataset: str = "d1",
        source_factory= None,
        code_hash: str | None = None,
    ) -> dict[str, object]:
        from experiments.exp267_273_zero_support_fn_recovery.source_pipeline import build_cache_unit

        with patch(
            "experiments.exp267_273_zero_support_fn_recovery.source_pipeline.load_bundle",
            return_value=_bundle(dataset),
        ):
            return build_cache_unit(
                self.root,
                "scattering",
                dataset,
                reset=reset,
                source_factory=source_factory or (lambda *_args, **_kwargs: _FakeSource()),
                db_hash=None,
                code_hash=code_hash,
            )

    def test_valid_cache_is_reused_only_when_config_hash_matches(self) -> None:
        first = self._build()
        second = self._build()
        self.assertEqual(first["cache_sha256"], second["cache_sha256"])
        changed = self._build(reset=False, code_hash="changed-code-hash")
        self.assertNotEqual(first["cache_sha256"], changed["cache_sha256"])
        self.assertEqual(changed["code_sha256"], "changed-code-hash")

    def test_stale_or_tampered_cache_rebuilds(self) -> None:
        first = self._build()
        archive = self.root / str(first["cache_file"])
        archive.write_bytes(b"tampered")
        rebuilt = self._build()
        self.assertEqual(first["cache_sha256"], rebuilt["cache_sha256"])
        self.assertNotEqual(archive.read_bytes(), b"tampered")

    def test_freeze_contains_no_evaluation_fields_and_is_canonical(self) -> None:
        from experiments.exp267_273_zero_support_fn_recovery.source_pipeline import (
            freeze_source_predictions,
            verify_prediction_hash,
        )

        self._build()
        prediction_path, manifest_path = freeze_source_predictions(
            self.root,
            sources=("scattering",),
            datasets=("d1",),
            expected_dataset_count=1,
            bundle_loader=_bundle,
        )
        rows = [json.loads(line) for line in prediction_path.read_text().splitlines()]
        self.assertEqual([row["alpha"] for row in rows], [0.005, 0.01, 0.02, 0.05])
        forbidden = {"y_test", "labels", "tp", "fp", "fn", "f1", "precision", "metrics"}
        self.assertFalse(set().union(*(set(row) for row in rows)) & forbidden)
        self.assertTrue(verify_prediction_hash(prediction_path, manifest_path))

    def test_prediction_hash_rejects_tampering(self) -> None:
        from experiments.exp267_273_zero_support_fn_recovery.source_pipeline import (
            freeze_source_predictions,
            verify_prediction_hash,
        )

        self._build()
        prediction_path, manifest_path = freeze_source_predictions(
            self.root,
            sources=("scattering",),
            datasets=("d1",),
            expected_dataset_count=1,
            bundle_loader=_bundle,
        )
        prediction_path.write_text(prediction_path.read_text() + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            verify_prediction_hash(prediction_path, manifest_path)

    def test_incomplete_atomic_cache_is_rebuilt_on_resume(self) -> None:
        from experiments.exp267_273_zero_support_fn_recovery.source_pipeline import cache_paths

        archive, metadata = cache_paths(self.root, "scattering", "d1")
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(b"partial archive")
        metadata.write_text('{"partial": true}', encoding="utf-8")
        rebuilt = self._build()
        self.assertTrue(archive.exists())
        self.assertTrue(metadata.exists())
        self.assertIn("cache_sha256", rebuilt)

    def test_cache_source_records_errors_in_deterministic_manifest(self) -> None:
        from experiments.exp267_273_zero_support_fn_recovery.source_pipeline import (
            cache_source,
            freeze_source_predictions,
        )

        with patch(
            "experiments.exp267_273_zero_support_fn_recovery.source_pipeline.load_bundle",
            side_effect=RuntimeError("bundle unavailable"),
        ):
            rows = cache_source(
                self.root,
                "scattering",
                datasets=("z", "a"),
                workers=1,
                expected_dataset_count=2,
            )
        self.assertEqual(rows, [])
        errors = (self.root / "source_cache" / "scattering" / "cache_errors.csv").read_text()
        self.assertLess(errors.index("a,"), errors.index("z,"))
        manifest = json.loads(
            (self.root / "source_cache" / "scattering" / "source_manifest.json").read_text()
        )
        self.assertFalse(manifest["complete"])
        with self.assertRaises(ValueError):
            freeze_source_predictions(
                self.root, sources=("scattering",), datasets=("a", "z"), expected_dataset_count=2,
                bundle_loader=_bundle,
            )

    def test_freeze_is_invariant_to_input_dataset_order(self) -> None:
        from experiments.exp267_273_zero_support_fn_recovery.source_pipeline import freeze_source_predictions

        self._build(dataset="d1")
        self._build(dataset="d2")
        first, _ = freeze_source_predictions(
            self.root, sources=("scattering",), datasets=("d2", "d1"), expected_dataset_count=2,
            output_dir=self.root / "first", bundle_loader=_bundle,
        )
        second, _ = freeze_source_predictions(
            self.root, sources=("scattering",), datasets=("d1", "d2"), expected_dataset_count=2,
            output_dir=self.root / "second", bundle_loader=_bundle,
        )
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_explicit_abstention_freezes_without_non_json_nan(self) -> None:
        from experiments.exp267_273_zero_support_fn_recovery.source_pipeline import freeze_source_predictions

        self._build(source_factory=lambda *_args, **_kwargs: _AbstainingSource())
        prediction_path, _ = freeze_source_predictions(
            self.root, sources=("scattering",), datasets=("d1",), expected_dataset_count=1,
            bundle_loader=_bundle,
        )
        rows = [json.loads(line) for line in prediction_path.read_text().splitlines()]
        self.assertEqual(rows[0]["attainable_p"], None)
        self.assertEqual(rows[0]["candidate_indices"], [])

    def test_coverage_contract_rejects_fixture_as_full_population(self) -> None:
        from experiments.exp267_273_zero_support_fn_recovery.source_pipeline import freeze_source_predictions

        self._build()
        with self.assertRaises(ValueError):
            freeze_source_predictions(
                self.root, sources=("scattering",), datasets=("d1",), expected_dataset_count=1117,
                bundle_loader=_bundle,
            )

    def test_alpha_boundary_is_inclusive_and_deterministic(self) -> None:
        from experiments.exp267_273_zero_support_fn_recovery.source_pipeline import candidate_indices

        reference = np.asarray([0.0, 1.0, 2.0])
        query = np.asarray([2.0, 1.0])
        self.assertEqual(candidate_indices(reference, query, 0.5), (0,))
        self.assertEqual(candidate_indices(reference, query, 0.499999), ())

    def test_explicit_abstention_is_complete(self) -> None:
        from experiments.exp267_273_zero_support_fn_recovery.source_pipeline import cache_result_complete

        self.assertTrue(
            cache_result_complete(
                {
                    "abstained": True,
                    "abstention_reason": "n_train_below_5",
                    "eligible": False,
                }
            )
        )
