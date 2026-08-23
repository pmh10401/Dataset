"""Fixture-only authority and atomic-publication tests for Exp451 migration."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.exp451_verified_accelerated_resume_antigravity.config_antigravity import (
    EXP450_SOURCE_CATALOG_SHA256,
    EXP450_SOURCE_CONTEXT_SHA256,
    EXP450_SOURCE_TRANSFORMS_SHA256,
)
from experiments.exp451_verified_accelerated_resume_antigravity.contracts_antigravity import (
    canonical_json_bytes,
)


class MigrationAuthorityTests(unittest.TestCase):
    """A changed historical authority must stop reuse before any checkpoint read."""

    def setUp(self) -> None:
        self.temporary = TemporaryDirectory(prefix="exp451_migration_fixture_antigravity_")
        self.root = Path(self.temporary.name)
        self.repository = Path(__file__).resolve().parents[3]
        self.catalog = {"catalog_sha256": EXP450_SOURCE_CATALOG_SHA256}

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_reconstruction_uses_historical_closure_and_fixed_authority(self) -> None:
        """Replacing a closure byte or catalog identity must invalidate source reuse."""
        from experiments.exp451_verified_accelerated_resume_antigravity.migration_antigravity import (
            TRANSFORMS_PATH_ANTIGRAVITY,
            reconstruct_exp450_source_context_antigravity,
        )

        context = reconstruct_exp450_source_context_antigravity(self.repository, self.catalog)

        self.assertEqual(context.sha256, EXP450_SOURCE_CONTEXT_SHA256)
        self.assertEqual(
            context.code_closure[TRANSFORMS_PATH_ANTIGRAVITY],
            EXP450_SOURCE_TRANSFORMS_SHA256,
        )
        with self.assertRaisesRegex(ValueError, "catalog"):
            reconstruct_exp450_source_context_antigravity(
                self.repository, {"catalog_sha256": "0" * 64}
            )

    def test_create_only_envelope_reuses_only_exact_bytes(self) -> None:
        """Changing an existing target artifact must never be silently overwritten."""
        from experiments.exp451_verified_accelerated_resume_antigravity.artifacts_antigravity import (
            read_exp451_dataset_condition_success_antigravity,
            write_exp451_dataset_condition_success_antigravity,
        )

        root = self.root / "target_antigravity"
        envelope = _target_envelope()
        path = write_exp451_dataset_condition_success_antigravity(root, envelope)
        self.assertEqual(
            read_exp451_dataset_condition_success_antigravity(
                root, path.relative_to(root).as_posix()
            ),
            envelope,
        )
        path.write_bytes(canonical_json_bytes({**envelope, "dataset": "changed_antigravity"}))
        with self.assertRaisesRegex(ValueError, "differs"):
            write_exp451_dataset_condition_success_antigravity(root, envelope)

    def test_invalid_source_migration_leaves_fixture_tree_byte_identical(self) -> None:
        """A malformed source payload must fail closed without mutating source evidence."""
        from experiments.exp451_verified_accelerated_resume_antigravity.migration_antigravity import (
            migrate_verified_exp450_results_antigravity,
        )

        source = self.root / "source_antigravity"
        source.mkdir()
        bad = source / "checkpoint_antigravity.json"
        bad.write_bytes(b'{"not":"a verified source checkpoint"}\n')
        before = _tree_hashes(source)
        with self.assertRaisesRegex(ValueError, "source"):
            migrate_verified_exp450_results_antigravity(
                repository=self.repository,
                source_root=source,
                target_root=self.root / "target_antigravity",
                catalog=self.catalog,
                target_context_sha256="f" * 64,
            )
        self.assertEqual(_tree_hashes(source), before)

    def test_source_envelope_rejects_backend_raw_identity_and_payload_tampering(self) -> None:
        """A source checkpoint must bind each scientific payload to its exact catalog identity."""
        from experiments.exp451_verified_accelerated_resume_antigravity.migration_antigravity import (
            _source_envelope_antigravity,
            reconstruct_exp450_source_context_antigravity,
        )

        context = reconstruct_exp450_source_context_antigravity(self.repository, self.catalog)
        source = self.root / "source_antigravity"
        relative = (
            "checkpoints_antigravity/A_antigravity/"
            f"{sha256(b'fixture_antigravity').hexdigest()}_antigravity/"
            "dataset_condition_success_antigravity.json"
        )
        entry = {"dataset": "fixture_antigravity", "train": {}, "test": {}}
        fixture_catalog = {"catalog_sha256": EXP450_SOURCE_CATALOG_SHA256, "datasets": [entry]}
        baseline = _source_envelope(context, entry)
        path = source / relative
        path.parent.mkdir(parents=True)
        for changed, expected in (
            ({**baseline, "backend": {}}, "backend"),
            ({**baseline, "raw_identity": {"dataset": "other_antigravity"}}, "raw identity"),
            ({**baseline, "payload": {"sources": [], "route": {}}, "payload_sha256": "0" * 64}, "payload"),
        ):
            path.write_bytes(canonical_json_bytes(changed))
            with self.assertRaisesRegex(ValueError, expected):
                _source_envelope_antigravity(
                    self.repository, source, relative, context, fixture_catalog
                )

    def test_nonofficial_source_root_requires_explicit_fixture_authority(self) -> None:
        """Production migration cannot silently treat an arbitrary directory as Exp450."""
        from experiments.exp451_verified_accelerated_resume_antigravity.migration_antigravity import (
            FixtureMigrationAuthorityAntigravity,
            migrate_verified_exp450_results_antigravity,
        )

        source = self.root / "source_antigravity"
        source.mkdir()
        with self.assertRaisesRegex(ValueError, "official"):
            migrate_verified_exp450_results_antigravity(
                repository=self.repository,
                source_root=source,
                target_root=self.root / "target_antigravity",
                catalog=self.catalog,
                target_context_sha256="f" * 64,
            )
        authority = FixtureMigrationAuthorityAntigravity(
            source_root=source,
            source_context_sha256="a" * 64,
            catalog_sha256=EXP450_SOURCE_CATALOG_SHA256,
        )
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            migrate_verified_exp450_results_antigravity(
                repository=self.repository,
                source_root=source,
                target_root=self.root / "target_antigravity",
                catalog=self.catalog,
                target_context_sha256="f" * 64,
                fixture_authority=authority,
            )

    def test_full_fixture_migration_reverification_and_failure_boundaries(self) -> None:
        """Only a fully typed 4,234-coordinate fixture publishes a manifest."""
        from experiments.exp451_verified_accelerated_resume_antigravity.migration_antigravity import (
            migrate_verified_exp450_results_antigravity,
            verify_migration_manifest_antigravity,
        )

        source, catalog, authority, source_paths = _full_source_fixture(self.root)
        target = self.root / "target_antigravity"
        before = _tree_hashes(source)
        manifest = migrate_verified_exp450_results_antigravity(
            repository=self.repository, source_root=source, target_root=target,
            catalog=catalog, target_context_sha256="f" * 64,
            fixture_authority=authority,
        )
        self.assertEqual(manifest.total_count, 4234)
        self.assertEqual(manifest.counts_by_condition, {"A": 1117, "B": 1117, "C": 1000, "D": 1000})
        self.assertEqual(
            verify_migration_manifest_antigravity(
                repository=self.repository, source_root=source, target_root=target,
                catalog=catalog, target_context_sha256="f" * 64,
                fixture_authority=authority,
            ),
            manifest,
        )
        self.assertEqual(_tree_hashes(source), before)

        target_leaf = target / manifest.entries[0].target_relative_path
        target_value = json.loads(target_leaf.read_text(encoding="utf-8"))
        target_leaf.write_bytes(canonical_json_bytes({**target_value, "backend": {}}))
        with self.assertRaisesRegex(ValueError, "entry"):
            verify_migration_manifest_antigravity(
                repository=self.repository, source_root=source, target_root=target,
                catalog=catalog, target_context_sha256="f" * 64,
                fixture_authority=authority,
            )

        source_leaf = source / source_paths[0]
        source_value = json.loads(source_leaf.read_text(encoding="utf-8"))
        changed_payload = {**source_value["payload"], "route": {**source_value["payload"]["route"], "n_test": 3}}
        source_leaf.write_bytes(canonical_json_bytes({
            **source_value, "payload": changed_payload,
            "payload_sha256": sha256(canonical_json_bytes(changed_payload)).hexdigest(),
        }))
        source_before_reject = _tree_hashes(source)
        partial_target = self.root / "typed_reject_target_antigravity"
        with self.assertRaisesRegex(ValueError, "typed payload"):
            migrate_verified_exp450_results_antigravity(
                repository=self.repository, source_root=source, target_root=partial_target,
                catalog=catalog, target_context_sha256="f" * 64,
                fixture_authority=authority,
            )
        self.assertFalse((partial_target / "migration_manifest_antigravity.json").exists())
        self.assertEqual(_tree_hashes(source), source_before_reject)

        source_leaf.write_bytes(canonical_json_bytes(source_value))
        (source / source_paths[-1]).unlink()
        source_before_partial = _tree_hashes(source)
        incomplete_target = self.root / "incomplete_target_antigravity"
        with self.assertRaisesRegex(ValueError, "counts"):
            migrate_verified_exp450_results_antigravity(
                repository=self.repository, source_root=source, target_root=incomplete_target,
                catalog=catalog, target_context_sha256="f" * 64,
                fixture_authority=authority,
            )
        self.assertFalse((incomplete_target / "migration_manifest_antigravity.json").exists())
        self.assertEqual(_tree_hashes(source), source_before_partial)

    def test_fixture_source_and_target_symlinks_and_root_overlap_are_rejected(self) -> None:
        """A traversal through either root must fail before checkpoint publication."""
        from experiments.exp451_verified_accelerated_resume_antigravity.artifacts_antigravity import (
            write_exp451_dataset_condition_success_antigravity,
        )
        from experiments.exp451_verified_accelerated_resume_antigravity.migration_antigravity import (
            FixtureMigrationAuthorityAntigravity,
            migrate_verified_exp450_results_antigravity,
        )

        source = self.root / "source_antigravity"
        source.mkdir()
        authority = FixtureMigrationAuthorityAntigravity(
            source_root=source, source_context_sha256="a" * 64, catalog_sha256="b" * 64,
        )
        symlink_base = source / "checkpoints_antigravity"
        symlink_base.symlink_to(self.root / "elsewhere_antigravity")
        with self.assertRaisesRegex(ValueError, "invalid"):
            migrate_verified_exp450_results_antigravity(
                repository=self.repository, source_root=source,
                target_root=self.root / "target_antigravity", catalog={"catalog_sha256": "b" * 64},
                target_context_sha256="f" * 64, fixture_authority=authority,
            )
        symlink_base.unlink()
        with self.assertRaisesRegex(ValueError, "overlap"):
            migrate_verified_exp450_results_antigravity(
                repository=self.repository, source_root=source,
                target_root=source / "nested_target_antigravity", catalog={"catalog_sha256": "b" * 64},
                target_context_sha256="f" * 64, fixture_authority=authority,
            )
        with self.assertRaisesRegex(ValueError, "overlap"):
            migrate_verified_exp450_results_antigravity(
                repository=self.repository, source_root=source,
                target_root=source.parent, catalog={"catalog_sha256": "b" * 64},
                target_context_sha256="f" * 64, fixture_authority=authority,
            )
        actual_target = self.root / "actual_target_antigravity"
        actual_target.mkdir()
        target_link = self.root / "target_link_antigravity"
        target_link.symlink_to(actual_target)
        with self.assertRaisesRegex(ValueError, "unsafe"):
            write_exp451_dataset_condition_success_antigravity(target_link, _target_envelope())


def _target_envelope() -> dict[str, object]:
    payload = {"route": {"n_test": 1}, "sources": []}
    return {
        "schema": "exp451-dataset-condition-success-antigravity.v1",
        "context_sha256": "e" * 64,
        "source_context_sha256": "c" * 64,
        "catalog_sha256": "d" * 64,
        "condition_id": "A",
        "dataset": "fixture_antigravity",
        "raw_identity": {"dataset": "fixture_antigravity"},
        "backend": {"tensor_backend": "mps"},
        "execution_origin": "imported_exp450_mps",
        "payload": payload,
        "payload_sha256": sha256(canonical_json_bytes(payload)).hexdigest(),
        "source_envelope_sha256": "b" * 64,
    }


def _source_envelope(context: object, entry: dict[str, object]) -> dict[str, object]:
    payload = {"sources": [], "route": {}}
    return {
        "schema": "exp450-dataset-condition-success-antigravity.v1",
        "context_sha256": context.sha256,
        "catalog_sha256": EXP450_SOURCE_CATALOG_SHA256,
        "condition_id": "A",
        "dataset": "fixture_antigravity",
        "raw_identity": entry,
        "backend": dict(context.backend),
        "payload": payload,
        "payload_sha256": sha256(canonical_json_bytes(payload)).hexdigest(),
    }


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _full_source_fixture(root: Path) -> tuple[Path, dict[str, object], object, tuple[str, ...]]:
    from experiments.exp451_verified_accelerated_resume_antigravity.migration_antigravity import (
        FixtureMigrationAuthorityAntigravity,
    )

    source = root / "source_full_antigravity"
    payload = _typed_payload_fixture()
    datasets = [
        {"dataset": f"fixture_{index:04d}_antigravity", "train": {"count": 5}, "test": {"count": 2}}
        for index in range(1117)
    ]
    bare = {"schema": "fixture-catalog-antigravity.v1", "mode": "full", "database": {}, "datasets": datasets}
    catalog = {**bare, "catalog_sha256": sha256(canonical_json_bytes(bare)).hexdigest()}
    authority = FixtureMigrationAuthorityAntigravity(
        source_root=source, source_context_sha256="a" * 64,
        catalog_sha256=catalog["catalog_sha256"],
    )
    paths: list[str] = []
    for index, entry in enumerate(datasets):
        for condition in ("A", "B", "C", "D"):
            if condition in {"C", "D"} and index >= 1000:
                continue
            relative = (
                f"checkpoints_antigravity/{condition}_antigravity/"
                f"{sha256(entry['dataset'].encode('utf-8')).hexdigest()}_antigravity/"
                "dataset_condition_success_antigravity.json"
            )
            envelope = {
                "schema": "exp450-dataset-condition-success-antigravity.v1",
                "context_sha256": "a" * 64,
                "catalog_sha256": catalog["catalog_sha256"],
                "condition_id": condition,
                "dataset": entry["dataset"],
                "raw_identity": entry,
                "backend": {
                    "tensor_backend": "mps", "torch_version": "2.13.0", "mps_available": True,
                    "mlx_available": False, "logical_cpu_count": 12, "cpu_workers": 11,
                    "inner_numerical_threads": 1,
                },
                "payload": payload,
                "payload_sha256": sha256(canonical_json_bytes(payload)).hexdigest(),
            }
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(canonical_json_bytes(envelope))
            paths.append(relative)
    return source, catalog, authority, tuple(paths)


def _typed_payload_fixture() -> dict[str, object]:
    from experiments.exp450_resampling_image_ablation_antigravity.artifacts_antigravity import (
        _route_payload,
        _source_payload,
    )
    from paper.equipment_anomaly_detection.hard_route_paper import (
        HardRouteResult,
        RouteLanes,
        SourceExpansion,
    )
    from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import (
        EXPECTED_HARD_SOURCES,
        HardSourceBundle,
        _completed_paper,
    )

    sources = HardSourceBundle(
        EXPECTED_HARD_SOURCES,
        {
            name: _completed_paper(
                name, np.asarray([0., 1., 2., 3., 4.], dtype=np.float64),
                np.asarray([1.5, 4.5], dtype=np.float64),
            )
            for name in EXPECTED_HARD_SOURCES
        },
    )
    lanes = RouteLanes(frozenset(), frozenset(), frozenset())
    route = HardRouteResult(
        lanes, frozenset(), frozenset(), frozenset(), lanes,
        SourceExpansion(frozenset(), frozenset()), lanes, frozenset(), lanes,
        frozenset(), frozenset(), 2,
    )
    return {"sources": _source_payload(sources, 5, 2), "route": _route_payload(route, 2)}


if __name__ == "__main__":
    unittest.main()
