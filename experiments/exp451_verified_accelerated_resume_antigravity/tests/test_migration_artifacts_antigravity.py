"""Fixture-only authority and atomic-publication tests for Exp451 migration."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

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
                _source_envelope_antigravity(source, relative, context, fixture_catalog)


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


if __name__ == "__main__":
    unittest.main()
