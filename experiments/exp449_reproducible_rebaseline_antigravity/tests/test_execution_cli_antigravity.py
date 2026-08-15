"""Task 6 command-boundary tests using a temporary Git authority fixture."""

from __future__ import annotations

from dataclasses import fields
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import subprocess
import unittest
from unittest.mock import patch

from experiments.exp449_reproducible_rebaseline_antigravity import operations_antigravity as operations
from experiments.exp449_reproducible_rebaseline_antigravity import execution_authority_antigravity as authority
from experiments.exp449_reproducible_rebaseline_antigravity.tests.test_execution_claim_antigravity import (
    _context_antigravity,
    _verified_antigravity,
)
from paper.equipment_anomaly_detection.artifacts_paper import canonical_json_bytes


class Exp449ExecutionCliTests(unittest.TestCase):
    """Tests name the unsafe command behavior each one prevents."""

    def test_prepare_smoke_a_grant_creates_only_two_untracked_authority_json_files(self) -> None:
        """Skipping create-only publication or the Git-shape gate could prepare executable state."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            repository = Path(temporary)
            subprocess.run(["git", "init", "-q", str(repository)], check=True)
            roots = tuple(repository / f"root_{index}_antigravity" for index in range(6))
            claim = repository / "claim_antigravity.lock"
            authority_dir = repository / "docs" / "authorities"
            grant_path = authority_dir / "smoke_a_execution_grant_antigravity.json"
            digest_path = authority_dir / "smoke_a_execution_grant_digest_antigravity.json"
            frozen = SimpleNamespace(**{
                field.name: getattr(operations.FROZEN_EXP449_CONFIG, field.name)
                for field in fields(operations.FROZEN_EXP449_CONFIG)
            })
            frozen.smoke_a_scientific_root, frozen.smoke_a_operations_root = roots[:2]
            frozen.smoke_b_scientific_root, frozen.smoke_b_operations_root = roots[2:4]
            frozen.official_scientific_root, frozen.official_operations_root = roots[4:]
            facts = {
                "database": {"path": "/fixture/database.sqlite", "observed_size": 17, "sha256": "a" * 64},
                "contexts": {"smoke_a": {"catalog_sha256": "b" * 64, "context_sha256": "c" * 64, "acceleration_assessment_sha256": "d" * 64}},
                "code_closure": {"fixture_antigravity.py": "e" * 64},
                "lineage_lock_paper_sha256": "f" * 64,
                "codemap_lock_sha256": "0" * 64,
                "package_inventory": {"fixture": "1"},
                "seeds_and_constants": {"fixture": 449},
                "acceleration": {"selected_backend": "cpu"},
                "workers": {"logical_cpu_count": 12, "dataset_workers": 11},
            }
            with patch.object(operations, "FROZEN_EXP449_CONFIG", frozen), \
                 patch.object(authority, "FROZEN_EXP449_CONFIG", frozen), \
                 patch.object(operations, "SMOKE_A_CLAIM_PATH_ANTIGRAVITY", claim), \
                 patch.object(authority, "SMOKE_A_CLAIM_PATH_ANTIGRAVITY", claim), \
                 patch.object(operations, "_REPOSITORY_ROOT_ANTIGRAVITY", repository), \
                 patch.object(operations, "_fixed_roots_antigravity", return_value=roots), \
                 patch.object(operations, "readiness_exp449_antigravity", return_value=facts), \
                 patch.object(operations, "_sha256_file_antigravity", return_value="f" * 64), \
                 patch.object(operations, "SMOKE_A_GRANT_PATH_ANTIGRAVITY", grant_path, create=True), \
                 patch.object(operations, "SMOKE_A_DIGEST_PATH_ANTIGRAVITY", digest_path, create=True), \
                 patch.object(authority, "SMOKE_A_GRANT_PATH_ANTIGRAVITY", grant_path), \
                 patch.object(authority, "SMOKE_A_DIGEST_PATH_ANTIGRAVITY", digest_path):
                paths = operations.prepare_smoke_a_grant_antigravity("1" * 40)
            self.assertEqual(paths, (grant_path, digest_path))
            self.assertTrue(grant_path.is_file())
            self.assertTrue(digest_path.is_file())
            self.assertEqual(
                subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository, check=True, capture_output=True, text=True).stdout.splitlines(),
                [
                    "?? docs/authorities/smoke_a_execution_grant_antigravity.json",
                    "?? docs/authorities/smoke_a_execution_grant_digest_antigravity.json",
                ],
            )
            self.assertTrue(all(not path.exists() for path in (*roots, claim)))

    def test_worker_rejects_an_unlocked_canonical_claim_descriptor(self) -> None:
        """Dropping the live-lock probe would let a direct private worker mint authority."""
        with TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            scientific, operational, claim_path = root / "science_antigravity", root / "operations_antigravity", root / "claim_antigravity.lock"
            context = _context_antigravity(scientific)
            verified = _verified_antigravity(scientific, operational, claim_path, context)
            descriptor = os.open(claim_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(descriptor, canonical_json_bytes(operations._claim_payload_antigravity(verified, context)))
                with self.assertRaisesRegex(ValueError, "not live"):
                    operations._authenticate_inherited_smoke_a_claim_fd_antigravity(descriptor, verified, context)
            finally:
                os.close(descriptor)
