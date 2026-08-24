"""Synthetic Git and source-tree tests for the Exp467 authority boundary."""

from __future__ import annotations

from hashlib import sha256
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from experiments.exp471_exp467_evaluation_recovery_antigravity.contracts_antigravity import (
    canonical_json_bytes_exp471_antigravity,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity import (
    contracts_antigravity as contracts,
)
from experiments.exp471_exp467_evaluation_recovery_antigravity import (
    source_authority_antigravity as source_authority,
)


CONTEXT_SHA256_ANTIGRAVITY = "c" * 64
CATALOG_SHA256_ANTIGRAVITY = "4" * 64
DATABASE_SHA256_ANTIGRAVITY = "5" * 64
INVENTORY_SHA256_ANTIGRAVITY = "3" * 64


@contextmanager
def _synthetic_constants_antigravity(fixture: object):
    with mock.patch.object(
        source_authority, "EXP467_SOURCE_COMMIT_ANTIGRAVITY", fixture.commit
    ), mock.patch.object(
        source_authority,
        "EXP467_SOURCE_CONTEXT_SHA256_ANTIGRAVITY",
        CONTEXT_SHA256_ANTIGRAVITY,
    ), mock.patch.object(
        contracts, "EXP467_SOURCE_COMMIT_ANTIGRAVITY", fixture.commit
    ), mock.patch.object(
        contracts,
        "EXP467_SOURCE_CONTEXT_SHA256_ANTIGRAVITY",
        CONTEXT_SHA256_ANTIGRAVITY,
    ):
        yield


def _run_antigravity(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


class SyntheticSourceAntigravity:
    def __init__(self, temporary: str) -> None:
        self.root = Path(temporary)
        self.repository = self.root / "repository_antigravity"
        self.source_root = self.root / "source_root_antigravity"
        self.target_root = self.root / "target_root_antigravity"
        self.import_marker = self.root / "verifier_imported_antigravity"
        self.repository.mkdir()
        self.source_root.mkdir()
        self.target_root.mkdir()
        (self.repository / "scratch").mkdir()
        package = (
            self.repository
            / "experiments"
            / "exp467_numba_calibration_resume_antigravity"
        )
        package.mkdir(parents=True)
        launcher_path = self.repository / "scratch" / "run_exp467_numba_calibration_resume_antigravity.py"
        verifier_path = package / "verifier_antigravity.py"
        launcher_path.write_text(
            f'''from pathlib import Path

class ContextAntigravity:
    sha256 = {CONTEXT_SHA256_ANTIGRAVITY!r}
    catalog_sha256 = {CATALOG_SHA256_ANTIGRAVITY!r}
    database_sha256 = {DATABASE_SHA256_ANTIGRAVITY!r}
    run_root = Path({str(self.source_root)!r})

class ManifestAntigravity:
    source_context_sha256 = "1" * 64

def runtime_dependency_relative_paths_antigravity():
    return (
        "scratch/run_exp467_numba_calibration_resume_antigravity.py",
        "experiments/exp467_numba_calibration_resume_antigravity/verifier_antigravity.py",
    )

def _full_inputs_antigravity():
    return ContextAntigravity(), {{"catalog_sha256": {CATALOG_SHA256_ANTIGRAVITY!r}}}

def _verify_migration_antigravity():
    context, catalog = _full_inputs_antigravity()
    return context, catalog, ManifestAntigravity()
''',
            encoding="utf-8",
        )
        verifier_path.write_text(
            f'''from pathlib import Path
Path({str(self.import_marker)!r}).write_text("imported", encoding="utf-8")

class VerificationAntigravity:
    total_successes = 4468
    imported_by_condition = {{"A": 1117, "B": 1117, "C": 1000, "D": 1000}}
    native_by_condition = {{"A": 0, "B": 0, "C": 117, "D": 117}}
    inventory_sha256 = {INVENTORY_SHA256_ANTIGRAVITY!r}
    catalog_sha256 = {CATALOG_SHA256_ANTIGRAVITY!r}
    database_sha256 = {DATABASE_SHA256_ANTIGRAVITY!r}

def verify_exp467_completion_antigravity(root, *, context, catalog, migration_manifest):
    root = Path(root)
    if catalog.get("catalog_sha256") != (root / "catalog_sha256_antigravity.txt").read_text():
        raise ValueError("catalog")
    if context.database_sha256 != (root / "database_sha256_antigravity.txt").read_text():
        raise ValueError("database")
    if (root / "checkpoint_envelope_antigravity.json").read_bytes() != b"source-envelope-antigravity\\n":
        raise ValueError("source envelope")
    if (root / "origin_count_antigravity.txt").read_text() != "4234:234":
        raise ValueError("origin count")
    return VerificationAntigravity()

def read_persisted_exp467_condition_manifests_antigravity(root, verification):
    opened = {{}}
    for condition in ("A", "B", "C", "D"):
        path = Path(root) / f"{{condition}}_condition_manifest_antigravity.json"
        raw = path.read_bytes()
        value = __import__("json").loads(raw)
        expected = ({{"condition": condition, "origin": "frozen"}})
        canonical = (__import__("json").dumps(expected, sort_keys=True, separators=(",", ":")) + "\\n").encode()
        if raw != canonical or value != expected:
            raise ValueError("condition manifest")
        opened[condition] = {{**value, "manifest_sha256": __import__("hashlib").sha256(raw).hexdigest()}}
    return opened

def provenance_summary_antigravity(verification):
    imported = sum(verification.imported_by_condition.values())
    native = sum(verification.native_by_condition.values())
    return {{"total_successes": verification.total_successes, "imported_total": imported, "native_total": native}}
''',
            encoding="utf-8",
        )
        (self.source_root / "catalog_sha256_antigravity.txt").write_text(CATALOG_SHA256_ANTIGRAVITY)
        (self.source_root / "database_sha256_antigravity.txt").write_text(DATABASE_SHA256_ANTIGRAVITY)
        (self.source_root / "checkpoint_envelope_antigravity.json").write_bytes(b"source-envelope-antigravity\n")
        (self.source_root / "origin_count_antigravity.txt").write_text("4234:234")
        for condition in ("A", "B", "C", "D"):
            value = {"condition": condition, "origin": "frozen"}
            (self.source_root / f"{condition}_condition_manifest_antigravity.json").write_bytes(
                canonical_json_bytes_exp471_antigravity(value)
            )
        _run_antigravity(self.repository, "init", "-q")
        _run_antigravity(self.repository, "config", "user.email", "synthetic@example.invalid")
        _run_antigravity(self.repository, "config", "user.name", "Synthetic Antigravity")
        _run_antigravity(self.repository, "add", ".")
        _run_antigravity(self.repository, "commit", "-q", "-m", "synthetic authority")
        self.commit = _run_antigravity(self.repository, "rev-parse", "HEAD")


class Exp471SourceAuthorityTestsAntigravity(unittest.TestCase):
    def test_synthetic_git_source_verifies_publishes_and_reopens(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp471_source_antigravity_") as temporary:
            fixture = SyntheticSourceAntigravity(temporary)
            with _synthetic_constants_antigravity(fixture):
                context = source_authority.reconstruct_exp467_source_context_antigravity(fixture.repository)
                self.assertEqual(context.sha256, CONTEXT_SHA256_ANTIGRAVITY)
                session = source_authority.open_exp467_source_session_antigravity(
                    fixture.repository, fixture.source_root
                )
                authority = session.authority
                self.assertTrue(source_authority._source_session_is_issued_antigravity(session))
                self.assertEqual((authority.total_successes, authority.imported_total, authority.native_total), (4468, 4234, 234))
                self.assertEqual(authority.source_context_sha256, CONTEXT_SHA256_ANTIGRAVITY)
                self.assertEqual(
                    authority.source_file_sha256["checkpoint_envelope_antigravity.json"],
                    "fa756373aeadb850fc0ea4ce51299b8106f3368558f6da519bd81014d959db94",
                )
                expected_tree = sha256(canonical_json_bytes_exp471_antigravity(dict(authority.source_file_sha256))).hexdigest()
                self.assertEqual(authority.source_tree_sha256, expected_tree)
                path = source_authority.publish_source_authority_antigravity(fixture.target_root, authority)
                self.assertEqual(path.name, "source_authority_antigravity.json")
                self.assertEqual(source_authority.publish_source_authority_antigravity(fixture.target_root, authority), path)
                reopened = source_authority.read_source_authority_antigravity(
                    fixture.target_root, fixture.repository, fixture.source_root
                )
                self.assertEqual(reopened, authority)

    def test_all_mutations_fail_closed_and_historical_gate_is_lazy(self) -> None:
        mutations = (
            ("catalog_sha256_antigravity.txt", "0" * 64),
            ("database_sha256_antigravity.txt", "0" * 64),
            ("checkpoint_envelope_antigravity.json", "changed\n"),
            ("A_condition_manifest_antigravity.json", '{"condition":"A","origin":"changed"}\n'),
            ("origin_count_antigravity.txt", "4235:233"),
        )
        for relative, replacement in mutations:
            with self.subTest(relative=relative), tempfile.TemporaryDirectory(prefix="exp471_mutation_antigravity_") as temporary:
                fixture = SyntheticSourceAntigravity(temporary)
                (fixture.source_root / relative).write_text(replacement, encoding="utf-8")
                with _synthetic_constants_antigravity(fixture):
                    with self.assertRaises(ValueError):
                        source_authority.open_exp467_source_session_antigravity(fixture.repository, fixture.source_root)

        with tempfile.TemporaryDirectory(prefix="exp471_git_gate_antigravity_") as temporary:
            fixture = SyntheticSourceAntigravity(temporary)
            verifier = fixture.repository / "experiments/exp467_numba_calibration_resume_antigravity/verifier_antigravity.py"
            verifier.write_bytes(verifier.read_bytes().replace(b"VerificationAntigravity", b"VerificationMutatedAntigravity", 1))
            with _synthetic_constants_antigravity(fixture):
                with self.assertRaisesRegex(ValueError, "worktree closure"):
                    source_authority.open_exp467_source_session_antigravity(fixture.repository, fixture.source_root)
            self.assertFalse(fixture.import_marker.exists())

        with tempfile.TemporaryDirectory(prefix="exp471_historical_gate_antigravity_") as temporary:
            fixture = SyntheticSourceAntigravity(temporary)
            changed_commit = ("0" if fixture.commit[0] != "0" else "1") + fixture.commit[1:]
            with mock.patch.object(
                source_authority, "EXP467_SOURCE_COMMIT_ANTIGRAVITY", changed_commit
            ):
                with self.assertRaisesRegex(ValueError, "historical closure"):
                    source_authority.open_exp467_source_session_antigravity(
                        fixture.repository, fixture.source_root
                    )
            self.assertFalse(fixture.import_marker.exists())

    def test_changed_source_and_authority_artifacts_are_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp471_publish_antigravity_") as temporary:
            fixture = SyntheticSourceAntigravity(temporary)
            with _synthetic_constants_antigravity(fixture):
                authority = source_authority.verify_exp467_source_antigravity(fixture.repository, fixture.source_root)
                path = source_authority.publish_source_authority_antigravity(fixture.target_root, authority)
                original = path.read_bytes()
                path.chmod(0o644)
                path.write_bytes(b"{}\n")
                with self.assertRaises(ValueError):
                    source_authority.publish_source_authority_antigravity(fixture.target_root, authority)
                self.assertEqual(path.read_bytes(), b"{}\n")
                path.write_bytes(original)
                (fixture.source_root / "checkpoint_envelope_antigravity.json").write_bytes(b"changed\n")
                with self.assertRaises(ValueError):
                    source_authority.read_source_authority_antigravity(
                        fixture.target_root, fixture.repository, fixture.source_root
                    )


if __name__ == "__main__":
    unittest.main()
