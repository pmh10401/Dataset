"""Synthetic Git and source-tree tests for the Exp467 authority boundary."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

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
SOURCE_TREE_SHA256_ANTIGRAVITY = "4e4a73274354ac2f41c24c4c766a6deac91e80d3220343da85f1ccc2adef7b7b"
CONDITION_MANIFEST_SHA256_ANTIGRAVITY = {
    "A": "6ebae25fb8e6e403190718183ee5ecc3f874aaf4f79b7cc91591fff3bbe2c222",
    "B": "2fea9b73ec729d9197c53e55c5e0322295185792c7a0d46372e1e922cda9c54f",
    "C": "6f414cf78c741877111fb6c92ab1b5eb2fa2a07aff6f79ceac3f816df6d4c961",
    "D": "66978957cf503c05eba2aa0d947b95e86a8461cd46bc4965fb6147c730f69bdc",
}


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
        self.root = Path(temporary).resolve()
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
        nested = self.source_root / "nested_antigravity"
        nested.mkdir()
        (nested / "payload_antigravity.bin").write_bytes(b"nested-antigravity\n")
        for condition in ("A", "B", "C", "D"):
            value = {"condition": condition, "origin": "frozen"}
            (self.source_root / f"{condition}_condition_manifest_antigravity.json").write_bytes(
                (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
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
                self.assertEqual(authority.source_tree_sha256, SOURCE_TREE_SHA256_ANTIGRAVITY)
                self.assertEqual(
                    dict(authority.condition_manifest_sha256),
                    CONDITION_MANIFEST_SHA256_ANTIGRAVITY,
                )
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
                score_calls: list[object] = []
                label_calls: list[object] = []
                with _synthetic_constants_antigravity(fixture):
                    with self.assertRaises(ValueError):
                        source_authority.open_exp467_source_session_antigravity(
                            fixture.repository,
                            fixture.source_root,
                            score_reader_antigravity=lambda: score_calls.append(object()),
                            label_callback_antigravity=lambda: label_calls.append(object()),
                        )
                self.assertEqual((score_calls, label_calls), ([], []))

        with tempfile.TemporaryDirectory(prefix="exp471_git_gate_antigravity_") as temporary:
            fixture = SyntheticSourceAntigravity(temporary)
            verifier = fixture.repository / "experiments/exp467_numba_calibration_resume_antigravity/verifier_antigravity.py"
            verifier.write_bytes(verifier.read_bytes().replace(b"VerificationAntigravity", b"VerificationMutatedAntigravity", 1))
            score_calls = []
            label_calls = []
            with _synthetic_constants_antigravity(fixture):
                with self.assertRaisesRegex(ValueError, "worktree closure"):
                    source_authority.open_exp467_source_session_antigravity(
                        fixture.repository,
                        fixture.source_root,
                        score_reader_antigravity=lambda: score_calls.append(object()),
                        label_callback_antigravity=lambda: label_calls.append(object()),
                    )
            self.assertEqual((score_calls, label_calls), ([], []))
            self.assertFalse(fixture.import_marker.exists())

        with tempfile.TemporaryDirectory(prefix="exp471_historical_gate_antigravity_") as temporary:
            fixture = SyntheticSourceAntigravity(temporary)
            changed_commit = ("0" if fixture.commit[0] != "0" else "1") + fixture.commit[1:]
            score_calls = []
            label_calls = []
            with mock.patch.object(
                source_authority, "EXP467_SOURCE_COMMIT_ANTIGRAVITY", changed_commit
            ):
                with self.assertRaisesRegex(ValueError, "historical closure"):
                    source_authority.open_exp467_source_session_antigravity(
                        fixture.repository,
                        fixture.source_root,
                        score_reader_antigravity=lambda: score_calls.append(object()),
                        label_callback_antigravity=lambda: label_calls.append(object()),
                    )
            self.assertEqual((score_calls, label_calls), ([], []))
            self.assertFalse(fixture.import_marker.exists())

    def test_descriptor_boundaries_reject_source_leaf_and_parent_swaps(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp471_source_swap_antigravity_") as temporary:
            fixture = SyntheticSourceAntigravity(temporary)
            checkpoint = fixture.source_root / "checkpoint_envelope_antigravity.json"
            replacement = fixture.root / "replacement_checkpoint_antigravity.json"
            replacement.write_bytes(checkpoint.read_bytes())
            fired = False

            def source_leaf_swap(event: str, relative: str) -> None:
                nonlocal fired
                if not fired and event == "source:after_leaf_open" and relative == checkpoint.name:
                    fired = True
                    checkpoint.rename(checkpoint.with_suffix(".saved_antigravity"))
                    checkpoint.symlink_to(replacement)

            with _synthetic_constants_antigravity(fixture), mock.patch.object(
                source_authority, "_race_hook_antigravity", source_leaf_swap, create=True
            ):
                with self.assertRaises(ValueError):
                    source_authority.verify_exp467_source_antigravity(
                        fixture.repository, fixture.source_root
                    )
            self.assertTrue(fired)

    def test_worktree_closure_rejects_leaf_and_parent_swaps_before_import(self) -> None:
        relative = "experiments/exp467_numba_calibration_resume_antigravity/verifier_antigravity.py"
        with tempfile.TemporaryDirectory(prefix="exp471_worktree_leaf_antigravity_") as temporary:
            fixture = SyntheticSourceAntigravity(temporary)
            verifier = fixture.repository / relative
            replacement = fixture.root / "replacement_verifier_antigravity.py"
            replacement.write_bytes(verifier.read_bytes())
            fired = False

            def worktree_leaf_swap(event: str, path: str) -> None:
                nonlocal fired
                if not fired and event == "worktree:after_leaf_open" and path == relative:
                    fired = True
                    verifier.rename(verifier.with_suffix(".saved_antigravity"))
                    verifier.symlink_to(replacement)

            with _synthetic_constants_antigravity(fixture), mock.patch.object(
                source_authority, "_race_hook_antigravity", worktree_leaf_swap
            ):
                with self.assertRaises(ValueError):
                    source_authority.open_exp467_source_session_antigravity(
                        fixture.repository, fixture.source_root
                    )
            self.assertTrue(fired)
            self.assertFalse(fixture.import_marker.exists())

    def test_verified_captured_module_bytes_are_executed_after_path_swap(self) -> None:
        relative = "experiments/exp467_numba_calibration_resume_antigravity/verifier_antigravity.py"
        with tempfile.TemporaryDirectory(prefix="exp471_captured_import_antigravity_") as temporary:
            fixture = SyntheticSourceAntigravity(temporary)
            verifier = fixture.repository / relative
            malicious_marker = fixture.root / "malicious_verifier_imported_antigravity"
            fired = False

            def post_snapshot_swap(event: str, path: str) -> None:
                nonlocal fired
                if not fired and event == "worktree:after_snapshot":
                    fired = True
                    verifier.write_text(
                        f'from pathlib import Path\nPath({str(malicious_marker)!r}).write_text("bad")\n',
                        encoding="utf-8",
                    )

            with _synthetic_constants_antigravity(fixture), mock.patch.object(
                source_authority, "_race_hook_antigravity", post_snapshot_swap
            ):
                authority = source_authority.verify_exp467_source_antigravity(
                    fixture.repository, fixture.source_root
                )
            self.assertTrue(fired)
            self.assertEqual(authority.total_successes, 4468)
            self.assertTrue(fixture.import_marker.exists())
            self.assertFalse(malicious_marker.exists())

        with tempfile.TemporaryDirectory(prefix="exp471_worktree_parent_antigravity_") as temporary:
            fixture = SyntheticSourceAntigravity(temporary)
            package = fixture.repository / "experiments/exp467_numba_calibration_resume_antigravity"
            outside = fixture.root / "replacement_package_antigravity"
            shutil.copytree(package, outside)
            fired = False

            def worktree_parent_swap(event: str, path: str) -> None:
                nonlocal fired
                if not fired and event == "worktree:after_directory_open" and path == "experiments/exp467_numba_calibration_resume_antigravity":
                    fired = True
                    package.rename(package.with_name("exp467_saved_package_antigravity"))
                    package.symlink_to(outside, target_is_directory=True)

            with _synthetic_constants_antigravity(fixture), mock.patch.object(
                source_authority, "_race_hook_antigravity", worktree_parent_swap
            ):
                with self.assertRaises(ValueError):
                    source_authority.open_exp467_source_session_antigravity(
                        fixture.repository, fixture.source_root
                    )
            self.assertTrue(fired)
            self.assertFalse(fixture.import_marker.exists())

        with tempfile.TemporaryDirectory(prefix="exp471_source_parent_swap_antigravity_") as temporary:
            fixture = SyntheticSourceAntigravity(temporary)
            nested = fixture.source_root / "nested_antigravity"
            outside = fixture.root / "outside_nested_antigravity"
            shutil.copytree(nested, outside)
            fired = False

            def source_parent_swap(event: str, relative: str) -> None:
                nonlocal fired
                if not fired and event == "source:after_directory_open" and relative == "nested_antigravity":
                    fired = True
                    nested.rename(fixture.source_root / "nested_saved_antigravity")
                    nested.symlink_to(outside, target_is_directory=True)

            with _synthetic_constants_antigravity(fixture), mock.patch.object(
                source_authority, "_race_hook_antigravity", source_parent_swap, create=True
            ):
                with self.assertRaises(ValueError):
                    source_authority.verify_exp467_source_antigravity(
                        fixture.repository, fixture.source_root
                    )
            self.assertTrue(fired)

    def test_authority_root_and_leaf_operations_are_descriptor_bound(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp471_authority_fd_antigravity_") as temporary:
            fixture = SyntheticSourceAntigravity(temporary)
            real_parent = fixture.root / "real_parent_antigravity"
            real_root = real_parent / "authority_root_antigravity"
            real_root.mkdir(parents=True)
            linked_parent = fixture.root / "linked_parent_antigravity"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            linked_root = linked_parent / real_root.name
            with _synthetic_constants_antigravity(fixture):
                authority = source_authority.verify_exp467_source_antigravity(
                    fixture.repository, fixture.source_root
                )
                with self.assertRaises(ValueError):
                    source_authority.publish_source_authority_antigravity(
                        linked_root, authority
                    )
                path = source_authority.publish_source_authority_antigravity(
                    real_root, authority
                )
                with self.assertRaises(ValueError):
                    source_authority.read_source_authority_antigravity(
                        linked_root, fixture.repository, fixture.source_root
                    )

                replacement = fixture.root / "replacement_authority_antigravity.json"
                replacement.write_bytes(path.read_bytes())
                fired = False

                def authority_leaf_swap(event: str, relative: str) -> None:
                    nonlocal fired
                    if not fired and event == "authority:after_leaf_open":
                        fired = True
                        path.chmod(0o644)
                        path.rename(path.with_suffix(".saved_antigravity"))
                        path.symlink_to(replacement)

                with mock.patch.object(
                    source_authority, "_race_hook_antigravity", authority_leaf_swap, create=True
                ):
                    with self.assertRaises(ValueError):
                        source_authority.read_source_authority_antigravity(
                            real_root, fixture.repository, fixture.source_root
                        )
                self.assertTrue(fired)

    def test_publication_failure_never_exposes_partial_final_name(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp471_partial_publish_antigravity_") as temporary:
            fixture = SyntheticSourceAntigravity(temporary)
            with _synthetic_constants_antigravity(fixture):
                authority = source_authority.verify_exp467_source_antigravity(
                    fixture.repository, fixture.source_root
                )

                def partial_write(descriptor: int, payload: bytes) -> None:
                    os.write(descriptor, payload[:7])
                    raise OSError("synthetic partial write")

                with mock.patch.object(
                    source_authority, "_write_all_antigravity", partial_write, create=True
                ):
                    with self.assertRaises(OSError):
                        source_authority.publish_source_authority_antigravity(
                            fixture.target_root, authority
                        )
                self.assertFalse(
                    (fixture.target_root / "source_authority_antigravity.json").exists()
                )
                self.assertEqual(
                    tuple(
                        path
                        for path in fixture.target_root.iterdir()
                        if "temporary_antigravity" in path.name
                    ),
                    (),
                )

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
