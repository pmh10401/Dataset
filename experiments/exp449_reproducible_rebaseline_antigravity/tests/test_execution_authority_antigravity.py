"""Pure, fixture-only contracts for the deferred Exp449 Smoke A authority."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from types import MappingProxyType
import tempfile
import unittest
from unittest.mock import patch

from experiments.exp449_reproducible_rebaseline_antigravity.config_antigravity import (
    FROZEN_EXP449_CONFIG,
    SMOKE_A_CLAIM_PATH_ANTIGRAVITY,
    SMOKE_A_DIGEST_PATH_ANTIGRAVITY,
    SMOKE_A_GRANT_PATH_ANTIGRAVITY,
)
from experiments.exp449_reproducible_rebaseline_antigravity.execution_authority_antigravity import (
    Exp449SmokeAGrantAntigravity,
    Exp449SmokeAGrantDigestAntigravity,
    canonical_grant_bytes_antigravity,
    canonical_grant_digest_bytes_antigravity,
    parse_grant_bytes_antigravity,
    parse_grant_digest_bytes_antigravity,
    verify_smoke_a_authorization_commit_antigravity,
    verify_authority_directory_structure_antigravity,
)
from experiments.exp449_reproducible_rebaseline_antigravity import execution_authority_antigravity as authority_module


class Exp449SmokeAAuthorityContractTests(unittest.TestCase):
    """A wrong authority payload must fail before any future mutation boundary."""

    _SHA_A = "a" * 64
    _SHA_B = "b" * 64
    _SHA_C = "c" * 64

    def _grant(self) -> Exp449SmokeAGrantAntigravity:
        return Exp449SmokeAGrantAntigravity(
            schema="exp449-smoke-a-execution-grant-antigravity.v1",
            experiment_id="Exp449",
            target="smoke-a",
            policy_id="hard_plus_stable",
            implementation_commit="1" * 40,
            scientific_root=FROZEN_EXP449_CONFIG.smoke_a_scientific_root,
            operations_root=FROZEN_EXP449_CONFIG.smoke_a_operations_root,
            claim_path=SMOKE_A_CLAIM_PATH_ANTIGRAVITY,
            nonce="0123456789abcdef0123456789abcdef",
            maximum_waves=2,
            label_access=False,
            logical_cpu_count=12,
            dataset_workers=11,
            database_path="/fixture/dataset.sqlite",
            database_size=17,
            database_sha256=self._SHA_A,
            smoke_catalog_sha256=self._SHA_B,
            smoke_datasets=FROZEN_EXP449_CONFIG.smoke_datasets,
            context_sha256=self._SHA_B,
            scientific_code_closure={
                "experiments/exp449_reproducible_rebaseline_antigravity/config_antigravity.py": self._SHA_A,
                "experiments/exp449_reproducible_rebaseline_antigravity/runner_antigravity.py": self._SHA_B,
            },
            authority_code_closure={
                "experiments/exp449_reproducible_rebaseline_antigravity/execution_authority_antigravity.py": self._SHA_C,
            },
            lineage_lock_sha256=self._SHA_A,
            codemap_lock_sha256=self._SHA_B,
            acceleration_assessment_sha256=self._SHA_C,
            canonical_backend="cpu",
            package_inventory={"numpy": "2.3.1", "pycatch22": "0.5.0"},
            seeds_and_constants={"seed": 449, "alphas": (0.01, 0.05)},
            acceleration={"selected_backend": "cpu", "mlx_assessed": True, "mps_assessed": True, "pytorch_assessed": True},
            design_facts={
                "dataset_count": 1117,
                "test_row_count": 195900,
                "preprocessing_contracts": (
                    "task4_train_only",
                    "task5_label_free_train_test_value_length_transductive",
                ),
                "cpu_repeat_tolerance": 1e-12,
                "accelerated_tolerance": 1e-5,
                "minimum_accelerated_speedup": 1.1,
            },
            grant_created_at="2026-08-15T11:45:00Z",
        )

    def _digest(self) -> Exp449SmokeAGrantDigestAntigravity:
        grant_bytes = canonical_grant_bytes_antigravity(self._grant())
        return Exp449SmokeAGrantDigestAntigravity(
            schema="exp449-smoke-a-execution-grant-digest-antigravity.v1",
            implementation_commit="1" * 40,
            grant_path=SMOKE_A_GRANT_PATH_ANTIGRAVITY,
            grant_size=len(grant_bytes),
            grant_sha256=self._SHA_C,
        )

    def test_fixed_paths_and_contract_fields_are_frozen(self) -> None:
        self.assertEqual(
            SMOKE_A_GRANT_PATH_ANTIGRAVITY,
            Path(__file__).resolve().parents[3] / "docs/authorities/smoke_a_execution_grant_antigravity.json",
        )
        self.assertEqual(
            SMOKE_A_DIGEST_PATH_ANTIGRAVITY,
            Path(__file__).resolve().parents[3] / "docs/authorities/smoke_a_execution_grant_digest_antigravity.json",
        )
        self.assertEqual(
            SMOKE_A_CLAIM_PATH_ANTIGRAVITY,
            Path("/Users/mac/Documents/Dataset/paper_proposal_exp449_smoke_a_antigravity_20260814_claim_antigravity.lock"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(Exp449SmokeAGrantDigestAntigravity)),
            ("schema", "implementation_commit", "grant_path", "grant_size", "grant_sha256"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(Exp449SmokeAGrantAntigravity)),
            (
                "schema", "experiment_id", "target", "policy_id", "implementation_commit", "scientific_root", "operations_root", "claim_path",
                "nonce", "maximum_waves", "label_access", "logical_cpu_count", "dataset_workers", "database_path",
                "database_size", "database_sha256", "smoke_catalog_sha256", "smoke_datasets", "context_sha256", "scientific_code_closure",
                "authority_code_closure", "lineage_lock_sha256", "codemap_lock_sha256", "acceleration_assessment_sha256",
                "canonical_backend", "package_inventory", "seeds_and_constants", "acceleration", "design_facts", "grant_created_at",
            ),
        )
        with self.assertRaises(FrozenInstanceError):
            self._grant().target = "smoke-b"  # type: ignore[misc]

    def test_canonical_round_trips_reject_noncanonical_bytes_and_extra_fields(self) -> None:
        grant = self._grant()
        grant_bytes = canonical_grant_bytes_antigravity(grant)
        self.assertEqual(parse_grant_bytes_antigravity(grant_bytes), grant)
        self.assertEqual(canonical_grant_bytes_antigravity(parse_grant_bytes_antigravity(grant_bytes)), grant_bytes)
        with self.assertRaisesRegex(ValueError, "canonical"):
            parse_grant_bytes_antigravity(grant_bytes + b"\n")
        payload = json.loads(grant_bytes)
        payload["unapproved"] = True
        with self.assertRaisesRegex(ValueError, "fields"):
            parse_grant_bytes_antigravity(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        payload = json.loads(grant_bytes)
        payload.pop("nonce")
        with self.assertRaisesRegex(ValueError, "fields"):
            parse_grant_bytes_antigravity(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        digest = self._digest()
        digest_bytes = canonical_grant_digest_bytes_antigravity(digest)
        self.assertEqual(parse_grant_digest_bytes_antigravity(digest_bytes), digest)
        self.assertEqual(canonical_grant_digest_bytes_antigravity(parse_grant_digest_bytes_antigravity(digest_bytes)), digest_bytes)

    def test_grant_rejects_wrong_frozen_identity_and_noncanonical_order(self) -> None:
        grant = self._grant()
        for field, value in (
            ("schema", "wrong.v1"),
            ("experiment_id", "Exp450"),
            ("target", "smoke-b"),
            ("policy_id", "hard_only"),
            ("implementation_commit", "A" * 40),
            ("scientific_root", FROZEN_EXP449_CONFIG.smoke_b_scientific_root),
            ("operations_root", FROZEN_EXP449_CONFIG.smoke_b_operations_root),
            ("claim_path", Path("/tmp/claim_antigravity.lock")),
            ("nonce", "A" * 32),
            ("maximum_waves", 1),
            ("maximum_waves", 2.0),
            ("label_access", True),
            ("logical_cpu_count", 11),
            ("logical_cpu_count", 12.0),
            ("dataset_workers", 12),
            ("dataset_workers", 11.0),
            ("database_sha256", "A" * 64),
            ("smoke_catalog_sha256", "A" * 64),
            ("acceleration_assessment_sha256", "A" * 64),
            ("canonical_backend", "mlx"),
            ("grant_created_at", "2026-08-15 11:45:00Z"),
        ):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    replace(grant, **{field: value})
        with self.assertRaisesRegex(ValueError, "ordered"):
            replace(grant, smoke_datasets=tuple(reversed(grant.smoke_datasets)))

    def test_nested_authority_data_is_recursively_immutable_and_sha_maps_are_strict(self) -> None:
        grant = self._grant()
        for value in (
            grant.scientific_code_closure,
            grant.authority_code_closure,
            grant.package_inventory,
            grant.seeds_and_constants,
            grant.acceleration,
            grant.design_facts,
        ):
            self.assertIsInstance(value, MappingProxyType)
        self.assertIsInstance(grant.seeds_and_constants["alphas"], tuple)
        with self.assertRaises(TypeError):
            grant.scientific_code_closure["changed"] = self._SHA_A  # type: ignore[index]
        with self.assertRaises(ValueError):
            replace(grant, scientific_code_closure={"bad": "not-a-sha"})
        with self.assertRaises(ValueError):
            replace(grant, authority_code_closure={})
        with self.assertRaises(ValueError):
            replace(self._digest(), implementation_commit="A" * 40)
        with self.assertRaises(ValueError):
            replace(self._digest(), grant_path=SMOKE_A_DIGEST_PATH_ANTIGRAVITY)

    def test_authority_directory_rejects_non_authority_entries_without_reading_grants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "authorities"
            self.assertEqual(verify_authority_directory_structure_antigravity(directory), ())
            directory.mkdir()
            (directory / SMOKE_A_GRANT_PATH_ANTIGRAVITY.name).write_text("{}")
            self.assertEqual(
                verify_authority_directory_structure_antigravity(directory),
                (SMOKE_A_GRANT_PATH_ANTIGRAVITY.name,),
            )
            for kind in ("third", "python", "directory", "symlink"):
                with self.subTest(kind=kind):
                    candidate = directory / {
                        "third": "third_antigravity.json",
                        "python": "authority_antigravity.py",
                        "directory": "nested_antigravity",
                        "symlink": "link_antigravity.json",
                    }[kind]
                    if kind == "directory":
                        candidate.mkdir()
                    elif kind == "symlink":
                        candidate.symlink_to(SMOKE_A_GRANT_PATH_ANTIGRAVITY.name)
                    else:
                        candidate.write_text("{}")
                    with self.assertRaisesRegex(ValueError, "unapproved"):
                        verify_authority_directory_structure_antigravity(directory)
                    if kind == "directory":
                        candidate.rmdir()
                    else:
                        candidate.unlink()

    def test_authority_directory_rejects_unsafe_types_even_with_approved_basenames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            for kind, basename in (
                ("directory", SMOKE_A_GRANT_PATH_ANTIGRAVITY.name),
                ("symlink", SMOKE_A_DIGEST_PATH_ANTIGRAVITY.name),
            ):
                with self.subTest(kind=kind):
                    directory = parent / f"{kind}_authority_directory_antigravity"
                    directory.mkdir()
                    candidate = directory / basename
                    if kind == "directory":
                        candidate.mkdir()
                    else:
                        target = directory / SMOKE_A_GRANT_PATH_ANTIGRAVITY.name
                        target.write_text("{}")
                        candidate.symlink_to(target.name)
                    with self.assertRaisesRegex(ValueError, "unapproved"):
                        verify_authority_directory_structure_antigravity(directory)

    def _git_antigravity(self, repository: Path, *arguments: str) -> str:
        return subprocess.run(
            ("git", *arguments), cwd=repository, check=True, capture_output=True, text=True,
        ).stdout.strip()

    def _authorization_repository_antigravity(
        self, *, extra_authority_path: bool = False, authority_mode: int = 0o644,
        source_delta: tuple[str, bytes] | None = None,
        initial_authority_paths: tuple[str, ...] = (),
        remove_initial_authority_paths: tuple[str, ...] = (),
        grant_payload_transform=None,
        grant_bytes_transform=None,
        digest_payload_transform=None,
        symlink_grant: bool = False,
    ) -> tuple[tempfile.TemporaryDirectory[str], Path, str, str]:
        """Build I -> A from literal Git bytes, never a production authority path."""
        directory = tempfile.TemporaryDirectory(dir=Path.cwd())
        repository = Path(directory.name) / "repository_antigravity"
        repository.mkdir()
        self._git_antigravity(repository, "init", "--quiet")
        self._git_antigravity(repository, "config", "user.email", "fixture@example.invalid")
        self._git_antigravity(repository, "config", "user.name", "fixture")
        (repository / "science_antigravity.py").write_bytes(b"scientific closure I\n")
        (repository / "authority_antigravity.py").write_bytes(b"authority closure I\n")
        lineage = repository / "paper/equipment_anomaly_detection/lineage_lock_paper.json"
        lineage.parent.mkdir(parents=True)
        lineage.write_bytes(b'{"lineage":"I"}\n')
        codemap = repository / "docs/codemap/codemap_exp449_antigravity.lock"
        codemap.parent.mkdir(parents=True)
        codemap.write_bytes(b'{"codemap":"I"}\n')
        if initial_authority_paths:
            initial_authority = repository / "docs/authorities"
            initial_authority.mkdir()
            for relative in initial_authority_paths:
                candidate = initial_authority / relative
                candidate.parent.mkdir(parents=True, exist_ok=True)
                candidate.write_bytes(canonical_grant_bytes_antigravity(self._grant()))
        self._git_antigravity(repository, "add", ".")
        self._git_antigravity(repository, "commit", "--quiet", "-m", "implementation I")
        implementation = self._git_antigravity(repository, "rev-parse", "HEAD")
        if source_delta is not None:
            path, contents = source_delta
            (repository / path).write_bytes(contents)
        grant = replace(
            self._grant(),
            implementation_commit=implementation,
            scientific_code_closure={"science_antigravity.py": sha256(b"scientific closure I\n").hexdigest()},
            authority_code_closure={"authority_antigravity.py": sha256(b"authority closure I\n").hexdigest()},
            lineage_lock_sha256=sha256(b'{"lineage":"I"}\n').hexdigest(),
            codemap_lock_sha256=sha256(b'{"codemap":"I"}\n').hexdigest(),
        )
        grant_bytes = canonical_grant_bytes_antigravity(grant)
        digest = replace(
            self._digest(), implementation_commit=implementation,
            grant_size=len(grant_bytes), grant_sha256=sha256(grant_bytes).hexdigest(),
        )
        authority = repository / "docs/authorities"
        authority.mkdir(exist_ok=True)
        grant_path = authority / SMOKE_A_GRANT_PATH_ANTIGRAVITY.name
        if "renamed_grant_antigravity.json" in initial_authority_paths:
            (authority / "renamed_grant_antigravity.json").replace(grant_path)
        if grant_bytes_transform is not None:
            grant_bytes = grant_bytes_transform(grant_bytes)
            digest = replace(digest, grant_size=len(grant_bytes), grant_sha256=sha256(grant_bytes).hexdigest())
        if grant_payload_transform is not None:
            grant_payload = json.loads(grant_bytes)
            grant_payload_transform(grant_payload)
            grant_bytes = json.dumps(grant_payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            digest = replace(digest, grant_size=len(grant_bytes), grant_sha256=sha256(grant_bytes).hexdigest())
        grant_path.write_bytes(grant_bytes)
        os.chmod(grant_path, authority_mode)
        digest_payload = json.loads(canonical_grant_digest_bytes_antigravity(digest))
        if digest_payload_transform is not None:
            digest_payload_transform(digest_payload)
        digest_path = authority / SMOKE_A_DIGEST_PATH_ANTIGRAVITY.name
        digest_path.write_bytes(json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        if symlink_grant:
            grant_path.unlink()
            grant_path.symlink_to("../../science_antigravity.py")
        for relative in remove_initial_authority_paths:
            (authority / relative).unlink()
        if extra_authority_path:
            (authority / "unexpected_antigravity.json").write_bytes(b"{}\n")
        self._git_antigravity(repository, "add", ".")
        self._git_antigravity(repository, "commit", "--quiet", "-m", "authority A")
        return directory, repository, implementation, self._git_antigravity(repository, "rev-parse", "HEAD")

    def test_authorization_commit_returns_one_immutable_verified_snapshot(self) -> None:
        """Reading authority bytes from a replaced path instead of retained FD bytes accepts mutable evidence."""
        directory, repository, implementation, approved = self._authorization_repository_antigravity()
        self.addCleanup(directory.cleanup)

        verified = verify_smoke_a_authorization_commit_antigravity(approved, repository_root=repository)

        self.assertEqual(verified.approved_authority_commit, approved)
        self.assertEqual(verified.implementation_commit, implementation)
        self.assertEqual(verified.grant.implementation_commit, implementation)
        self.assertEqual(verified.grant_bytes, canonical_grant_bytes_antigravity(verified.grant))
        self.assertEqual(verified.grant_sha256, sha256(verified.grant_bytes).hexdigest())
        self.assertEqual(len(verified.grant_stat_identity), 4)
        self.assertEqual(len(verified.digest_stat_identity), 4)
        with self.assertRaises(FrozenInstanceError):
            verified.implementation_commit = approved  # type: ignore[misc]

    def test_authorization_commit_rejects_wrong_head_dirty_or_non_direct_history(self) -> None:
        """Dropping clean A -> I identity checks allows an unreviewed authority state through."""
        directory, repository, implementation, approved = self._authorization_repository_antigravity()
        self.addCleanup(directory.cleanup)
        self._git_antigravity(repository, "checkout", "--quiet", implementation)
        with self.assertRaises(ValueError):
            verify_smoke_a_authorization_commit_antigravity(approved, repository_root=repository)
        self._git_antigravity(repository, "checkout", "--quiet", approved)
        (repository / "dirty_antigravity.txt").write_text("dirty\n")
        with self.assertRaises(ValueError):
            verify_smoke_a_authorization_commit_antigravity(approved, repository_root=repository)
        (repository / "dirty_antigravity.txt").unlink()
        self._git_antigravity(repository, "commit", "--allow-empty", "--quiet", "-m", "not direct")
        non_direct = self._git_antigravity(repository, "rev-parse", "HEAD")
        with self.assertRaises(ValueError):
            verify_smoke_a_authorization_commit_antigravity(non_direct, repository_root=repository)

    def test_authorization_commit_rejects_non_authority_delta_and_unsafe_authority_mode(self) -> None:
        """Accepting anything besides two added regular authority files expands authorization scope."""
        for arguments in (
            {"extra_authority_path": True},
            {"authority_mode": 0o755},
            {"source_delta": ("science_antigravity.py", b"scientific closure A\n")},
            {"source_delta": ("docs/codemap/codemap_exp449_antigravity.lock", b'{"codemap":"A"}\n')},
        ):
            with self.subTest(arguments=arguments):
                directory, repository, _implementation, approved = self._authorization_repository_antigravity(**arguments)
                self.addCleanup(directory.cleanup)
                with self.assertRaises(ValueError):
                    verify_smoke_a_authorization_commit_antigravity(approved, repository_root=repository)

    def test_authorization_commit_rejects_malformed_identity_and_digest_binding(self) -> None:
        """Ignoring canonical SHA, size, path, or implementation fields detaches a digest from its grant."""
        directory, repository, _implementation, approved = self._authorization_repository_antigravity()
        self.addCleanup(directory.cleanup)
        for invalid in ("A" * 40, "a" * 39, "not-a-commit"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    verify_smoke_a_authorization_commit_antigravity(invalid, repository_root=repository)
        digest_path = repository / "docs/authorities" / SMOKE_A_DIGEST_PATH_ANTIGRAVITY.name
        payload = json.loads(digest_path.read_text())
        payload["grant_sha256"] = "f" * 64
        digest_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        with self.assertRaises(ValueError):
            verify_smoke_a_authorization_commit_antigravity(approved, repository_root=repository)

    def test_authorization_delta_rejects_deleted_modified_renamed_symlinked_and_untracked_authority_paths(self) -> None:
        """Relaxing exact two-file A delta permits a path-level authority substitution."""
        cases = (
            {
                "initial_authority_paths": ("deleted_antigravity.json",),
                "remove_initial_authority_paths": ("deleted_antigravity.json",),
                "message": "delta",
            },
            {
                "initial_authority_paths": (SMOKE_A_GRANT_PATH_ANTIGRAVITY.name, SMOKE_A_DIGEST_PATH_ANTIGRAVITY.name),
                "message": "delta",
            },
            {
                "initial_authority_paths": ("renamed_grant_antigravity.json",),
                "message": "delta",
            },
            {"symlink_grant": True, "message": "authority directory"},
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                directory, repository, implementation, approved = self._authorization_repository_antigravity(
                    **{key: value for key, value in arguments.items() if key != "message"}
                )
                self.addCleanup(directory.cleanup)
                if arguments.get("initial_authority_paths") == ("renamed_grant_antigravity.json",):
                    renamed = self._git_antigravity(repository, "diff-tree", "--no-commit-id", "--name-status", "-r", "-M1%", implementation, approved)
                    self.assertIn("R", renamed)
                with self.assertRaisesRegex(ValueError, arguments["message"]):
                    verify_smoke_a_authorization_commit_antigravity(approved, repository_root=repository)
        directory, repository, _implementation, approved = self._authorization_repository_antigravity()
        self.addCleanup(directory.cleanup)
        (repository / "docs/authorities/untracked_antigravity.json").write_bytes(b"{}\n")
        with self.assertRaisesRegex(ValueError, "clean approved HEAD"):
            verify_smoke_a_authorization_commit_antigravity(approved, repository_root=repository)

    def test_authorization_snapshot_keeps_the_opened_grant_when_its_path_is_swapped(self) -> None:
        """Hashing then reparsing a path would accept a grant replaced between those two reads."""
        directory, repository, _implementation, approved = self._authorization_repository_antigravity()
        self.addCleanup(directory.cleanup)
        grant_path = repository / "docs/authorities" / SMOKE_A_GRANT_PATH_ANTIGRAVITY.name
        original = grant_path.read_bytes()
        replacement = json.loads(original)
        replacement["nonce"] = "f" * 32
        replacement_bytes = json.dumps(replacement, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        real_read = authority_module.os.read
        swapped = False

        def swap_after_first_grant_read(descriptor: int, size: int) -> bytes:
            nonlocal swapped
            value = real_read(descriptor, size)
            if not swapped and value == original:
                replacement_path = grant_path.with_name("replacement_grant_antigravity.json")
                replacement_path.write_bytes(replacement_bytes)
                os.replace(replacement_path, grant_path)
                swapped = True
            return value

        with patch.object(authority_module.os, "read", side_effect=swap_after_first_grant_read):
            verified = verify_smoke_a_authorization_commit_antigravity(approved, repository_root=repository)
        self.assertTrue(swapped)
        self.assertEqual(verified.grant_bytes, original)
        self.assertEqual(grant_path.read_bytes(), replacement_bytes)

    def test_authorization_parser_reaches_noncanonical_digest_and_closure_lock_gates(self) -> None:
        """A valid A delta must still reject each independent retained authority binding."""
        cases = (
            {"grant_bytes_transform": lambda value: value + b"\n", "message": "not canonical"},
            {"digest_payload_transform": lambda value: value.__setitem__("grant_size", value["grant_size"] + 1), "message": "does not bind"},
            {"digest_payload_transform": lambda value: value.__setitem__("grant_path", "/tmp/wrong_antigravity.json"), "message": "digest is invalid"},
            {"digest_payload_transform": lambda value: value.__setitem__("implementation_commit", "e" * 40), "message": "does not bind"},
            {"grant_payload_transform": lambda value: value["authority_code_closure"].__setitem__("authority_antigravity.py", "e" * 64), "message": "bound closure"},
            {"grant_payload_transform": lambda value: value.__setitem__("lineage_lock_sha256", "e" * 64), "message": "bound closure or lock"},
        )
        for arguments in cases:
            with self.subTest(message=arguments["message"]):
                directory, repository, _implementation, approved = self._authorization_repository_antigravity(
                    **{key: value for key, value in arguments.items() if key != "message"}
                )
                self.addCleanup(directory.cleanup)
                with self.assertRaisesRegex(ValueError, arguments["message"]):
                    verify_smoke_a_authorization_commit_antigravity(approved, repository_root=repository)
