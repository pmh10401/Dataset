"""Pure, fixture-only contracts for the deferred Exp449 Smoke A authority."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import json
from pathlib import Path
from types import MappingProxyType
import tempfile
import unittest

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
    verify_authority_directory_structure_antigravity,
)


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
