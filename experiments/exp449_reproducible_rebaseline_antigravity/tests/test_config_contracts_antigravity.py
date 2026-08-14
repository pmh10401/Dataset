"""Exp449 version-boundary tests with an immutable Exp448 byte inventory."""

from __future__ import annotations

import json
import ast
from hashlib import sha256
from pathlib import Path
import subprocess
import unittest
from dataclasses import FrozenInstanceError, fields, replace
from types import MappingProxyType

from experiments.exp449_reproducible_rebaseline_antigravity.config_antigravity import (
    FROZEN_EXP449_CONFIG,
    FrozenExp449Config,
)
from experiments.exp449_reproducible_rebaseline_antigravity.contracts_antigravity import (
    Exp449Context,
    Exp449ProposalManifest,
    Exp449ShardDescriptor,
    Exp449TerminalStatus,
    SmokeComparison,
)
from paper.equipment_anomaly_detection.contracts_paper import FrozenTask8BackendDecision


REPO_ROOT = Path(__file__).resolve().parents[3]
EXP448_DESIGN_COMMIT = "26830622"
EXP448_FROZEN_PATHS = (
    "experiments/exp448_exp447_text_label_boundary_repair_antigravity",
    "docs/codemap/codemap_exp448_antigravity.html",
    "docs/codemap/codemap_exp448_antigravity.json",
    "docs/codemap/codemap_exp448_antigravity.lock",
    "experiments/exp448_exp447_text_label_boundary_repair_antigravity/tests/"
    "test_paper_r4_full_route_parity_recovery_antigravity.py",
)


def _git_stdout(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _blob_inventory_at(ref: str) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for line in _git_stdout("ls-tree", "-r", "--full-tree", ref, "--", *EXP448_FROZEN_PATHS).splitlines():
        metadata, path = line.split("\t", 1)
        _mode, object_type, blob_hash = metadata.split()
        if object_type == "blob":
            inventory[path] = blob_hash
    return dict(sorted(inventory.items()))


def exp448_frozen_blob_inventory_at_design_commit(ref: str) -> dict[str, str]:
    return _blob_inventory_at(ref)


def current_exp448_blob_inventory() -> dict[str, str]:
    tracked_paths = set(
        _git_stdout("ls-files", "--cached", "--", *EXP448_FROZEN_PATHS).splitlines()
    )
    untracked_paths = set(
        _git_stdout("ls-files", "--others", "--exclude-standard", "--", *EXP448_FROZEN_PATHS).splitlines()
    )
    inventory: dict[str, str] = {}
    for path in sorted(tracked_paths | untracked_paths):
        candidate = REPO_ROOT / path
        if candidate.is_file():
            inventory[path] = _git_stdout("hash-object", str(candidate)).strip()
    return inventory


class Exp449VersionBoundaryTests(unittest.TestCase):
    def test_exp449_state_and_triplet_exist_without_mutating_exp448(self) -> None:
        state = json.loads((REPO_ROOT / "RESEARCH_STATE.json").read_text())
        self.assertEqual(state["active_experiment"], "Exp449")
        self.assertEqual(state["status"], "implementation_verified_awaiting_smoke_authorization")
        exp449 = state["exp449_reproducible_rebaseline_antigravity"]
        self.assertEqual(exp449["status"], "implementation_verified_awaiting_smoke_authorization")
        self.assertEqual(
            exp449["next_action"],
            "design_and_review_trusted_smoke_a_execution_authority_then_request_user_approval",
        )
        self.assertFalse(exp449["smoke_started"])
        self.assertFalse(exp449["official_run_started"])
        self.assertFalse(exp449["test_labels_opened"])
        self.assertEqual(exp449["official_label_access_count"], 0)
        self.assertEqual(
            state["active_documents"]["preregistration"],
            "docs/EXP449_PREREGISTRATION_antigravity_KO.md",
        )
        self.assertEqual(
            state["active_documents"]["runbook"],
            "docs/EXP449_RUNBOOK_antigravity_KO.md",
        )
        self.assertEqual(
            state["active_documents"]["implementation_report"],
            "docs/EXP449_IMPLEMENTATION_REPORT_antigravity.md",
        )
        for root in (
            FROZEN_EXP449_CONFIG.smoke_a_scientific_root,
            FROZEN_EXP449_CONFIG.smoke_a_operations_root,
            FROZEN_EXP449_CONFIG.smoke_b_scientific_root,
            FROZEN_EXP449_CONFIG.smoke_b_operations_root,
            FROZEN_EXP449_CONFIG.official_scientific_root,
            FROZEN_EXP449_CONFIG.official_operations_root,
        ):
            self.assertFalse(root.exists(), f"Exp449 root must remain absent: {root}")
        for suffix in ("html", "json", "lock"):
            self.assertTrue(
                (REPO_ROOT / f"docs/codemap/codemap_exp449_antigravity.{suffix}").is_file()
            )
        for document in (
            "docs/EXP449_PREREGISTRATION_antigravity_KO.md",
            "docs/EXP449_RUNBOOK_antigravity_KO.md",
            "docs/EXP449_IMPLEMENTATION_REPORT_antigravity.md",
        ):
            self.assertTrue((REPO_ROOT / document).is_file(), document)
        frozen = exp448_frozen_blob_inventory_at_design_commit(EXP448_DESIGN_COMMIT)
        self.assertTrue(frozen)
        self.assertEqual(current_exp448_blob_inventory(), frozen)

    def test_exp449_codemap_python_symbols_and_test_paths_are_real(self) -> None:
        """A final map must point to defining files, not planning-era aliases or phantom symbols."""
        payload = json.loads((REPO_ROOT / "docs/codemap/codemap_exp449_antigravity.json").read_text())
        for node in payload["nodes"]:
            path = REPO_ROOT / node["path"]
            self.assertTrue(path.is_file(), node["path"])
            for test_path in node.get("tests", []):
                self.assertTrue((REPO_ROOT / test_path).is_file(), test_path)
            if path.suffix != ".py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            symbols = {
                child.name for child in tree.body
                if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            }
            symbols.update(
                target.id for child in tree.body
                if isinstance(child, (ast.Assign, ast.AnnAssign))
                for target in (child.targets if isinstance(child, ast.Assign) else (child.target,))
                if isinstance(target, ast.Name)
            )
            for entrypoint in node.get("entrypoints", []):
                self.assertIn(entrypoint, symbols, f"{node['path']}:{entrypoint}")

    def test_every_implemented_exp449_codemap_node_has_current_semantics(self) -> None:
        """Task1--8 nodes must answer callers, effects, and coverage without stale planning language."""
        payload = json.loads((REPO_ROOT / "docs/codemap/codemap_exp449_antigravity.json").read_text())
        forbidden = ("future", "planned", "implementation planning only", "placeholder")
        nodes = [node for node in payload["nodes"] if node["id"].startswith("exp449_")]
        self.assertTrue(nodes)
        for node in nodes:
            serialized = json.dumps(node, ensure_ascii=False).casefold()
            self.assertFalse(any(term in serialized for term in forbidden), node["id"])
            self.assertTrue(node.get("callers"), f"missing callers: {node['id']}")
            self.assertTrue(node.get("effects"), f"missing effects: {node['id']}")
            self.assertTrue(node.get("tests"), f"missing tests: {node['id']}")
            for test_path in node["tests"]:
                self.assertTrue((REPO_ROOT / test_path).is_file(), test_path)


class Exp449FrozenConfigTests(unittest.TestCase):
    def test_frozen_config_is_exact_and_uses_only_new_absent_roots(self) -> None:
        self.assertIsInstance(FROZEN_EXP449_CONFIG, FrozenExp449Config)
        self.assertEqual(FROZEN_EXP449_CONFIG.experiment_id, "Exp449")
        self.assertEqual(FROZEN_EXP449_CONFIG.dataset_count, 1117)
        self.assertEqual(FROZEN_EXP449_CONFIG.test_row_count, 195900)
        self.assertEqual(FROZEN_EXP449_CONFIG.policy_id, "hard_plus_stable")
        self.assertEqual(FROZEN_EXP449_CONFIG.cpu_repeat_tolerance, 1e-12)
        self.assertEqual(FROZEN_EXP449_CONFIG.accelerated_tolerance, 1e-5)
        self.assertEqual(FROZEN_EXP449_CONFIG.minimum_accelerated_speedup, 1.1)
        self.assertEqual(FROZEN_EXP449_CONFIG.maximum_waves, 2)
        self.assertEqual(
            FROZEN_EXP449_CONFIG.preprocessing_contracts,
            (
                "task4_train_only",
                "task5_label_free_train_test_value_length_transductive",
            ),
        )
        self.assertEqual(len(FROZEN_EXP449_CONFIG.smoke_datasets), 23)
        self.assertIn("MelbournePedestrian_normal_10", FROZEN_EXP449_CONFIG.smoke_datasets)
        self.assertEqual(
            FROZEN_EXP449_CONFIG.smoke_datasets,
            tuple(sorted(FROZEN_EXP449_CONFIG.smoke_datasets)),
        )
        self.assertEqual(len(set(FROZEN_EXP449_CONFIG.smoke_datasets)), 23)
        self.assertEqual(
            tuple(field.name for field in fields(FrozenExp449Config)),
            (
                "experiment_id",
                "dataset_count",
                "test_row_count",
                "policy_id",
                "cpu_repeat_tolerance",
                "accelerated_tolerance",
                "minimum_accelerated_speedup",
                "maximum_waves",
                "preprocessing_contracts",
                "smoke_datasets",
                "smoke_a_scientific_root",
                "smoke_a_operations_root",
                "smoke_b_scientific_root",
                "smoke_b_operations_root",
                "official_scientific_root",
                "official_operations_root",
            ),
        )
        self.assertEqual(
            FROZEN_EXP449_CONFIG.smoke_a_scientific_root,
            Path("/Users/mac/Documents/Dataset/paper_proposal_exp449_smoke_a_antigravity_20260814"),
        )
        self.assertEqual(
            FROZEN_EXP449_CONFIG.smoke_a_operations_root,
            Path("/Users/mac/Documents/Dataset/paper_proposal_exp449_smoke_a_antigravity_20260814_ops_antigravity"),
        )
        self.assertEqual(
            FROZEN_EXP449_CONFIG.smoke_b_scientific_root,
            Path("/Users/mac/Documents/Dataset/paper_proposal_exp449_smoke_b_antigravity_20260814"),
        )
        self.assertEqual(
            FROZEN_EXP449_CONFIG.smoke_b_operations_root,
            Path("/Users/mac/Documents/Dataset/paper_proposal_exp449_smoke_b_antigravity_20260814_ops_antigravity"),
        )
        self.assertEqual(
            FROZEN_EXP449_CONFIG.official_scientific_root,
            Path("/Users/mac/Documents/Dataset/paper_proposal_exp449_official_antigravity_20260814"),
        )
        self.assertEqual(
            FROZEN_EXP449_CONFIG.official_operations_root,
            Path("/Users/mac/Documents/Dataset/paper_proposal_exp449_official_antigravity_20260814_ops_antigravity"),
        )
        with self.assertRaises(FrozenInstanceError):
            FROZEN_EXP449_CONFIG.policy_id = "mutable"  # type: ignore[misc]
        with self.assertRaises(ValueError):
            replace(
                FROZEN_EXP449_CONFIG,
                smoke_a_scientific_root=Path("/tmp/not_an_exp449_root_antigravity"),
            )


class Exp449ImmutableContractTests(unittest.TestCase):
    _SHA_A = "a" * 64
    _SHA_B = "b" * 64
    _SHA_C = "c" * 64
    _SHA_D = "d" * 64
    _SHA_E = "e" * 64
    _SHA_F = "f" * 64

    def _backend(self) -> FrozenTask8BackendDecision:
        return FrozenTask8BackendDecision(
            selected_backend="cpu",
            fallback_reason="MLX parity benchmark not yet approved",
            assessment_sha256=self._SHA_A,
            logical_cpu_count=2,
            dataset_workers=1,
            inner_numerical_threads=1,
            outer_execution_mode="parallel_task4_task5",
        )

    def _policy_coordinate_digests(self) -> dict[str, str]:
        return {
            "hard_only": self._SHA_A,
            "hard_plus_source": self._SHA_B,
            "hard_plus_stable": self._SHA_C,
            "hard_plus_unstable": self._SHA_D,
        }

    def _context(self, *, run_root: Path | None = None, mode: str = "smoke") -> Exp449Context:
        bare_catalog = {
            "schema": "paper-task8-catalog.v1",
            "mode": mode,
            "database": {"fixture": "contracts"},
            "datasets": [{"dataset": "Dataset_normal_1", "raw_sha256": self._SHA_B, "rows": [1, 2]}],
        }
        catalog = {
            **bare_catalog,
            "catalog_sha256": sha256(json.dumps(bare_catalog, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode() + b"\n").hexdigest(),
        }
        return Exp449Context.from_verified_inputs(
            run_root=(run_root or FROZEN_EXP449_CONFIG.smoke_a_scientific_root),
            mode=mode,
            database_sha256=self._SHA_A,
            catalog=catalog,
            code_closure={"module_antigravity.py": self._SHA_C},
            lineage_sha256=self._SHA_D,
            codemap_sha256=self._SHA_E,
            interpreter="/usr/bin/python3",
            environment={"python": "3.14.6", "numpy": "2.3.1"},
            acceleration=self._backend(),
            seeds_and_constants={"seed": 449, "alphas": (0.01, 0.05)},
        )

    def test_context_rejects_historical_parity_authority_before_other_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "historical authority"):
            Exp449Context.from_verified_inputs(
                run_root=Path("/tmp/exp449_fixture_antigravity"),
                mode="smoke",
                catalog={},
                acceleration={},
                acceptance_path=Path("old.json"),
            )

    def test_context_copies_nested_provenance_and_has_exact_schema(self) -> None:
        context = self._context()
        self.assertEqual(
            tuple(field.name for field in fields(Exp449Context)),
            (
                "experiment_id",
                "mode",
                "run_root",
                "database_sha256",
                "catalog_sha256",
                "code_closure",
                "lineage_sha256",
                "codemap_sha256",
                "interpreter",
                "environment",
                "backend_record",
                "seeds_and_constants",
                "historical_parity_status",
            ),
        )
        self.assertEqual(context.experiment_id, "Exp449")
        self.assertEqual(context.mode, "smoke")
        self.assertEqual(
            context.run_root,
            FROZEN_EXP449_CONFIG.smoke_a_scientific_root.resolve(),
        )
        self.assertEqual(context.historical_parity_status, "not_used")
        self.assertIsInstance(context.code_closure, MappingProxyType)
        self.assertIsInstance(context.code_closure["module_antigravity.py"], str)
        self.assertIsInstance(context.environment, MappingProxyType)
        self.assertIsInstance(context.seeds_and_constants, MappingProxyType)
        self.assertEqual(context.seeds_and_constants["alphas"], (0.01, 0.05))
        with self.assertRaises(TypeError):
            context.environment["python"] = "mutable"  # type: ignore[index]
        with self.assertRaises(ValueError):
            Exp449Context.from_verified_inputs(
                run_root=Path("/tmp/exp449_contract_fixture_antigravity"),
                mode="smoke",
                database_sha256="A" * 64,
                catalog={"schema": "paper-task8-catalog.v1", "mode": "smoke", "database": {}, "datasets": [], "catalog_sha256": "0" * 64},
                code_closure={"module_antigravity.py": self._SHA_C},
                lineage_sha256=self._SHA_D,
                codemap_sha256=self._SHA_E,
                interpreter="/usr/bin/python3",
                environment={"python": "3.14.6"},
                acceleration=self._backend(),
                seeds_and_constants={"seed": 449},
            )

    def test_context_rejects_alternate_or_mode_incompatible_frozen_roots(self) -> None:
        with self.assertRaisesRegex(ValueError, "run root"):
            self._context(run_root=Path("/tmp/alternate_absolute_root_antigravity"))
        with self.assertRaisesRegex(ValueError, "run root"):
            self._context(
                run_root=FROZEN_EXP449_CONFIG.official_scientific_root,
                mode="smoke",
            )
        self.assertEqual(
            self._context(
                run_root=FROZEN_EXP449_CONFIG.official_operations_root,
                mode="official",
            ).run_root,
            FROZEN_EXP449_CONFIG.official_operations_root.resolve(),
        )

    def test_manifest_terminal_and_smoke_contracts_are_exact_and_fail_closed(self) -> None:
        context = self._context()
        manifest = Exp449ProposalManifest(
            schema="exp449-proposal-manifest-antigravity.v1",
            context_sha256=self._SHA_A,
            catalog_sha256=context.catalog_sha256,
            backend_record=self._backend().artifact_record(),
            entries=(
                Exp449ShardDescriptor(
                    dataset="Dataset_normal_1",
                    relative_path="shards/Dataset_normal_1_antigravity.json",
                    size=17,
                    sha256=self._SHA_B,
                    schema="paper-proposal-typed-shard.v1",
                    policy_coordinate_sha256=self._policy_coordinate_digests(),
                    candidate_c_score_sha256=self._SHA_D,
                ),
            ),
            historical_parity_status="not_used",
        )
        self.assertEqual(
            tuple(field.name for field in fields(Exp449ProposalManifest)),
            (
                "schema",
                "context_sha256",
                "catalog_sha256",
                "backend_record",
                "entries",
                "historical_parity_status",
                "task45_checkpoints",
            ),
        )
        self.assertEqual(manifest.schema, "exp449-proposal-manifest-antigravity.v1")
        self.assertIsInstance(manifest.entries, tuple)
        self.assertEqual(manifest.entries[0].size, 17)
        with self.assertRaises(ValueError):
            Exp449ShardDescriptor(
                dataset="Dataset_normal_1",
                relative_path="shards/Dataset_normal_1_antigravity.json",
                size=-1,
                sha256="not-a-sha",
                schema="paper-proposal-typed-shard.v1",
                policy_coordinate_sha256=self._policy_coordinate_digests(),
                candidate_c_score_sha256=self._SHA_D,
            )
        status = Exp449TerminalStatus(
            mode="official",
            status="proposal_score_incomplete",
            completed_dataset_count=1116,
            failed_dataset_count=1,
            pending_dataset_count=0,
            wave_count=2,
        )
        self.assertEqual(status.status, "proposal_score_incomplete")
        with self.assertRaises(ValueError):
            Exp449TerminalStatus("official", "dataset_local_complete", 0, 0, 0, 0)
        comparison = SmokeComparison(
            smoke_a_manifest_sha256=self._SHA_A,
            smoke_b_manifest_sha256=self._SHA_B,
            compared_dataset_count=23,
            coordinate_identity=True,
            maximum_score_delta=0.0,
            passed=True,
        )
        self.assertTrue(comparison.passed)
        with self.assertRaises(ValueError):
            SmokeComparison(self._SHA_A, self._SHA_B, 23, True, float("nan"), False)

    def test_manifest_rejects_schema_and_entry_digest_closure_mismatches(self) -> None:
        entry = Exp449ShardDescriptor(
            dataset="Dataset_normal_1",
            relative_path="shards/Dataset_normal_1_antigravity.json",
            size=17,
            sha256=self._SHA_B,
            schema="paper-proposal-typed-shard.v1",
            policy_coordinate_sha256=self._policy_coordinate_digests(),
            candidate_c_score_sha256=self._SHA_D,
        )
        with self.assertRaisesRegex(ValueError, "manifest schema"):
            Exp449ProposalManifest(
                schema="exp449_proposal_manifest_v1",
                context_sha256=self._SHA_A,
                catalog_sha256=self._SHA_B,
                backend_record=self._backend().artifact_record(),
                entries=(entry,),
                historical_parity_status="not_used",
            )
        with self.assertRaisesRegex(ValueError, "coordinate"):
            Exp449ShardDescriptor(
                dataset="Dataset_normal_1",
                relative_path="shards/Dataset_normal_1_antigravity.json",
                size=17,
                sha256=self._SHA_B,
                schema="paper-proposal-typed-shard.v1",
                policy_coordinate_sha256={
                    "hard_only": self._SHA_A,
                    "hard_plus_source": self._SHA_B,
                    "hard_plus_stable": self._SHA_C,
                },
                candidate_c_score_sha256=self._SHA_D,
            )
        with self.assertRaisesRegex(ValueError, "ordered"):
            Exp449ProposalManifest(
                schema="exp449-proposal-manifest-antigravity.v1",
                context_sha256=self._SHA_A,
                catalog_sha256=self._SHA_B,
                backend_record=self._backend().artifact_record(),
                entries=(
                    Exp449ShardDescriptor(
                        dataset="Z_normal_1",
                        relative_path="shards/Z_normal_1_antigravity.json",
                        size=17,
                        sha256=self._SHA_B,
                        schema="paper-proposal-typed-shard.v1",
                        policy_coordinate_sha256=self._policy_coordinate_digests(),
                        candidate_c_score_sha256=self._SHA_D,
                    ),
                    entry,
                ),
                historical_parity_status="not_used",
            )

    def test_terminal_completion_and_smoke_pass_require_exact_inventory_counts(self) -> None:
        self.assertEqual(
            Exp449TerminalStatus("smoke", "dataset_local_complete", 23, 0, 0, 1).completed_dataset_count,
            23,
        )
        self.assertEqual(
            Exp449TerminalStatus("official", "dataset_local_complete", 1117, 0, 0, 2).completed_dataset_count,
            1117,
        )
        with self.assertRaisesRegex(ValueError, "completion"):
            Exp449TerminalStatus("smoke", "dataset_local_complete", 0, 0, 0, 0)
        with self.assertRaisesRegex(ValueError, "completion"):
            Exp449TerminalStatus("official", "dataset_local_complete", 23, 0, 0, 1)
        self.assertEqual(
            Exp449TerminalStatus("smoke", "proposal_score_incomplete", 22, 1, 0, 1).status,
            "proposal_score_incomplete",
        )
        with self.assertRaisesRegex(ValueError, "total"):
            Exp449TerminalStatus("smoke", "proposal_score_incomplete", 0, 0, 0, 0)
        with self.assertRaisesRegex(ValueError, "total"):
            Exp449TerminalStatus("official", "proposal_score_incomplete", 1116, 0, 0, 2)
        with self.assertRaisesRegex(ValueError, "23"):
            SmokeComparison(self._SHA_A, self._SHA_B, 0, True, 0.0, True)

    def test_manifest_accepts_distinct_per_dataset_coordinate_digests_and_rejects_key_drift(self) -> None:
        first = Exp449ShardDescriptor(
            dataset="A_normal_1",
            relative_path="shards/A_normal_1_antigravity.json",
            size=17,
            sha256=self._SHA_A,
            schema="paper-proposal-typed-shard.v1",
            policy_coordinate_sha256=self._policy_coordinate_digests(),
            candidate_c_score_sha256=self._SHA_E,
        )
        second_coordinates = {
            "hard_only": "0" * 64,
            "hard_plus_source": "1" * 64,
            "hard_plus_stable": "2" * 64,
            "hard_plus_unstable": "3" * 64,
        }
        second = Exp449ShardDescriptor(
            dataset="B_normal_1",
            relative_path="shards/B_normal_1_antigravity.json",
            size=23,
            sha256=self._SHA_B,
            schema="paper-proposal-typed-shard.v1",
            policy_coordinate_sha256=second_coordinates,
            candidate_c_score_sha256=self._SHA_F,
        )
        manifest = Exp449ProposalManifest(
            schema="exp449-proposal-manifest-antigravity.v1",
            context_sha256=self._SHA_C,
            catalog_sha256=self._SHA_D,
            backend_record=self._backend().artifact_record(),
            entries=(first, second),
            historical_parity_status="not_used",
        )
        self.assertEqual(manifest.entries[0].policy_coordinate_sha256, self._policy_coordinate_digests())
        self.assertEqual(manifest.entries[1].policy_coordinate_sha256, second_coordinates)
        with self.assertRaisesRegex(ValueError, "coordinate"):
            Exp449ShardDescriptor(
                dataset="C_normal_1",
                relative_path="shards/C_normal_1_antigravity.json",
                size=17,
                sha256=self._SHA_C,
                schema="paper-proposal-typed-shard.v1",
                policy_coordinate_sha256={
                    "hard_only": self._SHA_A,
                    "hard_plus_source": self._SHA_B,
                    "hard_plus_stable": self._SHA_C,
                },
                candidate_c_score_sha256=self._SHA_D,
            )
        with self.assertRaisesRegex(ValueError, "coordinate"):
            Exp449ShardDescriptor(
                dataset="D_normal_1",
                relative_path="shards/D_normal_1_antigravity.json",
                size=17,
                sha256=self._SHA_D,
                schema="paper-proposal-typed-shard.v1",
                policy_coordinate_sha256={
                    **self._policy_coordinate_digests(),
                    "unexpected_policy": self._SHA_E,
                },
                candidate_c_score_sha256=self._SHA_F,
            )
