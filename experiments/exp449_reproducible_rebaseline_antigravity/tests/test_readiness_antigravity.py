"""Behavioral Task 7 tests for Exp449's read-only execution gate."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.exp449_reproducible_rebaseline_antigravity import operations_antigravity as operations
from experiments.exp449_reproducible_rebaseline_antigravity.config_antigravity import FROZEN_EXP449_CONFIG


def _catalog_antigravity(mode: str) -> dict[str, object]:
    """A hand-built, read-only catalog identity; production derives this from SQLite."""
    names = FROZEN_EXP449_CONFIG.smoke_datasets if mode == "smoke" else ("fixture_full_antigravity",) * 1117
    if mode == "full":
        names = tuple(f"fixture_{index:04d}_antigravity" for index in range(1117))
    bare = {
        "schema": "paper-task8-catalog.v1", "mode": "smoke" if mode == "smoke" else "full",
        "database": {"unresolved_path": "/fixture/database_antigravity.sqlite", "sha256": "a" * 64},
        "datasets": [
            {"dataset": name, "test": {"count": 1 if mode == "smoke" else 175}}
            for name in names
        ],
    }
    if mode == "full":
        bare["datasets"][-1]["test"]["count"] = 195900 - 175 * 1116
    encoded = json.dumps(bare, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    return {**bare, "catalog_sha256": sha256(encoded).hexdigest()}


class Exp449ReadinessTests(unittest.TestCase):
    """Each test names the broken operational boundary it prevents."""

    def setUp(self) -> None:
        self.directory = TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.directory.name)
        self.roots = tuple(self.root / f"root_{index}_antigravity" for index in range(6))

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _ready(self, *, logical_cpu_count: int = 12) -> dict[str, object]:
        with patch.object(operations, "_current_clean_head_antigravity", return_value="c" * 40), \
             patch.object(operations, "_require_capable_interpreter_antigravity", return_value={"executable": "fixture-python", "python_version": "3.14.6"}), \
             patch.object(operations, "_readiness_catalogs_antigravity", return_value={"smoke": _catalog_antigravity("smoke"), "official": _catalog_antigravity("full")}), \
             patch.object(operations, "_dependency_facts_antigravity", return_value={"pycatch22": "0.5.0", "aeon": "1.5.0", "hydra": "constructed", "torch": "2.13.0"}), \
             patch.object(operations, "_fixed_roots_antigravity", return_value=self.roots):
            return operations._readiness_facts_antigravity("c" * 40, logical_cpu_count=logical_cpu_count)

    def test_readiness_records_exact_cpu_policy_acceleration_and_distinct_preprocessing(self) -> None:
        """Changing workers, dropping a backend fact, or collapsing preprocessing must fail readiness."""
        facts = self._ready()
        self.assertEqual(facts["workers"], {"logical_cpu_count": 12, "dataset_workers": 11})
        self.assertEqual(facts["preprocessing_contracts"], [
            "task4_train_only", "task5_label_free_train_test_value_length_transductive",
        ])
        self.assertEqual(facts["acceleration"]["canonical_backend"], "cpu")
        self.assertEqual({item["family"] for item in facts["acceleration"]["backends"]}, {"pytorch", "mlx", "mps"})
        self.assertIn("transfer", facts["acceleration"]["benchmark_requirement"])
        self.assertIn("compilation", facts["acceleration"]["benchmark_requirement"])
        self.assertIn("synchronization", facts["acceleration"]["benchmark_requirement"])
        self.assertTrue(facts["acceleration"]["one_heavy_process_only"])
        self.assertTrue(facts["acceleration"]["no_outer_pool_with_heavy_process"])

    def test_readiness_never_creates_any_absent_fixed_root(self) -> None:
        """Replacing lstat inspection with mkdir would turn a review into an execution side effect."""
        self._ready()
        self.assertTrue(all(not root.exists() and not root.is_symlink() for root in self.roots))

    def test_root_traversal_symlink_and_historical_destinations_are_refused(self) -> None:
        """Resolving before checking or accepting an old destination could redirect a future run."""
        with self.assertRaisesRegex(ValueError, "fixed roots"):
            operations._inspect_fixed_roots_antigravity((self.root / "../escape_antigravity",) * 6)
        target = self.root / "target_antigravity"
        target.mkdir()
        link = self.root / "linked_antigravity"
        link.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            operations._inspect_fixed_roots_antigravity((link, *self.roots[1:]))
        for token in ("exp448", "r4", "historical"):
            roots = list(self.roots)
            roots[0] = self.root / f"{token}_destination_antigravity"
            with self.assertRaisesRegex(ValueError, "historical"):
                operations._inspect_fixed_roots_antigravity(tuple(roots))

    def test_dirty_or_nonmatching_reviewed_commit_fails_before_input_inspection(self) -> None:
        """A stale review must not be able to inspect a database or future root state."""
        with patch.object(operations, "_current_clean_head_antigravity", return_value="d" * 40), \
             patch.object(operations, "_readiness_catalogs_antigravity", side_effect=AssertionError("catalog read")):
            with self.assertRaisesRegex(ValueError, "reviewed commit"):
                operations.readiness_exp449_antigravity("c" * 40)
        with self.assertRaisesRegex(ValueError, "reviewed commit"):
            operations._require_reviewed_head_antigravity("C" * 40)

    def test_unpinned_interpreter_or_dependency_preflight_blocks_readiness(self) -> None:
        """A different interpreter or optional-dependency substitution changes frozen scientific behavior."""
        with patch.object(operations, "_current_clean_head_antigravity", return_value="c" * 40), \
             patch.object(operations, "_require_capable_interpreter_antigravity", side_effect=ValueError("capable interpreter differs")):
            with self.assertRaisesRegex(ValueError, "capable interpreter"):
                operations.readiness_exp449_antigravity("c" * 40)
        with patch.object(operations, "_current_clean_head_antigravity", return_value="c" * 40), \
             patch.object(operations, "_require_capable_interpreter_antigravity", return_value={"executable": "fixture-python", "python_version": "3.14.6"}), \
             patch.object(operations, "_readiness_catalogs_antigravity", return_value={"smoke": _catalog_antigravity("smoke"), "official": _catalog_antigravity("full")}), \
             patch.object(operations, "_dependency_facts_antigravity", side_effect=ValueError("dependency unavailable:pycatch22==0.5.0")):
            with self.assertRaisesRegex(ValueError, "pycatch22"):
                operations.readiness_exp449_antigravity("c" * 40)

    def test_label_firewall_blocks_loaded_label_module_without_scoring(self) -> None:
        """A loaded label accessor would invalidate a supposedly values-only review."""
        module_name = "paper.equipment_anomaly_detection.label_access_paper"
        sentinel = object()
        with patch.dict(sys.modules, {module_name: sentinel}):
            with self.assertRaisesRegex(ValueError, "label module"):
                self._ready()

    def test_context_collapse_and_scoring_invocation_are_rejected(self) -> None:
        """Readiness must preserve both contracts and never reach the scoring seam."""
        with patch.object(operations, "compose_exp449_proposal_score_antigravity", side_effect=AssertionError("scoring invoked")), \
             patch.object(operations, "FROZEN_EXP449_CONFIG", operations.FROZEN_EXP449_CONFIG):
            self._ready()
        with patch.object(operations, "_preprocessing_contracts_antigravity", return_value=("task4_train_only", "task4_train_only")):
            with self.assertRaisesRegex(ValueError, "preprocessing"):
                self._ready()

    def test_read_only_status_reports_absent_roots_without_creating_them(self) -> None:
        """A status probe must not accidentally initialize an operations namespace."""
        with patch.object(operations, "_fixed_roots_antigravity", return_value=self.roots):
            result = operations.status_exp449_read_only_antigravity()
        self.assertEqual(result["status"], "not_started")
        self.assertTrue(all(item["state"] == "absent" for item in result["roots"]))
        self.assertTrue(all(not root.exists() for root in self.roots))

    def test_execution_subcommands_require_separate_authority_before_any_root_creation(self) -> None:
        """A CLI run command without the create-only authority cannot cross into execution."""
        with patch.object(operations, "_current_clean_head_antigravity", return_value="c" * 40), \
             patch.object(operations, "_fixed_roots_antigravity", return_value=self.roots):
            with self.assertRaisesRegex(ValueError, "execution authority"):
                operations._require_execution_authority_antigravity("smoke_a", "c" * 40)
        self.assertTrue(all(not root.exists() for root in self.roots))
        target = self.root / "authority_target_antigravity"
        target.mkdir()
        linked_operations = self.root / "linked_operations_antigravity"
        linked_operations.symlink_to(target, target_is_directory=True)
        hostile_roots = list(self.roots)
        hostile_roots[1] = linked_operations
        with patch.object(operations, "_current_clean_head_antigravity", return_value="c" * 40), \
             patch.object(operations, "_fixed_roots_antigravity", return_value=tuple(hostile_roots)):
            with self.assertRaisesRegex(ValueError, "symlink"):
                operations._require_execution_authority_antigravity("smoke_a", "c" * 40)

    def test_launcher_import_is_inactive_and_status_is_read_only(self) -> None:
        """Moving the operations import above __main__ would make a harmless launcher import active."""
        launcher = Path("scratch/run_exp449_reproducible_rebaseline_antigravity.py")
        imported = subprocess.run(
            [sys.executable, "-c", "import importlib,sys; importlib.import_module('scratch.run_exp449_reproducible_rebaseline_antigravity'); assert 'experiments.exp449_reproducible_rebaseline_antigravity.operations_antigravity' not in sys.modules"],
            cwd=Path.cwd(), capture_output=True, text=True,
        )
        self.assertEqual(imported.returncode, 0, imported.stderr)
        before = tuple(root.exists() or root.is_symlink() for root in operations._fixed_roots_antigravity())
        status = subprocess.run([sys.executable, str(launcher), "status"], cwd=Path.cwd(), capture_output=True, text=True)
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertEqual(json.loads(status.stdout)["status"], "not_started")
        self.assertEqual(tuple(root.exists() or root.is_symlink() for root in operations._fixed_roots_antigravity()), before)
