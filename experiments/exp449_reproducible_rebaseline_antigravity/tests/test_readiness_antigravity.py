"""Behavioral Task 7 tests for Exp449's read-only execution gate."""

from __future__ import annotations

from contextlib import nullcontext, redirect_stderr
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.exp449_reproducible_rebaseline_antigravity import operations_antigravity as operations
from experiments.exp449_reproducible_rebaseline_antigravity.config_antigravity import FROZEN_EXP449_CONFIG, SMOKE_A_CLAIM_PATH_ANTIGRAVITY
from experiments.exp449_reproducible_rebaseline_antigravity.execution_authority_antigravity import Exp449SmokeAGrantAntigravity


def _catalog_antigravity(mode: str) -> dict[str, object]:
    """A hand-built, read-only catalog identity; production derives this from SQLite."""
    names = FROZEN_EXP449_CONFIG.smoke_datasets if mode == "smoke" else ("fixture_full_antigravity",) * 1117
    if mode == "full":
        names = tuple(f"fixture_{index:04d}_antigravity" for index in range(1117))
    bare = {
        "schema": "paper-task8-catalog.v1", "mode": "smoke" if mode == "smoke" else "full",
        "database": {
            "unresolved_path": operations.FROZEN_PAPER_CONFIG.database_path,
            "resolved_path": str(Path(operations.FROZEN_PAPER_CONFIG.database_path).resolve(strict=False)),
            "size": 123,
            "sha256": operations.FROZEN_PAPER_CONFIG.database_sha256,
        },
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

    def _ready(self, *, logical_cpu_count: int = 12, builder=None, interpreter=None, dependencies=None, inventory=None, closure=None, config=None) -> dict[str, object]:
        builder = builder or (lambda mode: _catalog_antigravity(mode))
        interpreter = interpreter or {"executable": "fixture-python", "python_version": "3.14.6"}
        dependencies = dependencies or {"pycatch22": "0.5.0", "aeon": "1.5.0", "hydra": "constructed", "torch": "2.13.0"}
        inventory = inventory or {"schema": "fixture", "algorithm": "fixture", "records": [{"project": "fixture", "version": "1"}], "sha256": "f" * 64}
        closure_patch = patch.object(operations, "_closure_facts_antigravity", return_value=closure) if closure is not None else nullcontext()
        config_patch = patch.object(operations, "FROZEN_EXP449_CONFIG", config) if config is not None else nullcontext()
        with patch.object(operations, "_current_clean_head_antigravity", return_value="c" * 40), \
             patch.object(operations, "_require_capable_interpreter_antigravity", return_value=interpreter), \
             patch.object(operations, "_readiness_catalog_builder_antigravity", side_effect=builder), \
             patch.object(operations, "_dependency_facts_antigravity", return_value=dependencies), \
             patch.object(operations, "_package_inventory_facts_antigravity", return_value=inventory), \
             patch.object(operations, "_fixed_roots_antigravity", return_value=self.roots), closure_patch, config_patch:
            return operations._readiness_facts_antigravity("c" * 40, logical_cpu_count=logical_cpu_count)

    def test_authorized_readiness_rejects_an_authority_closure_that_does_not_match_recomputed_source(self) -> None:
        """Omitting authority-closure comparison would allow A -> I evidence to bind the wrong verifier source."""
        facts = self._ready()
        facts["roots"] = [{"path": str(path), "state": "absent"} for path in operations._fixed_roots_antigravity()]
        facts["seeds_and_constants"] = operations._validate_seeds_and_constants_antigravity(
            operations._seeds_and_constants_antigravity()
        )
        database = facts["database"]
        contexts = facts["contexts"]
        acceleration = facts["acceleration"]
        self.assertIsInstance(database, dict)
        self.assertIsInstance(contexts, dict)
        self.assertIsInstance(acceleration, dict)
        grant = Exp449SmokeAGrantAntigravity(
            schema="exp449-smoke-a-execution-grant-antigravity.v1", experiment_id="Exp449", target="smoke-a",
            policy_id="hard_plus_stable", implementation_commit="c" * 40,
            scientific_root=FROZEN_EXP449_CONFIG.smoke_a_scientific_root,
            operations_root=FROZEN_EXP449_CONFIG.smoke_a_operations_root,
            claim_path=SMOKE_A_CLAIM_PATH_ANTIGRAVITY, nonce="0" * 32,
            maximum_waves=2, label_access=False, logical_cpu_count=12, dataset_workers=11,
            database_path=str(database["path"]), database_size=int(database["observed_size"]), database_sha256=str(database["sha256"]),
            smoke_catalog_sha256=str(contexts["smoke_a"]["catalog_sha256"]), smoke_datasets=FROZEN_EXP449_CONFIG.smoke_datasets,
            context_sha256=str(contexts["smoke_a"]["context_sha256"]), scientific_code_closure=facts["code_closure"],
            authority_code_closure={"experiments/exp449_reproducible_rebaseline_antigravity/execution_authority_antigravity.py": "f" * 64},
            lineage_lock_sha256=str(facts["lineage_lock_paper_sha256"]), codemap_lock_sha256=str(facts["codemap_lock_sha256"]),
            acceleration_assessment_sha256=str(contexts["smoke_a"]["acceleration_assessment_sha256"]), canonical_backend="cpu",
            package_inventory=facts["package_inventory"], seeds_and_constants=facts["seeds_and_constants"], acceleration=acceleration,
            design_facts={
                "dataset_count": 1117, "test_row_count": 195900,
                "preprocessing_contracts": FROZEN_EXP449_CONFIG.preprocessing_contracts,
                "cpu_repeat_tolerance": 1e-12, "accelerated_tolerance": 1e-5, "minimum_accelerated_speedup": 1.1,
            }, grant_created_at="2026-08-15T11:45:00Z",
        )
        with self.assertRaisesRegex(ValueError, "authority"):
            operations._require_authorized_smoke_a_readiness_facts_antigravity(facts, grant)

    def test_closure_is_explicit_complete_safe_and_binds_paper_lineage_and_codemap_lock(self) -> None:
        """A named map or JSON hash cannot substitute for the executable scientific closure."""
        expected = {
            "experiments/exp449_reproducible_rebaseline_antigravity/artifacts_antigravity.py",
            "experiments/exp449_reproducible_rebaseline_antigravity/config_antigravity.py",
            "experiments/exp449_reproducible_rebaseline_antigravity/contracts_antigravity.py",
            "experiments/exp449_reproducible_rebaseline_antigravity/operations_antigravity.py",
            "experiments/exp449_reproducible_rebaseline_antigravity/runner_antigravity.py",
            "experiments/exp449_reproducible_rebaseline_antigravity/validation_antigravity.py",
            "experiments/exp449_reproducible_rebaseline_antigravity/verifier_antigravity.py",
            "paper/equipment_anomaly_detection/acceleration_paper.py",
            "paper/equipment_anomaly_detection/artifacts_paper.py",
            "paper/equipment_anomaly_detection/calibration_paper.py",
            "paper/equipment_anomaly_detection/catalog_paper.py",
            "paper/equipment_anomaly_detection/config_paper.py",
            "paper/equipment_anomaly_detection/contracts_paper.py",
            "paper/equipment_anomaly_detection/data_paper.py",
            "paper/equipment_anomaly_detection/hard_route_paper.py",
            "paper/equipment_anomaly_detection/lineage_paper.py",
            "paper/equipment_anomaly_detection/proposal_score_paper.py",
            "paper/equipment_anomaly_detection/routing_paper.py",
            "paper/equipment_anomaly_detection/score_adapter_paper.py",
            "paper/equipment_anomaly_detection/sources_paper/canonical_paper.py",
            "paper/equipment_anomaly_detection/sources_paper/hard_sources_paper.py",
            "paper/equipment_anomaly_detection/sources_paper/imaging_paper.py",
            "paper/equipment_anomaly_detection/sources_paper/knn_paper.py",
            "paper/equipment_anomaly_detection/sources_paper/legacy_route_sources_paper.py",
            "paper/equipment_anomaly_detection/sources_paper/low_frequency_paper.py",
            "paper/equipment_anomaly_detection/sources_paper/minirocket_paper.py",
            "paper/equipment_anomaly_detection/sources_paper/multirocket_hydra_paper.py",
            "paper/equipment_anomaly_detection/sources_paper/preprocessing_paper.py",
            "paper/equipment_anomaly_detection/sources_paper/rocket_paper.py",
            "paper/equipment_anomaly_detection/sources_paper/stability_paper.py",
        }
        closure = operations._closure_facts_antigravity()
        self.assertEqual(set(closure["code_closure"]), expected)
        self.assertEqual(
            closure["lineage_lock_paper_sha256"],
            operations._sha256_file_antigravity(operations.LINEAGE_LOCK_PATH),
        )
        self.assertEqual(
            closure["codemap_lock_sha256"],
            operations._sha256_file_antigravity(
                operations._REPOSITORY_ROOT_ANTIGRAVITY / "docs/codemap/codemap_exp449_antigravity.lock"
            ),
        )
        self.assertEqual(len(closure["code_closure"]), len(expected))

    def test_normalized_distribution_inventory_is_full_deterministic_and_version_sensitive(self) -> None:
        """A partial required-package check cannot detect another scientific package drift."""
        distributions = (
            SimpleNamespace(metadata={"Name": "Example_Pkg"}, version="1.0"),
            SimpleNamespace(metadata={"Name": "another.pkg"}, version="2.0"),
            SimpleNamespace(metadata={"Name": "Example-Pkg"}, version="1.0"),
        )
        inventory = operations._package_inventory_facts_antigravity(distributions)
        self.assertEqual(inventory["records"], [
            {"project": "another-pkg", "version": "2.0"},
            {"project": "example-pkg", "version": "1.0"},
        ])
        altered = operations._package_inventory_facts_antigravity((distributions[1],
            SimpleNamespace(metadata={"Name": "Example-Pkg"}, version="1.1"),
        ))
        self.assertNotEqual(inventory["sha256"], altered["sha256"])
        with self.assertRaisesRegex(ValueError, "conflicting"):
            operations._package_inventory_facts_antigravity(distributions + (
                SimpleNamespace(metadata={"Name": "example.pkg"}, version="9.9"),
            ))

    def test_package_inventory_is_bound_in_readiness_and_each_context_hash(self) -> None:
        """Changing, adding, or removing an installed distribution must describe a different run."""
        baseline = self._ready()
        altered_inventory = {"schema": "fixture", "algorithm": "fixture", "records": [{"project": "fixture", "version": "2"}, {"project": "added", "version": "1"}], "sha256": "e" * 64}
        altered = self._ready(inventory=altered_inventory)
        self.assertEqual(baseline["package_inventory"]["records"], [{"project": "fixture", "version": "1"}])
        self.assertNotEqual(baseline["contexts"]["official"]["context_sha256"], altered["contexts"]["official"]["context_sha256"])
        with self.assertRaisesRegex(ValueError, "dependency"):
            self._ready(dependencies={"pycatch22": "0.5.0", "aeon": "1.5.0", "hydra": "constructed", "torch": "2.13.1"})

    def test_constants_manifest_covers_every_scientific_category_and_rejects_each_mutation(self) -> None:
        """Seeds, alpha/threshold, routing, schemas, and recovery limits are separate frozen facts."""
        manifest = operations._seeds_and_constants_antigravity()
        self.assertEqual(manifest["schema"], "exp449-seeds-and-constants-antigravity.v1")
        for category in (
            "seeds", "alpha_and_thresholds", "route_promotion_topk_and_rate_rules",
            "policy_ids_and_order", "preprocessing_contracts", "dataset_and_test_counts",
            "tolerances", "task4_task5_recovery", "serialization_and_verifier_schemas",
        ):
            self.assertIn(category, manifest)
        for category, key in (
            ("seeds", "paper_seed"),
            ("alpha_and_thresholds", "source_alpha"),
            ("route_promotion_topk_and_rate_rules", "max_top"),
            ("serialization_and_verifier_schemas", "task45_success_schema"),
        ):
            altered = json.loads(json.dumps(manifest))
            altered[category][key] = "mutated_antigravity"
            with self.assertRaisesRegex(ValueError, "constants"):
                operations._validate_seeds_and_constants_antigravity(altered)

    @staticmethod
    def _config_with_antigravity(**changes: object) -> SimpleNamespace:
        """A complete frozen-config-shaped fixture for the readiness boundary only."""
        source = operations.FROZEN_EXP449_CONFIG
        fields = (
            "experiment_id", "dataset_count", "test_row_count", "policy_id",
            "cpu_repeat_tolerance", "accelerated_tolerance", "minimum_accelerated_speedup",
            "maximum_waves", "preprocessing_contracts", "smoke_datasets",
            "smoke_a_scientific_root", "smoke_a_operations_root", "smoke_b_scientific_root",
            "smoke_b_operations_root", "official_scientific_root", "official_operations_root",
        )
        return SimpleNamespace(**{field: changes.get(field, getattr(source, field)) for field in fields})

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
        self.assertEqual(set(facts["contexts"]), {"smoke_a", "smoke_b", "official"})
        self.assertTrue(all(value["historical_parity_status"] == "not_used" for value in facts["contexts"].values()))

    def test_observed_verified_catalog_database_digest_must_match_frozen_identity(self) -> None:
        """Returning a frozen SHA after a mismatched observed catalog would hide database substitution."""
        def wrong_digest(mode: str) -> dict[str, object]:
            catalog = _catalog_antigravity(mode)
            catalog["database"] = {**catalog["database"], "sha256": "b" * 64}
            bare = {key: value for key, value in catalog.items() if key != "catalog_sha256"}
            catalog["catalog_sha256"] = sha256(json.dumps(bare, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode() + b"\n").hexdigest()
            return catalog
        with self.assertRaisesRegex(ValueError, "database SHA256"):
            self._ready(builder=wrong_digest)
        facts = self._ready()
        self.assertEqual(facts["database"]["sha256"], operations.FROZEN_PAPER_CONFIG.database_sha256)
        self.assertEqual(facts["database"]["observed_size"], 123)

    def test_readiness_never_creates_any_absent_fixed_root(self) -> None:
        """Replacing lstat inspection with mkdir would turn a review into an execution side effect."""
        self._ready()
        self.assertTrue(all(not root.exists() and not root.is_symlink() for root in self.roots))

    def test_root_inspection_uses_exact_historical_path_authority_not_substrings(self) -> None:
        """Only an exact former root/component is unsafe; benign temporary spellings remain inspectable."""
        with self.assertRaisesRegex(ValueError, "fixed roots"):
            operations._inspect_fixed_roots_antigravity((self.root / "../escape_antigravity",) * 6)
        target = self.root / "target_antigravity"
        target.mkdir()
        link = self.root / "linked_antigravity"
        link.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            operations._inspect_fixed_roots_antigravity((link, *self.roots[1:]))

        for index, name in enumerate((
            "fixture_r4_suffix_antigravity",
            "unrelated_exp448ish_antigravity",
            "historical_notes_only_antigravity",
            "r4-exp448-long-random-antigravity",
        )):
            with self.subTest(benign=name):
                roots = list(self.roots)
                roots[0] = self.root / name
                inspected = operations._inspect_fixed_roots_antigravity(tuple(roots))
                self.assertEqual(inspected[0], {"path": str(roots[0]), "state": "absent"})

        for historical_root in operations._HISTORICAL_ROOTS_ANTIGRAVITY:
            for candidate in (historical_root, historical_root / "child_antigravity"):
                with self.subTest(historical_root=candidate):
                    roots = list(self.roots)
                    roots[0] = candidate
                    with self.assertRaisesRegex(ValueError, "historical"):
                        operations._inspect_fixed_roots_antigravity(tuple(roots))
        for component in operations._HISTORICAL_ROOT_COMPONENTS_ANTIGRAVITY:
            roots = list(self.roots)
            roots[0] = self.root / component / "destination_antigravity"
            with self.assertRaisesRegex(ValueError, "historical"):
                operations._inspect_fixed_roots_antigravity(tuple(roots))

    def test_dirty_or_nonmatching_reviewed_commit_fails_before_input_inspection(self) -> None:
        """A stale review must not be able to inspect a database or future root state."""
        with patch.object(operations, "_current_clean_head_antigravity", return_value="d" * 40), \
             patch.object(operations, "_readiness_catalog_builder_antigravity", side_effect=AssertionError("catalog read")):
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
             patch.object(operations, "_readiness_catalog_builder_antigravity", side_effect=lambda mode: _catalog_antigravity(mode)), \
             patch.object(operations, "_dependency_facts_antigravity", side_effect=ValueError("dependency unavailable:pycatch22==0.5.0")):
            with self.assertRaisesRegex(ValueError, "pycatch22"):
                operations.readiness_exp449_antigravity("c" * 40)

    def test_literal_pinned_interpreter_spelling_rejects_aliases_and_base_python(self) -> None:
        """Resolving the executable would silently accept a symlinked interpreter alias."""
        result = operations._require_capable_interpreter_antigravity()
        self.assertEqual(result["pinned_executable"], str(operations._CAPABLE_INTERPRETER_ANTIGRAVITY))
        alias = self.root / "python_alias_antigravity"
        alias.symlink_to(sys.executable)
        with patch.object(operations.sys, "executable", str(alias)):
            with self.assertRaisesRegex(ValueError, "capable interpreter"):
                operations._require_capable_interpreter_antigravity()
        with patch.object(operations.sys, "executable", "/usr/local/bin/python3"):
            with self.assertRaisesRegex(ValueError, "capable interpreter"):
                operations._require_capable_interpreter_antigravity()

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
        baseline = self._ready()
        altered_interpreter = self._ready(interpreter={"executable": "fixture-python-altered", "python_version": "3.14.6"})
        self.assertNotEqual(baseline["contexts"]["official"]["context_sha256"], altered_interpreter["contexts"]["official"]["context_sha256"])
        altered_workers = self._ready(logical_cpu_count=13)
        self.assertNotEqual(baseline["contexts"]["official"]["context_sha256"], altered_workers["contexts"]["official"]["context_sha256"])
        with self.assertRaisesRegex(ValueError, "dependency"):
            self._ready(dependencies={"pycatch22": "0.5.0", "aeon": "1.5.1", "hydra": "constructed", "torch": "2.13.0"})

    def test_context_hash_binds_closure_constants_and_approved_root(self) -> None:
        """Dropping any remaining verified input would permit a context to describe another run."""
        baseline = self._ready()
        closure = operations._closure_facts_antigravity()
        altered_closure = {
            **closure,
            "code_closure": {**closure["code_closure"], "experiments/exp449_reproducible_rebaseline_antigravity/operations_antigravity.py": "a" * 64},
        }
        closure_changed = self._ready(closure=altered_closure)
        self.assertNotEqual(baseline["contexts"]["official"]["context_sha256"], closure_changed["contexts"]["official"]["context_sha256"])
        altered_constants = self._ready(config=self._config_with_antigravity(policy_id="different_policy_antigravity"))
        self.assertNotEqual(baseline["contexts"]["official"]["context_sha256"], altered_constants["contexts"]["official"]["context_sha256"])
        with self.assertRaisesRegex(ValueError, "run root"):
            self._ready(config=self._config_with_antigravity(smoke_a_scientific_root=self.root / "unapproved_root_antigravity"))

    def test_read_only_status_reports_absent_roots_without_creating_them(self) -> None:
        """A status probe must not accidentally initialize an operations namespace."""
        with patch.object(operations, "_fixed_roots_antigravity", return_value=self.roots):
            result = operations.status_exp449_read_only_antigravity()
        self.assertEqual(result["status"], "not_started")
        self.assertTrue(all(item["state"] == "absent" for item in result["roots"]))
        self.assertTrue(all(not root.exists() for root in self.roots))

    def test_every_run_command_fails_closed_without_reading_any_authority_artifact(self) -> None:
        """An unpinned self-hash can never authorize an execution command."""
        for authority_shape in ("missing", "arbitrary", "self_hashed", "symlink", "path_swapped"):
            with self.subTest(authority_shape=authority_shape), \
                 patch.object(operations, "_current_clean_head_antigravity", return_value="c" * 40), \
                 patch.object(operations, "_fixed_roots_antigravity", return_value=self.roots), \
                 patch.object(operations, "_lstat_or_none_antigravity", side_effect=AssertionError("authority path read")):
                for command in ("run-smoke-a", "run-smoke-b", "run-official"):
                    with redirect_stderr(StringIO()):
                        with self.assertRaises(SystemExit) as exited:
                            operations.main_antigravity([command, "--reviewed-commit", "c" * 40])
                    self.assertEqual(exited.exception.code, 2)
        self.assertTrue(all(not root.exists() for root in self.roots))

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
