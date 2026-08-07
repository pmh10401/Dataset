"""Phase-C boundary tests for the fixed five-command Exp436 CLI.

Each test names a safety regression at the command boundary.  Database and
hardware work is replaced only below that boundary; filesystem publication is
always exercised with real temporary directories.
"""
from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp436_missing_safe_strict_consensus_antigravity import cli_antigravity as module
from experiments.exp436_missing_safe_strict_consensus_antigravity.config_antigravity import FROZEN_CONFIG
from experiments.exp436_missing_safe_strict_consensus_antigravity.contracts_antigravity import BackendDecision
from experiments.exp436_missing_safe_strict_consensus_antigravity.execution_profile_antigravity import EXP437_PROFILE


HEAD = "a" * 40


def _readiness(parent: Path, *, commit: str = HEAD) -> dict[str, object]:
    """Hand-built canonical readiness fixture; never uses CLI helpers."""
    decision = BackendDecision.cpu_fallback().to_payload()
    datasets = ["Demo"]
    lanes = {
        "exp291_prediction": FROZEN_CONFIG.exp291_prediction_sha256,
        "exp291_manifest": FROZEN_CONFIG.exp291_manifest_sha256,
        "exp292_prediction": FROZEN_CONFIG.exp292_prediction_sha256,
        "exp292_manifest": FROZEN_CONFIG.exp292_manifest_sha256,
    }
    config = json.loads(canonical_json_bytes(asdict(FROZEN_CONFIG)))
    return {
        "schema": "exp436.readiness-binding.v1", "run_identity": FROZEN_CONFIG.run_identity,
        "reviewed_commit": commit, "database_path": "/tmp/database_antigravity",
        "database_sha256": FROZEN_CONFIG.database_sha256,
        "exp291_prediction_path": "/tmp/exp291_prediction_antigravity",
        "exp291_prediction_sha256": lanes["exp291_prediction"],
        "exp291_manifest_path": "/tmp/exp291_manifest_antigravity",
        "exp291_manifest_sha256": lanes["exp291_manifest"],
        "exp292_prediction_path": "/tmp/exp292_prediction_antigravity",
        "exp292_prediction_sha256": lanes["exp292_prediction"],
        "exp292_manifest_path": "/tmp/exp292_manifest_antigravity",
        "exp292_manifest_sha256": lanes["exp292_manifest"],
        "datasets": datasets, "dataset_digest": sha256(canonical_json_bytes(datasets)).hexdigest(),
        "frozen_config": config, "frozen_config_sha256": sha256(canonical_json_bytes(config)).hexdigest(),
        "backend_decision": decision, "backend_decision_sha256": sha256(canonical_json_bytes(decision)).hexdigest(),
        "run_root": str((parent / "output_antigravity").resolve()),
        "checkpoint_root": str((parent / "checkpoint_antigravity").resolve()),
        "evaluation_root": str((parent / "evaluation_antigravity").resolve()),
        "evaluation_root_absent": True, "test_values_opened": 0, "test_labels_opened": 0,
        "label_access_count": 0, "focused_tests": {"returncode": 0, "command": ["unittest"]},
        "candidate_fixture": {"sha256": "b" * 64, "candidate_indices": []},
        "cpu_topology": {"logical_cpus": 12, "workers": 11, "inner_threads": 1},
        "runtime_versions": {"python": "test", "numpy": "test", "torch": "unavailable", "mlx": "unavailable"},
    }


class CliAntigravityTests(unittest.TestCase):
    def _deps(self, parent: Path, calls: list[str], *, state: str = "complete_review_only"):
        readiness = parent / "readiness_antigravity.json"
        return module._CliDependencies(
            output=parent / "output_antigravity", checkpoint=parent / "checkpoint_antigravity",
            readiness=readiness, current_head=lambda: HEAD,
            readiness_call=lambda: calls.append("readiness") or _readiness(parent),
            execute=lambda command: calls.append(command) or {"state": state},
            inspect=lambda: calls.append("inspect") or {"state": state},
            evaluate=lambda: calls.append("evaluate") or {"state": "evaluated", "success": False},
        )

    def _main(self, argv, deps):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = module._main_with_dependencies(argv, deps)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_parser_has_exactly_five_commands_and_no_flags(self):
        parser = module.build_parser()
        self.assertEqual(["readiness", "new", "resume", "inspect", "evaluate"],
                         re.search(r"\{([^}]+)\}", parser.format_help()).group(1).split(","))
        for command in ("readiness", "new", "resume", "inspect", "evaluate"):
            self.assertEqual(command, parser.parse_args([command]).command)
        with self.assertRaises(SystemExit):
            parser.parse_args(["new", "--workers", "1"])

    def test_exp437_parser_and_dependencies_are_bound_to_the_new_profile(self):
        parser = module.build_parser(profile=EXP437_PROFILE)
        self.assertIn(EXP437_PROFILE.cli_program, parser.prog)
        deps = module._production_dependencies(EXP437_PROFILE)
        self.assertEqual(deps.output, EXP437_PROFILE.output)
        self.assertEqual(deps.checkpoint, EXP437_PROFILE.checkpoint)
        self.assertEqual(deps.readiness, EXP437_PROFILE.readiness)
        with self.assertRaises(SystemExit):
            parser.parse_args(["new", "--workers", "1"])

    def test_exp437_readiness_projects_the_selected_profile_config(self):
        with TemporaryDirectory() as directory:
            parent = Path(directory) / "run_parent_antigravity"
            output = parent / "output_antigravity"
            checkpoint = parent / "checkpoint_antigravity"
            readiness = parent / "readiness_antigravity.json"
            evaluation = parent / "evaluation_antigravity"
            source = {
                "database": "/tmp/database_antigravity",
                "exp291_prediction": "/tmp/exp291_prediction_antigravity",
                "exp291_manifest": "/tmp/exp291_manifest_antigravity",
                "exp292_prediction": "/tmp/exp292_prediction_antigravity",
                "exp292_manifest": "/tmp/exp292_manifest_antigravity",
            }

            @contextmanager
            def session(*_args, **_kwargs):
                yield SimpleNamespace(
                    source_paths=SimpleNamespace(to_payload=lambda: source),
                    context=SimpleNamespace(datasets=("D0000",)),
                    label_access_count=0,
                )

            digests = {
                module.DATABASE_PATH: FROZEN_CONFIG.database_sha256,
                module.LANE_PATHS[0]: FROZEN_CONFIG.exp291_prediction_sha256,
                module.LANE_PATHS[1]: FROZEN_CONFIG.exp291_manifest_sha256,
                module.LANE_PATHS[2]: FROZEN_CONFIG.exp292_prediction_sha256,
                module.LANE_PATHS[3]: FROZEN_CONFIG.exp292_manifest_sha256,
            }
            decision = BackendDecision.cpu_fallback()
            with patch.object(module, "_fixed_paths", return_value=(parent, output, checkpoint, readiness, evaluation)), \
                    patch.object(module, "_clean_head", return_value=HEAD), \
                    patch.object(module, "_backend_decision_and_fixture", return_value=(decision, {"sha256": "b" * 64, "candidate_indices": []})), \
                    patch.object(module, "_focused_test_evidence", return_value={"returncode": 0, "command": ["unittest"]}), \
                    patch.object(module, "_runtime_versions", return_value={"python": "test", "numpy": "test", "torch": "test", "mlx": "test"}), \
                    patch.object(module, "_sha256_file", side_effect=lambda path: digests[Path(path)]), \
                    patch("experiments.exp436_missing_safe_strict_consensus_antigravity.inputs_antigravity.open_exp436_input_session", session):
                payload = module._production_readiness(EXP437_PROFILE)
            self.assertEqual(payload["run_identity"], EXP437_PROFILE.run_identity)
            self.assertEqual(payload["frozen_config"]["experiment_id"], EXP437_PROFILE.experiment_id)

    def test_exp437_wrapper_runs_from_outside_repository_with_no_override_flags(self):
        wrapper = Path(__file__).resolve().parents[3] / "scratch/run_exp437_metadata_roundtrip_antigravity.py"
        completed = subprocess.run(
            [sys.executable, str(wrapper), "--help"],
            cwd="/tmp",
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("run_exp437_metadata_roundtrip_antigravity.py", completed.stdout)
        self.assertIn("{readiness,new,resume,inspect,evaluate}", completed.stdout)
        self.assertNotIn("--workers", completed.stdout)
        self.assertNotIn("--backend", completed.stdout)

    def test_new_requires_a_canonical_binding_from_the_current_clean_head(self):
        with TemporaryDirectory() as directory:
            parent, calls = Path(directory), []
            deps = self._deps(parent, calls)
            deps.readiness.write_bytes(canonical_json_bytes(_readiness(parent)))
            code, _out, err = self._main(["new"], deps)
            self.assertEqual((code, calls), (0, ["new"]), err)

    def test_new_rejects_noncanonical_readiness_before_execution(self):
        with TemporaryDirectory() as directory:
            parent, calls = Path(directory), []
            deps = self._deps(parent, calls)
            deps.readiness.write_text(json.dumps(_readiness(parent)), encoding="utf-8")
            code, _out, err = self._main(["new"], deps)
            self.assertEqual(code, 2); self.assertIn("readiness", err); self.assertEqual(calls, [])

    def test_new_rechecks_the_head_between_readiness_validation_and_execution(self):
        with TemporaryDirectory() as directory:
            parent, calls = Path(directory), []
            deps = self._deps(parent, calls)
            deps.readiness.write_bytes(canonical_json_bytes(_readiness(parent)))
            deps = module._CliDependencies(deps.output, deps.checkpoint, deps.readiness,
                current_head=iter([HEAD, "b" * 40]).__next__, readiness_call=deps.readiness_call,
                execute=deps.execute, inspect=deps.inspect, evaluate=deps.evaluate)
            code, _out, err = self._main(["new"], deps)
            self.assertEqual(code, 2); self.assertIn("HEAD", err); self.assertEqual(calls, [])

    def test_new_rejects_existing_output_checkpoint_and_evaluation_roots(self):
        for name in ("output_antigravity", "checkpoint_antigravity", "evaluation_antigravity"):
            with self.subTest(root=name), TemporaryDirectory() as directory:
                parent, calls = Path(directory), []
                deps = self._deps(parent, calls); deps.readiness.write_bytes(canonical_json_bytes(_readiness(parent)))
                (parent / name).mkdir()
                code, _out, _err = self._main(["new"], deps)
                self.assertEqual((code, calls), (2, []))

    def test_atomic_readiness_creates_previously_absent_parent_with_only_canonical_file(self):
        with TemporaryDirectory() as directory:
            parent = Path(directory) / "run_parent_antigravity"
            payload = _readiness(parent)
            module._publish_readiness_atomically(parent / "readiness_antigravity.json", payload)
            self.assertEqual(tuple(item.name for item in parent.iterdir()), ("readiness_antigravity.json",))
            self.assertEqual((parent / "readiness_antigravity.json").read_bytes(), canonical_json_bytes(payload))

    def test_main_readiness_returns_zero_after_canonical_publication_without_state(self):
        """Removing the readiness success branch would turn this valid CLI call into exit 1."""
        with TemporaryDirectory() as directory:
            parent, calls = Path(directory) / "run_parent_antigravity", []
            payload = _readiness(parent)

            def publish_readiness():
                calls.append("readiness")
                module._publish_readiness_atomically(parent / "readiness_antigravity.json", payload)
                return payload

            base = self._deps(parent, calls)
            deps = module._CliDependencies(base.output, base.checkpoint, base.readiness,
                base.current_head, publish_readiness, base.execute, base.inspect, base.evaluate)
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.object(module, "_production_dependencies", return_value=deps), \
                    redirect_stdout(stdout), redirect_stderr(stderr):
                code = module.main(["readiness"])
            self.assertEqual((code, calls, stderr.getvalue()), (0, ["readiness"], ""))
            published = json.loads(deps.readiness.read_bytes())
            self.assertEqual(published, payload)
            self.assertNotIn("state", published)

    def test_atomic_readiness_rejects_existing_parent_without_clobber(self):
        with TemporaryDirectory() as directory:
            parent = Path(directory) / "run_parent_antigravity"; parent.mkdir()
            marker = parent / "marker_antigravity"; marker.write_bytes(b"preserve")
            with self.assertRaisesRegex(ValueError, "absent"):
                module._publish_readiness_atomically(parent / "readiness_antigravity.json", _readiness(parent))
            self.assertEqual(marker.read_bytes(), b"preserve")

    def test_atomic_readiness_rejects_a_symlink_parent(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); target = root / "target_antigravity"; target.mkdir()
            parent = root / "run_parent_antigravity"; parent.symlink_to(target, target_is_directory=True)
            with self.assertRaises(ValueError):
                module._publish_readiness_atomically(parent / "readiness_antigravity.json", _readiness(parent))
            self.assertEqual(tuple(target.iterdir()), ())

    def test_atomic_readiness_failure_cleans_staging_and_never_leaves_empty_parent(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); parent = root / "run_parent_antigravity"
            with patch.object(module, "_write_create_only", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    module._publish_readiness_atomically(parent / "readiness_antigravity.json", _readiness(parent))
            self.assertFalse(parent.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_atomic_readiness_race_does_not_replace_winner(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); parent = root / "run_parent_antigravity"; winner = {"done": False}
            def race(source, destination):
                if not winner["done"]:
                    winner["done"] = True; parent.mkdir(); (parent / "winner_antigravity").write_bytes(b"winner")
                raise FileExistsError(destination)
            with patch.object(module, "_exclusive_rename", side_effect=race):
                with self.assertRaises(OSError):
                    module._publish_readiness_atomically(parent / "readiness_antigravity.json", _readiness(parent))
            self.assertEqual((parent / "winner_antigravity").read_bytes(), b"winner")

    def test_readiness_fixture_construction_uses_scalar_queries_and_frozen_threshold(self):
        """A 2-D/missing-threshold smoke fixture must fail before any backend run."""
        from experiments.exp436_missing_safe_strict_consensus_antigravity.contracts_antigravity import BackendBenchmarkEvidence
        seen, mps_seen = [], []
        def observe(fixtures):
            seen.extend(fixtures)
            return BackendBenchmarkEvidence.unavailable("mlx", "fixture")
        with patch("experiments.exp436_missing_safe_strict_consensus_antigravity.mlx_antigravity.benchmark_mlx_adoption", side_effect=observe), \
                patch("experiments.exp436_missing_safe_strict_consensus_antigravity.mps_antigravity.benchmark_mps_adoption", side_effect=lambda fixtures: mps_seen.extend(fixtures) or BackendBenchmarkEvidence.unavailable("mps", "fixture")):
            decision, fixture = module._backend_decision_and_fixture()
        self.assertEqual(decision.backend, "cpu")
        self.assertEqual((len(seen), len(mps_seen)), (1, 1))
        for observed in (*seen, *mps_seen):
            self.assertEqual((observed.query.ndim, observed.k, float(observed.threshold)), (1, 3, 0.0))
        self.assertEqual(fixture["candidate_indices"], [0])

    def test_readiness_uses_exclusive_rename_helper_not_replacing_os_rename(self):
        with TemporaryDirectory() as directory:
            parent = Path(directory) / "run_parent_antigravity"
            with patch.object(module, "_exclusive_rename", wraps=module._exclusive_rename) as renamed:
                module._publish_readiness_atomically(parent / "readiness_antigravity.json", _readiness(parent))
            self.assertEqual(renamed.call_count, 1)

    def test_public_readiness_runs_without_cli_dependency_injection(self):
        """The real entrypoint binds its own sources before atomically publishing."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "run_parent_antigravity"
            output = parent / "output_antigravity"
            checkpoint = parent / "checkpoint_antigravity"
            evaluation = parent / "evaluation_antigravity"
            readiness = parent / "readiness_antigravity.json"
            source = {
                "database": "/tmp/database_antigravity",
                "exp291_prediction": "/tmp/exp291_prediction_antigravity",
                "exp291_manifest": "/tmp/exp291_manifest_antigravity",
                "exp292_prediction": "/tmp/exp292_prediction_antigravity",
                "exp292_manifest": "/tmp/exp292_manifest_antigravity",
            }
            @contextmanager
            def session(*_args, **_kwargs):
                yield SimpleNamespace(
                    source_paths=SimpleNamespace(to_payload=lambda: source),
                    context=SimpleNamespace(datasets=("D0000",)), label_access_count=0,
                )
            digests = {
                module.DATABASE_PATH: FROZEN_CONFIG.database_sha256,
                module.LANE_PATHS[0]: FROZEN_CONFIG.exp291_prediction_sha256,
                module.LANE_PATHS[1]: FROZEN_CONFIG.exp291_manifest_sha256,
                module.LANE_PATHS[2]: FROZEN_CONFIG.exp292_prediction_sha256,
                module.LANE_PATHS[3]: FROZEN_CONFIG.exp292_manifest_sha256,
            }
            from experiments.exp436_missing_safe_strict_consensus_antigravity.contracts_antigravity import BackendBenchmarkEvidence
            with patch.object(module, "RUN_PARENT", parent), \
                    patch.object(module, "FROZEN_OUTPUT", output), \
                    patch.object(module, "FROZEN_CHECKPOINT", checkpoint), \
                    patch.object(module, "FROZEN_EVALUATION", evaluation), \
                    patch.object(module, "FROZEN_READINESS", readiness), \
                    patch.object(module, "_clean_head", return_value=HEAD), \
                    patch("experiments.exp436_missing_safe_strict_consensus_antigravity.mlx_antigravity.benchmark_mlx_adoption", return_value=BackendBenchmarkEvidence.unavailable("mlx", "test_unavailable")), \
                    patch("experiments.exp436_missing_safe_strict_consensus_antigravity.mps_antigravity.benchmark_mps_adoption", return_value=BackendBenchmarkEvidence.unavailable("mps", "test_unavailable")), \
                    patch.object(module, "_focused_test_evidence", return_value={"returncode": 0, "command": ["unittest"]}), \
                    patch.object(module, "_runtime_versions", return_value={"python": "test", "numpy": "test", "torch": "test", "mlx": "test"}), \
                    patch.object(module, "_sha256_file", side_effect=lambda path: digests[Path(path)]), \
                    patch("experiments.exp436_missing_safe_strict_consensus_antigravity.inputs_antigravity.open_exp436_input_session", session):
                payload = module.run_readiness()
            self.assertEqual(payload["reviewed_commit"], HEAD)
            self.assertEqual(json.loads(readiness.read_bytes())["candidate_fixture"], payload["candidate_fixture"])
            self.assertEqual(payload["candidate_fixture"]["candidate_indices"], [0])
            self.assertEqual(tuple(path.name for path in parent.iterdir()), ("readiness_antigravity.json",))

    def test_readiness_payload_binds_evidence_and_absent_evaluation_root(self):
        with TemporaryDirectory() as directory:
            payload = _readiness(Path(directory))
            self.assertEqual(module._validate_readiness_payload(payload, expected_commit=HEAD)["cpu_topology"],
                             {"logical_cpus": 12, "workers": 11, "inner_threads": 1})

    def test_readiness_rejects_any_label_access_or_missing_candidate_fixture(self):
        with TemporaryDirectory() as directory:
            payload = _readiness(Path(directory)); payload["label_access_count"] = 1
            with self.assertRaises(ValueError): module._validate_readiness_payload(payload, expected_commit=HEAD)
            payload = _readiness(Path(directory)); del payload["candidate_fixture"]
            with self.assertRaises(ValueError): module._validate_readiness_payload(payload, expected_commit=HEAD)

    def test_readiness_rejects_output_checkpoint_or_evaluation_presence_evidence(self):
        with TemporaryDirectory() as directory:
            for key in ("evaluation_root_absent",):
                payload = _readiness(Path(directory)); payload[key] = False
                with self.subTest(key=key), self.assertRaises(ValueError):
                    module._validate_readiness_payload(payload, expected_commit=HEAD)

    def test_resume_rejects_failed_before_execute(self):
        with TemporaryDirectory() as directory:
            parent, calls = Path(directory), []
            deps = self._deps(parent, calls); deps.output.mkdir(); deps.checkpoint.mkdir()
            with patch.object(module, "_resume_state", return_value="failed"):
                code, _out, err = self._main(["resume"], deps)
            self.assertEqual(code, 2); self.assertIn("failed", err); self.assertEqual(calls, [])

    def test_resume_rejects_running_and_fake_complete_states_before_execute(self):
        for state in ("running", "complete"):
            with self.subTest(state=state), TemporaryDirectory() as directory:
                parent, calls = Path(directory), []
                deps = self._deps(parent, calls); deps.output.mkdir(); deps.checkpoint.mkdir()
                with patch.object(module, "_resume_state", return_value=state):
                    code, _out, _err = self._main(["resume"], deps)
                self.assertEqual((code, calls), (2, []))

    def test_resume_only_executes_the_interrupted_state(self):
        with TemporaryDirectory() as directory:
            parent, calls = Path(directory), []
            deps = self._deps(parent, calls); deps.output.mkdir(); deps.checkpoint.mkdir()
            with patch.object(module, "_resume_state", return_value="interrupted"):
                code, _out, err = self._main(["resume"], deps)
            self.assertEqual((code, calls), (0, ["resume"]), err)

    def test_inspect_is_read_only_and_does_not_publish(self):
        with TemporaryDirectory() as directory:
            parent, calls = Path(directory), []
            deps = self._deps(parent, calls); deps.output.mkdir(); deps.checkpoint.mkdir()
            marker = deps.output / "immutable_antigravity"; marker.write_bytes(b"unchanged")
            code, _out, err = self._main(["inspect"], deps)
            self.assertEqual((code, calls), (0, ["inspect"]), err); self.assertEqual(marker.read_bytes(), b"unchanged")

    def test_evaluate_exit_zero_means_completed_not_champion(self):
        with TemporaryDirectory() as directory:
            parent, calls = Path(directory), []
            deps = self._deps(parent, calls); deps.output.mkdir(); deps.checkpoint.mkdir()
            code, stdout, err = self._main(["evaluate"], deps)
            self.assertEqual((code, calls), (0, ["evaluate"]), err)
            self.assertFalse(json.loads(stdout)["success"])

    def test_prediction_and_inspect_do_not_import_evaluation_module(self):
        name = "experiments.exp436_missing_safe_strict_consensus_antigravity.evaluation_antigravity"
        sys.modules.pop(name, None)
        with TemporaryDirectory() as directory:
            parent, calls = Path(directory), []
            deps = self._deps(parent, calls); deps.output.mkdir(); deps.checkpoint.mkdir()
            self._main(["inspect"], deps)
        self.assertNotIn(name, sys.modules)

    def test_evaluate_imports_only_the_exact_evaluation_module_lazily(self):
        name = "experiments.exp436_missing_safe_strict_consensus_antigravity.evaluation_antigravity"
        sys.modules.pop(name, None)
        with patch.object(module.importlib, "import_module", side_effect=ImportError("exact firewall")) as imported:
            with self.assertRaises(ImportError): module.run_evaluate()
        self.assertEqual(imported.call_args.args[0], name)

    def test_wrapper_help_runs_outside_the_repository_without_pythonpath(self):
        runner = Path(__file__).resolve().parents[3] / "scratch" / "run_exp436_missing_safe_strict_consensus_antigravity.py"
        env = dict(os.environ); env.pop("PYTHONPATH", None)
        with TemporaryDirectory() as directory:
            completed = subprocess.run([sys.executable, str(runner), "--help"], cwd=directory, env=env, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("readiness", completed.stdout)

    def test_cli_errors_are_json_free_and_use_exit_two(self):
        with TemporaryDirectory() as directory:
            parent, calls = Path(directory), []
            code, out, err = self._main(["new"], self._deps(parent, calls))
            self.assertEqual((code, out, calls), (2, "", [])); self.assertIn("new failed", err)


if __name__ == "__main__":
    unittest.main()
