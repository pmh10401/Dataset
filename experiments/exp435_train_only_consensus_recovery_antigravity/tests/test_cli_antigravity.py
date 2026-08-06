"""TDD checks for the fixed Exp435 command boundary."""

from __future__ import annotations

import io
import json
import builtins
from contextlib import contextmanager
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch
from dataclasses import asdict
from hashlib import sha256
from types import SimpleNamespace

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp435_train_only_consensus_recovery_antigravity import cli_antigravity as cli_module
from experiments.exp435_train_only_consensus_recovery_antigravity import reporting_antigravity as reporting_module
from experiments.exp435_train_only_consensus_recovery_antigravity.checkpoint_antigravity import Exp435CheckpointStore
from experiments.exp435_train_only_consensus_recovery_antigravity.contracts_antigravity import (
    Exp435Candidate,
    Exp435DatasetResult,
    InstanceScoreEvidence,
    BackendDecision,
)
from experiments.exp435_train_only_consensus_recovery_antigravity.cli_antigravity import (
    _CliDependencies,
    _write_new,
    build_parser,
    main,
)
from experiments.exp435_train_only_consensus_recovery_antigravity.config_antigravity import FROZEN_CONFIG
from experiments.exp435_train_only_consensus_recovery_antigravity.progress_antigravity import ProgressWriter


class Exp435CliTests(unittest.TestCase):
    def _valid_readiness(self, commit: str) -> dict[str, object]:
        lanes = {
            "exp291_prediction": FROZEN_CONFIG.exp291_prediction_sha256,
            "exp291_manifest": FROZEN_CONFIG.exp291_manifest_sha256,
            "exp292_prediction": FROZEN_CONFIG.exp292_prediction_sha256,
            "exp292_manifest": FROZEN_CONFIG.exp292_manifest_sha256,
        }

        return {
            "schema": "exp435.readiness.v1",
            "state": "ready",
            "experiment_id": FROZEN_CONFIG.experiment_id,
            "run_identity": FROZEN_CONFIG.run_identity,
            "git_commit": commit,
            "config_sha256": sha256(canonical_json_bytes(asdict(FROZEN_CONFIG))).hexdigest(),
            "input_sha256": {"database": FROZEN_CONFIG.database_sha256, **lanes},
            "database_sha256": FROZEN_CONFIG.database_sha256,
            "lane_artifact_sha256": lanes,
            "backend": "cpu",
            "backend_decision": BackendDecision.cpu_fallback("fixture").to_dict(),
            "runtime_versions": {"python": "test"},
            "scientific_parity": True,
            "output_root_absent": True,
            "checkpoint_root_absent": True,
            "evaluation_root_absent": True,
            "test_values_opened": 0,
            "test_labels_opened": 0,
            "label_access_count": 0,
            "focused_tests": {"returncode": 0},
        }

    def _completed_runtime(self, parent: Path) -> tuple[Path, Path, Exp435CheckpointStore]:
        output = parent / "output_antigravity"
        checkpoint = parent / "checkpoint_antigravity"
        context = {
            "schema": "fixture.context.v1",
            "run_identity": FROZEN_CONFIG.run_identity,
            "datasets": ["Demo"],
            "test_sizes": {"Demo": 101},
            "database_sha256": "d" * 64,
            "code_commit": "f" * 40,
            "runtime_versions": {"python": "test"},
            "backend_decision": BackendDecision.cpu_fallback("fixture").to_dict(),
        }
        store = Exp435CheckpointStore.create(checkpoint, context)
        evidence = InstanceScoreEvidence.fixture(
            dataset="Demo", instance_index=0, length=16,
            prototype_abnormal=True, gradient_abnormal=True,
        )
        digest = store.commit(Exp435DatasetResult(
            dataset="Demo", status="scored", test_size=101,
            candidates=(Exp435Candidate("Demo", 0, 16, evidence.evidence_digest),),
            evidence=(evidence,),
        ))
        writer = ProgressWriter.create(
            output,
            run_identity=store.run_identity,
            checkpoint=checkpoint,
            total_datasets=1,
            backend="cpu",
            context_fingerprint=store.context_fingerprint,
            heartbeat_seconds=None,
        )
        writer.update(
            stage="scoring", current_dataset="Demo", dataset_metadata=None,
            completed=1, success=1, abstention=0, error=0, candidate_count=1,
        )
        writer.append_event({"kind": "dataset_complete", "dataset": "Demo", "record_sha256": digest})
        writer.mark_complete(candidate_count=1)
        writer.close()
        reporting_module.publish_exp435(
            store, output, _expectations=reporting_module._PublicationExpectations(total=1),
        )
        return output, checkpoint, store

    def _unfinished_runtime(self, parent: Path) -> tuple[Path, Path, Exp435CheckpointStore]:
        output = parent / "output_antigravity"
        checkpoint = parent / "checkpoint_antigravity"
        context = {
            "schema": "fixture.context.v1",
            "run_identity": FROZEN_CONFIG.run_identity,
            "datasets": ["Demo"],
            "test_sizes": {"Demo": 101},
            "database_sha256": "d" * 64,
            "code_commit": "f" * 40,
            "runtime_versions": {"python": "test"},
            "backend_decision": BackendDecision.cpu_fallback("fixture").to_dict(),
        }
        store = Exp435CheckpointStore.create(checkpoint, context)
        writer = ProgressWriter.create(
            output,
            run_identity=store.run_identity,
            checkpoint=checkpoint,
            total_datasets=1,
            backend="cpu",
            context_fingerprint=store.context_fingerprint,
            heartbeat_seconds=None,
        )
        writer.close()
        return output, checkpoint, store

    def _local_publication(self, output: Path, store: Exp435CheckpointStore):
        return reporting_module.inspect_publication(
            output, store, _expectations=reporting_module._PublicationExpectations(total=1),
        )

    def _dependencies(self, parent: Path, calls: list[str]) -> _CliDependencies:
        output = parent / "output_antigravity"
        checkpoint = parent / "checkpoint_antigravity"
        readiness = parent / "readiness_antigravity.json"

        def readiness_call() -> dict[str, object]:
            calls.append("readiness")
            return {
                "schema": "exp435.readiness.v1",
                "state": "ready",
                "git_commit": "a" * 40,
                "backend": "cpu",
                "output_root_absent": True,
                "checkpoint_root_absent": True,
            }

        def execute(mode: str) -> dict[str, object]:
            calls.append(mode)
            return {
                "state": "complete_review_only",
                "completed_datasets": 1117,
                "failed_datasets": 0,
            }

        return _CliDependencies(
            output=output,
            checkpoint=checkpoint,
            readiness=readiness,
            current_head=lambda: "a" * 40,
            readiness_call=readiness_call,
            execute=execute,
            inspect=lambda: {"state": "complete_review_only"},
            evaluate=lambda: {"state": "evaluated"},
        )

    def _main(self, argv: list[str], dependencies: _CliDependencies):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(argv, _dependencies=dependencies)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_parser_has_exactly_five_modes(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()
        for name in ("readiness", "new", "resume", "inspect", "evaluate"):
            self.assertIn(name, help_text)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["new", "--output-root", "/tmp/alternate"])

    def test_cli_run_identity_is_derived_from_the_frozen_config(self) -> None:
        self.assertEqual(cli_module.RUN_IDENTITY, FROZEN_CONFIG.run_identity)

    def test_absolute_scratch_runner_help_works_outside_repo_without_pythonpath(self) -> None:
        """The documented wrapper must bootstrap imports when launched directly."""
        runner = Path(__file__).resolve().parents[3] / "scratch" / (
            "run_exp435_train_only_consensus_recovery_antigravity.py"
        )
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        with TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, str(runner), "--help"],
                cwd=directory,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        match = re.search(r"\{([^}]+)\}", completed.stdout)
        self.assertIsNotNone(match, completed.stdout)
        self.assertEqual(
            match.group(1).split(","),
            ["readiness", "new", "resume", "inspect", "evaluate"],
        )

    def test_readiness_uses_only_fixed_file_and_no_alternate_path(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            calls: list[str] = []
            dependencies = self._dependencies(parent, calls)
            code, _stdout, stderr = self._main(
                ["readiness", "--readiness-output", str(parent / "other.json")],
                dependencies,
            )
            self.assertEqual(code, 2)
            self.assertIn("unrecognized", stderr)
            self.assertEqual(calls, [])

    def test_new_rechecks_clean_commit_before_execute(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            calls: list[str] = []
            dependencies = self._dependencies(parent, calls)
            dependencies.readiness.write_text(
                json.dumps({"schema": "exp435.readiness.v1", "state": "ready", "git_commit": "a" * 40}),
                encoding="utf-8",
            )
            code, _stdout, stderr = self._main(["new"], dependencies)
            self.assertEqual(code, 2)
            self.assertIn("readiness", stderr)
            self.assertEqual(calls, [])

    def test_readiness_action_is_the_single_writer(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory) / "new_parent"
            output = parent / "output_antigravity"
            checkpoint = parent / "checkpoint_antigravity"
            readiness = parent / "readiness_antigravity.json"
            calls: list[str] = []
            payload = self._valid_readiness("a" * 40)

            def readiness_call() -> dict[str, object]:
                calls.append("readiness")
                parent.mkdir()
                _write_new(readiness, canonical_json_bytes(payload))
                return payload

            dependencies = _CliDependencies(
                output=output,
                checkpoint=checkpoint,
                readiness=readiness,
                current_head=lambda: "a" * 40,
                readiness_call=readiness_call,
                execute=lambda mode: {"state": "complete_review_only"},
                inspect=lambda: {"state": "complete_review_only"},
                evaluate=lambda: {"state": "evaluated"},
            )
            code, stdout, stderr = self._main(["readiness"], dependencies)
            self.assertEqual(code, 0, stderr)
            self.assertEqual(calls, ["readiness"])
            self.assertEqual(readiness.read_bytes(), canonical_json_bytes(payload))
            self.assertEqual(json.loads(stdout), payload)

    def test_prediction_modes_do_not_import_evaluation(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            calls: list[str] = []
            dependencies = self._dependencies(parent, calls)
            with patch.dict("sys.modules", {"experiments.exp435_train_only_consensus_recovery_antigravity.evaluation_antigravity": None}):
                code, _stdout, stderr = self._main(["inspect"], dependencies)
            self.assertEqual(code, 2)
            self.assertNotIn("evaluation", stderr.lower())

    def test_new_preflight_requires_evaluation_root_absent(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            calls: list[str] = []
            dependencies = self._dependencies(parent, calls)
            (parent / "evaluation_antigravity").mkdir()
            code, _stdout, stderr = self._main(["new"], dependencies)
            self.assertEqual(code, 2)
            self.assertIn("new", stderr)
            self.assertEqual(calls, [])

    def test_ensure_run_parent_creates_empty_parent_and_rejects_unknown_content(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory) / "run_parent"
            cli_module._ensure_run_parent(parent)
            self.assertTrue(parent.is_dir())
            cli_module._ensure_run_parent(parent)
            (parent / "unknown.txt").write_text("do not accept", encoding="utf-8")
            with self.assertRaises(ValueError):
                cli_module._ensure_run_parent(parent)

    def test_inspect_after_publication_replays_without_runtime_imports(self) -> None:
        with TemporaryDirectory() as directory:
            output, checkpoint, _store = self._completed_runtime(Path(directory))
            with patch.object(cli_module, "FROZEN_OUTPUT", output), \
                    patch.object(cli_module, "FROZEN_CHECKPOINT", checkpoint), \
                    patch.object(cli_module, "inspect_publication", self._local_publication), \
                    patch.dict("sys.modules", {
                        "experiments.exp435_train_only_consensus_recovery_antigravity.inputs_antigravity": None,
                        "experiments.exp435_train_only_consensus_recovery_antigravity.runtime_antigravity": None,
                    }):
                # Bind the store argument through the production callback shape.
                with patch.object(cli_module, "inspect_publication", side_effect=lambda out, store: self._local_publication(out, store)):
                    result = cli_module.run_inspect()
            self.assertEqual(result["state"], "complete_review_only")
            self.assertEqual(result["prediction_sha256"], json.loads(
                (output / "publication_antigravity" / "manifest_antigravity.json").read_bytes()
            )["prediction_sha256"])

    def test_repeated_resume_replays_publication_without_database_or_workers(self) -> None:
        with TemporaryDirectory() as directory:
            output, checkpoint, _store = self._completed_runtime(Path(directory))
            with patch.object(cli_module, "FROZEN_OUTPUT", output), \
                    patch.object(cli_module, "FROZEN_CHECKPOINT", checkpoint), \
                    patch.object(cli_module, "_clean_head", return_value="f" * 40), \
                    patch.object(cli_module, "inspect_publication", side_effect=lambda out, store: self._local_publication(out, store)), \
                    patch.dict("sys.modules", {
                        "experiments.exp435_train_only_consensus_recovery_antigravity.inputs_antigravity": None,
                        "experiments.exp435_train_only_consensus_recovery_antigravity.runtime_antigravity": None,
                    }):
                result = cli_module.run_resume()
            self.assertEqual(result["state"], "complete_review_only")
            self.assertEqual(result["prediction_sha256"], json.loads(
                (output / "publication_antigravity" / "manifest_antigravity.json").read_bytes()
            )["prediction_sha256"])

    def test_unfinished_resume_passes_exact_final_head_to_input_session(self) -> None:
        with TemporaryDirectory() as directory:
            output, checkpoint, _store = self._unfinished_runtime(Path(directory))
            received: list[str] = []

            @contextmanager
            def fake_session(*args, **kwargs):
                received.append(kwargs["verified_code_commit"])
                yield SimpleNamespace(context={})

            class Summary:
                status = "running"

                def to_payload(self):
                    return {"state": "running"}

            inputs_module = __import__(
                "experiments.exp435_train_only_consensus_recovery_antigravity.inputs_antigravity",
                fromlist=["open_exp435_input_session"],
            )
            runtime_module = __import__(
                "experiments.exp435_train_only_consensus_recovery_antigravity.runtime_antigravity",
                fromlist=["inspect_exp435", "resume_exp435"],
            )
            with patch.object(cli_module, "FROZEN_OUTPUT", output), \
                    patch.object(cli_module, "FROZEN_CHECKPOINT", checkpoint), \
                    patch.object(cli_module, "_clean_head", side_effect=["f" * 40, "f" * 40]), \
                    patch.object(runtime_module, "inspect_exp435", return_value=SimpleNamespace(status="running")), \
                    patch.object(inputs_module, "open_exp435_input_session", fake_session), \
                    patch.object(runtime_module, "_runtime_context", return_value={}), \
                    patch.object(runtime_module, "resume_exp435", return_value=Summary()):
                result = cli_module._run_prediction("resume")
            self.assertEqual(result["state"], "running")
            self.assertEqual(received, ["f" * 40])

    def test_unfinished_resume_rejects_head_changed_before_input_session(self) -> None:
        with TemporaryDirectory() as directory:
            output, checkpoint, _store = self._unfinished_runtime(Path(directory))
            opened = False

            @contextmanager
            def fail_session(*args, **kwargs):
                nonlocal opened
                opened = True
                raise AssertionError("input session must not open after resume HEAD change")
                yield  # pragma: no cover

            class Summary:
                status = "running"

                def to_payload(self):
                    return {"state": "running"}

            inputs_module = __import__(
                "experiments.exp435_train_only_consensus_recovery_antigravity.inputs_antigravity",
                fromlist=["open_exp435_input_session"],
            )
            runtime_module = __import__(
                "experiments.exp435_train_only_consensus_recovery_antigravity.runtime_antigravity",
                fromlist=["inspect_exp435", "resume_exp435"],
            )
            with patch.object(cli_module, "FROZEN_OUTPUT", output), \
                    patch.object(cli_module, "FROZEN_CHECKPOINT", checkpoint), \
                    patch.object(cli_module, "_clean_head", side_effect=["f" * 40, "e" * 40]), \
                    patch.object(runtime_module, "inspect_exp435", return_value=SimpleNamespace(status="running")), \
                    patch.object(inputs_module, "open_exp435_input_session", fail_session), \
                    patch.object(runtime_module, "_runtime_context", return_value={}), \
                    patch.object(runtime_module, "resume_exp435", return_value=Summary()):
                with self.assertRaises(ValueError):
                    cli_module._run_prediction("resume")
            self.assertFalse(opened)

    def test_failed_resume_rejects_before_input_and_preserves_runtime_tree(self) -> None:
        with TemporaryDirectory() as directory:
            output, checkpoint, store = self._unfinished_runtime(Path(directory))
            writer = ProgressWriter.open_existing(
                output,
                run_identity=store.run_identity,
                checkpoint=checkpoint,
                backend="cpu",
                total_datasets=1,
                context_fingerprint=store.context_fingerprint,
                process_is_live=lambda pid: False,
                heartbeat_seconds=None,
            )
            writer.mark_failed("Demo", RuntimeError("synthetic failure"))
            writer.close()
            root = Path(directory)
            before = {
                path.relative_to(root): path.read_bytes()
                for path in sorted(root.rglob("*")) if path.is_file()
            }
            opened = False

            @contextmanager
            def fail_session(*args, **kwargs):
                nonlocal opened
                opened = True
                raise AssertionError("failed roots cannot open input")
                yield  # pragma: no cover

            inputs_module = __import__(
                "experiments.exp435_train_only_consensus_recovery_antigravity.inputs_antigravity",
                fromlist=["open_exp435_input_session"],
            )
            with patch.object(cli_module, "FROZEN_OUTPUT", output), \
                    patch.object(cli_module, "FROZEN_CHECKPOINT", checkpoint), \
                    patch.object(cli_module, "_clean_head", return_value="f" * 40), \
                    patch.object(inputs_module, "open_exp435_input_session", fail_session):
                with self.assertRaisesRegex(ValueError, "failed.*cannot resume"):
                    cli_module.run_resume()
            after = {
                path.relative_to(root): path.read_bytes()
                for path in sorted(root.rglob("*")) if path.is_file()
            }
            self.assertFalse(opened)
            self.assertEqual(after, before)

    def test_new_rejects_head_changed_by_delayed_imports_before_session(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            readiness = parent / "readiness_antigravity.json"
            readiness.write_bytes(canonical_json_bytes(self._valid_readiness("a" * 40)))
            state = {"imports_done": False, "session_opened": False}

            def clean_head() -> str:
                return "b" * 40 if state["imports_done"] else "a" * 40

            original_import = builtins.__import__

            def import_hook(name, globals=None, locals=None, fromlist=(), level=0):
                result = original_import(name, globals, locals, fromlist, level)
                if name.endswith("inputs_antigravity") or name.endswith("runtime_antigravity"):
                    state["imports_done"] = True
                return result

            inputs_module = __import__(
                "experiments.exp435_train_only_consensus_recovery_antigravity.inputs_antigravity",
                fromlist=["open_exp435_input_session"],
            )

            def fail_session(*args, **kwargs):
                state["session_opened"] = True
                raise AssertionError("input session must not open after a HEAD change")

            with patch.object(cli_module, "FROZEN_READINESS", readiness), \
                    patch.object(cli_module, "_clean_head", side_effect=clean_head), \
                    patch.object(inputs_module, "open_exp435_input_session", fail_session), \
                    patch("builtins.__import__", side_effect=import_hook):
                with self.assertRaises(ValueError):
                    cli_module._run_prediction("new")
            self.assertFalse(state["session_opened"])

    def test_new_passes_exact_final_head_to_input_session(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            readiness = parent / "readiness_antigravity.json"
            readiness.write_bytes(canonical_json_bytes(self._valid_readiness("a" * 40)))
            received: list[str] = []

            @contextmanager
            def fake_session(*args, **kwargs):
                received.append(kwargs["verified_code_commit"])
                yield SimpleNamespace(context={})

            class Summary:
                status = "running"

                def to_payload(self):
                    return {"state": "running"}

            inputs_module = __import__(
                "experiments.exp435_train_only_consensus_recovery_antigravity.inputs_antigravity",
                fromlist=["open_exp435_input_session"],
            )
            runtime_module = __import__(
                "experiments.exp435_train_only_consensus_recovery_antigravity.runtime_antigravity",
                fromlist=["run_exp435_new"],
            )
            with patch.object(cli_module, "FROZEN_READINESS", readiness), \
                    patch.object(cli_module, "_clean_head", side_effect=["a" * 40, "a" * 40]), \
                    patch.object(inputs_module, "open_exp435_input_session", fake_session), \
                    patch.object(runtime_module, "_runtime_context", return_value={}), \
                    patch.object(runtime_module, "run_exp435_new", return_value=Summary()):
                result = cli_module._run_prediction("new")
            self.assertEqual(result["state"], "running")
            self.assertEqual(received, ["a" * 40])

    def test_production_readiness_head_change_writes_no_artifact(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory) / "run_parent"
            readiness = parent / "readiness_antigravity.json"
            lane_hashes = [
                FROZEN_CONFIG.database_sha256,
                FROZEN_CONFIG.exp291_prediction_sha256,
                FROZEN_CONFIG.exp291_manifest_sha256,
                FROZEN_CONFIG.exp292_prediction_sha256,
                FROZEN_CONFIG.exp292_manifest_sha256,
            ]
            with patch.object(cli_module, "RUN_PARENT", parent), \
                    patch.object(cli_module, "FROZEN_READINESS", readiness), \
                    patch.object(cli_module, "FROZEN_OUTPUT", parent / "output_antigravity"), \
                    patch.object(cli_module, "FROZEN_CHECKPOINT", parent / "checkpoint_antigravity"), \
                    patch.object(cli_module, "FROZEN_EVALUATION", parent / "evaluation_antigravity"), \
                    patch.object(cli_module, "_clean_head", side_effect=["a" * 40, "b" * 40]), \
                    patch.object(cli_module, "_sha256_file", side_effect=lane_hashes), \
                    patch.object(cli_module, "_synthetic_mlx_decision", return_value=BackendDecision.cpu_fallback("test")), \
                    patch.object(cli_module, "_focused_test_evidence", return_value={"returncode": 0}), \
                    patch.object(cli_module, "_runtime_versions", return_value={"python": "test"}):
                with self.assertRaises(ValueError):
                    cli_module.run_readiness()
            self.assertFalse(readiness.exists())
            self.assertTrue(parent.is_dir())
            self.assertEqual(tuple(parent.iterdir()), ())

    def test_production_readiness_stable_head_writes_only_canonical_readiness(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory) / "run_parent"
            readiness = parent / "readiness_antigravity.json"
            lane_hashes = [
                FROZEN_CONFIG.database_sha256,
                FROZEN_CONFIG.exp291_prediction_sha256,
                FROZEN_CONFIG.exp291_manifest_sha256,
                FROZEN_CONFIG.exp292_prediction_sha256,
                FROZEN_CONFIG.exp292_manifest_sha256,
            ]
            with patch.object(cli_module, "RUN_PARENT", parent), \
                    patch.object(cli_module, "FROZEN_READINESS", readiness), \
                    patch.object(cli_module, "FROZEN_OUTPUT", parent / "output_antigravity"), \
                    patch.object(cli_module, "FROZEN_CHECKPOINT", parent / "checkpoint_antigravity"), \
                    patch.object(cli_module, "FROZEN_EVALUATION", parent / "evaluation_antigravity"), \
                    patch.object(cli_module, "_clean_head", return_value="a" * 40), \
                    patch.object(cli_module, "_sha256_file", side_effect=lane_hashes), \
                    patch.object(cli_module, "_synthetic_mlx_decision", return_value=BackendDecision.cpu_fallback("test")), \
                    patch.object(cli_module, "_focused_test_evidence", return_value={"returncode": 0}), \
                    patch.object(cli_module, "_runtime_versions", return_value={"python": "test"}):
                payload = cli_module.run_readiness()
            self.assertEqual(readiness.read_bytes(), canonical_json_bytes(payload))
            self.assertEqual(tuple(item.name for item in parent.iterdir()), ("readiness_antigravity.json",))


if __name__ == "__main__":
    unittest.main()
