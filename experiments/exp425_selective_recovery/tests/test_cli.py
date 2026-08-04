"""CLI mode and read-only inspection tests for Exp425."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.exp425_selective_recovery.checkpoints import Exp425CheckpointStore
from experiments.exp425_selective_recovery.runtime import AtomicProgressWriter, Exp425RunCounts, RuntimeProgress
from experiments.exp425_selective_recovery.tests.test_runtime import (
    _commit_runtime_import,
    _runtime_context,
    _runtime_import_record,
    _runtime_provenance,
)
from experiments.exp417_causal_multihorizon_shadow import contracts as exp417_contracts
from experiments.exp417_causal_multihorizon_shadow import config as exp417_config
from experiments.exp417_causal_multihorizon_shadow.contracts import BackendAudit
from experiments.exp425_selective_recovery.database import Exp425DatabaseSnapshot


class _SnapshotProbe(Exp425DatabaseSnapshot):
    """Typed launch placeholder: PID rejection must happen before DB use."""

    def close(self) -> None:
        return None


def _load_cli_module() -> object:
    root = Path(__file__).resolve().parents[3]
    entrypoint = root / "scratch" / "run_exp425_selective_recovery.py"
    specification = importlib.util.spec_from_file_location("exp425_cli_under_test", entrypoint)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _tree_state(root: Path) -> dict[str, tuple[bytes, int, int]]:
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


@contextmanager
def _tiny_runtime_config() -> object:
    """Keep the real checkpoint decoder on the tiny fixture's frozen lanes."""
    config = _runtime_context().config
    lane_defaults = list(exp417_contracts.FrozenLaneSnapshot.__init__.__defaults__)
    lane_defaults[-2] = config
    with (
        patch.object(exp417_contracts, "FROZEN_CONFIG", config),
        patch.object(exp417_config, "FROZEN_CONFIG", config),
        patch.object(exp417_config.Exp417FrozenConfig, "validate", lambda _self: None),
        patch.object(exp417_contracts.FrozenLaneSnapshot.__init__, "__defaults__", tuple(lane_defaults)),
        patch.object(exp417_contracts.FrozenRunContext.__init__, "__defaults__", (config,)),
    ):
        yield


def _runtime_tree(directory: Path, *, state: str, complete: bool) -> tuple[Path, Path, str]:
    """Build a real small checkpoint/output pair without model or DB work."""
    directory.mkdir(parents=True, exist_ok=True)
    output, checkpoint = directory / "output", directory / "checkpoints"
    context = _runtime_context()
    provenance = _runtime_provenance(context, "cli-runtime")
    store = Exp425CheckpointStore.create(checkpoint, provenance, context)
    if complete:
        for dataset in context.datasets:
            _commit_runtime_import(store, provenance, _runtime_import_record(dataset))
    writer = AtomicProgressWriter.create(
        output_root=output, run_identity=provenance.run_identity, command="new",
        checkpoint_root=checkpoint, total_datasets=len(context.datasets), backend="mps",
    )
    inspection = Exp425CheckpointStore.inspect_read_only(checkpoint, expected_run_identity=provenance.run_identity)
    for ordinal, (dataset, origin, status, reason, elapsed) in enumerate(inspection.record_summaries, 1):
        writer.write(
            RuntimeProgress.initializing(writer.paths, provenance.run_identity, "new", len(context.datasets), "mps"),
            event="dataset_terminal",
            detail={"dataset": dataset, "ordinal": ordinal, "origin": origin,
                    "status": status, "reason": reason, "elapsed_seconds": elapsed},
        )
    writer.write(RuntimeProgress.initializing(writer.paths, provenance.run_identity, "new", len(context.datasets), "mps"), event="phase_change")
    if state != "initializing":
        completed = len(inspection.record_summaries)
        writer.write(
            RuntimeProgress(
                writer.paths, provenance.run_identity, "new", len(context.datasets), "mps",
                state=state, stage=state,
                counts=Exp425RunCounts(len(context.datasets), completed=completed, imported=completed),
                terminal_samples=tuple(elapsed for _, _, _, _, elapsed in inspection.record_summaries),
            ),
            event="failure" if state == "failed" else "phase_change",
        )
    writer.close()
    return output, checkpoint, provenance.run_identity


class Exp425CliTests(unittest.TestCase):
    def _direct_main(self, argv: list[str]) -> tuple[int, str, str]:
        module = _load_cli_module()
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = module.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_cli_requires_one_of_the_explicit_modes(self) -> None:
        root = Path(__file__).resolve().parents[3]
        entrypoint = root / "scratch" / "run_exp425_selective_recovery.py"
        result = subprocess.run([sys.executable, str(entrypoint)], cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("new", result.stderr)

    def test_new_existing_root_fails_before_launch_factory_import(self) -> None:
        root = Path(__file__).resolve().parents[3]
        entrypoint = root / "scratch" / "run_exp425_selective_recovery.py"
        with TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            result = subprocess.run(
                [sys.executable, str(entrypoint), "new", "--output-root", str(output),
                 "--checkpoint-root", str(Path(directory) / "checkpoints"),
                 "--launch-factory", "does_not_exist:factory"],
                cwd=root, capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("already exists", result.stderr)
        self.assertNotIn("does_not_exist", result.stderr)

    def test_cli_rejects_unknown_mode(self) -> None:
        root = Path(__file__).resolve().parents[3]
        entrypoint = root / "scratch" / "run_exp425_selective_recovery.py"
        with TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(entrypoint), "readiness", "--output-root", str(Path(directory) / "output"),
                 "--checkpoint-root", str(Path(directory) / "checkpoint")],
                cwd=root, capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)

    def test_new_existing_checkpoint_root_fails_before_launch_factory_import(self) -> None:
        root = Path(__file__).resolve().parents[3]
        entrypoint = root / "scratch" / "run_exp425_selective_recovery.py"
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "existing-checkpoint"
            checkpoint.mkdir()
            result = subprocess.run(
                [sys.executable, str(entrypoint), "new", "--output-root", str(Path(directory) / "output"),
                 "--checkpoint-root", str(checkpoint), "--launch-factory", "does_not_exist:factory"],
                cwd=root, capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("already exists", result.stderr)
        self.assertNotIn("does_not_exist", result.stderr)

    def test_resume_absent_roots_fails_before_launch_factory_import(self) -> None:
        root = Path(__file__).resolve().parents[3]
        entrypoint = root / "scratch" / "run_exp425_selective_recovery.py"
        with TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(entrypoint), "resume", "--output-root", str(Path(directory) / "output"),
                 "--checkpoint-root", str(Path(directory) / "checkpoint"), "--launch-factory", "does_not_exist:factory"],
                cwd=root, capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("is absent", result.stderr)
        self.assertNotIn("does_not_exist", result.stderr)

    def test_inspect_returns_one_for_validated_incomplete_and_failed_states(self) -> None:
        with _tiny_runtime_config():
            with TemporaryDirectory() as directory:
                output, checkpoint, identity = _runtime_tree(Path(directory), state="initializing", complete=False)
                code, stdout, stderr = self._direct_main([
                    "inspect", "--output-root", str(output), "--checkpoint-root", str(checkpoint), "--run-identity", identity,
                ])
                self.assertEqual(code, 1, stderr)
                self.assertEqual(json.loads(stdout)["state"], "initializing")
            with TemporaryDirectory() as directory:
                output, checkpoint, identity = _runtime_tree(Path(directory), state="failed", complete=False)
                code, stdout, stderr = self._direct_main([
                    "inspect", "--output-root", str(output), "--checkpoint-root", str(checkpoint), "--run-identity", identity,
                ])
                self.assertEqual(code, 1, stderr)
                self.assertEqual(json.loads(stdout)["state"], "failed")

    def test_inspect_requires_matching_output_checkpoint_and_run_identity(self) -> None:
        with _tiny_runtime_config():
            with TemporaryDirectory() as directory:
                root = Path(directory)
                output, checkpoint, identity = _runtime_tree(root / "first", state="complete_review_only", complete=True)
                _, foreign_checkpoint, _ = _runtime_tree(root / "second", state="complete_review_only", complete=True)
                code, _stdout, stderr = self._direct_main([
                    "inspect", "--output-root", str(output), "--checkpoint-root", str(foreign_checkpoint), "--run-identity", identity,
                ])
                self.assertEqual(code, 2)
                self.assertIn("does not bind", stderr)
                code, _stdout, stderr = self._direct_main([
                    "inspect", "--output-root", str(output), "--checkpoint-root", str(checkpoint), "--run-identity", "foreign-run",
                ])
                self.assertEqual(code, 2)
                self.assertIn("identity", stderr)

    def test_complete_inspect_is_read_only_and_never_enters_launch_or_mutating_paths(self) -> None:
        with _tiny_runtime_config():
            with TemporaryDirectory() as directory:
                output, checkpoint, identity = _runtime_tree(Path(directory), state="complete_review_only", complete=True)
                before = _tree_state(Path(directory))
                module = _load_cli_module()
                stdout, stderr = io.StringIO(), io.StringIO()
                with (
                    patch.object(module, "_factory", side_effect=AssertionError("launch factory invoked")),
                    patch.object(module, "execute_runtime_command", side_effect=AssertionError("runtime launch invoked")),
                    patch("experiments.exp425_selective_recovery.checkpoints.Exp425CheckpointStore.open", side_effect=AssertionError("checkpoint open invoked")),
                    patch("experiments.exp425_selective_recovery.runtime.AtomicProgressWriter.claim_pid", side_effect=AssertionError("PID claim invoked")),
                    patch("experiments.exp425_selective_recovery.database.open_database_snapshot", side_effect=AssertionError("database snapshot invoked")),
                    redirect_stdout(stdout), redirect_stderr(stderr),
                ):
                    code = module.main([
                        "inspect", "--output-root", str(output), "--checkpoint-root", str(checkpoint), "--run-identity", identity,
                    ])
                self.assertEqual(code, 0, stderr.getvalue())
                self.assertEqual(json.loads(stdout.getvalue())["state"], "complete_review_only")
                self.assertEqual(before, _tree_state(Path(directory)))

    def test_complete_resume_returns_zero_without_launch_factory_or_pid_claim(self) -> None:
        with _tiny_runtime_config():
            with TemporaryDirectory() as directory:
                output, checkpoint, identity = _runtime_tree(Path(directory), state="complete_review_only", complete=True)
                before = _tree_state(Path(directory))
                module = _load_cli_module()
                stdout, stderr = io.StringIO(), io.StringIO()
                with (
                    patch.object(module, "_factory", side_effect=AssertionError("launch factory invoked")),
                    patch.object(module, "execute_runtime_command", side_effect=AssertionError("runtime launch invoked")),
                    patch("experiments.exp425_selective_recovery.runtime.AtomicProgressWriter.claim_pid", side_effect=AssertionError("PID claim invoked")),
                    redirect_stdout(stdout), redirect_stderr(stderr),
                ):
                    code = module.main([
                        "resume", "--output-root", str(output), "--checkpoint-root", str(checkpoint), "--run-identity", identity,
                    ])
                self.assertEqual(code, 0, stderr.getvalue())
                published = json.loads(stdout.getvalue())
                self.assertEqual(published["state"], "complete_review_only")
                self.assertEqual(published["output_root"], str(output.absolute()))
                self.assertEqual(before, _tree_state(Path(directory)))

    def test_resume_mismatch_rejects_before_launch_factory(self) -> None:
        with _tiny_runtime_config():
            with TemporaryDirectory() as directory:
                root = Path(directory)
                output, _checkpoint, identity = _runtime_tree(root / "first", state="initializing", complete=False)
                _, foreign_checkpoint, _ = _runtime_tree(root / "second", state="initializing", complete=False)
                module = _load_cli_module()
                stdout, stderr = io.StringIO(), io.StringIO()
                with (
                    patch.object(module, "_factory", side_effect=AssertionError("launch factory invoked")),
                    redirect_stdout(stdout), redirect_stderr(stderr),
                ):
                    code = module.main([
                        "resume", "--output-root", str(output), "--checkpoint-root", str(foreign_checkpoint),
                        "--run-identity", identity, "--launch-factory", "unused:factory",
                    ])
                self.assertEqual(code, 2)
                self.assertIn("does not bind", stderr.getvalue())
                self.assertEqual(stdout.getvalue(), "")

    def test_resume_rejects_live_and_foreign_pid_leases_after_preflight(self) -> None:
        with _tiny_runtime_config():
            with TemporaryDirectory() as directory:
                output, checkpoint, identity = _runtime_tree(Path(directory), state="initializing", complete=False)
                context = _runtime_context()
                provenance = _runtime_provenance(context, identity)
                audit = BackendAudit("mps", True, 0.0, 1.0, 1.1, False)
                launch = {
                    "provenance": provenance, "context": context,
                    "snapshot": object.__new__(_SnapshotProbe), "audit": audit,
                }
                for lease_identity, expected in ((identity, "owned by a live process"), ("foreign-run", "does not match this run")):
                    (output / "run.pid").write_bytes(json.dumps({
                        "schema": "exp425.runtime-pid.v1", "run_identity": lease_identity,
                        "pid": os.getpid(), "nonce": "lease-nonce",
                    }, sort_keys=True, separators=(",", ":")).encode())
                    module = _load_cli_module()
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with (
                        patch.object(module, "_factory", return_value=lambda: launch),
                        redirect_stdout(stdout), redirect_stderr(stderr),
                    ):
                        code = module.main([
                            "resume", "--output-root", str(output), "--checkpoint-root", str(checkpoint),
                            "--run-identity", identity, "--launch-factory", "fixture:launch",
                        ])
                    self.assertEqual(code, 1, stderr.getvalue())
                    self.assertIn(expected, stderr.getvalue())
                    self.assertEqual(json.loads(stdout.getvalue())["pid_path"], str((output / "run.pid").absolute()))
                    (output / "run.pid").unlink()
