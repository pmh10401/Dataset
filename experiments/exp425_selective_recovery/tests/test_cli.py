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
from types import SimpleNamespace
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


def _corrupt_progress(output: Path, **changes: object) -> None:
    progress_path = output / "progress.json"
    payload = json.loads(progress_path.read_text())
    payload.update(changes)
    progress_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _checkpoint_ahead_of_progress_tree(directory: Path) -> tuple[Path, Path, str]:
    """Simulate the allowed checkpoint-commit → terminal-event crash boundary."""
    output, checkpoint, identity = _runtime_tree(directory, state="complete_review_only", complete=True)
    events_path = output / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    terminal_seen = 0
    retained: list[dict[str, object]] = []
    for event in events:
        if event.get("event") == "dataset_terminal":
            terminal_seen += 1
            if terminal_seen == 3:
                continue
        retained.append(event)
    events_path.write_text("".join(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n" for event in retained))
    _corrupt_progress(
        output,
        state="running", stage="running", current_dataset=None, current_phase=None,
        completed_datasets=2, remaining_datasets=1, imported_datasets=2,
        native_datasets=0, recomputed_datasets=0, abstained_datasets=0,
        oof_identity_insufficient_datasets=0, failed_datasets=0,
    )
    return output, checkpoint, identity


def _sparse_imported_resume_tree(directory: Path, *, command: str = "resume", state: str = "running") -> tuple[Path, Path, str]:
    """A real sparse Exp424 import set after restore, before terminal replay."""
    directory.mkdir(parents=True, exist_ok=True)
    output, checkpoint = directory / "output", directory / "checkpoints"
    context = _runtime_context()
    provenance = _runtime_provenance(context, "sparse-import-resume")
    store = Exp425CheckpointStore.create(checkpoint, provenance, context)
    for dataset in (context.datasets[0], context.datasets[2]):
        _commit_runtime_import(store, provenance, _runtime_import_record(dataset))
    writer = AtomicProgressWriter.create(output_root=output, run_identity=provenance.run_identity,
                                         command=command, checkpoint_root=checkpoint,
                                         total_datasets=len(context.datasets), backend="mps")
    records = tuple(store.iter_records())
    counts = Exp425RunCounts(len(context.datasets), completed=len(records), imported=len(records))
    writer.write(RuntimeProgress(writer.paths, provenance.run_identity, command, len(context.datasets), "mps",
                                 state=state, stage=state, counts=counts), event="phase_change")
    writer.close()
    return output, checkpoint, provenance.run_identity


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
    def test_resume_accepts_restored_sparse_import_counters_without_terminal_events(self) -> None:
        """A restore-published full checkpoint count is legitimate only for resume/running."""
        with _tiny_runtime_config(), TemporaryDirectory() as directory:
            output, checkpoint, identity = _sparse_imported_resume_tree(Path(directory))
            module = _load_cli_module()
            runtime, inspected = module._inspect_pair(output, checkpoint, identity)
            self.assertEqual((runtime["command"], runtime["completed_datasets"], inspected.completed_count), ("resume", 2, 2))

    def test_sparse_full_unlogged_counters_reject_new_and_importing_states(self) -> None:
        with _tiny_runtime_config(), TemporaryDirectory() as directory:
            module = _load_cli_module()
            for command, state in (("new", "running"), ("resume", "importing")):
                output, checkpoint, identity = _sparse_imported_resume_tree(Path(directory) / f"{command}-{state}", command=command, state=state)
                with self.assertRaisesRegex(ValueError, "counters"):
                    module._inspect_pair(output, checkpoint, identity)
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
                    "snapshot": object.__new__(_SnapshotProbe), "audit": audit, "adoption": object(),
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

    def test_launch_factory_contract_errors_return_two(self) -> None:
        for specification, expected in (("not-a-factory", "module:callable"), ("math:pi", "not callable")):
            with self.subTest(specification=specification), TemporaryDirectory() as directory:
                code, _stdout, stderr = self._direct_main([
                    "new", "--output-root", str(Path(directory) / "output"),
                    "--checkpoint-root", str(Path(directory) / "checkpoint"),
                    "--launch-factory", specification,
                ])
                self.assertEqual(code, 2)
                self.assertIn(expected, stderr)
        with TemporaryDirectory() as directory:
            module = _load_cli_module()
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                patch.object(module, "_factory", return_value=lambda: ()),
                redirect_stdout(stdout), redirect_stderr(stderr),
            ):
                code = module.main([
                    "new", "--output-root", str(Path(directory) / "output"),
                    "--checkpoint-root", str(Path(directory) / "checkpoint"),
                    "--launch-factory", "fixture:launch",
                ])
            self.assertEqual(code, 2)
            self.assertIn("must return a mapping", stderr.getvalue())
        with TemporaryDirectory() as directory:
            module = _load_cli_module()
            stdout, stderr = io.StringIO(), io.StringIO()
            with (
                patch.object(module, "_factory", return_value=lambda: (_ for _ in ()).throw(RuntimeError("assembly failed"))),
                redirect_stdout(stdout), redirect_stderr(stderr),
            ):
                code = module.main([
                    "new", "--output-root", str(Path(directory) / "output"),
                    "--checkpoint-root", str(Path(directory) / "checkpoint"),
                    "--launch-factory", "fixture:launch",
                ])
            self.assertEqual(code, 2)
            self.assertIn("assembly failed", stderr.getvalue())

    def test_inspect_rejects_unknown_state_without_mutating_tree(self) -> None:
        with _tiny_runtime_config(), TemporaryDirectory() as directory:
            output, checkpoint, identity = _runtime_tree(Path(directory), state="complete_review_only", complete=True)
            _corrupt_progress(output, state="unknown_state", stage="unknown_state")
            before = _tree_state(Path(directory))
            code, stdout, stderr = self._direct_main([
                "inspect", "--output-root", str(output), "--checkpoint-root", str(checkpoint), "--run-identity", identity,
            ])
            self.assertEqual(code, 2)
            self.assertEqual(stdout, "")
            self.assertIn("state", stderr)
            self.assertEqual(before, _tree_state(Path(directory)))

    def test_inspect_rejects_complete_counter_corruption_without_mutating_tree(self) -> None:
        corruptions = {
            "completed_datasets": 2, "remaining_datasets": 1, "imported_datasets": 2,
            "native_datasets": 1, "recomputed_datasets": 1, "abstained_datasets": 1,
            "oof_identity_insufficient_datasets": 1, "failed_datasets": 1,
        }
        with _tiny_runtime_config():
            for field, value in corruptions.items():
                with self.subTest(field=field), TemporaryDirectory() as directory:
                    output, checkpoint, identity = _runtime_tree(Path(directory), state="complete_review_only", complete=True)
                    _corrupt_progress(output, **{field: value})
                    before = _tree_state(Path(directory))
                    code, stdout, stderr = self._direct_main([
                        "inspect", "--output-root", str(output), "--checkpoint-root", str(checkpoint), "--run-identity", identity,
                    ])
                    self.assertEqual(code, 2)
                    self.assertEqual(stdout, "")
                    self.assertIn("counter", stderr)
                    self.assertEqual(before, _tree_state(Path(directory)))

    def test_inspect_rejects_impossible_incomplete_accounting_without_mutating_tree(self) -> None:
        with _tiny_runtime_config(), TemporaryDirectory() as directory:
            output, checkpoint, identity = _runtime_tree(Path(directory), state="initializing", complete=False)
            _corrupt_progress(output, completed_datasets=1, remaining_datasets=2, imported_datasets=0, native_datasets=0)
            before = _tree_state(Path(directory))
            code, stdout, stderr = self._direct_main([
                "inspect", "--output-root", str(output), "--checkpoint-root", str(checkpoint), "--run-identity", identity,
            ])
            self.assertEqual(code, 2)
            self.assertEqual(stdout, "")
            self.assertIn("counter", stderr)
            self.assertEqual(before, _tree_state(Path(directory)))

    def test_checkpoint_ahead_of_nonterminal_progress_is_resumable_and_inspectable(self) -> None:
        with _tiny_runtime_config(), TemporaryDirectory() as directory:
            output, checkpoint, identity = _checkpoint_ahead_of_progress_tree(Path(directory))
            before = _tree_state(Path(directory))
            code, stdout, stderr = self._direct_main([
                "inspect", "--output-root", str(output), "--checkpoint-root", str(checkpoint), "--run-identity", identity,
            ])
            self.assertEqual(code, 1, stderr)
            self.assertEqual(json.loads(stdout)["state"], "running")
            self.assertEqual(before, _tree_state(Path(directory)))
            module = _load_cli_module()
            stdout, stderr = io.StringIO(), io.StringIO()
            launch = {"provenance": object(), "context": object(), "snapshot": object(), "audit": object(), "adoption": object()}
            with (
                patch.object(module, "_factory", return_value=lambda: launch),
                patch.object(module, "execute_runtime_command", return_value=SimpleNamespace(state="failed")) as execute,
                redirect_stdout(stdout), redirect_stderr(stderr),
            ):
                code = module.main([
                    "resume", "--output-root", str(output), "--checkpoint-root", str(checkpoint),
                    "--run-identity", identity, "--launch-factory", "fixture:launch",
                ])
            self.assertEqual(code, 1, stderr.getvalue())
            execute.assert_called_once()

    def test_nonterminal_progress_rejects_event_prefix_accounting_mismatch(self) -> None:
        with _tiny_runtime_config(), TemporaryDirectory() as directory:
            output, checkpoint, identity = _checkpoint_ahead_of_progress_tree(Path(directory))
            _corrupt_progress(output, completed_datasets=1, remaining_datasets=2, imported_datasets=1)
            before = _tree_state(Path(directory))
            code, stdout, stderr = self._direct_main([
                "inspect", "--output-root", str(output), "--checkpoint-root", str(checkpoint), "--run-identity", identity,
            ])
            self.assertEqual(code, 2)
            self.assertEqual(stdout, "")
            self.assertIn("counter", stderr)
            self.assertEqual(before, _tree_state(Path(directory)))

    def test_nonterminal_progress_rejects_nonprefix_checkpoint_suffix(self) -> None:
        with _tiny_runtime_config(), TemporaryDirectory() as directory:
            output, checkpoint, identity = _checkpoint_ahead_of_progress_tree(Path(directory))
            events_path = output / "events.jsonl"
            events = [json.loads(line) for line in events_path.read_text().splitlines()]
            terminals = [index for index, event in enumerate(events) if event.get("event") == "dataset_terminal"]
            events[terminals[0]], events[terminals[1]] = events[terminals[1]], events[terminals[0]]
            events_path.write_text("".join(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n" for event in events))
            before = _tree_state(Path(directory))
            code, stdout, _stderr = self._direct_main([
                "inspect", "--output-root", str(output), "--checkpoint-root", str(checkpoint), "--run-identity", identity,
            ])
            self.assertEqual(code, 2)
            self.assertEqual(stdout, "")
            self.assertEqual(before, _tree_state(Path(directory)))
