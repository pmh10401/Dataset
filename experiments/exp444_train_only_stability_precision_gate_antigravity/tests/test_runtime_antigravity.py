"""TDD coverage for the Exp444 parent-owned ordered runtime."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock
from concurrent.futures import Future

import numpy as np

from experiments.exp444_train_only_stability_precision_gate_antigravity import runtime_antigravity as runtime
from experiments.exp444_train_only_stability_precision_gate_antigravity.contracts_antigravity import (
    CandidateInputAntigravity,
    CandidateStabilityResultAntigravity,
    DepthViewScoreAntigravity,
    OmissionMaskAntigravity,
    SourceCandidateAntigravity,
    StabilityViewAntigravity,
)
from experiments.exp444_train_only_stability_precision_gate_antigravity.checkpoint_antigravity import (
    ProcessIdentityAntigravity,
)


def _inputs_antigravity(count: int = 6) -> tuple[CandidateInputAntigravity, ...]:
    values: list[CandidateInputAntigravity] = []
    for sequence in range(count):
        candidate = SourceCandidateAntigravity(
            f"Dataset_{sequence:02d}", sequence, 8, (1, 2), sequence + 1
        )
        train = np.arange(32, dtype="<f4").reshape(4, 8) + sequence
        test = np.arange(8, dtype="<f4") + sequence
        values.append(CandidateInputAntigravity(candidate, (0, 1, 2, 3), train, test))
    return tuple(values)


def _result_antigravity(item: CandidateInputAntigravity) -> CandidateStabilityResultAntigravity:
    score = (
        DepthViewScoreAntigravity(1, 2.0, 1.0, True, 2.0, 0.5),
        DepthViewScoreAntigravity(2, 2.0, 1.0, True, 2.0, 0.5),
    )
    view = StabilityViewAntigravity(
        OmissionMaskAntigravity(0, (0,), hashlib.sha256(b"mask").hexdigest()),
        (1, 2, 3), score, (1, 2), True,
    )
    return CandidateStabilityResultAntigravity(item.candidate, (view,), 1, 1, "stable_review")


class Exp444RuntimeAntigravityTests(unittest.TestCase):
    def test_parallel_and_sequential_bytes_match_antigravity(self) -> None:
        inputs = _inputs_antigravity()
        sequential = runtime.sequential_reference_antigravity(inputs, score_task=_result_antigravity)
        parallel = runtime.ordered_parallel_map_antigravity(inputs, workers=3, score_task=_result_antigravity)
        self.assertEqual(runtime.canonical_results_bytes_antigravity(parallel), runtime.canonical_results_bytes_antigravity(sequential))

    def test_scheduler_preserves_case_sensitive_input_order_and_bounded_pending_antigravity(self) -> None:
        inputs = _inputs_antigravity(20)
        submitted: list[int] = []
        sequence_zero: Future | None = None
        commits: list[int] = []
        snapshots: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

        class ExecutorFixture:
            def __init__(self, **_: object) -> None:
                pass
            def __enter__(self) -> "ExecutorFixture":
                return self
            def __exit__(self, *_: object) -> None:
                return None
            def submit(self, _fn: object, value: CandidateInputAntigravity) -> Future:
                nonlocal sequence_zero
                sequence = value.candidate.original_rank - 1
                submitted.append(sequence)
                future = Future()
                if sequence == 0:
                    sequence_zero = future
                else:
                    future.set_result(_result_antigravity(value))
                if len(submitted) == 9:
                    assert sequence_zero is not None
                    sequence_zero.set_result(_result_antigravity(inputs[0]))
                return future
            def shutdown(self, **_: object) -> None:
                return None

        def progress(*, stage: str, pending: tuple[int, ...], buffered: tuple[int, ...], **_: object) -> None:
            snapshots.append((stage, pending, buffered))

        result = runtime.ordered_parallel_map_antigravity(
            inputs,
            workers=3,
            executor_factory=ExecutorFixture,
            commit=lambda sequence, _candidate, _result: commits.append(sequence),
            progress=progress,
        )
        self.assertEqual(submitted, list(range(20)))
        self.assertTrue(any(0 in pending and buffered for _, pending, buffered in snapshots))
        self.assertLessEqual(max(len(pending) + len(buffered) for _, pending, buffered in snapshots), 9)
        self.assertEqual(commits, list(range(20)))
        self.assertEqual(tuple(item.candidate.original_rank for item in result), tuple(range(1, 21)))

    def test_worker_failure_contains_dataset_candidate_and_view_context_antigravity(self) -> None:
        inputs = _inputs_antigravity(2)

        class ErrorExecutorAntigravity:
            def __init__(self, **_: object) -> None:
                pass
            def submit(self, _fn: object, candidate: CandidateInputAntigravity) -> Future:
                future = Future()
                if candidate.candidate.original_rank == 1:
                    future.set_exception(_ViewFailureAntigravity("view failed"))
                else:
                    future.set_result(_result_antigravity(candidate))
                return future
            def shutdown(self, **_: object) -> None:
                return None

        with self.assertRaises(runtime.WorkerFailureAntigravity) as captured:
            runtime.ordered_parallel_map_antigravity(
                inputs, workers=2, executor_factory=ErrorExecutorAntigravity
            )
        self.assertIn("Dataset_00", str(captured.exception))
        self.assertEqual(captured.exception.candidate_index, 0)
        self.assertEqual(captured.exception.view_index, 7)
        self.assertIn("view=7", str(captured.exception))

    def test_worker_initializer_pins_all_inner_numeric_threads_antigravity(self) -> None:
        names = runtime._THREAD_VARIABLES_ANTIGRAVITY
        for name in names:
            os.environ.pop(name, None)
        runtime._worker_initializer_antigravity()
        self.assertEqual(tuple(os.environ[name] for name in names), ("1",) * 7)

    def test_non_cpu_backend_refuses_outer_pool_antigravity(self) -> None:
        with self.assertRaisesRegex(ValueError, "CPU"):
            runtime.ordered_parallel_map_antigravity(_inputs_antigravity(), distance_backend="mlx")

    def test_no_evaluator_is_imported_by_worker_module_antigravity(self) -> None:
        source = Path(runtime.__file__).read_text()
        self.assertNotIn("evaluation_antigravity", source)
        self.assertNotIn("from .evaluation", source)

    def test_sigint_exact_prefix_resume_and_complete_finalization_antigravity(self) -> None:
        inputs = _inputs_antigravity(30)

        def interrupted_map(active, *, commit, progress, **_kwargs):
            progress(stage="active", sequence=0)
            commit(0, active[0], _result_antigravity(active[0]))
            raise KeyboardInterrupt("fixture SIGINT")

        def completed_map(active, *, commit, progress, **_kwargs):
            results = tuple(_result_antigravity(item) for item in active)
            for sequence, (item, result) in enumerate(zip(active, results, strict=True)):
                progress(stage="active", sequence=sequence)
                commit(sequence, item, result)
            return results

        with tempfile.TemporaryDirectory(prefix="exp444_task7_runtime_antigravity_") as directory:
            root = Path(directory) / "run_antigravity"
            with (
                mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(9_100_001, "start-A")),
                mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)),
                mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=interrupted_map),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    runtime.run_new_antigravity("reviewed", run_root=root)

            attempt = root / "attempts_antigravity/attempt_0000_antigravity"
            state = __import__("json").loads((attempt / "checkpoint_antigravity/state_antigravity.json").read_text())
            progress = __import__("json").loads((attempt / "progress_antigravity.json").read_text())
            self.assertEqual((state["status"], state["completed_units"]), ("interrupted", 1))
            self.assertEqual((progress["stage"], progress["completed"], progress["active"]), ("interrupted", 1, False))

            with (
                mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(9_100_002, "start-B")),
                mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)),
                mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=completed_map),
            ):
                outcome = runtime.run_resume_antigravity(
                    "reviewed", "attempt_0000_antigravity", run_root=root
                )
            self.assertEqual((outcome.status, outcome.completed_count, outcome.resumed), ("complete", 30, True))
            self.assertEqual(tuple(item.candidate.original_rank for item in outcome.results), tuple(range(1, 31)))

            with (
                mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(9_100_003, "start-C")),
                mock.patch.object(runtime, "_load_source_inputs_antigravity", side_effect=AssertionError("complete re-entry must not open source")),
                mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=AssertionError("complete re-entry must not schedule")),
            ):
                final = runtime.run_resume_antigravity(
                    "reviewed", "attempt_0000_antigravity", run_root=root
                )
            self.assertTrue(final.finalization_only)
            self.assertEqual(final.completed_count, 30)

    def test_context_mismatch_rejects_before_progress_mutation_antigravity(self) -> None:
        inputs = _inputs_antigravity(30)

        def interrupted_map(_active, **_kwargs):
            raise KeyboardInterrupt("fixture SIGINT")

        with tempfile.TemporaryDirectory(prefix="exp444_task7_context_antigravity_") as directory:
            root = Path(directory) / "run_antigravity"
            with (
                mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(9_200_001, "start-A")),
                mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)),
                mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=interrupted_map),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    runtime.run_new_antigravity("reviewed", run_root=root)
            attempt = root / "attempts_antigravity/attempt_0000_antigravity"
            context_path = attempt / "checkpoint_antigravity/context_antigravity.json"
            payload = __import__("json").loads(context_path.read_text())
            payload["scientific_config_sha256"] = "0" * 64
            context_path.write_bytes(runtime.canonical_json_bytes_antigravity(payload))
            progress_path = attempt / "progress_antigravity.json"
            before = progress_path.read_bytes()
            with mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(9_200_002, "start-B")):
                with self.assertRaisesRegex(ValueError, "context"):
                    runtime.run_resume_antigravity("reviewed", "attempt_0000_antigravity", run_root=root)
            self.assertEqual(before, progress_path.read_bytes())

    def test_resume_rejects_live_stored_owner_with_real_pid_probe_antigravity(self) -> None:
        """Removing the real PID probes would let a second owner mutate the prefix."""
        inputs = _inputs_antigravity(30)

        def interrupted_map(_active, **_kwargs):
            raise KeyboardInterrupt("fixture SIGINT")

        with tempfile.TemporaryDirectory(prefix="exp444_task7_live_owner_antigravity_") as directory:
            root = Path(directory) / "run_antigravity"
            live_owner = ProcessIdentityAntigravity(os.getpid(), "live-owner-antigravity")
            with (
                mock.patch.object(runtime, "current_process_identity_antigravity", return_value=live_owner),
                mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)),
                mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=interrupted_map),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    runtime.run_new_antigravity("reviewed", run_root=root)

            attempt = root / "attempts_antigravity/attempt_0000_antigravity"
            progress_path = attempt / "progress_antigravity.json"
            checkpoint_path = attempt / "checkpoint_antigravity/state_antigravity.json"
            before = (progress_path.read_bytes(), checkpoint_path.read_bytes())
            replacement = ProcessIdentityAntigravity(os.getpid() + 100000, "replacement-antigravity")
            with (
                mock.patch.object(runtime, "current_process_identity_antigravity", return_value=replacement),
                mock.patch.object(runtime, "_load_source_inputs_antigravity", side_effect=AssertionError("live owner must fail before source access")),
            ):
                with self.assertRaisesRegex(ValueError, "active"):
                    runtime.run_resume_antigravity("reviewed", "attempt_0000_antigravity", run_root=root)
            self.assertEqual(before, (progress_path.read_bytes(), checkpoint_path.read_bytes()))

    def test_live_progress_names_next_commit_candidate_while_worker_is_blocked_antigravity(self) -> None:
        """Moving candidate_start back after result() would restore a null live state."""
        inputs = _inputs_antigravity(30)
        submitted = threading.Event()
        release = threading.Event()
        futures: list[tuple[Future, CandidateInputAntigravity]] = []

        class BlockingExecutorAntigravity:
            def __init__(self, **_: object) -> None:
                pass
            def __enter__(self):
                return self
            def __exit__(self, *_: object) -> None:
                return None
            def submit(self, _fn: object, candidate: CandidateInputAntigravity) -> Future:
                future = Future()
                futures.append((future, candidate))
                if len(futures) == 11:
                    submitted.set()
                if release.is_set():
                    future.set_result(_result_antigravity(candidate))
                return future
            def shutdown(self, **_: object) -> None:
                return None

        with tempfile.TemporaryDirectory(prefix="exp444_task7_live_progress_antigravity_") as directory:
            root = Path(directory) / "run_antigravity"
            errors: list[BaseException] = []

            def target_antigravity() -> None:
                try:
                    runtime.run_new_antigravity(
                        "reviewed", run_root=root, executor_factory=BlockingExecutorAntigravity
                    )
                except BaseException as error:
                    errors.append(error)

            with (
                mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(9_300_001, "start-live-progress")),
                mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)),
            ):
                thread = threading.Thread(target=target_antigravity, daemon=True)
                thread.start()
                self.assertTrue(submitted.wait(2.0), "scheduler did not submit the first worker batch")
                attempt = root / "attempts_antigravity/attempt_0000_antigravity"
                payload = json.loads((attempt / "progress_antigravity.json").read_text())
                self.assertEqual(payload["current_dataset"], "Dataset_00")
                self.assertEqual(payload["current_candidate_index"], 0)
                self.assertEqual(payload["stage"], "candidate")
                self.assertIsNone(payload["current_view"])
                self.assertIsNone(payload["view_count"])
                release.set()
                for future, candidate in tuple(futures):
                    if not future.done():
                        future.set_result(_result_antigravity(candidate))
                thread.join(5.0)
            self.assertFalse(thread.is_alive(), "runtime did not finish after blocked workers were released")
            self.assertEqual(errors, [])

    def test_scheduler_sigint_cancels_without_waiting_for_active_worker_antigravity(self) -> None:
        """Restoring executor context-manager exit would block interruption return."""
        inputs = _inputs_antigravity(2)
        context_exit_called = threading.Event()
        release_context_exit = threading.Event()
        shutdown_calls: list[dict[str, object]] = []

        interrupted = Future()
        interrupted.set_exception(KeyboardInterrupt("fixture SIGINT"))
        active = Future()
        active.set_running_or_notify_cancel()

        class BlockingExitExecutorAntigravity:
            def __init__(self, **_: object) -> None:
                self._futures = [interrupted, active]
            def __enter__(self):
                return self
            def __exit__(self, *_: object) -> None:
                context_exit_called.set()
                release_context_exit.wait(2.0)
            def submit(self, _fn: object, _candidate: CandidateInputAntigravity) -> Future:
                return self._futures.pop(0)
            def shutdown(self, **kwargs: object) -> None:
                shutdown_calls.append(dict(kwargs))

        errors: list[BaseException] = []

        def target_antigravity() -> None:
            try:
                runtime.ordered_parallel_map_antigravity(
                    inputs, workers=2, executor_factory=BlockingExitExecutorAntigravity
                )
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=target_antigravity, daemon=True)
        thread.start()
        thread.join(0.2)
        returned_without_wait = not thread.is_alive()
        release_context_exit.set()
        thread.join(2.0)
        self.assertTrue(returned_without_wait, "SIGINT waited for an active worker")
        self.assertFalse(context_exit_called.is_set(), "scheduler used waiting context-manager shutdown")
        self.assertEqual(shutdown_calls, [{"wait": False, "cancel_futures": True}])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], KeyboardInterrupt)

    def test_complete_reentry_rejects_missing_or_corrupt_attempt_evidence_antigravity(self) -> None:
        """Skipping terminal progress/PID/log validation would accept these attempts."""
        inputs = _inputs_antigravity(30)

        def completed_map(active, *, commit, progress, **_kwargs):
            results = tuple(_result_antigravity(item) for item in active)
            for sequence, (item, result) in enumerate(zip(active, results, strict=True)):
                progress(stage="active", sequence=sequence)
                commit(sequence, item, result)
            return results

        cases = (
            "missing_progress", "active_progress", "wrong_completed",
            "missing_pid", "active_pid", "missing_events", "corrupt_events",
            "missing_text", "empty_text", "bad_checkpoint_hash", "bad_context",
        )
        for offset, case in enumerate(cases):
            with self.subTest(case=case), tempfile.TemporaryDirectory(prefix=f"exp444_task7_complete_{case}_antigravity_") as directory:
                root = Path(directory) / "run_antigravity"
                owner = ProcessIdentityAntigravity(9_400_001 + offset * 2, f"owner-{case}")
                replacement = ProcessIdentityAntigravity(9_400_002 + offset * 2, f"replacement-{case}")
                with (
                    mock.patch.object(runtime, "current_process_identity_antigravity", return_value=owner),
                    mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)),
                    mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=completed_map),
                ):
                    runtime.run_new_antigravity("reviewed", run_root=root)

                attempt = root / "attempts_antigravity/attempt_0000_antigravity"
                progress_path = attempt / "progress_antigravity.json"
                pid_path = attempt / "pid_antigravity.json"
                events_path = attempt / "events_antigravity.jsonl"
                text_path = attempt / "text_antigravity.log"
                state_path = attempt / "checkpoint_antigravity/state_antigravity.json"
                context_path = attempt / "checkpoint_antigravity/context_antigravity.json"
                if case == "missing_progress":
                    progress_path.unlink()
                elif case == "missing_pid":
                    pid_path.unlink()
                elif case in {"active_progress", "wrong_completed"}:
                    payload = json.loads(progress_path.read_text())
                    payload["active" if case == "active_progress" else "completed"] = True if case == "active_progress" else 29
                    progress_path.write_bytes(runtime.canonical_json_bytes_antigravity(payload))
                elif case == "active_pid":
                    payload = json.loads(pid_path.read_text())
                    payload["active"] = True
                    pid_path.write_bytes(runtime.canonical_json_bytes_antigravity(payload))
                elif case == "missing_events":
                    events_path.unlink()
                elif case == "corrupt_events":
                    events_path.write_text("not-json-antigravity\n")
                elif case == "missing_text":
                    text_path.unlink()
                elif case == "empty_text":
                    text_path.write_text("")
                elif case == "bad_checkpoint_hash":
                    payload = json.loads(state_path.read_text())
                    payload["payload_sha256"] = "0" * 64
                    state_path.write_bytes(runtime.canonical_json_bytes_antigravity(payload))
                elif case == "bad_context":
                    payload = json.loads(context_path.read_text())
                    payload["scientific_config_sha256"] = "0" * 64
                    context_path.write_bytes(runtime.canonical_json_bytes_antigravity(payload))
                else:  # pragma: no cover - exhaustive literal cases above
                    raise AssertionError(case)

                with mock.patch.object(runtime, "current_process_identity_antigravity", return_value=replacement):
                    with self.assertRaisesRegex(ValueError, "progress|PID|log|checkpoint|context|complete"):
                        runtime.run_resume_antigravity(
                            "reviewed", "attempt_0000_antigravity", run_root=root
                        )

    def test_resume_rejects_corrupt_committed_authority_before_any_owner_bytes_change_antigravity(self) -> None:
        """Opening progress before full record validation would rebind a corrupt attempt."""
        inputs = _inputs_antigravity(30)

        def interrupted_map(active, *, commit, progress, **_kwargs):
            progress(stage="active", sequence=0)
            commit(0, active[0], _result_antigravity(active[0]))
            raise KeyboardInterrupt("fixture SIGINT")

        for offset, corruption in enumerate(("record", "payload_sha")):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory(prefix=f"exp444_task7_preflight_{corruption}_antigravity_") as directory:
                root = Path(directory) / "run_antigravity"
                owner = ProcessIdentityAntigravity(9_500_001 + offset * 2, f"owner-{corruption}")
                replacement = ProcessIdentityAntigravity(9_500_002 + offset * 2, f"replacement-{corruption}")
                with (
                    mock.patch.object(runtime, "current_process_identity_antigravity", return_value=owner),
                    mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)),
                    mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=interrupted_map),
                ):
                    with self.assertRaises(KeyboardInterrupt):
                        runtime.run_new_antigravity("reviewed", run_root=root)

                attempt = root / "attempts_antigravity/attempt_0000_antigravity"
                checkpoint = attempt / "checkpoint_antigravity"
                if corruption == "record":
                    record_path = checkpoint / "records_antigravity/0000_record_antigravity.json"
                    payload = json.loads(record_path.read_text())
                    payload["result"]["candidate"]["dataset"] = "TamperedDatasetAntigravity"
                    record_path.write_bytes(runtime.canonical_json_bytes_antigravity(payload))
                else:
                    state_path = checkpoint / "state_antigravity.json"
                    payload = json.loads(state_path.read_text())
                    payload["payload_sha256"] = "0" * 64
                    state_path.write_bytes(runtime.canonical_json_bytes_antigravity(payload))
                before = {
                    path.relative_to(attempt).as_posix(): path.read_bytes()
                    for path in sorted(attempt.rglob("*"))
                    if path.is_file()
                }
                with (
                    mock.patch.object(runtime, "current_process_identity_antigravity", return_value=replacement),
                    mock.patch.object(runtime, "_load_source_inputs_antigravity", side_effect=AssertionError("corrupt authority must fail before source access")),
                ):
                    with self.assertRaisesRegex(ValueError, "checkpoint|record|payload"):
                        runtime.run_resume_antigravity(
                            "reviewed", "attempt_0000_antigravity", run_root=root
                        )
                after = {
                    path.relative_to(attempt).as_posix(): path.read_bytes()
                    for path in sorted(attempt.rglob("*"))
                    if path.is_file()
                }
                self.assertEqual(before, after)

    def test_resume_revalidates_checkpoint_at_progress_rebind_boundary_antigravity(self) -> None:
        """A record changed after initial preflight must not permit progress rebinding."""
        inputs = _inputs_antigravity(30)

        def interrupted_map(active, *, commit, progress, **_kwargs):
            progress(stage="active", sequence=0)
            commit(0, active[0], _result_antigravity(active[0]))
            raise KeyboardInterrupt("fixture SIGINT")

        with tempfile.TemporaryDirectory(prefix="exp444_task7_preflight_race_antigravity_") as directory:
            root = Path(directory) / "run_antigravity"
            owner = ProcessIdentityAntigravity(9_600_001, "owner-race")
            replacement = ProcessIdentityAntigravity(9_600_002, "replacement-race")
            with (
                mock.patch.object(runtime, "current_process_identity_antigravity", return_value=owner),
                mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)),
                mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=interrupted_map),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    runtime.run_new_antigravity("reviewed", run_root=root)

            attempt = root / "attempts_antigravity/attempt_0000_antigravity"
            record_path = attempt / "checkpoint_antigravity/records_antigravity/0000_record_antigravity.json"
            original_preflight = runtime.Exp444CheckpointStoreAntigravity.inspect_interrupted_antigravity
            corrupted_authority: dict[str, bytes] = {}
            calls = 0

            def racing_preflight(*args, **kwargs):
                nonlocal calls
                calls += 1
                original_preflight(*args, **kwargs)
                if calls == 1:
                    payload = json.loads(record_path.read_text())
                    payload["result"]["candidate"]["dataset"] = "RacingDatasetAntigravity"
                    record_path.write_bytes(runtime.canonical_json_bytes_antigravity(payload))
                    corrupted_authority.update({
                        path.relative_to(attempt).as_posix(): path.read_bytes()
                        for path in sorted(attempt.rglob("*"))
                        if path.is_file()
                    })

            with (
                mock.patch.object(runtime, "current_process_identity_antigravity", return_value=replacement),
                mock.patch.object(runtime.Exp444CheckpointStoreAntigravity, "inspect_interrupted_antigravity", side_effect=racing_preflight),
                mock.patch.object(runtime, "_load_source_inputs_antigravity", side_effect=AssertionError("racing corruption must fail before source access")),
            ):
                with self.assertRaisesRegex(ValueError, "record"):
                    runtime.run_resume_antigravity(
                        "reviewed", "attempt_0000_antigravity", run_root=root
                    )
            after = {
                path.relative_to(attempt).as_posix(): path.read_bytes()
                for path in sorted(attempt.rglob("*"))
                if path.is_file()
            }
            self.assertGreaterEqual(calls, 2)
            self.assertEqual(corrupted_authority, after)

    def test_resume_reconciles_exact_next_result_orphan_before_scheduling_antigravity(self) -> None:
        """Dropping progress reconciliation leaves checkpoint and progress one apart."""
        inputs = _inputs_antigravity(30)

        def interrupted_map(active, *, commit, progress, **_kwargs):
            progress(stage="active", sequence=0)
            commit(0, active[0], _result_antigravity(active[0]))
            raise KeyboardInterrupt("fixture SIGINT")

        with tempfile.TemporaryDirectory(prefix="exp444_task7_result_orphan_antigravity_") as directory:
            root = Path(directory) / "run_antigravity"
            owner = ProcessIdentityAntigravity(9_700_001, "owner-result-orphan")
            replacement = ProcessIdentityAntigravity(9_700_002, "replacement-result-orphan")
            with (
                mock.patch.object(runtime, "current_process_identity_antigravity", return_value=owner),
                mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)),
                mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=interrupted_map),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    runtime.run_new_antigravity("reviewed", run_root=root)

            attempt = root / "attempts_antigravity/attempt_0000_antigravity"
            orphan_result = _result_antigravity(inputs[1])
            orphan_record = runtime.canonical_json_bytes_antigravity({
                "sequence": 1,
                "result": orphan_result.to_payload(),
            })
            orphan_path = attempt / "checkpoint_antigravity/records_antigravity/0001_record_antigravity.json"
            orphan_path.write_bytes(orphan_record)
            orphan_sha = hashlib.sha256(orphan_record).hexdigest()

            def completed_map(active, *, commit, progress, **_kwargs):
                state = json.loads((attempt / "checkpoint_antigravity/state_antigravity.json").read_text())
                live = json.loads((attempt / "progress_antigravity.json").read_text())
                self.assertEqual((state["completed_units"], live["completed"]), (2, 2))
                self.assertEqual((live["recomputed"], live["success"], live["error"]), (2, 2, 0))
                self.assertEqual(active[0].candidate.original_rank, 3)
                results = tuple(_result_antigravity(item) for item in active)
                for sequence, (candidate, result) in enumerate(zip(active, results, strict=True)):
                    progress(stage="active", sequence=sequence)
                    commit(sequence, candidate, result)
                return results

            with (
                mock.patch.object(runtime, "current_process_identity_antigravity", return_value=replacement),
                mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)),
                mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=completed_map),
            ):
                outcome = runtime.run_resume_antigravity(
                    "reviewed", "attempt_0000_antigravity", run_root=root
                )
            state = json.loads((attempt / "checkpoint_antigravity/state_antigravity.json").read_text())
            progress_payload = json.loads((attempt / "progress_antigravity.json").read_text())
            events = [json.loads(line) for line in (attempt / "events_antigravity.jsonl").read_text().splitlines()]
            reconciled = next(event for event in events if event["event"] == "orphan_reconciled")
            self.assertEqual((outcome.status, outcome.completed_count), ("complete", 30))
            self.assertEqual((state["completed_units"], progress_payload["completed"]), (30, 30))
            self.assertEqual(reconciled["record_sha256"], orphan_sha)
            self.assertIn("고아 레코드 복구", (attempt / "text_antigravity.log").read_text())

    def test_resume_reconciles_exact_next_error_orphan_to_terminal_error_antigravity(self) -> None:
        """A durable error orphan must terminate both authorities without scheduling."""
        inputs = _inputs_antigravity(30)

        def interrupted_map(active, *, commit, progress, **_kwargs):
            progress(stage="active", sequence=0)
            commit(0, active[0], _result_antigravity(active[0]))
            raise KeyboardInterrupt("fixture SIGINT")

        with tempfile.TemporaryDirectory(prefix="exp444_task7_error_orphan_antigravity_") as directory:
            root = Path(directory) / "run_antigravity"
            owner = ProcessIdentityAntigravity(9_800_001, "owner-error-orphan")
            replacement = ProcessIdentityAntigravity(9_800_002, "replacement-error-orphan")
            with (
                mock.patch.object(runtime, "current_process_identity_antigravity", return_value=owner),
                mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)),
                mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=interrupted_map),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    runtime.run_new_antigravity("reviewed", run_root=root)

            attempt = root / "attempts_antigravity/attempt_0000_antigravity"
            orphan_record = runtime.canonical_json_bytes_antigravity({
                "sequence": 1,
                "result": None,
                "status": "error",
                "error": {"type": "RuntimeError", "message": "orphan worker failed"},
            })
            orphan_path = attempt / "checkpoint_antigravity/records_antigravity/0001_record_antigravity.json"
            orphan_path.write_bytes(orphan_record)
            with (
                mock.patch.object(runtime, "current_process_identity_antigravity", return_value=replacement),
                mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)),
                mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=AssertionError("terminal orphan must not schedule")),
            ):
                with self.assertRaises(runtime.WorkerFailureAntigravity) as captured:
                    runtime.run_resume_antigravity(
                        "reviewed", "attempt_0000_antigravity", run_root=root
                    )
            state = json.loads((attempt / "checkpoint_antigravity/state_antigravity.json").read_text())
            progress_payload = json.loads((attempt / "progress_antigravity.json").read_text())
            events = [json.loads(line) for line in (attempt / "events_antigravity.jsonl").read_text().splitlines()]
            self.assertEqual(captured.exception.candidate_index, 1)
            self.assertIn("orphan worker failed", str(captured.exception))
            self.assertEqual((state["status"], state["completed_units"], state["error_count"]), ("error", 2, 1))
            self.assertEqual((progress_payload["stage"], progress_payload["completed"], progress_payload["error"]), ("error", 2, 1))
            self.assertEqual((progress_payload["recomputed"], progress_payload["success"]), (2, 1))
            self.assertEqual(events[-2]["event"], "orphan_reconciled")
            self.assertEqual(events[-1]["event"], "terminal")


class _ViewFailureAntigravity(RuntimeError):
    view_index = 7


if __name__ == "__main__":
    unittest.main()
