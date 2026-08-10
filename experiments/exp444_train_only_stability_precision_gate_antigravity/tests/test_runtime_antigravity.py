"""TDD coverage for the Exp444 parent-owned ordered runtime."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

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
        inputs = _inputs_antigravity()
        observed: list[int] = []
        peak = 0

        class FutureFixture:
            def __init__(self, value: CandidateInputAntigravity) -> None:
                self.value = value
                self.done_flag = True
            def done(self) -> bool:
                return self.done_flag
            def result(self) -> CandidateStabilityResultAntigravity:
                observed.append(self.value.candidate.original_rank)
                return _result_antigravity(self.value)
            def cancel(self) -> bool:
                return True

        class ExecutorFixture:
            def __init__(self, **_: object) -> None:
                self.pending: list[FutureFixture] = []
            def __enter__(self) -> "ExecutorFixture":
                return self
            def __exit__(self, *_: object) -> None:
                return None
            def submit(self, _fn: object, value: CandidateInputAntigravity) -> FutureFixture:
                nonlocal peak
                future = FutureFixture(value)
                self.pending.append(future)
                peak = max(peak, len(self.pending))
                return future
            def shutdown(self, **_: object) -> None:
                return None

        with mock.patch.object(runtime, "ProcessPoolExecutor", ExecutorFixture):
            result = runtime.ordered_parallel_map_antigravity(inputs, workers=3)
        self.assertEqual(tuple(item.candidate.original_rank for item in result), tuple(range(1, 7)))
        self.assertEqual(tuple(sorted(observed)), tuple(range(1, 7)))
        self.assertLessEqual(peak, 3 * 3)

    def test_worker_failure_contains_dataset_candidate_and_view_context_antigravity(self) -> None:
        inputs = _inputs_antigravity(2)
        with self.assertRaises(runtime.WorkerFailureAntigravity) as captured:
            runtime.sequential_reference_antigravity(inputs, score_task=_raising_antigravity)
        self.assertIn("Dataset_00", str(captured.exception))
        self.assertEqual(captured.exception.candidate_index, 0)

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

        def interrupted_map(active, *, commit, **_kwargs):
            commit(0, active[0], _result_antigravity(active[0]))
            raise KeyboardInterrupt("fixture SIGINT")

        def completed_map(active, *, commit, **_kwargs):
            results = tuple(_result_antigravity(item) for item in active)
            for sequence, (item, result) in enumerate(zip(active, results, strict=True)):
                commit(sequence, item, result)
            return results

        with tempfile.TemporaryDirectory(prefix="exp444_task7_runtime_antigravity_") as directory:
            root = Path(directory) / "run_antigravity"
            with (
                mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(61001, "start-A")),
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
                mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(61002, "start-B")),
                mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)),
                mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=completed_map),
            ):
                outcome = runtime.run_resume_antigravity(
                    "reviewed", "attempt_0000_antigravity", run_root=root
                )
            self.assertEqual((outcome.status, outcome.completed_count, outcome.resumed), ("complete", 30, True))
            self.assertEqual(tuple(item.candidate.original_rank for item in outcome.results), tuple(range(1, 31)))

            with (
                mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(61003, "start-C")),
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
                mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(62001, "start-A")),
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
            with mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(62002, "start-B")):
                with self.assertRaisesRegex(ValueError, "context"):
                    runtime.run_resume_antigravity("reviewed", "attempt_0000_antigravity", run_root=root)
            self.assertEqual(before, progress_path.read_bytes())


if __name__ == "__main__":
    unittest.main()


def _raising_antigravity(_candidate: CandidateInputAntigravity) -> CandidateStabilityResultAntigravity:
    raise RuntimeError("view failed")
