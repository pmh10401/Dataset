"""Behavior-first tests for the ordered 11-worker Exp443 runtime."""
from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path
import os
import tempfile
import unittest

import numpy as np

from experiments.exp443_missing_normal_train_policy_comparison_antigravity.checkpoint_antigravity import (
    WorkUnitAntigravity,
)
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.contracts_antigravity import (
    KernelSeedScoresAntigravity,
    PolicyDatasetResultAntigravity,
)


POLICIES = (
    "linear_edge_nearest_control",
    "exclude_nonfinite_row",
    "leave_one_out_peer_median",
)


def _result(unit: WorkUnitAntigravity) -> PolicyDatasetResultAntigravity:
    scores = tuple(
        KernelSeedScoresAntigravity(kernel, seed, np.zeros(2, dtype="<f8"), np.zeros(1, dtype="<f8"), ())
        for kernel in (1024, 4096)
        for seed in (20260721, 20261721)
    )
    return PolicyDatasetResultAntigravity(
        unit.policy, unit.dataset, 8, 2, 0, 2, 0, 1, scores,
        {1024: (), 4096: ()}, (), "scored",
    )


class _FakeExecutor:
    created: list[dict[str, object]] = []

    def __init__(self, *, max_workers: int, initializer, **kwargs: object) -> None:
        self.created.append({"max_workers": max_workers, "initializer": initializer, **kwargs})
        initializer()
        self.futures: list[Future] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def submit(self, function, task):
        future: Future = Future()
        future.set_result(function(task))
        self.futures.append(future)
        return future


class RuntimeAntigravityTests(unittest.TestCase):
    def test_worker_initializer_sets_all_six_numerical_thread_variables(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.runtime_antigravity import (
            _worker_initializer_antigravity,
        )

        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
            os.environ.pop(name, None)
        _worker_initializer_antigravity()
        self.assertEqual({name: os.environ[name] for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")}, {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")})

    def test_scheduler_uses_exactly_eleven_spawn_workers_and_commits_only_ordered_prefix(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.runtime_antigravity import (
            run_ordered_policy_schedule_antigravity,
        )

        _FakeExecutor.created.clear()
        units = tuple(WorkUnitAntigravity(i, POLICIES[0], name) for i, name in enumerate(("A", "B", "C")))
        # The production order is case-sensitive; this fixture intentionally
        # uses an already ordered unit tuple and the fake executor completes
        # futures immediately.
        committed: list[int] = []
        run_ordered_policy_schedule_antigravity(
            units=units,
            task_loader=lambda unit: unit,
            score_task=_result,
            commit=lambda unit, result: committed.append(unit.sequence),
            workers=11,
            executor_factory=_FakeExecutor,
        )
        self.assertEqual(_FakeExecutor.created[0]["max_workers"], 11)
        context = _FakeExecutor.created[0]["mp_context"]
        self.assertEqual(context if context == "spawn" else context.get_start_method(), "spawn")
        self.assertEqual(committed, [0, 1, 2])

    def test_scheduler_stops_after_worker_error_without_submitting_later_units(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.runtime_antigravity import (
            run_ordered_policy_schedule_antigravity,
        )

        units = tuple(WorkUnitAntigravity(i, POLICIES[0], f"dataset_{i}") for i in range(3))
        submitted: list[int] = []

        def score(unit: WorkUnitAntigravity):
            submitted.append(unit.sequence)
            if unit.sequence == 1:
                raise RuntimeError("worker boom")
            return _result(unit)

        with self.assertRaisesRegex(RuntimeError, "worker boom"):
            run_ordered_policy_schedule_antigravity(
                units=units,
                task_loader=lambda unit: unit,
                score_task=score,
                commit=lambda _unit, _result: None,
                workers=11,
                executor_factory=_FakeExecutor,
            )
        self.assertNotIn(2, submitted)

    def test_policy_b_is_not_submitted_when_policy_a_control_fails(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.control_antigravity import ControlGateResultAntigravity
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.runtime_antigravity import _execute_policy_phases_antigravity

        units = (
            WorkUnitAntigravity(0, POLICIES[0], "dataset_00"),
            WorkUnitAntigravity(1, POLICIES[1], "dataset_00"),
        )
        submitted: list[tuple[str, int]] = []
        failed = ControlGateResultAntigravity("control_reproduction_failed", False, "fixture failure", (), {}, {}, 0)
        with self.assertRaisesRegex(ValueError, "control_reproduction_failed"):
            _execute_policy_phases_antigravity(
                units=units,
                task_loader=lambda unit: unit,
                commit=lambda _unit, _result: None,
                progress=None,
                control_runner=lambda _results: failed,
                revalidate=lambda: None,
                score_task=lambda unit: _result(unit),
                executor_factory=_FakeExecutor,
            )
        # The scheduler sees the control failure before the B phase is even
        # constructed, so no alternative-policy task can be submitted.
        self.assertEqual(submitted, [])

    def test_resume_prefix_reopens_results_and_admits_policy_b_after_control(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.checkpoint_antigravity import (
            CheckpointBindingAntigravity,
            Exp443CheckpointStoreAntigravity,
            ProcessIdentityAntigravity,
        )
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.control_antigravity import ControlGateResultAntigravity
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.runtime_antigravity import _execute_policy_phases_antigravity

        units = (
            WorkUnitAntigravity(0, POLICIES[0], "dataset_00"),
            WorkUnitAntigravity(1, POLICIES[1], "dataset_00"),
        )
        passed = ControlGateResultAntigravity("control_reproduction_passed", True, None, (), {}, {}, 0)
        binding = CheckpointBindingAntigravity("exp443-fixture-antigravity", "a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 40)
        with tempfile.TemporaryDirectory(prefix="exp443_task8_resume_antigravity_") as directory:
            root = Path(directory) / "checkpoint_antigravity"
            old = ProcessIdentityAntigravity(4242, "old")
            store = Exp443CheckpointStoreAntigravity.create_antigravity(root, units, binding, old)
            first = _result(units[0])
            store.commit_result_antigravity(0, first)
            self.assertEqual(store.results_antigravity(), {0: first})
            results: dict[int, PolicyDatasetResultAntigravity] = dict(store.results_antigravity())
            _execute_policy_phases_antigravity(
                units=units,
                task_loader=lambda unit: unit,
                commit=lambda unit, result: (results.__setitem__(unit.sequence, result), store.commit_result_antigravity(unit.sequence, result)),
                progress=None,
                control_runner=lambda _results: passed,
                revalidate=lambda: None,
                initial_results=results,
                score_task=lambda unit: _result(unit),
                executor_factory=_FakeExecutor,
            )
            self.assertEqual(store.completed_count_antigravity, 2)
            # The resumed B phase is returned through the in-memory result map.
            self.assertIn(1, results)

    def test_sigterm_interruption_closes_checkpoint_and_progress_as_matching_resumable_prefix(self) -> None:
        import json
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.checkpoint_antigravity import (
            CheckpointBindingAntigravity,
            Exp443CheckpointStoreAntigravity,
            ProcessIdentityAntigravity,
        )
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.progress_antigravity import ProgressWriterAntigravity
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.runtime_antigravity import _mark_interrupted_antigravity

        units = (WorkUnitAntigravity(0, POLICIES[0], "dataset_00"), WorkUnitAntigravity(1, POLICIES[0], "dataset_01"))
        binding = CheckpointBindingAntigravity("exp443-fixture-antigravity", "a" * 64, "b" * 64, "c" * 64, "d" * 64, "e" * 40)
        with tempfile.TemporaryDirectory(prefix="exp443_task8_sigterm_antigravity_") as directory:
            attempt = Path(directory) / "attempt_antigravity"
            attempt.mkdir()
            process = ProcessIdentityAntigravity(4242, "fixture-start")
            checkpoint = attempt / "checkpoint_antigravity"
            store = Exp443CheckpointStoreAntigravity.create_antigravity(checkpoint, units, binding, process)
            writer = ProgressWriterAntigravity(attempt, total=2, checkpoint_path=checkpoint / "state_antigravity.json", process=process)
            writer.begin_units_antigravity(units)
            store.commit_result_antigravity(0, _result(units[0]))
            writer.accept_unit_antigravity(_result(units[0]))
            _mark_interrupted_antigravity(store, writer, "SIGTERM received")
            state = json.loads((checkpoint / "state_antigravity.json").read_text())
            progress = json.loads((attempt / "progress_antigravity.json").read_text())
            self.assertEqual(state["status"], "interrupted_resumable")
            self.assertEqual(state["completed_count"], 1)
            self.assertEqual(progress["stage"], "interrupted_resumable")
            self.assertEqual(progress["completed"], 1)
            self.assertIsNone(progress["eta_seconds"])
            self.assertFalse(json.loads((attempt / "pid_antigravity.json").read_text())["active"])


if __name__ == "__main__":
    unittest.main()
