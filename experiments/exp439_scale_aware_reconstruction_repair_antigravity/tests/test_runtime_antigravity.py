from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
import os
import threading
import time
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.exp439_scale_aware_reconstruction_repair_antigravity.contracts_antigravity import (
    BackendDecision,
    Exp439DatasetResult,
    InstanceChannelEvidence,
    InstanceScoreEvidence,
    RawValueSeries,
)
from experiments.exp439_scale_aware_reconstruction_repair_antigravity.inputs_antigravity import (
    Exp439RawDataset,
)
from experiments.exp439_scale_aware_reconstruction_repair_antigravity.runtime_antigravity import (
    CpuDatasetTaskAntigravity,
    LaneMembershipSnapshotAntigravity,
    _build_context_antigravity,
    _score_accelerated_task_antigravity,
    _run_antigravity,
    inspect_exp439_antigravity,
    _score_cpu_task_antigravity,
    run_ordered_cpu_schedule_antigravity,
    run_serial_accelerator_schedule_antigravity,
    worker_count_antigravity,
)


def _row(dataset: str, split: str, index: int, phase: float) -> RawValueSeries:
    x = np.arange(16, dtype=np.float32)
    values = np.sin(x * np.float32(0.2) + np.float32(phase)).astype(np.float32)
    raw = values.astype("<f4").tobytes()
    return RawValueSeries(dataset, index, split, values, sha256(raw).hexdigest())


def _raw_pair(dataset: str) -> tuple[Exp439RawDataset, Exp439RawDataset]:
    train = Exp439RawDataset(
        dataset,
        "TRAIN",
        tuple(_row(dataset, "TRAIN", index, index * 0.1) for index in range(5)),
    )
    test = Exp439RawDataset(
        dataset,
        "TEST",
        (_row(dataset, "TEST", 0, 0.15), _row(dataset, "TEST", 1, 1.5)),
    )
    return train, test


def _result(dataset: str) -> Exp439DatasetResult:
    channels = tuple(
        InstanceChannelEvidence(name, 2.0, 1.0, True, True, None)
        for name in ("high_frequency", "low_frequency", "amplitude_complexity")
    )
    return Exp439DatasetResult(
        dataset,
        (InstanceScoreEvidence(dataset, 0, 16, channels, True),),
        (),
        (),
    )


class _Session:
    def __init__(self, datasets: tuple[str, ...]) -> None:
        self.datasets = datasets
        self.loaded_by = []

    def load_train(self, dataset: str) -> Exp439RawDataset:
        self.loaded_by.append(("train", dataset, threading.current_thread().name))
        return _raw_pair(dataset)[0]

    def load_test_values(self, dataset: str, train: Exp439RawDataset) -> Exp439RawDataset:
        self.loaded_by.append(("test", dataset, threading.current_thread().name))
        self.assert_bound = train.dataset == dataset
        return _raw_pair(dataset)[1]


class _SourcePaths:
    def to_payload(self):
        return {
            "database": "/tmp/database_antigravity.db",
            "exp291_prediction": "/tmp/exp291_prediction_antigravity.jsonl",
            "exp291_manifest": "/tmp/exp291_manifest_antigravity.json",
            "exp292_prediction": "/tmp/exp292_prediction_antigravity.jsonl",
            "exp292_manifest": "/tmp/exp292_manifest_antigravity.json",
        }


class _Context:
    def __init__(self, datasets):
        self.datasets = datasets
        self.code_commit = "a" * 40
        self.database_sha256 = "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
        self.test_identity_manifest = {
            dataset: _raw_pair(dataset)[1].identity_lengths for dataset in datasets
        }
        self.lanes = type(
            "Lanes",
            (),
            {name: {} for name in ("hard", "standard", "priority", "automatic_normal")},
        )()


class _BoundSession(_Session):
    def __init__(self, datasets):
        super().__init__(datasets)
        self.context = _Context(datasets)
        self.source_paths = _SourcePaths()
        self.label_access_count = 0


def _thread_executor(max_workers: int, initializer):
    initializer()
    return ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="antigravity-test")


class Exp439RuntimeAntigravityTests(unittest.TestCase):
    def test_runtime_accepts_json_roundtripped_readiness_backend_decision(self) -> None:
        datasets = ("A",)
        decision = BackendDecision("cpu", None, (), "synthetic CPU fixture")
        readiness = {
            "schema": "exp439.readiness.v1",
            "reviewed_commit": "a" * 40,
            "database_sha256": (
                "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
            ),
            "datasets": ["A"],
            "backend_decision": {
                "selected_backend": "cpu",
                "selected_batch_size": None,
                "evidence": [],
                "fallback_reason": "synthetic CPU fixture",
            },
            "backend_provenance": {},
            "source_sha256": {},
        }

        context, _ = _build_context_antigravity(
            session=_BoundSession(datasets),
            reviewed_commit="a" * 40,
            decision=decision,
            output_root=Path("/tmp/output_antigravity"),
            checkpoint_root=Path("/tmp/checkpoint_antigravity"),
            readiness_payload=readiness,
        )

        self.assertEqual(context["backend_decision"], readiness["backend_decision"])

    def test_official_worker_count_and_inner_threads_are_max_minus_one(self) -> None:
        self.assertEqual(worker_count_antigravity(12), 11)
        self.assertEqual(worker_count_antigravity(1), 1)
        for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
        ):
            self.assertEqual(os.environ[name], "1")

    def test_cpu_task_scores_every_test_row_without_label_access(self) -> None:
        train, test = _raw_pair("A")
        lanes = LaneMembershipSnapshotAntigravity(
            hard=frozenset({("A", 0)}),
            standard=frozenset(),
            priority=frozenset(),
            automatic_normal=frozenset(),
        )
        result = _score_cpu_task_antigravity(CpuDatasetTaskAntigravity("A", train, test, lanes))
        self.assertEqual(result.dataset, "A")
        self.assertEqual(tuple(item.instance_index for item in result.evidence), (0, 1))
        self.assertFalse(result.evidence[0].outside_all_frozen_lanes)
        self.assertTrue(result.evidence[1].outside_all_frozen_lanes)

    def test_parallel_scheduler_is_bounded_and_commits_in_frozen_order(self) -> None:
        datasets = tuple("ABCDE")
        delays = {dataset: (len(datasets) - index) * 0.002 for index, dataset in enumerate(datasets)}
        for workers in (1, 2, 11):
            with self.subTest(workers=workers):
                session = _Session(datasets)
                committed: list[str] = []
                observations: list[tuple[int, int]] = []

                def score(task):
                    time.sleep(delays[task.dataset])
                    return _result(task.dataset)

                run_ordered_cpu_schedule_antigravity(
                    datasets=datasets,
                    session=session,
                    lanes=LaneMembershipSnapshotAntigravity.empty_antigravity(),
                    workers=workers,
                    commit=lambda result, metadata: committed.append(result.dataset),
                    progress=lambda **payload: None,
                    score_task=score,
                    executor_factory=_thread_executor,
                    observer=lambda pending, buffered: observations.append((pending, buffered)),
                )
                self.assertEqual(tuple(committed), datasets)
                self.assertTrue(all(pending + buffered <= min(workers, len(datasets)) for pending, buffered in observations))
                self.assertTrue(all(item[2] == threading.current_thread().name for item in session.loaded_by))

    def test_real_process_pool_accepts_the_picklable_cpu_task(self) -> None:
        session = _Session(("A",))
        committed: list[str] = []
        run_ordered_cpu_schedule_antigravity(
            datasets=session.datasets,
            session=session,
            lanes=LaneMembershipSnapshotAntigravity.empty_antigravity(),
            workers=11,
            commit=lambda result, metadata: committed.append(result.dataset),
            progress=lambda **payload: None,
        )
        self.assertEqual(committed, ["A"])

    def test_worker_exception_is_attributed_and_stops_the_schedule(self) -> None:
        session = _Session(("A", "B", "C"))
        committed: list[str] = []

        def score(task):
            if task.dataset == "B":
                raise RuntimeError("synthetic worker failure")
            return _result(task.dataset)

        with self.assertRaisesRegex(RuntimeError, "synthetic worker failure"):
            run_ordered_cpu_schedule_antigravity(
                datasets=session.datasets,
                session=session,
                lanes=LaneMembershipSnapshotAntigravity.empty_antigravity(),
                workers=2,
                commit=lambda result, metadata: committed.append(result.dataset),
                progress=lambda **payload: None,
                score_task=score,
                executor_factory=_thread_executor,
            )
        self.assertNotIn("C", committed)

    def test_accelerator_schedule_is_strictly_serial(self) -> None:
        session = _Session(("A", "B", "C"))
        active = 0
        maximum_active = 0
        committed: list[str] = []

        def score(task):
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            try:
                return _result(task.dataset)
            finally:
                active -= 1

        run_serial_accelerator_schedule_antigravity(
            datasets=session.datasets,
            session=session,
            lanes=LaneMembershipSnapshotAntigravity.empty_antigravity(),
            commit=lambda result, metadata: committed.append(result.dataset),
            progress=lambda **payload: None,
            score_task=score,
        )
        self.assertEqual(maximum_active, 1)
        self.assertEqual(tuple(committed), session.datasets)

    def test_accelerated_task_uses_the_selected_exact_stage(self) -> None:
        train, test = _raw_pair("A")
        task = CpuDatasetTaskAntigravity(
            "A", train, test, LaneMembershipSnapshotAntigravity.empty_antigravity()
        )
        cpu = _score_cpu_task_antigravity(task)

        class NumbaEngine:
            backend = "numba"

            def __init__(self):
                self.calls = 0

            def compute(self, values, embedding_dimension, delay):
                from experiments.exp439_scale_aware_reconstruction_repair_antigravity.aspe_antigravity import amplitude_sensitive_permutation_entropy_antigravity
                self.calls += 1
                return amplitude_sensitive_permutation_entropy_antigravity(values, embedding_dimension, delay)

        numba = NumbaEngine()
        numba_result = _score_accelerated_task_antigravity(
            task, backend="numba", batch_size=16, engine=numba
        )
        self.assertGreater(numba.calls, 0)
        self.assertEqual(numba_result.candidate_coordinates, cpu.candidate_coordinates)

        class DenseEngine:
            backend = "mlx"

            def __init__(self):
                self.calls = 0

            def score_batch(self, queries, references, k, *, batch_size):
                self.calls += 1
                delta = references[None, :, :].astype(np.float64) - queries[:, None, :].astype(np.float64)
                distances = np.sqrt(np.mean(delta * delta, axis=2))
                nearest = np.partition(distances, k - 1, axis=1)[:, :k]
                return np.median(nearest, axis=1).astype(np.float32)

        dense = DenseEngine()
        dense_result = _score_accelerated_task_antigravity(
            task, backend="mlx", batch_size=16, engine=dense
        )
        self.assertGreater(dense.calls, 0)
        self.assertEqual(dense_result.candidate_coordinates, cpu.candidate_coordinates)

    def test_new_interrupt_resume_matches_uninterrupted_record_bytes(self) -> None:
        datasets = ("A", "B", "C")
        decision = BackendDecision("cpu", None, (), "synthetic CPU fixture")
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            output = base / "output_antigravity"
            checkpoint = base / "checkpoint_antigravity"
            calls = {"B": 0}

            def interrupt_once(task):
                if task.dataset == "B" and calls["B"] == 0:
                    calls["B"] += 1
                    raise KeyboardInterrupt("synthetic interrupt")
                return _result(task.dataset)

            with self.assertRaises(KeyboardInterrupt):
                _run_antigravity(
                    mode="new",
                    output_root=output,
                    checkpoint_root=checkpoint,
                    reviewed_commit="a" * 40,
                    decision=decision,
                    input_session_factory=lambda: _BoundSession(datasets),
                    workers=1,
                    executor_factory=_thread_executor,
                    score_task=interrupt_once,
                )
            resumed = _run_antigravity(
                mode="resume",
                output_root=output,
                checkpoint_root=checkpoint,
                reviewed_commit="a" * 40,
                decision=decision,
                input_session_factory=lambda: _BoundSession(datasets),
                workers=2,
                executor_factory=_thread_executor,
                score_task=lambda task: _result(task.dataset),
            )
            self.assertEqual(resumed.completed_datasets, 3)
            inspected = inspect_exp439_antigravity(output, checkpoint)
            self.assertEqual(inspected.completed_datasets, 3)
            self.assertEqual(inspected.prediction_label_access_count, 0)

            fresh_output = base / "fresh_output_antigravity"
            fresh_checkpoint = base / "fresh_checkpoint_antigravity"
            _run_antigravity(
                mode="new",
                output_root=fresh_output,
                checkpoint_root=fresh_checkpoint,
                reviewed_commit="a" * 40,
                decision=decision,
                input_session_factory=lambda: _BoundSession(datasets),
                workers=11,
                executor_factory=_thread_executor,
                score_task=lambda task: _result(task.dataset),
            )

            def record_bytes(root):
                return tuple(
                    path.read_bytes()
                    for path in sorted((root / "records_antigravity").iterdir())
                )

            self.assertEqual(record_bytes(checkpoint), record_bytes(fresh_checkpoint))

    def test_worker_failure_marks_both_roots_failed_and_new_never_clobbers(self) -> None:
        datasets = ("A", "B")
        decision = BackendDecision("cpu", None, (), "synthetic CPU fixture")
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            output = base / "output_antigravity"
            checkpoint = base / "checkpoint_antigravity"

            def fail(task):
                raise RuntimeError("synthetic failure")

            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                _run_antigravity(
                    mode="new",
                    output_root=output,
                    checkpoint_root=checkpoint,
                    reviewed_commit="a" * 40,
                    decision=decision,
                    input_session_factory=lambda: _BoundSession(datasets),
                    workers=2,
                    executor_factory=_thread_executor,
                    score_task=fail,
                )
            self.assertEqual(json.loads((output / "progress_antigravity.json").read_text())["state"], "failed")
            self.assertEqual(json.loads((checkpoint / "state_antigravity.json").read_text())["status"], "failed")
            before = (checkpoint / "state_antigravity.json").read_bytes()
            with self.assertRaises(ValueError):
                _run_antigravity(
                    mode="new",
                    output_root=output,
                    checkpoint_root=checkpoint,
                    reviewed_commit="a" * 40,
                    decision=decision,
                    input_session_factory=lambda: _BoundSession(datasets),
                    workers=1,
                    executor_factory=_thread_executor,
                    score_task=lambda task: _result(task.dataset),
                )
            self.assertEqual(before, (checkpoint / "state_antigravity.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
