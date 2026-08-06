"""Runtime topology tests for Exp436."""
from __future__ import annotations

import os
import inspect
import multiprocessing
import pickle
import subprocess
import ast
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.exp403_408.artifact_io import canonical_json_bytes
from experiments.exp436_missing_safe_strict_consensus_antigravity.contracts_antigravity import BackendDecision
from experiments.exp436_missing_safe_strict_consensus_antigravity.contracts_antigravity import (
    BackendBatchEvidence,
    BackendBenchmarkEvidence,
    Exp436Candidate,
    Exp436DatasetResult,
    InstanceScoreEvidence,
)
from experiments.exp436_missing_safe_strict_consensus_antigravity.backend_antigravity import select_backend
from experiments.exp436_missing_safe_strict_consensus_antigravity.config_antigravity import FROZEN_CONFIG


class RuntimeImportTests(unittest.TestCase):
    def test_import_sets_all_six_numerical_thread_variables(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        for name in runtime_antigravity._THREAD_ENV:
            self.assertEqual(os.environ[name], "1")

    def test_fresh_process_sets_all_thread_variables_before_runtime_import(self) -> None:
        code = "import os; names=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'); [os.environ.pop(n,None) for n in names]; import experiments.exp436_missing_safe_strict_consensus_antigravity.runtime_antigravity; print('|'.join(os.environ[n] for n in names))"
        completed = subprocess.run([os.environ.get("PYTHON", "python3"), "-c", code], cwd=Path(__file__).parents[3], capture_output=True, text=True, check=True)
        self.assertEqual(completed.stdout.strip(), "1|1|1|1|1|1")


class RuntimeContextTests(unittest.TestCase):
    def test_context_binds_every_readiness_and_session_authority(self) -> None:
        """Changing an external readiness binding must alter the persisted context."""

        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity

        datasets = tuple(f"D{index:04d}" for index in range(1117))
        lanes = SimpleNamespace(
            hard={}, standard={}, priority={}, automatic_normal={},
            exp291_prediction_sha256="b" * 64,
            exp291_manifest_sha256="c" * 64,
            exp292_prediction_sha256="d" * 64,
            exp292_manifest_sha256="e" * 64,
        )
        session = SimpleNamespace(
            context=SimpleNamespace(
                datasets=datasets,
                code_commit="a" * 40,
                database_sha256="f" * 64,
                lanes=lanes,
                test_identity_manifest={dataset: ((2, 16),) for dataset in datasets},
                runtime_versions={"python": "3", "numpy": "2"},
            ),
            source_paths=SimpleNamespace(to_payload=lambda: {
                "database": "/tmp/database_antigravity",
                "exp291_prediction": "/tmp/exp291_prediction_antigravity",
                "exp291_manifest": "/tmp/exp291_manifest_antigravity",
                "exp292_prediction": "/tmp/exp292_prediction_antigravity",
                "exp292_manifest": "/tmp/exp292_manifest_antigravity",
            }),
        )
        decision = BackendDecision.cpu_fallback()
        readiness = {
            "schema": "exp436.readiness-binding.v1", "run_identity": "exp436-missing-safe-strict-consensus-antigravity-20260806",
            "reviewed_commit": "a" * 40, "database_path": "/tmp/database_antigravity", "database_sha256": "f" * 64,
            "exp291_prediction_path": "/tmp/exp291_prediction_antigravity", "exp291_prediction_sha256": "b" * 64,
            "exp291_manifest_path": "/tmp/exp291_manifest_antigravity", "exp291_manifest_sha256": "c" * 64,
            "exp292_prediction_path": "/tmp/exp292_prediction_antigravity", "exp292_prediction_sha256": "d" * 64,
            "exp292_manifest_path": "/tmp/exp292_manifest_antigravity", "exp292_manifest_sha256": "e" * 64,
            "datasets": list(datasets), "dataset_digest": sha256(canonical_json_bytes(list(datasets))).hexdigest(),
            "frozen_config": json.loads(canonical_json_bytes(asdict(FROZEN_CONFIG))),
            "frozen_config_sha256": sha256(canonical_json_bytes(json.loads(canonical_json_bytes(asdict(FROZEN_CONFIG))))).hexdigest(),
            "backend_decision": decision.to_payload(), "backend_decision_sha256": sha256(canonical_json_bytes(decision.to_payload())).hexdigest(),
            "run_root": "/tmp/output_antigravity", "checkpoint_root": "/tmp/checkpoint_antigravity",
        }
        context = runtime_antigravity._build_run_context(
            session=session, readiness=readiness, readiness_path="/tmp/readiness_antigravity.json",
            run_root="/tmp/output_antigravity", checkpoint_root="/tmp/checkpoint_antigravity", decision=decision,
        )
        self.assertEqual(context.to_payload()["readiness_artifact_sha256"], sha256(canonical_json_bytes(readiness)).hexdigest())
        self.assertEqual(context.to_payload()["test_identity_manifest"]["D0000"], [[2, 16]])


def _datasets_antigravity() -> tuple[str, ...]:
    return tuple(f"D{index:04d}" for index in range(1117))


def _result_antigravity(dataset: str, *, candidate: bool = False) -> Exp436DatasetResult:
    evidence = InstanceScoreEvidence.fixture(
        dataset=dataset, instance_index=2, length=16,
        prototype_abnormal=candidate, gradient_abnormal=candidate, acf_abnormal=candidate,
    )
    candidates = () if not candidate else (Exp436Candidate(dataset, 2, 16, evidence.evidence_digest),)
    return Exp436DatasetResult(dataset, "scored", (2,), candidates, (evidence,))


def _session_antigravity() -> object:
    datasets = _datasets_antigravity()
    lanes = SimpleNamespace(
        hard={}, standard={}, priority={}, automatic_normal={},
        exp291_prediction_sha256="b" * 64, exp291_manifest_sha256="c" * 64,
        exp292_prediction_sha256="d" * 64, exp292_manifest_sha256="e" * 64,
    )
    rows = (SimpleNamespace(values=(1.0,) * 16),) * 4
    context = SimpleNamespace(
        datasets=datasets, code_commit="a" * 40, database_sha256="f" * 64, lanes=lanes,
        test_identity_manifest={dataset: ((2, 16),) for dataset in datasets}, runtime_versions={"python": "3", "numpy": "2"},
    )
    return SimpleNamespace(
        context=context,
        source_paths=SimpleNamespace(to_payload=lambda: {
            "database": "/tmp/database_antigravity", "exp291_prediction": "/tmp/exp291_prediction_antigravity",
            "exp291_manifest": "/tmp/exp291_manifest_antigravity", "exp292_prediction": "/tmp/exp292_prediction_antigravity",
            "exp292_manifest": "/tmp/exp292_manifest_antigravity",
        }),
        load_train=lambda _dataset: SimpleNamespace(series=rows),
        load_test_values=lambda _dataset, _train: SimpleNamespace(series=(SimpleNamespace(values=(1.0,) * 16),)),
    )


def _readiness_antigravity(runtime: object, root: Path) -> Path:
    datasets = _datasets_antigravity(); decision = BackendDecision.cpu_fallback()
    config = json.loads(canonical_json_bytes(asdict(FROZEN_CONFIG)))
    payload = {
        "schema": "exp436.readiness-binding.v1", "run_identity": FROZEN_CONFIG.run_identity, "reviewed_commit": "a" * 40,
        "database_path": "/tmp/database_antigravity", "database_sha256": "f" * 64,
        "exp291_prediction_path": "/tmp/exp291_prediction_antigravity", "exp291_prediction_sha256": "b" * 64,
        "exp291_manifest_path": "/tmp/exp291_manifest_antigravity", "exp291_manifest_sha256": "c" * 64,
        "exp292_prediction_path": "/tmp/exp292_prediction_antigravity", "exp292_prediction_sha256": "d" * 64,
        "exp292_manifest_path": "/tmp/exp292_manifest_antigravity", "exp292_manifest_sha256": "e" * 64,
        "datasets": list(datasets), "dataset_digest": sha256(canonical_json_bytes(list(datasets))).hexdigest(),
        "frozen_config": config, "frozen_config_sha256": sha256(canonical_json_bytes(config)).hexdigest(),
        "backend_decision": decision.to_payload(), "backend_decision_sha256": sha256(canonical_json_bytes(decision.to_payload())).hexdigest(),
        "run_root": str((root / "output_antigravity").resolve()), "checkpoint_root": str((root / "checkpoint_antigravity").resolve()),
    }
    path = root / "readiness_antigravity.json"; path.write_bytes(canonical_json_bytes(payload))
    return path


def _spawn_score_antigravity(task: object) -> tuple[str, tuple[str, ...]]:
    from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
    return runtime_antigravity._score_cpu_task(task).status, tuple(os.environ[name] for name in runtime_antigravity._THREAD_ENV)


def _real_task_antigravity() -> object:
    from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
    from experiments.exp436_missing_safe_strict_consensus_antigravity.contracts_antigravity import RawValueSeries, MissingSafeRawDataset
    rows = tuple(RawValueSeries(index, tuple(float(index + offset) for offset in range(16)), f"{index:064x}") for index in range(4))
    test = MissingSafeRawDataset("Spawn", "TEST", (RawValueSeries(2, tuple(float(10 + offset) for offset in range(16)), "f" * 64),))
    return runtime_antigravity.CpuDatasetTask("Spawn", MissingSafeRawDataset("Spawn", "TRAIN", rows), test,
                                               runtime_antigravity.LaneMembershipSnapshot(frozenset(), frozenset(), frozenset(), frozenset()), BackendDecision.cpu_fallback())


def _mlx_decision_antigravity() -> BackendDecision:
    batches = tuple(BackendBatchEvidence("mlx", size, "completed", 2, ((2.0, 1.0),) * 5, "a" * 64, "b" * 64, True, True, 0.0, 0.0, 2.0, 1.0, 2.0, True, True, True, "passed") for size in (16, 32, 64, 128, 256))
    mlx = BackendBenchmarkEvidence("mlx", True, batches, "python", "numpy", "gpu:0", "info", True, "paired-synchronized-v1", 16, 1.0, None, mlx_version="mlx", mlx_metal_version="metal")
    return select_backend(mlx, BackendBenchmarkEvidence.unavailable("mps", "missing"))


def _mps_decision_antigravity() -> BackendDecision:
    batches = tuple(BackendBatchEvidence("mps", size, "completed", 2, ((2.0, 1.0),) * 5, "a" * 64, "b" * 64, True, True, 0.0, 0.0, 2.0, 1.0, 2.0, True, True, True, "passed") for size in (16, 32, 64, 128, 256))
    mps = BackendBenchmarkEvidence("mps", True, batches, "python", "numpy", "mps", "info", True, "paired-synchronized-v1", 16, 1.0, None, pytorch_version="torch")
    return select_backend(BackendBenchmarkEvidence.unavailable("mlx", "missing"), mps)


class _InlineExecutorAntigravity:
    instances: list["_InlineExecutorAntigravity"] = []
    candidate = False

    def __init__(self, *, max_workers: int, initializer: object) -> None:
        self.max_workers, self.initializer, self.pending_peak = max_workers, initializer, 0
        self.submissions: list[str] = []; self.futures: list[object] = []; self.shutdown_calls = 0; type(self).instances.append(self)

    def submit(self, _function: object, task: object) -> object:
        from concurrent.futures import Future
        future = Future(); self.submissions.append(task.dataset); self.pending_peak = max(self.pending_peak, len(self.submissions))
        future.cancel_called_antigravity = False
        def cancel() -> bool:
            future.cancel_called_antigravity = True
            return True
        future.cancel = cancel
        future.set_result(_result_antigravity(task.dataset, candidate=self.candidate)); self.futures.append(future); return future

    def shutdown(self, **_kwargs: object) -> None:
        self.shutdown_calls += 1


class RuntimeCpuTopologyTests(unittest.TestCase):
    def setUp(self) -> None:
        _InlineExecutorAntigravity.instances.clear(); _InlineExecutorAntigravity.candidate = False

    def _schedule(self, *, count: int, candidate: bool = False) -> tuple[object, object, list[int], list[tuple[int, int]]]:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        decision = BackendDecision.cpu_fallback(); session = _session_antigravity()
        class Store:
            def __init__(self) -> None: self.committed: list[str] = []
            def commit(self, result: object, **_kwargs: object) -> str: self.committed.append(result.dataset); return "a" * 64
        class Writer:
            def __init__(self) -> None: self.events: list[str] = []; self.updates: list[dict[str, object]] = []
            def update(self, **kwargs: object) -> None: self.updates.append(kwargs)
            def record_dataset_complete(self, **kwargs: object) -> None: self.events.append(str(kwargs["dataset"]))
        store, writer, widths, occupancy = Store(), Writer(), [], []
        _InlineExecutorAntigravity.candidate = candidate
        def all_complete(futures: object, **_kwargs: object) -> tuple[set[object], set[object]]:
            widths.append(len(futures))
            return set(futures), set()
        executor = _InlineExecutorAntigravity(max_workers=FROZEN_CONFIG.cpu_workers, initializer=runtime_antigravity._worker_initializer)
        counters = {"completed": 0, "success": 0, "abstention": 0, "error": 0, "candidates": 0, "excluded": 0, "nonfinite": 0}
        with patch.object(runtime_antigravity, "wait", all_complete):
            runtime_antigravity._run_cpu_schedule(executor=executor, missing=tuple(f"D{index:04d}" for index in range(count)), session=session,
                                                   lanes=runtime_antigravity.LaneMembershipSnapshot(frozenset(), frozenset(), frozenset(), frozenset()),
                                                   decision=decision, store=store, writer=writer, counters=counters, _observer=lambda pending, buffered: occupancy.append((pending, buffered)))
        return store, writer, widths, occupancy

    def test_cpu_uses_frozen_eleven_workers_and_commits_sorted_prefix(self) -> None:
        store, writer, widths, occupancy = self._schedule(count=24)
        self.assertEqual(_InlineExecutorAntigravity.instances[0].max_workers, FROZEN_CONFIG.cpu_workers)
        self.assertLessEqual(max(widths), FROZEN_CONFIG.cpu_workers)
        self.assertTrue(all(pending + buffered <= FROZEN_CONFIG.cpu_workers for pending, buffered in occupancy))
        self.assertEqual(store.committed, [f"D{index:04d}" for index in range(24)])
        self.assertEqual(writer.events, store.committed)

    def test_forced_reverse_completion_buffers_then_commits_only_frozen_prefix(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        decision = BackendDecision.cpu_fallback(); session = _session_antigravity(); commits: list[str] = []; occupancy: list[tuple[int, int]] = []
        class Store:
            def commit(self, result: object, **_kwargs: object) -> str: commits.append(result.dataset); return "a" * 64
        class Writer:
            def update(self, **_kwargs: object) -> None: return None
            def record_dataset_complete(self, **_kwargs: object) -> None: return None
        executor = _InlineExecutorAntigravity(max_workers=11, initializer=runtime_antigravity._worker_initializer)
        def reverse_one(futures: object, **_kwargs: object) -> tuple[set[object], set[object]]:
            values = list(futures); return {values[-1]}, set(values[:-1])
        counters = {"completed": 0, "success": 0, "abstention": 0, "error": 0, "candidates": 0, "excluded": 0, "nonfinite": 0}
        with patch.object(runtime_antigravity, "wait", reverse_one):
            runtime_antigravity._run_cpu_schedule(executor=executor, missing=tuple(f"D{index:04d}" for index in range(13)), session=session, lanes=runtime_antigravity.LaneMembershipSnapshot(frozenset(), frozenset(), frozenset(), frozenset()), decision=decision, store=Store(), writer=Writer(), counters=counters, _observer=lambda pending, buffered: occupancy.append((pending, buffered)))
        self.assertEqual(commits, [f"D{index:04d}" for index in range(13)])
        self.assertTrue(any(buffered > 0 for _, buffered in occupancy))
        self.assertTrue(all(pending + buffered <= 11 for pending, buffered in occupancy))

    def test_candidate_101_fails_before_creating_the_101st_checkpoint_intent(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        decision = BackendDecision.cpu_fallback(); session = _session_antigravity(); _InlineExecutorAntigravity.candidate = True
        class Store:
            def __init__(self) -> None: self.committed: list[str] = []
            def commit(self, result: object, **_kwargs: object) -> str: self.committed.append(result.dataset); return "a" * 64
        class Writer:
            def update(self, **_kwargs: object) -> None: return None
            def record_dataset_complete(self, **_kwargs: object) -> None: return None
        store, writer = Store(), Writer()
        def first_complete(futures: object, **_kwargs: object) -> tuple[set[object], set[object]]:
            values = list(futures); return {values[0]}, set(values[1:])
        executor = _InlineExecutorAntigravity(max_workers=FROZEN_CONFIG.cpu_workers, initializer=runtime_antigravity._worker_initializer)
        counters = {"completed": 0, "success": 0, "abstention": 0, "error": 0, "candidates": 0, "excluded": 0, "nonfinite": 0}
        with patch.object(runtime_antigravity, "wait", first_complete):
            with self.assertRaisesRegex(ValueError, "candidate budget"):
                runtime_antigravity._run_cpu_schedule(executor=executor, missing=tuple(f"D{index:04d}" for index in range(112)), session=session,
                                                       lanes=runtime_antigravity.LaneMembershipSnapshot(frozenset(), frozenset(), frozenset(), frozenset()),
                                                       decision=decision, store=store, writer=writer, counters=counters)
        self.assertEqual(len(store.committed), 100)
        self.assertEqual(counters["candidates"], 100)
        self.assertTrue(any(future.cancel_called_antigravity for future in executor.futures))
        self.assertGreaterEqual(executor.shutdown_calls, 1)

    def test_cpu_worker_task_is_picklable_and_contains_no_parent_authority(self) -> None:
        task = _real_task_antigravity()
        self.assertEqual(task, pickle.loads(pickle.dumps(task)))
        self.assertEqual(tuple(task.__dataclass_fields__), ("dataset", "train", "test", "lanes", "decision"))

    def test_worker_rejects_arbitrary_path_or_label_carrying_objects_before_scoring(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        task = _real_task_antigravity()
        forged = runtime_antigravity.CpuDatasetTask(task.dataset, SimpleNamespace(split="TRAIN", path="/tmp/forged", labels=(1,)), task.test, task.lanes, task.decision)
        with self.assertRaisesRegex(ValueError, "CPU task"):
            runtime_antigravity._score_cpu_task(forged)

    def test_bounded_real_spawn_scores_values_without_parent_resources(self) -> None:
        task = _real_task_antigravity()
        context = multiprocessing.get_context("spawn")
        with context.Pool(1) as pool:
            status, threads = pool.apply(_spawn_score_antigravity, (task,))
        self.assertIn(status, {"scored", "abstained"})
        self.assertEqual(threads, ("1",) * 6)

    def test_serial_and_spawn_worker_results_are_identical(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        task = _real_task_antigravity(); serial = runtime_antigravity._score_cpu_task(task)
        with multiprocessing.get_context("spawn").Pool(1) as pool:
            parallel_status, _ = pool.apply(_spawn_score_antigravity, (task,))
        self.assertEqual(parallel_status, serial.status)

    def test_runtime_import_firewall_has_no_exp435_labels_evaluation_or_publication_route(self) -> None:
        source = Path(__file__).parents[1].joinpath("runtime_antigravity.py").read_text()
        tree = ast.parse(source)
        imports = "\n".join(ast.unparse(node) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)))
        self.assertNotIn("exp435", imports.lower())
        self.assertNotIn("evaluation", imports.lower())
        self.assertNotIn("publication", imports.lower())
        self.assertNotIn("label", imports.lower())

    def test_public_runtime_signatures_expose_no_backend_or_worker_override(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        for function in (runtime_antigravity.run_exp436_new, runtime_antigravity.resume_exp436, runtime_antigravity.inspect_exp436):
            names = set(inspect.signature(function).parameters)
            self.assertFalse({"backend", "backend_decision", "engine", "executor", "worker_count", "dataset_scorer"} & names)

    def test_cpu_interrupt_cancels_and_reaps_the_executor(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        class Store:
            def commit(self, _result: object, **_kwargs: object) -> str: return "a" * 64
        class Writer:
            def update(self, **_kwargs: object) -> None: return None
            def record_dataset_complete(self, **_kwargs: object) -> None: return None
        executor = _InlineExecutorAntigravity(max_workers=FROZEN_CONFIG.cpu_workers, initializer=runtime_antigravity._worker_initializer)
        executor._processes = {}
        counters = {"completed": 0, "success": 0, "abstention": 0, "error": 0, "candidates": 0, "excluded": 0, "nonfinite": 0}
        with patch.object(runtime_antigravity, "wait", side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                runtime_antigravity._run_cpu_schedule(executor=executor, missing=("D0000",), session=_session_antigravity(),
                                                       lanes=runtime_antigravity.LaneMembershipSnapshot(frozenset(), frozenset(), frozenset(), frozenset()),
                                                       decision=BackendDecision.cpu_fallback(), store=Store(), writer=Writer(), counters=counters)
        self.assertGreaterEqual(executor.shutdown_calls, 1)

    def test_production_cpu_path_constructs_exact_frozen_pool_and_never_engine(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        with TemporaryDirectory() as directory:
            root = Path(directory); readiness_path = _readiness_antigravity(runtime_antigravity, root); readiness = json.loads(readiness_path.read_text())
            with patch.object(runtime_antigravity, "ProcessPoolExecutor", _InlineExecutorAntigravity), patch.object(runtime_antigravity, "_run_cpu_schedule", side_effect=KeyboardInterrupt()), patch.object(runtime_antigravity, "_engine", side_effect=AssertionError("CPU must not create accelerator engine")):
                with self.assertRaises(KeyboardInterrupt):
                    runtime_antigravity._run(run_root=root / "output_antigravity", checkpoint_root=root / "checkpoint_antigravity", readiness_path=readiness_path, readiness=readiness, input_session=_session_antigravity(), decision=BackendDecision.cpu_fallback(), resume=False)
            self.assertEqual(_InlineExecutorAntigravity.instances[-1].max_workers, 11)

    def test_cpu_worker_error_records_one_failed_terminal_without_checkpoint_advance(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        with TemporaryDirectory() as directory:
            root = Path(directory); readiness_path = _readiness_antigravity(runtime_antigravity, root); readiness = json.loads(readiness_path.read_text())
            def fail_with_active_dataset(*, writer: object, **_kwargs: object) -> None:
                writer.update(stage="awaiting_worker_result", current_dataset="D0000", dataset_metadata={"train_count": 4, "test_count": 1, "train_length_histogram": {"16": 4}, "test_length_histogram": {"16": 1}}, in_flight_datasets=("D0000",))
                raise RuntimeError("worker boom")
            with patch.object(runtime_antigravity, "ProcessPoolExecutor", _InlineExecutorAntigravity), patch.object(runtime_antigravity, "_run_cpu_schedule", side_effect=fail_with_active_dataset):
                with self.assertRaisesRegex(RuntimeError, "worker boom"):
                    runtime_antigravity._run(run_root=root / "output_antigravity", checkpoint_root=root / "checkpoint_antigravity", readiness_path=readiness_path, readiness=readiness, input_session=_session_antigravity(), decision=BackendDecision.cpu_fallback(), resume=False)
            events = [json.loads(line) for line in (root / "output_antigravity" / "events_antigravity.jsonl").read_text().splitlines()]
            self.assertEqual([event["kind"] for event in events].count("failed"), 1)
            self.assertEqual(events[-1]["payload"]["offending_dataset"], "D0000")
            self.assertEqual(events[-1]["payload"]["prior_stage"], "awaiting_worker_result")
            self.assertEqual(json.loads((root / "checkpoint_antigravity" / "record_set_antigravity.json").read_text())["completed"], 0)

    def test_cpu_future_failure_names_the_failing_future_not_the_last_submitted_dataset(self) -> None:
        from concurrent.futures import Future
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        class FailingExecutor:
            instance: object | None = None
            def __init__(self, **_kwargs: object) -> None:
                self.submissions: list[str] = []; self.shutdown_calls = 0; self._processes = {}; type(self).instance = self
            def submit(self, _function: object, task: object) -> Future:
                future = Future(); self.submissions.append(task.dataset)
                if task.dataset == "D0005": future.set_exception(RuntimeError("D0005 worker failure"))
                else: future.set_result(_result_antigravity(task.dataset))
                return future
            def shutdown(self, **_kwargs: object) -> None: self.shutdown_calls += 1
        with TemporaryDirectory() as directory:
            root = Path(directory); readiness_path = _readiness_antigravity(runtime_antigravity, root); readiness = json.loads(readiness_path.read_text())
            with patch.object(runtime_antigravity, "ProcessPoolExecutor", FailingExecutor):
                with self.assertRaisesRegex(RuntimeError, "D0005 worker failure"):
                    runtime_antigravity._run(run_root=root / "output_antigravity", checkpoint_root=root / "checkpoint_antigravity", readiness_path=readiness_path, readiness=readiness, input_session=_session_antigravity(), decision=BackendDecision.cpu_fallback(), resume=False)
            events = [json.loads(line) for line in (root / "output_antigravity" / "events_antigravity.jsonl").read_text().splitlines()]
            progress = json.loads((root / "output_antigravity" / "progress_antigravity.json").read_text())
            self.assertEqual(FailingExecutor.instance.submissions, [f"D{index:04d}" for index in range(11)])
            self.assertEqual(events[-1]["payload"]["offending_dataset"], "D0005")
            self.assertEqual(events[-1]["payload"]["prior_stage"], "awaiting_worker_result")
            self.assertEqual(progress["latest_error"]["dataset"], "D0005")
            self.assertEqual(json.loads((root / "checkpoint_antigravity" / "record_set_antigravity.json").read_text())["completed"], 0)
            self.assertGreaterEqual(FailingExecutor.instance.shutdown_calls, 1)


class RuntimeReadOnlyTests(unittest.TestCase):
    def _root(self, terminal: str) -> tuple[Path, Path, Path, Path]:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        from experiments.exp436_missing_safe_strict_consensus_antigravity.checkpoint_antigravity import Exp436CheckpointStore
        from experiments.exp436_missing_safe_strict_consensus_antigravity.progress_antigravity import ProgressWriter
        directory = TemporaryDirectory(); self.addCleanup(directory.cleanup); root = Path(directory.name)
        checkpoint, output = root / "checkpoint_antigravity", root / "output_antigravity"
        readiness_path = _readiness_antigravity(runtime_antigravity, root); readiness = json.loads(readiness_path.read_text())
        context = runtime_antigravity._build_run_context(session=_session_antigravity(), readiness=readiness, readiness_path=readiness_path, run_root=output, checkpoint_root=checkpoint, decision=BackendDecision.cpu_fallback())
        store = Exp436CheckpointStore.create(checkpoint, context)
        writer = ProgressWriter.create(output, run_identity=store.run_identity, checkpoint=checkpoint, total_datasets=1117, backend="cpu", context_fingerprint=store.context_fingerprint, heartbeat_seconds=None)
        if terminal == "failed": writer.mark_failed("D0000", "loading", RuntimeError("boom"))
        # Construct a syntactically valid but scientifically impossible
        # terminal artifact to exercise public fail-closed resume behavior.
        elif terminal == "checkpoint_complete": writer._mark_terminal("checkpoint_complete", None, "idle", None)
        else: writer.mark_interrupted("D0000", "loading", KeyboardInterrupt())
        writer.close()
        return root, output, checkpoint, readiness_path

    def test_failed_resume_rejects_before_input_factory_and_without_mutation(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        root, output, checkpoint, readiness = self._root("failed")
        before = {path: path.read_bytes() for path in (*output.iterdir(), *checkpoint.iterdir()) if path.is_file()}
        factory_calls = []
        with self.assertRaisesRegex(ValueError, "failed runtime"):
            runtime_antigravity.resume_exp436(readiness_path=readiness, run_root=output, checkpoint_root=checkpoint, reviewed_commit="a" * 40, input_session_factory=lambda: factory_calls.append(True))
        self.assertEqual(factory_calls, [])
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_incomplete_checkpoint_complete_rejects_before_factory_and_without_mutation(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        root, output, checkpoint, readiness = self._root("checkpoint_complete")
        before = {path: path.read_bytes() for path in tuple(output.iterdir()) + tuple(checkpoint.rglob("*")) if path.is_file()}
        with self.assertRaisesRegex(ValueError, "checkpoint complete"):
            runtime_antigravity.resume_exp436(readiness_path=readiness, run_root=output, checkpoint_root=checkpoint, reviewed_commit="a" * 40, input_session_factory=lambda: self.fail("incomplete terminal must reject before factory"))
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_inspect_is_byte_identical_and_never_requests_an_input_session(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        root, output, checkpoint, readiness = self._root("interrupted")
        tracked = tuple(output.iterdir()) + tuple(checkpoint.rglob("*"))
        before = {path: path.read_bytes() for path in tracked if path.is_file()}
        summary = runtime_antigravity.inspect_exp436(readiness_path=readiness, run_root=output, checkpoint_root=checkpoint, reviewed_commit="a" * 40)
        self.assertEqual(summary.completed_datasets, 0)
        self.assertEqual(summary.status, "interrupted")
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_invalid_readiness_rejects_new_before_factory(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        with TemporaryDirectory() as directory:
            root = Path(directory); readiness = _readiness_antigravity(runtime_antigravity, root)
            payload = json.loads(readiness.read_text()); payload["dataset_digest"] = "0" * 64; readiness.write_bytes(canonical_json_bytes(payload))
            with self.assertRaisesRegex(ValueError, "dataset digest"):
                runtime_antigravity.run_exp436_new(readiness_path=readiness, run_root=root / "output_antigravity", checkpoint_root=root / "checkpoint_antigravity", reviewed_commit="a" * 40, input_session_factory=lambda: self.fail("factory must not run"))

    def test_existing_new_root_rejects_before_factory(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        with TemporaryDirectory() as directory:
            root = Path(directory); readiness = _readiness_antigravity(runtime_antigravity, root); (root / "output_antigravity").mkdir()
            with self.assertRaisesRegex(ValueError, "roots must be absent"):
                runtime_antigravity.run_exp436_new(readiness_path=readiness, run_root=root / "output_antigravity", checkpoint_root=root / "checkpoint_antigravity", reviewed_commit="a" * 40, input_session_factory=lambda: self.fail("factory must not run"))

    def test_inspect_rejects_stored_readiness_mismatch_without_mutation(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        root, output, checkpoint, readiness = self._root("interrupted")
        payload = json.loads(readiness.read_text()); payload["database_path"] = "/tmp/other_database_antigravity"; readiness.write_bytes(canonical_json_bytes(payload))
        before = {path: path.read_bytes() for path in tuple(output.iterdir()) + tuple(checkpoint.rglob("*")) if path.is_file()}
        with self.assertRaisesRegex(ValueError, "stored context"):
            runtime_antigravity.inspect_exp436(readiness_path=readiness, run_root=output, checkpoint_root=checkpoint, reviewed_commit="a" * 40)
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_resume_stored_readiness_mismatch_never_calls_factory(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        root, output, checkpoint, readiness = self._root("interrupted")
        payload = json.loads(readiness.read_text()); payload["database_path"] = "/tmp/other_database_antigravity"; readiness.write_bytes(canonical_json_bytes(payload))
        with self.assertRaisesRegex(ValueError, "stored context"):
            runtime_antigravity.resume_exp436(readiness_path=readiness, run_root=output, checkpoint_root=checkpoint, reviewed_commit="a" * 40, input_session_factory=lambda: self.fail("factory must not run"))

    def test_complete_resume_never_opens_factory_engine_or_pool_and_preserves_terminal_bytes(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        root, output, checkpoint, readiness = self._root("interrupted")
        # A strict complete-prefix seam avoids materializing 1,117 fsynced fixture records.
        datasets = _datasets_antigravity(); completed = {dataset: "a" * 64 for dataset in datasets}
        store = SimpleNamespace(run_identity=FROZEN_CONFIG.run_identity, context_fingerprint="c" * 64, datasets=datasets, completed_digests=lambda: completed, inspect=lambda: SimpleNamespace(completed=1117))
        before = {path: path.read_bytes() for path in tuple(output.iterdir()) + tuple(checkpoint.rglob("*")) if path.is_file()}
        with patch.object(runtime_antigravity.Exp436CheckpointStore, "open_classified_read_only", return_value=store), patch.object(runtime_antigravity, "_validate_stored_context"), patch.object(runtime_antigravity.ProgressWriter, "preflight_existing", return_value={"progress": {"state": "checkpoint_complete"}}), patch.object(runtime_antigravity, "_summary", return_value="complete"), patch.object(runtime_antigravity, "_engine", side_effect=AssertionError("complete resume must not create engine")), patch.object(runtime_antigravity, "ProcessPoolExecutor", side_effect=AssertionError("complete resume must not create pool")):
            self.assertEqual(runtime_antigravity.resume_exp436(readiness_path=readiness, run_root=output, checkpoint_root=checkpoint, reviewed_commit="a" * 40, input_session_factory=lambda: self.fail("complete resume must not create session")), "complete")
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_real_final_intent_lag_replays_once_without_factory_engine_or_pool(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        from experiments.exp436_missing_safe_strict_consensus_antigravity.checkpoint_antigravity import Exp436CheckpointStore, _name, _result_payload
        root, output, checkpoint, readiness = self._root("interrupted")
        store = Exp436CheckpointStore.open_read_only(checkpoint)
        metadata = {"train_count": 4, "test_count": 1, "train_length_histogram": {"16": 4}, "test_length_histogram": {"16": 1}}
        records: dict[str, str] = {}; results = []
        for index, dataset in enumerate(store.datasets):
            result = _result_antigravity(dataset); raw = canonical_json_bytes(_result_payload(result, index, metadata))
            (checkpoint / "records_antigravity" / _name(dataset)).write_bytes(raw)
            records[dataset] = sha256(raw).hexdigest(); results.append(result)
        stable_records = dict(tuple(records.items())[:-1])
        record_set = Exp436CheckpointStore._set_payload_static(store.context_fingerprint, tuple(stable_records), stable_records, tuple(results[:-1]))
        (checkpoint / "record_set_antigravity.json").write_bytes(canonical_json_bytes(record_set))
        final_dataset = store.datasets[-1]
        (checkpoint / "intents_antigravity" / _name(final_dataset)).write_bytes(canonical_json_bytes({"schema": "exp436.commit-intent.v1", "dataset": final_dataset, "record_sha256": records[final_dataset]}))
        with patch.object(runtime_antigravity, "_engine", side_effect=AssertionError("complete replay must not create an engine")), patch.object(runtime_antigravity, "ProcessPoolExecutor", side_effect=AssertionError("complete replay must not create a pool")):
            summary = runtime_antigravity.resume_exp436(readiness_path=readiness, run_root=output, checkpoint_root=checkpoint, reviewed_commit="a" * 40, input_session_factory=lambda: self.fail("complete replay must not create an input session"))
        self.assertEqual(summary.status, "checkpoint_complete")
        events = [json.loads(line) for line in (output / "events_antigravity.jsonl").read_text().splitlines()]
        self.assertEqual(sum(event["kind"] == "dataset_complete" for event in events), 1117)
        self.assertEqual(sum(event["kind"] == "checkpoint_complete" for event in events), 1)
        before = {path: path.read_bytes() for path in tuple(output.iterdir()) + tuple(checkpoint.rglob("*")) if path.is_file()}
        repeated = runtime_antigravity.resume_exp436(readiness_path=readiness, run_root=output, checkpoint_root=checkpoint, reviewed_commit="a" * 40, input_session_factory=lambda: self.fail("repeated complete replay must not create an input session"))
        self.assertEqual(repeated.status, "checkpoint_complete")
        self.assertEqual(before, {path: path.read_bytes() for path in before})


class RuntimeAcceleratorTests(unittest.TestCase):
    def _normal_serial_run(self, decision: BackendDecision) -> tuple[list[object], int]:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        engine = object(); seen: list[object] = []
        class Store:
            datasets = ("D0000", "D0001"); run_identity = FROZEN_CONFIG.run_identity; context_fingerprint = "c" * 64
            def completed_digests(self): return {}
        class Writer:
            progress = {"state": "running", "stage": "idle"}
            def update(self, **_kwargs): pass
            def start_heartbeat(self): pass
            def stop_heartbeat(self): pass
            def close(self): pass
            def mark_checkpoint_complete(self): self.progress["state"] = "checkpoint_complete"
        store, writer = Store(), Writer()
        with TemporaryDirectory() as directory:
            root=Path(directory); readiness_path=_readiness_antigravity(runtime_antigravity, root); readiness=json.loads(readiness_path.read_text())
            with patch.object(runtime_antigravity.Exp436CheckpointStore, "create", return_value=store), patch.object(runtime_antigravity.ProgressWriter, "create", return_value=writer), patch.object(runtime_antigravity, "_reconcile_progress"), patch.object(runtime_antigravity, "_counts", return_value={"completed":0,"success":0,"abstention":0,"error":0,"candidates":0,"excluded":0,"nonfinite":0}), patch.object(runtime_antigravity, "_engine", return_value=engine) as make_engine, patch.object(runtime_antigravity, "_load", return_value=(object(), object(), {"train_count":1,"test_count":1,"train_length_histogram":{"1":1},"test_length_histogram":{"1":1}})), patch.object(runtime_antigravity, "_score_accelerated", side_effect=lambda _a,_b,_c,_d,current: seen.append(current) or _result_antigravity("D0000" if len(seen)==1 else "D0001")), patch.object(runtime_antigravity, "_commit"), patch.object(runtime_antigravity, "_mark_checkpoint_complete"), patch.object(runtime_antigravity, "_summary", return_value="complete"), patch.object(runtime_antigravity, "ProcessPoolExecutor", side_effect=AssertionError("accelerator must remain serial")):
                self.assertEqual(runtime_antigravity._run(run_root=root/"output_antigravity", checkpoint_root=root/"checkpoint_antigravity", readiness_path=readiness_path, readiness=readiness, input_session=_session_antigravity(), decision=decision, resume=False), "complete")
        return seen, make_engine.call_count

    def test_normal_mlx_run_reuses_one_engine_for_every_dataset_without_pool(self) -> None:
        seen, created = self._normal_serial_run(_mlx_decision_antigravity())
        self.assertEqual(created, 1); self.assertEqual(len(seen), 2); self.assertIs(seen[0], seen[1])

    def test_normal_mps_run_reuses_one_engine_for_every_dataset_without_pool(self) -> None:
        seen, created = self._normal_serial_run(_mps_decision_antigravity())
        self.assertEqual(created, 1); self.assertEqual(len(seen), 2); self.assertIs(seen[0], seen[1])
    def test_mlx_and_mps_engine_factories_validate_exactly_one_selected_instance(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        from experiments.exp436_missing_safe_strict_consensus_antigravity import mlx_antigravity, mps_antigravity
        mlx_engine, mps_engine, validated = object(), object(), []
        with patch.object(mlx_antigravity, "MlxEngine", return_value=mlx_engine), patch.object(mps_antigravity, "MpsEngine", return_value=mps_engine), patch.object(runtime_antigravity, "validate_backend_engine", side_effect=lambda decision, engine: validated.append((decision.backend, engine)) or engine):
            self.assertIs(runtime_antigravity._engine(_mlx_decision_antigravity()), mlx_engine)
            self.assertIs(runtime_antigravity._engine(_mps_decision_antigravity()), mps_engine)
        self.assertEqual(validated, [("mlx", mlx_engine), ("mps", mps_engine)])

    def test_accelerator_interrupt_persists_one_terminal_event_and_never_constructs_pool(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        with TemporaryDirectory() as directory:
            root = Path(directory); readiness_path = _readiness_antigravity(runtime_antigravity, root); readiness = json.loads(readiness_path.read_text())
            decision = _mlx_decision_antigravity()
            with patch.object(runtime_antigravity, "_engine", return_value=object()), patch.object(runtime_antigravity, "_score_accelerated", side_effect=KeyboardInterrupt()), patch.object(runtime_antigravity, "ProcessPoolExecutor", side_effect=AssertionError("accelerator must not create pool")):
                with self.assertRaises(KeyboardInterrupt):
                    runtime_antigravity._run(run_root=root / "output_antigravity", checkpoint_root=root / "checkpoint_antigravity", readiness_path=readiness_path, readiness=readiness, input_session=_session_antigravity(), decision=decision, resume=False)
            events = [json.loads(line) for line in (root / "output_antigravity" / "events_antigravity.jsonl").read_text().splitlines()]
            self.assertEqual([event["kind"] for event in events].count("interrupted"), 1)
            self.assertEqual(json.loads((root / "output_antigravity" / "progress_antigravity.json").read_text())["state"], "interrupted")

    def test_mps_interrupt_persists_terminal_without_cpu_fallback_or_pool(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        with TemporaryDirectory() as directory:
            root = Path(directory); readiness_path = _readiness_antigravity(runtime_antigravity, root); readiness = json.loads(readiness_path.read_text())
            with patch.object(runtime_antigravity, "_engine", return_value=object()), patch.object(runtime_antigravity, "_score_accelerated", side_effect=KeyboardInterrupt()), patch.object(runtime_antigravity, "ProcessPoolExecutor", side_effect=AssertionError("MPS must not create pool")):
                with self.assertRaises(KeyboardInterrupt):
                    runtime_antigravity._run(run_root=root / "output_antigravity", checkpoint_root=root / "checkpoint_antigravity", readiness_path=readiness_path, readiness=readiness, input_session=_session_antigravity(), decision=_mps_decision_antigravity(), resume=False)
            self.assertEqual(json.loads((root / "output_antigravity" / "progress_antigravity.json").read_text())["backend"], "mps")

    def test_accelerator_runtime_error_fails_once_without_cpu_fallback_or_checkpoint_advance(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        with TemporaryDirectory() as directory:
            root=Path(directory); readiness_path=_readiness_antigravity(runtime_antigravity,root); readiness=json.loads(readiness_path.read_text())
            with patch.object(runtime_antigravity,"_engine",return_value=object()), patch.object(runtime_antigravity,"_score_accelerated",side_effect=RuntimeError("device lost")), patch.object(runtime_antigravity,"ProcessPoolExecutor",side_effect=AssertionError("no fallback pool")):
                with self.assertRaisesRegex(RuntimeError,"device lost"):
                    runtime_antigravity._run(run_root=root/"output_antigravity",checkpoint_root=root/"checkpoint_antigravity",readiness_path=readiness_path,readiness=readiness,input_session=_session_antigravity(),decision=_mlx_decision_antigravity(),resume=False)
            progress=json.loads((root/"output_antigravity"/"progress_antigravity.json").read_text())
            self.assertEqual(progress["state"],"failed"); self.assertEqual(json.loads((root/"checkpoint_antigravity"/"record_set_antigravity.json").read_text())["completed"],0)

    def test_runtime_engine_provenance_mismatch_is_one_failed_terminal_without_fallback(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        with TemporaryDirectory() as directory:
            root=Path(directory); readiness_path=_readiness_antigravity(runtime_antigravity,root); readiness=json.loads(readiness_path.read_text())
            with patch.object(runtime_antigravity,"_engine",side_effect=ValueError("engine device provenance does not match decision")), patch.object(runtime_antigravity,"ProcessPoolExecutor",side_effect=AssertionError("no CPU fallback")):
                with self.assertRaisesRegex(ValueError,"provenance"):
                    runtime_antigravity._run(run_root=root/"output_antigravity",checkpoint_root=root/"checkpoint_antigravity",readiness_path=readiness_path,readiness=readiness,input_session=_session_antigravity(),decision=_mlx_decision_antigravity(),resume=False)
            events=[json.loads(line) for line in (root/"output_antigravity"/"events_antigravity.jsonl").read_text().splitlines()]
            self.assertEqual([event["kind"] for event in events].count("failed"),1); self.assertEqual(json.loads((root/"checkpoint_antigravity"/"record_set_antigravity.json").read_text())["completed"],0)

    def test_interrupted_accelerator_resume_constructs_one_fresh_engine_not_old_instance(self) -> None:
        from experiments.exp436_missing_safe_strict_consensus_antigravity import runtime_antigravity
        with TemporaryDirectory() as directory:
            root=Path(directory); readiness_path=_readiness_antigravity(runtime_antigravity,root); readiness=json.loads(readiness_path.read_text()); old,new=object(),object()
            with patch.object(runtime_antigravity,"_engine",side_effect=[old,new]) as engines, patch.object(runtime_antigravity,"_score_accelerated",side_effect=KeyboardInterrupt()), patch.object(runtime_antigravity,"ProcessPoolExecutor",side_effect=AssertionError("serial accelerator has no pool")):
                with self.assertRaises(KeyboardInterrupt): runtime_antigravity._run(run_root=root/"output_antigravity",checkpoint_root=root/"checkpoint_antigravity",readiness_path=readiness_path,readiness=readiness,input_session=_session_antigravity(),decision=_mlx_decision_antigravity(),resume=False)
                with self.assertRaises(KeyboardInterrupt): runtime_antigravity._run(run_root=root/"output_antigravity",checkpoint_root=root/"checkpoint_antigravity",readiness_path=readiness_path,readiness=readiness,input_session=_session_antigravity(),decision=_mlx_decision_antigravity(),resume=True)
            self.assertEqual(engines.call_count,2); self.assertIsNot(old,new)
