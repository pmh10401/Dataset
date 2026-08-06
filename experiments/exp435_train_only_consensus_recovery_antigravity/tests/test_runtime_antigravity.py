"""TDD checks for deterministic Exp435 serial/parallel runtime scheduling."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import subprocess
import sys
import time
import unittest
from concurrent.futures import ProcessPoolExecutor
from pickle import dumps, loads
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch

from experiments.exp435_train_only_consensus_recovery_antigravity.contracts_antigravity import (
    BackendBatchEvidence,
    BackendDecision,
    Exp435Candidate,
    Exp435DatasetResult,
    InstanceScoreEvidence,
)
from experiments.exp435_train_only_consensus_recovery_antigravity.checkpoint_antigravity import (
    Exp435CheckpointStore,
)
from experiments.exp435_train_only_consensus_recovery_antigravity.runtime_antigravity import (
    LaneMembershipSnapshot,
    _lane_snapshot,
    _runtime_context,
    _score_worker,
    inspect_exp435,
    run_exp435_new,
)
from experiments.exp435_train_only_consensus_recovery_antigravity.config_antigravity import FROZEN_CONFIG
from experiments.exp435_train_only_consensus_recovery_antigravity.scoring_antigravity import (
    CandidateBudgetExceeded,
)


class _Session:
    def __init__(self, datasets: tuple[str, ...]):
        self.context = {
            "datasets": list(datasets),
            "test_sizes": {name: 2 for name in datasets},
            "lanes": {
                "hard": MappingProxyType({}),
                "standard": MappingProxyType({}),
                "priority": MappingProxyType({}),
                "automatic_normal": MappingProxyType({}),
            },
        }


def _score(train: object, test: object, lanes: object, backend: object) -> Exp435DatasetResult:
    dataset = getattr(train, "dataset", None) or getattr(test, "dataset", None)
    if dataset is None and isinstance(train, dict):
        dataset = train["dataset"]
    evidence = InstanceScoreEvidence.fixture(dataset=str(dataset), instance_index=0)
    # Deliberately vary completion latency; runtime must still commit name order.
    if str(dataset).endswith("B"):
        time.sleep(0.02)
    return Exp435DatasetResult(dataset=str(dataset), status="scored", test_size=2, evidence=(evidence,))


def _budget_score(train: object, test: object, lanes: object, backend: object) -> Exp435DatasetResult:
    del test, lanes, backend
    dataset = str(getattr(train, "get", lambda _key: "A")("dataset"))
    evidence = tuple(
        InstanceScoreEvidence.fixture(
            dataset=dataset,
            instance_index=index,
            length=16,
            prototype_abnormal=True,
            gradient_abnormal=True,
        )
        for index in range(101)
    )
    candidates = tuple(
        Exp435Candidate(dataset, item.instance_index, item.length, item.evidence_digest)
        for item in evidence
    )
    return Exp435DatasetResult(
        dataset=dataset,
        status="scored",
        test_size=101,
        candidates=candidates,
        evidence=evidence,
    )


def _parallel_failure_score(train: object, test: object, lanes: object, backend: object) -> Exp435DatasetResult:
    del test, lanes, backend
    dataset = str(getattr(train, "get", lambda _key: "A")("dataset"))
    if dataset == "B":
        raise RuntimeError("synthetic worker failure")
    time.sleep(1.0)
    return _score(train, {"dataset": dataset}, None, None)


def _parallel_budget_score(train: object, test: object, lanes: object, backend: object) -> Exp435DatasetResult:
    del test, lanes, backend
    dataset = str(getattr(train, "get", lambda _key: "A")("dataset"))
    if dataset == "A":
        return _budget_score(train, {"dataset": dataset}, None, None)
    time.sleep(1.5)
    return _score(train, {"dataset": dataset}, None, None)


def _interrupt_score(train: object, test: object, lanes: object, backend: object) -> Exp435DatasetResult:
    del train, test, lanes, backend
    raise KeyboardInterrupt("synthetic interrupt")


class _BlockedExecutor:
    """In-process executor fixture that exposes exact submitted work."""

    instances: list["_BlockedExecutor"] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.submitted: list[tuple[object, tuple[object, ...]]] = []
        self.shutdown_calls: list[tuple[bool, bool]] = []
        type(self).instances.append(self)

    def submit(self, function: object, *args: object):
        from concurrent.futures import Future

        future = Future()
        self.submitted.append((future, (function, *args)))
        return future

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        self.shutdown_calls.append((wait, cancel_futures))


def _adopted_backend() -> BackendDecision:
    evidence = BackendBatchEvidence(
        batch_size=16,
        finite_outputs=True,
        candidate_identity_match=True,
        parity_passed=True,
        normalized_error_max=0.0,
        absolute_error_max=0.0,
        speed_passed=True,
        cpu_median_seconds=1.0,
        mlx_median_seconds=0.5,
        speedup=2.0,
        reason="fixture",
    )
    return BackendDecision(
        backend="mlx",
        adopted=True,
        smoke_seed=43501,
        finite_outputs=True,
        candidate_identity_match=True,
        parity_passed=True,
        normalized_error_max=0.0,
        absolute_error_max=0.0,
        speedup=2.0,
        batch_size=16,
        selected_batch_size=16,
        cpu_median_seconds=1.0,
        mlx_median_seconds=0.5,
        mlx_version="fixture-mlx",
        mlx_metal_version="fixture-metal",
        device="fixture-device",
        device_info="fixture-device-info",
        metal_available=True,
        timings=((16, 1.0, 0.5),),
        batch_evidence=(evidence,),
        reason="fixture",
    )


class RuntimeTests(unittest.TestCase):
    def test_lane_snapshot_roundtrips_and_preserves_all_four_lanes(self) -> None:
        lanes = SimpleNamespace(
            hard=MappingProxyType({"A": (0,)}),
            standard=MappingProxyType({"A": (1,)}),
            priority=MappingProxyType({"A": (2,)}),
            automatic_normal=MappingProxyType({"A": (3,)}),
        )
        snapshot = _lane_snapshot(lanes, ("A",), {"A": 4})
        self.assertIsInstance(loads(dumps(snapshot)), LaneMembershipSnapshot)
        self.assertTrue(all(snapshot.contains("A", index) for index in range(4)))

    def test_lane_snapshot_is_accepted_by_an_actual_process_worker(self) -> None:
        lanes = SimpleNamespace(
            hard=MappingProxyType({"A": (0,)}),
            standard=MappingProxyType({"A": (1,)}),
            priority=MappingProxyType({"A": (2,)}),
            automatic_normal=MappingProxyType({"A": (3,)}),
        )
        snapshot = _lane_snapshot(lanes, ("A",), {"A": 4})
        with ProcessPoolExecutor(max_workers=1) as executor:
            result = executor.submit(
                _score_worker,
                _score,
                "A",
                {"dataset": "A"},
                {"dataset": "A"},
                snapshot,
                None,
            ).result()
        self.assertEqual(result.dataset, "A")

    def test_runtime_context_derives_and_freezes_exact_test_sizes(self) -> None:
        session = _Session(("A", "B"))
        session.context.pop("test_sizes")
        session.context["test_identity_manifest"] = {
            "A": [[0, 8], [1, 8]],
            "B": [[0, 4]],
        }
        projected = _runtime_context(session, BackendDecision.cpu_fallback("fixture"))
        self.assertEqual(projected["test_sizes"], {"A": 2, "B": 1})
        session.context["test_sizes"] = {"A": 2, "B": 2}
        with self.assertRaisesRegex(ValueError, "disagree"):
            _runtime_context(session, BackendDecision.cpu_fallback("fixture"))

    def test_runtime_context_checkpoint_and_progress_use_the_frozen_run_identity(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            session = _Session(("A",))
            session.context["code_commit"] = "f" * 40
            projected = _runtime_context(session, BackendDecision.cpu_fallback("fixture"))
            self.assertEqual(projected["run_identity"], FROZEN_CONFIG.run_identity)
            store = Exp435CheckpointStore.create(root / "checkpoint", projected)
            self.assertEqual(store.run_identity, FROZEN_CONFIG.run_identity)

    def test_thread_limits_are_set_before_numpy_or_artifact_import_in_clean_process(self) -> None:
        script = r'''
import builtins
import json
import os

names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")
observed = []
real_import = builtins.__import__

def traced(name, *args, **kwargs):
    if name == "numpy" or name.startswith("numpy.") or name == "experiments.exp403_408.artifact_io":
        observed.append((name, {key: os.environ.get(key) for key in names}))
    return real_import(name, *args, **kwargs)

builtins.__import__ = traced
import experiments.exp435_train_only_consensus_recovery_antigravity.runtime_antigravity
print(json.dumps(observed, sort_keys=True))
'''
        environment = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
            environment[name] = "999"
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[3],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        observed = json.loads(result.stdout)
        self.assertTrue(observed)
        self.assertTrue(all(all(value == "1" for value in values.values()) for _, values in observed))

    def test_serial_and_parallel_checkpoint_order_and_digest_are_equal(self) -> None:
        datasets = ("A", "B")
        for worker_count in (1, 7):
            with self.subTest(worker_count=worker_count), TemporaryDirectory() as directory:
                root = Path(directory)
                session = _Session(datasets)
                # Fixture sessions expose values directly; the runtime boundary
                # accepts a scorer that owns the decoding contract in production.
                session.load_train = lambda dataset: {"dataset": dataset}  # type: ignore[attr-defined]
                session.load_test_values = lambda dataset, train: {"dataset": dataset}  # type: ignore[attr-defined]
                run_exp435_new(
                    run_root=root / "output",
                    checkpoint_root=root / "checkpoint",
                    input_session=session,
                    backend_decision=None,
                    dataset_scorer=_score,
                    worker_count=worker_count,
                )
                records = sorted((root / "checkpoint" / "records_antigravity").glob("*.json"))
                names = [path.read_text() for path in records]
                if worker_count == 1:
                    serial = names
                else:
                    self.assertEqual(names, serial)

    def test_inspect_reconstructs_runtime_context_without_writing(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            session = _Session(("A", "B"))
            session.load_train = lambda dataset: {"dataset": dataset}  # type: ignore[attr-defined]
            session.load_test_values = lambda dataset, train: {"dataset": dataset}  # type: ignore[attr-defined]
            run_exp435_new(
                run_root=root / "output",
                checkpoint_root=root / "checkpoint",
                input_session=session,
                backend_decision=None,
                dataset_scorer=_score,
                worker_count=1,
            )
            paths = sorted(path for path in root.rglob("*") if path.is_file())
            before = {path.relative_to(root): path.read_bytes() for path in paths}
            summary = inspect_exp435(
                run_root=root / "output",
                checkpoint_root=root / "checkpoint",
                expected_context=session.context,
            )
            self.assertEqual(summary.completed_datasets, 2)
            after = {
                path.relative_to(root): path.read_bytes()
                for path in sorted(path for path in root.rglob("*") if path.is_file())
            }
            self.assertEqual(before, after)

    def test_adopted_default_mlx_path_is_serial_and_uses_one_engine(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            session = _Session(("A", "B"))
            session.load_train = lambda dataset: {"dataset": dataset}  # type: ignore[attr-defined]
            session.load_test_values = lambda dataset, train: {"dataset": dataset}  # type: ignore[attr-defined]
            with patch(
                "experiments.exp435_train_only_consensus_recovery_antigravity.mlx_antigravity.MlxEngine",
                return_value=object(),
            ) as engine, patch(
                "experiments.exp435_train_only_consensus_recovery_antigravity.runtime_antigravity._default_mlx_dataset_scorer",
                side_effect=lambda train, test, lanes, backend, active_engine: _score(train, test, lanes, backend),
            ) as scorer:
                summary = run_exp435_new(
                    run_root=root / "output",
                    checkpoint_root=root / "checkpoint",
                    input_session=session,
                    backend_decision=_adopted_backend(),
                    dataset_scorer=None,
                )
            self.assertEqual(summary.completed_datasets, 2)
            self.assertEqual(engine.call_count, 1)
            self.assertEqual(scorer.call_count, 2)

    def test_inspect_rejects_cpu_output_bound_to_swapped_mlx_checkpoint(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            session = _Session(("A",))
            session.context["code_commit"] = "f" * 40
            session.load_train = lambda dataset: {"dataset": dataset}  # type: ignore[attr-defined]
            session.load_test_values = lambda dataset, train: {"dataset": dataset}  # type: ignore[attr-defined]
            run_exp435_new(
                run_root=root / "output",
                checkpoint_root=root / "checkpoint-cpu",
                input_session=session,
                backend_decision=None,
                dataset_scorer=_score,
                worker_count=1,
            )
            mlx_context = _runtime_context(session, _adopted_backend())
            Exp435CheckpointStore.create(root / "checkpoint-mlx", mlx_context)
            with self.assertRaisesRegex(ValueError, "binding"):
                inspect_exp435(
                    run_root=root / "output",
                    checkpoint_root=root / "checkpoint-mlx",
                    expected_context=session.context,
                )

    def test_candidate_budget_failure_is_recorded_before_no_record_is_published(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            session = _Session(("A",))
            session.context["test_sizes"] = {"A": 101}
            session.load_train = lambda dataset: {"dataset": dataset}  # type: ignore[attr-defined]
            session.load_test_values = lambda dataset, train: {"dataset": dataset}  # type: ignore[attr-defined]
            with self.assertRaises(CandidateBudgetExceeded):
                run_exp435_new(
                    run_root=root / "output",
                    checkpoint_root=root / "checkpoint",
                    input_session=session,
                    backend_decision=None,
                    dataset_scorer=_budget_score,
                    worker_count=1,
                )
            events = [
                json.loads(line)
                for line in (root / "output" / "events_antigravity.jsonl").read_text().splitlines()
            ]
            budget_events = [item for item in events if item.get("kind") == "candidate_budget_exceeded"]
            self.assertEqual(len(budget_events), 1)
            self.assertEqual(budget_events[0]["dataset"], "A")
            self.assertEqual(budget_events[0]["observed"], 101)
            self.assertEqual(budget_events[0]["budget"], 100)
            self.assertEqual(tuple((root / "checkpoint" / "records_antigravity").iterdir()), ())
            inspected = inspect_exp435(
                run_root=root / "output",
                checkpoint_root=root / "checkpoint",
                expected_context=session.context,
            )
            self.assertEqual(inspected.status, "failed")
            self.assertEqual(inspected.completed_datasets, 1)
            self.assertEqual(inspected.successful_datasets, 0)
            self.assertEqual(inspected.abstained_datasets, 0)
            self.assertEqual(inspected.failed_datasets, 1)

    def test_parallel_worker_failure_records_offending_dataset_and_does_not_wait_for_sibling(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            session = _Session(("A", "B"))
            session.load_train = lambda dataset: {"dataset": dataset}  # type: ignore[attr-defined]
            session.load_test_values = lambda dataset, train: {"dataset": dataset}  # type: ignore[attr-defined]
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, "synthetic worker failure"):
                run_exp435_new(
                    run_root=root / "output",
                    checkpoint_root=root / "checkpoint",
                    input_session=session,
                    backend_decision=None,
                    dataset_scorer=_parallel_failure_score,
                    worker_count=7,
                )
            self.assertLess(time.monotonic() - started, 2.5)
            events = [
                json.loads(line)
                for line in (root / "output" / "events_antigravity.jsonl").read_text().splitlines()
            ]
            failures = [item for item in events if item.get("kind") == "failure"]
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["dataset"], "B")

    def test_parallel_progress_never_reports_or_submits_an_eighth_blocked_dataset(self) -> None:
        datasets = tuple(f"Dataset-{index}" for index in range(1, 9))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            session = _Session(datasets)
            loaded: list[str] = []
            session.load_train = lambda dataset: loaded.append(dataset) or {"dataset": dataset}  # type: ignore[attr-defined]
            session.load_test_values = lambda dataset, train: {"dataset": dataset}  # type: ignore[attr-defined]
            _BlockedExecutor.instances.clear()
            waits = 0

            def release_when_observed(pending, *, return_when):
                del return_when
                nonlocal waits
                waits += 1
                executor = _BlockedExecutor.instances[0]
                submitted = [args[2] for _future, args in executor.submitted]
                if waits == 1:
                    self.assertEqual(submitted, list(datasets[:7]))
                    self.assertEqual(loaded, list(datasets[:7]))
                    payload = json.loads((root / "output" / "progress_antigravity.json").read_text())
                    self.assertNotEqual(payload["current_dataset"], datasets[7])
                    self.assertEqual(payload["stage"], "scoring")
                    future = next(iter(pending))
                    worker, scorer, dataset, train, test, lanes, backend = executor.submitted[0][1]
                    future.set_result(worker(scorer, dataset, train, test, lanes, backend))
                    return {future}, set()
                self.assertEqual(submitted, list(datasets))
                for future, args in executor.submitted:
                    if not future.done():
                        worker, scorer, dataset, train, test, lanes, backend = args
                        future.set_result(worker(scorer, dataset, train, test, lanes, backend))
                return set(pending), set()

            with patch(
                "experiments.exp435_train_only_consensus_recovery_antigravity.runtime_antigravity.ProcessPoolExecutor",
                _BlockedExecutor,
            ), patch(
                "experiments.exp435_train_only_consensus_recovery_antigravity.runtime_antigravity.wait",
                side_effect=release_when_observed,
            ):
                summary = run_exp435_new(
                    run_root=root / "output",
                    checkpoint_root=root / "checkpoint",
                    input_session=session,
                    backend_decision=None,
                    dataset_scorer=_score,
                    worker_count=7,
                )
            self.assertEqual(summary.completed_datasets, len(datasets))
            self.assertEqual(waits, 2)

    def test_keyboard_interrupt_has_one_terminal_state_and_read_only_summary(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            session = _Session(("A",))
            session.load_train = lambda dataset: {"dataset": dataset}  # type: ignore[attr-defined]
            session.load_test_values = lambda dataset, train: {"dataset": dataset}  # type: ignore[attr-defined]
            with self.assertRaises(KeyboardInterrupt):
                run_exp435_new(
                    run_root=root / "output",
                    checkpoint_root=root / "checkpoint",
                    input_session=session,
                    backend_decision=None,
                    dataset_scorer=_interrupt_score,
                    worker_count=7,
                )
            events = [
                json.loads(line)
                for line in (root / "output" / "events_antigravity.jsonl").read_text().splitlines()
            ]
            interrupted = [item for item in events if item.get("kind") == "state" and item.get("state") == "interrupted"]
            self.assertEqual(len(interrupted), 1)
            inspected = inspect_exp435(
                run_root=root / "output",
                checkpoint_root=root / "checkpoint",
                expected_context=session.context,
            )
            self.assertEqual(inspected.status, "interrupted")
            self.assertEqual(inspected.completed_datasets, 0)
            self.assertEqual(inspected.failed_datasets, 0)

    def test_parallel_budget_overflow_is_persisted_before_slow_sibling_finishes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            session = _Session(("A", "B"))
            session.context["test_sizes"] = {"A": 101, "B": 2}
            session.load_train = lambda dataset: {"dataset": dataset}  # type: ignore[attr-defined]
            session.load_test_values = lambda dataset, train: {"dataset": dataset}  # type: ignore[attr-defined]
            started = time.monotonic()
            with self.assertRaises(CandidateBudgetExceeded):
                run_exp435_new(
                    run_root=root / "output",
                    checkpoint_root=root / "checkpoint",
                    input_session=session,
                    backend_decision=None,
                    dataset_scorer=_parallel_budget_score,
                    worker_count=7,
                )
            self.assertLess(time.monotonic() - started, 1.4)
            events = [
                json.loads(line)
                for line in (root / "output" / "events_antigravity.jsonl").read_text().splitlines()
            ]
            budget_events = [item for item in events if item.get("kind") == "candidate_budget_exceeded"]
            self.assertEqual(len(budget_events), 1)
            self.assertEqual(budget_events[0]["dataset"], "A")
            self.assertEqual(budget_events[0]["observed"], 101)
            self.assertEqual(tuple((root / "checkpoint" / "records_antigravity").iterdir()), ())


if __name__ == "__main__":
    unittest.main()
