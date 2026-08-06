"""TDD checks for deterministic Exp435 serial/parallel runtime scheduling."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
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
from experiments.exp435_train_only_consensus_recovery_antigravity.runtime_antigravity import (
    LaneMembershipSnapshot,
    _lane_snapshot,
    _runtime_context,
    _score_worker,
    inspect_exp435,
    run_exp435_new,
)
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
