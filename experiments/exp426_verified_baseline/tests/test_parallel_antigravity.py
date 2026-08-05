"""Fixed twelve-process executor tests for Exp426."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.exp426_verified_baseline.contracts_antigravity import (
    Exp426DatasetRecord,
)
from experiments.exp426_verified_baseline.parallel_antigravity import (
    Exp426TrainWorkerTask,
    Exp426SourceWorkerTask,
    _Exp426ProbeTask,
    _iter_ordered_workers_for_test,
    _run_ordered_workers_for_test,
    run_ordered_workers,
)
from experiments.exp417_causal_multihorizon_shadow.contracts import (
    RawSeries,
    RawTrainDataset,
)
from experiments.exp425_selective_recovery.predecessor import (
    open_validated_exp424_source,
)
from experiments.exp425_selective_recovery.tests.test_eligibility import (
    _series_with_identity_classes,
)
from experiments.exp426_verified_baseline.source_antigravity import (
    read_verified_source_tasks,
)
from experiments.exp426_verified_baseline.tests.test_source_antigravity import _source_fixture


class Exp426ParallelTests(unittest.TestCase):
    thread_names = (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    )

    @staticmethod
    def record(dataset: str) -> Exp426DatasetRecord:
        return Exp426DatasetRecord(
            dataset, "exp424_import", "scored", (),
            "b" * 64, "c" * 64, None, None,
            "verified_exp424_nonterminal",
        )

    def test_twelve_processes_limit_inner_threads_prefetch_and_emit_dataset_order(self) -> None:
        tasks = tuple(
            _Exp426ProbeTask(
                f"dataset-{index:02d}",
                self.record(f"dataset-{index:02d}"),
                delay_seconds=(24 - index) * 0.01,
            )
            for index in range(24)
        )

        results, audit = _run_ordered_workers_for_test(
            tasks, workers=12, prefetch=24
        )

        self.assertEqual(
            tuple(result.dataset for result in results),
            tuple(f"dataset-{index:02d}" for index in range(24)),
        )
        self.assertEqual(len(audit.worker_pids), 12)
        self.assertNotIn(os.getpid(), audit.worker_pids)
        self.assertLessEqual(audit.max_in_flight, 24)
        self.assertTrue(all(
            dict(limits) == {name: "1" for name in self.thread_names}
            for limits in audit.thread_limits
        ))

    def test_one_and_twelve_processes_have_identical_scientific_results(self) -> None:
        tasks = tuple(
            _Exp426ProbeTask(
                f"dataset-{index:02d}", self.record(f"dataset-{index:02d}"), 0.0
            )
            for index in range(24)
        )
        single, _single_audit = _run_ordered_workers_for_test(
            tasks, workers=1, prefetch=2
        )
        twelve, _twelve_audit = _run_ordered_workers_for_test(
            tasks, workers=12, prefetch=24
        )
        self.assertEqual(
            [result.scientific_sha256 for result in single],
            [result.scientific_sha256 for result in twelve],
        )

    def test_task_input_is_consumed_lazily_with_at_most_prefetch_items(self) -> None:
        allowed = [24]

        def guarded_tasks():
            for index in range(30):
                if index >= allowed[0]:
                    raise AssertionError("executor consumed beyond bounded prefetch")
                yield _Exp426ProbeTask(
                    f"dataset-{index:02d}",
                    self.record(f"dataset-{index:02d}"),
                    0.01,
                )

        observed = []
        for result in _iter_ordered_workers_for_test(
            guarded_tasks(), workers=12, prefetch=24
        ):
            observed.append(result.dataset)
            allowed[0] += 1
        self.assertEqual(len(observed), 30)

    def test_public_executor_fixes_twelve_workers_and_worker_errors_name_dataset(self) -> None:
        task = _Exp426ProbeTask("dataset-error", self.record("dataset-error"), 0.0, fail=True)
        with self.assertRaisesRegex(ValueError, "workers"):
            tuple(run_ordered_workers((task,), workers=1))
        with self.assertRaisesRegex(ValueError, "test probe"):
            tuple(run_ordered_workers((task,), workers=12))
        with self.assertRaisesRegex(RuntimeError, "dataset-error"):
            _run_ordered_workers_for_test((task,), workers=1, prefetch=1)

    def test_train_worker_task_cannot_carry_paths_connections_labels_or_checkpoint(self) -> None:
        train = RawTrainDataset(
            "dataset-train",
            (RawSeries(0, tuple(float(index) for index in range(64))),),
        )
        task = Exp426TrainWorkerTask(
            "dataset-train", train, "a" * 64, "b" * 64, "c" * 64
        )
        values = vars(task)
        self.assertEqual(
            set(values),
            {
                "dataset", "train", "fold_contract_sha256",
                "source_record_sha256", "source_result_sha256",
            },
        )
        self.assertFalse(any(isinstance(value, Path) for value in values.values()))
        self.assertFalse(any(
            token in type(value).__name__.lower()
            for value in values.values()
            for token in ("connection", "checkpoint", "label")
        ))

    def test_real_source_classification_is_identical_with_one_and_twelve_workers(self) -> None:
        with TemporaryDirectory() as directory:
            root, contract, _datasets = _source_fixture(Path(directory))
            with open_validated_exp424_source(root, contract) as source:
                tasks = tuple(
                    Exp426SourceWorkerTask(item.dataset, item.raw, item.expected_digest)
                    for item in read_verified_source_tasks(source)
                )
            single, _ = _run_ordered_workers_for_test(tasks, workers=1, prefetch=2)
            twelve, _ = _run_ordered_workers_for_test(tasks, workers=12, prefetch=24)
            self.assertEqual(
                [item.scientific_sha256 for item in single],
                [item.scientific_sha256 for item in twelve],
            )
            self.assertEqual(
                sum(hasattr(item.value, "terminal_reason") for item in twelve), 1
            )

    def test_real_train_task_runs_in_worker_and_returns_verified_abstention(self) -> None:
        identities = _series_with_identity_classes(64, early_count=4, other_count=6)
        train = RawTrainDataset(
            "dataset-train",
            tuple(
                RawSeries(index, identities[index % len(identities)].values)
                for index in range(100)
            ),
        )
        task = Exp426TrainWorkerTask(
            "dataset-train", train, "a" * 64, "b" * 64, "c" * 64
        )
        results, audit = _run_ordered_workers_for_test(
            (task,), workers=1, prefetch=1
        )
        evidence = results[0].value
        self.assertEqual(evidence.record.origin, "verified_train_abstention")
        self.assertEqual(evidence.record.status, "abstained")
        self.assertEqual(len(audit.worker_pids), 1)


if __name__ == "__main__":
    unittest.main()
