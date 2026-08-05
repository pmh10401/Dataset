"""Resumable parent-owned runtime and durable log tests for Exp426."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import time
import unittest

from experiments.exp417_causal_multihorizon_shadow.contracts import RawSeries, RawTrainDataset
from experiments.exp426_verified_baseline.contracts_antigravity import (
    Exp426DatasetRecord,
    Exp426Provenance,
    WorkerEvidence,
)
from experiments.exp426_verified_baseline.parallel_antigravity import Exp426TrainWorkerTask
from experiments.exp426_verified_baseline.source_antigravity import (
    Exp424SourceClassification,
)
from experiments.exp426_verified_baseline.runtime_antigravity import (
    _RuntimeDependencies,
    inspect_exp426,
    run_exp426_new,
    run_exp426_resume,
)


@dataclass(frozen=True)
class _FakeSourceTask:
    dataset: str
    classification: Exp424SourceClassification


class _FakeDatabase:
    def __init__(self, trains: dict[str, RawTrainDataset]) -> None:
        self.trains = trains
        self.loads: list[str] = []
        self.boundaries: list[str] = []


class _FakeRuntime:
    def __init__(self, classifications: tuple[Exp424SourceClassification, ...]) -> None:
        self.tasks = tuple(
            _FakeSourceTask(item.dataset, item) for item in classifications
        )
        self.runner_calls: list[tuple[int, int]] = []
        self.worker_datasets: list[str] = []
        self.revalidations = 0

    def source_tasks(self, _source: object):
        return iter(self.tasks)

    def revalidate(self, _source: object) -> None:
        self.revalidations += 1

    def run_workers(self, tasks, *, workers: int, prefetch: int):
        self.runner_calls.append((workers, prefetch))
        for ordinal, task in enumerate(tasks):
            self.worker_datasets.append(task.dataset)
            if isinstance(task, _FakeSourceTask):
                value = task.classification
            elif isinstance(task, Exp426TrainWorkerTask):
                value = WorkerEvidence(
                    ordinal,
                    task.dataset,
                    Exp426DatasetRecord(
                        task.dataset, "verified_train_abstention", "abstained", (),
                        task.source_record_sha256, task.source_result_sha256,
                        "d" * 64, "e" * 64, "oof_identity_insufficient",
                    ),
                )
            else:
                raise AssertionError(f"unexpected task: {task!r}")
            yield SimpleNamespace(dataset=task.dataset, value=value)

    @staticmethod
    def load_train(database: _FakeDatabase, dataset: str, _context: object) -> RawTrainDataset:
        database.loads.append(dataset)
        return database.trains[dataset]

    @staticmethod
    def verify_database(database: _FakeDatabase, name: str) -> None:
        database.boundaries.append(name)

    def dependencies(self) -> _RuntimeDependencies:
        return _RuntimeDependencies(
            source_tasks=self.source_tasks,
            source_revalidate=self.revalidate,
            worker_runner=self.run_workers,
            load_train=self.load_train,
            verify_database=self.verify_database,
            heartbeat_seconds=None,
            process_is_live=lambda _pid: False,
        )


class Exp426RuntimeTests(unittest.TestCase):
    @staticmethod
    def datasets(total: int) -> tuple[str, ...]:
        return tuple(f"dataset-{index:02d}" for index in range(total))

    @staticmethod
    def provenance(datasets: tuple[str, ...]) -> Exp426Provenance:
        return Exp426Provenance(
            "exp426-runtime", "1" * 40, "2" * 64, "3" * 64,
            "4" * 64, "5" * 64,
            {
                "hard": "6" * 64, "standard": "7" * 64,
                "priority": "8" * 64, "automatic_normal": "9" * 64,
            },
            "a" * 64, datasets, 12,
        )

    @staticmethod
    def checkpoint_context(datasets: tuple[str, ...]) -> dict[str, object]:
        return {
            "schema": "exp426.frozen-context.v1",
            "database_sha256": "2" * 64,
            "datasets": list(datasets),
            "test_sizes": {dataset: 4 for dataset in datasets},
            "lane_coordinates": [],
            "code_commit": "1" * 40,
            "runtime_versions": {"numpy": "test", "python": "3.14-test"},
        }

    @staticmethod
    def classifications(
        datasets: tuple[str, ...], imported: int,
    ) -> tuple[Exp424SourceClassification, ...]:
        result = []
        for index, dataset in enumerate(datasets):
            source_record = f"{index + 10:064x}"
            source_result = f"{index + 100:064x}"
            if index < imported:
                record = Exp426DatasetRecord(
                    dataset, "exp424_import", "scored", (),
                    source_record, source_result, None, None,
                    "verified_exp424_nonterminal",
                )
                result.append(Exp424SourceClassification(
                    dataset, source_record, source_result, record, None,
                    100, (64,), 4, (64,),
                ))
            else:
                result.append(Exp424SourceClassification(
                    dataset, source_record, source_result, None,
                    "oof_identity_insufficient", 0, (), 4, (64,),
                ))
        return tuple(result)

    @staticmethod
    def trains(datasets: tuple[str, ...]) -> dict[str, RawTrainDataset]:
        return {
            dataset: RawTrainDataset(
                dataset,
                tuple(
                    RawSeries(index, tuple(float(value + index) for value in range(64)))
                    for index in range(3)
                ),
            )
            for dataset in datasets
        }

    def test_mixed_run_imports_twelve_and_loads_train_only_for_three_terminals(self) -> None:
        datasets = self.datasets(15)
        classifications = self.classifications(datasets, imported=12)
        fake = _FakeRuntime(classifications)
        terminal = datasets[12:]
        database = _FakeDatabase(self.trains(terminal))
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            result = run_exp426_new(
                parent / "output", parent / "checkpoint",
                self.provenance(datasets), self.checkpoint_context(datasets),
                object(), database, object(), _dependencies=fake.dependencies(),
            )

            self.assertEqual(result.counts.completed, 15)
            self.assertEqual(result.counts.imported, 12)
            self.assertEqual(result.counts.verified_train_abstentions, 3)
            self.assertEqual(result.counts.failed, 0)
            self.assertEqual(database.loads, list(terminal))
            self.assertEqual(fake.runner_calls, [(12, 24), (12, 24)])
            self.assertEqual(fake.revalidations, 2)
            self.assertTrue(all("TEST" not in boundary for boundary in database.boundaries))

            events = [
                json.loads(line)
                for line in (parent / "output" / "events.jsonl").read_text().splitlines()
            ]
            states = [item["state"] for item in events if item["kind"] == "state"]
            self.assertEqual(states, [
                "initializing", "validating_source", "importing",
                "verifying_train", "finalizing", "complete_review_only",
            ])
            completed = [item for item in events if item["kind"] == "dataset_complete"]
            self.assertEqual(len(completed), 15)
            for item in completed:
                self.assertEqual(item["backend"], "cpu")
                self.assertEqual(item["workers"], 12)
                self.assertGreater(item["test_size"], 0)
                self.assertTrue(item["test_lengths"])
                self.assertIn("train_size", item)
                self.assertIn("train_lengths", item)

    def test_crash_boundaries_resume_without_duplicate_event_or_completed_recompute(self) -> None:
        for boundary in (
            "after_worker_result", "after_record_commit",
            "after_event_append", "after_progress_replace",
        ):
            with self.subTest(boundary=boundary), TemporaryDirectory() as directory:
                datasets = self.datasets(2)
                fake = _FakeRuntime(self.classifications(datasets, imported=1))
                database = _FakeDatabase(self.trains((datasets[1],)))
                reached = False

                def crash(name: str, dataset: str | None) -> None:
                    nonlocal reached
                    if not reached and name == boundary and dataset == datasets[0]:
                        reached = True
                        raise RuntimeError("injected crash")

                parent = Path(directory)
                with self.assertRaisesRegex(RuntimeError, "injected crash"):
                    run_exp426_new(
                        parent / "output", parent / "checkpoint",
                        self.provenance(datasets), self.checkpoint_context(datasets),
                        object(), database, object(),
                        _dependencies=fake.dependencies(), _hook=crash,
                    )
                interrupted_progress = json.loads(
                    (parent / "output" / "progress.json").read_text()
                )
                self.assertNotEqual(interrupted_progress["state"], "failed")
                self.assertEqual(interrupted_progress["counts"]["error"], 0)
                before_resume_calls = fake.worker_datasets.count(datasets[0])

                result = run_exp426_resume(
                    parent / "output", parent / "checkpoint",
                    self.provenance(datasets), self.checkpoint_context(datasets),
                    object(), database, object(), _dependencies=fake.dependencies(),
                )

                self.assertEqual(result.counts.completed, 2)
                after_resume_calls = fake.worker_datasets.count(datasets[0])
                expected = 2 if boundary == "after_worker_result" else 1
                self.assertEqual(after_resume_calls, expected)
                events = [
                    json.loads(line)
                    for line in (parent / "output" / "events.jsonl").read_text().splitlines()
                ]
                dataset_events = [
                    item for item in events
                    if item["kind"] == "dataset_complete" and item["dataset"] == datasets[0]
                ]
                self.assertEqual(len(dataset_events), 1)
                self.assertGreaterEqual(before_resume_calls, 1)

    def test_keyboard_interrupt_resumes_with_zero_errors_and_can_publish(self) -> None:
        datasets = self.datasets(2)
        fake = _FakeRuntime(self.classifications(datasets, imported=1))
        database = _FakeDatabase(self.trains((datasets[1],)))
        interrupted = False

        def interrupt(name: str, _dataset: str | None) -> None:
            nonlocal interrupted
            if not interrupted and name == "after_worker_result":
                interrupted = True
                raise KeyboardInterrupt("operator interruption")

        with TemporaryDirectory() as directory:
            parent = Path(directory)
            output, checkpoint = parent / "output", parent / "checkpoint"
            provenance = self.provenance(datasets)
            context = self.checkpoint_context(datasets)
            with self.assertRaisesRegex(KeyboardInterrupt, "operator interruption"):
                run_exp426_new(
                    output, checkpoint, provenance, context,
                    object(), database, object(),
                    _dependencies=fake.dependencies(), _hook=interrupt,
                )
            progress = json.loads((output / "progress.json").read_text())
            self.assertEqual(progress["state"], "interrupted")
            self.assertEqual(progress["counts"]["error"], 0)

            result = run_exp426_resume(
                output, checkpoint, provenance, context,
                object(), database, object(), _dependencies=fake.dependencies(),
            )
            self.assertEqual(result.state, "complete_review_only")
            final_progress = json.loads((output / "progress.json").read_text())
            self.assertEqual(final_progress["counts"]["error"], 0)

            from experiments.exp426_verified_baseline.reporting_antigravity import (
                _PublicationExpectations,
                publish_exp426,
            )

            publication = publish_exp426(
                output, checkpoint, provenance, context,
                _expectations=_PublicationExpectations(2, 1, 1),
            )
            self.assertEqual(publication.manifest["failed_datasets"], 0)

    def test_failed_runtime_root_is_immutable_and_cannot_resume(self) -> None:
        datasets = self.datasets(1)
        fake = _FakeRuntime(self.classifications(datasets, imported=1))
        base = fake.dependencies()

        def fail_workers(_tasks, *, workers: int, prefetch: int):
            self.assertEqual((workers, prefetch), (12, 24))
            raise ValueError("scientific worker failure")
            yield  # pragma: no cover - keep this callable an iterator

        dependencies = replace(base, worker_runner=fail_workers)
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            output, checkpoint = parent / "output", parent / "checkpoint"
            provenance = self.provenance(datasets)
            context = self.checkpoint_context(datasets)
            with self.assertRaisesRegex(ValueError, "scientific worker failure"):
                run_exp426_new(
                    output, checkpoint, provenance, context,
                    object(), _FakeDatabase({}), object(),
                    _dependencies=dependencies,
                )
            progress = json.loads((output / "progress.json").read_text())
            self.assertEqual(progress["state"], "failed")
            self.assertEqual(progress["counts"]["error"], 1)

            with self.assertRaisesRegex(ValueError, "failed runtime root"):
                run_exp426_resume(
                    output, checkpoint, provenance, context,
                    object(), _FakeDatabase({}), object(),
                    _dependencies=fake.dependencies(),
                )

    def test_inspect_is_read_only_and_progress_contains_required_monitoring_fields(self) -> None:
        datasets = self.datasets(2)
        fake = _FakeRuntime(self.classifications(datasets, imported=1))
        database = _FakeDatabase(self.trains((datasets[1],)))
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "output"
            checkpoint = parent / "checkpoint"
            run_exp426_new(
                output, checkpoint, self.provenance(datasets),
                self.checkpoint_context(datasets), object(), database, object(),
                _dependencies=fake.dependencies(),
            )
            before = {
                path.relative_to(parent): (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
                for path in parent.rglob("*") if path.is_file()
            }
            inspection = inspect_exp426(output, checkpoint, expected_run_identity="exp426-runtime")
            after = {
                path.relative_to(parent): (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
                for path in parent.rglob("*") if path.is_file()
            }
            self.assertEqual(before, after)
            self.assertEqual(inspection.progress["state"], "complete_review_only")
            required = {
                "pid", "current_dataset", "dataset_metadata", "backend", "workers",
                "counts", "throughput_datasets_per_second", "elapsed_seconds",
                "eta_seconds", "checkpoint", "latest_error",
            }
            self.assertTrue(required.issubset(inspection.progress))
            self.assertIsNone(inspection.progress["current_dataset"])
            text = (output / "progress.json").read_text()
            for forbidden in ("test_label", "ground_truth", "actual_anomaly", "f1", "precision", "recall"):
                self.assertNotIn(forbidden, text.lower())

    def test_complete_resume_returns_without_source_database_worker_or_file_writes(self) -> None:
        datasets = self.datasets(2)
        fake = _FakeRuntime(self.classifications(datasets, imported=1))
        database = _FakeDatabase(self.trains((datasets[1],)))
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "output"
            checkpoint = parent / "checkpoint"
            run_exp426_new(
                output, checkpoint, self.provenance(datasets),
                self.checkpoint_context(datasets), object(), database, object(),
                _dependencies=fake.dependencies(),
            )
            before_files = {
                path.relative_to(parent): (
                    path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns,
                )
                for path in parent.rglob("*") if path.is_file()
            }
            before_activity = (
                tuple(fake.runner_calls), tuple(fake.worker_datasets),
                fake.revalidations, tuple(database.loads), tuple(database.boundaries),
            )

            result = run_exp426_resume(
                output, checkpoint, self.provenance(datasets),
                self.checkpoint_context(datasets), object(), database, object(),
                _dependencies=fake.dependencies(),
            )

            after_files = {
                path.relative_to(parent): (
                    path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns,
                )
                for path in parent.rglob("*") if path.is_file()
            }
            after_activity = (
                tuple(fake.runner_calls), tuple(fake.worker_datasets),
                fake.revalidations, tuple(database.loads), tuple(database.boundaries),
            )
            self.assertEqual(result.state, "complete_review_only")
            self.assertEqual(before_files, after_files)
            self.assertEqual(before_activity, after_activity)

    def test_heartbeat_updates_progress_and_stops_before_terminal_publication(self) -> None:
        datasets = self.datasets(1)
        fake = _FakeRuntime(self.classifications(datasets, imported=1))
        base = fake.dependencies()

        def slow_runner(tasks, *, workers: int, prefetch: int):
            for result in base.worker_runner(tasks, workers=workers, prefetch=prefetch):
                time.sleep(0.05)
                yield result

        dependencies = replace(
            base, worker_runner=slow_runner, heartbeat_seconds=0.01,
        )
        observed_elapsed: list[float] = []
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "output"

            def observe(name: str, _dataset: str | None) -> None:
                if name == "after_worker_result":
                    progress = json.loads((output / "progress.json").read_text())
                    observed_elapsed.append(progress["elapsed_seconds"])

            run_exp426_new(
                output, parent / "checkpoint", self.provenance(datasets),
                self.checkpoint_context(datasets), object(), _FakeDatabase({}), object(),
                _dependencies=dependencies, _hook=observe,
            )
            self.assertTrue(observed_elapsed)
            self.assertGreater(observed_elapsed[0], 0.01)
            progress_before = (output / "progress.json").read_bytes()
            modified_before = (output / "progress.json").stat().st_mtime_ns
            time.sleep(0.03)
            self.assertEqual(progress_before, (output / "progress.json").read_bytes())
            self.assertEqual(modified_before, (output / "progress.json").stat().st_mtime_ns)
            pid_payload = json.loads((output / "run.pid").read_text())
            self.assertFalse(pid_payload["active"])

    def test_resume_rejects_a_matching_live_pid(self) -> None:
        datasets = self.datasets(1)
        fake = _FakeRuntime(self.classifications(datasets, imported=1))
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "output"
            checkpoint = parent / "checkpoint"

            def crash(name: str, _dataset: str | None) -> None:
                if name == "after_worker_result":
                    raise RuntimeError("leave an incomplete run")

            with self.assertRaisesRegex(RuntimeError, "leave an incomplete run"):
                run_exp426_new(
                    output, checkpoint, self.provenance(datasets),
                    self.checkpoint_context(datasets), object(), _FakeDatabase({}), object(),
                    _dependencies=fake.dependencies(), _hook=crash,
                )
            pid_payload = json.loads((output / "run.pid").read_text())
            pid_payload.update({"active": True, "pid": os.getpid()})
            (output / "run.pid").write_text(
                json.dumps(pid_payload, sort_keys=True, separators=(",", ":")),
            )
            dependencies = replace(
                fake.dependencies(), process_is_live=lambda _pid: True,
            )
            with self.assertRaisesRegex(ValueError, "still active"):
                run_exp426_resume(
                    output, checkpoint, self.provenance(datasets),
                    self.checkpoint_context(datasets), object(), _FakeDatabase({}), object(),
                    _dependencies=dependencies,
                )


if __name__ == "__main__":
    unittest.main()
