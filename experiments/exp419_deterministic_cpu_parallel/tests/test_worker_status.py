from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.exp419_deterministic_cpu_parallel.worker_status import (
    WorkerStatus,
    append_runtime_log,
    write_worker_status,
)
from experiments.exp419_deterministic_cpu_parallel.contracts import (
    RuntimeLogRecord,
)


class WorkerStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def test_status_writer_publishes_safe_current_dataset_snapshot(self) -> None:
        status = WorkerStatus(
            shard_index=0,
            worker_pid=123,
            expected_datasets=2,
            completed_datasets=0,
            phase="dataset_starting",
            dataset="alpha",
            dataset_index=0,
            seed=None,
            fold=None,
            error="",
        )

        path = write_worker_status(self.root, status)

        self.assertEqual(path, self.root / "exp419_worker_status.json")
        self.assertEqual(
            json.loads(path.read_text(encoding="utf-8")),
            {
                "completed_datasets": 0,
                "dataset": "alpha",
                "dataset_index": 0,
                "error": "",
                "expected_datasets": 2,
                "fold": None,
                "phase": "dataset_starting",
                "schema": "exp419.worker-status.v1",
                "seed": None,
                "shard_index": 0,
                "worker_pid": 123,
            },
        )

    def test_status_payload_rejects_prediction_material(self) -> None:
        with self.assertRaisesRegex(ValueError, "status field"):
            WorkerStatus.from_payload(
                {
                    "completed_datasets": 0,
                    "dataset": None,
                    "dataset_index": None,
                    "error": "",
                    "expected_datasets": 1,
                    "fold": None,
                    "phase": "worker_starting",
                    "prediction": [0],
                    "schema": "exp419.worker-status.v1",
                    "seed": None,
                    "shard_index": 0,
                    "worker_pid": 123,
                }
            )

    def test_runtime_log_rejects_test_prediction_label_and_metric_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden"):
            RuntimeLogRecord.from_mapping(
                {
                    "run_id": "run-1",
                    "timestamp": "2026-07-31T00:00:00Z",
                    "pid": 123,
                    "shard_index": 0,
                    "current_dataset": None,
                    "phase": "worker_starting",
                    "seed": None,
                    "fold": None,
                    "completed_datasets": 0,
                    "elapsed_seconds": 0.0,
                    "status": "running",
                    "error": "",
                    "test_values": [1.0],
                }
            )

    def test_runtime_log_writer_appends_only_safe_progress_fields(self) -> None:
        record = RuntimeLogRecord(
            run_id="run-1",
            timestamp="2026-07-31T00:00:00Z",
            pid=123,
            shard_index=0,
            current_dataset="alpha",
            phase="preprocessing",
            seed=None,
            fold=None,
            completed_datasets=1,
            elapsed_seconds=1.25,
            status="running",
            error="",
        )

        path = append_runtime_log(self.root, record)

        self.assertEqual(path, self.root / "worker-0.runtime.jsonl")
        self.assertEqual(
            json.loads(path.read_text(encoding="utf-8")),
            record.to_payload(),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
