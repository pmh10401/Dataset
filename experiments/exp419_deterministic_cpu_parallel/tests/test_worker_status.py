from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.exp419_deterministic_cpu_parallel.worker_status import (
    WorkerStatus,
    write_worker_status,
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
