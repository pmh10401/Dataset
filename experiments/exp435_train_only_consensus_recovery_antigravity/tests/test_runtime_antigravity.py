"""TDD checks for deterministic Exp435 serial/parallel runtime scheduling."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

from experiments.exp435_train_only_consensus_recovery_antigravity.contracts_antigravity import (
    Exp435DatasetResult,
    InstanceScoreEvidence,
)
from experiments.exp435_train_only_consensus_recovery_antigravity.runtime_antigravity import (
    run_exp435_new,
)


class _Session:
    def __init__(self, datasets: tuple[str, ...]):
        self.context = {"datasets": list(datasets), "test_sizes": {name: 2 for name in datasets}}


def _score(train: object, test: object, lanes: object, backend: object) -> Exp435DatasetResult:
    dataset = getattr(train, "dataset", None) or getattr(test, "dataset", None)
    if dataset is None and isinstance(train, dict):
        dataset = train["dataset"]
    evidence = InstanceScoreEvidence.fixture(dataset=str(dataset), instance_index=0)
    # Deliberately vary completion latency; runtime must still commit name order.
    if str(dataset).endswith("B"):
        time.sleep(0.02)
    return Exp435DatasetResult(dataset=str(dataset), status="scored", test_size=2, evidence=(evidence,))


class RuntimeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
