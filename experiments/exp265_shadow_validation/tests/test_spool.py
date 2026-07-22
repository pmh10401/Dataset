from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.exp265_shadow_validation.contracts import validate_run_input
from experiments.exp265_shadow_validation.routing import route_shadow_run
from experiments.exp265_shadow_validation.spool import ShadowSpool, SpoolConflictError
from experiments.exp265_shadow_validation.tests.test_contracts import valid_run_input


def payload(run_id: str = "run-001") -> dict[str, object]:
    raw = valid_run_input()
    raw["run_id"] = run_id
    return route_shadow_run(validate_run_input(raw))


class SpoolTest(unittest.TestCase):
    def test_enqueue_is_idempotent_for_same_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spool = ShadowSpool(Path(directory) / "spool.sqlite")
            spool.enqueue(payload())
            spool.enqueue(payload())
            self.assertEqual([row.run_id for row in spool.pending(10)], ["run-001"])

    def test_conflicting_prediction_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spool = ShadowSpool(Path(directory) / "spool.sqlite")
            spool.enqueue(payload())
            changed = payload()
            changed["prediction_sha256"] = "b" * 64
            with self.assertRaises(SpoolConflictError):
                spool.enqueue(changed)

    def test_delivery_state_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spool = ShadowSpool(Path(directory) / "spool.sqlite")
            spool.enqueue(payload())
            spool.mark_delivered("run-001", {"ok": True, "receipt": "r1"})
            self.assertEqual(spool.pending(10), [])
            delivered = spool.get("run-001")
            self.assertEqual(delivered.state, "delivered")
            self.assertEqual(delivered.receipt["receipt"], "r1")

    def test_retryable_failure_keeps_row_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spool = ShadowSpool(Path(directory) / "spool.sqlite")
            spool.enqueue(payload())
            spool.mark_failure("run-001", "offline", permanent=False)
            row = spool.get("run-001")
            self.assertEqual(row.state, "pending")
            self.assertEqual(row.attempt_count, 1)


if __name__ == "__main__":
    unittest.main()
