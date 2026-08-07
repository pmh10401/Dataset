from copy import deepcopy
import unittest

from experiments.exp438_frequency_complexity_evidence_antigravity.predecessor_antigravity import (
    classify_exp437_snapshot_antigravity,
    revalidate_exp437_snapshot_antigravity,
)


EXP437_PARENT_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/"
    "exp437_metadata_roundtrip_antigravity_20260807"
)


def _running_snapshot_antigravity(**changes):
    snapshot = {
        "run_identity": "exp437-metadata-roundtrip-antigravity-20260807",
        "run_parent": EXP437_PARENT_ANTIGRAVITY,
        "state": "running",
        "pid": 43210,
        "pid_active": True,
        "pid_os_live": True,
        "backend": "cpu",
        "completed": 938,
        "checkpoint_completed": 939,
        "total": 1117,
        "error": 0,
        "heartbeat": "2026-08-07T04:50:00Z",
        "heartbeat_age_seconds": 1.0,
        "publication_exists": False,
        "evaluation_exists": False,
        "concurrency_authorized": True,
    }
    snapshot.update(changes)
    return snapshot


def _terminal_snapshot_antigravity(**changes):
    snapshot = _running_snapshot_antigravity(
        state="checkpoint_complete",
        pid_active=False,
        pid_os_live=False,
        completed=1117,
        checkpoint_completed=1117,
        heartbeat_age_seconds=0.0,
        publication_exists=True,
    )
    snapshot.update(changes)
    return snapshot


class Exp438PredecessorAntigravityTests(unittest.TestCase):
    def test_accepts_fresh_live_zero_error_cpu_snapshot(self):
        self.assertEqual(
            classify_exp437_snapshot_antigravity(_running_snapshot_antigravity()),
            "concurrent",
        )

    def test_accepts_terminal_inactive_complete_snapshot(self):
        self.assertEqual(
            classify_exp437_snapshot_antigravity(_terminal_snapshot_antigravity()),
            "terminal",
        )

    def test_rejects_each_unsafe_concurrent_condition(self):
        cases = (
            {"pid_os_live": False},
            {"heartbeat_age_seconds": 300.001},
            {"heartbeat_age_seconds": -0.001},
            {"backend": "mlx"},
            {"backend": "mps"},
            {"state": "failed"},
            {"state": "interrupted"},
            {"error": 1},
            {"completed": 940, "checkpoint_completed": 939},
            {"completed": 1117, "checkpoint_completed": 1117},
            {"publication_exists": True},
            {"evaluation_exists": True},
            {"concurrency_authorized": False},
            {"run_identity": "wrong"},
            {"run_parent": "/tmp/wrong"},
        )
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                classify_exp437_snapshot_antigravity(
                    _running_snapshot_antigravity(**changes)
                )

    def test_revalidation_accepts_forward_progress_and_recovery_pid(self):
        bound = _running_snapshot_antigravity()
        current = _running_snapshot_antigravity(
            pid=54321,
            completed=945,
            checkpoint_completed=946,
            heartbeat_age_seconds=2.0,
        )
        self.assertEqual(
            revalidate_exp437_snapshot_antigravity(bound, current), "concurrent"
        )

    def test_revalidation_accepts_terminal_transition(self):
        self.assertEqual(
            revalidate_exp437_snapshot_antigravity(
                _running_snapshot_antigravity(), _terminal_snapshot_antigravity()
            ),
            "terminal",
        )

    def test_revalidation_rejects_regression_or_changed_binding(self):
        bound = _running_snapshot_antigravity()
        cases = (
            {"completed": 937},
            {"checkpoint_completed": 938},
            {"total": 1118},
            {"backend": "mps"},
            {"run_identity": "wrong"},
            {"run_parent": "/tmp/wrong"},
        )
        for changes in cases:
            current = deepcopy(bound)
            current.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                revalidate_exp437_snapshot_antigravity(bound, current)


if __name__ == "__main__":
    unittest.main()
