from copy import deepcopy
import unittest

from operational.recover_exp437_checkpoint_antigravity import (
    REVIEWED_EXP437_COMMIT_ANTIGRAVITY,
    _validate_incident_antigravity,
)


def _incident_antigravity(**changes):
    evidence = {
        "source_head": REVIEWED_EXP437_COMMIT_ANTIGRAVITY,
        "source_clean": True,
        "run_identity": "exp437-metadata-roundtrip-antigravity-20260807",
        "state": "running",
        "recorded_pid": 81739,
        "pid_active": True,
        "pid_os_live": False,
        "backend": "cpu",
        "error": 0,
        "progress_completed": 938,
        "event_completed": 938,
        "checkpoint_completed": 938,
        "physical_record_files": 939,
        "total": 1117,
        "reconciliation_reasons": ["intent_with_record"],
        "label_access_count": 0,
        "test_values_opened": 0,
        "test_labels_opened": 0,
        "publication_exists": False,
        "evaluation_exists": False,
    }
    evidence.update(changes)
    return evidence


class Exp437RecoveryLauncherAntigravityTests(unittest.TestCase):
    def test_accepts_exact_stale_pid_intent_with_record_incident(self):
        evidence = _incident_antigravity()
        self.assertEqual(_validate_incident_antigravity(evidence), evidence)

    def test_rejects_any_changed_safety_boundary(self):
        cases = (
            {"source_head": "a" * 40},
            {"source_clean": False},
            {"run_identity": "wrong"},
            {"state": "failed"},
            {"recorded_pid": 1},
            {"pid_active": False},
            {"pid_os_live": True},
            {"backend": "mps"},
            {"error": 1},
            {"progress_completed": 937},
            {"event_completed": 937},
            {"checkpoint_completed": 939},
            {"physical_record_files": 938},
            {"total": 1118},
            {"reconciliation_reasons": []},
            {"label_access_count": 1},
            {"test_values_opened": 1},
            {"test_labels_opened": 1},
            {"publication_exists": True},
            {"evaluation_exists": True},
        )
        for changes in cases:
            evidence = deepcopy(_incident_antigravity())
            evidence.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                _validate_incident_antigravity(evidence)


if __name__ == "__main__":
    unittest.main()
