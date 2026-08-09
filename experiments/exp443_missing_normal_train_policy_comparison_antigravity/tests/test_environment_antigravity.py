"""RED tests for the exact isolated Exp443 environment evidence."""
from __future__ import annotations

import hashlib
from pathlib import Path
import unittest

from experiments.exp443_missing_normal_train_policy_comparison_antigravity.config_antigravity import FROZEN_CONFIG
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.environment_antigravity import (
    EnvironmentEvidenceAntigravity,
    verify_environment_antigravity,
)


class Exp443EnvironmentAntigravityTests(unittest.TestCase):
    def test_environment_evidence_records_exact_versions_threads_and_cpu_backend(self) -> None:
        evidence = verify_environment_antigravity()
        self.assertIsInstance(evidence, EnvironmentEvidenceAntigravity)
        self.assertEqual(evidence.python_version, "3.14.6")
        self.assertEqual(evidence.logical_cpu_count, 12)
        self.assertEqual(evidence.outer_workers, 11)
        self.assertEqual(evidence.inner_threads, 1)
        self.assertEqual(evidence.package_versions["aeon"], "1.5.0")
        self.assertEqual(evidence.package_versions["numpy"], "2.3.5")
        self.assertEqual(evidence.package_versions["scikit-learn"], "1.8.0")
        self.assertEqual(evidence.acceleration_decision, "inapplicable_official_cpu_transformer")
        self.assertEqual(evidence.to_payload()["lock_sha256"], FROZEN_CONFIG.environment_lock_sha256)
        self.assertEqual(EnvironmentEvidenceAntigravity.from_payload(evidence.to_payload()), evidence)

    def test_environment_evidence_rejects_wrong_lock_or_worker_contract(self) -> None:
        evidence = verify_environment_antigravity()
        for field_name, value in (("lock_sha256", "0" * 64), ("outer_workers", 1), ("inner_threads", 2)):
            payload = evidence.to_payload()
            payload[field_name] = value
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError):
                    EnvironmentEvidenceAntigravity.from_payload(payload)


if __name__ == "__main__":
    unittest.main()
