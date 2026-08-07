from __future__ import annotations

import unittest

from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.config_antigravity import (
    FROZEN_CONFIG,
    cpu_worker_count_antigravity,
    frozen_config_payload_antigravity,
)


class Exp440ConfigAntigravityTests(unittest.TestCase):
    def test_frozen_scientific_literals_and_cpu_policy(self) -> None:
        self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp440")
        self.assertEqual(FROZEN_CONFIG.dataset_count, 1117)
        self.assertEqual(FROZEN_CONFIG.adjacent_offsets, (-1, 0, 1))
        self.assertEqual(FROZEN_CONFIG.minimum_consensus_depths, 2)
        self.assertEqual(FROZEN_CONFIG.outside_hard_budget, 100)
        self.assertEqual(cpu_worker_count_antigravity(12), 11)
        self.assertEqual(cpu_worker_count_antigravity(1), 1)
        self.assertTrue(FROZEN_CONFIG.review_only)
        self.assertFalse(FROZEN_CONFIG.operational_alert_eligible)

    def test_payload_is_json_safe_and_complete(self) -> None:
        payload = frozen_config_payload_antigravity()
        self.assertEqual(payload["batch_sizes"], [16, 32, 64, 128, 256])
        self.assertEqual(payload["adjacent_offsets"], [-1, 0, 1])
        self.assertEqual(set(payload), set(FROZEN_CONFIG.__dataclass_fields__))


if __name__ == "__main__":
    unittest.main()
