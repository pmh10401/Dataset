"""RED/GREEN tests for the isolated Exp444 environment contract."""
from __future__ import annotations

import unittest

from experiments.exp444_train_only_stability_precision_gate_antigravity.environment_antigravity import (
    EnvironmentEvidenceAntigravity,
    verify_environment_antigravity,
)
from experiments.exp444_train_only_stability_precision_gate_antigravity.config_antigravity import FROZEN_CONFIG, inner_thread_environment_names_antigravity


class Exp444EnvironmentAntigravityTests(unittest.TestCase):
    def test_environment_evidence_api_is_present(self) -> None:
        self.assertTrue(callable(verify_environment_antigravity))
        self.assertTrue(hasattr(EnvironmentEvidenceAntigravity, "from_payload"))

    def test_environment_evidence_round_trip_and_seven_thread_controls(self) -> None:
        evidence = EnvironmentEvidenceAntigravity(
            "3.14.6", "synthetic-platform", "arm64", 12, 11, 1,
            {"mlx": "0.32.0", "numba": "0.66.0", "numpy": "2.4.6", "pip": "26.1.2"},
            FROZEN_CONFIG.environment_lock_sha256,
            {name: "1" for name in inner_thread_environment_names_antigravity()},
            (), FROZEN_CONFIG.acceleration_decision,
            "synthetic pre-run parity gate evidence", None, "0.32.0", "0.66.0", None, None,
        )
        self.assertEqual(EnvironmentEvidenceAntigravity.from_payload(evidence.to_payload()), evidence)
        with self.assertRaises(ValueError):
            EnvironmentEvidenceAntigravity(
                "3.14.6", "synthetic-platform", "arm64", 12, 11, 1,
                {"mlx": "0.32.0", "numba": "0.66.0", "numpy": "2.4.6", "pip": "26.1.2"},
                FROZEN_CONFIG.environment_lock_sha256,
                {name: ("2" if name == "NUMBA_NUM_THREADS" else "1") for name in inner_thread_environment_names_antigravity()},
                (), FROZEN_CONFIG.acceleration_decision,
                "synthetic pre-run parity gate evidence", None, "0.32.0", "0.66.0", None, None,
            )


if __name__ == "__main__":
    unittest.main()
