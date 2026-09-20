import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp411_near_repeat_qualification.backends import (
    assess_backend,
    pairwise_rms,
    probe_backends,
)


class BackendTests(unittest.TestCase):
    def test_numba_cpu_matches_numpy(self) -> None:
        fit = np.asarray([[0.0, 0.0], [3.0, 4.0]])
        query = np.asarray([[0.0, 0.0], [4.0, 4.0]])
        np.testing.assert_allclose(
            pairwise_rms(fit, query, backend="numba_cpu"),
            pairwise_rms(fit, query, backend="numpy_cpu"),
            atol=1e-6,
            rtol=0.0,
        )

    def test_torch_cpu_matches_numpy(self) -> None:
        fit = np.asarray([[0.0, 0.0], [3.0, 4.0]])
        query = np.asarray([[0.0, 0.0], [4.0, 4.0]])
        np.testing.assert_allclose(
            pairwise_rms(fit, query, backend="torch_cpu"),
            pairwise_rms(fit, query, backend="numpy_cpu"),
            atol=1e-6,
            rtol=0.0,
        )

    def test_mps_device_failure_is_recorded_not_raised(self) -> None:
        with patch(
            "experiments.exp411_near_repeat_qualification.backends._probe_torch_mps",
            side_effect=RuntimeError("MPS device unavailable"),
        ):
            result = probe_backends()
        self.assertFalse(result["torch_mps"].available)
        self.assertIn("MPS device", result["torch_mps"].reason)
        self.assertEqual(result["mlx_status"], "excluded_by_user_decision")

    def test_backend_cannot_be_canonical_below_speed_gate(self) -> None:
        result = assess_backend(
            name="torch_cpu",
            max_abs_error=0.0,
            tag_identity=True,
            selected_quantile_identity=True,
            cpu_seconds=1.0,
            backend_seconds=0.95,
        )
        self.assertFalse(result.canonical_eligible)
        self.assertAlmostEqual(result.speedup, 1.0 / 0.95)

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown"):
            pairwise_rms(
                np.asarray([[0.0, 0.0]]),
                np.asarray([[0.0, 0.0]]),
                backend="unknown",
            )


if __name__ == "__main__":
    unittest.main()
