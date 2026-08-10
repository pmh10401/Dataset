"""RED/GREEN contract tests for the Exp444 acceleration gate."""
from __future__ import annotations

from dataclasses import dataclass
import math
import unittest


@dataclass(frozen=True)
class _Run:
    backend: str
    seconds: float
    lanes: tuple[tuple[str, int, str], ...]
    max_diff: float = 0.0
    finite: bool = True
    synchronized: bool = True
    masks_match: bool = True
    candidate_presence_match: bool = True
    survival_counts_match: bool = True
    lane_coordinates_match: bool = True
    available: bool = True
    device: str | None = "Apple GPU"
    version: str | None = "0.32.0"
    compile_warmup_included: bool = True
    materialization_included: bool = True
    process_count: int = 1


def _run(backend: str, *, seconds: float, lanes: tuple[tuple[str, int, str], ...], **kwargs: object) -> _Run:
    return _Run(backend, seconds, lanes, **kwargs)


class BackendContractAntigravityTests(unittest.TestCase):
    def test_backend_requires_identity_and_speed_gate_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.backend_antigravity import (
            choose_backend_antigravity,
        )

        decision = choose_backend_antigravity(
            cpu=_run("cpu", seconds=2.0, lanes=(("A", 1, "stable_review"),)),
            mlx=_run("mlx", seconds=2.0 / 1.1, lanes=(("A", 1, "stable_review"),), max_diff=1e-6),
        )
        self.assertEqual(decision.selected_backend, "mlx")

    def test_fast_backend_with_lane_difference_falls_back_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.backend_antigravity import (
            choose_backend_antigravity,
        )

        decision = choose_backend_antigravity(
            cpu=_run("cpu", seconds=2.0, lanes=(("A", 1, "stable_review"),)),
            mlx=_run("mlx", seconds=1.0, lanes=(("A", 1, "unstable_review"),)),
        )
        self.assertEqual(decision.selected_backend, "cpu")
        self.assertIn("lane", decision.fallback_reason or "")

    def test_exact_speed_boundary_is_inclusive_and_below_is_not_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.backend_antigravity import (
            choose_backend_antigravity,
        )

        lanes = (("A", 1, "stable_review"),)
        exact = choose_backend_antigravity(
            cpu=_run("cpu", seconds=1.1, lanes=lanes),
            numba=_run("numba", seconds=1.0, lanes=lanes),
        )
        below = choose_backend_antigravity(
            cpu=_run("cpu", seconds=1.099, lanes=lanes),
            numba=_run("numba", seconds=1.0, lanes=lanes),
        )
        self.assertEqual(exact.selected_backend, "numba")
        self.assertEqual(below.selected_backend, "cpu")

    def test_nonfinite_or_unsynchronized_run_falls_back_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.backend_antigravity import (
            choose_backend_antigravity,
        )

        lanes = (("A", 1, "stable_review"),)
        for bad in (
            _run("mlx", seconds=0.1, lanes=lanes, finite=False),
            _run("mlx", seconds=0.1, lanes=lanes, synchronized=False),
            _run("mlx", seconds=0.1, lanes=lanes, max_diff=1.1e-5),
            _run("mlx", seconds=0.1, lanes=lanes, compile_warmup_included=False),
            _run("mlx", seconds=0.1, lanes=lanes, materialization_included=False),
            _run("mlx", seconds=0.1, lanes=lanes, process_count=2),
        ):
            decision = choose_backend_antigravity(
                cpu=_run("cpu", seconds=1.0, lanes=lanes),
                mlx=bad,
            )
            self.assertEqual(decision.selected_backend, "cpu")

    def test_distance_backends_match_float64_reference_and_enforce_k3_antigravity(self) -> None:
        import numpy as np
        from experiments.exp444_train_only_stability_precision_gate_antigravity.backend_antigravity import (
            score_distance_cpu_antigravity,
            score_distance_numba_antigravity,
        )

        query = np.asarray([0.0, 1.0, 2.0, 3.0], dtype="<f4")
        references = np.asarray(
            [[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [9.0, 9.0, 9.0, 9.0]],
            dtype="<f4",
        )
        expected = score_distance_cpu_antigravity(query, references, 3)
        self.assertTrue(math.isfinite(expected))
        self.assertAlmostEqual(score_distance_numba_antigravity(query, references, 3), expected, delta=1e-12)
        with self.assertRaises(ValueError):
            score_distance_numba_antigravity(query, references, 2)

    def test_run_payload_records_sync_transfer_and_device_provenance_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.backend_antigravity import (
            BackendRunAntigravity,
        )

        run = BackendRunAntigravity(
            backend="mlx",
            seconds=1.0,
            lanes=(("A", 1, "stable_review"),),
            max_diff=0.0,
            available=True,
            synchronized=True,
            device="Apple GPU",
            version="0.32.0",
        )
        payload = run.to_payload()
        self.assertTrue(payload["synchronization_included"])
        self.assertTrue(payload["transfer_included"])
        self.assertEqual(payload["device"], "Apple GPU")


if __name__ == "__main__":
    unittest.main()
