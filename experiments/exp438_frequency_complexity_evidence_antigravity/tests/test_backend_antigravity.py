import unittest

import numpy as np

from experiments.exp438_frequency_complexity_evidence_antigravity.backend_antigravity import (
    BackendFixtureAntigravity,
    benchmark_candidate_antigravity,
    select_backend_antigravity,
)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class Exp438BackendAntigravityTests(unittest.TestCase):
    def fixture(self) -> BackendFixtureAntigravity:
        return BackendFixtureAntigravity(
            values=np.asarray([0.2, 0.8, 1.2], dtype=np.float32),
            thresholds=np.asarray([0.5, 0.5, 1.0], dtype=np.float32),
        )

    def test_benchmark_uses_two_warmups_five_pairs_and_selects_fastest_batch(self) -> None:
        clock = _Clock()
        calls = {"cpu": 0, "device": 0, "sync": 0}

        def cpu(fixture, batch_size):
            calls["cpu"] += 1
            clock.value += 2.0
            return fixture.values

        def device(fixture, batch_size):
            calls["device"] += 1
            clock.value += 0.5 if batch_size == 32 else 1.0
            return fixture.values.copy()

        def sync():
            calls["sync"] += 1

        evidence = benchmark_candidate_antigravity(
            "mlx",
            self.fixture(),
            cpu,
            device,
            batch_sizes=(16, 32),
            clock=clock,
            synchronize=sync,
        )
        self.assertEqual(calls, {"cpu": 10, "device": 14, "sync": 14})
        passing = [item for item in evidence if item.passed]
        self.assertEqual(len(passing), 2)
        decision = select_backend_antigravity(evidence)
        self.assertEqual(decision.selected_backend, "mlx")
        self.assertEqual(decision.selected_batch_size, 32)

    def test_candidate_or_numeric_mismatch_fails_and_cpu_falls_back(self) -> None:
        clock = _Clock()

        def cpu(fixture, batch_size):
            clock.value += 2.0
            return fixture.values

        def wrong(fixture, batch_size):
            clock.value += 1.0
            return np.asarray([0.2, 0.1, 1.2], dtype=np.float32)

        evidence = benchmark_candidate_antigravity(
            "mps",
            self.fixture(),
            cpu,
            wrong,
            batch_sizes=(16,),
            clock=clock,
            synchronize=lambda: None,
        )
        self.assertFalse(evidence[0].passed)
        self.assertFalse(evidence[0].candidate_parity)
        decision = select_backend_antigravity(evidence)
        self.assertEqual(decision.selected_backend, "cpu")
        self.assertIsNone(decision.selected_batch_size)
        self.assertIn("no candidate passed", decision.fallback_reason)

    def test_cross_backend_exact_tie_is_lexical(self) -> None:
        evidence = []
        for backend in ("mps", "mlx"):
            clock = _Clock()

            def cpu(fixture, batch_size, clock=clock):
                clock.value += 2.0
                return fixture.values

            def device(fixture, batch_size, clock=clock):
                clock.value += 1.0
                return fixture.values

            evidence.extend(
                benchmark_candidate_antigravity(
                    backend,
                    self.fixture(),
                    cpu,
                    device,
                    batch_sizes=(16,),
                    clock=clock,
                    synchronize=lambda: None,
                )
            )
        decision = select_backend_antigravity(tuple(evidence))
        self.assertEqual(decision.selected_backend, "mlx")


if __name__ == "__main__":
    unittest.main()
