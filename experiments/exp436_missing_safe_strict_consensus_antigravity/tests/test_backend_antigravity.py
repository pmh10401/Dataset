"""Contract tests for neutral Exp436 dense-scoring backend selection."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.exp436_missing_safe_strict_consensus_antigravity import mlx_antigravity as mlx_module
from experiments.exp436_missing_safe_strict_consensus_antigravity import mps_antigravity as mps_module
from experiments.exp436_missing_safe_strict_consensus_antigravity.backend_antigravity import (
    score_with_backend,
    select_backend,
)
from experiments.exp436_missing_safe_strict_consensus_antigravity.contracts_antigravity import (
    BackendBatchEvidence,
    BackendBenchmarkEvidence,
)
from experiments.exp436_missing_safe_strict_consensus_antigravity.mlx_antigravity import MlxFixture
from experiments.exp436_missing_safe_strict_consensus_antigravity.mps_antigravity import MpsFixture


def _available(backend: str, median: float) -> BackendBenchmarkEvidence:
    """Build a fully passing evidence record through the public test helper."""
    batches = tuple(
        BackendBatchEvidence(
            backend, size, "completed", 2, ((median * 2.0, median),) * 5,
            "a" * 64, "b" * 64, True, True, 0.0, 0.0, median * 2.0, median,
            2.0, True, True, True, "passed",
        )
        for size in (16, 32, 64, 128, 256)
    )
    return BackendBenchmarkEvidence(
        backend, True, batches, "python", "numpy", "gpu:0" if backend == "mlx" else "mps", "info", True,
        "paired-synchronized-v1", 16, median, None,
        mlx_version="mlx" if backend == "mlx" else None,
        mlx_metal_version="metal" if backend == "mlx" else None,
        pytorch_version="torch" if backend == "mps" else None,
    )


class BackendSelectorTests(unittest.TestCase):
    def test_mlx_and_mps_fixture_digest_algorithms_are_symmetric(self) -> None:
        query = np.asarray([0.1, 0.1], dtype=np.float32)
        references = np.asarray([[0., 0.], [1., 1.], [2., 2.]], dtype=np.float32)
        mlx_fixtures = (MlxFixture(query, references, 0.5),)
        mps_fixtures = (MpsFixture(query, references, 0.5),)
        self.assertEqual(mlx_module._digest_fixtures(mlx_fixtures), mps_module._digest_fixtures(mps_fixtures))
        mlx_scores, mps_scores = mlx_module._cpu_score_suite(mlx_fixtures), mps_module._cpu_score_suite(mps_fixtures)
        mlx_coordinates = mlx_module._candidate_coordinates(mlx_fixtures, mlx_scores)
        mps_coordinates = mps_module._candidate_coordinates(mps_fixtures, mps_scores)
        self.assertEqual(mlx_module._fixture_output_sha(mlx_scores, mlx_coordinates), mps_module._fixture_output_sha(mps_scores, mps_coordinates))

    def test_selector_uses_lexical_tie_break_and_cpu_workers(self) -> None:
        decision = select_backend(_available("mlx", 1.0), _available("mps", 1.0))
        self.assertEqual(decision.backend, "mlx")
        self.assertTrue(decision.adopted)
        self.assertEqual(decision.outer_worker_count, 1)

        cpu = select_backend(
            BackendBenchmarkEvidence.unavailable("mlx", "framework_unavailable"),
            BackendBenchmarkEvidence.unavailable("mps", "framework_unavailable"),
        )
        self.assertEqual(cpu.backend, "cpu")
        self.assertFalse(cpu.adopted)
        self.assertEqual(cpu.outer_worker_count, 11)

    def test_adopted_backend_requires_matching_supplied_engine_and_never_falls_back(self) -> None:
        decision = select_backend(_available("mlx", 1.0), BackendBenchmarkEvidence.unavailable("mps", "missing"))
        query = np.asarray([0.0, 0.0], dtype=np.float32)
        references = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "engine"):
            score_with_backend(decision, query, references, 3)

    def test_dispatch_validates_full_provenance_and_cpu_remains_exact_scalar(self) -> None:
        decision = select_backend(_available("mlx", 1.0), BackendBenchmarkEvidence.unavailable("mps", "missing"))
        query = np.asarray([0.0, 0.0], dtype=np.float32)
        references = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)

        class Engine:
            backend = "mlx"
            metal_available = True
            device = "gpu:0"
            device_info = "info"
            mlx_version = "mlx"
            mlx_metal_version = "metal"

            def __init__(self) -> None:
                self.batch_sizes: list[int] = []

            def score(self, query, references, k, *, batch_size):
                self.batch_sizes.append(batch_size)
                return 0.0

        engine = Engine()
        self.assertEqual(score_with_backend(decision, query, references, 3, engine=engine), 0.0)
        self.assertEqual(engine.batch_sizes, [16])
        engine.device = "wrong"
        with self.assertRaisesRegex(ValueError, "device provenance"):
            score_with_backend(decision, query, references, 3, engine=engine)
        cpu = select_backend(BackendBenchmarkEvidence.unavailable("mlx", "missing"), BackendBenchmarkEvidence.unavailable("mps", "missing"))
        self.assertEqual(score_with_backend(cpu, query, references, 3), 1.0)

    def test_dispatch_rejects_nonfrozen_k_before_any_cpu_or_device_score(self) -> None:
        query = np.asarray([0.0, 0.0], dtype=np.float32)
        references = np.asarray([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=np.float32)
        cpu = select_backend(BackendBenchmarkEvidence.unavailable("mlx", "missing"), BackendBenchmarkEvidence.unavailable("mps", "missing"))
        adopted = select_backend(_available("mlx", 1.0), BackendBenchmarkEvidence.unavailable("mps", "missing"))

        class Engine:
            backend = "mlx"
            metal_available = True
            device = "gpu:0"
            device_info = "info"
            mlx_version = "mlx"
            mlx_metal_version = "metal"

            def __init__(self) -> None:
                self.calls: list[int] = []

            def score(self, query, references, k, *, batch_size):
                self.calls.append(k)
                return 0.0

        engine = Engine()
        for decision, active in ((cpu, None), (adopted, engine)):
            for k in (1, 2, 4):
                with self.subTest(backend=decision.backend, k=k):
                    with self.assertRaisesRegex(ValueError, "frozen"):
                        score_with_backend(decision, query, references, k, engine=active)
        self.assertEqual(engine.calls, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
