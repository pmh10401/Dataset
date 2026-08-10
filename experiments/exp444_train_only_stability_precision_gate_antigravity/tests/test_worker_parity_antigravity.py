"""Read-only one-worker/11-worker parity tests for Exp444."""
from __future__ import annotations

import unittest

from experiments.exp444_train_only_stability_precision_gate_antigravity import runtime_antigravity as runtime
from experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_runtime_antigravity import (
    _inputs_antigravity,
    _result_antigravity,
)


class Exp444WorkerParityAntigravityTests(unittest.TestCase):
    def test_one_worker_and_eleven_worker_have_identical_canonical_bytes_antigravity(self) -> None:
        inputs = _inputs_antigravity()
        serial = runtime.ordered_parallel_map_antigravity(inputs, workers=1, score_task=_result_antigravity)
        parallel = runtime.ordered_parallel_map_antigravity(inputs, workers=11, score_task=_result_antigravity)
        self.assertEqual(runtime.canonical_results_bytes_antigravity(serial), runtime.canonical_results_bytes_antigravity(parallel))
        self.assertEqual(tuple(item.candidate.coordinate for item in serial), tuple(item.candidate.coordinate for item in parallel))

    def test_production_worker_count_is_frozen_to_eleven_antigravity(self) -> None:
        self.assertEqual(runtime.production_worker_count_antigravity(), 11)


if __name__ == "__main__":
    unittest.main()
