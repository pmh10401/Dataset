from dataclasses import replace
import unittest

from experiments.exp419_deterministic_cpu_parallel.config import FROZEN_CONFIG


class Exp419ConfigTests(unittest.TestCase):
    def test_exact_runtime_contract_is_frozen(self):
        self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp419")
        self.assertEqual(FROZEN_CONFIG.worker_candidates, (1, 2, 4, 6))
        self.assertEqual(FROZEN_CONFIG.start_method, "spawn")
        self.assertEqual(FROZEN_CONFIG.speedup_gate, 2.0)
        self.assertEqual(FROZEN_CONFIG.memory_limit_bytes, 27_487_790_694)
        self.assertEqual(FROZEN_CONFIG.torch_threads, 1)

    def test_modified_contract_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "worker_candidates"):
            replace(FROZEN_CONFIG, worker_candidates=(1, 8)).validate()
