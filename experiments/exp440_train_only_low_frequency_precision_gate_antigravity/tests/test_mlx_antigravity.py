from __future__ import annotations
import numpy as np,unittest
from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.mlx_antigravity import benchmark_mlx_antigravity
class MlxTests(unittest.TestCase):
 def test_benchmark_is_truthful_and_never_required_for_correctness(self):
  q=np.arange(64,dtype=np.float32);r=np.stack([q+i for i in range(8)])
  evidence=benchmark_mlx_antigravity(q,r,repeats=2);self.assertEqual(evidence.backend,'mlx');self.assertTrue(evidence.available or evidence.fallback_reason is not None)
if __name__=='__main__':unittest.main()
