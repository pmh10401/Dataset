from __future__ import annotations
import time,unittest
from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.runtime_antigravity import ordered_parallel_map_antigravity
class RuntimeTests(unittest.TestCase):
 def test_parallel_completion_is_committed_in_input_order(self):
  values=list(range(22))
  def work(x):time.sleep((21-x)*0.0001);return x*x
  self.assertEqual(ordered_parallel_map_antigravity(values,work,workers=11),tuple(x*x for x in values))
 def test_worker_override_is_rejected(self):
  with self.assertRaises(ValueError):ordered_parallel_map_antigravity([1],lambda x:x,workers=1)
if __name__=='__main__':unittest.main()
