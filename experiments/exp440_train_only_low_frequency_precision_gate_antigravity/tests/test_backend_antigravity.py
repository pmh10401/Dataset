from __future__ import annotations
from dataclasses import replace
import unittest
from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.backend_antigravity import AccelerationEvidenceAntigravity,select_backend_antigravity


def _e(name='mlx',speed=1.2,parity=True)->AccelerationEvidenceAntigravity:return AccelerationEvidenceAntigravity(name,True,True,parity,0.0,speed,0.2,0.1,'1.0','Apple GPU',None)
class BackendTests(unittest.TestCase):
 def test_only_finite_synchronized_exact_fast_backend_passes(self):
  self.assertTrue(_e().passed)
  for bad in (replace(_e(),candidate_parity=False),replace(_e(),max_abs_error=1e-4),replace(_e(),speedup=1.09),replace(_e(),synchronized=False),replace(_e(),version=None)):
   self.assertFalse(bad.passed)
 def test_selector_uses_fastest_pass_or_cpu_11_workers(self):
  self.assertEqual(select_backend_antigravity((_e('mlx',1.2),_e('numba',1.5))).backend,'numba')
  decision=select_backend_antigravity((AccelerationEvidenceAntigravity.unavailable('mlx','missing'),));self.assertEqual((decision.backend,decision.workers),('cpu',11))
if __name__=='__main__':unittest.main()
