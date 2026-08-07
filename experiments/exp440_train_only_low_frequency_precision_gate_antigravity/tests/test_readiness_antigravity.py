from __future__ import annotations
import unittest
from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.readiness_antigravity import validate_clean_head_antigravity
class ReadinessTests(unittest.TestCase):
 def test_clean_head_requires_exact_commit_and_empty_status(self):
  self.assertEqual(validate_clean_head_antigravity('a'*40,'a'*40,''),'a'*40)
  for head,expected,status in [('b'*40,'a'*40,''),('a'*40,'a'*40,' M x')]:
   with self.assertRaises(ValueError):validate_clean_head_antigravity(head,expected,status)
if __name__=='__main__':unittest.main()
