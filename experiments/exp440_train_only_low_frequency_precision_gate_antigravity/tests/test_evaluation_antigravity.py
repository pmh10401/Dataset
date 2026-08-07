from __future__ import annotations
import unittest
from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.evaluation_antigravity import decide_success_antigravity
class EvaluationTests(unittest.TestCase):
 def test_every_preregistered_gate_is_required(self):
  facts={'candidate_count':100,'rescued_datasets':10,'added_false_positives':30,'candidate_precision':.35,'micro_f1_delta':.01,'macro_f1_delta':.01,'micro_ci_lower':.001,'macro_ci_lower':.001,'maximum_strong_dataset_loss':.2}
  self.assertTrue(decide_success_antigravity(facts)['winner_selected'])
  for key in facts:
   bad=dict(facts);bad[key]={'candidate_count':101,'rescued_datasets':9,'added_false_positives':31,'candidate_precision':.34,'micro_f1_delta':0,'macro_f1_delta':0,'micro_ci_lower':0,'macro_ci_lower':0,'maximum_strong_dataset_loss':.201}[key];self.assertFalse(decide_success_antigravity(bad)['winner_selected'],key)
if __name__=='__main__':unittest.main()
