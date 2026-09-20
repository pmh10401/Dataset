from __future__ import annotations
import numpy as np
import unittest
from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.features_antigravity import fit_group_calibration_antigravity
from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.scoring_antigravity import score_test_row_antigravity


class Exp440ScoringAntigravityTests(unittest.TestCase):
    def test_two_depth_strict_consensus_and_empirical_rank(self)->None:
        x=np.arange(64,dtype=np.float32);matrix=np.stack([np.sin(x/4)+i*0.01 for i in range(5)]).astype(np.float32)
        calibration=fit_group_calibration_antigravity(matrix,preferred_depths=(2,2,2,2,2))
        scores,candidate=score_test_row_antigravity('A',7,(10*np.sin(x/4)).astype(np.float32),calibration)
        agreeing=tuple(s for s in scores if s.exceeds)
        if len(agreeing)>=2:
            self.assertIsNotNone(candidate);self.assertEqual(candidate.agreement_count,len(agreeing));self.assertTrue(all(0<s.empirical_rank<=1 for s in scores))
        else:self.assertIsNone(candidate)

    def test_score_equal_to_threshold_does_not_exceed(self)->None:
        x=np.arange(64,dtype=np.float32);matrix=np.stack([np.sin(x/4)+i*0.01 for i in range(5)]).astype(np.float32)
        calibration=fit_group_calibration_antigravity(matrix,preferred_depths=(2,2,2,2,2))
        scores,_=score_test_row_antigravity('A',0,matrix[0],calibration)
        self.assertTrue(all(s.exceeds==(s.score>s.threshold) for s in scores))


if __name__=='__main__':unittest.main()
