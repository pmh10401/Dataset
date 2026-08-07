from __future__ import annotations
import numpy as np
import unittest
from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.features_antigravity import adjacent_depths_antigravity,fit_group_calibration_antigravity


class Exp440FeaturesAntigravityTests(unittest.TestCase):
    def test_adjacent_depth_boundaries(self)->None:
        self.assertEqual(adjacent_depths_antigravity(1,1),())
        self.assertEqual(adjacent_depths_antigravity(1,2),(1,2))
        self.assertEqual(adjacent_depths_antigravity(3,5),(2,3,4))
        self.assertEqual(adjacent_depths_antigravity(5,5),(4,5))

    def test_fold_local_oof_uses_only_other_rows_and_higher_quantile(self)->None:
        x=np.arange(64,dtype=np.float32);matrix=np.stack([np.sin(x/(i+2))+i*0.1 for i in range(5)]).astype(np.float32)
        calls=[]
        def spy(query,references,k):
            calls.append(references.shape[0]);delta=references.astype(np.float64)-query.astype(np.float64);dist=np.sqrt(np.mean(delta*delta,axis=1));return float(np.median(np.partition(dist,k-1)[:k]))
        group=fit_group_calibration_antigravity(matrix,distance=spy,preferred_depths=(2,2,2,2,2))
        self.assertTrue(calls);self.assertTrue(all(count in {3,4} for count in calls))
        for depth in group.depths:self.assertEqual(depth.threshold,float(np.quantile(depth.oof_scores,0.99,method='higher')))


if __name__=='__main__':unittest.main()
