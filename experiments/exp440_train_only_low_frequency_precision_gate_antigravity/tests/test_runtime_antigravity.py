from __future__ import annotations
from hashlib import sha256
import time,unittest
import numpy as np
from experiments.exp439_scale_aware_reconstruction_repair_antigravity.contracts_antigravity import RawValueSeries
from experiments.exp439_scale_aware_reconstruction_repair_antigravity.inputs_antigravity import Exp439RawDataset
from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.runtime_antigravity import CpuDatasetTaskAntigravity,ordered_parallel_map_antigravity,score_dataset_task_antigravity
class RuntimeTests(unittest.TestCase):
 def test_parallel_completion_is_committed_in_input_order(self):
  values=list(range(22))
  def work(x):time.sleep((21-x)*0.0001);return x*x
  self.assertEqual(ordered_parallel_map_antigravity(values,work,workers=11),tuple(x*x for x in values))
 def test_worker_override_is_rejected(self):
  with self.assertRaises(ValueError):ordered_parallel_map_antigravity([1],lambda x:x,workers=1)
 def test_dataset_worker_scores_without_labels(self):
  x=np.arange(64,dtype=np.float32)
  def row(i,values,split):
   raw=np.asarray(values,dtype='<f4').tobytes();return RawValueSeries('A',i,split,np.frombuffer(raw,dtype='<f4'),sha256(raw).hexdigest())
  train=Exp439RawDataset('A','TRAIN',tuple(row(i,np.sin(x/4)+i*.01,'TRAIN') for i in range(5)))
  test=Exp439RawDataset('A','TEST',(row(0,10*np.sin(x/4),'TEST'),row(1,[float('nan')]*64,'TEST')))
  result=score_dataset_task_antigravity(CpuDatasetTaskAntigravity('A',train,test));self.assertEqual(result.test_row_count,2);self.assertEqual(result.abstention_indices,(1,))
if __name__=='__main__':unittest.main()
