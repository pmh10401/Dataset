from __future__ import annotations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.progress_antigravity import ProgressWriterAntigravity
class ProgressTests(unittest.TestCase):
 def test_atomic_progress_contains_required_observability(self):
  with TemporaryDirectory() as tmp:
   root=Path(tmp)/'output_antigravity';w=ProgressWriterAntigravity.create(root,total=1117,checkpoint=Path(tmp)/'checkpoint_antigravity',backend='cpu');w.start_dataset('A',5,7,(16,),(16,),'calibration');w.complete_dataset('A','scored',provisional=2);w.close('interrupted')
   p=json.loads((root/'progress_antigravity.json').read_text())
   for key in ('pid','current_dataset','train_count','test_count','train_lengths','test_lengths','stage','backend','completed','total','imported','recomputed','success','abstention','error','provisional_candidates','retained_candidates','throughput','elapsed_seconds','eta_seconds','checkpoint','heartbeat','latest_error'):self.assertIn(key,p)
   self.assertGreaterEqual(len((root/'events_antigravity.jsonl').read_text().splitlines()),3);self.assertTrue((root/'parent_antigravity.log').read_text())
if __name__=='__main__':unittest.main()
