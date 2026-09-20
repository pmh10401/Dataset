from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.checkpoint_antigravity import Exp440CheckpointStoreAntigravity
from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.contracts_antigravity import Exp440DatasetResultAntigravity

class CheckpointTests(unittest.TestCase):
 def test_ordered_prefix_no_clobber_and_reopen(self):
  with TemporaryDirectory() as tmp:
   root=Path(tmp)/'checkpoint_antigravity';store=Exp440CheckpointStoreAntigravity.create(root,('A','B'),'c'*40)
   with self.assertRaises(ValueError):store.commit(1,Exp440DatasetResultAntigravity('B','scored',(),(),0,1))
   store.commit(0,Exp440DatasetResultAntigravity('A','scored',(),(),0,1));self.assertEqual(store.completed,1)
   self.assertEqual(Exp440CheckpointStoreAntigravity.open_read_only(root).completed,1)
   with self.assertRaises(FileExistsError):Exp440CheckpointStoreAntigravity.create(root,('A','B'),'c'*40)
 def test_tampered_record_fails_closed(self):
  with TemporaryDirectory() as tmp:
   root=Path(tmp)/'checkpoint_antigravity';s=Exp440CheckpointStoreAntigravity.create(root,('A',),'c'*40);s.commit(0,Exp440DatasetResultAntigravity('A','scored',(),(),0,1));next((root/'records_antigravity').iterdir()).write_text('{}')
   with self.assertRaises(ValueError):Exp440CheckpointStoreAntigravity.open_read_only(root)
if __name__=='__main__':unittest.main()
