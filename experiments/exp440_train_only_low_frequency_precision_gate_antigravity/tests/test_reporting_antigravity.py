from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.reporting_antigravity import publish_payloads_antigravity,inspect_publication_antigravity
class ReportingTests(unittest.TestCase):
 def test_publication_is_no_clobber_hash_bound_and_label_free(self):
  with TemporaryDirectory() as tmp:
   root=Path(tmp)/'publication_antigravity';frozen=publish_payloads_antigravity(root,datasets=('A','B'),candidates=({'dataset':'A','instance_index':2,'global_rank':1},),backend={'backend':'cpu'},reviewed_commit='a'*40)
   self.assertEqual(inspect_publication_antigravity(root).prediction_sha256,frozen.prediction_sha256)
   prediction=(root/'prediction_antigravity.jsonl').read_text();self.assertNotIn('"test_label"',prediction);self.assertIn('"contains_test_labels":false',prediction)
   with self.assertRaises(FileExistsError):publish_payloads_antigravity(root,datasets=('A','B'),candidates=(),backend={},reviewed_commit='a'*40)
if __name__=='__main__':unittest.main()
