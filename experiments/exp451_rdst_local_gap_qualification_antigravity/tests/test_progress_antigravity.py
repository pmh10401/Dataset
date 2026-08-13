import tempfile
import unittest
from pathlib import Path
from experiments.exp451_rdst_local_gap_qualification_antigravity.progress_antigravity import PROGRESS_KEYS_ANTIGRAVITY, ProgressWriterAntigravity, select_outer_workers_antigravity


class ProgressTestsAntigravity(unittest.TestCase):
    def payload(self, completed):
        return {"pid":1,"dataset":"D_antigravity","family":"F","train_rows":5,"test_rows":3,"series_length":4,"seed":None,"fold":None,"stage":"commit","backend":"cpu","completed":completed,"total":48,"success":completed,"error":0,"reused":0,"datasets_per_second":1.0,"elapsed_seconds":1.0,"eta_seconds":47.0,"checkpoint":"checkpoint_antigravity.json","latest_error":None}
    def test_worker_policy(self):
        self.assertEqual(select_outer_workers_antigravity(12, "cpu"), 11)
        self.assertEqual(select_outer_workers_antigravity(12, "mlx"), 1)

    def test_writer_replaces_json_and_appends_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = ProgressWriterAntigravity(Path(temporary))
            writer.write_antigravity(self.payload(1))
            writer.write_antigravity(self.payload(2))
            self.assertIn('"completed":2', (Path(temporary) / "progress_antigravity.json").read_text())
            self.assertEqual(len((Path(temporary) / "progress_antigravity.txt").read_text().splitlines()), 2)

    def test_full_progress_schema_is_exact(self):
        self.assertEqual(PROGRESS_KEYS_ANTIGRAVITY, {"pid","dataset","family","train_rows","test_rows","series_length","seed","fold","stage","backend","completed","total","success","error","reused","datasets_per_second","elapsed_seconds","eta_seconds","checkpoint","latest_error"})
        payload = self.payload(1)
        with tempfile.TemporaryDirectory() as temporary:
            ProgressWriterAntigravity(Path(temporary)).write_antigravity(payload)
            import json
            self.assertEqual(set(json.loads((Path(temporary)/"progress_antigravity.json").read_text())), PROGRESS_KEYS_ANTIGRAVITY)


if __name__ == "__main__": unittest.main()
