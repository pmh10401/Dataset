"""Full-cohort coverage, failure denominators and freeze-boundary tests."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("run_full_antigravity.py")


class FullCohortTests(unittest.TestCase):
    def module(self):
        self.assertTrue(SCRIPT.exists(), "full-cohort runner is not implemented")
        spec = importlib.util.spec_from_file_location("full_rockad_test", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_full_catalog_keeps_small_unrouted_and_large_datasets(self):
        module = self.module()
        rows = [{"dataset_name": "large", "train_count": 18000, "series_length": 4000},
                {"dataset_name": "unrouted", "train_count": 1, "method_name": None},
                {"dataset_name": "small", "train_count": 6}]
        self.assertEqual([r["dataset_name"] for r in module.full_catalog(rows)], ["large", "small", "unrouted"])
        with self.assertRaises(ValueError):
            module.full_catalog(rows + [rows[0]])

    def test_paired_summary_excludes_failure_and_uses_family_balanced_average(self):
        module = self.module()
        def row(name, family, auc, fn, fp):
            metrics = {"auc": auc, "ap": auc, "true_positives": 1,
                       "false_negatives": fn, "false_positives": fp, "true_negatives": 9}
            return {"dataset": name, "family": family, "models": {
                "service": {"status": "ok", "service_policy": metrics},
                "rockad": {"status": "ok", "service_policy": metrics}}}
        rows = [row("a1", "A", .6, 1, 0), row("a2", "A", .8, 2, 1), row("b1", "B", .9, 0, 0),
                {"dataset": "failed", "family": "C", "models": {"rockad": {"status": "error"}}}]
        result = module.paired_summary(rows, "service_policy")
        self.assertEqual(result["paired_datasets"], 3)
        self.assertEqual(result["paired_families"], 2)
        self.assertEqual(result["service"]["false_negatives"], 3)
        self.assertEqual(result["service"]["false_positives"], 1)
        self.assertAlmostEqual(result["service"]["family_mean_auc"], .8)
        self.assertAlmostEqual(result["service"]["dataset_mean_auc"], 23/30)
        empty = module.paired_summary([], "service_policy")
        self.assertIsNone(empty["service"]["dataset_mean_auc"])

    def test_freeze_rejects_missing_foreign_and_corrupted_score_files(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            protocol = {"catalog": [{"dataset_name": "a"}, {"dataset_name": "b"}]}
            (folder / "protocol_antigravity.json").write_text(json.dumps(protocol))
            module.write_score(folder / "score_0000_antigravity.json.gz", {"dataset": "a", "models": {}})
            with self.assertRaises(ValueError):
                module.score_inventory(folder, protocol)
            module.write_score(folder / "score_0001_antigravity.json.gz", {"dataset": "wrong", "models": {}})
            with self.assertRaises(ValueError):
                module.score_inventory(folder, protocol)

    def test_score_write_preserves_existing_checkpoint(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "score_antigravity.json.gz"
            module.write_score(path, {"dataset": "a", "models": {}})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                module.write_score(path, {"dataset": "b", "models": {}})
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
