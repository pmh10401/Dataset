"""Publication must verify frozen inputs and keep ranking-only observations."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("finalize_antigravity.py")


class FinalizeTests(unittest.TestCase):
    def module(self):
        self.assertTrue(SCRIPT.exists(), "verified publication finalizer is not implemented")
        spec = importlib.util.spec_from_file_location("finalize_rockad_test", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_ranking_summary_includes_uncalibrated_pairs_but_not_failures(self):
        module = self.module()
        rows = [
            {"dataset": "a", "family": "A", "models": {"service": {"ranking": {"auc": .9, "ap": .8}}, "rockad": {"status": "ranking_only", "ranking": {"auc": .7, "ap": .6}}}},
            {"dataset": "b", "family": "B", "models": {"service": {"ranking": {"auc": .5, "ap": .4}}, "rockad": {"status": "ok", "ranking": {"auc": .9, "ap": .8}}}},
            {"dataset": "c", "family": "C", "models": {"rockad": {"status": "error"}}},
        ]
        summary = module.ranking_summary(rows)
        self.assertEqual(summary["paired_datasets"], 2)
        self.assertAlmostEqual(summary["service"]["dataset_mean_auc"], .7)
        self.assertAlmostEqual(summary["rockad"]["dataset_mean_auc"], .8)

    def test_atomic_json_preserves_old_result_on_invalid_values(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "result_antigravity.json"
            module.atomic_json(path, {"valid": 1})
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                module.atomic_json(path, {"invalid": float("nan")})
            self.assertEqual(path.read_bytes(), original)

    def test_missing_freeze_blocks_finalization_before_database_access(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(FileNotFoundError):
                module.validate_snapshot(Path(name), Path(name)/"missing_source.db")

    def test_user_exclusion_preserves_completed_scores_and_freezes_scope(self):
        module = self.module()
        self.assertTrue(hasattr(module, "exclude_cornell_and_freeze"), "explicit exclusion finalization is missing")
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            database = folder / "source.db"
            database.write_bytes(b"fixture")
            protocol = {"catalog": [{"dataset_name": "CornellWhaleChallenge"}, {"dataset_name": "completed"}],
                        "database_sha256": module.run.digest(database), "runner_sha256": module.run.digest(module.run.__file__)}
            module.run.write_json(folder / "protocol_antigravity.json", protocol)
            checkpoint = folder / "score_0001_antigravity.json.gz"
            module.run.write_score(checkpoint, {"dataset": "completed", "models": {}})
            original = checkpoint.read_bytes()
            module.exclude_cornell_and_freeze(folder, database)
            self.assertEqual(checkpoint.read_bytes(), original)
            excluded = module.run.read_score(folder / "score_0000_antigravity.json.gz")
            self.assertEqual(excluded["models"]["rockad"]["status"], "excluded")
            self.assertNotIn("rows", excluded["models"]["rockad"])
            freeze = module.json.loads((folder / "score_freeze_antigravity.json").read_text())
            self.assertEqual(len(freeze["score_files"]), 2)
            self.assertEqual(freeze["effective_datasets"], 1)
            self.assertEqual(freeze["excluded_datasets"], ["CornellWhaleChallenge"])

    def test_exclusion_refuses_other_missing_results(self):
        module = self.module()
        self.assertTrue(hasattr(module, "exclude_cornell_and_freeze"), "explicit exclusion finalization is missing")
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            module.run.write_json(folder / "protocol_antigravity.json", {"catalog": [
                {"dataset_name": "CornellWhaleChallenge"}, {"dataset_name": "missing"}]})
            with self.assertRaises(ValueError):
                module.exclude_cornell_and_freeze(folder, folder / "absent.db")
            self.assertFalse((folder / "score_0000_antigravity.json.gz").exists())


if __name__ == "__main__":
    unittest.main()
