from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Exp177185QueueContractTest(unittest.TestCase):
    def test_queue_contains_exact_ordered_batch(self):
        source = (ROOT / "run_rank_experiments_sequential.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        assignments = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "EXP177_185_BATCH"
        }
        batch = assignments["EXP177_185_BATCH"]
        self.assertEqual(len(batch), 9)
        self.assertTrue(batch[0].startswith("experiment_177_"))
        self.assertTrue(batch[-1].startswith("experiment_185_"))

    def test_runner_records_failure_and_continues(self):
        source = (ROOT / "run_rank_experiments_sequential.py").read_text(encoding="utf-8")
        self.assertIn("QUEUE recorded failure", source)
        self.assertIn("continuing", source)
        self.assertIn("hard_deadline_unix", source)

    def test_dashboard_catalogs_every_batch_stage(self):
        source = (ROOT / "serve_rank_dashboard.py").read_text(encoding="utf-8")
        for number in range(177, 186):
            self.assertIn(f'"{number}"', source)


if __name__ == "__main__":
    unittest.main()
