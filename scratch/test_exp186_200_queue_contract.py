from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Exp186200QueueContractTest(unittest.TestCase):
    def test_queue_contains_exact_ordered_batch(self):
        source = (ROOT / "run_rank_experiments_sequential.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        
        # Look for EXP186_200_BATCH definition
        batch = None
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "EXP186_200_BATCH"
            ):
                batch = ast.literal_eval(node.value)
                break
                
        self.assertIsNotNone(batch, "EXP186_200_BATCH not defined in run_rank_experiments_sequential.py")
        self.assertEqual(len(batch), 15)
        self.assertTrue(batch[0].startswith("experiment_186_"))
        self.assertTrue(batch[-1].startswith("experiment_200_"))

        # Verify strict order
        for idx, stage in enumerate(range(186, 201)):
            self.assertIn(str(stage), batch[idx])

    def test_dashboard_catalogs_every_batch_stage(self):
        source = (ROOT / "serve_rank_dashboard.py").read_text(encoding="utf-8")
        for stage in range(186, 201):
            self.assertIn(f'"{stage}"', source)

    def test_queue_uses_correct_runner_script(self):
        source = (ROOT / "run_rank_experiments_sequential.py").read_text(encoding="utf-8")
        self.assertIn("experiments/exp186_200/run_experiment_186_200.py", source)


if __name__ == "__main__":
    unittest.main()
