from __future__ import annotations

import inspect
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.exp206_210 import artifacts


class Exp206210StructureTests(unittest.TestCase):
    def test_prediction_loaders_do_not_accept_y_test(self):
        self.assertNotIn("y_test", inspect.signature(artifacts.load_population_values_only).parameters)

    def test_evaluation_label_loader_is_separate(self):
        self.assertTrue(callable(artifacts.load_test_labels_for_evaluation))

    def test_stage_ids_are_frozen(self):
        from experiments.exp206_210.run_experiment_206_210 import STAGE_META
        self.assertEqual(list(STAGE_META), ["206", "207", "208", "209", "210"])

    def test_runner_can_start_by_absolute_script_path_without_pythonpath(self):
        script = Path(__file__).with_name("run_experiment_206_210.py").resolve()
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(script), "--help"],
                cwd=directory,
                env={"PATH": str(Path(sys.executable).parent)},
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
