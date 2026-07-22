from __future__ import annotations

import unittest
import subprocess
import sys
from pathlib import Path

from experiments.exp252_sktime_statistical_features_safe_hurst.config import VENV_PYTHON
from scratch.run_exp252_safe_hurst import EXPERIMENT_ID, build_command


REPO_ROOT = Path(__file__).resolve().parents[3]


class QueueDashboardContractTest(unittest.TestCase):
    def test_deferred_command_uses_isolated_sktime_interpreter(self) -> None:
        command = build_command(dataset_limit=None, dataset_list_file=None, reset_cache=False)
        self.assertEqual(command[0], str(VENV_PYTHON))
        self.assertIn("experiments.exp252_sktime_statistical_features_safe_hurst.run_pipeline", command)
        self.assertIn("7", command)

    def test_dashboard_registers_exp252(self) -> None:
        source = (REPO_ROOT / "serve_rank_dashboard.py").read_text()
        self.assertIn(EXPERIMENT_ID, source)
        self.assertIn("exp252_sktime_statistical_features_safe_hurst", source)

    def test_deferred_runner_can_start_outside_repository(self) -> None:
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scratch/run_exp252_safe_hurst.py"), "--help"],
            cwd="/tmp",
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
