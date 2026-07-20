from __future__ import annotations

import unittest
from pathlib import Path

from experiments.exp251_sktime_statistical_features.config import VENV_PYTHON
from scratch.run_exp251_after_current import EXPERIMENT_ID, build_command


REPO_ROOT = Path(__file__).resolve().parents[3]


class QueueDashboardContractTest(unittest.TestCase):
    def test_deferred_command_uses_isolated_sktime_interpreter(self) -> None:
        command = build_command(dataset_limit=None, dataset_list_file=None, reset_cache=False)
        self.assertEqual(command[0], str(VENV_PYTHON))
        self.assertIn("experiments.exp251_sktime_statistical_features.run_pipeline", command)
        self.assertIn("7", command)

    def test_dashboard_registers_exp251(self) -> None:
        source = (REPO_ROOT / "serve_rank_dashboard.py").read_text()
        self.assertIn(EXPERIMENT_ID, source)
        self.assertIn("exp251_sktime_statistical_features", source)


if __name__ == "__main__":
    unittest.main()
