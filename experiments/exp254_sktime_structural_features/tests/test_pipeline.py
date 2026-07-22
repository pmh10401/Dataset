from __future__ import annotations

import unittest
from pathlib import Path

from experiments.exp254_sktime_structural_features.run_pipeline import parser


class PipelineTest(unittest.TestCase):
    def test_cli_stages(self) -> None:
        choices = parser()._option_string_actions["--stage"].choices
        self.assertEqual(set(choices), {"audit", "cache", "predict", "evaluate", "all"})

    def test_dashboard_registration(self) -> None:
        root = Path(__file__).resolve().parents[3]
        self.assertIn("experiment_254_sktime_structural_features", (root / "serve_rank_dashboard.py").read_text())


if __name__ == "__main__":
    unittest.main()
