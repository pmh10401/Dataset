from __future__ import annotations

import unittest
from pathlib import Path

from experiments.exp253_guarded_standard_to_hard_expansion.run_pipeline import parser


class PipelineContractTest(unittest.TestCase):
    def test_pipeline_exposes_separate_stages(self) -> None:
        choices = parser()._option_string_actions["--stage"].choices
        self.assertEqual(set(choices), {"audit", "predict", "evaluate", "all"})

    def test_dashboard_registers_exp253(self) -> None:
        root = Path(__file__).resolve().parents[3]
        source = (root / "serve_rank_dashboard.py").read_text()
        self.assertIn("experiment_253_guarded_standard_to_hard_expansion", source)


if __name__ == "__main__":
    unittest.main()
