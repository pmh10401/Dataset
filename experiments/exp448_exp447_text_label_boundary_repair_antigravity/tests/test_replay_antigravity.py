"""Source-free replay tests for the Exp448 inspect boundary."""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest


class ReplayAntigravityTests(unittest.TestCase):
    def test_replay_semantics_match_evaluator_on_complete_temporary_artifact_antigravity(self) -> None:
        # The temporary fixture has synthetic labels only. The replay module is
        # passed an already-captured identity dependency and never receives a
        # label session or imports source/evaluation authority itself.
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity.evaluation_antigravity import _evaluate_with_dependencies_antigravity
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity.tests.test_evaluation_antigravity import _DependenciesAntigravity
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity.replay_antigravity import (
            inspect_replay_with_dependencies_antigravity,
        )

        with tempfile.TemporaryDirectory(prefix="exp447_replay_antigravity_") as directory:
            dependencies = _DependenciesAntigravity(Path(directory) / "evaluation_root_antigravity")
            expected = _evaluate_with_dependencies_antigravity("a" * 40, dependencies)
            observed = inspect_replay_with_dependencies_antigravity(dependencies)
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
