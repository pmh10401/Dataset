"""Exp445 evaluation boundary tests using only temporary frozen fixtures."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


class Exp445EvaluationAntigravityTests(unittest.TestCase):
    def test_profile_evaluator_waits_for_two_inspections_before_labels_antigravity(self) -> None:
        """Exp445 must not cross its TEST-label boundary before two frozen replays agree."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import evaluation_antigravity as evaluation
        from experiments.exp444_train_only_stability_precision_gate_antigravity.profiles_antigravity import EXP445_PROFILE_ANTIGRAVITY
        from experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_evaluation_antigravity import _DependenciesAntigravity

        with tempfile.TemporaryDirectory(prefix="exp445_evaluation_antigravity_") as directory:
            dependencies = _DependenciesAntigravity(Path(directory))
            result = evaluation._evaluate_for_profile_antigravity(
                EXP445_PROFILE_ANTIGRAVITY, dependencies=dependencies,
            )
            self.assertEqual(result.status, "descriptive_complete_no_promotion")
            self.assertEqual(
                dependencies.events[:6],
                ["inspect_publication", "identity", "hard", "inspect_publication", "identity", "open_labels"],
            )
            payload = json.loads((dependencies.evaluation_root_antigravity() / "evaluation_antigravity.json").read_text())
            self.assertEqual(payload["schema"], "exp445.evaluation.v1")

    def test_complete_exp445_evaluation_replays_without_labels_antigravity(self) -> None:
        """A complete Exp445 evaluation is a source-free, label-free immutable replay."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import evaluation_antigravity as evaluation
        from experiments.exp444_train_only_stability_precision_gate_antigravity.profiles_antigravity import EXP445_PROFILE_ANTIGRAVITY
        from experiments.exp444_train_only_stability_precision_gate_antigravity.tests.test_evaluation_antigravity import _DependenciesAntigravity

        with tempfile.TemporaryDirectory(prefix="exp445_evaluation_replay_antigravity_") as directory:
            dependencies = _DependenciesAntigravity(Path(directory))
            first = evaluation._evaluate_for_profile_antigravity(EXP445_PROFILE_ANTIGRAVITY, dependencies=dependencies)
            with mock.patch.object(dependencies, "open_labels_antigravity", side_effect=AssertionError("labels must remain closed")):
                replay = evaluation._evaluate_for_profile_antigravity(EXP445_PROFILE_ANTIGRAVITY, dependencies=dependencies)
            self.assertEqual(replay.evaluation_sha256, first.evaluation_sha256)
            self.assertEqual(dependencies.events.count("open_labels"), 1)
