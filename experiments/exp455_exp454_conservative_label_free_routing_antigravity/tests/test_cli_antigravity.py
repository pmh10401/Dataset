"""Closed-command and synthetic-flow tests for Exp455 Task 5."""

from __future__ import annotations

from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


_COMMIT_ANTIGRAVITY = "a" * 40
_ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
_LAUNCHER_ANTIGRAVITY = _ROOT_ANTIGRAVITY / "scratch/run_exp455_exp454_conservative_label_free_routing_antigravity.py"


class CliAntigravityTests(unittest.TestCase):
    def test_parser_accepts_only_evaluate_with_one_lowercase_commit_or_bare_inspect_antigravity(self) -> None:
        """Break caught: an abbreviation or runtime override reaches a lifecycle phase."""
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.cli_antigravity import parser_antigravity

        parser = parser_antigravity()
        self.assertEqual(parser.parse_args(["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY]).command, "evaluate")
        self.assertEqual(parser.parse_args(["inspect"]).command, "inspect")
        rejected = (
            ("eval", "--reviewed-commit", _COMMIT_ANTIGRAVITY),
            ("evaluate", "--reviewed-com", _COMMIT_ANTIGRAVITY),
            ("evaluate", "--reviewed-commit", "A" * 40),
            ("evaluate", "--reviewed-commit", "a" * 39),
            ("evaluate",),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--reviewed-commit", "b" * 40),
            ("inspect", "--reviewed-commit", _COMMIT_ANTIGRAVITY),
            ("inspect", "extra"),
            ("new",),
            ("resume",),
            ("retry",),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--root", "/tmp"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--threshold", "1"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--candidate", "other"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--family", "Other"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--method", "other"),
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--override"),
        )
        for argv in rejected:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(argv)

    def test_dispatcher_loads_only_selected_phase_antigravity(self) -> None:
        """Break caught: inspect imports evaluator or evaluate imports replay."""
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity import cli_antigravity as cli

        with (
            patch.object(cli, "_load_evaluate_antigravity", side_effect=AssertionError("evaluator loaded")),
            patch.object(cli, "_load_inspect_antigravity", return_value=lambda: 0) as inspect,
        ):
            self.assertEqual(cli.main_antigravity(["inspect"]), 0)
            inspect.assert_called_once_with()
        with (
            patch.object(cli, "_load_evaluate_antigravity", return_value=lambda commit: 0) as evaluate,
            patch.object(cli, "_load_inspect_antigravity", side_effect=AssertionError("replay loaded")),
        ):
            self.assertEqual(cli.main_antigravity(["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY]), 0)
            evaluate.assert_called_once_with()

    def test_synthetic_full_flow_has_eight_folds_forty_eight_rows_four_comparisons_eight_leaves_and_no_labels_antigravity(self) -> None:
        """Break caught: closed evaluation skips LODO, double-counts rows, or bypasses replay."""
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity import cli_antigravity as cli
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.artifacts_antigravity import publish_evaluation_antigravity
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.evaluation_antigravity import evaluate_routing_antigravity
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.replay_antigravity import _inspect_replay_with_dependencies_antigravity
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.router_antigravity import leave_one_dataset_out_antigravity
        from experiments.exp455_exp454_conservative_label_free_routing_antigravity.tests.test_evaluation_antigravity import RoutingEvaluationAntigravityTests

        authority, expected_features, _expected_decisions = RoutingEvaluationAntigravityTests._inputs_antigravity()
        with tempfile.TemporaryDirectory(prefix="exp455_cli_flow_antigravity_") as temporary:
            root = Path(temporary) / "result"
            evaluation, frozen = cli._evaluate_with_dependencies_antigravity(
                _COMMIT_ANTIGRAVITY,
                lambda commit: self.assertEqual(commit, _COMMIT_ANTIGRAVITY),
                lambda: authority,
                lambda captured: expected_features,
                leave_one_dataset_out_antigravity,
                evaluate_routing_antigravity,
                publish_evaluation_antigravity,
                root,
            )
            self.assertEqual(len(evaluation.decisions), 8)
            self.assertEqual(len(authority.rows), 48)
            self.assertEqual(len(evaluation.comparisons), 4)
            self.assertEqual(len(frozen.leaf_sha256), 8)
            self.assertEqual(authority.label_query_count, 0)
            self.assertEqual(_inspect_replay_with_dependencies_antigravity(root, authority, expected_features), evaluation)

    def test_launcher_is_outside_cwd_safe_and_only_calls_cli_main_antigravity(self) -> None:
        """Break caught: launcher relies on the repository current directory or imports a phase directly."""
        source = _LAUNCHER_ANTIGRAVITY.read_text(encoding="utf-8")
        self.assertIn("main_antigravity", source)
        self.assertNotIn("evaluation_antigravity", source)
        self.assertNotIn("replay_antigravity", source)
        self.assertIn("Path(__file__).resolve().parents[1]", source)


if __name__ == "__main__":
    unittest.main()
