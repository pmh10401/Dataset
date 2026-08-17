"""Closed-command and synthetic candidate-flow tests for Exp456 Task 4."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


_COMMIT_ANTIGRAVITY = "a" * 40
_ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
_LAUNCHER_ANTIGRAVITY = _ROOT_ANTIGRAVITY / "scratch/run_exp456_exp455_dataset_key_order_repair_antigravity.py"
_OFFICIAL_ROOT_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/exp456_exp455_dataset_key_order_repair_antigravity_20260817")


class CliAntigravityTests(unittest.TestCase):
    def test_closed_cli_module_and_launcher_expose_only_reviewed_evaluate_or_bare_inspect_antigravity(self) -> None:
        """Break caught: an absent boundary or parser override reaches a lifecycle phase."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.cli_antigravity import parser_antigravity

        self.assertTrue(_LAUNCHER_ANTIGRAVITY.is_file())
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
            ("evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY, "--override"),
        )
        for argv in rejected:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(argv)

    def test_dispatcher_loads_only_selected_phase_antigravity(self) -> None:
        """Break caught: inspect imports evaluation or evaluate imports replay."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity import cli_antigravity as cli

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

    def test_exact_command_tokens_reject_subcommand_help_and_equals_form_before_loading_antigravity(self) -> None:
        """Break caught: argparse expands the closed command surface before phase selection."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity import cli_antigravity as cli

        for argv in (
            ("inspect", "--help"),
            ("evaluate", "--help"),
            ("evaluate", f"--reviewed-commit={_COMMIT_ANTIGRAVITY}"),
        ):
            with self.subTest(argv=argv):
                with (
                    patch.object(cli, "_load_evaluate_antigravity", side_effect=AssertionError("evaluator loaded")),
                    patch.object(cli, "_load_inspect_antigravity", side_effect=AssertionError("replay loaded")),
                    redirect_stderr(io.StringIO()),
                    redirect_stdout(io.StringIO()),
                    self.assertRaises(SystemExit) as exited,
                ):
                    cli.main_antigravity(argv)
                self.assertNotEqual(exited.exception.code, 0)

        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as top_level_help:
            cli.main_antigravity(["--help"])
        self.assertEqual(top_level_help.exception.code, 0)

    def test_evaluate_orders_authority_features_lodo_evaluation_publication_and_aggregate_without_official_root_antigravity(self) -> None:
        """Break caught: evaluate skips a frozen stage, publishes elsewhere, or emits non-aggregate output."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity import cli_antigravity as cli
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.artifacts_antigravity import publish_evaluation_antigravity
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.dashboard_export_antigravity import export_aggregate_dashboard_antigravity
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.evaluation_antigravity import evaluate_routing_antigravity
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.router_antigravity import leave_one_dataset_out_antigravity
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.tests.test_evaluation_antigravity import RoutingEvaluationAntigravityTests

        authority, expected_features, _expected_decisions = RoutingEvaluationAntigravityTests._inputs_antigravity()
        calls: list[str] = []
        with tempfile.TemporaryDirectory(prefix="exp456_cli_flow_antigravity_") as temporary:
            root = Path(temporary) / "result"
            self.assertNotEqual(root, _OFFICIAL_ROOT_ANTIGRAVITY)
            evaluation, frozen = cli._evaluate_with_dependencies_antigravity(
                _COMMIT_ANTIGRAVITY,
                lambda commit: (calls.append("validate"), self.assertEqual(commit, _COMMIT_ANTIGRAVITY)),
                lambda: (calls.append("authority"), authority)[1],
                lambda captured: (calls.append("features"), self.assertIs(captured, authority), expected_features)[2],
                lambda features, outcomes: (calls.append("lodo"), leave_one_dataset_out_antigravity(features, outcomes))[1],
                lambda captured, features, decisions: (calls.append("evaluate"), evaluate_routing_antigravity(captured, features, decisions))[1],
                lambda destination, result: (calls.append("publish"), publish_evaluation_antigravity(destination, result))[1],
                root,
            )
            output = io.StringIO()
            with redirect_stdout(output):
                cli.emit_aggregate_json_antigravity(evaluation, export_aggregate_dashboard_antigravity)
            self.assertEqual(calls, ["validate", "authority", "features", "lodo", "evaluate", "publish"])
            self.assertEqual(len(evaluation.decisions), 8)
            self.assertEqual(len(authority.rows), 48)
            self.assertEqual(len(evaluation.comparisons), 4)
            self.assertEqual(len(frozen.leaf_sha256), 8)
            self.assertEqual(authority.label_query_count, 0)
            self.assertEqual(
                output.getvalue(),
                json.dumps(export_aggregate_dashboard_antigravity(evaluation), sort_keys=True, separators=(",", ":")) + "\n",
            )

    def test_inspect_calls_public_replay_at_injected_root_and_emits_same_aggregate_json_antigravity(self) -> None:
        """Break caught: inspect bypasses public replay or receives an implicit official root."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity import cli_antigravity as cli
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.dashboard_export_antigravity import export_aggregate_dashboard_antigravity
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.tests.test_evaluation_antigravity import RoutingEvaluationAntigravityTests
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.evaluation_antigravity import evaluate_routing_antigravity
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.router_antigravity import leave_one_dataset_out_antigravity

        authority, features, decisions = RoutingEvaluationAntigravityTests._inputs_antigravity()
        evaluation = evaluate_routing_antigravity(authority, features, leave_one_dataset_out_antigravity(features, {row.dataset: (row.baseline_confusion, row.candidate_confusion) for row in authority.rows if row.candidate_confusion is not None}))
        with tempfile.TemporaryDirectory(prefix="exp456_cli_inspect_antigravity_") as temporary:
            root = Path(temporary) / "result"
            called: list[Path] = []
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(cli._inspect_with_dependencies_antigravity(root, lambda destination: (called.append(destination), evaluation)[1], export_aggregate_dashboard_antigravity), 0)
            self.assertEqual(called, [root])
            self.assertNotEqual(root, _OFFICIAL_ROOT_ANTIGRAVITY)
            self.assertIn('"experiment_id":"Exp456"', output.getvalue())
            self.assertNotIn("dataset_evidence", output.getvalue())

    def test_launcher_is_outside_cwd_safe_and_only_calls_package_cli_antigravity(self) -> None:
        """Break caught: launcher relies on CWD or imports an execution phase directly."""
        source = _LAUNCHER_ANTIGRAVITY.read_text(encoding="utf-8")
        self.assertIn("main_antigravity", source)
        self.assertNotIn("evaluation_antigravity", source)
        self.assertNotIn("replay_antigravity", source)
        self.assertIn("Path(__file__).resolve().parents[1]", source)


if __name__ == "__main__":
    unittest.main()
