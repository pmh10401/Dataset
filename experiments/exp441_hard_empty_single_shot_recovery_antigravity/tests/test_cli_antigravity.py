"""Behavior tests for Exp441's deliberately small CLI surface."""
from __future__ import annotations

import contextlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


class CliAntigravityTests(unittest.TestCase):
    """The production changes caught here are route expansion and non-JSON success output."""

    def test_public_cli_has_exact_four_commands(self) -> None:
        module = importlib.import_module(
            "experiments.exp441_hard_empty_single_shot_recovery_antigravity.cli_antigravity"
        )
        parser = module.build_parser_antigravity()
        self.assertEqual(set(parser._subparsers._group_actions[0].choices), {"readiness", "new", "inspect", "evaluate"})
        with self.assertRaises(SystemExit):
            parser.parse_args(["resume"])

    def test_readiness_route_returns_zero_and_prints_one_canonical_json_document(self) -> None:
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import cli_antigravity as cli

        result = {"readiness_sha256": "a" * 64}
        stream = io.StringIO()
        with patch.object(cli, "current_clean_head_antigravity", return_value="b" * 40), patch(
            "experiments.exp441_hard_empty_single_shot_recovery_antigravity.readiness_antigravity.run_readiness_antigravity",
            return_value=result,
        ):
            with contextlib.redirect_stdout(stream):
                self.assertEqual(cli.main_antigravity(["readiness"]), 0)
        self.assertEqual(json.loads(stream.getvalue()), {"command": "readiness", "result": result})
        self.assertEqual(stream.getvalue().encode(), json.dumps(json.loads(stream.getvalue()), sort_keys=True, separators=(",", ":")).encode() + b"\n")

    def test_parser_rejects_every_override_surface(self) -> None:
        """A route accepting path/worker/backend/seed/budget flags would unfreeze Exp441."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import cli_antigravity as cli

        parser = cli.build_parser_antigravity()
        for flag in ("--path", "--workers", "--backend", "--seed", "--budget", "--source", "--threshold", "--bootstrap"):
            with self.assertRaises(SystemExit, msg=flag): parser.parse_args(["readiness", flag, "x"])

    def test_evaluate_is_the_sole_lazy_evaluator_route(self) -> None:
        """Prediction routes must remain importable when the future evaluator is absent."""
        from experiments.exp441_hard_empty_single_shot_recovery_antigravity import cli_antigravity as cli

        self.assertNotIn("experiments.exp441_hard_empty_single_shot_recovery_antigravity.evaluation_antigravity", sys.modules)
        with self.assertRaises(ModuleNotFoundError): cli.main_antigravity(["evaluate"])

    def test_outside_cwd_wrapper_help_needs_only_the_wrapper_and_cli(self) -> None:
        """Resolving imports relative to CWD would break the documented outside-CWD command."""
        wrapper = Path(__file__).resolve().parents[3] / "scratch" / "run_exp441_hard_empty_single_shot_recovery_antigravity.py"
        completed = subprocess.run([sys.executable, str(wrapper), "--help"], cwd="/tmp", check=True, capture_output=True, text=True)
        self.assertIn("{readiness,new,inspect,evaluate}", completed.stdout)
