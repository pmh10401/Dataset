"""Closed command and outside-CWD launcher regression tests for Exp452."""
from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch


_COMMIT_ANTIGRAVITY = "a" * 40
_FIXED_PYTHON_ANTIGRAVITY = "/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python"
_THREADS_ANTIGRAVITY = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS", "AEON_NUM_THREADS", "MLX_NUM_THREADS",
)
_ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
_LAUNCHER_ANTIGRAVITY = _ROOT_ANTIGRAVITY / "scratch/run_exp452_exp449_index_order_repair_antigravity.py"


class CliAntigravityTests(unittest.TestCase):
    def test_parser_exposes_only_three_exact_forms_antigravity(self) -> None:
        """Break caught: an abbreviation, override, extra command, or inspect argument reaches a phase."""
        from experiments.exp452_exp449_index_order_repair_antigravity.cli_antigravity import build_parser_antigravity

        parser = build_parser_antigravity()
        self.assertEqual(parser.parse_args(["readiness", "--reviewed-commit", _COMMIT_ANTIGRAVITY]).command, "readiness")
        self.assertEqual(parser.parse_args(["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY]).command, "evaluate")
        self.assertEqual(parser.parse_args(["inspect"]).command, "inspect")
        rejected = (
            ("read", "--reviewed-commit", _COMMIT_ANTIGRAVITY),
            ("readiness", "--reviewed-com", _COMMIT_ANTIGRAVITY),
            ("predict", "--reviewed-commit", _COMMIT_ANTIGRAVITY),
            ("new",), ("resume",), ("retry",),
            ("evaluate", "--reviewed-commit", "A" * 40),
            ("evaluate", "--reviewed-commit", "a" * 39),
            ("inspect", "--reviewed-commit", _COMMIT_ANTIGRAVITY),
        )
        for arguments in rejected:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(arguments)

    def test_safe_commands_leave_evaluator_and_label_modules_unloaded_antigravity(self) -> None:
        """Break caught: help, readiness, or inspect eagerly imports evaluator or TEST-label code."""
        probe = (
            "import json,sys;sys.path.insert(0,%r);" % str(_ROOT_ANTIGRAVITY) +
            "from experiments.exp452_exp449_index_order_repair_antigravity import cli_antigravity as c;"
            "c.build_parser_antigravity().parse_args(['--help'])"
        )
        result = subprocess.run(
            (_FIXED_PYTHON_ANTIGRAVITY, "-c", probe), cwd="/tmp", capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        loaded_probe = (
            "import json,sys;sys.path.insert(0,%r);" % str(_ROOT_ANTIGRAVITY) +
            "from experiments.exp452_exp449_index_order_repair_antigravity import cli_antigravity as c;"
            "print(json.dumps(sorted(name for name in sys.modules if name.endswith(('_antigravity.evaluation_antigravity','_antigravity.sources_antigravity','_antigravity.dashboard_export_antigravity')))))"
        )
        loaded = subprocess.run(
            (_FIXED_PYTHON_ANTIGRAVITY, "-c", loaded_probe), cwd="/tmp", capture_output=True, text=True, check=False
        )
        self.assertEqual(loaded.returncode, 0, loaded.stderr)
        self.assertEqual(json.loads(loaded.stdout), [])
        from experiments.exp452_exp449_index_order_repair_antigravity import cli_antigravity as cli
        with patch.object(cli, "_load_readiness_antigravity", return_value=lambda value: {"commit": value}), patch.object(cli, "_load_inspect_antigravity", return_value=lambda: {"replay": "safe"}), patch.object(cli, "_load_evaluate_antigravity", side_effect=AssertionError("evaluator imported")), patch.object(cli, "_print_result_antigravity"):
            self.assertEqual(cli.main_antigravity(["readiness", "--reviewed-commit", _COMMIT_ANTIGRAVITY]), 0)
            self.assertEqual(cli.main_antigravity(["inspect"]), 0)

    def test_evaluate_runs_clean_reviewed_map_gate_before_evaluator_import_antigravity(self) -> None:
        """Break caught: evaluator import or label reachability happens before the clean reviewed-HEAD/map gate."""
        from experiments.exp452_exp449_index_order_repair_antigravity import cli_antigravity as cli

        events: list[tuple[str, str]] = []
        def gate(commit: str) -> None:
            events.append(("gate", commit))
        def evaluator():
            events.append(("evaluator_import", ""))
            return lambda commit: {"commit": commit}
        with patch.object(cli, "_validate_reviewed_map_antigravity", side_effect=gate), patch.object(cli, "_load_evaluate_antigravity", side_effect=evaluator), patch.object(cli, "_print_result_antigravity"):
            self.assertEqual(cli.main_antigravity(["evaluate", "--reviewed-commit", _COMMIT_ANTIGRAVITY]), 0)
        self.assertEqual(events, [("gate", _COMMIT_ANTIGRAVITY), ("evaluator_import", "")])

    def test_launcher_pins_exact_threads_and_runs_outside_repository_antigravity(self) -> None:
        """Break caught: launcher depends on cwd or imports before its seven one-thread pins."""
        environment = {**os.environ, **{name: "999" for name in _THREADS_ANTIGRAVITY}}
        help_result = subprocess.run(
            (_FIXED_PYTHON_ANTIGRAVITY, str(_LAUNCHER_ANTIGRAVITY), "--help"), cwd="/tmp", env=environment,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("readiness", help_result.stdout)
        probe = "import json,os,runpy;runpy.run_path(%r);print(json.dumps({name:os.environ[name] for name in %r},sort_keys=True))" % (str(_LAUNCHER_ANTIGRAVITY), _THREADS_ANTIGRAVITY)
        result = subprocess.run(
            (_FIXED_PYTHON_ANTIGRAVITY, "-c", probe), cwd="/tmp", env=environment,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {name: "1" for name in _THREADS_ANTIGRAVITY})


if __name__ == "__main__":
    unittest.main()
