"""Closed command and launcher tests for Exp449."""
from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


_THREADS_ANTIGRAVITY = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "AEON_NUM_THREADS",
)
_LAUNCHER_ANTIGRAVITY = Path(__file__).resolve().parents[3] / "scratch/run_exp449_residual_fn_method_qualification_antigravity.py"


class CliAntigravityTests(unittest.TestCase):
    def test_parser_accepts_only_four_exact_forms_antigravity(self) -> None:
        """Break caught: a shorthand, override, retry, or inspect argument reaches a phase."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.cli_antigravity import build_parser_antigravity

        parser = build_parser_antigravity(); commit = "a" * 40
        self.assertEqual(parser.parse_args(["readiness", "--reviewed-commit", commit]).command, "readiness")
        self.assertEqual(parser.parse_args(["predict", "--reviewed-commit", commit]).command, "predict")
        self.assertEqual(parser.parse_args(["evaluate", "--reviewed-commit", commit]).command, "evaluate")
        self.assertEqual(parser.parse_args(["inspect"]).command, "inspect")
        rejected = (("read", "--reviewed-commit", commit), ("readiness", "--reviewed-com", commit),
                    ("predict", "--root", "x", "--reviewed-commit", commit), ("retry",), ("resume",),
                    ("evaluate", "--reviewed-commit", "A" * 40), ("inspect", "--reviewed-commit", commit))
        for arguments in rejected:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(arguments)

    def test_dispatch_imports_only_the_selected_phase_antigravity(self) -> None:
        """Break caught: help or a safe phase imports label/evaluator dependencies eagerly."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import cli_antigravity as cli

        commit = "a" * 40
        with patch.object(cli, "_load_readiness_antigravity", return_value=lambda value: {"commit": value}) as readiness, patch.object(cli, "_load_predict_antigravity", side_effect=AssertionError("predict imported")), patch.object(cli, "_load_evaluate_antigravity", side_effect=AssertionError("evaluate imported")), patch.object(cli, "_load_inspect_antigravity", side_effect=AssertionError("inspect imported")), patch.object(cli, "_print_result_antigravity"):
            self.assertEqual(cli.main_antigravity(["readiness", "--reviewed-commit", commit]), 0)
        readiness.assert_called_once_with()

    def test_launcher_runs_outside_repository_with_thread_pins_antigravity(self) -> None:
        """Break caught: the launcher imports before pinning one-thread native libraries."""
        environment = {**os.environ, **{name: "999" for name in _THREADS_ANTIGRAVITY}}
        help_result = subprocess.run([sys.executable, str(_LAUNCHER_ANTIGRAVITY), "--help"], cwd="/tmp", env=environment, capture_output=True, text=True, check=False)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("readiness", help_result.stdout)
        probe = "import json, os, runpy; runpy.run_path(%r); print(json.dumps({name: os.environ[name] for name in %r}, sort_keys=True))" % (str(_LAUNCHER_ANTIGRAVITY), _THREADS_ANTIGRAVITY)
        result = subprocess.run([sys.executable, "-c", probe], cwd="/tmp", env=environment, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {name: "1" for name in _THREADS_ANTIGRAVITY})


if __name__ == "__main__":
    unittest.main()
