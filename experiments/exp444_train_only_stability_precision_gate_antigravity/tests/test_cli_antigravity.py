"""Behavior tests for Exp444's fixed operator boundary."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch


class CliAntigravityTests(unittest.TestCase):
    def test_exactly_five_commands_reject_every_override_surface_antigravity(self) -> None:
        """An accepted scientific or path flag would bypass the reviewed frozen contract."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import cli_antigravity as cli

        parser = cli.build_parser_antigravity()
        self.assertEqual(set(parser._subparsers._group_actions[0].choices), {"readiness", "new", "resume", "inspect", "evaluate"})
        for flag in ("--root", "--backend", "--workers", "--seed", "--view", "--fraction", "--source", "--threshold", "--budget"):
            with self.assertRaises(SystemExit, msg=flag):
                parser.parse_args(["new", flag, "x"])

    def test_resume_accepts_only_one_safe_attempt_basename_antigravity(self) -> None:
        """A path-like resume identifier could redirect mutation outside the frozen attempt."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import cli_antigravity as cli

        parser = cli.build_parser_antigravity()
        self.assertEqual(parser.parse_args(["resume", "--reviewed-commit", "a" * 40, "--attempt", "attempt_0000_antigravity"]).attempt, "attempt_0000_antigravity")
        for name in ("../attempt_0000_antigravity", "attempt_1_antigravity", "attempt_0000_antigravity/child"):
            with self.assertRaises(SystemExit, msg=name):
                parser.parse_args(["resume", "--reviewed-commit", "a" * 40, "--attempt", name])

    def test_evaluate_inspects_before_lazy_evaluator_import_antigravity(self) -> None:
        """Importing the evaluator first could make TEST labels reachable after a bad publication."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import cli_antigravity as cli

        imported: list[str] = []
        module_name = "experiments.exp444_train_only_stability_precision_gate_antigravity.evaluation_antigravity"
        fake = types.ModuleType(module_name)
        fake.evaluate_frozen_publication_antigravity = lambda: {"status": "ok"}  # type: ignore[attr-defined]
        stream = io.StringIO()
        with patch.object(cli, "inspect_publication_antigravity", side_effect=ValueError("publication invalid")), patch.object(cli.importlib, "import_module", side_effect=lambda name: imported.append(name) or fake), contextlib.redirect_stdout(stream):
            self.assertEqual(cli.main_antigravity(["evaluate"]), 2)
        self.assertEqual(imported, [])
        self.assertEqual(json.loads(stream.getvalue()), {"error": "publication invalid"})

    def test_wrapper_sets_threads_before_first_numpy_import_antigravity(self) -> None:
        """Importing NumPy before the wrapper pins threads would leak multi-threaded numerical state."""
        wrapper = Path(__file__).resolve().parents[3] / "scratch" / "run_exp444_train_only_stability_precision_gate_antigravity.py"
        completed = subprocess.run([sys.executable, str(wrapper), "thread-probe-antigravity"], check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(completed.stdout), {name: "1" for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "NUMBA_NUM_THREADS",
        )})

    def test_wrapper_error_is_one_canonical_json_object_antigravity(self) -> None:
        """Traceback output would corrupt machine-readable operator evidence."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import cli_antigravity as cli

        stream = io.StringIO()
        with patch.object(cli, "_dispatch_antigravity", side_effect=ValueError("broken")), contextlib.redirect_stdout(stream):
            self.assertEqual(cli.main_antigravity(["inspect"]), 2)
        self.assertEqual(json.loads(stream.getvalue()), {"error": "broken"})
