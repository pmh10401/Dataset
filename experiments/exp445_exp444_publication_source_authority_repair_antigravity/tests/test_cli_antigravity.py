"""Closed operator-surface tests for the Exp445 repair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch


class Exp445CliAntigravityTests(unittest.TestCase):
    def test_parser_accepts_only_four_fixed_commands_antigravity(self) -> None:
        """Any override would bypass the reviewed Exp445 identity and authority."""
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import cli_antigravity as cli

        parser = cli.build_parser_antigravity()
        self.assertEqual(
            set(parser._subparsers._group_actions[0].choices),
            {"readiness", "new", "inspect", "evaluate"},
        )
        self.assertEqual(
            parser.parse_args(["readiness", "--reviewed-commit", "a" * 40]).reviewed_commit,
            "a" * 40,
        )
        self.assertEqual(
            parser.parse_args(["new", "--reviewed-commit", "a" * 40]).reviewed_commit,
            "a" * 40,
        )
        for command, flag in (
            ("resume", "--reviewed-commit"),
            ("new", "--root"),
            ("new", "--path"),
            ("new", "--profile"),
            ("new", "--source"),
            ("new", "--seed"),
            ("new", "--mask"),
            ("new", "--threshold"),
            ("new", "--workers"),
            ("new", "--backend"),
            ("evaluate", "--evaluation"),
        ):
            with self.subTest(command=command, flag=flag), self.assertRaises(SystemExit):
                parser.parse_args([command, flag, "x"])

    def test_evaluate_inspects_twice_before_lazy_evaluator_import_antigravity(self) -> None:
        """A failed frozen publication must never make label-capable code importable."""
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import cli_antigravity as cli

        events: list[str] = []
        publication = object()
        module_name = "experiments.exp444_train_only_stability_precision_gate_antigravity.evaluation_antigravity"
        evaluator = types.ModuleType(module_name)
        evaluator.evaluate_exp445_frozen_publication_antigravity = lambda: events.append("evaluate") or {"status": "ok"}  # type: ignore[attr-defined]
        with (
            patch.object(cli, "inspect_prediction_antigravity", side_effect=lambda: events.append("inspect") or publication),
            patch.object(cli.importlib, "import_module", side_effect=lambda name: events.append(f"import:{name}") or evaluator),
        ):
            self.assertEqual(cli._dispatch_antigravity(argparse.Namespace(command="evaluate")), {"status": "ok"})
        self.assertEqual(events, ["inspect", "inspect", f"import:{module_name}", "evaluate"])

    def test_changed_second_inspection_blocks_evaluator_import_antigravity(self) -> None:
        """The lazy label-capable import is forbidden when frozen inspection disagrees."""
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import cli_antigravity as cli

        class PublicationAntigravity:
            def __init__(self, digest: str) -> None:
                self.publication_root = Path("/tmp/exp445-publication-antigravity")
                self.manifest = {"prediction_sha256": digest}

        imported: list[str] = []
        with (
            patch.object(cli, "inspect_prediction_antigravity", side_effect=(PublicationAntigravity("a" * 64), PublicationAntigravity("b" * 64))),
            patch.object(cli.importlib, "import_module", side_effect=lambda name: imported.append(name)),
        ):
            with self.assertRaisesRegex(ValueError, "changed between independent inspections"):
                cli._dispatch_antigravity(argparse.Namespace(command="evaluate"))
        self.assertEqual(imported, [])

    def test_launcher_pins_inner_threads_before_numpy_capable_import_antigravity(self) -> None:
        """A late thread pin would violate the fixed CPU/11-inner-1 topology."""
        launcher = Path(__file__).resolve().parents[3] / "scratch" / "run_exp445_exp444_publication_source_authority_repair_antigravity.py"
        completed = subprocess.run(
            [sys.executable, str(launcher), "thread-probe-antigravity"],
            check=False, capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout), {
            name: "1" for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
                "NUMBA_NUM_THREADS",
            )
        })

    def test_inspect_keeps_sqlite_and_evaluator_unloaded_antigravity(self) -> None:
        """A missing fixed publication must fail before any source or label boundary opens."""
        statement = """
import sys
from experiments.exp445_exp444_publication_source_authority_repair_antigravity import cli_antigravity as cli
assert 'sqlite3' not in sys.modules
assert 'experiments.exp444_train_only_stability_precision_gate_antigravity.evaluation_antigravity' not in sys.modules
assert cli.main_antigravity(('inspect',)) == 2
assert 'sqlite3' not in sys.modules
assert 'experiments.exp444_train_only_stability_precision_gate_antigravity.evaluation_antigravity' not in sys.modules
"""
        completed = subprocess.run(
            [sys.executable, "-B", "-c", statement],
            cwd=Path(__file__).resolve().parents[3], check=False, capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
