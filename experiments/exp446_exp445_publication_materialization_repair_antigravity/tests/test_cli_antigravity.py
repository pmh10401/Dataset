"""Three-command, source-free Exp446 operator CLI tests."""
from __future__ import annotations

import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
from unittest import mock


@dataclass(frozen=True)
class _MappingProxyCliOutputAntigravity:
    """Representative frozen lifecycle output: mapping proxies must serialize without copying."""

    nested: Mapping[str, object]


class Exp446CliAntigravityTests(unittest.TestCase):
    def test_new_result_serializes_nested_mappingproxy_dataclass_antigravity(self) -> None:
        """Post-publication new output must never deepcopy a frozen mapping proxy."""
        from experiments.exp446_exp445_publication_materialization_repair_antigravity import cli_antigravity as cli

        result = {
            "runtime": _MappingProxyCliOutputAntigravity(MappingProxyType({"z": 2, "a": ("x", 1)})),
            "publication": MappingProxyType({"path": Path("/frozen/publication"), "marker": "abc"}),
        }
        output = io.StringIO()
        with mock.patch.object(cli, "_dispatch_antigravity", return_value=result), redirect_stdout(output):
            self.assertEqual(cli.main_antigravity(("new", "--reviewed-commit", "a" * 40)), 0)
        self.assertEqual(json.loads(output.getvalue()), {
            "command": "new",
            "result": {"publication": {"marker": "abc", "path": "/frozen/publication"}, "runtime": {"nested": {"a": ["x", 1], "z": 2}}},
        })

    def test_inspect_result_serializes_nested_mappingproxy_dataclass_antigravity(self) -> None:
        """Source-free inspect output must use the same frozen-safe converter without inspecting artifacts."""
        from experiments.exp446_exp445_publication_materialization_repair_antigravity import cli_antigravity as cli

        result = _MappingProxyCliOutputAntigravity(MappingProxyType({"b": True, "a": None}))
        output = io.StringIO()
        with mock.patch.object(cli, "_dispatch_antigravity", return_value=result), redirect_stdout(output):
            self.assertEqual(cli.main_antigravity(("inspect",)), 0)
        self.assertEqual(json.loads(output.getvalue()), {
            "command": "inspect",
            "result": {"nested": {"a": None, "b": True}},
        })

    def test_parser_accepts_only_three_closed_commands_antigravity(self) -> None:
        """Adding an operator override or historical command must fail parsing."""
        from experiments.exp446_exp445_publication_materialization_repair_antigravity.cli_antigravity import build_parser_antigravity

        parser = build_parser_antigravity()
        self.assertEqual(parser.parse_args(("readiness", "--reviewed-commit", "a" * 40)).command, "readiness")
        self.assertEqual(parser.parse_args(("new", "--reviewed-commit", "b" * 40)).command, "new")
        self.assertEqual(parser.parse_args(("inspect",)).command, "inspect")
        for argv in (("resume",), ("retry",), ("evaluate",), ("inspect", "--root", "/tmp"), ("new", "--backend", "cpu"), ("new", "--reviewed-c", "a" * 40), ("inspect", "extra")):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                parser.parse_args(argv)

    def test_inspect_subprocess_does_not_import_sqlite_evaluator_or_labels_antigravity(self) -> None:
        """The source-free inspect dispatch seam must not import SQLite, labels, or official artifacts."""
        repository = Path(__file__).resolve().parents[3]
        program = """
import sys
from unittest import mock
from experiments.exp446_exp445_publication_materialization_repair_antigravity import cli_antigravity as cli
assert 'sqlite3' not in sys.modules
assert 'experiments.exp444_train_only_stability_precision_gate_antigravity.evaluation_antigravity' not in sys.modules
with mock.patch.object(cli, 'inspect_prediction_antigravity', return_value={'frozen': 'inspect'}):
    assert cli.main_antigravity(('inspect',)) == 0
bad = [name for name in sys.modules if 'sqlite' in name.lower() or 'evaluation_antigravity' in name or 'label' in name.lower()]
assert not bad, bad
"""
        completed = subprocess.run((sys.executable, "-c", program), cwd=repository, text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)

    def test_launcher_sets_threads_without_hidden_operator_command_antigravity(self) -> None:
        """The launcher must pin threads at import time while every argv stays a CLI argv."""
        launcher = Path(__file__).resolve().parents[3] / "scratch" / "run_exp446_exp445_publication_materialization_repair_antigravity.py"
        program = f"""
import importlib.util
import json
import os
import sys
sys.argv = ['launcher', 'inspect']
spec = importlib.util.spec_from_file_location('exp446_launcher_antigravity', {str(launcher)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps({{name: os.environ[name] for name in module._THREAD_ENV_NAMES_ANTIGRAVITY}}, sort_keys=True))
"""
        completed = subprocess.run((sys.executable, "-c", program), cwd=launcher.parent.parent, text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(set(__import__("json").loads(completed.stdout).values()), {"1"})
