"""Closed-command and source-free import-boundary tests for the Exp448 CLI."""
from __future__ import annotations

from contextlib import redirect_stderr
from dataclasses import dataclass
import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


_PACKAGE_ANTIGRAVITY = "experiments.exp448_exp447_text_label_boundary_repair_antigravity"
_LAUNCHER_ANTIGRAVITY = Path(__file__).resolve().parents[3] / "scratch/run_exp448_exp447_text_label_boundary_repair_antigravity.py"
_THREAD_NAMES_ANTIGRAVITY = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


@dataclass(frozen=True, slots=True)
class _ResultAntigravity:
    name: str
    count: int
    nested: tuple[object, ...]


class CliAntigravityTests(unittest.TestCase):
    def test_parser_exposes_exact_closed_commands_and_rejects_overrides_antigravity(self) -> None:
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity.cli_antigravity import build_parser_antigravity

        parser = build_parser_antigravity()
        commands = next(action for action in parser._actions if action.dest == "command")
        self.assertEqual(tuple(commands.choices), ("readiness", "evaluate", "inspect"))
        self.assertFalse(parser.allow_abbrev)
        self.assertEqual(parser.parse_args(["inspect"]).command, "inspect")
        self.assertEqual(
            parser.parse_args(["readiness", "--reviewed-commit", "a" * 40]).reviewed_commit,
            "a" * 40,
        )
        rejected = (
            ("readiness",),
            ("readiness", "--reviewed-commit", "A" * 40),
            ("evaluate", "--reviewed-commit", "a" * 39),
            ("inspect", "--reviewed-commit", "a" * 40),
            ("read", "--reviewed-commit", "a" * 40),
            ("new",), ("resume",), ("retry",),
            ("readiness", "--reviewed-com", "a" * 40),
        )
        for arguments in rejected:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(arguments)
        for forbidden in ("--root", "--hash", "--source", "--lane", "--threshold", "--budget"):
            with self.subTest(forbidden=forbidden), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["readiness", "--reviewed-commit", "a" * 40, forbidden, "override"])

    def test_dispatch_keeps_readiness_and_inspect_lazy_antigravity(self) -> None:
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity import cli_antigravity as cli

        with patch.object(cli, "_load_readiness_antigravity", return_value=lambda commit: {"kind": "readiness", "commit": commit}) as readiness, patch.object(cli, "_load_evaluate_antigravity", side_effect=AssertionError("evaluation loaded")), patch.object(cli, "_load_inspect_antigravity", side_effect=AssertionError("inspect loaded")), patch.object(cli, "_print_result_antigravity"):
            self.assertEqual(cli.main_antigravity(["readiness", "--reviewed-commit", "a" * 40]), 0)
        readiness.assert_called_once_with()

        with patch.object(cli, "_load_inspect_antigravity", return_value=lambda dependencies: {"kind": "inspect"}) as inspect, patch.object(cli, "_inspect_dependencies_antigravity", return_value=object()), patch.object(cli, "_load_readiness_antigravity", side_effect=AssertionError("readiness loaded")), patch.object(cli, "_load_evaluate_antigravity", side_effect=AssertionError("evaluation loaded")), patch.object(cli, "_print_result_antigravity"):
            self.assertEqual(cli.main_antigravity(["inspect"]), 0)
        inspect.assert_called_once_with()

    def test_evaluate_checks_clean_head_before_evaluator_import_antigravity(self) -> None:
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity import cli_antigravity as cli

        events: list[str] = []

        def clean(commit: str) -> None:
            events.append(f"clean:{commit}")

        def evaluator(commit: str) -> dict[str, str]:
            events.append(f"evaluate:{commit}")
            return {"kind": "evaluation"}

        with patch.object(cli, "_load_clean_review_antigravity", return_value=clean), patch.object(cli, "_load_evaluate_antigravity", return_value=evaluator), patch.object(cli, "_print_result_antigravity"):
            self.assertEqual(cli.main_antigravity(["evaluate", "--reviewed-commit", "a" * 40]), 0)
        self.assertEqual(events, [f"clean:{'a' * 40}", f"evaluate:{'a' * 40}"])

    def test_serializer_uses_dataclass_fields_without_private_asdict_antigravity(self) -> None:
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity import cli_antigravity as cli

        self.assertEqual(
            cli._json_safe_antigravity(_ResultAntigravity("result", 2, ({"inner": (3,)},))),
            {"name": "result", "count": 2, "nested": [{"inner": [3]}]},
        )
        self.assertNotIn("asdict", Path(cli.__file__).read_text(encoding="utf-8"))

    def test_fresh_subprocess_firewall_for_help_readiness_and_inspect_antigravity(self) -> None:
        repository = Path(__file__).resolve().parents[3]
        for command in ("help", "readiness", "inspect"):
            with self.subTest(command=command):
                invocation = "parser = build_parser_antigravity(); parser.parse_args(['--help'])" if command == "help" else (
                    "try:\n main_antigravity(['readiness', '--reviewed-commit', 'a' * 40])\nexcept ValueError:\n pass" if command == "readiness" else
                    "try:\n main_antigravity(['inspect'])\nexcept ValueError:\n pass"
                )
                probe = (
                    "import sys\n"
                    f"from {_PACKAGE_ANTIGRAVITY}.cli_antigravity import build_parser_antigravity, main_antigravity\n"
                    f"{invocation}\n"
                    f"forbidden = ('sqlite3', '{_PACKAGE_ANTIGRAVITY}.evaluation_antigravity') if {command!r} == 'readiness' else ('sqlite3', '{_PACKAGE_ANTIGRAVITY}.sources_antigravity', '{_PACKAGE_ANTIGRAVITY}.evaluation_antigravity')\n"
                    "assert not any(name in sys.modules for name in forbidden), sorted(name for name in forbidden if name in sys.modules)\n"
                )
                completed = subprocess.run([sys.executable, "-c", probe], cwd=repository, check=False, capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_launcher_runs_outside_repository_and_overwrites_thread_pins_antigravity(self) -> None:
        environment = {**os.environ, **{name: "999" for name in _THREAD_NAMES_ANTIGRAVITY}}
        completed = subprocess.run([sys.executable, str(_LAUNCHER_ANTIGRAVITY), "--help"], cwd="/tmp", check=False, capture_output=True, text=True, env=environment)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("readiness", completed.stdout)
        probe = (
            "import json, os, runpy; "
            f"runpy.run_path({str(_LAUNCHER_ANTIGRAVITY)!r}); "
            f"print(json.dumps({{name: os.environ[name] for name in {_THREAD_NAMES_ANTIGRAVITY!r}}}, sort_keys=True))"
        )
        completed = subprocess.run([sys.executable, "-c", probe], cwd="/tmp", check=False, capture_output=True, text=True, env=environment)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(__import__("json").loads(completed.stdout), {name: "1" for name in _THREAD_NAMES_ANTIGRAVITY})


if __name__ == "__main__":
    unittest.main()
