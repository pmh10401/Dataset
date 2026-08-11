"""Closed-command and import-boundary tests for the Exp447 CLI."""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import redirect_stderr
import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


_PACKAGE_ANTIGRAVITY = "experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity"
_LAUNCHER_ANTIGRAVITY = Path(__file__).resolve().parents[3] / "scratch/run_exp447_exp446_postfreeze_performance_evaluation_antigravity.py"


@dataclass(frozen=True, slots=True)
class _ResultAntigravity:
    name: str
    count: int
    nested: tuple[object, ...]


class CliAntigravityTests(unittest.TestCase):
    def test_parser_exposes_only_closed_commands_and_exact_commit_antigravity(self) -> None:
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.cli_antigravity import build_parser_antigravity

        parser = build_parser_antigravity()
        commands = next(action for action in parser._actions if action.dest == "command")
        self.assertEqual(tuple(commands.choices), ("readiness", "evaluate", "inspect"))
        self.assertFalse(parser.allow_abbrev)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["readiness"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["readiness", "--reviewed-commit", "A" * 40])
        self.assertEqual(
            parser.parse_args(["evaluate", "--reviewed-commit", "a" * 40]).reviewed_commit,
            "a" * 40,
        )
        for forbidden in ("new", "resume", "retry"):
            with self.subTest(forbidden=forbidden), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args([forbidden])
        for forbidden in ("--root", "--hash", "--source", "--lane", "--threshold", "--budget"):
            with self.subTest(forbidden=forbidden), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["readiness", "--reviewed-commit", "a" * 40, forbidden, "override"])

    def test_dispatch_is_lazy_and_inspect_is_label_free_antigravity(self) -> None:
        # Break caught: a top-level evaluator import made --help/readiness/inspect
        # initialize the label authority before their command branch was chosen.
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity import cli_antigravity as cli

        self.assertNotIn("sqlite3", sys.modules)
        with patch.object(cli, "_load_readiness_antigravity", return_value=lambda commit: {"kind": "readiness", "commit": commit}) as readiness, patch.object(cli, "_load_evaluate_antigravity", side_effect=AssertionError("evaluation loaded")), patch.object(cli, "_load_inspect_antigravity", side_effect=AssertionError("inspect loaded")), patch.object(cli, "_print_result_antigravity"):
            self.assertEqual(cli.main_antigravity(["readiness", "--reviewed-commit", "a" * 40]), 0)
        readiness.assert_called_once_with()

        with patch.object(cli, "_load_inspect_antigravity", return_value=lambda dependencies: {"kind": "inspect"}) as inspect, patch.object(cli, "_inspect_dependencies_antigravity", return_value=object()), patch.object(cli, "_load_readiness_antigravity", side_effect=AssertionError("readiness loaded")), patch.object(cli, "_load_evaluate_antigravity", side_effect=AssertionError("evaluation loaded")), patch.object(cli, "_print_result_antigravity"):
            self.assertEqual(cli.main_antigravity(["inspect"]), 0)
        inspect.assert_called_once_with()
        self.assertNotIn("sqlite3", sys.modules)

    def test_evaluate_rechecks_clean_commit_before_evaluator_import_antigravity(self) -> None:
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity import cli_antigravity as cli

        events: list[str] = []
        def clean(commit: str) -> None:
            events.append(f"clean:{commit}")
        def evaluator(commit: str) -> dict[str, str]:
            events.append(f"evaluate:{commit}")
            return {"kind": "evaluation"}
        with patch.object(cli, "_load_clean_review_antigravity", return_value=clean), patch.object(cli, "_load_evaluate_antigravity", return_value=evaluator), patch.object(cli, "_print_result_antigravity"):
            self.assertEqual(cli.main_antigravity(["evaluate", "--reviewed-commit", "a" * 40]), 0)
        self.assertEqual(events, [f"clean:{'a' * 40}", f"evaluate:{'a' * 40}"])

    def test_serializer_uses_slots_dataclass_fields_without_asdict_antigravity(self) -> None:
        from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity import cli_antigravity as cli

        result = _ResultAntigravity("result", 2, ({"inner": (3,)},))
        self.assertEqual(cli._json_safe_antigravity(result), {"name": "result", "count": 2, "nested": [{"inner": [3]}]})
        self.assertNotIn("asdict", Path(cli.__file__).read_text(encoding="utf-8"))

    def test_launcher_runs_help_outside_repository_and_pins_threads_antigravity(self) -> None:
        # A subprocess makes the import order observable: launcher environment
        # assignment must precede package import and must not depend on cwd.
        with subprocess.Popen(
            [sys.executable, str(_LAUNCHER_ANTIGRAVITY), "--help"],
            cwd="/tmp",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "OMP_NUM_THREADS": "999", "OPENBLAS_NUM_THREADS": "999", "MKL_NUM_THREADS": "999", "VECLIB_MAXIMUM_THREADS": "999", "NUMEXPR_NUM_THREADS": "999", "BLIS_NUM_THREADS": "999", "NUMBA_NUM_THREADS": "999"},
        ) as process:
            stdout, stderr = process.communicate(timeout=20)
        self.assertEqual(process.returncode, 0, stderr)
        self.assertIn("readiness", stdout)
        self.assertIn("evaluate", stdout)
        self.assertIn("inspect", stdout)
        probe = (
            "import json, os, runpy; "
            f"runpy.run_path({str(_LAUNCHER_ANTIGRAVITY)!r}); "
            "print(json.dumps({key: os.environ[key] for key in "
            "('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',"
            "'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','NUMBA_NUM_THREADS')}))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe], cwd="/tmp", check=False,
            capture_output=True, text=True,
            env={**os.environ, "OMP_NUM_THREADS": "999", "OPENBLAS_NUM_THREADS": "999", "MKL_NUM_THREADS": "999", "VECLIB_MAXIMUM_THREADS": "999", "NUMEXPR_NUM_THREADS": "999", "BLIS_NUM_THREADS": "999", "NUMBA_NUM_THREADS": "999"},
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), '{"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1"}')

    def test_real_inspect_loader_subprocess_keeps_label_authority_unloaded_antigravity(self) -> None:
        # This must be an unshared interpreter. An earlier test may have loaded
        # evaluator/source modules, masking an eager import in the CLI loader.
        probe = (
            "import sys; "
            "from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.cli_antigravity "
            "import _load_inspect_antigravity; "
            "loaded = _load_inspect_antigravity(); "
            "assert callable(loaded); "
            "forbidden = ('sqlite3', "
            "'experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.sources_antigravity', "
            "'experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.evaluation_antigravity'); "
            "assert not any(name in sys.modules for name in forbidden), sorted(name for name in forbidden if name in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe], cwd=Path(__file__).resolve().parents[3],
            check=False, capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_real_inspect_dispatch_subprocess_keeps_label_authority_unloaded_antigravity(self) -> None:
        # The actual inspect branch constructs its dependency adapter after the
        # replay loader. The absent official root safely stops replay before it
        # can read a publication; this still exposes forbidden eager imports.
        probe = (
            "import sys\n"
            "from experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.cli_antigravity import main_antigravity\n"
            "try:\n main_antigravity(['inspect'])\n"
            "except ValueError:\n pass\n"
            "else:\n raise AssertionError('temporary inspect must not succeed')\n"
            "forbidden = ('sqlite3', "
            "'experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.sources_antigravity', "
            "'experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.evaluation_antigravity'); "
            "assert not any(name in sys.modules for name in forbidden), sorted(name for name in forbidden if name in sys.modules); "
            "assert 'experiments.exp447_exp446_postfreeze_performance_evaluation_antigravity.publication_identity_antigravity' in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe], cwd=Path(__file__).resolve().parents[3],
            check=False, capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
