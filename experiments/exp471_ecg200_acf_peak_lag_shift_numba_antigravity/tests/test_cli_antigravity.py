from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


class Exp471CliTestAntigravity(unittest.TestCase):
    def test_cli_rejects_official_surface_antigravity(self) -> None:
        """Break caught: an unapproved official/routing surface reaches the closed CLI."""
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.cli_antigravity import parser_antigravity

        parser = parser_antigravity()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["official"])

    def test_parser_has_only_required_closed_subcommands_antigravity(self) -> None:
        """Break caught: an alias, optional command, or unknown flag reaches the CLI."""
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.cli_antigravity import (
            PRIVATE_RUN1_ANTIGRAVITY,
            PRIVATE_RUN2_ANTIGRAVITY,
            parser_antigravity,
        )

        parser = parser_antigravity()
        actions = [action for action in parser._actions if action.dest == "command"]
        self.assertEqual(len(actions), 1)
        self.assertEqual(tuple(actions[0].choices), ("evaluate", "inspect"))
        evaluate = parser.parse_args(("evaluate", "--root", str(PRIVATE_RUN1_ANTIGRAVITY), "--commit", "a" * 40))
        inspect = parser.parse_args(("inspect", "--root", str(PRIVATE_RUN2_ANTIGRAVITY)))
        self.assertEqual(
            (evaluate.command, evaluate.root, evaluate.commit),
            ("evaluate", PRIVATE_RUN1_ANTIGRAVITY, "a" * 40),
        )
        self.assertEqual((inspect.command, inspect.root), ("inspect", PRIVATE_RUN2_ANTIGRAVITY))
        with self.assertRaises(SystemExit):
            parser.parse_args(("evaluate", "--root", str(PRIVATE_RUN1_ANTIGRAVITY), "--commit", "A" * 40))
        with self.assertRaises(SystemExit):
            parser.parse_args(("inspect", "--root", str(PRIVATE_RUN1_ANTIGRAVITY), "--unknown"))
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(("inspect", "--root", ""))

    def test_launcher_help_from_outside_repo_exposes_only_closed_commands_antigravity(self) -> None:
        """Break caught: direct scratch execution cannot import the project from another working directory."""
        launcher = Path(__file__).resolve().parents[3] / (
            "scratch/run_exp471_ecg200_acf_peak_lag_shift_numba_antigravity.py"
        )
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                (sys.executable, str(launcher), "--help"), cwd=directory,
                check=False, capture_output=True, text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("evaluate", completed.stdout)
        self.assertIn("inspect", completed.stdout)
        for forbidden in ("official", "route", "promote", "benchmark"):
            self.assertNotIn(forbidden, completed.stdout)

    def test_main_emits_one_strict_json_object_without_nan_antigravity(self) -> None:
        """Break caught: CLI adds prose, permits NaN, or dispatches a different closed command."""
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity import cli_antigravity

        payload = {"decision": {"passed": True}, "hashes": {}, "root": "x"}
        root = Path("x")
        with mock.patch.object(cli_antigravity, "_PRIVATE_ROOTS_ANTIGRAVITY", (root,)), \
             mock.patch.object(cli_antigravity, "_require_clean_reviewed_commit_antigravity"), \
             mock.patch.object(cli_antigravity, "evaluate_pipeline_antigravity", return_value=payload):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = cli_antigravity.main_antigravity(("evaluate", "--root", "x", "--commit", "a" * 40))
        self.assertEqual(code, 0)
        raw = stdout.getvalue()
        self.assertTrue(raw.endswith("\n"))
        self.assertNotIn("NaN", raw)
        self.assertEqual(json.loads(raw), payload)

    def test_inspect_command_uses_temporary_publication_only_antigravity(self) -> None:
        """Break caught: inspect invokes the evaluator instead of stored-evidence replay."""
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity import cli_antigravity

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "publication_antigravity"
            root.mkdir()
            inspected = mock.Mock()
            inspected.decision = mock.Mock(passed=False, eligible_for_private_reproducibility=False, failure_reasons=("gate",))
            inspected.hashes = {"context_antigravity.json": "b" * 64}
            with mock.patch.object(cli_antigravity, "_PRIVATE_ROOTS_ANTIGRAVITY", (root,)), \
                 mock.patch.object(cli_antigravity, "inspect_result_antigravity", return_value=inspected) as inspect_result:
                with mock.patch.object(cli_antigravity, "evaluate_pipeline_antigravity") as evaluate:
                    with mock.patch.object(cli_antigravity, "_load_frozen_train_antigravity") as loader:
                        stdout = io.StringIO()
                        with contextlib.redirect_stdout(stdout):
                            code = cli_antigravity.main_antigravity(("inspect", "--root", str(root)))
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["root"], str(root))
            inspect_result.assert_called_once_with(root)
            evaluate.assert_not_called()
            loader.assert_not_called()

    def test_cli_git_checks_use_its_exact_repository_root_antigravity(self) -> None:
        """Break caught: git provenance changes when a caller starts the CLI from another worktree."""
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity import cli_antigravity

        root = Path(cli_antigravity.__file__).resolve().parents[2]
        responses = (
            subprocess.CompletedProcess(("git",), 0, stdout="a" * 40 + "\n", stderr=""),
            subprocess.CompletedProcess(("git",), 0, stdout="", stderr=""),
        )
        with mock.patch.object(cli_antigravity.subprocess, "run", side_effect=responses) as run:
            cli_antigravity._require_clean_reviewed_commit_antigravity("a" * 40)
        self.assertEqual(run.call_count, 2)
        self.assertTrue(all(call.kwargs["cwd"] == root for call in run.call_args_list))

    def test_cli_git_checks_reject_mismatched_and_dirty_reviewed_commit_antigravity(self) -> None:
        """Break caught: evaluate proceeds from an unreviewed HEAD or dirty worktree."""
        from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity import cli_antigravity

        mismatch = subprocess.CompletedProcess(("git",), 0, stdout="b" * 40 + "\n", stderr="")
        with mock.patch.object(cli_antigravity.subprocess, "run", return_value=mismatch) as run:
            with self.assertRaisesRegex(ValueError, "current HEAD"):
                cli_antigravity._require_clean_reviewed_commit_antigravity("a" * 40)
        self.assertEqual(run.call_count, 1)

        responses = (
            subprocess.CompletedProcess(("git",), 0, stdout="a" * 40 + "\n", stderr=""),
            subprocess.CompletedProcess(("git",), 0, stdout=" M file_antigravity.py\n", stderr=""),
        )
        with mock.patch.object(cli_antigravity.subprocess, "run", side_effect=responses):
            with self.assertRaisesRegex(ValueError, "clean worktree"):
                cli_antigravity._require_clean_reviewed_commit_antigravity("a" * 40)


if __name__ == "__main__":
    unittest.main()
