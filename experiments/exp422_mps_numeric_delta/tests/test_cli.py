"""Guarded Exp422 CLI behavior without database or accelerator access."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.exp422_mps_numeric_delta.tests.test_manifest import _capability, _core
import scratch.run_exp422_mps_numeric_delta as cli


class Exp422CompletionGuardTests(unittest.TestCase):
    def test_live_exp419_pid_is_rejected_before_exit_status(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pidfile = root / "exp419.pid"
            pidfile.write_text(str(os.getpid()), encoding="ascii")

            with self.assertRaisesRegex(ValueError, "Exp419 is still running"):
                cli.require_exp419_finished(pidfile, root / "exp419.exit")

    def test_missing_or_nonzero_exp419_exit_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pidfile = root / "exp419.pid"
            exitfile = root / "exp419.exit"
            with self.assertRaisesRegex(ValueError, "exit file"):
                cli.require_exp419_finished(pidfile, exitfile)
            exitfile.write_text("1\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "exit code"):
                cli.require_exp419_finished(pidfile, exitfile)

    def test_absent_pid_and_zero_exit_means_exp419_finished(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            exitfile = root / "exp419.exit"
            exitfile.write_text("0\n", encoding="ascii")
            cli.require_exp419_finished(root / "exp419.pid", exitfile)


class Exp422CliTests(unittest.TestCase):
    def test_official_exp419_and_nonempty_output_roots_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Exp419"):
            cli.validate_output_root(
                Path("/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_official")
            )
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "attempt"
            root.mkdir()
            (root / "prior.json").write_text("{}", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "empty|fresh"):
                cli.validate_output_root(root)

    def test_output_root_cannot_hide_exp419_behind_a_symlinked_parent(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            official = root / "exp419_official"
            official.mkdir()
            alias = root / "diagnostic-alias"
            alias.symlink_to(official, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "Exp419"):
                cli.validate_output_root(alias / "exp422-attempt")

    def test_parser_requires_exact_arguments_and_bounds_repeats(self) -> None:
        required = [
            "--database", "/tmp/db.sqlite",
            "--output-root", "/tmp/exp422-attempt",
            "--exp419-pidfile", "/tmp/exp419.pid",
            "--exp419-exitfile", "/tmp/exp419.exit",
        ]
        self.assertEqual(cli.parse_args([*required, "--repeats", "7"]).repeats, 7)
        with self.assertRaises(SystemExit):
            cli.parse_args([*required, "--repeats", "0"])
        with self.assertRaises(SystemExit):
            cli.parse_args([*required, "--repeats", "8"])
        with self.assertRaises(SystemExit):
            cli.parse_args(["--output-root", "/tmp/exp422-attempt"])

    def test_failed_core_skips_score_and_still_writes_one_manifest(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "exp422-attempt"
            database = root / "normal-train.sqlite"
            database.write_bytes(b"fixture boundary only")
            exitfile = root / "exp419.exit"
            exitfile.write_text("0\n", encoding="ascii")
            argv = [
                "--database", str(database),
                "--output-root", str(output),
                "--exp419-pidfile", str(root / "exp419.pid"),
                "--exp419-exitfile", str(exitfile),
                "--repeats", "1",
            ]
            with patch.object(cli, "detect_capability", return_value=_capability()), patch.object(
                cli, "preflight_database", return_value="ok"
            ), patch.object(cli, "verify_source_manifest", return_value="b" * 64), patch.object(
                cli, "source_manifest_git_commit", return_value="c" * 40
            ), patch.object(
                cli, "current_git_commit", return_value="a" * 40
            ), patch.object(cli, "run_core_parity", return_value=_core()), patch.object(
                cli, "select_train_sample", side_effect=AssertionError("score selector must be skipped")
            ), patch.object(
                cli, "run_score_parity", side_effect=AssertionError("score stage must be skipped")
            ):
                returncode = cli.main(argv)

            self.assertEqual(returncode, 0)
            payload = json.loads((output / "manifest.json").read_bytes())
            self.assertEqual(payload["state"], "failed")
            self.assertIsNone(payload["score_stage"])
            self.assertEqual(list(output.iterdir()), [output / "manifest.json"])


if __name__ == "__main__":
    unittest.main()
