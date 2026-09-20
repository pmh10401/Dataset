"""Tests for the separately preregistered Exp422-IR runner boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import scratch.run_exp422_interrupted419_diagnostic as cli


class Exp422InterruptedGuardTests(unittest.TestCase):
    def test_only_dead_pid_and_exit_143_are_accepted(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pidfile = root / "exp419.pid"
            exitfile = root / "exp419.exit"
            pidfile.write_text("999999999\n", encoding="ascii")
            exitfile.write_text("143\n", encoding="ascii")
            with patch.object(cli, "_pid_is_alive", return_value=False):
                self.assertEqual(cli.require_exp419_interrupted(pidfile, exitfile), 143)

    def test_live_pid_and_other_exit_codes_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pidfile = root / "exp419.pid"
            exitfile = root / "exp419.exit"
            pidfile.write_text(str(os.getpid()), encoding="ascii")
            exitfile.write_text("143\n", encoding="ascii")
            with patch.object(cli, "_pid_is_alive", return_value=True):
                with self.assertRaisesRegex(ValueError, "still running"):
                    cli.require_exp419_interrupted(pidfile, exitfile)
            with patch.object(cli, "_pid_is_alive", return_value=False):
                exitfile.write_text("0\n", encoding="ascii")
                with self.assertRaisesRegex(ValueError, "143"):
                    cli.require_exp419_interrupted(pidfile, exitfile)


class ProgressLoggerTests(unittest.TestCase):
    def test_progress_events_are_durable_jsonl(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "progress.jsonl"
            logger = cli.ProgressLogger(path)
            logger.emit("core_started", completed=0, total=60)
            logger.emit("core_finished", completed=60, total=60, state="passed")
            logger.close()
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["event"] for row in rows], ["core_started", "core_finished"])
            self.assertEqual(rows[-1]["completed"], 60)
            self.assertEqual(rows[-1]["total"], 60)
            self.assertTrue(rows[-1]["pid"] > 0)


if __name__ == "__main__":
    unittest.main()
