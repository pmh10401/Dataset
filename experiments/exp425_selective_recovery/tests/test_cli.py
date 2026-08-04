"""CLI mode and read-only inspection tests for Exp425."""

from __future__ import annotations

import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from pathlib import Path


class Exp425CliTests(unittest.TestCase):
    def test_cli_requires_one_of_the_explicit_modes(self) -> None:
        root = Path(__file__).resolve().parents[3]
        entrypoint = root / "scratch" / "run_exp425_selective_recovery.py"
        result = subprocess.run([sys.executable, str(entrypoint)], cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("new", result.stderr)

    def test_new_existing_root_fails_before_launch_factory_import(self) -> None:
        root = Path(__file__).resolve().parents[3]
        entrypoint = root / "scratch" / "run_exp425_selective_recovery.py"
        with TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            result = subprocess.run(
                [sys.executable, str(entrypoint), "new", "--output-root", str(output),
                 "--checkpoint-root", str(Path(directory) / "checkpoints"),
                 "--launch-factory", "does_not_exist:factory"],
                cwd=root, capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn("already exists", result.stderr)
        self.assertNotIn("does_not_exist", result.stderr)
