"""CLI mode and read-only inspection tests for Exp425."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class Exp425CliTests(unittest.TestCase):
    def test_cli_requires_one_of_the_explicit_modes(self) -> None:
        root = Path(__file__).resolve().parents[3]
        entrypoint = root / "scratch" / "run_exp425_selective_recovery.py"
        result = subprocess.run([sys.executable, str(entrypoint)], cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("new", result.stderr)

