from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class WrapperTest(unittest.TestCase):
    def test_wrapper_runs_from_external_directory(self) -> None:
        root = Path(__file__).resolve().parents[3]
        script = root / "scratch" / "run_exp291_final_operational_routing_freeze.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--help"], cwd="/tmp",
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--phase", completed.stdout)


if __name__ == "__main__":
    unittest.main()
