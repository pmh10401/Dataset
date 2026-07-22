from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class ScratchWrapperTest(unittest.TestCase):
    def test_wrapper_imports_project_from_external_working_directory(self) -> None:
        root = Path(__file__).resolve().parents[3]
        script = root / "scratch" / "run_exp289_label_consistency_audit.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd="/tmp",
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--output-root", completed.stdout)


if __name__ == "__main__":
    unittest.main()
