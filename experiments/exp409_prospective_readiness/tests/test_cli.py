from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ENTRYPOINT = (
    Path(__file__).resolve().parents[3]
    / "scratch"
    / "run_exp409_prospective_readiness.py"
)


class CliTests(unittest.TestCase):
    def test_full_run_flag_does_not_exist(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ENTRYPOINT), "--help"],
            text=True,
            capture_output=True,
        )

        self.assertEqual(completed.returncode, 0)
        self.assertNotIn("--stage", completed.stdout)
        self.assertNotIn("--full", completed.stdout)
        self.assertIn("--representative-smoke", completed.stdout)
        self.assertIn("--dataset-limit", completed.stdout)

    def test_output_dir_must_not_exist(self) -> None:
        with TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ENTRYPOINT),
                    "--representative-smoke",
                    "--dataset-limit",
                    "4",
                    "--output-dir",
                    directory,
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("must not exist", completed.stderr)

    def test_limit_below_four_is_rejected_before_execution(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "new-output"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ENTRYPOINT),
                    "--representative-smoke",
                    "--dataset-limit",
                    "3",
                    "--output-dir",
                    str(output),
                ],
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("4..20", completed.stderr)


if __name__ == "__main__":
    unittest.main()
