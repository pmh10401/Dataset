from pathlib import Path
import subprocess
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scratch" / "launch_exp411_full.sh"


class LaunchContractTests(unittest.TestCase):
    def test_print_command_has_fixed_thread_and_resume_contract(self) -> None:
        result = subprocess.run(
            ["bash", str(SCRIPT), "--print-command"],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in (
            "NUMBA_NUM_THREADS=12",
            "OMP_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "--full-qualification",
            "--checkpoint-dir",
            "--benchmark-report",
            "/private/tmp/exp411-full.stdout.log",
            "/private/tmp/exp411-full.stderr.log",
        ):
            self.assertIn(expected, result.stdout)
        self.assertNotIn("submitted", result.stdout.lower())

    def test_unknown_launcher_argument_fails_without_submission(self) -> None:
        result = subprocess.run(
            ["bash", str(SCRIPT), "--unknown"],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
