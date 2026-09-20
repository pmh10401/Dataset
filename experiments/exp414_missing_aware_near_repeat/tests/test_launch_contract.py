import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest


REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "scratch" / "launch_exp414_full.sh"
HELPER = REPO_ROOT / "scratch" / "run_exp414_launchd_once.sh"


class LaunchContractTests(unittest.TestCase):
    def test_print_command_has_fixed_one_shot_execution_contract(self) -> None:
        result = subprocess.run(
            ["bash", str(LAUNCHER), "--print-command"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in (
            "NUMBA_NUM_THREADS=12",
            "OMP_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "VECLIB_MAXIMUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
            "com.openai.exp414.full",
            "/private/tmp/exp414-full.stdout.log",
            "/private/tmp/exp414-full.stderr.log",
            "/Users/mac/Documents/Dataset/univariate_ts.db",
            "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/"
            "exp404_prediction_manifest.json",
            "/Users/mac/Documents/Dataset/"
            "exp411_near_repeat_qualification.checkpoint",
            "/Users/mac/Documents/Dataset/"
            "exp414_missing_aware_near_repeat.checkpoint",
            "/Users/mac/Documents/Dataset/exp414_acceleration_benchmark.json",
            "/Users/mac/Documents/Dataset/exp414_missing_aware_near_repeat",
            "run_exp414_launchd_once.sh",
        ):
            self.assertIn(expected, result.stdout)
        self.assertLess(
            result.stdout.index("run_exp414_launchd_once.sh"),
            result.stdout.index("run_exp414_missing_aware_near_repeat.py"),
        )
        self.assertNotIn("submitted", result.stdout.lower())

    def test_helper_removes_only_its_job_after_child_success(self) -> None:
        result, calls = self._run_helper("/usr/bin/true")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, "remove com.openai.exp414.full\n")

    def test_helper_removes_only_its_job_after_child_failure(self) -> None:
        result, calls = self._run_helper("/usr/bin/false")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, "remove com.openai.exp414.full\n")

    def test_unknown_launcher_argument_fails_without_submission(self) -> None:
        result = subprocess.run(
            ["bash", str(LAUNCHER), "--unknown"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr.lower())

    def test_print_command_rejects_trailing_arguments(self) -> None:
        result = subprocess.run(
            ["bash", str(LAUNCHER), "--print-command", "unexpected"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("usage:", result.stderr.lower())

    def _run_helper(self, child: str) -> tuple[subprocess.CompletedProcess[str], str]:
        with TemporaryDirectory() as raw:
            root = Path(raw)
            launchctl = root / "launchctl"
            log = root / "launchctl.log"
            launchctl.write_text(
                "#!/bin/bash\n"
                'printf "%s\\n" "$*" >> "$EXP414_LAUNCHCTL_LOG"\n',
                encoding="utf-8",
            )
            launchctl.chmod(0o755)
            environment = os.environ.copy()
            environment["PATH"] = f"{root}:/usr/bin:/bin"
            environment["EXP414_LAUNCHCTL_LOG"] = str(log)
            result = subprocess.run(
                ["bash", str(HELPER), child],
                text=True,
                capture_output=True,
                check=False,
                env=environment,
            )
            calls = log.read_text(encoding="utf-8") if log.exists() else ""
        return result, calls


if __name__ == "__main__":
    unittest.main()
