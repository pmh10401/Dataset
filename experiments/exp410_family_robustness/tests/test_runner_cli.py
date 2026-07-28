import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.exp410_family_robustness.runner import run_smoke


REPO_ROOT = Path(__file__).resolve().parents[3]


class RunnerCliTests(unittest.TestCase):
    def test_cli_requires_explicit_mode_and_new_output(self) -> None:
        script = REPO_ROOT / "scratch" / "run_exp410_family_robustness.py"
        missing = subprocess.run(
            [sys.executable, str(script)],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("one of the arguments", missing.stderr)

    def test_representative_smoke_publishes_without_real_full_scan(self) -> None:
        with TemporaryDirectory() as raw:
            output = Path(raw) / "smoke"
            hashes = run_smoke(output)
            summary = json.loads(
                (output / "exp410_summary.json").read_text(encoding="utf-8")
            )
            self.assertTrue(summary["smoke"])
            self.assertEqual(
                summary["difficulty_stratification"]["status"],
                "not_evaluated",
            )
            self.assertFalse(summary["operational_policy_changed"])
            self.assertIn("MANIFEST.json", hashes)

    def test_full_runner_is_not_called_by_smoke(self) -> None:
        with TemporaryDirectory() as raw, patch(
            "experiments.exp410_family_robustness.runner.scan_dataset_profiles",
            side_effect=AssertionError("real profile scan forbidden in smoke"),
        ):
            run_smoke(Path(raw) / "smoke")


if __name__ == "__main__":
    unittest.main()
