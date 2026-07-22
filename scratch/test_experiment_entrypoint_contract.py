from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "scratch"


class ExperimentEntrypointContractTests(unittest.TestCase):
    def test_every_wrapper_importing_experiments_bootstraps_before_import(self):
        missing = []
        for path in sorted(SCRATCH.glob("run_*.py")):
            text = path.read_text(encoding="utf-8")
            package_positions = [
                position for marker in ("from experiments", "import experiments")
                if (position := text.find(marker)) >= 0
            ]
            if not package_positions:
                continue
            bootstrap = text.find("sys.path.insert")
            if bootstrap < 0 or bootstrap > min(package_positions):
                missing.append(path.name)
        self.assertEqual(missing, [], f"wrappers missing repository bootstrap: {missing}")

    def test_sequential_runner_child_environment_contains_repository_root(self):
        from run_rank_experiments_sequential import build_child_environment

        env = build_child_environment({"PYTHONPATH": "/tmp/example"}, worker_count="7")
        paths = env["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(paths[0], str(ROOT))
        self.assertIn("/tmp/example", paths)
        self.assertEqual(env["RANK_EXPERIMENT_WORKERS"], "7")
        result = subprocess.run(
            [sys.executable, "-c", "import experiments"],
            cwd="/tmp",
            env=env,
            text=True,
            capture_output=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_direct_wrapper_execution_can_import_project_package(self):
        result = subprocess.run(
            [sys.executable, str(SCRATCH / "run_exp265_shadow_exporter.py"), "--help"],
            cwd="/tmp",
            text=True,
            capture_output=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
