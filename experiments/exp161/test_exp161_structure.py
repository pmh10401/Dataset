from __future__ import annotations

import unittest
import inspect
import subprocess
import sys
from pathlib import Path

# Paths to verify
PROJECT_DIR = Path("/Users/minho/Documents/timesries project")
EXP_DIR = PROJECT_DIR / "experiments" / "exp161"
DOCS_DIR = PROJECT_DIR / "docs"


class TestExp161Structure(unittest.TestCase):

    def test_file_existence(self):
        # Verify the structure has all required source files
        required_src_files = [
            EXP_DIR / "__init__.py",
            EXP_DIR / "alternative_source_audit_core.py",
            EXP_DIR / "run_experiment_161_alternative_source_audit.py",
            EXP_DIR / "test_exp161_alternative_source_audit.py",
            EXP_DIR / "test_exp161_structure.py",
        ]
        for path in required_src_files:
            self.assertTrue(path.exists(), f"Missing required file: {path}")

    def test_docs_existence(self):
        # Verify required documentation files exist
        required_docs = [
            DOCS_DIR / "EXP161_PREREGISTRATION_KO.md",
            DOCS_DIR / "EXP161_RUNBOOK_KO.md",
        ]
        for path in required_docs:
            self.assertTrue(path.exists(), f"Missing required doc: {path}")

    def test_core_exports(self):
        # Verify core module exposes required functions
        import experiments.exp161.alternative_source_audit_core as core

        self.assertTrue(hasattr(core, "is_target"))
        self.assertTrue(hasattr(core, "detect_computed_exp99"))
        self.assertTrue(hasattr(core, "detect_computed_exp101"))
        self.assertTrue(hasattr(core, "detect_computed_exp46"))
        self.assertTrue(hasattr(core, "is_policy_eligible"))
        self.assertTrue(hasattr(core, "calculate_legacy_index_overlap"))
        self.assertTrue(hasattr(core, "SOURCE_DISPLAY_NAMES"))
        self.assertTrue(hasattr(core, "EXP46_DATASETS"))

    def test_full_audit_accepts_separate_output_directory(self):
        from experiments.exp161.run_experiment_161_alternative_source_audit import run_audit

        self.assertIn("output_dir", inspect.signature(run_audit).parameters)

    def test_runner_is_directly_executable(self):
        result = subprocess.run(
            [sys.executable, str(EXP_DIR / "run_experiment_161_alternative_source_audit.py"), "--help"],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
