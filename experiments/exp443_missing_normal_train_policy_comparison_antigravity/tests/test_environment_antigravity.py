"""RED tests for the exact isolated Exp443 environment evidence."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from experiments.exp443_missing_normal_train_policy_comparison_antigravity.config_antigravity import FROZEN_CONFIG
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.environment_antigravity import (
    EnvironmentEvidenceAntigravity,
    verify_environment_antigravity,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts/create_exp443_environment_antigravity.sh"
_INNER_THREAD_ENV = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)


def _tree_state(root: Path) -> tuple[tuple[object, ...], ...]:
    state: list[tuple[object, ...]] = []
    for path in sorted((root, *root.rglob("*")), key=lambda item: str(item)):
        relative = "." if path == root else str(path.relative_to(root))
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if path.is_symlink():
            body: object = ("symlink", os.readlink(path))
        elif path.is_file():
            body = ("file", path.read_bytes())
        else:
            body = ("directory",)
        state.append((relative, mode, body))
    return tuple(state)


class Exp443EnvironmentAntigravityTests(unittest.TestCase):
    def test_environment_evidence_records_exact_versions_threads_and_cpu_backend(self) -> None:
        with patch.dict(os.environ, {name: "1" for name in _INNER_THREAD_ENV}):
            evidence = verify_environment_antigravity()
        self.assertIsInstance(evidence, EnvironmentEvidenceAntigravity)
        self.assertEqual(evidence.python_version, "3.14.6")
        self.assertEqual(evidence.logical_cpu_count, 12)
        self.assertEqual(evidence.outer_workers, 11)
        self.assertEqual(evidence.inner_threads, 1)
        self.assertEqual(dict(evidence.inner_thread_env), {name: "1" for name in _INNER_THREAD_ENV})
        self.assertEqual(evidence.package_versions["aeon"], "1.5.0")
        self.assertEqual(evidence.package_versions["numpy"], "2.3.5")
        self.assertEqual(evidence.package_versions["scikit-learn"], "1.8.0")
        self.assertEqual(evidence.acceleration_decision, "inapplicable_official_cpu_transformer")
        self.assertEqual(evidence.to_payload()["lock_sha256"], FROZEN_CONFIG.environment_lock_sha256)
        self.assertEqual(EnvironmentEvidenceAntigravity.from_payload(evidence.to_payload()), evidence)

    def test_environment_evidence_rejects_wrong_lock_or_worker_contract(self) -> None:
        with patch.dict(os.environ, {name: "1" for name in _INNER_THREAD_ENV}):
            evidence = verify_environment_antigravity()
        for field_name, value in (("lock_sha256", "0" * 64), ("outer_workers", 1), ("inner_threads", 2)):
            payload = evidence.to_payload()
            payload[field_name] = value
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError):
                    EnvironmentEvidenceAntigravity.from_payload(payload)

    def test_environment_rejects_missing_or_non_one_inner_thread_env(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                verify_environment_antigravity()
        with patch.dict(os.environ, {name: "1" for name in _INNER_THREAD_ENV} | {"OMP_NUM_THREADS": "2"}):
            with self.assertRaises(ValueError):
                verify_environment_antigravity()

    def test_environment_rejects_discovered_real_threadpool_with_more_than_one_thread(self) -> None:
        pools = [{"user_api": "blas", "internal_api": "openblas", "num_threads": 2, "prefix": "lib"}]
        with patch.dict(os.environ, {name: "1" for name in _INNER_THREAD_ENV}):
            with patch("threadpoolctl.threadpool_info", return_value=pools):
                with self.assertRaises(ValueError):
                    verify_environment_antigravity()

    def test_existing_mismatching_venv_is_byte_and_mode_unchanged_on_failure(self) -> None:
        fixed_root = Path(FROZEN_CONFIG.isolated_environment_root)
        self.assertTrue(fixed_root.is_dir())
        with tempfile.TemporaryDirectory(prefix="exp443_venv_fixture_antigravity_") as temporary:
            backup = Path(temporary) / "real_venv_backup_antigravity"
            os.replace(fixed_root, backup)
            try:
                fixture = fixed_root
                (fixture / "bin").mkdir(parents=True, mode=0o755)
                fake_python = fixture / "bin/python"
                fake_python.write_text("#!/bin/sh\nprintf '%s\\n' '3.13.0'\n", encoding="utf-8")
                fake_python.chmod(0o755)
                before = _tree_state(fixture)
                result = subprocess.run(
                    ["bash", str(_SCRIPT)],
                    cwd=_REPO_ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                after = _tree_state(fixture)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("refusing", result.stderr.lower())
                self.assertEqual(before, after)
            finally:
                if fixed_root.exists():
                    for child in sorted(fixed_root.rglob("*"), reverse=True):
                        if child.is_file() or child.is_symlink():
                            child.unlink()
                        else:
                            child.rmdir()
                    fixed_root.rmdir()
                os.replace(backup, fixed_root)


if __name__ == "__main__":
    unittest.main()
