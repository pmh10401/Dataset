"""RED tests for the exact isolated Exp443 environment evidence."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest

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


class _EnvironmentControlsAntigravity:
    def __enter__(self):
        self._previous = {name: os.environ.get(name) for name in _INNER_THREAD_ENV}
        os.environ.update({name: "1" for name in _INNER_THREAD_ENV})
        return self

    def __exit__(self, *_exc):
        for name, value in self._previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


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
        with _EnvironmentControlsAntigravity():
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
        with _EnvironmentControlsAntigravity():
            evidence = verify_environment_antigravity()
        for field_name, value in (("lock_sha256", "0" * 64), ("outer_workers", 1), ("inner_threads", 2)):
            payload = evidence.to_payload()
            payload[field_name] = value
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError):
                    EnvironmentEvidenceAntigravity.from_payload(payload)

    def test_environment_rejects_missing_or_non_one_inner_thread_env(self) -> None:
        previous = {name: os.environ.get(name) for name in _INNER_THREAD_ENV}
        try:
            for name in _INNER_THREAD_ENV:
                os.environ.pop(name, None)
            with self.assertRaises(ValueError):
                verify_environment_antigravity()
            os.environ.update({name: "1" for name in _INNER_THREAD_ENV})
            os.environ["OMP_NUM_THREADS"] = "2"
            with self.assertRaises(ValueError):
                verify_environment_antigravity()
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_environment_rejects_round_tripped_real_evidence_with_more_than_one_thread(self) -> None:
        with _EnvironmentControlsAntigravity():
            evidence = verify_environment_antigravity()
        payload = evidence.to_payload()
        payload["blas_thread_pools"] = [{
            "user_api": "blas", "internal_api": "openblas",
            "num_threads": 2, "prefix": "lib",
        }]
        with self.assertRaises(ValueError):
            EnvironmentEvidenceAntigravity.from_payload(payload)

    def test_environment_rejects_forged_optional_versions_and_mps_types(self) -> None:
        with _EnvironmentControlsAntigravity():
            evidence = verify_environment_antigravity()
        for field_name, value in (
            ("pytorch_version", 1),
            ("mlx_version", 1),
            ("mps_version", 1),
            ("mps_available", 1),
        ):
            payload = evidence.to_payload()
            payload[field_name] = value
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError):
                    EnvironmentEvidenceAntigravity.from_payload(payload)

    def test_existing_mismatching_fixture_is_byte_and_mode_unchanged_on_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp443_venv_fixture_antigravity_") as temporary:
            fixture = Path(temporary) / "venv_fixture_antigravity"
            (fixture / "bin").mkdir(parents=True, mode=0o755)
            fake_python = fixture / "bin/python"
            fake_python.write_text("#!/bin/sh\nprintf '%s\\n' '3.13.0'\n", encoding="utf-8")
            fake_python.chmod(0o755)
            before = _tree_state(fixture)
            result = subprocess.run(
                [
                    "bash", "-c",
                    'source "$1"; exp443_validate_existing_venv_antigravity "$2" "$3"',
                    "bash", str(_SCRIPT), str(fixture),
                    str(Path(FROZEN_CONFIG.environment_lock_path)),
                ],
                cwd=_REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            after = _tree_state(fixture)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing", result.stderr.lower())
            self.assertEqual(before, after)

    def test_fresh_default_freeze_validation_cleans_fixture_temp_on_success_and_failure(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp443_fresh_freeze_antigravity_") as temporary:
            root = Path(temporary)
            lock = root / "requirements_lock_antigravity.txt"
            payload = root / "freeze_payload_antigravity.txt"
            fake_python = root / "python_antigravity"
            freeze_path = root / "default_freeze_antigravity.tmp"
            lock.write_text("a==2\nz==1\n", encoding="utf-8")
            payload.write_text("z==1\na==2\n", encoding="utf-8")
            fake_python.write_text(
                "#!/bin/sh\n"
                "if [ \"$3\" = freeze ]; then cat \"%s\"; else exit 7; fi\n" % payload,
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            command = 'source "$1"; exp443_validate_fresh_freeze_antigravity "$2" "$3" "$4"'
            success = subprocess.run(
                ["bash", "-c", command, "bash", str(_SCRIPT), str(fake_python), str(lock), str(freeze_path)],
                cwd=_REPO_ROOT, text=True, capture_output=True, check=False,
            )
            self.assertEqual(success.returncode, 0, success.stderr)
            self.assertFalse(freeze_path.exists())

            payload.write_text("wrong==9\n", encoding="utf-8")
            failure = subprocess.run(
                ["bash", "-c", command, "bash", str(_SCRIPT), str(fake_python), str(lock), str(freeze_path)],
                cwd=_REPO_ROOT, text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(failure.returncode, 0)
            self.assertIn("fresh Exp443 freeze validation", failure.stderr)
            self.assertFalse(freeze_path.exists())

    def test_bootstrap_freeze_failure_cleans_fixture_temp_without_lock(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp443_bootstrap_freeze_antigravity_") as temporary:
            root = Path(temporary)
            lock = root / "requirements_lock_antigravity.txt"
            fake_python = root / "python_antigravity"
            fake_python.write_text("#!/bin/sh\nexit 7\n", encoding="utf-8")
            fake_python.chmod(0o755)
            command = 'source "$1"; exp443_bootstrap_lock_antigravity "$2" "$3"'
            failure = subprocess.run(
                ["bash", "-c", command, "bash", str(_SCRIPT), str(fake_python), str(lock)],
                cwd=_REPO_ROOT, text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(failure.returncode, 0)
            self.assertIn("fresh Exp443 bootstrap freeze failed", failure.stderr)
            self.assertFalse(lock.exists())
            self.assertEqual(list(root.glob("requirements_lock_antigravity.txt.tmp.*")), [])

    def test_bootstrap_atomic_link_publication_handles_signals_and_never_overwrites(self) -> None:
        """The final lock is absent or complete across both publication signals.

        This uses real shell subprocesses, a fake ``pip freeze`` executable,
        and a private temporary directory.  The two hooks define deterministic
        signal windows immediately before and after atomic publication; no
        production venv or tracked lock is used.
        """
        with tempfile.TemporaryDirectory(prefix="exp443_bootstrap_publish_antigravity_") as temporary:
            root = Path(temporary)
            fake_python = root / "python_antigravity"
            fake_python.write_text(
                "#!/bin/sh\n"
                "if [ \"$3\" = freeze ]; then printf '%s\\n' 'z==1' 'a==2'; else exit 7; fi\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            expected = b"a==2\nz==1\n"

            def run(lock: Path, hook: str = "") -> subprocess.CompletedProcess[str]:
                command = (
                    'source "$1"; '
                    + hook
                    + 'exp443_bootstrap_lock_antigravity "$2" "$3"'
                )
                return subprocess.run(
                    ["bash", "-c", command, "bash", str(_SCRIPT), str(fake_python), str(lock)],
                    cwd=_REPO_ROOT, text=True, capture_output=True, check=False,
                )

            before_lock = root / "before_requirements_lock_antigravity.txt"
            before = run(
                before_lock,
                'exp443_before_bootstrap_publication_antigravity() { kill -TERM "$$"; }; ',
            )
            self.assertEqual(before.returncode, 143, before.stderr)
            self.assertFalse(before_lock.exists())
            self.assertEqual(list(root.glob(f"{before_lock.name}.tmp.antigravity.*")), [])

            after_lock = root / "after_requirements_lock_antigravity.txt"
            after = run(
                after_lock,
                'exp443_after_bootstrap_publication_antigravity() { kill -TERM "$$"; }; ',
            )
            self.assertEqual(after.returncode, 143, after.stderr)
            self.assertEqual(after_lock.read_bytes(), expected)
            self.assertEqual(list(root.glob(f"{after_lock.name}.tmp.antigravity.*")), [])

            success_lock = root / "success_requirements_lock_antigravity.txt"
            success = run(success_lock)
            self.assertEqual(success.returncode, 0, success.stderr)
            self.assertEqual(success_lock.read_bytes(), expected)
            self.assertEqual(list(root.glob(f"{success_lock.name}.tmp.antigravity.*")), [])

            existing_lock = root / "existing_requirements_lock_antigravity.txt"
            existing = b"existing==9\n"
            existing_lock.write_bytes(existing)
            preexisting = run(existing_lock)
            self.assertNotEqual(preexisting.returncode, 0)
            self.assertEqual(existing_lock.read_bytes(), existing)
            self.assertEqual(list(root.glob(f"{existing_lock.name}.tmp.antigravity.*")), [])


if __name__ == "__main__":
    unittest.main()
