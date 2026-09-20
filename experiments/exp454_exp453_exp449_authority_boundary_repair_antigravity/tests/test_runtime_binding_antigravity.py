"""Exact executable/prefix runtime-binding tests for Exp454 Task 3."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[3]
HELPER_ANTIGRAVITY = ROOT_ANTIGRAVITY / "scratch/exp454_runtime_binding_antigravity.py"
LAUNCHER_ANTIGRAVITY = (
    ROOT_ANTIGRAVITY / "scratch/run_exp454_exp453_exp449_authority_boundary_repair_antigravity.py"
)
PROBE_ANTIGRAVITY = Path(__file__).with_name("probe_runtime_launcher_antigravity.py")
FIXED_EXECUTABLE_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python"
)
FIXED_PREFIX_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314"
)
EXP444_EXECUTABLE_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314/bin/python"
)
EXP444_PREFIX_ANTIGRAVITY = (
    "/Users/mac/Documents/Dataset/venvs/exp444_stability_antigravity_py314"
)


def _load_helper_antigravity():
    spec = importlib.util.spec_from_file_location(
        "exp454_runtime_binding_antigravity", HELPER_ANTIGRAVITY
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Exp454 runtime binding helper is not importable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _probe_antigravity(executable: str, *, bind: bool, extra_argv: list[str] | None = None, env: dict[str, str] | None = None):
    command = [executable, str(PROBE_ANTIGRAVITY), *(extra_argv or [])]
    merged = dict(os.environ)
    if env:
        merged.update(env)
    merged["EXP454_PROBE_BIND_ANTIGRAVITY"] = "1" if bind else "0"
    return subprocess.run(command, text=True, capture_output=True, check=False, cwd=str(ROOT_ANTIGRAVITY), env=merged)


class RuntimeBindingAntigravityTest(unittest.TestCase):
    def test_exp444_without_binding_keeps_wrong_literal_runtime_antigravity(self) -> None:
        """Break caught: resolve()/samefile() identity hides the Exp444 venv."""
        result = _probe_antigravity(EXP444_EXECUTABLE_ANTIGRAVITY, bind=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["executable"], EXP444_EXECUTABLE_ANTIGRAVITY)
        self.assertEqual(payload["prefix"], EXP444_PREFIX_ANTIGRAVITY)
        self.assertIsNone(payload["reexec_marker"])
        self.assertEqual(payload["exp454_imported"], [])
        self.assertEqual(Path(payload["executable"]).resolve(), Path(FIXED_EXECUTABLE_ANTIGRAVITY).resolve())
        self.assertTrue(Path(payload["executable"]).samefile(FIXED_EXECUTABLE_ANTIGRAVITY))
        self.assertNotEqual(payload["executable"], FIXED_EXECUTABLE_ANTIGRAVITY)
        self.assertNotEqual(payload["prefix"], FIXED_PREFIX_ANTIGRAVITY)

    def test_helper_compares_literal_executable_and_prefix_only_antigravity(self) -> None:
        """Break caught: runtime identity is decided by resolve() or samefile()."""
        self.assertTrue(HELPER_ANTIGRAVITY.is_file())
        source = HELPER_ANTIGRAVITY.read_text(encoding="utf-8")
        tree = ast.parse(source)
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertNotIn("resolve", called)
        self.assertNotIn("samefile", called)
        compact = source.replace(" ", "")
        self.assertIn("sys.executable==FIXED_EXECUTABLE_ANTIGRAVITY", compact)
        self.assertIn("sys.prefix==FIXED_PREFIX_ANTIGRAVITY", compact)
        self.assertIn("importlib.metadata", source)
        self.assertIn("platform.python_version", source)
        self.assertIn("observe_exact_runtime_antigravity", source)

    def test_wrong_venv_reexecs_exactly_once_and_preserves_argv_antigravity(self) -> None:
        """Break caught: wrong venv imports Exp454 or drops --reviewed-commit."""
        commit = "ab" * 20
        result = _probe_antigravity(
            EXP444_EXECUTABLE_ANTIGRAVITY,
            bind=True,
            extra_argv=["readiness", "--reviewed-commit", commit],
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["executable"], FIXED_EXECUTABLE_ANTIGRAVITY)
        self.assertEqual(payload["prefix"], FIXED_PREFIX_ANTIGRAVITY)
        self.assertEqual(payload["reexec_marker"], "1")
        self.assertEqual(payload["exp454_imported"], [])
        self.assertEqual(payload["argv"][1:], ["readiness", "--reviewed-commit", commit])

    def test_fixed_venv_does_not_reexec_antigravity(self) -> None:
        """Break caught: the already-fixed interpreter is replaced again."""
        result = _probe_antigravity(
            FIXED_EXECUTABLE_ANTIGRAVITY,
            bind=True,
            extra_argv=["inspect"],
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["executable"], FIXED_EXECUTABLE_ANTIGRAVITY)
        self.assertEqual(payload["prefix"], FIXED_PREFIX_ANTIGRAVITY)
        self.assertIsNone(payload["reexec_marker"])
        self.assertEqual(payload["exp454_imported"], [])
        self.assertEqual(payload["argv"][1:], ["inspect"])

    def test_failed_exec_and_marker_mismatch_stop_before_package_import_antigravity(self) -> None:
        """Break caught: a failed or looping re-exec still imports Exp454 or opens a root."""
        helper = _load_helper_antigravity()
        before = set(sys.modules)
        with tempfile.TemporaryDirectory(prefix="exp454_runtime_fail_antigravity_") as temporary:
            missing = Path(temporary) / "missing_python_antigravity"
            original_executable = helper.FIXED_EXECUTABLE_ANTIGRAVITY
            original_execve = helper.os.execve
            helper.FIXED_EXECUTABLE_ANTIGRAVITY = str(missing)

            def fail_execve(*_arguments: object) -> None:
                raise OSError("exec failed")

            helper.os.execve = fail_execve
            try:
                with self.assertRaisesRegex(RuntimeError, "re-exec failed"):
                    helper.ensure_fixed_runtime_antigravity(PROBE_ANTIGRAVITY, ("readiness",))
            finally:
                helper.FIXED_EXECUTABLE_ANTIGRAVITY = original_executable
                helper.os.execve = original_execve
        added = set(sys.modules) - before
        self.assertFalse(
            any(
                name == "experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity"
                or name.startswith("experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity.")
                for name in added
            )
        )

        result = _probe_antigravity(
            EXP444_EXECUTABLE_ANTIGRAVITY,
            bind=True,
            env={"EXP454_FIXED_RUNTIME_REEXEC_ANTIGRAVITY": "1"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("re-exec failed", result.stderr)
        self.assertNotIn("import experiments.exp454", result.stderr)
        self.assertNotIn("open_test_labels_once_antigravity", result.stderr)

    def test_fixed_path_wrong_package_version_rejected_before_import_antigravity(self) -> None:
        """Break caught: a drifted package version on the fixed path still proceeds to Exp454 import."""
        result = _probe_antigravity(
            FIXED_EXECUTABLE_ANTIGRAVITY,
            bind=True,
            extra_argv=["evaluate"],
            env={"EXP454_RUNTIME_METADATA_PROBE_ANTIGRAVITY": json.dumps({"numpy": "2.4.6"})},
        )
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("re-exec failed", result.stderr)
        self.assertEqual(result.stdout.strip(), "")
        self.assertNotIn("import experiments.exp454", result.stderr)
        self.assertNotIn("open_test_labels_once_antigravity", result.stderr)
        self.assertNotIn("lifecycle_antigravity", result.stderr)

    def test_official_launcher_binds_before_thread_env_and_package_import_antigravity(self) -> None:
        """Break caught: the official launcher imports Exp454 or sets threads before re-exec."""
        self.assertTrue(LAUNCHER_ANTIGRAVITY.is_file())
        source = LAUNCHER_ANTIGRAVITY.read_text(encoding="utf-8")
        bind_at = source.index("ensure_fixed_runtime_antigravity")
        thread_at = source.index("OMP_NUM_THREADS")
        package_at = source.index(
            "experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity"
        )
        self.assertLess(bind_at, thread_at)
        self.assertLess(thread_at, package_at)
        self.assertLess(thread_at, source.index("cli_antigravity"))
        self.assertNotIn("evaluation_antigravity", source)
        self.assertNotIn("replay_antigravity", source)
        self.assertNotIn(".resolve(", source.split("ensure_fixed_runtime_antigravity", 1)[0])
        self.assertNotIn("samefile", source)


if __name__ == "__main__":
    unittest.main()
