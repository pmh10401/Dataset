"""Synthetic, label-incapable readiness tests for Exp453."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest

from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.config_antigravity import (
    FIXED_EXECUTABLE_ANTIGRAVITY,
    FIXED_PREFIX_ANTIGRAVITY,
)


_COMMIT_ANTIGRAVITY = "c" * 40


class ReadinessAntigravityTest(unittest.TestCase):
    def test_readiness_runs_review_first_and_records_exact_runtime_antigravity(self) -> None:
        """Break caught: readiness can authorize drift or hide the exact interpreter binding."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity import (
            lifecycle_antigravity as lifecycle,
        )
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity import (
            readiness_antigravity as readiness,
        )

        events: list[str] = []
        authority = SimpleNamespace(
            predecessor_attestation_sha256="a" * 64,
            prediction_aggregate_sha256="b" * 64,
            exp452_stopped_receipt_sha256="8" * 64,
            exp452_failure_audit_sha256="e" * 64,
            exp452_evaluation_root_absent=True,
            dataset_count=48,
            prediction_count=96,
        )
        documents = (
            ("approved_design", "1" * 64),
            ("implementation_report", "2" * 64),
            ("preregistration", "3" * 64),
            ("runbook", "4" * 64),
        )
        database = readiness._DatabaseIdentityAntigravity("5" * 64, 1048138, 1, 99)
        runtime = readiness._RuntimeStateAntigravity(
            FIXED_EXECUTABLE_ANTIGRAVITY,
            FIXED_PREFIX_ANTIGRAVITY,
            "3.14.6",
            "2.3.5",
            "1.4.0",
            "1.7.2",
        )
        backend = readiness._BackendStateAntigravity("2.8.0", "0.29.3", "0.29.3", True, False)
        with tempfile.TemporaryDirectory(prefix="exp453_readiness_antigravity_") as temporary:
            parent = Path(temporary)
            python_version, numpy_version, aeon_version, sklearn_version = ("3.14.6", "2.3.5", "1.4.0", "1.7.2")
            paths = lifecycle._LifecyclePathsAntigravity(
                parent / "receipts_antigravity",
                parent / "audit_antigravity.json",
                parent / "evaluation_antigravity",
                FIXED_EXECUTABLE_ANTIGRAVITY,
                FIXED_PREFIX_ANTIGRAVITY,
                python_version,
                numpy_version,
                aeon_version,
                sklearn_version,
            )

            def review(value: str) -> None:
                self.assertEqual(value, _COMMIT_ANTIGRAVITY)
                events.append("review")

            def initial_absence() -> None:
                events.append("initial_paths")
                self.assertFalse(paths.receipt_directory.exists())
                self.assertFalse(paths.audit_path.exists())
                self.assertFalse(paths.evaluation_root.exists())

            def final_absence() -> None:
                events.append("final_paths")
                self.assertTrue(paths.receipt_directory.is_dir())
                self.assertFalse(paths.audit_path.exists())
                self.assertFalse(paths.evaluation_root.exists())

            dependencies = readiness._ReadinessDependenciesAntigravity(
                validate_reviewed_map=review,
                document_hashes=lambda: events.append("documents") or documents,
                require_initial_paths_absent=initial_absence,
                require_runtime_paths_absent=final_absence,
                verify_database=lambda: events.append("database") or database,
                capture_predecessors=lambda: events.append("predecessor") or authority,
                validate_predecessors=lambda value: (
                    events.append("validate_predecessor")
                    or tuple(f"dataset_{index}" for index in range(48))
                    if value is authority
                    else ()
                ),
                runtime_state=lambda: events.append("runtime") or runtime,
                backend_state=lambda: events.append("backend") or backend,
                run_readiness_phase=lambda commit, operation, precondition: lifecycle._run_phase_with_paths_antigravity(
                    "readiness", commit, operation, paths, path_precondition=precondition
                ),
            )
            manifest = readiness._readiness_with_dependencies_antigravity(_COMMIT_ANTIGRAVITY, dependencies)

            self.assertEqual(
                events,
                [
                    "review", "initial_paths", "documents", "database", "predecessor",
                    "validate_predecessor", "runtime", "backend", "final_paths", "review",
                ],
            )
            self.assertEqual(lifecycle.validate_receipt_inventory_antigravity(paths.receipt_directory).state, "S1")
            self.assertEqual(manifest["label_access_count"], 0)
            self.assertEqual(manifest["dataset_count"], 48)
            self.assertEqual(manifest["prediction_count"], 96)
            self.assertNotIn("order_mismatch_count", manifest)
            self.assertEqual(manifest["exp452_stopped_receipt_sha256"], "8" * 64)
            self.assertEqual(manifest["exp452_failure_audit_sha256"], "e" * 64)
            self.assertTrue(manifest["exp452_evaluation_root_absent"])
            self.assertEqual(manifest["prediction_aggregate_sha256"], "b" * 64)
            self.assertEqual(manifest["runtime"]["executable"], FIXED_EXECUTABLE_ANTIGRAVITY)
            self.assertEqual(manifest["runtime"]["prefix"], FIXED_PREFIX_ANTIGRAVITY)
            self.assertEqual(manifest["runtime"]["python_version"], "3.14.6")
            self.assertEqual(manifest["runtime"]["numpy_version"], "2.3.5")
            self.assertEqual(manifest["runtime"]["aeon_version"], "1.4.0")
            self.assertEqual(manifest["runtime"]["sklearn_version"], "1.7.2")
            self.assertEqual(manifest["backend"]["canonical_backend"], "cpu")
            self.assertFalse(manifest["backend"]["tensor_eligible"])
            self.assertFalse(manifest["backend"]["mlx_mps_workload_run"])

    def test_readiness_failure_stops_and_does_not_reach_later_gates_antigravity(self) -> None:
        """Break caught: a failed predecessor/readiness gate remains retryable or reaches labels."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity import (
            lifecycle_antigravity as lifecycle,
        )
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity import (
            readiness_antigravity as readiness,
        )

        with tempfile.TemporaryDirectory(prefix="exp453_readiness_stop_antigravity_") as temporary:
            parent = Path(temporary)
            python_version, numpy_version, aeon_version, sklearn_version = ("3.14.6", "2.3.5", "1.4.0", "1.7.2")
            paths = lifecycle._LifecyclePathsAntigravity(
                parent / "receipts_antigravity",
                parent / "audit_antigravity.json",
                parent / "evaluation_antigravity",
                FIXED_EXECUTABLE_ANTIGRAVITY,
                FIXED_PREFIX_ANTIGRAVITY,
                python_version,
                numpy_version,
                aeon_version,
                sklearn_version,
            )
            events: list[str] = []
            dependencies = readiness._ReadinessDependenciesAntigravity(
                validate_reviewed_map=lambda _commit: events.append("review"),
                document_hashes=lambda: (
                    ("approved_design", "1" * 64),
                    ("implementation_report", "2" * 64),
                    ("preregistration", "3" * 64),
                    ("runbook", "4" * 64),
                ),
                require_initial_paths_absent=lambda: events.append("initial_paths"),
                require_runtime_paths_absent=lambda: events.append("forbidden_final"),
                verify_database=lambda: (_ for _ in ()).throw(ValueError("database drift")),
                capture_predecessors=lambda: events.append("forbidden_predecessor"),
                validate_predecessors=lambda _authority: (),
                runtime_state=lambda: (_ for _ in ()).throw(AssertionError("runtime reached")),
                backend_state=lambda: (_ for _ in ()).throw(AssertionError("backend reached")),
                run_readiness_phase=lambda commit, operation, precondition: lifecycle._run_phase_with_paths_antigravity(
                    "readiness", commit, operation, paths, path_precondition=precondition
                ),
            )
            with self.assertRaisesRegex(ValueError, "database drift"):
                readiness._readiness_with_dependencies_antigravity(_COMMIT_ANTIGRAVITY, dependencies)
            self.assertEqual(events, ["review", "initial_paths"])
            self.assertTrue(lifecycle.validate_receipt_inventory_antigravity(paths.receipt_directory).stopped)
            self.assertTrue(paths.audit_path.is_file())
            with self.assertRaisesRegex(ValueError, "permanently stopped"):
                readiness._readiness_with_dependencies_antigravity(_COMMIT_ANTIGRAVITY, dependencies)

    def test_database_identity_is_descriptor_pinned_without_sql_antigravity(self) -> None:
        """Break caught: readiness trusts a path hash or opens SQLite metadata/labels."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity import (
            readiness_antigravity as readiness,
        )

        with tempfile.TemporaryDirectory(prefix="exp453_readiness_db_antigravity_") as temporary:
            database = Path(temporary) / "database_antigravity.bin"
            payload = b"synthetic database bytes\n"
            database.write_bytes(payload)
            inode = os.stat(database, follow_symlinks=False).st_ino
            identity = readiness._verify_database_descriptor_antigravity(
                database, hashlib.sha256(payload).hexdigest(), inode
            )
            self.assertEqual(identity.sha256, hashlib.sha256(payload).hexdigest())
            self.assertEqual(identity.inode, inode)
            self.assertEqual(identity.nlink, 1)
            hardlink = Path(temporary) / "database_hardlink_antigravity.bin"
            os.link(database, hardlink)
            with self.assertRaisesRegex(ValueError, "inode/link"):
                readiness._verify_database_descriptor_antigravity(
                    database, hashlib.sha256(payload).hexdigest(), inode
                )

    def test_backend_probe_is_metadata_only_and_torch_is_isolated_antigravity(self) -> None:
        """Break caught: backend readiness imports MLX/PyTorch in-process or runs tensors."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity import (
            readiness_antigravity as readiness,
        )

        calls: list[tuple[object, ...]] = []

        def version(name: str) -> str:
            return {"torch": "2.8.0", "mlx": "0.29.3", "mlx-metal": "0.29.3"}[name]

        def run(arguments: tuple[str, ...]):  # type: ignore[no-untyped-def]
            calls.append(arguments)
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"pytorch_version": "2.8.0", "mps_built": True, "mps_available": False}),
                stderr="",
            )

        state = readiness._probe_backend_metadata_antigravity(
            FIXED_EXECUTABLE_ANTIGRAVITY, distribution_version=version, run_process=run
        )
        self.assertEqual(state.pytorch_version, "2.8.0")
        self.assertEqual(len(calls), 1)
        command_text = " ".join(calls[0])
        self.assertIn("import torch", command_text)
        self.assertNotIn("tensor(", command_text)
        self.assertNotIn("mlx.core", command_text)

    def test_reviewed_map_requires_clean_map_only_child_antigravity(self) -> None:
        """Break caught: readiness accepts a clean but stale/non-map-child code map."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity import (
            readiness_antigravity as readiness,
        )

        parent = "d" * 40
        responses = {
            ("rev-parse", "HEAD"): _COMMIT_ANTIGRAVITY,
            ("status", "--porcelain=v1"): "",
            ("rev-parse", "HEAD^"): parent,
            ("diff", "--name-only", "HEAD^", "HEAD"): "\n".join(sorted(readiness._MAP_TRIPLET_ANTIGRAVITY)),
        }
        validations: list[Path] = []
        readiness._validate_reviewed_map_antigravity(
            _COMMIT_ANTIGRAVITY,
            repository=Path("/synthetic/repository_antigravity"),
            git=lambda *arguments, repository: responses[arguments],
            validate_current_map=lambda repository: validations.append(repository),
        )
        self.assertEqual(validations, [Path("/synthetic/repository_antigravity")])
        responses[("status", "--porcelain=v1")] = " M tracked_antigravity.py"
        with self.assertRaisesRegex(ValueError, "clean"):
            readiness._validate_reviewed_map_antigravity(
                _COMMIT_ANTIGRAVITY,
                repository=Path("/synthetic/repository_antigravity"),
                git=lambda *arguments, repository: responses[arguments],
                validate_current_map=lambda _repository: None,
            )

    def test_runtime_state_uses_literal_executable_and_prefix_antigravity(self) -> None:
        """Break caught: readiness identity still uses resolve() or samefile()."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity import (
            readiness_antigravity as readiness,
        )

        source = inspect.getsource(readiness._runtime_state_antigravity)
        self.assertNotIn("resolve(", source)
        self.assertNotIn("samefile", source)
        self.assertIn("sys.executable", source)
        self.assertIn("sys.prefix", source)

    def test_readiness_import_is_label_and_backend_workload_firewalled_antigravity(self) -> None:
        """Break caught: importing readiness makes labels/evaluator/network/dashboard reachable."""
        import experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.readiness_antigravity as readiness

        source = inspect.getsource(readiness)
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        for forbidden in ("sqlite3", "numpy", "torch", "mlx", "requests", "urllib", "httpx"):
            self.assertNotIn(forbidden, imported)
        for forbidden in (
            "sources_antigravity", "evaluator_antigravity", "replay_antigravity",
            "dashboard_export_antigravity", "open_test_labels_once_antigravity", "SELECT ",
        ):
            self.assertNotIn(forbidden, source)

        repository = str(Path(__file__).resolve().parents[3])
        code = (
            f"import json,sys; sys.path.insert(0,{repository!r}); before=set(sys.modules); "
            "import experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.readiness_antigravity; "
            "after=set(sys.modules)-before; "
            "print(json.dumps(sorted(after)))"
        )
        result = subprocess.run(
            (sys.executable, "-I", "-c", code),
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            capture_output=True,
            check=True,
        )
        loaded = set(json.loads(result.stdout))
        self.assertFalse(
            any(
                name in loaded or any(item.startswith(name + ".") for item in loaded)
                for name in ("sqlite3", "numpy", "torch", "mlx", "requests", "urllib", "httpx")
            )
        )


if __name__ == "__main__":
    unittest.main()
