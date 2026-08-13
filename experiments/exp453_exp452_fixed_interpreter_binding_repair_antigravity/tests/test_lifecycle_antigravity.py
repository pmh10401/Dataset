"""Private-root tests for Exp453's readiness/evaluate/inspect lifecycle."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.config_antigravity import (
    FIXED_EXECUTABLE_ANTIGRAVITY,
    FIXED_PREFIX_ANTIGRAVITY,
    RUNTIME_VERSIONS_ANTIGRAVITY,
)


_COMMIT_ANTIGRAVITY = "b" * 40


class LifecycleAntigravityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="exp453_lifecycle_antigravity_")
        self.parent = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _paths_antigravity(self, lifecycle):  # type: ignore[no-untyped-def]
        python_version, numpy_version, aeon_version, sklearn_version = RUNTIME_VERSIONS_ANTIGRAVITY
        return lifecycle._LifecyclePathsAntigravity(
            receipt_directory=self.parent / "receipts_antigravity",
            audit_path=self.parent / "failure_audit_antigravity.json",
            evaluation_root=self.parent / "evaluation_antigravity",
            executable=FIXED_EXECUTABLE_ANTIGRAVITY,
            prefix=FIXED_PREFIX_ANTIGRAVITY,
            python_version=python_version,
            numpy_version=numpy_version,
            aeon_version=aeon_version,
            sklearn_version=sklearn_version,
        )

    def _publish_root_antigravity(self, root: Path) -> None:
        root.mkdir(mode=0o700)
        (root / "identity_antigravity.json").write_bytes(b'{"schema":"synthetic"}\n')

    def test_receipts_record_literal_executable_prefix_and_versions_antigravity(self) -> None:
        """Break caught: lifecycle omits exact runtime identity from the receipt chain."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity import (
            lifecycle_antigravity as lifecycle,
        )

        paths = self._paths_antigravity(lifecycle)
        lifecycle._run_phase_with_paths_antigravity(
            "readiness", _COMMIT_ANTIGRAVITY, lambda: "ready", paths
        )
        payload = json.loads((paths.receipt_directory / "readiness_complete_antigravity.json").read_text())
        self.assertEqual(payload["experiment_id"], "Exp453")
        self.assertEqual(payload["executable"], FIXED_EXECUTABLE_ANTIGRAVITY)
        self.assertEqual(payload["prefix"], FIXED_PREFIX_ANTIGRAVITY)
        self.assertEqual(payload["python_version"], "3.14.6")
        self.assertEqual(payload["numpy_version"], "2.3.5")
        self.assertEqual(payload["aeon_version"], "1.4.0")
        self.assertEqual(payload["sklearn_version"], "1.7.2")
        self.assertNotIn("resolve", json.dumps(payload))

    def test_only_readiness_evaluate_bare_inspect_complete_in_order_antigravity(self) -> None:
        """Break caught: a repeated/out-of-order phase runs or bare inspect invents a commit."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity import (
            lifecycle_antigravity as lifecycle,
        )

        paths = self._paths_antigravity(lifecycle)
        calls: list[str] = []
        self.assertEqual(
            lifecycle._run_phase_with_paths_antigravity(
                "readiness", _COMMIT_ANTIGRAVITY, lambda: calls.append("readiness") or "ready", paths
            ),
            "ready",
        )

        def evaluate() -> str:
            calls.append("evaluate")
            self._publish_root_antigravity(paths.evaluation_root)
            return "evaluated"

        self.assertEqual(
            lifecycle._run_phase_with_paths_antigravity("evaluate", _COMMIT_ANTIGRAVITY, evaluate, paths),
            "evaluated",
        )
        self.assertEqual(
            lifecycle._run_phase_with_paths_antigravity(
                "inspect", None, lambda commit: calls.append(f"inspect:{commit}") or "inspected", paths
            ),
            "inspected",
        )
        classification = lifecycle.validate_receipt_inventory_antigravity(
            paths.receipt_directory,
            evaluation_root=paths.evaluation_root,
        )
        self.assertEqual(classification.state, "S5")
        self.assertEqual(classification.reviewed_commit, _COMMIT_ANTIGRAVITY)
        self.assertFalse(classification.stopped)
        self.assertEqual(calls, ["readiness", "evaluate", f"inspect:{_COMMIT_ANTIGRAVITY}"])
        self.assertEqual(
            sorted(path.name for path in paths.receipt_directory.iterdir()),
            sorted((
                "readiness_complete_antigravity.json",
                "evaluate_started_antigravity.json",
                "evaluate_complete_antigravity.json",
                "inspect_started_antigravity.json",
                "inspect_complete_antigravity.json",
            )),
        )
        self.assertFalse(paths.audit_path.exists())

    def test_invalid_commit_out_of_order_and_repeat_stop_before_operation_antigravity(self) -> None:
        """Break caught: invalid identity or phase predecessor can execute work or retry."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity import (
            lifecycle_antigravity as lifecycle,
        )

        paths = self._paths_antigravity(lifecycle)
        for invalid in ("B" * 40, "b" * 39, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                lifecycle._run_phase_with_paths_antigravity(
                    "readiness", invalid, lambda: self.fail("invalid commit ran"), paths
                )
        self.assertFalse(paths.receipt_directory.exists())
        with self.assertRaisesRegex(ValueError, "exact predecessor"):
            lifecycle._run_phase_with_paths_antigravity(
                "evaluate", _COMMIT_ANTIGRAVITY, lambda: self.fail("out of order ran"), paths
            )
        classification = lifecycle.validate_receipt_inventory_antigravity(paths.receipt_directory)
        self.assertTrue(classification.stopped)
        self.assertEqual(classification.state, "STOPPED")
        self.assertTrue(paths.audit_path.is_file())
        audit_before = paths.audit_path.read_bytes()
        receipts_before = tuple((path.name, path.read_bytes()) for path in sorted(paths.receipt_directory.iterdir()))
        with self.assertRaisesRegex(ValueError, "permanently stopped"):
            lifecycle._run_phase_with_paths_antigravity(
                "readiness", _COMMIT_ANTIGRAVITY, lambda: self.fail("retry ran"), paths
            )
        self.assertEqual(paths.audit_path.read_bytes(), audit_before)
        self.assertEqual(
            tuple((path.name, path.read_bytes()) for path in sorted(paths.receipt_directory.iterdir())),
            receipts_before,
        )

    def test_occupied_symlink_partial_receipt_root_and_audit_namespaces_block_antigravity(self) -> None:
        """Break caught: an owned lifecycle is inferred from occupied, linked, or partial paths."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity import (
            lifecycle_antigravity as lifecycle,
        )

        python_version, numpy_version, aeon_version, sklearn_version = RUNTIME_VERSIONS_ANTIGRAVITY
        cases = ("empty_receipts", "receipt_symlink", "partial", "root", "audit")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory(prefix="exp453_namespace_antigravity_") as temporary:
                parent = Path(temporary)
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
                if case == "empty_receipts":
                    paths.receipt_directory.mkdir()
                elif case == "receipt_symlink":
                    target = parent / "target_antigravity"
                    target.mkdir()
                    paths.receipt_directory.symlink_to(target, target_is_directory=True)
                elif case == "partial":
                    paths.receipt_directory.mkdir()
                    (paths.receipt_directory / "evaluate_started_antigravity.json").write_bytes(b"{}\n")
                elif case == "root":
                    paths.evaluation_root.write_bytes(b"occupied\n")
                else:
                    paths.audit_path.write_bytes(b"occupied\n")
                with self.assertRaises(ValueError):
                    lifecycle._run_phase_with_paths_antigravity(
                        "readiness", _COMMIT_ANTIGRAVITY, lambda: self.fail(f"{case} ran"), paths
                    )
                if case in {"empty_receipts", "receipt_symlink", "partial"}:
                    self.assertFalse(paths.audit_path.exists())
                if case == "root":
                    self.assertEqual(paths.evaluation_root.read_bytes(), b"occupied\n")
                if case == "audit":
                    self.assertEqual(paths.audit_path.read_bytes(), b"occupied\n")
                    self.assertTrue(lifecycle.validate_receipt_inventory_antigravity(paths.receipt_directory).stopped)

    def test_operation_failure_has_started_then_stopped_never_complete_and_no_retry_antigravity(self) -> None:
        """Break caught: evaluate failure is followed by a complete receipt or remains retryable."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity import (
            lifecycle_antigravity as lifecycle,
        )

        paths = self._paths_antigravity(lifecycle)
        lifecycle._run_phase_with_paths_antigravity("readiness", _COMMIT_ANTIGRAVITY, lambda: "ready", paths)
        with self.assertRaisesRegex(RuntimeError, "evaluate boom"):
            lifecycle._run_phase_with_paths_antigravity(
                "evaluate",
                _COMMIT_ANTIGRAVITY,
                lambda: (_ for _ in ()).throw(RuntimeError("evaluate boom")),
                paths,
            )
        self.assertTrue((paths.receipt_directory / "evaluate_started_antigravity.json").is_file())
        self.assertFalse((paths.receipt_directory / "evaluate_complete_antigravity.json").exists())
        self.assertTrue((paths.receipt_directory / "stopped_antigravity.json").is_file())
        self.assertTrue(paths.audit_path.is_file())
        self.assertIn(b'"permanent_stop":true', paths.audit_path.read_bytes())
        with self.assertRaisesRegex(ValueError, "permanently stopped"):
            lifecycle._run_phase_with_paths_antigravity(
                "evaluate", _COMMIT_ANTIGRAVITY, lambda: self.fail("retry ran"), paths
            )

    def test_root_drift_and_receipt_multilink_fail_closed_antigravity(self) -> None:
        """Break caught: inspect mutates publication bytes or a receipt gains another name."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity import (
            lifecycle_antigravity as lifecycle,
        )

        paths = self._paths_antigravity(lifecycle)
        lifecycle._run_phase_with_paths_antigravity("readiness", _COMMIT_ANTIGRAVITY, lambda: None, paths)
        lifecycle._run_phase_with_paths_antigravity(
            "evaluate",
            _COMMIT_ANTIGRAVITY,
            lambda: self._publish_root_antigravity(paths.evaluation_root),
            paths,
        )
        with self.assertRaisesRegex(ValueError, "root changed"):
            lifecycle._run_phase_with_paths_antigravity(
                "inspect",
                None,
                lambda _commit: (paths.evaluation_root / "identity_antigravity.json").write_bytes(b"changed\n"),
                paths,
            )
        self.assertTrue(lifecycle.validate_receipt_inventory_antigravity(paths.receipt_directory).stopped)

        other = self.parent / "other_antigravity"
        other.mkdir()
        receipt = paths.receipt_directory / "readiness_complete_antigravity.json"
        os.link(receipt, other / "receipt_hardlink_antigravity.json")
        classification = lifecycle.validate_receipt_inventory_antigravity(paths.receipt_directory)
        self.assertTrue(classification.invalid)

    def test_readiness_path_precondition_runs_before_receipt_creation_antigravity(self) -> None:
        """Break caught: readiness creates its receipt namespace before the absence preflight."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity import (
            lifecycle_antigravity as lifecycle,
        )

        paths = self._paths_antigravity(lifecycle)
        observations: list[bool] = []

        def precondition() -> None:
            observations.append(paths.receipt_directory.exists())

        lifecycle._run_phase_with_paths_antigravity(
            "readiness",
            _COMMIT_ANTIGRAVITY,
            lambda: None,
            paths,
            path_precondition=precondition,
        )
        self.assertEqual(observations, [False])

    def test_wrong_runtime_fields_are_rejected_antigravity(self) -> None:
        """Break caught: lifecycle accepts a resolved or version-drifted runtime record."""
        from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity import (
            lifecycle_antigravity as lifecycle,
        )

        python_version, numpy_version, aeon_version, sklearn_version = RUNTIME_VERSIONS_ANTIGRAVITY
        with self.assertRaises(ValueError):
            lifecycle._LifecyclePathsAntigravity(
                self.parent / "receipts_antigravity",
                self.parent / "audit_antigravity.json",
                self.parent / "evaluation_antigravity",
                "/resolved/but/wrong/python",
                FIXED_PREFIX_ANTIGRAVITY,
                python_version,
                numpy_version,
                aeon_version,
                sklearn_version,
            )
        with self.assertRaises(ValueError):
            lifecycle._LifecyclePathsAntigravity(
                self.parent / "receipts_antigravity",
                self.parent / "audit_antigravity.json",
                self.parent / "evaluation_antigravity",
                FIXED_EXECUTABLE_ANTIGRAVITY,
                FIXED_PREFIX_ANTIGRAVITY,
                python_version,
                "2.4.6",
                aeon_version,
                sklearn_version,
            )
        original = lifecycle.importlib.metadata.version

        def drifted_numpy_antigravity(name: str) -> str:
            if name == "numpy":
                return "2.4.6"
            return original(name)

        lifecycle.importlib.metadata.version = drifted_numpy_antigravity
        try:
            with self.assertRaises(ValueError):
                lifecycle._production_paths_antigravity()
        finally:
            lifecycle.importlib.metadata.version = original


if __name__ == "__main__":
    unittest.main()
