"""Receipt lifecycle behavior; all paths are synthetic temporary namespaces."""
from __future__ import annotations

import tempfile
import unittest
import os
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class LifecycleAntigravityTests(unittest.TestCase):
    @staticmethod
    def _seed_state_antigravity(lifecycle, receipts: Path, state: str, commit: str) -> None:
        transitions = (("readiness", "complete"), ("predict", "started"), ("predict", "complete"),
                       ("evaluate", "started"), ("evaluate", "complete"), ("inspect", "started"),
                       ("inspect", "complete"))
        for phase, event in transitions[:int(state[1:])]:
            lifecycle.append_receipt_antigravity(receipts, phase, event, commit)

    def test_exact_predecessor_runs_only_its_phase_operation_antigravity(self) -> None:
        """Break caught: a direct, repeated, or out-of-order command is allowed to do phase work."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import lifecycle_antigravity as lifecycle  # noqa: PLC0415

        commit = "a" * 40
        with tempfile.TemporaryDirectory(prefix="exp449_lifecycle_antigravity_") as temporary:
            parent = Path(temporary)
            receipts, audit = parent / "receipts", parent / "audit_antigravity.json"
            prediction, evaluation = parent / "prediction", parent / "evaluation"
            calls: list[str] = []
            for phase, expected in (("readiness", "readiness"), ("predict", "predict"), ("evaluate", "evaluate"), ("inspect", "inspect")):
                result = lifecycle.run_phase_antigravity(
                    phase, commit, lambda phase=phase: calls.append(phase) or phase,
                    receipt_dir=receipts, audit_path=audit,
                    prediction_root=prediction, evaluation_root=evaluation,
                )
                self.assertEqual(result, expected)
            self.assertEqual(calls, ["readiness", "predict", "evaluate", "inspect"])
            self.assertEqual(lifecycle.validate_receipt_inventory_antigravity(receipts).state, "S7")
            self.assertFalse(audit.exists())

    def test_valid_wrong_predecessor_stops_before_create_only_audit_antigravity(self) -> None:
        """Break caught: a valid S-state command bypasses terminal stop or overwrites failure evidence."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import lifecycle_antigravity as lifecycle  # noqa: PLC0415

        commit = "b" * 40
        with tempfile.TemporaryDirectory(prefix="exp449_lifecycle_stop_antigravity_") as temporary:
            parent = Path(temporary)
            receipts, audit = parent / "receipts", parent / "audit_antigravity.json"
            lifecycle.append_receipt_antigravity(receipts, "readiness", "complete", commit)
            with self.assertRaisesRegex(ValueError, "exact predecessor"):
                lifecycle.run_phase_antigravity(
                    "evaluate", commit, lambda: self.fail("out-of-order operation ran"),
                    receipt_dir=receipts, audit_path=audit,
                    prediction_root=parent / "prediction", evaluation_root=parent / "evaluation",
                )
            state = lifecycle.validate_receipt_inventory_antigravity(receipts)
            self.assertTrue(state.stopped)
            self.assertTrue(audit.is_file())
            before = receipts.read_bytes() if receipts.is_file() else tuple(sorted(path.read_bytes() for path in receipts.iterdir()))
            with self.assertRaises(ValueError):
                lifecycle.run_phase_antigravity(
                    "predict", commit, lambda: self.fail("stopped operation ran"),
                    receipt_dir=receipts, audit_path=audit,
                    prediction_root=parent / "prediction", evaluation_root=parent / "evaluation",
                )
            after = receipts.read_bytes() if receipts.is_file() else tuple(sorted(path.read_bytes() for path in receipts.iterdir()))
            self.assertEqual(after, before)

    def test_reviewed_commit_mismatch_is_rejected_before_any_receipt_mutation_antigravity(self) -> None:
        """Break caught: a later receipt binds a different review and leaves a poisoned partial namespace."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import lifecycle_antigravity as lifecycle  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_lifecycle_chain_antigravity_") as temporary:
            receipts = Path(temporary) / "receipts"
            lifecycle.append_receipt_antigravity(receipts, "readiness", "complete", "c" * 40)
            before = (receipts / "readiness_complete_antigravity.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "reviewed commit"):
                lifecycle.append_receipt_antigravity(receipts, "predict", "started", "d" * 40)
            self.assertEqual(lifecycle.validate_receipt_inventory_antigravity(receipts).state, "S1")
            self.assertEqual((receipts / "readiness_complete_antigravity.json").read_bytes(), before)
            self.assertFalse((receipts / "predict_started_antigravity.json").exists())

    def test_command_state_matrix_stops_every_valid_nonpredecessor_antigravity(self) -> None:
        """Break caught: any one of the 28 invalid valid-state command pairs is allowed to run work."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import lifecycle_antigravity as lifecycle  # noqa: PLC0415

        expected = {"readiness": "S0", "predict": "S1", "evaluate": "S3", "inspect": "S5"}
        commit = "e" * 40
        with tempfile.TemporaryDirectory(prefix="exp449_lifecycle_matrix_antigravity_") as temporary:
            parent = Path(temporary)
            for phase, predecessor in expected.items():
                for state in ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"):
                    with self.subTest(phase=phase, state=state):
                        receipts, audit = parent / f"{phase}_{state}", parent / f"{phase}_{state}_audit.json"
                        self._seed_state_antigravity(lifecycle, receipts, state, commit)
                        calls: list[str] = []
                        if state == predecessor:
                            self.assertEqual(lifecycle.run_phase_antigravity(phase, commit, lambda: calls.append("ran") or "result", receipt_dir=receipts, audit_path=audit, prediction_root=parent / "prediction", evaluation_root=parent / "evaluation"), "result")
                            self.assertEqual(calls, ["ran"])
                            self.assertFalse(audit.exists())
                        else:
                            with self.assertRaisesRegex(ValueError, "exact predecessor"):
                                lifecycle.run_phase_antigravity(phase, commit, lambda: calls.append("ran"), receipt_dir=receipts, audit_path=audit, prediction_root=parent / "prediction", evaluation_root=parent / "evaluation")
                            self.assertEqual(calls, [])
                            self.assertTrue(lifecycle.validate_receipt_inventory_antigravity(receipts).stopped)
                            self.assertTrue(audit.is_file())

    def test_stopped_and_invalid_receipts_are_byte_preserved_without_audit_antigravity(self) -> None:
        """Break caught: terminal or unowned receipt evidence is appended, replaced, or audited."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import lifecycle_antigravity as lifecycle  # noqa: PLC0415

        commit = "f" * 40
        with tempfile.TemporaryDirectory(prefix="exp449_lifecycle_preserve_antigravity_") as temporary:
            parent = Path(temporary)
            stopped = parent / "stopped"; self._seed_state_antigravity(lifecycle, stopped, "S1", commit)
            lifecycle.append_receipt_antigravity(stopped, "predict", "stopped", commit)
            stopped_before = tuple((path.name, path.read_bytes()) for path in sorted(stopped.iterdir()))
            with self.assertRaisesRegex(ValueError, "permanently stopped"):
                lifecycle.run_phase_antigravity("predict", commit, lambda: self.fail("stopped ran"), receipt_dir=stopped, audit_path=parent / "stopped_audit.json", prediction_root=parent / "prediction", evaluation_root=parent / "evaluation")
            self.assertEqual(tuple((path.name, path.read_bytes()) for path in sorted(stopped.iterdir())), stopped_before)
            self.assertFalse((parent / "stopped_audit.json").exists())
            invalid = parent / "invalid"; invalid.mkdir(); (invalid / "foreign_antigravity.json").write_text("{}\n")
            invalid_before = (invalid / "foreign_antigravity.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "inventory is invalid"):
                lifecycle.run_phase_antigravity("readiness", commit, lambda: self.fail("invalid ran"), receipt_dir=invalid, audit_path=parent / "invalid_audit.json", prediction_root=parent / "prediction", evaluation_root=parent / "evaluation")
            self.assertEqual((invalid / "foreign_antigravity.json").read_bytes(), invalid_before)
            self.assertFalse((parent / "invalid_audit.json").exists())

    def test_occupied_audit_is_preserved_after_stopped_receipt_antigravity(self) -> None:
        """Break caught: occupied audit evidence is overwritten or normal phase work begins before stop."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import lifecycle_antigravity as lifecycle  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_lifecycle_audit_antigravity_") as temporary:
            parent = Path(temporary); audit, receipts = parent / "audit_antigravity.json", parent / "receipts"
            audit.write_bytes(b'{"foreign":true}\n')
            before = audit.read_bytes()
            with self.assertRaisesRegex(ValueError, "audit destination is occupied"):
                lifecycle.run_phase_antigravity("readiness", "1" * 40, lambda: self.fail("occupied audit ran"), receipt_dir=receipts, audit_path=audit, prediction_root=parent / "prediction", evaluation_root=parent / "evaluation")
            self.assertEqual(audit.read_bytes(), before)
            self.assertTrue(lifecycle.validate_receipt_inventory_antigravity(receipts).stopped)

    def test_operation_failure_appends_stopped_and_never_a_complete_leaf_antigravity(self) -> None:
        """Break caught: an operation error is followed by success receipt or omits its permanent stop."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import lifecycle_antigravity as lifecycle  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_lifecycle_failure_antigravity_") as temporary:
            parent = Path(temporary); receipts = parent / "receipts"
            self._seed_state_antigravity(lifecycle, receipts, "S1", "2" * 40)
            with self.assertRaisesRegex(RuntimeError, "work failure"):
                lifecycle.run_phase_antigravity("predict", "2" * 40, lambda: (_ for _ in ()).throw(RuntimeError("work failure")), receipt_dir=receipts, audit_path=parent / "audit_antigravity.json", prediction_root=parent / "prediction", evaluation_root=parent / "evaluation")
            self.assertTrue((receipts / "predict_started_antigravity.json").is_file())
            self.assertFalse((receipts / "predict_complete_antigravity.json").exists())
            self.assertTrue((receipts / "stopped_antigravity.json").is_file())

    def test_receipt_commit_and_predecessor_hash_are_descriptor_derived_antigravity(self) -> None:
        """Break caught: post-validation Path.read_bytes races replace receipt review or hash authority."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import lifecycle_antigravity as lifecycle  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_lifecycle_descriptor_antigravity_") as temporary:
            receipts = Path(temporary) / "receipts"
            lifecycle.append_receipt_antigravity(receipts, "readiness", "complete", "3" * 40)
            with patch.object(Path, "read_bytes", side_effect=AssertionError("path read forbidden")):
                self.assertEqual(lifecycle.receipt_reviewed_commit_antigravity(receipts), "3" * 40)
                lifecycle.append_receipt_antigravity(receipts, "predict", "started", "3" * 40)
            self.assertEqual(lifecycle.validate_receipt_inventory_antigravity(receipts).state, "S2")

    def test_predecessor_commit_mismatch_stops_using_owned_chain_commit_antigravity(self) -> None:
        """Break caught: S1/S3/S5 with a wrong lower40 leaves a valid receipt nonterminal."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import lifecycle_antigravity as lifecycle  # noqa: PLC0415

        expected = {"S1": "predict", "S3": "evaluate", "S5": "inspect"}
        with tempfile.TemporaryDirectory(prefix="exp449_lifecycle_commit_mismatch_antigravity_") as temporary:
            parent = Path(temporary)
            for state, phase in expected.items():
                with self.subTest(state=state, phase=phase):
                    receipts, audit = parent / state, parent / f"{state}.json"
                    self._seed_state_antigravity(lifecycle, receipts, state, "4" * 40)
                    with self.assertRaisesRegex(ValueError, "receipt chain"):
                        lifecycle.run_phase_antigravity(phase, "5" * 40, lambda: self.fail("mismatch ran"), receipt_dir=receipts, audit_path=audit, prediction_root=parent / "prediction", evaluation_root=parent / "evaluation")
                    self.assertTrue(lifecycle.validate_receipt_inventory_antigravity(receipts).stopped)
                    terminal = lifecycle.validate_receipt_inventory_antigravity(receipts)
                    self.assertEqual(terminal.reviewed_commit, "4" * 40)
                    self.assertTrue(audit.is_file())

    def test_receipt_writer_fault_matrix_never_reports_a_durable_success_antigravity(self) -> None:
        """Break caught: a create/write/fsync/reread/close failure returns a receipt as though it committed."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import lifecycle_antigravity as lifecycle  # noqa: PLC0415

        points = ("directory_create", "leaf_create", "leaf_write", "leaf_fsync", "leaf_close", "parent_fsync", "leaf_reread", "reread_close", "directory_close")
        with tempfile.TemporaryDirectory(prefix="exp449_lifecycle_fault_matrix_antigravity_") as temporary:
            parent = Path(temporary)
            for point in points:
                with self.subTest(point=point):
                    receipts = parent / point
                    def fail(active: str) -> None:
                        if active == point:
                            raise OSError(point)
                    with self.assertRaisesRegex(OSError, point):
                        lifecycle.append_receipt_antigravity(receipts, "readiness", "complete", "6" * 40, fault_hook=fail)

    def test_primary_receipt_failure_survives_close_failure_antigravity(self) -> None:
        """Break caught: a close fault replaces the earlier write primary error."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import lifecycle_antigravity as lifecycle  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_lifecycle_primary_antigravity_") as temporary:
            def fail(point: str) -> None:
                if point == "leaf_write":
                    raise RuntimeError("primary write")
                if point == "leaf_close":
                    raise OSError("secondary close")
            with self.assertRaisesRegex(RuntimeError, "primary write") as raised:
                lifecycle.append_receipt_antigravity(Path(temporary) / "receipts", "readiness", "complete", "7" * 40, fault_hook=fail)
            self.assertIn("leaf_close_error=OSError: secondary close", "\n".join(raised.exception.__notes__))

    def test_unsafe_foreign_partial_and_noncanonical_receipts_are_invalid_and_preserved_antigravity(self) -> None:
        """Break caught: symlink, multilink, foreign, partial, or noncanonical receipt state becomes owned."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import lifecycle_antigravity as lifecycle  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_lifecycle_invalid_matrix_antigravity_") as temporary:
            parent = Path(temporary)
            for defect in ("directory_symlink", "leaf_symlink", "leaf_hardlink", "foreign", "partial", "noncanonical"):
                with self.subTest(defect=defect):
                    root = parent / defect
                    if defect == "directory_symlink":
                        target = parent / "directory_target"; target.mkdir(); root.symlink_to(target, target_is_directory=True)
                    elif defect in ("foreign", "partial", "noncanonical"):
                        root.mkdir()
                        name = "foreign_antigravity.json" if defect == "foreign" else "readiness_complete_antigravity.json"
                        root.joinpath(name).write_bytes(b"{}\n" if defect != "noncanonical" else b'{"schema": "bad"}\n')
                    else:
                        lifecycle.append_receipt_antigravity(root, "readiness", "complete", "8" * 40)
                        leaf = root / "readiness_complete_antigravity.json"
                        if defect == "leaf_symlink":
                            target = parent / "leaf_target_antigravity.json"; target.write_bytes(leaf.read_bytes()); leaf.unlink(); leaf.symlink_to(target)
                        else:
                            os.link(leaf, parent / "extra_hardlink_antigravity.json")
                    before = None if root.is_symlink() else tuple((item.name, item.read_bytes()) for item in sorted(root.iterdir()))
                    result = lifecycle.validate_receipt_inventory_antigravity(root)
                    self.assertTrue(result.invalid)
                    if before is not None:
                        self.assertEqual(tuple((item.name, item.read_bytes()) for item in sorted(root.iterdir())), before)

    def test_durable_complete_returns_without_terminal_stop_antigravity(self) -> None:
        """Break caught: cleanup after a durable complete appends stopped or reclassifies success."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import lifecycle_antigravity as lifecycle  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_lifecycle_complete_boundary_antigravity_") as temporary:
            parent = Path(temporary); receipts = parent / "receipts"
            lifecycle.append_receipt_antigravity(receipts, "readiness", "complete", "9" * 40)
            self.assertEqual(lifecycle.run_phase_antigravity("predict", "9" * 40, lambda: "prediction", receipt_dir=receipts, audit_path=parent / "audit.json", prediction_root=parent / "prediction", evaluation_root=parent / "evaluation"), "prediction")
            self.assertEqual(lifecycle.validate_receipt_inventory_antigravity(receipts).state, "S3")
            self.assertFalse((receipts / "stopped_antigravity.json").exists())
            self.assertFalse((parent / "audit.json").exists())

    def test_append_rejects_same_name_receipt_directory_rebind_antigravity(self) -> None:
        """Break caught: a retained receipt directory is renamed and the new same-name directory receives authority."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import lifecycle_antigravity as lifecycle  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_lifecycle_rebind_antigravity_") as temporary:
            parent = Path(temporary); receipts = parent / "receipts"; moved = parent / "moved"
            lifecycle.append_receipt_antigravity(receipts, "readiness", "complete", "a" * 40)
            def replace(point: str) -> None:
                if point == "leaf_reread":
                    os.rename(receipts, moved); receipts.mkdir()
            with self.assertRaisesRegex(ValueError, "rebound"):
                lifecycle.append_receipt_antigravity(receipts, "predict", "started", "a" * 40, fault_hook=replace)
            self.assertTrue(lifecycle.validate_receipt_inventory_antigravity(receipts).invalid)
            self.assertEqual(lifecycle.validate_receipt_inventory_antigravity(moved).state, "S2")

    def test_audit_symlink_and_writer_failure_preserve_primary_antigravity(self) -> None:
        """Break caught: audit symlink is overwritten or audit persistence replaces the phase primary error."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import lifecycle_antigravity as lifecycle  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_lifecycle_audit_fault_antigravity_") as temporary:
            parent = Path(temporary); target = parent / "target_antigravity.json"; target.write_bytes(b"foreign\n")
            audit = parent / "audit_antigravity.json"; audit.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "occupied"):
                lifecycle.run_phase_antigravity("readiness", "b" * 40, lambda: self.fail("audit symlink ran"), receipt_dir=parent / "receipts", audit_path=audit, prediction_root=parent / "prediction", evaluation_root=parent / "evaluation")
            self.assertEqual(target.read_bytes(), b"foreign\n")
            self.assertTrue(audit.is_symlink())
            receipts = parent / "write_failure_receipts"
            original = lifecycle.write_failure_audit_antigravity
            with patch.object(lifecycle, "write_failure_audit_antigravity", side_effect=OSError("audit write")):
                with self.assertRaisesRegex(RuntimeError, "phase primary") as raised:
                    lifecycle.run_phase_antigravity("readiness", "c" * 40, lambda: (_ for _ in ()).throw(RuntimeError("phase primary")), receipt_dir=receipts, audit_path=parent / "write_failure.json", prediction_root=parent / "prediction", evaluation_root=parent / "evaluation")
            self.assertIn("failure_audit_error=OSError: audit write", "\n".join(raised.exception.__notes__))
            self.assertTrue(lifecycle.validate_receipt_inventory_antigravity(receipts).stopped)
            self.assertIsNotNone(original)

    def test_private_inspect_wrapper_is_ordered_and_source_free_seam_antigravity(self) -> None:
        """Break caught: inspect opens a source dependency or replays before receipt-derived prediction capture."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import lifecycle_antigravity as lifecycle  # noqa: PLC0415
        from experiments.exp449_residual_fn_method_qualification_antigravity import replay_antigravity as replay  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_lifecycle_inspect_seam_antigravity_") as temporary:
            parent = Path(temporary); receipts = parent / "receipts"; evaluation = parent / "evaluation"; evaluation.mkdir()
            self._seed_state_antigravity(lifecycle, receipts, "S5", "d" * 40)
            events: list[str] = []
            capture = SimpleNamespace(authority=SimpleNamespace(reviewed_commit="d" * 40))
            result = replay._inspect_with_dependencies_antigravity(
                lambda: events.append("receipt") or "d" * 40,
                lambda: events.append("capture") or capture,
                lambda root, value: events.append("replay") or "replayed",
                evaluation_root=evaluation, receipt_dir=receipts, audit_path=parent / "audit.json",
            )
            self.assertEqual(result, "replayed")
            self.assertEqual(events, ["receipt", "capture", "replay"])
            self.assertEqual(lifecycle.validate_receipt_inventory_antigravity(receipts).state, "S7")

    def test_private_predict_wrapper_commits_zero_label_immutable_root_antigravity(self) -> None:
        """Break caught: prediction opens a label dependency or completes before its root exists and is immutable."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import lifecycle_antigravity as lifecycle  # noqa: PLC0415
        from experiments.exp449_residual_fn_method_qualification_antigravity import prediction_antigravity as prediction  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_lifecycle_predict_seam_antigravity_") as temporary:
            parent = Path(temporary); receipts = parent / "receipts"; root = parent / "prediction"; events: list[str] = []
            self._seed_state_antigravity(lifecycle, receipts, "S1", "e" * 40)
            def operation():
                events.append("zero-label-operation"); root.mkdir(); (root / "immutable_antigravity.json").write_bytes(b"{}\n")
                return "authority"
            self.assertEqual(prediction._predict_with_lifecycle_dependencies_antigravity("e" * 40, operation, prediction_root=root, receipt_dir=receipts, audit_path=parent / "audit.json"), "authority")
            self.assertEqual(events, ["zero-label-operation"])
            self.assertEqual(lifecycle.validate_receipt_inventory_antigravity(receipts).state, "S3")
            self.assertFalse((parent / "audit.json").exists())

    def test_private_evaluate_wrapper_captures_prediction_before_evaluator_antigravity(self) -> None:
        """Break caught: evaluator/label opener is reachable before descriptor prediction capture or repeats capture."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import lifecycle_antigravity as lifecycle  # noqa: PLC0415
        from experiments.exp449_residual_fn_method_qualification_antigravity import evaluation_antigravity as evaluation  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_lifecycle_evaluate_seam_antigravity_") as temporary:
            parent = Path(temporary); receipts = parent / "receipts"; prediction_root = parent / "prediction"; prediction_root.mkdir(); (prediction_root / "leaf.json").write_bytes(b"{}\n")
            evaluation_root = parent / "evaluation"; events: list[str] = []
            self._seed_state_antigravity(lifecycle, receipts, "S3", "f" * 40)
            capture = SimpleNamespace(authority=SimpleNamespace(reviewed_commit="f" * 40))
            def evaluator(value):
                self.assertIs(value, capture); events.append("evaluator"); evaluation_root.mkdir(); (evaluation_root / "leaf.json").write_bytes(b"{}\n"); return "evaluation"
            self.assertEqual(evaluation._evaluate_with_lifecycle_dependencies_antigravity("f" * 40, lambda: events.append("capture") or capture, evaluator, evaluation_root=evaluation_root, receipt_dir=receipts, audit_path=parent / "audit.json"), "evaluation")
            self.assertEqual(events, ["capture", "evaluator"])
            self.assertEqual(lifecycle.validate_receipt_inventory_antigravity(receipts).state, "S5")

    def test_private_inspect_subprocess_firewall_keeps_forbidden_modules_unloaded_antigravity(self) -> None:
        """Break caught: actual synthetic inspect dispatch imports sources, sqlite, evaluator, aeon, network, or dashboard."""
        script = r'''
import json, tempfile
from pathlib import Path
from types import SimpleNamespace
from experiments.exp449_residual_fn_method_qualification_antigravity import lifecycle_antigravity as life
from experiments.exp449_residual_fn_method_qualification_antigravity import replay_antigravity as replay
with tempfile.TemporaryDirectory() as raw:
    parent=Path(raw); receipts=parent/'receipts'; evaluation=parent/'evaluation'; evaluation.mkdir()
    for phase,event in (("readiness","complete"),("predict","started"),("predict","complete"),("evaluate","started"),("evaluate","complete")):
        life.append_receipt_antigravity(receipts, phase, event, 'a'*40)
    capture=SimpleNamespace(authority=SimpleNamespace(reviewed_commit='a'*40))
    result=replay._inspect_with_dependencies_antigravity(lambda:'a'*40, lambda:capture, lambda root,value:'ok', evaluation_root=evaluation, receipt_dir=receipts, audit_path=parent/'audit.json')
    import sys
    forbidden=('sqlite3','experiments.exp449_residual_fn_method_qualification_antigravity.sources_antigravity','experiments.exp449_residual_fn_method_qualification_antigravity.evaluation_antigravity','aeon')
    print(json.dumps({'result':result,'loaded':[name for name in forbidden if name in sys.modules]}))
'''
        completed = subprocess.run([sys.executable, "-c", script], cwd=Path.cwd(), check=True, text=True, capture_output=True)
        observed = json.loads(completed.stdout)
        self.assertEqual(observed, {"result": "ok", "loaded": []})

    def test_predict_wrapper_failure_uses_one_unified_audit_not_legacy_inner_audit_antigravity(self) -> None:
        """Break caught: receipt-gated predict invokes legacy audit or duplicates the unified failure audit."""
        from experiments.exp449_residual_fn_method_qualification_antigravity import lifecycle_antigravity as lifecycle  # noqa: PLC0415
        from experiments.exp449_residual_fn_method_qualification_antigravity import prediction_antigravity as prediction  # noqa: PLC0415

        with tempfile.TemporaryDirectory(prefix="exp449_lifecycle_predict_audit_antigravity_") as temporary:
            parent = Path(temporary); receipts = parent / "receipts"; root = parent / "prediction"; audit = parent / "audit.json"
            self._seed_state_antigravity(lifecycle, receipts, "S1", "1" * 40)
            calls: list[str] = []
            original = lifecycle.write_failure_audit_antigravity
            def unified(*args, **kwargs):
                calls.append("unified"); return original(*args, **kwargs)
            with patch.object(prediction, "_handle_lifecycle_failure_antigravity", side_effect=AssertionError("legacy audit called")), patch.object(lifecycle, "write_failure_audit_antigravity", side_effect=unified):
                with self.assertRaisesRegex(RuntimeError, "operation failure"):
                    prediction._predict_with_lifecycle_dependencies_antigravity("1" * 40, lambda: (_ for _ in ()).throw(RuntimeError("operation failure")), prediction_root=root, receipt_dir=receipts, audit_path=audit)
            self.assertEqual(calls, ["unified"])
            self.assertTrue(audit.is_file())
            self.assertTrue(lifecycle.validate_receipt_inventory_antigravity(receipts).stopped)
