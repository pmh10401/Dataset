"""Single-use Exp446 replay boundaries on deterministic values-only fixtures."""
from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from experiments.exp444_train_only_stability_precision_gate_antigravity.checkpoint_antigravity import ProcessIdentityAntigravity
from experiments.exp444_train_only_stability_precision_gate_antigravity.contracts_antigravity import (
    CandidateInputAntigravity, CandidateStabilityResultAntigravity,
    DepthViewScoreAntigravity, OmissionMaskAntigravity, SourceCandidateAntigravity,
    StabilityViewAntigravity,
)


_COMMIT_ANTIGRAVITY = "a" * 40


def _inputs_antigravity() -> tuple[CandidateInputAntigravity, ...]:
    return tuple(
        CandidateInputAntigravity(
            SourceCandidateAntigravity(f"Fixture_{index:02d}", index, 8, (1, 2), index + 1),
            (0, 1, 2, 3), np.arange(32, dtype="<f4").reshape(4, 8), np.arange(8, dtype="<f4"),
        )
        for index in range(30)
    )


def _result_antigravity(item: CandidateInputAntigravity) -> CandidateStabilityResultAntigravity:
    scores = (
        DepthViewScoreAntigravity(1, 2.0, 1.0, True, 2.0, 0.5),
        DepthViewScoreAntigravity(2, 2.0, 1.0, True, 2.0, 0.5),
    )
    view = StabilityViewAntigravity(
        OmissionMaskAntigravity(0, (0,), hashlib.sha256(b"fixture-mask").hexdigest()),
        (1, 2, 3), scores, (1, 2), True,
    )
    return CandidateStabilityResultAntigravity(item.candidate, (view,), 1, 1, "stable_review")


def _wrapper_sha_antigravity(index: int, item: CandidateInputAntigravity) -> str:
    payload = {"sequence": index, "result": _result_antigravity(item).to_payload()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n").hexdigest()


def _authority_antigravity(inputs: tuple[CandidateInputAntigravity, ...]) -> SimpleNamespace:
    prefix = tuple(_wrapper_sha_antigravity(index, item) for index, item in enumerate(inputs))
    aggregate = hashlib.sha256(json.dumps(list(prefix), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n").hexdigest()
    return SimpleNamespace(record_sha256_prefix=prefix, record_prefix_payload_sha256=aggregate)


class Exp446LifecycleAntigravityTests(unittest.TestCase):
    """Each test catches a missing parity or single-use lifecycle branch."""

    def _patched_profile_antigravity(self, profile: object, root: Path) -> object:
        patched = object.__new__(type(profile))
        for name in (
            "experiment_id", "schema", "run_identity", "reviewed_design_path",
            "preregistration_path", "predecessor_required",
        ):
            object.__setattr__(patched, name, getattr(profile, name))
        object.__setattr__(patched, "run_root", root)
        return patched

    def _run_antigravity(
        self, *, changed_index: int | None = None, changed_aggregate: bool = False,
        changed_after_readiness: bool = False, labels: int = 0, workers: int = 11,
        close_fault: bool = False, preexisting_root: str | None = None,
        ordered_failure: BaseException | None = None,
        write_fault: str | None = None, reviewed_commit: str = _COMMIT_ANTIGRAVITY,
        commit_error: BaseException | None = None,
    ) -> tuple[Path, object]:
        """Exercise the real checkpoint/progress path with a 30-record CPU fixture."""
        from experiments.exp446_exp445_publication_materialization_repair_antigravity import lifecycle_antigravity
        from experiments.exp444_train_only_stability_precision_gate_antigravity import runtime_antigravity as runtime

        inputs = _inputs_antigravity()
        exp444 = _authority_antigravity(inputs)
        exp445 = _authority_antigravity(inputs)
        if changed_index is not None:
            prefix = list(exp445.record_sha256_prefix)
            prefix[changed_index] = "f" * 64
            exp445 = SimpleNamespace(
                record_sha256_prefix=tuple(prefix),
                record_prefix_payload_sha256=exp445.record_prefix_payload_sha256,
            )
        if changed_aggregate:
            exp445 = SimpleNamespace(
                record_sha256_prefix=exp445.record_sha256_prefix,
                record_prefix_payload_sha256="e" * 64,
            )
        temporary = Path(tempfile.mkdtemp(prefix="exp446_lifecycle_antigravity_"))
        self.addCleanup(shutil.rmtree, temporary, ignore_errors=True)
        with self.subTest(fixture_root=temporary):
            root = temporary / "exp446_exp445_publication_materialization_repair_antigravity_20260811"
            profile = self._patched_profile_antigravity(lifecycle_antigravity._EXP446_PROFILE_ANTIGRAVITY, root)
            if preexisting_root == "directory":
                root.mkdir()
            elif preexisting_root == "symlink":
                target = temporary / "existing_target_antigravity"
                target.mkdir()
                root.symlink_to(target, target_is_directory=True)
            calls = 0
            def verify_exp445():
                nonlocal calls
                calls += 1
                if changed_after_readiness and calls > 1:
                    return SimpleNamespace(
                        record_sha256_prefix=exp445.record_sha256_prefix,
                        record_prefix_payload_sha256="d" * 64,
                    )
                return exp445
            def ordered(active, *, commit, progress, **kwargs):
                self.assertEqual((kwargs["workers"], kwargs["distance_backend"]), (11, "cpu"))
                if ordered_failure is not None:
                    raise ordered_failure
                for index, item in enumerate(active):
                    progress(stage="active", sequence=index)
                    commit(index, item, _result_antigravity(item))
                return tuple(_result_antigravity(item) for item in active)
            with ExitStack() as patches:
                for patch in (
                mock.patch.object(lifecycle_antigravity, "_EXP446_PROFILE_ANTIGRAVITY", profile),
                mock.patch.object(runtime, "EXP446_PROFILE_ANTIGRAVITY", profile),
                mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity.EXP446_PROFILE_ANTIGRAVITY", profile),
                mock.patch.object(runtime, "readiness_for_profile_antigravity"),
                mock.patch.object(
                    runtime, "_validate_reviewed_commit_antigravity",
                    side_effect=commit_error,
                    return_value=_COMMIT_ANTIGRAVITY,
                ),
                mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(99446, "exp446-test")),
                mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, labels)),
                mock.patch.object(runtime, "production_worker_count_antigravity", return_value=workers),
                mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=ordered),
                mock.patch.object(runtime, "_verify_exp445_predecessor_antigravity", return_value=exp444),
                mock.patch.object(runtime, "_verify_exp446_predecessor_antigravity", side_effect=verify_exp445),
                mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.provenance_antigravity.verify_exp444_provenance_antigravity"),
                ):
                    patches.enter_context(patch)
                if close_fault:
                    from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import ProgressWriterAntigravity
                    patches.enter_context(mock.patch.object(ProgressWriterAntigravity, "close_antigravity", side_effect=OSError("terminal close fault")))
                if write_fault is not None:
                    from experiments.exp444_train_only_stability_precision_gate_antigravity import checkpoint_antigravity, progress_antigravity
                    if write_fault == "marker":
                        patches.enter_context(mock.patch.object(
                            checkpoint_antigravity.Exp444CheckpointStoreAntigravity,
                            "write_terminal_parity_antigravity", side_effect=OSError("marker write fault"),
                        ))
                    else:
                        method = {
                            "progress": "_write_progress_antigravity",
                            "pid": "_write_pid_antigravity",
                            "event": "_append_event_antigravity",
                            "text": "_append_text_antigravity",
                        }[write_fault]
                        patches.enter_context(mock.patch.object(
                            progress_antigravity.ProgressWriterAntigravity, method,
                            side_effect=OSError(f"{write_fault} write fault"),
                        ))
                try:
                    outcome = lifecycle_antigravity.run_new_antigravity(reviewed_commit)
                except BaseException as error:
                    outcome = error
            return root, outcome

    def _require_terminal_parity_antigravity(self, root: Path) -> None:
        """Exercise the public eligibility gate against an already-closed fixture."""
        from experiments.exp446_exp445_publication_materialization_repair_antigravity import lifecycle_antigravity

        inputs = _inputs_antigravity()
        profile = self._patched_profile_antigravity(lifecycle_antigravity._EXP446_PROFILE_ANTIGRAVITY, root)
        with (
            mock.patch.object(lifecycle_antigravity, "_EXP446_PROFILE_ANTIGRAVITY", profile),
            mock.patch(
                "experiments.exp445_exp444_publication_source_authority_repair_antigravity.predecessor_antigravity.verify_exp444_predecessor_antigravity",
                return_value=_authority_antigravity(inputs),
            ),
            mock.patch(
                "experiments.exp446_exp445_publication_materialization_repair_antigravity.predecessor_antigravity.verify_exp445_predecessor_antigravity",
                return_value=_authority_antigravity(inputs),
            ),
        ):
            lifecycle_antigravity._require_terminal_parity_antigravity()

    def _assert_exact_error_antigravity(
        self, root: Path, outcome: BaseException, *, sequence: int, message: str,
    ) -> None:
        """The fixed profile persists one exact-next error and aligned inactive progress."""
        self.assertEqual(str(outcome), message)
        attempt = root / "attempts_antigravity/attempt_0000_antigravity"
        record = json.loads(
            (attempt / f"checkpoint_antigravity/records_antigravity/{sequence:04d}_record_antigravity.json").read_text()
        )
        state = json.loads((attempt / "checkpoint_antigravity/state_antigravity.json").read_text())
        progress = json.loads((attempt / "progress_antigravity.json").read_text())
        pid = json.loads((attempt / "pid_antigravity.json").read_text())
        self.assertEqual(record, {
            "sequence": sequence, "result": None, "status": "error",
            "error": {"type": type(outcome).__name__, "message": message},
        })
        self.assertEqual(
            (state["status"], state["completed_units"], state["error_count"], state["error"]),
            ("error", sequence + 1, 1, record["error"]),
        )
        self.assertEqual(
            (progress["stage"], progress["active"], progress["completed"], progress["error"], progress["latest_error"], pid["active"]),
            ("error", False, sequence + 1, 1, message, False),
        )

    def test_all_thirty_wrappers_require_exp444_and_exp445_parity_before_success_checkpoint_antigravity(self) -> None:
        """Removing either authority check must reject an otherwise valid wrapper."""
        root, outcome = self._run_antigravity()
        self.assertEqual((outcome.status, outcome.completed_count, outcome.workers, outcome.label_access_count), ("complete", 30, 11, 0))
        self.assertTrue((root / "attempts_antigravity/attempt_0000_antigravity/terminal_parity_antigravity.json").is_file())

    def test_wrapper_mismatch_at_first_middle_and_last_index_preserves_exact_next_error_antigravity(self) -> None:
        """Skipping an index gate would allow a changed wrapper to become a success record."""
        for index in (0, 15, 29):
            with self.subTest(index=index):
                root, outcome = self._run_antigravity(changed_index=index)
                self.assertIsInstance(outcome, ValueError)
                self.assertIn("record parity mismatch", str(outcome))
                attempt = root / "attempts_antigravity/attempt_0000_antigravity"
                state = json.loads((attempt / "checkpoint_antigravity/state_antigravity.json").read_text())
                progress = json.loads((attempt / "progress_antigravity.json").read_text())
                self.assertEqual((state["status"], state["completed_units"], progress["stage"], progress["active"]), ("error", index + 1, "error", False))
                self.assertFalse((attempt / "publication_antigravity").exists())

    def test_aggregate_or_final_predecessor_change_blocks_terminal_complete_antigravity(self) -> None:
        """Dropping either final gate would create terminal eligibility from stale authority."""
        for kwargs in ({"changed_aggregate": True}, {"changed_after_readiness": True}):
            with self.subTest(kwargs=kwargs):
                root, outcome = self._run_antigravity(**kwargs)
                self.assertIsInstance(outcome, ValueError)
                self.assertIn("predecessor", str(outcome))
                state = json.loads((root / "attempts_antigravity/attempt_0000_antigravity/checkpoint_antigravity/state_antigravity.json").read_text())
                self.assertEqual(state["status"], "error")

    def test_worker_and_label_contracts_fail_before_root_allocation_antigravity(self) -> None:
        """Relaxing fixed CPU/11 or zero-label rules would allocate an invalid run root."""
        for kwargs in ({"workers": 10}, {"labels": 1}):
            with self.subTest(kwargs=kwargs):
                root, outcome = self._run_antigravity(**kwargs)
                self.assertIsInstance(outcome, ValueError)
                self.assertFalse(root.exists())

    def test_publication_is_rejected_before_terminal_parity_antigravity(self) -> None:
        """A publication route that imports its writer first could bypass terminal parity."""
        from experiments.exp446_exp445_publication_materialization_repair_antigravity import lifecycle_antigravity
        with tempfile.TemporaryDirectory(prefix="exp446_publication_gate_antigravity_") as temporary:
            profile = self._patched_profile_antigravity(
                lifecycle_antigravity._EXP446_PROFILE_ANTIGRAVITY,
                Path(temporary) / "exp446_exp445_publication_materialization_repair_antigravity_20260811",
            )
            with mock.patch.object(lifecycle_antigravity, "_EXP446_PROFILE_ANTIGRAVITY", profile):
                with self.assertRaisesRegex(ValueError, "terminal parity"):
                    lifecycle_antigravity.publish_prediction_antigravity()

    def test_terminal_close_fault_never_leaves_publication_marker_antigravity(self) -> None:
        """Creating eligibility before terminal close would expose a half-closed attempt."""
        root, outcome = self._run_antigravity(close_fault=True)
        self.assertIsInstance(outcome, OSError)
        self.assertFalse((root / "attempts_antigravity/attempt_0000_antigravity/terminal_parity_antigravity.json").exists())

    def test_existing_and_symlink_root_refuse_single_use_retry_antigravity(self) -> None:
        """Removing the root check would permit a same-root retry."""
        for root_kind in ("directory", "symlink"):
            with self.subTest(root_kind=root_kind):
                root, outcome = self._run_antigravity(preexisting_root=root_kind)
                self.assertIsInstance(outcome, ValueError)
                self.assertEqual(str(outcome), "Exp444 new run root must be absent and antigravity-namespaced")
                self.assertTrue(root.exists() or root.is_symlink())

    def test_wrong_commit_rejects_before_temp_root_allocation_antigravity(self) -> None:
        """The commit gate must run before a failed caller can consume the single-use root."""
        root, outcome = self._run_antigravity(
            reviewed_commit="b" * 40,
            commit_error=ValueError("reviewed commit is not the approved clean head"),
        )
        self.assertIsInstance(outcome, ValueError)
        self.assertEqual(str(outcome), "reviewed commit is not the approved clean head")
        self.assertFalse(root.exists())

    def test_worker_and_scorer_failure_preserve_exact_next_error_content_antigravity(self) -> None:
        """A worker exception must not become an unbound terminal marker or misaligned progress."""
        for failure in (RuntimeError("worker fixture fault"), ValueError("scorer fixture fault")):
            with self.subTest(failure=type(failure).__name__):
                root, outcome = self._run_antigravity(ordered_failure=failure)
                self.assertIsInstance(outcome, type(failure))
                self._assert_exact_error_antigravity(root, outcome, sequence=0, message=str(failure))
                self.assertFalse((root / "attempts_antigravity/attempt_0000_antigravity/terminal_parity_antigravity.json").exists())

    def test_terminal_marker_rejects_forged_or_noncanonical_terminal_evidence_antigravity(self) -> None:
        """A marker must authenticate canonical, complete terminal evidence rather than its own claims."""
        for corruption in ("forged", "marker_symlink", "extra_marker_key", "missing_marker", "mutated_marker", "leaf_symlink", "mutated_leaf"):
            with self.subTest(corruption=corruption):
                root, outcome = self._run_antigravity()
                self.assertEqual(outcome.status, "complete")
                attempt = root / "attempts_antigravity/attempt_0000_antigravity"
                marker = attempt / "terminal_parity_antigravity.json"
                record = attempt / "checkpoint_antigravity/records_antigravity/0000_record_antigravity.json"
                if corruption == "forged":
                    record.write_bytes(record.read_bytes() + b" ")
                    payload = json.loads(marker.read_text())
                    payload["terminal_sha256"]["checkpoint_antigravity/records_antigravity/0000_record_antigravity.json"] = hashlib.sha256(record.read_bytes()).hexdigest()
                    marker.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                elif corruption == "marker_symlink":
                    target = attempt / "marker_target_antigravity.json"
                    marker.replace(target)
                    marker.symlink_to(target.name)
                elif corruption == "extra_marker_key":
                    payload = json.loads(marker.read_text())
                    payload["forged"] = True
                    marker.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                elif corruption == "missing_marker":
                    marker.unlink()
                elif corruption == "mutated_marker":
                    marker.write_text("{}\n")
                elif corruption == "leaf_symlink":
                    target = record.with_name("record_target_antigravity.json")
                    record.replace(target)
                    record.symlink_to(target.name)
                else:
                    record.write_bytes(record.read_bytes() + b" ")
                with self.assertRaisesRegex(ValueError, "terminal parity"):
                    self._require_terminal_parity_antigravity(root)

    def test_write_fault_matrix_never_exposes_terminal_marker_antigravity(self) -> None:
        """Each terminal-evidence channel must fail closed without a publication capability."""
        for channel in ("progress", "pid", "event", "text", "marker"):
            with self.subTest(channel=channel):
                root, outcome = self._run_antigravity(write_fault=channel)
                self.assertIsInstance(outcome, OSError)
                attempt = root / "attempts_antigravity/attempt_0000_antigravity"
                self.assertFalse((attempt / "terminal_parity_antigravity.json").exists())
                if channel == "marker":
                    progress = json.loads((attempt / "progress_antigravity.json").read_text())
                    pid = json.loads((attempt / "pid_antigravity.json").read_text())
                    self.assertEqual((progress["stage"], progress["active"], pid["active"]), ("complete", False, False))

    def test_valid_terminal_marker_requires_inactive_complete_progress_and_pid_antigravity(self) -> None:
        """A valid eligibility marker is paired with inactive, complete terminal ownership evidence."""
        root, outcome = self._run_antigravity()
        self.assertEqual(outcome.status, "complete")
        attempt = root / "attempts_antigravity/attempt_0000_antigravity"
        progress = json.loads((attempt / "progress_antigravity.json").read_text())
        pid = json.loads((attempt / "pid_antigravity.json").read_text())
        self.assertEqual((progress["stage"], progress["active"], progress["completed"], pid["active"]), ("complete", False, 30, False))
        self._require_terminal_parity_antigravity(root)


if __name__ == "__main__":
    unittest.main()
