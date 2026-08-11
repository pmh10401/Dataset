"""Single-use Exp446 replay boundaries on deterministic values-only fixtures."""
from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
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
_CAPTURED_POOLS_ANTIGRAVITY: list["_CapturingProcessPoolExecutorAntigravity"] = []


class _CapturingProcessPoolExecutorAntigravity(ProcessPoolExecutor):
    """Real executor that retains child handles across its parent's shutdown call."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.child_handles_antigravity = []
        _CAPTURED_POOLS_ANTIGRAVITY.append(self)

    def shutdown(self, *args, **kwargs) -> None:
        if self._processes is not None:
            self.child_handles_antigravity.extend(self._processes.values())
        return super().shutdown(*args, **kwargs)


def _checkpoint_tree_bytes_antigravity(root: Path) -> dict[str, bytes]:
    """Raw complete checkpoint authority, independent of JSON interpretation."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


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


def _child_value_error_antigravity(item: CandidateInputAntigravity) -> CandidateStabilityResultAntigravity:
    """Picklable spawned-worker fixture: errors must cross a real Future."""
    if item.candidate.instance_index == 0:
        raise ValueError("scorer child fixture fault")
    time.sleep(0.5)
    return _result_antigravity(item)


def _child_transport_crash_antigravity(item: CandidateInputAntigravity) -> CandidateStabilityResultAntigravity:
    """Picklable spawned-worker fixture: one child exits before serializing a result."""
    if item.candidate.instance_index == 0:
        os._exit(47)
    time.sleep(0.5)
    return _result_antigravity(item)


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
        failure_seam: str | None = None,
        real_scheduler_seam: str | None = None,
        capture_pool: bool = False, checkpoint_capture: list[dict[str, bytes]] | None = None,
        write_fault: str | None = None, reviewed_commit: str = _COMMIT_ANTIGRAVITY,
        commit_error: BaseException | None = None, reuse_root: Path | None = None,
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
            root = temporary / "exp446_exp445_publication_materialization_repair_antigravity_20260811" if reuse_root is None else reuse_root
            profile = self._patched_profile_antigravity(lifecycle_antigravity._EXP446_PROFILE_ANTIGRAVITY, root)
            if preexisting_root == "directory":
                root.mkdir()
            elif preexisting_root == "symlink":
                target = temporary / "existing_target_antigravity"
                target.mkdir()
                root.symlink_to(target, target_is_directory=True)
            calls = 0
            checkpoint_root = root / "attempts_antigravity/attempt_0000_antigravity/checkpoint_antigravity"
            def capture_checkpoint_before_fault() -> None:
                if checkpoint_capture is not None:
                    checkpoint_capture.append(_checkpoint_tree_bytes_antigravity(checkpoint_root))
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
                if failure_seam == "worker_dispatch":
                    raise runtime.WorkerFailureAntigravity(0, active[0], RuntimeError("worker dispatch fixture fault"))
                if failure_seam == "scorer":
                    try:
                        runtime.score_stability_candidate_antigravity(active[0])
                    except BaseException as error:
                        raise runtime.WorkerFailureAntigravity(0, active[0], error) from error
                for index, item in enumerate(active):
                    progress(stage="active", sequence=index)
                    commit(index, item, _result_antigravity(item))
                return tuple(_result_antigravity(item) for item in active)
            with ExitStack() as patches:
                patch_list = [
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
                mock.patch.object(
                    runtime, "score_stability_candidate_antigravity",
                    side_effect=ValueError("scorer fixture fault") if (failure_seam == "scorer" or real_scheduler_seam == "scorer") else runtime.score_stability_candidate_antigravity,
                ),
                mock.patch.object(runtime, "_verify_exp445_predecessor_antigravity", return_value=exp444),
                mock.patch.object(runtime, "_verify_exp446_predecessor_antigravity", side_effect=verify_exp445),
                mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.provenance_antigravity.verify_exp444_provenance_antigravity"),
                ]
                if real_scheduler_seam is None:
                    patch_list.append(mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=ordered))
                else:
                    scorer = (
                        _child_transport_crash_antigravity
                        if real_scheduler_seam == "worker_transport"
                        else _child_value_error_antigravity
                    )
                    patch_list.append(mock.patch.object(runtime, "_score_candidate_worker_antigravity", scorer))
                for patch in patch_list:
                    patches.enter_context(patch)
                if capture_pool:
                    patches.enter_context(mock.patch.object(runtime, "ProcessPoolExecutor", _CapturingProcessPoolExecutorAntigravity))
                if close_fault:
                    from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import ProgressWriterAntigravity
                    patches.enter_context(mock.patch.object(ProgressWriterAntigravity, "close_antigravity", side_effect=OSError("terminal close fault")))
                if write_fault is not None:
                    from experiments.exp444_train_only_stability_precision_gate_antigravity import checkpoint_antigravity, progress_antigravity
                    if write_fault == "marker":
                        def marker_fault(*args, **kwargs):
                            capture_checkpoint_before_fault()
                            raise OSError("marker write fault")
                        patches.enter_context(mock.patch.object(
                            checkpoint_antigravity.Exp444CheckpointStoreAntigravity,
                            "write_terminal_parity_antigravity", side_effect=marker_fault,
                        ))
                    else:
                        persistent = write_fault.startswith("persistent_")
                        channel = write_fault.removeprefix("persistent_")
                        fired = False
                        if channel == "progress":
                            original = progress_antigravity._write_atomic_json_antigravity
                            def terminal_progress_fault(path, payload):
                                nonlocal fired
                                if (persistent or not fired) and path.name == "progress_antigravity.json" and payload.get("stage") == "complete":
                                    fired = True
                                    capture_checkpoint_before_fault()
                                    raise OSError("progress write fault")
                                return original(path, payload)
                            patches.enter_context(mock.patch.object(
                                progress_antigravity, "_write_atomic_json_antigravity", new=terminal_progress_fault,
                            ))
                        elif channel == "pid":
                            original = progress_antigravity.ProgressWriterAntigravity._write_pid_antigravity
                            def terminal_pid_fault(writer, active):
                                nonlocal fired
                                if (persistent or not fired) and active is False:
                                    fired = True
                                    capture_checkpoint_before_fault()
                                    raise OSError("pid write fault")
                                return original(writer, active)
                            patches.enter_context(mock.patch.object(
                                progress_antigravity.ProgressWriterAntigravity, "_write_pid_antigravity", new=terminal_pid_fault,
                            ))
                        elif channel == "event":
                            original = progress_antigravity.ProgressWriterAntigravity._append_event_antigravity
                            def terminal_event_fault(writer, event, stage, extra=None):
                                nonlocal fired
                                if (persistent or not fired) and event == "terminal":
                                    fired = True
                                    capture_checkpoint_before_fault()
                                    raise OSError("event write fault")
                                return original(writer, event, stage, extra)
                            patches.enter_context(mock.patch.object(
                                progress_antigravity.ProgressWriterAntigravity, "_append_event_antigravity", new=terminal_event_fault,
                            ))
                        elif channel == "text":
                            original = progress_antigravity.ProgressWriterAntigravity._append_text_antigravity
                            def terminal_text_fault(writer, line):
                                nonlocal fired
                                if (persistent or not fired) and line.startswith("종료 "):
                                    fired = True
                                    capture_checkpoint_before_fault()
                                    raise OSError("text write fault")
                                return original(writer, line)
                            patches.enter_context(mock.patch.object(
                                progress_antigravity.ProgressWriterAntigravity, "_append_text_antigravity", new=terminal_text_fault,
                            ))
                        else:
                            raise AssertionError(f"unexpected terminal write fault: {write_fault}")
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
                self._assert_exact_error_antigravity(root, outcome, sequence=index, message=str(outcome))
                attempt = root / "attempts_antigravity/attempt_0000_antigravity"
                self.assertFalse((attempt / "publication_antigravity").exists())

    def test_aggregate_or_final_predecessor_change_blocks_terminal_complete_antigravity(self) -> None:
        """Dropping either final gate would create terminal eligibility from stale authority."""
        for kwargs in ({"changed_aggregate": True}, {"changed_after_readiness": True}):
            with self.subTest(kwargs=kwargs):
                root, outcome = self._run_antigravity(**kwargs)
                self.assertIsInstance(outcome, ValueError)
                self.assertIn("predecessor", str(outcome))
                self._assert_exact_error_antigravity(root, outcome, sequence=29, message=str(outcome))

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
        from experiments.exp444_train_only_stability_precision_gate_antigravity import reporting_antigravity
        with tempfile.TemporaryDirectory(prefix="exp446_publication_gate_antigravity_") as temporary:
            profile = self._patched_profile_antigravity(
                lifecycle_antigravity._EXP446_PROFILE_ANTIGRAVITY,
                Path(temporary) / "exp446_exp445_publication_materialization_repair_antigravity_20260811",
            )
            with (
                mock.patch.object(lifecycle_antigravity, "_EXP446_PROFILE_ANTIGRAVITY", profile),
                mock.patch.object(reporting_antigravity, "EXP446_PROFILE_ANTIGRAVITY", profile),
            ):
                with self.assertRaisesRegex(ValueError, "terminal parity"):
                    lifecycle_antigravity.publish_prediction_antigravity()
            self.assertFalse(profile.run_root.exists() or profile.run_root.is_symlink())

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
        """A real spawned executor preserves transport and scorer failures in durable order."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import runtime_antigravity as runtime
        from concurrent.futures.process import BrokenProcessPool

        for seam in ("worker_transport", "scorer"):
            with self.subTest(seam=seam):
                root, outcome = self._run_antigravity(real_scheduler_seam=seam)
                self.assertIsInstance(outcome, runtime.WorkerFailureAntigravity)
                self._assert_exact_error_antigravity(root, outcome, sequence=0, message=str(outcome))
                self.assertIsInstance(outcome.__cause__, BrokenProcessPool if seam == "worker_transport" else ValueError)
                if seam == "scorer":
                    self.assertEqual(str(outcome.__cause__), "scorer child fixture fault")
                self.assertFalse((root / "attempts_antigravity/attempt_0000_antigravity/terminal_parity_antigravity.json").exists())

    def test_real_process_pool_failure_reaps_all_child_handles_antigravity(self) -> None:
        """Returning before executor cleanup would leave real delayed children alive."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import runtime_antigravity as runtime

        for seam in ("worker_transport", "scorer"):
            with self.subTest(seam=seam):
                _CAPTURED_POOLS_ANTIGRAVITY.clear()
                root, outcome = self._run_antigravity(real_scheduler_seam=seam, capture_pool=True)
                self.assertIsInstance(outcome, runtime.WorkerFailureAntigravity)
                self.assertEqual(len(_CAPTURED_POOLS_ANTIGRAVITY), 1)
                handles = _CAPTURED_POOLS_ANTIGRAVITY[0].child_handles_antigravity
                self.assertGreater(len(handles), 0)
                for process in handles:
                    self.assertIsNotNone(process.pid)
                    self.assertFalse(process.is_alive())
                    self.assertIsNotNone(process.exitcode)
                self.assertFalse((root / "attempts_antigravity/attempt_0000_antigravity/terminal_parity_antigravity.json").exists())

    def test_terminal_marker_rejects_forged_or_noncanonical_terminal_evidence_antigravity(self) -> None:
        """A marker must authenticate canonical, complete terminal evidence rather than its own claims."""
        for corruption in (
            "forged", "marker_symlink", "extra_marker_key", "missing_marker", "mutated_marker",
            "leaf_symlink", "mutated_leaf", "extra_state_key", "extra_progress_key",
            "extra_pid_key", "extra_terminal_event_key", "altered_terminal_text",
            "state_process_wrong_type", "progress_metric_wrong_type", "pid_heartbeat_wrong_type",
            "forged_preceding_event", "prefixed_text",
            "state_completed_float", "state_total_bool", "progress_completed_float",
            "progress_total_bool", "event_pid_float", "event_completed_bool",
            "event_total_float", "event_candidate_index_bool", "forged_event_dataset_pair",
            "missing_event_final_lf", "event_crlf", "missing_text_final_lf", "text_crlf",
        ):
            with self.subTest(corruption=corruption):
                root, outcome = self._run_antigravity()
                self.assertEqual(outcome.status, "complete")
                attempt = root / "attempts_antigravity/attempt_0000_antigravity"
                marker = attempt / "terminal_parity_antigravity.json"
                record = attempt / "checkpoint_antigravity/records_antigravity/0000_record_antigravity.json"
                state = attempt / "checkpoint_antigravity/state_antigravity.json"
                progress = attempt / "progress_antigravity.json"
                pid = attempt / "pid_antigravity.json"
                events = attempt / "events_antigravity.jsonl"
                text = attempt / "text_antigravity.log"
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
                elif corruption == "extra_state_key":
                    payload = json.loads(state.read_text()); payload["forged"] = True
                    state.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                    payload = json.loads(marker.read_text())
                    payload["terminal_sha256"]["checkpoint_antigravity/state_antigravity.json"] = hashlib.sha256(state.read_bytes()).hexdigest()
                    marker.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                elif corruption == "extra_progress_key":
                    payload = json.loads(progress.read_text()); payload["forged"] = True
                    progress.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                    payload = json.loads(marker.read_text())
                    payload["terminal_sha256"]["progress_antigravity.json"] = hashlib.sha256(progress.read_bytes()).hexdigest()
                    marker.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                elif corruption == "extra_pid_key":
                    payload = json.loads(pid.read_text()); payload["forged"] = True
                    pid.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                    payload = json.loads(marker.read_text())
                    payload["terminal_sha256"]["pid_antigravity.json"] = hashlib.sha256(pid.read_bytes()).hexdigest()
                    marker.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                elif corruption == "extra_terminal_event_key":
                    rows = [json.loads(line) for line in events.read_text().splitlines()]
                    rows[-1]["forged"] = True
                    events.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))
                    payload = json.loads(marker.read_text())
                    payload["terminal_sha256"]["events_antigravity.jsonl"] = hashlib.sha256(events.read_bytes()).hexdigest()
                    marker.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                elif corruption == "altered_terminal_text":
                    text.write_text(text.read_text().replace("종료 stage=complete", "종료 stage=forged"))
                    payload = json.loads(marker.read_text())
                    payload["terminal_sha256"]["text_antigravity.log"] = hashlib.sha256(text.read_bytes()).hexdigest()
                    marker.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                elif corruption == "state_process_wrong_type":
                    payload = json.loads(state.read_text()); payload["process"] = "forged"
                    state.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                    payload = json.loads(marker.read_text())
                    payload["terminal_sha256"]["checkpoint_antigravity/state_antigravity.json"] = hashlib.sha256(state.read_bytes()).hexdigest()
                    marker.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                elif corruption == "progress_metric_wrong_type":
                    payload = json.loads(progress.read_text()); payload["elapsed_seconds"] = "forged"
                    progress.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                    payload = json.loads(marker.read_text())
                    payload["terminal_sha256"]["progress_antigravity.json"] = hashlib.sha256(progress.read_bytes()).hexdigest()
                    marker.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                elif corruption == "pid_heartbeat_wrong_type":
                    payload = json.loads(pid.read_text()); payload["heartbeat"] = 0
                    pid.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                    payload = json.loads(marker.read_text())
                    payload["terminal_sha256"]["pid_antigravity.json"] = hashlib.sha256(pid.read_bytes()).hexdigest()
                    marker.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                elif corruption == "forged_preceding_event":
                    rows = [json.loads(line) for line in events.read_text().splitlines()]
                    rows[0]["stage"] = "forged"
                    events.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))
                    payload = json.loads(marker.read_text())
                    payload["terminal_sha256"]["events_antigravity.jsonl"] = hashlib.sha256(events.read_bytes()).hexdigest()
                    marker.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                elif corruption == "prefixed_text":
                    text.write_text("forged prefix\n" + text.read_text())
                    payload = json.loads(marker.read_text())
                    payload["terminal_sha256"]["text_antigravity.log"] = hashlib.sha256(text.read_bytes()).hexdigest()
                    marker.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                elif corruption == "state_completed_float":
                    payload = json.loads(state.read_text()); payload["completed_units"] = 30.0
                    state.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                    payload = json.loads(marker.read_text())
                    payload["terminal_sha256"]["checkpoint_antigravity/state_antigravity.json"] = hashlib.sha256(state.read_bytes()).hexdigest()
                    marker.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                elif corruption == "state_total_bool":
                    payload = json.loads(state.read_text()); payload["total_units"] = True
                    state.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                    payload = json.loads(marker.read_text())
                    payload["terminal_sha256"]["checkpoint_antigravity/state_antigravity.json"] = hashlib.sha256(state.read_bytes()).hexdigest()
                    marker.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                elif corruption in {"progress_completed_float", "progress_total_bool"}:
                    payload = json.loads(progress.read_text())
                    payload["completed" if corruption == "progress_completed_float" else "total"] = 30.0 if corruption == "progress_completed_float" else True
                    progress.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                    payload = json.loads(marker.read_text())
                    payload["terminal_sha256"]["progress_antigravity.json"] = hashlib.sha256(progress.read_bytes()).hexdigest()
                    marker.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                elif corruption in {"event_pid_float", "event_completed_bool", "event_total_float", "event_candidate_index_bool", "forged_event_dataset_pair"}:
                    rows = [json.loads(line) for line in events.read_text().splitlines()]
                    if corruption == "event_pid_float": rows[0]["pid"] = float(rows[0]["pid"])
                    elif corruption == "event_completed_bool": rows[1]["completed"] = False
                    elif corruption == "event_total_float": rows[1]["total"] = 30.0
                    elif corruption == "event_candidate_index_bool": rows[1]["candidate_index"] = False
                    else:
                        rows[1]["dataset"] = rows[2]["dataset"] = "Forged_dataset"
                    events.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))
                    if corruption == "forged_event_dataset_pair":
                        lines = text.read_text().splitlines()
                        lines[1] = "후보 시작 dataset=Forged_dataset index=0 train=4 test=4 raw_length=8"
                        lines[2] = "후보 완료 dataset=Forged_dataset status=success 완료=1/30"
                        text.write_text("\n".join(lines) + "\n")
                    payload = json.loads(marker.read_text())
                    payload["terminal_sha256"]["events_antigravity.jsonl"] = hashlib.sha256(events.read_bytes()).hexdigest()
                    if corruption == "forged_event_dataset_pair":
                        payload["terminal_sha256"]["text_antigravity.log"] = hashlib.sha256(text.read_bytes()).hexdigest()
                    marker.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
                elif corruption in {"missing_event_final_lf", "event_crlf"}:
                    payload = events.read_bytes()
                    events.write_bytes(payload[:-1] if corruption == "missing_event_final_lf" else payload.replace(b"\n", b"\r\n"))
                    marker_payload = json.loads(marker.read_text())
                    marker_payload["terminal_sha256"]["events_antigravity.jsonl"] = hashlib.sha256(events.read_bytes()).hexdigest()
                    marker.write_text(json.dumps(marker_payload, sort_keys=True, separators=(",", ":")) + "\n")
                elif corruption in {"missing_text_final_lf", "text_crlf"}:
                    payload = text.read_bytes()
                    text.write_bytes(payload[:-1] if corruption == "missing_text_final_lf" else payload.replace(b"\n", b"\r\n"))
                    marker_payload = json.loads(marker.read_text())
                    marker_payload["terminal_sha256"]["text_antigravity.log"] = hashlib.sha256(text.read_bytes()).hexdigest()
                    marker.write_text(json.dumps(marker_payload, sort_keys=True, separators=(",", ":")) + "\n")
                else:
                    record.write_bytes(record.read_bytes() + b" ")
                with self.assertRaisesRegex(ValueError, "terminal parity"):
                    self._require_terminal_parity_antigravity(root)

    def test_terminal_authority_rejects_same_size_cross_read_replacement_antigravity(self) -> None:
        """The final namespace pass catches a replacement after its only read."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import checkpoint_antigravity

        root, outcome = self._run_antigravity()
        self.assertEqual(outcome.status, "complete")
        state = root / "attempts_antigravity/attempt_0000_antigravity/checkpoint_antigravity/state_antigravity.json"
        original = checkpoint_antigravity._read_pinned_regular_leaf_antigravity
        fired = False

        def replace_after_read(directory_fd, name):
            nonlocal fired
            payload, identity = original(directory_fd, name)
            if not fired and name == "state_antigravity.json":
                fired = True
                replacement = state.with_name("state_replacement_antigravity.json")
                replacement.write_bytes(payload)
                os.replace(replacement, state)
            return payload, identity

        with mock.patch.object(checkpoint_antigravity, "_read_pinned_regular_leaf_antigravity", new=replace_after_read):
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
                progress = json.loads((attempt / "progress_antigravity.json").read_text())
                pid = json.loads((attempt / "pid_antigravity.json").read_text())
                if channel == "marker":
                    self.assertEqual((progress["completed"], progress["total"], progress["stage"], progress["active"], pid["active"]), (30, 30, "complete", False, False))
                elif channel == "progress":
                    self.assertEqual((progress["completed"], progress["total"], progress["stage"], progress["active"], pid["active"]), (30, 30, "running", True, False))
                elif channel == "pid":
                    self.assertEqual((progress["completed"], progress["total"], progress["stage"], progress["active"], pid["active"]), (30, 30, "complete", False, True))
                else:
                    self.assertEqual((progress["completed"], progress["total"], progress["stage"], progress["active"], pid["active"]), (30, 30, "complete", False, False))

    def test_marker_finalization_failure_is_immutable_complete_authority_antigravity(self) -> None:
        """A marker write after durable complete must leave explicit non-eligibility evidence."""
        root, outcome = self._run_antigravity(write_fault="marker")
        self.assertIsInstance(outcome, OSError)
        attempt = root / "attempts_antigravity/attempt_0000_antigravity"
        failure = attempt / "terminal_finalization_failure_antigravity.json"
        self.assertTrue(failure.is_file())
        payload = json.loads(failure.read_text())
        self.assertEqual(payload["schema"], "exp446.terminal-finalization-failure.v1")
        self.assertEqual(payload["channel"], "marker")
        self.assertEqual(payload["error"], {"type": "OSError", "message": "marker write fault"})
        self.assertFalse((attempt / "terminal_parity_antigravity.json").exists())
        progress = json.loads((attempt / "progress_antigravity.json").read_text())
        pid = json.loads((attempt / "pid_antigravity.json").read_text())
        self.assertEqual((progress["stage"], progress["active"], pid["active"]), ("complete", False, False))
        with self.assertRaisesRegex(ValueError, "terminal parity"):
            self._require_terminal_parity_antigravity(root)
        retry_root, retry = self._run_antigravity(reuse_root=root)
        self.assertIsInstance(retry, ValueError)
        self.assertEqual(retry_root, root)

    def test_persistent_terminal_channel_failure_records_exact_recovery_authority_antigravity(self) -> None:
        """Terminal faults preserve the exact root cause and PID recovery truth without a retry."""
        for channel in ("progress", "pid", "event", "text"):
            with self.subTest(channel=channel):
                root, outcome = self._run_antigravity(write_fault=f"persistent_{channel}")
                self.assertIsInstance(outcome, OSError)
                attempt = root / "attempts_antigravity/attempt_0000_antigravity"
                failure = json.loads((attempt / "terminal_finalization_failure_antigravity.json").read_text())
                self.assertEqual((failure["schema"], failure["channel"]), ("exp446.terminal-finalization-failure.v1", channel))
                self.assertEqual(failure["error"], {"type": "OSError", "message": f"{channel} write fault"})
                state = json.loads((attempt / "checkpoint_antigravity/state_antigravity.json").read_text())
                pid = json.loads((attempt / "pid_antigravity.json").read_text())
                self.assertEqual((state["status"], state["completed_units"], state["error_count"]), ("complete", 30, 0))
                self.assertEqual(failure["pid_recovery"]["status"], "unresolved" if channel == "pid" else "inactive")
                self.assertEqual(pid["active"], channel == "pid")
                self.assertFalse((attempt / "terminal_parity_antigravity.json").exists())
                with self.assertRaisesRegex(ValueError, "terminal parity"):
                    self._require_terminal_parity_antigravity(root)
                retry_root, retry = self._run_antigravity(reuse_root=root)
                self.assertIsInstance(retry, ValueError)
                self.assertEqual(retry_root, root)

    def test_terminal_finalization_failure_never_mutates_complete_checkpoint_tree_antigravity(self) -> None:
        """A closeout or marker error must not alter any already-complete checkpoint bytes."""
        for fault in ("persistent_progress", "persistent_pid", "persistent_event", "persistent_text", "marker"):
            with self.subTest(fault=fault):
                captures: list[dict[str, bytes]] = []
                root, outcome = self._run_antigravity(write_fault=fault, checkpoint_capture=captures)
                self.assertIsInstance(outcome, OSError)
                checkpoint_root = root / "attempts_antigravity/attempt_0000_antigravity/checkpoint_antigravity"
                self.assertEqual(len(captures), 1)
                self.assertEqual(captures[0], _checkpoint_tree_bytes_antigravity(checkpoint_root))
                self.assertEqual(len(captures[0]), 32)

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
