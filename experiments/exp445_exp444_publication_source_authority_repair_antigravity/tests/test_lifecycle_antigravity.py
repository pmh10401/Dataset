"""Parity-gated Exp445 new-only lifecycle behavior."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments.exp444_train_only_stability_precision_gate_antigravity.contracts_antigravity import (
    CandidateInputAntigravity,
    CandidateStabilityResultAntigravity,
    DepthViewScoreAntigravity,
    OmissionMaskAntigravity,
    SourceCandidateAntigravity,
    StabilityViewAntigravity,
)
from experiments.exp444_train_only_stability_precision_gate_antigravity.checkpoint_antigravity import ProcessIdentityAntigravity


_COMMIT_ANTIGRAVITY = "a" * 40


def _inputs_antigravity() -> tuple[CandidateInputAntigravity, ...]:
    values = []
    for sequence in range(30):
        candidate = SourceCandidateAntigravity(f"Dataset_{sequence:02d}", sequence, 8, (1, 2), sequence + 1)
        values.append(CandidateInputAntigravity(candidate, (0, 1, 2, 3), np.arange(32, dtype="<f4").reshape(4, 8), np.arange(8, dtype="<f4")))
    return tuple(values)


def _result_antigravity(item: CandidateInputAntigravity) -> CandidateStabilityResultAntigravity:
    score = (DepthViewScoreAntigravity(1, 2.0, 1.0, True, 2.0, 0.5), DepthViewScoreAntigravity(2, 2.0, 1.0, True, 2.0, 0.5))
    view = StabilityViewAntigravity(OmissionMaskAntigravity(0, (0,), hashlib.sha256(b"mask").hexdigest()), (1, 2, 3), score, (1, 2), True)
    return CandidateStabilityResultAntigravity(item.candidate, (view,), 1, 1, "stable_review")


def _record_sha_antigravity(sequence: int, result: CandidateStabilityResultAntigravity) -> str:
    raw = json.dumps({"sequence": sequence, "result": result.to_payload()}, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
    return hashlib.sha256(raw).hexdigest()


class Exp445LifecycleAntigravityTests(unittest.TestCase):
    def _test_profile_antigravity(self, profile: object, root: Path):
        """Create only a test-local object; production profiles remain closed."""
        patched = object.__new__(type(profile))
        for name, value in (
            ("experiment_id", profile.experiment_id), ("schema", profile.schema),
            ("run_identity", profile.run_identity), ("run_root", root),
            ("reviewed_design_path", profile.reviewed_design_path),
            ("preregistration_path", profile.preregistration_path),
            ("predecessor_required", profile.predecessor_required),
        ):
            object.__setattr__(patched, name, value)
        return patched

    def _authority_antigravity(self, inputs: tuple[CandidateInputAntigravity, ...]):
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity.predecessor_antigravity import Exp444PredecessorAuthorityAntigravity
        prefix = tuple(_record_sha_antigravity(index, _result_antigravity(item)) for index, item in enumerate(inputs))
        aggregate = hashlib.sha256(json.dumps(list(prefix), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n").hexdigest()
        return Exp444PredecessorAuthorityAntigravity(_COMMIT_ANTIGRAVITY, Path("/predecessor/attempt_antigravity"), prefix, aggregate)

    def test_all_matching_records_complete_once_before_any_publication_antigravity(self) -> None:
        """Skipping a pre-write comparison could silently publish a changed candidate wrapper."""
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import lifecycle_antigravity
        from experiments.exp444_train_only_stability_precision_gate_antigravity import runtime_antigravity as runtime

        inputs = _inputs_antigravity()
        authority = self._authority_antigravity(inputs)
        scheduler_arguments: dict[str, object] = {}
        with tempfile.TemporaryDirectory(prefix="exp445_lifecycle_antigravity_") as directory:
            root = Path(directory) / "exp445_exp444_publication_source_authority_repair_antigravity_20260810"
            profile = lifecycle_antigravity._EXP445_PROFILE_ANTIGRAVITY
            patched_profile = self._test_profile_antigravity(profile, root)
            def complete(active, *, commit, progress, **kwargs):
                scheduler_arguments.update(kwargs)
                for index, item in enumerate(active):
                    progress(stage="active", sequence=index)
                    commit(index, item, _result_antigravity(item))
                return tuple(_result_antigravity(item) for item in active)
            with (
                mock.patch.object(lifecycle_antigravity, "_EXP445_PROFILE_ANTIGRAVITY", patched_profile),
                mock.patch.object(runtime, "EXP445_PROFILE_ANTIGRAVITY", patched_profile), mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity.EXP445_PROFILE_ANTIGRAVITY", patched_profile),
                mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity.EXP445_PROFILE_ANTIGRAVITY", patched_profile),
                mock.patch.object(runtime, "readiness_for_profile_antigravity"), mock.patch.object(runtime, "current_clean_head_antigravity", return_value=_COMMIT_ANTIGRAVITY),
                mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(99101, "exp445-test")),
                mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)),
                mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=complete),
                mock.patch.object(runtime, "_verify_exp445_predecessor_antigravity", return_value=authority),
                mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.provenance_antigravity.verify_exp444_provenance_antigravity"),
            ):
                outcome = lifecycle_antigravity.run_new_antigravity(_COMMIT_ANTIGRAVITY)
            self.assertEqual((outcome.status, outcome.completed_count, outcome.workers), ("complete", 30, 11))
            self.assertEqual((scheduler_arguments["workers"], scheduler_arguments["distance_backend"]), (11, "cpu"))
            attempt = root / "attempts_antigravity/attempt_0000_antigravity"
            context = json.loads((attempt / "checkpoint_antigravity/context_antigravity.json").read_text())
            self.assertEqual((context["experiment_id"], context["schema"], context["run_identity"]), (
                "Exp445", "exp445", "exp445-exp444-publication-source-authority-repair-antigravity-20260810",
            ))
            self.assertEqual((context["outer_workers"], context["inner_threads"]), (11, 1))
            self.assertFalse((attempt / "publication_antigravity").exists())
            self.assertEqual(len(list((attempt / "checkpoint_antigravity/records_antigravity").glob("*_record_antigravity.json"))), 30)

    def test_live_profile_readiness_runs_before_new_root_mutation_antigravity(self) -> None:
        """New execution must not bypass its live label-free readiness gate."""
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import lifecycle_antigravity
        from experiments.exp444_train_only_stability_precision_gate_antigravity import runtime_antigravity as runtime

        inputs = _inputs_antigravity(); authority = self._authority_antigravity(inputs)
        with tempfile.TemporaryDirectory(prefix="exp445_live_readiness_antigravity_") as directory:
            root = Path(directory) / "exp445_exp444_publication_source_authority_repair_antigravity_20260810"
            profile = self._test_profile_antigravity(lifecycle_antigravity._EXP445_PROFILE_ANTIGRAVITY, root)
            readiness_calls: list[tuple[object, str]] = []
            def readiness(active_profile, commit):
                readiness_calls.append((active_profile, commit))
            def root_mutation(path):
                self.assertEqual(readiness_calls, [(profile, _COMMIT_ANTIGRAVITY)])
                raise RuntimeError("stop after readiness")
            with (
                mock.patch.object(lifecycle_antigravity, "_EXP445_PROFILE_ANTIGRAVITY", profile),
                mock.patch.object(runtime, "EXP445_PROFILE_ANTIGRAVITY", profile), mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity.EXP445_PROFILE_ANTIGRAVITY", profile),
                mock.patch.object(runtime, "readiness_for_profile_antigravity", side_effect=readiness),
                mock.patch.object(runtime, "current_clean_head_antigravity", return_value=_COMMIT_ANTIGRAVITY),
                mock.patch.object(runtime, "_verify_exp445_predecessor_antigravity", return_value=authority),
                mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)),
                mock.patch.object(runtime, "_attempt_root_antigravity", side_effect=root_mutation),
                mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.provenance_antigravity.verify_exp444_provenance_antigravity"),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop after readiness"):
                    lifecycle_antigravity.run_new_antigravity(_COMMIT_ANTIGRAVITY)

    def test_checkpoint_setup_failure_has_terminal_exp445_evidence_antigravity(self) -> None:
        """A post-root checkpoint failure cannot leave an apparently live Exp445 root."""
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import lifecycle_antigravity
        from experiments.exp444_train_only_stability_precision_gate_antigravity import runtime_antigravity as runtime

        inputs = _inputs_antigravity(); authority = self._authority_antigravity(inputs)
        with tempfile.TemporaryDirectory(prefix="exp445_setup_failure_antigravity_") as directory:
            root = Path(directory) / "exp445_exp444_publication_source_authority_repair_antigravity_20260810"
            profile = self._test_profile_antigravity(lifecycle_antigravity._EXP445_PROFILE_ANTIGRAVITY, root)
            with (
                mock.patch.object(lifecycle_antigravity, "_EXP445_PROFILE_ANTIGRAVITY", profile), mock.patch.object(runtime, "EXP445_PROFILE_ANTIGRAVITY", profile), mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity.EXP445_PROFILE_ANTIGRAVITY", profile),
                mock.patch.object(runtime, "readiness_for_profile_antigravity"), mock.patch.object(runtime, "readiness_for_profile_antigravity"), mock.patch.object(runtime, "current_clean_head_antigravity", return_value=_COMMIT_ANTIGRAVITY),
                mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(99104, "exp445-test")),
                mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)), mock.patch.object(runtime, "_verify_exp445_predecessor_antigravity", return_value=authority),
                mock.patch.object(runtime.Exp444CheckpointStoreAntigravity, "create", side_effect=OSError("checkpoint setup fault")),
                mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.provenance_antigravity.verify_exp444_provenance_antigravity"),
            ):
                with self.assertRaisesRegex(OSError, "checkpoint setup fault"):
                    lifecycle_antigravity.run_new_antigravity(_COMMIT_ANTIGRAVITY)
            attempt = root / "attempts_antigravity/attempt_0000_antigravity"
            failure = json.loads((attempt / "setup_failure_antigravity.json").read_text())
            self.assertEqual((failure["stage"], failure["error"]["message"]), ("setup", "checkpoint setup fault"))
            progress = json.loads((attempt / "progress_antigravity.json").read_text())
            self.assertEqual((progress["active"], progress["stage"]), (False, "error"))

    def test_progress_setup_failure_reconciles_checkpoint_to_terminal_error_antigravity(self) -> None:
        """Progress initialization failure must not leave its already-created checkpoint running."""
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import lifecycle_antigravity
        from experiments.exp444_train_only_stability_precision_gate_antigravity import runtime_antigravity as runtime
        from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import ProgressWriterAntigravity

        inputs = _inputs_antigravity(); authority = self._authority_antigravity(inputs)
        with tempfile.TemporaryDirectory(prefix="exp445_progress_setup_failure_antigravity_") as directory:
            root = Path(directory) / "exp445_exp444_publication_source_authority_repair_antigravity_20260810"
            profile = self._test_profile_antigravity(lifecycle_antigravity._EXP445_PROFILE_ANTIGRAVITY, root)
            with (
                mock.patch.object(lifecycle_antigravity, "_EXP445_PROFILE_ANTIGRAVITY", profile), mock.patch.object(runtime, "EXP445_PROFILE_ANTIGRAVITY", profile), mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity.EXP445_PROFILE_ANTIGRAVITY", profile), mock.patch.object(runtime, "readiness_for_profile_antigravity"), mock.patch.object(runtime, "current_clean_head_antigravity", return_value=_COMMIT_ANTIGRAVITY), mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(99107, "exp445-test")), mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)), mock.patch.object(runtime, "_verify_exp445_predecessor_antigravity", return_value=authority), mock.patch.object(ProgressWriterAntigravity, "create", side_effect=OSError("progress setup fault")), mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.provenance_antigravity.verify_exp444_provenance_antigravity"),
            ):
                with self.assertRaisesRegex(OSError, "progress setup fault"):
                    lifecycle_antigravity.run_new_antigravity(_COMMIT_ANTIGRAVITY)
            attempt = root / "attempts_antigravity/attempt_0000_antigravity"; checkpoint = attempt / "checkpoint_antigravity"
            self.assertEqual(json.loads((checkpoint / "state_antigravity.json").read_text())["status"], "error")
            progress = json.loads((attempt / "progress_antigravity.json").read_text())
            self.assertEqual((progress["active"], progress["stage"], progress["completed"], progress["error"]), (False, "error", 1, 1))

    def test_nonfinal_progress_fault_reconciles_success_then_writes_exact_next_error_antigravity(self) -> None:
        """A durable success cannot be followed by a stale progress prefix."""
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import lifecycle_antigravity
        from experiments.exp444_train_only_stability_precision_gate_antigravity import runtime_antigravity as runtime
        from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import ProgressWriterAntigravity

        inputs = _inputs_antigravity(); authority = self._authority_antigravity(inputs); original = ProgressWriterAntigravity.complete_candidate_antigravity
        with tempfile.TemporaryDirectory(prefix="exp445_nonfinal_progress_fault_antigravity_") as directory:
            root = Path(directory) / "exp445_exp444_publication_source_authority_repair_antigravity_20260810"; profile = self._test_profile_antigravity(lifecycle_antigravity._EXP445_PROFILE_ANTIGRAVITY, root)
            def ordered(active, *, commit, progress, **_kwargs):
                for index, item in enumerate(active): progress(stage="active", sequence=index); commit(index, item, _result_antigravity(item))
            def fail_sequence_seven(writer, *args, **kwargs):
                if writer._current_candidate_index == 7:
                    raise OSError("progress sequence seven fault")
                return original(writer, *args, **kwargs)
            with (
                mock.patch.object(lifecycle_antigravity, "_EXP445_PROFILE_ANTIGRAVITY", profile), mock.patch.object(runtime, "EXP445_PROFILE_ANTIGRAVITY", profile), mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity.EXP445_PROFILE_ANTIGRAVITY", profile), mock.patch.object(runtime, "readiness_for_profile_antigravity"), mock.patch.object(runtime, "readiness_for_profile_antigravity"), mock.patch.object(runtime, "current_clean_head_antigravity", return_value=_COMMIT_ANTIGRAVITY), mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(99105, "exp445-test")), mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)), mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=ordered), mock.patch.object(runtime, "_verify_exp445_predecessor_antigravity", return_value=authority), mock.patch.object(ProgressWriterAntigravity, "complete_candidate_antigravity", new=fail_sequence_seven), mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.provenance_antigravity.verify_exp444_provenance_antigravity"),
            ):
                with self.assertRaisesRegex(OSError, "sequence seven fault"):
                    lifecycle_antigravity.run_new_antigravity(_COMMIT_ANTIGRAVITY)
            attempt = root / "attempts_antigravity/attempt_0000_antigravity"; records = attempt / "checkpoint_antigravity/records_antigravity"
            self.assertEqual([item.name for item in sorted(records.iterdir())], [f"{index:04d}_record_antigravity.json" for index in range(9)])
            self.assertEqual(json.loads((records / "0008_record_antigravity.json").read_text())["status"], "error")
            progress = json.loads((attempt / "progress_antigravity.json").read_text())
            self.assertEqual((progress["stage"], progress["completed"], progress["error"]), ("error", 9, 1))

    def test_final_progress_fault_reconciles_then_completes_antigravity(self) -> None:
        """The last durable success is reconciled before either terminal authority completes."""
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import lifecycle_antigravity
        from experiments.exp444_train_only_stability_precision_gate_antigravity import runtime_antigravity as runtime
        from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import ProgressWriterAntigravity

        inputs = _inputs_antigravity(); authority = self._authority_antigravity(inputs); original = ProgressWriterAntigravity.complete_candidate_antigravity; faulted = [False]
        with tempfile.TemporaryDirectory(prefix="exp445_final_progress_fault_antigravity_") as directory:
            root = Path(directory) / "exp445_exp444_publication_source_authority_repair_antigravity_20260810"; profile = self._test_profile_antigravity(lifecycle_antigravity._EXP445_PROFILE_ANTIGRAVITY, root)
            def ordered(active, *, commit, progress, **_kwargs):
                for index, item in enumerate(active): progress(stage="active", sequence=index); commit(index, item, _result_antigravity(item));
                return tuple(_result_antigravity(item) for item in active)
            def fail_final_once(writer, *args, **kwargs):
                if writer._current_candidate_index == 29 and not faulted[0]: faulted[0] = True; raise OSError("progress final fault")
                return original(writer, *args, **kwargs)
            with (
                mock.patch.object(lifecycle_antigravity, "_EXP445_PROFILE_ANTIGRAVITY", profile), mock.patch.object(runtime, "EXP445_PROFILE_ANTIGRAVITY", profile), mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity.EXP445_PROFILE_ANTIGRAVITY", profile), mock.patch.object(runtime, "readiness_for_profile_antigravity"), mock.patch.object(runtime, "readiness_for_profile_antigravity"), mock.patch.object(runtime, "current_clean_head_antigravity", return_value=_COMMIT_ANTIGRAVITY), mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(99106, "exp445-test")), mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)), mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=ordered), mock.patch.object(runtime, "_verify_exp445_predecessor_antigravity", return_value=authority), mock.patch.object(ProgressWriterAntigravity, "complete_candidate_antigravity", new=fail_final_once), mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.provenance_antigravity.verify_exp444_provenance_antigravity"),
            ):
                outcome = lifecycle_antigravity.run_new_antigravity(_COMMIT_ANTIGRAVITY)
            self.assertEqual((outcome.status, outcome.completed_count), ("complete", 30))
            progress = json.loads((root / "attempts_antigravity/attempt_0000_antigravity/progress_antigravity.json").read_text())
            self.assertEqual((progress["stage"], progress["completed"], progress["error"]), ("complete", 30, 0))

    def test_final_event_or_text_fault_after_progress_json_keeps_complete_authority_antigravity(self) -> None:
        """A post-JSON append fault must not downgrade an already accepted final prefix."""
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import lifecycle_antigravity
        from experiments.exp444_train_only_stability_precision_gate_antigravity import runtime_antigravity as runtime
        from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import ProgressWriterAntigravity

        for kind in ("event", "text"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(prefix=f"exp445_{kind}_after_json_antigravity_") as directory:
                inputs = _inputs_antigravity(); authority = self._authority_antigravity(inputs)
                root = Path(directory) / "exp445_exp444_publication_source_authority_repair_antigravity_20260810"; profile = self._test_profile_antigravity(lifecycle_antigravity._EXP445_PROFILE_ANTIGRAVITY, root)
                def ordered(active, *, commit, progress, **_kwargs):
                    for index, item in enumerate(active): progress(stage="active", sequence=index); commit(index, item, _result_antigravity(item))
                    return tuple(_result_antigravity(item) for item in active)
                original_event = ProgressWriterAntigravity._append_event_antigravity; original_text = ProgressWriterAntigravity._append_text_antigravity
                def event_fault(writer, event, stage, extra=None):
                    if event == "candidate_complete" and writer._counters["completed"] == 30: raise OSError("final event append fault")
                    return original_event(writer, event, stage, extra)
                def text_fault(writer, line):
                    if line.startswith("후보 완료") and "완료=30/30" in line: raise OSError("final text append fault")
                    return original_text(writer, line)
                with (
                    mock.patch.object(lifecycle_antigravity, "_EXP445_PROFILE_ANTIGRAVITY", profile), mock.patch.object(runtime, "EXP445_PROFILE_ANTIGRAVITY", profile), mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity.EXP445_PROFILE_ANTIGRAVITY", profile), mock.patch.object(runtime, "readiness_for_profile_antigravity"), mock.patch.object(runtime, "current_clean_head_antigravity", return_value=_COMMIT_ANTIGRAVITY), mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(99200, f"exp445-{kind}")), mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)), mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=ordered), mock.patch.object(runtime, "_verify_exp445_predecessor_antigravity", return_value=authority), mock.patch.object(ProgressWriterAntigravity, "_append_event_antigravity", new=event_fault if kind == "event" else original_event), mock.patch.object(ProgressWriterAntigravity, "_append_text_antigravity", new=text_fault if kind == "text" else original_text), mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.provenance_antigravity.verify_exp444_provenance_antigravity"),
                ):
                    outcome = lifecycle_antigravity.run_new_antigravity(_COMMIT_ANTIGRAVITY)
                self.assertEqual((outcome.status, outcome.completed_count), ("complete", 30))
                progress = json.loads((root / "attempts_antigravity/attempt_0000_antigravity/progress_antigravity.json").read_text())
                state = json.loads((root / "attempts_antigravity/attempt_0000_antigravity/checkpoint_antigravity/state_antigravity.json").read_text())
                self.assertEqual((progress["stage"], progress["completed"], state["status"], state["completed_units"]), ("complete", 30, "complete", 30))

    def test_nonfinal_wrapper_mismatch_writes_only_exact_next_error_record_antigravity(self) -> None:
        """Writing a successful mismatching result before rejecting it would invalidate failure evidence."""
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import lifecycle_antigravity
        from experiments.exp444_train_only_stability_precision_gate_antigravity import runtime_antigravity as runtime

        inputs = _inputs_antigravity(); authority = self._authority_antigravity(inputs)
        wrong = list(authority.record_sha256_prefix); wrong[7] = "0" * 64
        authority = type(authority)(authority.reviewed_commit, authority.attempt_root, tuple(wrong), authority.record_prefix_payload_sha256)
        with tempfile.TemporaryDirectory(prefix="exp445_nonfinal_antigravity_") as directory:
            root = Path(directory) / "exp445_exp444_publication_source_authority_repair_antigravity_20260810"; profile = lifecycle_antigravity._EXP445_PROFILE_ANTIGRAVITY
            patched_profile = self._test_profile_antigravity(profile, root)
            def complete(active, *, commit, progress, **_kwargs):
                for index, item in enumerate(active): progress(stage="active", sequence=index); commit(index, item, _result_antigravity(item))
            with (
                mock.patch.object(lifecycle_antigravity, "_EXP445_PROFILE_ANTIGRAVITY", patched_profile), mock.patch.object(runtime, "EXP445_PROFILE_ANTIGRAVITY", patched_profile), mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity.EXP445_PROFILE_ANTIGRAVITY", patched_profile), mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity.EXP445_PROFILE_ANTIGRAVITY", patched_profile), mock.patch.object(runtime, "readiness_for_profile_antigravity"), mock.patch.object(runtime, "current_clean_head_antigravity", return_value=_COMMIT_ANTIGRAVITY), mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(99102, "exp445-test")), mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)), mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=complete), mock.patch.object(runtime, "_verify_exp445_predecessor_antigravity", return_value=authority), mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.provenance_antigravity.verify_exp444_provenance_antigravity"),
            ):
                with self.assertRaisesRegex(ValueError, "predecessor record parity"):
                    lifecycle_antigravity.run_new_antigravity(_COMMIT_ANTIGRAVITY)
            records = root / "attempts_antigravity/attempt_0000_antigravity/checkpoint_antigravity/records_antigravity"
            self.assertEqual([item.name for item in sorted(records.iterdir())], [f"{index:04d}_record_antigravity.json" for index in range(8)])
            error = json.loads((records / "0007_record_antigravity.json").read_text())
            self.assertEqual((error["sequence"], error["status"], error["result"]), (7, "error", None))
            self.assertFalse((root / "attempts_antigravity/attempt_0000_antigravity/publication_antigravity").exists())

    def test_final_aggregate_mismatch_never_commits_the_thirtieth_success_antigravity(self) -> None:
        """Completing before aggregate revalidation would leave an irreversible false complete state."""
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import lifecycle_antigravity
        from experiments.exp444_train_only_stability_precision_gate_antigravity import runtime_antigravity as runtime

        inputs = _inputs_antigravity(); authority = self._authority_antigravity(inputs)
        authority = type(authority)(authority.reviewed_commit, authority.attempt_root, authority.record_sha256_prefix, "0" * 64)
        with tempfile.TemporaryDirectory(prefix="exp445_final_antigravity_") as directory:
            root = Path(directory) / "exp445_exp444_publication_source_authority_repair_antigravity_20260810"; profile = lifecycle_antigravity._EXP445_PROFILE_ANTIGRAVITY
            patched_profile = self._test_profile_antigravity(profile, root)
            def complete(active, *, commit, progress, **_kwargs):
                for index, item in enumerate(active): progress(stage="active", sequence=index); commit(index, item, _result_antigravity(item))
            with (
                mock.patch.object(lifecycle_antigravity, "_EXP445_PROFILE_ANTIGRAVITY", patched_profile), mock.patch.object(runtime, "EXP445_PROFILE_ANTIGRAVITY", patched_profile), mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity.EXP445_PROFILE_ANTIGRAVITY", patched_profile), mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity.EXP445_PROFILE_ANTIGRAVITY", patched_profile), mock.patch.object(runtime, "readiness_for_profile_antigravity"), mock.patch.object(runtime, "current_clean_head_antigravity", return_value=_COMMIT_ANTIGRAVITY), mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(99103, "exp445-test")), mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)), mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=complete), mock.patch.object(runtime, "_verify_exp445_predecessor_antigravity", return_value=authority), mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.provenance_antigravity.verify_exp444_provenance_antigravity"),
            ):
                with self.assertRaisesRegex(ValueError, "aggregate parity"):
                    lifecycle_antigravity.run_new_antigravity(_COMMIT_ANTIGRAVITY)
            attempt = root / "attempts_antigravity/attempt_0000_antigravity"; records = attempt / "checkpoint_antigravity/records_antigravity"
            self.assertEqual([item.name for item in sorted(records.iterdir())], [f"{index:04d}_record_antigravity.json" for index in range(30)])
            final = json.loads((records / "0029_record_antigravity.json").read_text())
            self.assertEqual((final["sequence"], final["status"], final["result"]), (29, "error", None))
            self.assertEqual(json.loads((attempt / "checkpoint_antigravity/state_antigravity.json").read_text())["status"], "error")
            progress = json.loads((attempt / "progress_antigravity.json").read_text())
            self.assertEqual((progress["stage"], progress["completed"], progress["error"]), ("error", 30, 1))

    def test_fresh_predecessor_tuple_mutation_blocks_record_twenty_nine_antigravity(self) -> None:
        """A valid initial predecessor cannot authorize a changed fresh final tuple."""
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import lifecycle_antigravity
        from experiments.exp444_train_only_stability_precision_gate_antigravity import runtime_antigravity as runtime

        inputs = _inputs_antigravity(); initial = self._authority_antigravity(inputs); changed = list(initial.record_sha256_prefix); changed[3] = "0" * 64
        fresh = type(initial)(initial.reviewed_commit, initial.attempt_root, tuple(changed), initial.record_prefix_payload_sha256)
        with tempfile.TemporaryDirectory(prefix="exp445_fresh_tuple_antigravity_") as directory:
            root = Path(directory) / "exp445_exp444_publication_source_authority_repair_antigravity_20260810"; profile = self._test_profile_antigravity(lifecycle_antigravity._EXP445_PROFILE_ANTIGRAVITY, root)
            def ordered(active, *, commit, progress, **_kwargs):
                for index, item in enumerate(active): progress(stage="active", sequence=index); commit(index, item, _result_antigravity(item))
            with (
                mock.patch.object(lifecycle_antigravity, "_EXP445_PROFILE_ANTIGRAVITY", profile), mock.patch.object(runtime, "EXP445_PROFILE_ANTIGRAVITY", profile), mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity.EXP445_PROFILE_ANTIGRAVITY", profile), mock.patch.object(runtime, "readiness_for_profile_antigravity"), mock.patch.object(runtime, "current_clean_head_antigravity", return_value=_COMMIT_ANTIGRAVITY), mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(99201, "exp445-test")), mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)), mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=ordered), mock.patch.object(runtime, "_verify_exp445_predecessor_antigravity", side_effect=(initial, fresh)), mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.provenance_antigravity.verify_exp444_provenance_antigravity"),
            ):
                with self.assertRaisesRegex(ValueError, "authority changed"):
                    lifecycle_antigravity.run_new_antigravity(_COMMIT_ANTIGRAVITY)
            records = root / "attempts_antigravity/attempt_0000_antigravity/checkpoint_antigravity/records_antigravity"
            final = json.loads((records / "0029_record_antigravity.json").read_text())
            self.assertEqual((final["status"], final["result"]), ("error", None))
            self.assertFalse((root / "attempts_antigravity/attempt_0000_antigravity/publication_antigravity").exists())


if __name__ == "__main__":
    unittest.main()
