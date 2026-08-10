"""TDD tests for the Exp444 self-written progress/PID evidence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments.exp444_train_only_stability_precision_gate_antigravity.artifacts_antigravity import (
    canonical_json_bytes_antigravity,
)
from experiments.exp444_train_only_stability_precision_gate_antigravity.config_antigravity import (
    FROZEN_CONFIG,
    scientific_config_sha256_antigravity,
)
from experiments.exp444_train_only_stability_precision_gate_antigravity.contracts_antigravity import (
    CandidateStabilityResultAntigravity,
    CheckpointContextAntigravity,
    DepthViewScoreAntigravity,
    OmissionMaskAntigravity,
    SourceCandidateAntigravity,
    StabilityViewAntigravity,
)


def _write_checkpoint_state_antigravity(
    attempt: Path,
    *,
    pid: int,
    start_identity: str,
    completed: int,
    total: int,
) -> Path:
    checkpoint = attempt / "checkpoint_antigravity"
    checkpoint.mkdir(exist_ok=True)
    prefix = [hashlib.sha256(f"record-{index}".encode()).hexdigest() for index in range(completed)]
    payload = {
        "context_sha256": "1" * 64,
        "status": "interrupted",
        "completed_units": completed,
        "total_units": total,
        "error_count": 0,
        "current_unit": None,
        "current_view": None,
        "payload_sha256": hashlib.sha256(canonical_json_bytes_antigravity(prefix)).hexdigest(),
        "record_sha256_prefix": prefix,
        "process": {"pid": pid, "start_identity": start_identity},
        "error": None,
        "updated_at": "2026-08-10T00:00:00.000Z",
    }
    state = checkpoint / "state_antigravity.json"
    state.write_bytes(canonical_json_bytes_antigravity(payload))
    return state


def _committed_success_fixture_antigravity(attempt: Path):
    """Build one real stable prefix plus its already-durable progress state."""
    from experiments.exp444_train_only_stability_precision_gate_antigravity.checkpoint_antigravity import (
        Exp444CheckpointStoreAntigravity,
        ProcessIdentityAntigravity,
    )
    from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import (
        ProgressWriterAntigravity,
    )

    process = ProcessIdentityAntigravity(44_601, "progress-recovery")
    context = CheckpointContextAntigravity(
        FROZEN_CONFIG.experiment_id, FROZEN_CONFIG.run_identity,
        scientific_config_sha256_antigravity(), FROZEN_CONFIG.environment_lock_sha256,
        30, FROZEN_CONFIG.cpu_workers, FROZEN_CONFIG.inner_threads, "a" * 40,
    )
    checkpoint = attempt / "checkpoint_antigravity"
    store = Exp444CheckpointStoreAntigravity.create(checkpoint, context, process)
    candidate = SourceCandidateAntigravity("Recovery", 0, 16, (1, 2), 1)
    scores = (
        DepthViewScoreAntigravity(1, 2.0, 1.0, True, 2.0, 0.5),
        DepthViewScoreAntigravity(2, 2.0, 1.0, True, 2.0, 0.5),
    )
    view = StabilityViewAntigravity(
        OmissionMaskAntigravity(0, (0,), hashlib.sha256(b"recovery").hexdigest()),
        (1, 2, 3, 4), scores, (1, 2), True,
    )
    result = CandidateStabilityResultAntigravity(candidate, (view,), 1, 1, "stable_review")
    writer = ProgressWriterAntigravity.create(
        attempt, total=30, process=process,
        checkpoint_path=checkpoint / "state_antigravity.json",
    )
    writer.start_candidate_antigravity("Recovery", 0, train_count=4, test_count=1, raw_length=16)
    store.commit_antigravity(0, result)
    writer.complete_candidate_antigravity()
    return writer, store.committed_prefix_authority_antigravity(0), {
        "train_count": 4, "test_count": 1, "test_rows": 1,
        "raw_length": 16, "target_length": 16,
    }


def _committed_candidate_fixture_antigravity(attempt: Path, sequence: int):
    """Build a live exact-next candidate state for sequence zero or one."""
    from experiments.exp444_train_only_stability_precision_gate_antigravity.checkpoint_antigravity import (
        Exp444CheckpointStoreAntigravity,
        ProcessIdentityAntigravity,
    )
    from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import (
        ProgressWriterAntigravity,
    )

    process = ProcessIdentityAntigravity(44_602, f"candidate-recovery-{sequence}")
    context = CheckpointContextAntigravity(
        FROZEN_CONFIG.experiment_id, FROZEN_CONFIG.run_identity,
        scientific_config_sha256_antigravity(), FROZEN_CONFIG.environment_lock_sha256,
        30, FROZEN_CONFIG.cpu_workers, FROZEN_CONFIG.inner_threads, "b" * 40,
    )
    checkpoint = attempt / "checkpoint_antigravity"
    store = Exp444CheckpointStoreAntigravity.create(checkpoint, context, process)
    writer = ProgressWriterAntigravity.create(
        attempt, total=30, process=process,
        checkpoint_path=checkpoint / "state_antigravity.json",
    )

    def result(index: int) -> CandidateStabilityResultAntigravity:
        candidate = SourceCandidateAntigravity(f"Live_{index}", index, 16, (1, 2), index + 1)
        scores = (
            DepthViewScoreAntigravity(1, 2.0, 1.0, True, 2.0, 0.5),
            DepthViewScoreAntigravity(2, 2.0, 1.0, True, 2.0, 0.5),
        )
        view = StabilityViewAntigravity(
            OmissionMaskAntigravity(0, (0,), hashlib.sha256(f"live-{index}".encode()).hexdigest()),
            (1, 2, 3, 4), scores, (1, 2), True,
        )
        return CandidateStabilityResultAntigravity(candidate, (view,), 1, 1, "stable_review")

    for index in range(sequence):
        writer.start_candidate_antigravity(f"Live_{index}", index, train_count=4, test_count=1, raw_length=16)
        store.commit_antigravity(index, result(index))
        writer.complete_candidate_antigravity()
    writer.start_candidate_antigravity(f"Live_{sequence}", sequence, train_count=4, test_count=1, raw_length=16)
    store.commit_antigravity(sequence, result(sequence))
    return writer, store.committed_prefix_authority_antigravity(sequence), {
        "train_count": 4, "test_count": 1, "test_rows": 1,
        "raw_length": 16, "target_length": 16,
    }


class Exp444ProgressAntigravityTests(unittest.TestCase):
    def test_exp445_committed_recovery_rejects_boolean_current_candidate_index_antigravity(self) -> None:
        """JSON booleans are never valid exact candidate indices, including 0/1 equality traps."""
        for sequence, forged in ((0, False), (1, True)):
            with self.subTest(sequence=sequence), tempfile.TemporaryDirectory(prefix="exp445_boolean_index_antigravity_") as directory:
                attempt = Path(directory) / "attempt_0000_antigravity"; attempt.mkdir()
                writer, authority, dimensions = _committed_candidate_fixture_antigravity(attempt, sequence)
                progress_path = attempt / "progress_antigravity.json"
                payload = json.loads(progress_path.read_text())
                payload["current_candidate_index"] = forged
                progress_path.write_bytes(canonical_json_bytes_antigravity(payload))
                with self.assertRaisesRegex(ValueError, "recovery authority"):
                    writer.reconcile_committed_success_antigravity(
                        authority, dataset=f"Live_{sequence}", dimensions=dimensions,
                    )

    def test_exp445_preexisting_log_failure_marker_blocks_clean_success_antigravity(self) -> None:
        """A pre-existing regular file or symlink is not trusted as this run's create-only marker."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import ProgressWriterAntigravity

        for kind in ("regular", "symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(prefix="exp445_existing_marker_antigravity_") as directory:
                attempt = Path(directory) / "attempt_0000_antigravity"; attempt.mkdir()
                writer = ProgressWriterAntigravity.create(
                    attempt, total=1, pid=44_604,
                    _exp445_diagnostic_log_failures_antigravity=True,
                )
                marker = attempt / "log_failure_antigravity.json"
                if kind == "regular":
                    marker.write_bytes(canonical_json_bytes_antigravity({"forged": True}))
                    before = marker.read_bytes()
                else:
                    target = attempt / "marker_target_antigravity.json"
                    target.write_text("target\n")
                    marker.symlink_to(target)
                    before = target.read_bytes()
                original_event = ProgressWriterAntigravity._append_event_antigravity
                def one_start_fault(instance, event, stage, extra=None):
                    if event == "candidate_start":
                        raise OSError("preexisting marker append fault")
                    return original_event(instance, event, stage, extra)
                with mock.patch.object(ProgressWriterAntigravity, "_append_event_antigravity", new=one_start_fault):
                    writer.start_candidate_antigravity("Marker", 0, train_count=5, test_count=1, raw_length=16)
                    writer.complete_candidate_antigravity()
                    writer.close_antigravity("complete")
                self.assertTrue(writer.log_failure_blocks_success_antigravity)
                self.assertEqual((marker.read_bytes() if kind == "regular" else target.read_bytes()), before)

    def test_exp445_committed_recovery_rejects_forged_official_history_antigravity(self) -> None:
        """A durable prefix fixes all Exp445 progress counters and candidate dimensions."""
        mutations = (
            ("imported", lambda progress: progress.update({"imported": 1, "recomputed": 0})),
            ("recomputed", lambda progress: progress.update({"imported": 1, "recomputed": 0})),
            ("abstention", lambda progress: progress.update({"abstention": 1})),
            ("train_missing", lambda progress: progress.update({"train_missing": 0})),
            ("train_used", lambda progress: progress.update({"train_used": 0})),
            ("train_excluded", lambda progress: progress.update({"train_excluded": 0})),
        )
        for name, mutate in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix="exp445_forged_progress_antigravity_") as directory:
                attempt = Path(directory) / "attempt_0000_antigravity"; attempt.mkdir()
                writer, authority, dimensions = _committed_success_fixture_antigravity(attempt)
                progress_path = attempt / "progress_antigravity.json"
                payload = json.loads(progress_path.read_text())
                mutate(payload)
                progress_path.write_bytes(canonical_json_bytes_antigravity(payload))
                with self.assertRaisesRegex(ValueError, "recovery authority"):
                    writer.reconcile_committed_success_antigravity(
                        authority, dataset="Recovery", dimensions=dimensions,
                    )

    def test_exp445_diagnostic_log_failures_do_not_block_core_terminal_authority_antigravity(self) -> None:
        """Persistent append faults must leave Exp445 progress/PID complete and auditable."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import ProgressWriterAntigravity

        with tempfile.TemporaryDirectory(prefix="exp445_log_failure_progress_antigravity_") as directory:
            attempt = Path(directory) / "attempt_0000_antigravity"
            attempt.mkdir()
            writer = ProgressWriterAntigravity.create(
                attempt, total=1, pid=44_501,
                _exp445_diagnostic_log_failures_antigravity=True,
            )
            with mock.patch.object(ProgressWriterAntigravity, "_append_event_antigravity", side_effect=OSError("persistent event fault")), mock.patch.object(ProgressWriterAntigravity, "_append_text_antigravity", side_effect=OSError("persistent text fault")):
                writer.start_candidate_antigravity("A", 0, train_count=5, test_count=1, raw_length=16)
                first_marker = (attempt / "log_failure_antigravity.json").read_bytes()
                writer.complete_candidate_antigravity()
                writer.close_antigravity("complete")
            progress = json.loads((attempt / "progress_antigravity.json").read_text())
            pid = json.loads((attempt / "pid_antigravity.json").read_text())
            marker = json.loads((attempt / "log_failure_antigravity.json").read_text())
            self.assertEqual((progress["stage"], progress["completed"], pid["active"]), ("complete", 1, False))
            self.assertEqual(marker["schema"], "exp445.log_failure.v1")
            self.assertEqual((attempt / "log_failure_antigravity.json").read_bytes(), first_marker)

    def test_exp445_secondary_log_marker_failure_never_blocks_core_closeout_antigravity(self) -> None:
        """If marker creation also fails, the terminal in-memory flag still blocks clean success."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity import progress_antigravity

        with tempfile.TemporaryDirectory(prefix="exp445_log_marker_secondary_antigravity_") as directory:
            attempt = Path(directory) / "attempt_0000_antigravity"
            attempt.mkdir()
            writer = progress_antigravity.ProgressWriterAntigravity.create(
                attempt, total=1, pid=44_503,
                _exp445_diagnostic_log_failures_antigravity=True,
            )
            with mock.patch.object(progress_antigravity.ProgressWriterAntigravity, "_append_event_antigravity", side_effect=OSError("persistent event fault")), mock.patch.object(progress_antigravity.ProgressWriterAntigravity, "_append_text_antigravity", side_effect=OSError("persistent text fault")), mock.patch.object(progress_antigravity, "write_create_only_antigravity", side_effect=OSError("marker write fault")):
                writer.start_candidate_antigravity("A", 0, train_count=5, test_count=1, raw_length=16)
                writer.complete_candidate_antigravity()
                writer.close_antigravity("complete")
            progress = json.loads((attempt / "progress_antigravity.json").read_text())
            pid = json.loads((attempt / "pid_antigravity.json").read_text())
            self.assertEqual((progress["stage"], pid["active"], writer.terminal_log_failure_antigravity), ("complete", False, True))
            self.assertFalse((attempt / "log_failure_antigravity.json").exists())

    def test_exp445_persistent_log_fault_does_not_mask_orphan_error_closeout_antigravity(self) -> None:
        """A durable orphan error still reaches inactive terminal progress if every append fails."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import ProgressWriterAntigravity

        with tempfile.TemporaryDirectory(prefix="exp445_log_failure_orphan_antigravity_") as directory:
            attempt = Path(directory) / "attempt_0000_antigravity"
            attempt.mkdir()
            writer = ProgressWriterAntigravity.create(
                attempt, total=2, pid=44_502,
                _exp445_diagnostic_log_failures_antigravity=True,
            )
            digest = hashlib.sha256(b"durable error").hexdigest()
            with mock.patch.object(ProgressWriterAntigravity, "_append_event_antigravity", side_effect=OSError("persistent event fault")), mock.patch.object(ProgressWriterAntigravity, "_append_text_antigravity", side_effect=OSError("persistent text fault")):
                writer.reconcile_orphan_antigravity(
                    "B", 0, record_sha256=digest, status="error", error="original compute error",
                    train_count=5, test_count=1, raw_length=16,
                )
                writer.close_antigravity("error", "original compute error")
            progress = json.loads((attempt / "progress_antigravity.json").read_text())
            pid = json.loads((attempt / "pid_antigravity.json").read_text())
            self.assertEqual((progress["stage"], progress["completed"], progress["error"], pid["active"]), ("error", 1, 1, False))
            self.assertTrue((attempt / "log_failure_antigravity.json").is_file())

    def test_progress_records_current_dimensions_and_eta_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import ProgressWriterAntigravity

        with tempfile.TemporaryDirectory(prefix="exp444_task6_progress_antigravity_") as directory:
            attempt = Path(directory) / "attempt_0000_antigravity"
            attempt.mkdir()
            progress = attempt / "progress_antigravity.json"
            writer = ProgressWriterAntigravity.create(attempt, total=30, pid=42001, checkpoint_path=attempt / "checkpoint_antigravity/state_antigravity.json")
            writer.start_candidate_antigravity("A", 7, train_count=20, test_count=50, raw_length=144)
            writer.start_view_antigravity(view_index=3, view_count=16, excluded_count=2)
            state = json.loads(progress.read_text())
            self.assertEqual(state["current_dataset"], "A")
            self.assertEqual(state["current_candidate_index"], 7)
            self.assertEqual(state["current_view"], 3)
            self.assertEqual(state["raw_length"], 144)
            self.assertEqual(state["view_count"], 16)
            self.assertIsNotNone(state["eta_seconds"])
            self.assertEqual(state["error"], 0)

    def test_completion_clears_dimensions_and_terminal_eta_and_logs_atomically_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import ProgressWriterAntigravity

        with tempfile.TemporaryDirectory(prefix="exp444_task6_progress_antigravity_") as directory:
            attempt = Path(directory) / "attempt_0000_antigravity"
            attempt.mkdir()
            writer = ProgressWriterAntigravity.create(attempt, total=1, pid=42002)
            writer.start_candidate_antigravity("A", 0, train_count=5, test_count=1, raw_length=16)
            writer.complete_candidate_antigravity()
            writer.close_antigravity("complete")
            progress_bytes = (attempt / "progress_antigravity.json").read_bytes()
            state = json.loads(progress_bytes)
            self.assertFalse(state["active"])
            self.assertEqual(state["stage"], "complete")
            self.assertIsNone(state["current_dataset"])
            self.assertIsNone(state["current_candidate_index"])
            self.assertIsNone(state["current_view"])
            self.assertIsNone(state["eta_seconds"])
            self.assertEqual(state["completed"], 1)
            pid = json.loads((attempt / "pid_antigravity.json").read_text())
            self.assertFalse(pid["active"])
            events = (attempt / "events_antigravity.jsonl").read_text().splitlines()
            self.assertEqual(json.loads(events[-1])["event"], "terminal")
            self.assertIn("종료 stage=complete", (attempt / "text_antigravity.log").read_text())
            before = progress_bytes
            with self.assertRaisesRegex(ValueError, "terminal"):
                writer.start_candidate_antigravity("B", 0, train_count=1, test_count=1, raw_length=1)
            self.assertEqual(before, (attempt / "progress_antigravity.json").read_bytes())

    def test_interrupted_progress_restores_counters_and_rejects_live_resume_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import ProgressWriterAntigravity

        with tempfile.TemporaryDirectory(prefix="exp444_task6_progress_antigravity_") as directory:
            attempt = Path(directory) / "attempt_0000_antigravity"
            attempt.mkdir()
            writer = ProgressWriterAntigravity.create(attempt, total=30, pid=42003, start_identity="start-A")
            writer.start_candidate_antigravity("A", 0, train_count=5, test_count=1, raw_length=16)
            writer.complete_candidate_antigravity()
            writer.close_antigravity("interrupted")
            _write_checkpoint_state_antigravity(
                attempt, pid=42003, start_identity="start-A", completed=1, total=30
            )
            with self.assertRaisesRegex(ValueError, "active"):
                ProgressWriterAntigravity.open_interrupted(attempt, total=30, pid=42004, start_identity="start-B", pid_probe=lambda _: True)
            resumed = ProgressWriterAntigravity.open_interrupted(attempt, total=30, pid=42004, start_identity="start-B", pid_probe=lambda _: False)
            self.assertEqual(resumed.completed_count_antigravity, 1)
            self.assertEqual(resumed.error_count_antigravity, 0)
            self.assertEqual(json.loads((attempt / "progress_antigravity.json").read_text())["active"], True)

    def test_orphan_reconciliation_is_exact_next_validated_and_truthful_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import ProgressWriterAntigravity

        with tempfile.TemporaryDirectory(prefix="exp444_task7_progress_orphan_antigravity_") as directory:
            attempt = Path(directory) / "attempt_0000_antigravity"
            attempt.mkdir()
            writer = ProgressWriterAntigravity.create(
                attempt, total=30, pid=9_120_001, start_identity="owner-orphan"
            )
            writer.start_candidate_antigravity("A", 0, train_count=5, test_count=1, raw_length=16)
            writer.complete_candidate_antigravity()
            writer.close_antigravity("interrupted")
            _write_checkpoint_state_antigravity(
                attempt, pid=9_120_001, start_identity="owner-orphan", completed=1, total=30
            )
            resumed = ProgressWriterAntigravity.open_interrupted(
                attempt,
                total=30,
                pid=9_120_002,
                start_identity="replacement-orphan",
                pid_probe=lambda _: False,
            )
            evidence_paths = (
                attempt / "progress_antigravity.json",
                attempt / "events_antigravity.jsonl",
                attempt / "text_antigravity.log",
            )
            before = tuple(path.read_bytes() for path in evidence_paths)
            with self.assertRaisesRegex(ValueError, "hash"):
                resumed.reconcile_orphan_antigravity(
                    "B", 1, record_sha256="bad", status="success",
                    train_count=6, test_count=1, raw_length=20,
                )
            self.assertEqual(before, tuple(path.read_bytes() for path in evidence_paths))
            digest = hashlib.sha256(b"accepted orphan record").hexdigest()
            resumed.reconcile_orphan_antigravity(
                "B", 1, record_sha256=digest, status="success",
                train_count=6, test_count=1, raw_length=20,
            )
            state = json.loads((attempt / "progress_antigravity.json").read_text())
            event = json.loads((attempt / "events_antigravity.jsonl").read_text().splitlines()[-1])
            self.assertEqual((state["completed"], state["recomputed"], state["success"], state["error"]), (2, 2, 2, 0))
            self.assertEqual((state["current_dataset"], state["current_candidate_index"]), (None, None))
            self.assertEqual((event["event"], event["dataset"], event["record_sha256"]), ("orphan_reconciled", "B", digest))
            self.assertIn("고아 레코드 복구 dataset=B", (attempt / "text_antigravity.log").read_text())

    def test_resume_authority_preflight_runs_before_first_owner_write_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import ProgressWriterAntigravity

        with tempfile.TemporaryDirectory(prefix="exp444_task7_progress_preflight_antigravity_") as directory:
            attempt = Path(directory) / "attempt_0000_antigravity"
            attempt.mkdir()
            writer = ProgressWriterAntigravity.create(
                attempt, total=30, pid=42003, start_identity="start-A"
            )
            writer.start_candidate_antigravity(
                "A", 0, train_count=5, test_count=1, raw_length=16
            )
            writer.complete_candidate_antigravity()
            writer.close_antigravity("interrupted")
            _write_checkpoint_state_antigravity(
                attempt, pid=42003, start_identity="start-A", completed=1, total=30
            )
            before = {
                path.relative_to(attempt).as_posix(): path.read_bytes()
                for path in sorted(attempt.rglob("*"))
                if path.is_file()
            }
            calls = 0

            def reject_changed_authority_antigravity() -> None:
                nonlocal calls
                calls += 1
                raise ValueError("checkpoint record changed")

            with self.assertRaisesRegex(ValueError, "record changed"):
                ProgressWriterAntigravity.open_interrupted(
                    attempt,
                    total=30,
                    pid=42004,
                    start_identity="start-B",
                    pid_probe=lambda _: False,
                    authority_preflight=reject_changed_authority_antigravity,
                )
            after = {
                path.relative_to(attempt).as_posix(): path.read_bytes()
                for path in sorted(attempt.rglob("*"))
                if path.is_file()
            }
            self.assertEqual(calls, 1)
            self.assertEqual(before, after)

    def test_stale_temps_are_discarded_without_becoming_progress_authority_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import ProgressWriterAntigravity

        with tempfile.TemporaryDirectory(prefix="exp444_task6_progress_antigravity_") as directory:
            attempt = Path(directory) / "attempt_0000_antigravity"
            attempt.mkdir()
            writer = ProgressWriterAntigravity.create(attempt, total=30, pid=42003, start_identity="start-A")
            writer.start_candidate_antigravity("A", 0, train_count=5, test_count=1, raw_length=16)
            writer.complete_candidate_antigravity()
            writer.close_antigravity("interrupted")
            _write_checkpoint_state_antigravity(
                attempt, pid=42003, start_identity="start-A", completed=1, total=30
            )
            stale_progress = attempt / "progress_antigravity.tmp"
            stale_pid = attempt / "pid_antigravity.tmp"
            stale_progress.write_bytes(b'{"completed":999}\n')
            stale_pid.write_bytes(b'{"pid":999}\n')

            resumed = ProgressWriterAntigravity.open_interrupted(
                attempt, total=30, pid=42004, start_identity="start-B", pid_probe=lambda _: False
            )

            self.assertEqual(resumed.completed_count_antigravity, 1)
            self.assertFalse(stale_progress.exists())
            self.assertFalse(stale_pid.exists())

    def test_resume_validates_all_progress_semantics_before_mutation_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import ProgressWriterAntigravity

        corruptions = {
            "completed exceeds total": {"completed": 31},
            "backend changed": {"backend": "mlx"},
            "inactive current dataset": {"current_dataset": "should-be-null"},
            "negative dimensions": {"train_count": -1},
            "counter mismatch": {"success": 0},
            "checkpoint path changed": {"checkpoint": "/tmp/wrong_antigravity.json"},
        }
        for label, mutation in corruptions.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory(prefix="exp444_task6_progress_antigravity_") as directory:
                attempt = Path(directory) / "attempt_0000_antigravity"
                attempt.mkdir()
                writer = ProgressWriterAntigravity.create(attempt, total=30, pid=42003, start_identity="start-A")
                writer.start_candidate_antigravity("A", 0, train_count=5, test_count=1, raw_length=16)
                writer.complete_candidate_antigravity()
                writer.close_antigravity("interrupted")
                _write_checkpoint_state_antigravity(
                    attempt, pid=42003, start_identity="start-A", completed=1, total=30
                )
                progress_path = attempt / "progress_antigravity.json"
                payload = json.loads(progress_path.read_text())
                payload.update(mutation)
                progress_path.write_bytes(canonical_json_bytes_antigravity(payload))
                before = {path.name: path.read_bytes() for path in attempt.iterdir() if path.is_file()}

                with self.assertRaisesRegex(ValueError, "invalid|changed|mismatch"):
                    ProgressWriterAntigravity.open_interrupted(
                        attempt, total=30, pid=42004, start_identity="start-B", pid_probe=lambda _: False
                    )
                after = {path.name: path.read_bytes() for path in attempt.iterdir() if path.is_file()}
                self.assertEqual(before, after)

    def test_replacement_pid_old_writer_and_parent_owner_are_enforced_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import ProgressWriterAntigravity

        with tempfile.TemporaryDirectory(prefix="exp444_task6_progress_antigravity_") as directory:
            attempt = Path(directory) / "attempt_0000_antigravity"
            attempt.mkdir()
            writer = ProgressWriterAntigravity.create(attempt, total=30, pid=42003, start_identity="start-A")
            with mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity.os.getpid", return_value=os.getpid() + 1):
                with self.assertRaisesRegex(ValueError, "parent owner"):
                    writer.start_candidate_antigravity("A", 0, train_count=5, test_count=1, raw_length=16)
            writer.start_candidate_antigravity("A", 0, train_count=5, test_count=1, raw_length=16)
            writer.complete_candidate_antigravity()
            writer.close_antigravity("interrupted")
            _write_checkpoint_state_antigravity(
                attempt, pid=42003, start_identity="start-A", completed=1, total=30
            )
            with self.assertRaisesRegex(ValueError, "PID|numeric"):
                ProgressWriterAntigravity.open_interrupted(
                    attempt, total=30, pid=42003, start_identity="different-token", pid_probe=lambda _: False
                )
            with self.assertRaisesRegex(ValueError, "terminal"):
                writer.close_antigravity("interrupted")
            with self.assertRaisesRegex(ValueError, "terminal"):
                writer.start_candidate_antigravity("B", 1, train_count=5, test_count=1, raw_length=16)

    def test_resume_rejects_checkpoint_identity_mismatch_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import ProgressWriterAntigravity

        with tempfile.TemporaryDirectory(prefix="exp444_task6_progress_antigravity_") as directory:
            attempt = Path(directory) / "attempt_0000_antigravity"
            attempt.mkdir()
            writer = ProgressWriterAntigravity.create(attempt, total=30, pid=42003, start_identity="start-A")
            writer.start_candidate_antigravity("A", 0, train_count=5, test_count=1, raw_length=16)
            writer.complete_candidate_antigravity()
            writer.close_antigravity("interrupted")
            checkpoint = _write_checkpoint_state_antigravity(
                attempt, pid=99999, start_identity="wrong-owner", completed=1, total=30
            )
            before = checkpoint.read_bytes()
            with self.assertRaisesRegex(ValueError, "checkpoint.*identity|checkpoint.*PID"):
                ProgressWriterAntigravity.open_interrupted(
                    attempt, total=30, pid=42004, start_identity="start-B", pid_probe=lambda _: False
                )
            self.assertEqual(before, checkpoint.read_bytes())

    def test_resume_rejects_corrupt_checkpoint_semantics_before_mutation_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.progress_antigravity import ProgressWriterAntigravity

        corruptions = {
            "payload hash": {"payload_sha256": "0" * 64},
            "completed mismatch": {"completed_units": 2},
            "active checkpoint current": {"current_unit": 0},
            "bad context hash": {"context_sha256": "not-a-sha"},
        }
        for label, mutation in corruptions.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory(prefix="exp444_task6_progress_antigravity_") as directory:
                attempt = Path(directory) / "attempt_0000_antigravity"
                attempt.mkdir()
                writer = ProgressWriterAntigravity.create(attempt, total=30, pid=42003, start_identity="start-A")
                writer.start_candidate_antigravity("A", 0, train_count=5, test_count=1, raw_length=16)
                writer.complete_candidate_antigravity()
                writer.close_antigravity("interrupted")
                checkpoint = _write_checkpoint_state_antigravity(
                    attempt, pid=42003, start_identity="start-A", completed=1, total=30
                )
                payload = json.loads(checkpoint.read_text())
                payload.update(mutation)
                checkpoint.write_bytes(canonical_json_bytes_antigravity(payload))
                before = {path.name: path.read_bytes() for path in attempt.iterdir() if path.is_file()}

                with self.assertRaisesRegex(ValueError, "checkpoint.*invalid|checkpoint.*mismatch|checkpoint.*changed"):
                    ProgressWriterAntigravity.open_interrupted(
                        attempt, total=30, pid=42004, start_identity="start-B", pid_probe=lambda _: False
                    )
                after = {path.name: path.read_bytes() for path in attempt.iterdir() if path.is_file()}
                self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
