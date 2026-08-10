"""TDD tests for the Exp444 ordered, create-only checkpoint."""
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


def _context_antigravity(total: int = 30) -> CheckpointContextAntigravity:
    return CheckpointContextAntigravity(
        FROZEN_CONFIG.experiment_id,
        FROZEN_CONFIG.run_identity,
        scientific_config_sha256_antigravity(),
        FROZEN_CONFIG.environment_lock_sha256,
        total,
        FROZEN_CONFIG.cpu_workers,
        FROZEN_CONFIG.inner_threads,
        "a" * 40,
    )


def _result_antigravity(sequence: int) -> CandidateStabilityResultAntigravity:
    source = SourceCandidateAntigravity(
        f"Fixture_{sequence:02d}", sequence, 16, (1, 2), sequence + 1
    )
    scores = (
        DepthViewScoreAntigravity(1, 2.0, 1.0, True, 2.0, 0.5),
        DepthViewScoreAntigravity(2, 2.0, 1.0, True, 2.0, 0.5),
    )
    view = StabilityViewAntigravity(
        OmissionMaskAntigravity(0, (0,), hashlib.sha256(f"{sequence}".encode()).hexdigest()),
        tuple(range(1, 5)), scores, (1, 2), True,
    )
    return CandidateStabilityResultAntigravity(source, (view,), 1, 1, "stable_review")


def _process_antigravity(pid: int = 41001, start: str = "start-A"):
    from experiments.exp444_train_only_stability_precision_gate_antigravity.checkpoint_antigravity import ProcessIdentityAntigravity
    return ProcessIdentityAntigravity(pid, start)


class Exp444CheckpointAntigravityTests(unittest.TestCase):
    def test_committed_prefix_authority_pins_exact_checkpoint_bytes_antigravity(self) -> None:
        """Progress recovery must receive a checkpoint-owned immutable prefix, not loose paths."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity.checkpoint_antigravity import (
            CommittedPrefixAuthorityAntigravity,
            Exp444CheckpointStoreAntigravity,
        )

        with tempfile.TemporaryDirectory(prefix="exp445_committed_prefix_antigravity_") as directory:
            root = Path(directory) / "checkpoint_antigravity"
            store = Exp444CheckpointStoreAntigravity.create(root, _context_antigravity(), _process_antigravity())
            store.commit_antigravity(0, _result_antigravity(0))
            authority = store.committed_prefix_authority_antigravity(0)
            self.assertIs(type(authority), CommittedPrefixAuthorityAntigravity)
            self.assertEqual(
                (authority.sequence, authority.total_units, authority.record_sha256_prefix),
                (0, 30, store.record_sha256_prefix_antigravity),
            )
            self.assertEqual(authority.result, _result_antigravity(0))

            state_path = root / "state_antigravity.json"
            state = json.loads(state_path.read_text())
            state["unexpected"] = "forged"
            state_path.write_bytes(canonical_json_bytes_antigravity(state))
            with self.assertRaisesRegex(ValueError, "authority|schema"):
                store.committed_prefix_authority_antigravity(0)

    def test_committed_prefix_authority_rejects_forged_state_and_namespace_antigravity(self) -> None:
        """No symlink, extra entry, forged state, or changed record can become recovery authority."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity.checkpoint_antigravity import Exp444CheckpointStoreAntigravity

        def mutate_state(path: Path, mutation) -> None:
            payload = json.loads(path.read_text())
            mutation(payload)
            path.write_bytes(canonical_json_bytes_antigravity(payload))

        cases = (
            ("extra_state_key", lambda root: mutate_state(root / "state_antigravity.json", lambda state: state.update({"extra": 1}))),
            ("forged_completed", lambda root: mutate_state(root / "state_antigravity.json", lambda state: state.update({"completed_units": 0}))),
            ("forged_context", lambda root: mutate_state(root / "state_antigravity.json", lambda state: state.update({"context_sha256": "0" * 64}))),
            ("forged_error", lambda root: mutate_state(root / "state_antigravity.json", lambda state: state.update({"status": "error", "error_count": 1, "error": {"type": "RuntimeError", "message": "forged"}}))),
            ("forged_prefix", lambda root: mutate_state(root / "state_antigravity.json", lambda state: state.update({"record_sha256_prefix": ["0" * 64], "payload_sha256": hashlib.sha256(canonical_json_bytes_antigravity(["0" * 64])).hexdigest()}))),
            ("extra_root_entry", lambda root: (root / "extra_antigravity.json").write_text("{}\n")),
            ("record_symlink", lambda root: ((root / "records_antigravity/0000_record_antigravity.json").unlink(), (root / "records_antigravity/0000_record_antigravity.json").symlink_to(root / "context_antigravity.json"))),
        )
        for name, mutate in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory(prefix="exp445_committed_prefix_attack_antigravity_") as directory:
                root = Path(directory) / "checkpoint_antigravity"
                store = Exp444CheckpointStoreAntigravity.create(root, _context_antigravity(), _process_antigravity())
                store.commit_antigravity(0, _result_antigravity(0))
                mutate(root)
                with self.assertRaisesRegex(ValueError, "authority|changed|invalid"):
                    store.committed_prefix_authority_antigravity(0)

    def test_resume_rejects_a_different_reviewed_commit_before_checkpoint_mutation_antigravity(self) -> None:
        """A different reviewed commit cannot take ownership of a valid interrupted checkpoint."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity.checkpoint_antigravity import Exp444CheckpointStoreAntigravity

        with tempfile.TemporaryDirectory(prefix="exp444_checkpoint_commit_antigravity_") as directory:
            root = Path(directory) / "checkpoint_antigravity"
            store = Exp444CheckpointStoreAntigravity.create(root, _context_antigravity(), _process_antigravity())
            store.mark_interrupted_antigravity()
            before = (root / "state_antigravity.json").read_bytes()
            other = CheckpointContextAntigravity(
                FROZEN_CONFIG.experiment_id, FROZEN_CONFIG.run_identity,
                scientific_config_sha256_antigravity(), FROZEN_CONFIG.environment_lock_sha256,
                30, FROZEN_CONFIG.cpu_workers, FROZEN_CONFIG.inner_threads, "b" * 40,
            )
            with self.assertRaisesRegex(ValueError, "context"):
                Exp444CheckpointStoreAntigravity.open_interrupted(
                    root, other, _process_antigravity(41002, "replacement"), pid_probe=lambda _: False,
                )
            self.assertEqual(before, (root / "state_antigravity.json").read_bytes())

    def test_create_only_order_and_terminal_immutability_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.checkpoint_antigravity import Exp444CheckpointStoreAntigravity

        with tempfile.TemporaryDirectory(prefix="exp444_task6_checkpoint_antigravity_") as directory:
            root = Path(directory) / "checkpoint_antigravity"
            store = Exp444CheckpointStoreAntigravity.create(root, _context_antigravity(), _process_antigravity())
            with self.assertRaisesRegex(ValueError, "next"):
                store.commit_antigravity(1, _result_antigravity(1))
            for sequence in range(30):
                store.commit_antigravity(sequence, _result_antigravity(sequence))
            state = json.loads((root / "state_antigravity.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["completed_units"], 30)
            self.assertEqual(len(state["record_sha256_prefix"]), 30)
            before = (root / "state_antigravity.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "terminal"):
                store.commit_antigravity(0, _result_antigravity(0))
            self.assertEqual(before, (root / "state_antigravity.json").read_bytes())

    def test_orphan_exact_next_is_reconciled_and_corrupt_orphan_rejected_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity
        from experiments.exp444_train_only_stability_precision_gate_antigravity.checkpoint_antigravity import Exp444CheckpointStoreAntigravity

        with tempfile.TemporaryDirectory(prefix="exp444_task6_checkpoint_antigravity_") as directory:
            root = Path(directory) / "checkpoint_antigravity"
            store = Exp444CheckpointStoreAntigravity.create(root, _context_antigravity(), _process_antigravity())
            store.commit_antigravity(0, _result_antigravity(0))
            # Simulate a crash after the record fsync and before state replace.
            result = _result_antigravity(1)
            record = {"sequence": 1, "result": result.to_payload()}
            record_path = root / "records_antigravity" / "0001_record_antigravity.json"
            record_path.write_bytes(canonical_json_bytes_antigravity(record))
            resumed = Exp444CheckpointStoreAntigravity.open_interrupted(
                root, _context_antigravity(), _process_antigravity(41002, "start-B"), pid_probe=lambda _: False
            )
            self.assertEqual(resumed.completed_count_antigravity, 2)

            with tempfile.TemporaryDirectory(prefix="exp444_task6_checkpoint_antigravity_") as second:
                bad_root = Path(second) / "checkpoint_antigravity"
                bad = Exp444CheckpointStoreAntigravity.create(bad_root, _context_antigravity(), _process_antigravity())
                bad.commit_antigravity(0, _result_antigravity(0))
                (bad_root / "records_antigravity" / "0001_record_antigravity.json").write_bytes(b"not-json")
                with self.assertRaisesRegex(ValueError, "orphan|record"):
                    Exp444CheckpointStoreAntigravity.open_interrupted(
                        bad_root, _context_antigravity(), _process_antigravity(41003, "start-C"), pid_probe=lambda _: False
                    )

    def test_read_only_inspection_describes_result_and_error_orphans_without_writes_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity
        from experiments.exp444_train_only_stability_precision_gate_antigravity.checkpoint_antigravity import Exp444CheckpointStoreAntigravity

        cases = (
            ("result", {"sequence": 1, "result": _result_antigravity(1).to_payload()}),
            (
                "error",
                {
                    "sequence": 1,
                    "result": None,
                    "status": "error",
                    "error": {"type": "RuntimeError", "message": "durable failure"},
                },
            ),
        )
        for offset, (case, payload) in enumerate(cases):
            with self.subTest(case=case), tempfile.TemporaryDirectory(prefix="exp444_task7_inspection_antigravity_") as directory:
                root = Path(directory) / "checkpoint_antigravity"
                store = Exp444CheckpointStoreAntigravity.create(
                    root, _context_antigravity(), _process_antigravity(9_110_001 + offset * 2, f"owner-{case}")
                )
                store.commit_antigravity(0, _result_antigravity(0))
                record_bytes = canonical_json_bytes_antigravity(payload)
                (root / "records_antigravity/0001_record_antigravity.json").write_bytes(record_bytes)
                before = {
                    path.relative_to(root).as_posix(): path.read_bytes()
                    for path in sorted(root.rglob("*")) if path.is_file()
                }
                inspection = Exp444CheckpointStoreAntigravity.inspect_interrupted_antigravity(
                    root,
                    _context_antigravity(),
                    _process_antigravity(9_110_002 + offset * 2, f"replacement-{case}"),
                    pid_probe=lambda _: False,
                )
                after = {
                    path.relative_to(root).as_posix(): path.read_bytes()
                    for path in sorted(root.rglob("*")) if path.is_file()
                }
                self.assertEqual(before, after)
                self.assertEqual((inspection.stored_completed_units, inspection.accepted_completed_units), (1, 2))
                self.assertEqual(inspection.orphan_sequence, 1)
                self.assertEqual(inspection.accepted_record_sha256_prefix[-1], hashlib.sha256(record_bytes).hexdigest())
                if case == "result":
                    self.assertEqual(inspection.orphan_result, _result_antigravity(1))
                    self.assertFalse(inspection.orphan_is_error_antigravity)
                else:
                    self.assertIsNone(inspection.orphan_result)
                    self.assertEqual((inspection.orphan_error_type, inspection.orphan_error_message), ("RuntimeError", "durable failure"))
                    self.assertTrue(inspection.orphan_is_error_antigravity)

    def test_resume_pid_rules_interruption_and_complete_finalization_reopen_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.checkpoint_antigravity import Exp444CheckpointStoreAntigravity

        with tempfile.TemporaryDirectory(prefix="exp444_task6_checkpoint_antigravity_") as directory:
            root = Path(directory) / "checkpoint_antigravity"
            store = Exp444CheckpointStoreAntigravity.create(root, _context_antigravity(), _process_antigravity())
            store.commit_antigravity(0, _result_antigravity(0))
            store.mark_interrupted_antigravity()
            with self.assertRaisesRegex(ValueError, "active"):
                Exp444CheckpointStoreAntigravity.open_interrupted(
                    root, _context_antigravity(), _process_antigravity(41002, "start-B"), pid_probe=lambda _: True
                )
            with self.assertRaisesRegex(ValueError, "distinct"):
                Exp444CheckpointStoreAntigravity.open_interrupted(
                    root, _context_antigravity(), _process_antigravity(), pid_probe=lambda _: False
                )
            resumed = Exp444CheckpointStoreAntigravity.open_interrupted(
                root, _context_antigravity(), _process_antigravity(41002, "start-B"), pid_probe=lambda _: False
            )
            resumed.commit_antigravity(1, _result_antigravity(1))
            for sequence in range(2, 30):
                resumed.commit_antigravity(sequence, _result_antigravity(sequence))
            before = (root / "state_antigravity.json").read_bytes()
            final = Exp444CheckpointStoreAntigravity.open_interrupted(
                root, _context_antigravity(), _process_antigravity(41004, "start-D"),
                pid_probe=lambda _: False, allow_complete_finalization=True,
            )
            self.assertTrue(final.finalization_only_antigravity)
            self.assertEqual(before, (root / "state_antigravity.json").read_bytes())
            with self.assertRaisesRegex(ValueError, "finalization|terminal"):
                final.commit_antigravity(0, _result_antigravity(0))

    def test_stale_atomic_temp_is_discarded_and_never_authoritative_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.checkpoint_antigravity import Exp444CheckpointStoreAntigravity

        with tempfile.TemporaryDirectory(prefix="exp444_task6_checkpoint_antigravity_") as directory:
            root = Path(directory) / "checkpoint_antigravity"
            store = Exp444CheckpointStoreAntigravity.create(root, _context_antigravity(), _process_antigravity())
            store.commit_antigravity(0, _result_antigravity(0))
            store.mark_interrupted_antigravity()
            stale = root / "state_antigravity.tmp"
            stale.write_bytes(b'{"completed_units":999}\n')

            resumed = Exp444CheckpointStoreAntigravity.open_interrupted(
                root, _context_antigravity(), _process_antigravity(41002, "start-B"), pid_probe=lambda _: False
            )

            self.assertFalse(stale.exists())
            self.assertEqual(resumed.completed_count_antigravity, 1)
            self.assertEqual(set(resumed.results_antigravity()), {0})

    def test_resume_rejects_tampered_prefix_hash_before_mutation_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity
        from experiments.exp444_train_only_stability_precision_gate_antigravity.checkpoint_antigravity import Exp444CheckpointStoreAntigravity

        for status in ("running", "interrupted", "complete"):
            with self.subTest(status=status), tempfile.TemporaryDirectory(prefix="exp444_task6_checkpoint_antigravity_") as directory:
                root = Path(directory) / "checkpoint_antigravity"
                store = Exp444CheckpointStoreAntigravity.create(root, _context_antigravity(), _process_antigravity())
                limit = 30 if status == "complete" else 1
                for sequence in range(limit):
                    store.commit_antigravity(sequence, _result_antigravity(sequence))
                if status == "interrupted":
                    store.mark_interrupted_antigravity()
                state_path = root / "state_antigravity.json"
                payload = json.loads(state_path.read_text())
                payload["payload_sha256"] = "0" * 64
                state_path.write_bytes(canonical_json_bytes_antigravity(payload))
                before = state_path.read_bytes()

                with self.assertRaisesRegex(ValueError, "payload hash"):
                    Exp444CheckpointStoreAntigravity.open_interrupted(
                        root, _context_antigravity(), _process_antigravity(41002, "start-B"),
                        pid_probe=lambda _: False, allow_complete_finalization=status == "complete",
                    )
                self.assertEqual(before, state_path.read_bytes())

    def test_read_only_interrupted_preflight_rejects_changed_record_without_mutation_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.artifacts_antigravity import canonical_json_bytes_antigravity
        from experiments.exp444_train_only_stability_precision_gate_antigravity.checkpoint_antigravity import Exp444CheckpointStoreAntigravity

        with tempfile.TemporaryDirectory(prefix="exp444_task7_preflight_antigravity_") as directory:
            root = Path(directory) / "checkpoint_antigravity"
            store = Exp444CheckpointStoreAntigravity.create(
                root, _context_antigravity(), _process_antigravity()
            )
            store.commit_antigravity(0, _result_antigravity(0))
            store.mark_interrupted_antigravity()
            record_path = root / "records_antigravity/0000_record_antigravity.json"
            payload = json.loads(record_path.read_text())
            payload["result"]["candidate"]["dataset"] = "TamperedDatasetAntigravity"
            record_path.write_bytes(canonical_json_bytes_antigravity(payload))
            before = {
                path.relative_to(root).as_posix(): path.read_bytes()
                for path in sorted(root.rglob("*"))
                if path.is_file()
            }
            with self.assertRaisesRegex(ValueError, "record"):
                Exp444CheckpointStoreAntigravity.inspect_interrupted_antigravity(
                    root,
                    _context_antigravity(),
                    _process_antigravity(41002, "start-B"),
                    pid_probe=lambda _: False,
                )
            after = {
                path.relative_to(root).as_posix(): path.read_bytes()
                for path in sorted(root.rglob("*"))
                if path.is_file()
            }
            self.assertEqual(before, after)

    def test_same_numeric_pid_and_interrupted_old_writer_are_rejected_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.checkpoint_antigravity import Exp444CheckpointStoreAntigravity

        with tempfile.TemporaryDirectory(prefix="exp444_task6_checkpoint_antigravity_") as directory:
            root = Path(directory) / "checkpoint_antigravity"
            store = Exp444CheckpointStoreAntigravity.create(root, _context_antigravity(), _process_antigravity())
            store.commit_antigravity(0, _result_antigravity(0))
            store.mark_interrupted_antigravity()
            before = (root / "state_antigravity.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "PID|numeric"):
                Exp444CheckpointStoreAntigravity.open_interrupted(
                    root, _context_antigravity(), _process_antigravity(41001, "different-token"), pid_probe=lambda _: False
                )
            with self.assertRaisesRegex(ValueError, "interrupted|writable"):
                store.commit_antigravity(1, _result_antigravity(1))
            self.assertEqual(before, (root / "state_antigravity.json").read_bytes())

    def test_parent_owner_and_terminal_error_record_are_enforced_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.checkpoint_antigravity import Exp444CheckpointStoreAntigravity

        with tempfile.TemporaryDirectory(prefix="exp444_task6_checkpoint_antigravity_") as directory:
            root = Path(directory) / "checkpoint_antigravity"
            store = Exp444CheckpointStoreAntigravity.create(root, _context_antigravity(), _process_antigravity())
            with mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.checkpoint_antigravity.os.getpid", return_value=os.getpid() + 1):
                with self.assertRaisesRegex(ValueError, "parent owner"):
                    store.commit_antigravity(0, _result_antigravity(0))
            store.commit_error_antigravity(0, RuntimeError("fixture failure"))
            state = json.loads((root / "state_antigravity.json").read_text())
            record = json.loads((root / "records_antigravity/0000_record_antigravity.json").read_text())
            self.assertEqual(state["status"], "error")
            self.assertEqual(state["error_count"], 1)
            self.assertEqual(record["status"], "error")
            before = (root / "state_antigravity.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "terminal"):
                store.commit_antigravity(1, _result_antigravity(1))
            with self.assertRaisesRegex(ValueError, "terminal error"):
                Exp444CheckpointStoreAntigravity.open_interrupted(
                    root, _context_antigravity(), _process_antigravity(41002, "start-B"), pid_probe=lambda _: False
                )
            self.assertEqual(before, (root / "state_antigravity.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
