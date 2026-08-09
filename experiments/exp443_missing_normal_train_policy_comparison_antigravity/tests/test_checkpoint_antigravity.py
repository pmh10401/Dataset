"""Behavior tests for the Exp443 ordered checkpoint store."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.exp443_missing_normal_train_policy_comparison_antigravity.contracts_antigravity import (
    KernelSeedScoresAntigravity,
    PolicyDatasetResultAntigravity,
)


POLICIES = (
    "linear_edge_nearest_control",
    "exclude_nonfinite_row",
    "leave_one_out_peer_median",
)
SEEDS = (20260721, 20261721)


def _units():
    from experiments.exp443_missing_normal_train_policy_comparison_antigravity.checkpoint_antigravity import (
        WorkUnitAntigravity,
    )

    return tuple(
        WorkUnitAntigravity(sequence=sequence, policy=policy, dataset=dataset)
        for sequence, (policy, dataset) in enumerate(
            ( (policy, dataset) for policy in POLICIES for dataset in ("A", "B") )
        )
    )


def _binding():
    from experiments.exp443_missing_normal_train_policy_comparison_antigravity.checkpoint_antigravity import (
        CheckpointBindingAntigravity,
    )

    digest = "a" * 64
    return CheckpointBindingAntigravity(
        run_identity="exp443-test-antigravity",
        scientific_config_sha256=digest,
        environment_lock_sha256="b" * 64,
        source_snapshot_sha256="c" * 64,
        database_sha256="d" * 64,
        reviewed_commit="e" * 40,
    )


def _process():
    from experiments.exp443_missing_normal_train_policy_comparison_antigravity.checkpoint_antigravity import (
        ProcessIdentityAntigravity,
    )

    return ProcessIdentityAntigravity(pid=4242, start_identity="start-A")


def _result(policy: str, dataset: str) -> PolicyDatasetResultAntigravity:
    scores = []
    for kernel in (1024, 4096):
        for seed in SEEDS:
            scores.append(
                KernelSeedScoresAntigravity(
                    kernel,
                    seed,
                    np.zeros(5, dtype="<f8"),
                    np.zeros(2, dtype="<f8"),
                    (),
                )
            )
    return PolicyDatasetResultAntigravity(
        policy,
        dataset,
        8,
        5,
        0,
        5,
        0,
        2,
        tuple(scores),
        {1024: (), 4096: ()},
        (),
        "scored",
    )


class CheckpointStoreAntigravityTests(unittest.TestCase):
    def test_rejects_unsorted_or_duplicate_policy_major_units(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.checkpoint_antigravity import (
            Exp443CheckpointStoreAntigravity,
        )

        units = _units()
        with tempfile.TemporaryDirectory(prefix="exp443_task7_checkpoint_antigravity_") as directory:
            root = Path(directory) / "checkpoint_antigravity"
            with self.assertRaisesRegex(ValueError, "order"):
                Exp443CheckpointStoreAntigravity.create_antigravity(
                    root, (units[1], units[0], *units[2:]), _binding(), _process()
                )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                Exp443CheckpointStoreAntigravity.create_antigravity(
                    root,
                    (*units[:-1], units[-1].__class__(sequence=5, policy=units[-2].policy, dataset=units[-2].dataset)),
                    _binding(),
                    _process(),
                )

    def test_create_commits_only_next_record_and_completes_exact_prefix(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.checkpoint_antigravity import (
            Exp443CheckpointStoreAntigravity,
        )

        units = _units()
        with tempfile.TemporaryDirectory(prefix="exp443_task7_checkpoint_antigravity_") as directory:
            root = Path(directory) / "checkpoint_antigravity"
            store = Exp443CheckpointStoreAntigravity.create_antigravity(
                root, units, _binding(), _process()
            )
            with self.assertRaisesRegex(ValueError, "next"):
                store.commit_result_antigravity(1, _result(units[1].policy, units[1].dataset))
            digests = []
            for unit in units:
                digest = store.commit_result_antigravity(
                    unit.sequence, _result(unit.policy, unit.dataset)
                )
                payload = (root / "records_antigravity" / f"{unit.sequence:04d}_record_antigravity.json").read_bytes()
                self.assertEqual(digest, hashlib.sha256(payload).hexdigest())
                digests.append(digest)
            state = json.loads((root / "state_antigravity.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertEqual(state["completed_count"], len(units))
            self.assertEqual(state["record_sha256_prefix"], digests)
            with self.assertRaisesRegex(ValueError, "terminal"):
                store.commit_result_antigravity(0, _result(units[0].policy, units[0].dataset))

    def test_wrong_identity_out_of_order_and_error_stop_the_attempt(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.checkpoint_antigravity import (
            Exp443CheckpointStoreAntigravity,
        )

        units = _units()
        with tempfile.TemporaryDirectory(prefix="exp443_task7_checkpoint_antigravity_") as directory:
            root = Path(directory) / "checkpoint_antigravity"
            store = Exp443CheckpointStoreAntigravity.create_antigravity(
                root, units, _binding(), _process()
            )
            with self.assertRaisesRegex(ValueError, "identity"):
                store.commit_result_antigravity(0, _result(units[1].policy, units[1].dataset))
            store.commit_result_antigravity(0, _result(units[0].policy, units[0].dataset))
            store.commit_error_antigravity(1, units[1], RuntimeError("worker failed"))
            state = json.loads((root / "state_antigravity.json").read_text())
            self.assertEqual(state["status"], "error")
            self.assertEqual(state["completed_count"], 2)
            with self.assertRaisesRegex(ValueError, "terminal"):
                store.commit_result_antigravity(2, _result(units[2].policy, units[2].dataset))
            with self.assertRaisesRegex(ValueError, "terminal"):
                Exp443CheckpointStoreAntigravity.open_for_resume_antigravity(
                    root, _binding(), lambda _: False
                )

    def test_resume_rejects_active_binding_and_corrupted_prefix(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.checkpoint_antigravity import (
            Exp443CheckpointStoreAntigravity,
        )

        units = _units()
        with tempfile.TemporaryDirectory(prefix="exp443_task7_checkpoint_antigravity_") as directory:
            root = Path(directory) / "checkpoint_antigravity"
            store = Exp443CheckpointStoreAntigravity.create_antigravity(
                root, units, _binding(), _process()
            )
            store.commit_result_antigravity(0, _result(units[0].policy, units[0].dataset))
            with self.assertRaisesRegex(ValueError, "active"):
                Exp443CheckpointStoreAntigravity.open_for_resume_antigravity(
                    root, _binding(), lambda _: True
                )
            resumed = Exp443CheckpointStoreAntigravity.open_for_resume_antigravity(
                root, _binding(), lambda _: False
            )
            resumed.commit_result_antigravity(1, _result(units[1].policy, units[1].dataset))
            record = root / "records_antigravity" / "0001_record_antigravity.json"
            record.write_bytes(record.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "record"):
                Exp443CheckpointStoreAntigravity.open_for_resume_antigravity(
                    root, _binding(), lambda _: False
                )

    def test_resume_rejects_missing_record_and_malformed_state_prefix(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.artifacts_antigravity import (
            canonical_json_bytes_antigravity,
        )
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.checkpoint_antigravity import (
            Exp443CheckpointStoreAntigravity,
        )

        units = _units()
        with tempfile.TemporaryDirectory(prefix="exp443_task7_checkpoint_antigravity_") as directory:
            root = Path(directory) / "checkpoint_antigravity"
            store = Exp443CheckpointStoreAntigravity.create_antigravity(
                root, units, _binding(), _process()
            )
            store.commit_result_antigravity(0, _result(units[0].policy, units[0].dataset))
            (root / "records_antigravity" / "0000_record_antigravity.json").unlink()
            with self.assertRaisesRegex(ValueError, "inventory"):
                Exp443CheckpointStoreAntigravity.open_for_resume_antigravity(
                    root, _binding(), lambda _: False
                )

        with tempfile.TemporaryDirectory(prefix="exp443_task7_checkpoint_antigravity_") as directory:
            root = Path(directory) / "checkpoint_antigravity"
            store = Exp443CheckpointStoreAntigravity.create_antigravity(
                root, units, _binding(), _process()
            )
            store.commit_result_antigravity(0, _result(units[0].policy, units[0].dataset))
            state_path = root / "state_antigravity.json"
            state = json.loads(state_path.read_text())
            state["completed_count"] = 2
            state_path.write_bytes(canonical_json_bytes_antigravity(state))
            with self.assertRaisesRegex(ValueError, "prefix"):
                Exp443CheckpointStoreAntigravity.open_for_resume_antigravity(
                    root, _binding(), lambda _: False
                )

        with tempfile.TemporaryDirectory(prefix="exp443_task7_checkpoint_antigravity_") as directory:
            root = Path(directory) / "checkpoint_antigravity"
            Exp443CheckpointStoreAntigravity.create_antigravity(
                root, units, _binding(), _process()
            )
            state_path = root / "state_antigravity.json"
            state = json.loads(state_path.read_text())
            state["updated_at"] = "broken-heartbeat"
            state_path.write_bytes(canonical_json_bytes_antigravity(state))
            with self.assertRaisesRegex(ValueError, "heartbeat"):
                Exp443CheckpointStoreAntigravity.open_for_resume_antigravity(
                    root, _binding(), lambda _: False
                )

    def test_binding_change_completed_attempt_and_interruption_are_gated(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.checkpoint_antigravity import (
            Exp443CheckpointStoreAntigravity,
        )

        units = _units()
        with tempfile.TemporaryDirectory(prefix="exp443_task7_checkpoint_antigravity_") as directory:
            root = Path(directory) / "checkpoint_antigravity"
            store = Exp443CheckpointStoreAntigravity.create_antigravity(
                root, units, _binding(), _process()
            )
            store.commit_result_antigravity(0, _result(units[0].policy, units[0].dataset))
            store.mark_interrupted_resumable_antigravity()
            state = json.loads((root / "state_antigravity.json").read_text())
            self.assertEqual(state["status"], "interrupted_resumable")
            resumed = Exp443CheckpointStoreAntigravity.open_for_resume_antigravity(
                root, _binding(), lambda _: False
            )
            with self.assertRaisesRegex(ValueError, "binding"):
                Exp443CheckpointStoreAntigravity.open_for_resume_antigravity(
                    root,
                    _binding().__class__(
                        run_identity="changed",
                        scientific_config_sha256=_binding().scientific_config_sha256,
                        environment_lock_sha256=_binding().environment_lock_sha256,
                        source_snapshot_sha256=_binding().source_snapshot_sha256,
                        database_sha256=_binding().database_sha256,
                        reviewed_commit=_binding().reviewed_commit,
                    ),
                    lambda _: False,
                )
            for unit in units[1:]:
                resumed.commit_result_antigravity(unit.sequence, _result(unit.policy, unit.dataset))
            with self.assertRaisesRegex(ValueError, "complete"):
                Exp443CheckpointStoreAntigravity.open_for_resume_antigravity(
                    root, _binding(), lambda _: False
                )


if __name__ == "__main__":
    unittest.main()
