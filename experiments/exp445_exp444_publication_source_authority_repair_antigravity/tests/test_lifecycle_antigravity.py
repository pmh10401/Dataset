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
            profile = lifecycle_antigravity.EXP445_PROFILE_ANTIGRAVITY
            patched_profile = self._test_profile_antigravity(profile, root)
            def complete(active, *, commit, progress, **kwargs):
                scheduler_arguments.update(kwargs)
                for index, item in enumerate(active):
                    progress(stage="active", sequence=index)
                    commit(index, item, _result_antigravity(item))
                return tuple(_result_antigravity(item) for item in active)
            with (
                mock.patch.object(lifecycle_antigravity, "EXP445_PROFILE_ANTIGRAVITY", patched_profile),
                mock.patch.object(runtime, "EXP445_PROFILE_ANTIGRAVITY", patched_profile),
                mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity.EXP445_PROFILE_ANTIGRAVITY", patched_profile),
                mock.patch.object(runtime, "current_clean_head_antigravity", return_value=_COMMIT_ANTIGRAVITY),
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

    def test_nonfinal_wrapper_mismatch_writes_only_exact_next_error_record_antigravity(self) -> None:
        """Writing a successful mismatching result before rejecting it would invalidate failure evidence."""
        from experiments.exp445_exp444_publication_source_authority_repair_antigravity import lifecycle_antigravity
        from experiments.exp444_train_only_stability_precision_gate_antigravity import runtime_antigravity as runtime

        inputs = _inputs_antigravity(); authority = self._authority_antigravity(inputs)
        wrong = list(authority.record_sha256_prefix); wrong[7] = "0" * 64
        authority = type(authority)(authority.reviewed_commit, authority.attempt_root, tuple(wrong), authority.record_prefix_payload_sha256)
        with tempfile.TemporaryDirectory(prefix="exp445_nonfinal_antigravity_") as directory:
            root = Path(directory) / "exp445_exp444_publication_source_authority_repair_antigravity_20260810"; profile = lifecycle_antigravity.EXP445_PROFILE_ANTIGRAVITY
            patched_profile = self._test_profile_antigravity(profile, root)
            def complete(active, *, commit, progress, **_kwargs):
                for index, item in enumerate(active): progress(stage="active", sequence=index); commit(index, item, _result_antigravity(item))
            with (
                mock.patch.object(lifecycle_antigravity, "EXP445_PROFILE_ANTIGRAVITY", patched_profile), mock.patch.object(runtime, "EXP445_PROFILE_ANTIGRAVITY", patched_profile), mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity.EXP445_PROFILE_ANTIGRAVITY", patched_profile), mock.patch.object(runtime, "current_clean_head_antigravity", return_value=_COMMIT_ANTIGRAVITY), mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(99102, "exp445-test")), mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)), mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=complete), mock.patch.object(runtime, "_verify_exp445_predecessor_antigravity", return_value=authority), mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.provenance_antigravity.verify_exp444_provenance_antigravity"),
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
            root = Path(directory) / "exp445_exp444_publication_source_authority_repair_antigravity_20260810"; profile = lifecycle_antigravity.EXP445_PROFILE_ANTIGRAVITY
            patched_profile = self._test_profile_antigravity(profile, root)
            def complete(active, *, commit, progress, **_kwargs):
                for index, item in enumerate(active): progress(stage="active", sequence=index); commit(index, item, _result_antigravity(item))
            with (
                mock.patch.object(lifecycle_antigravity, "EXP445_PROFILE_ANTIGRAVITY", patched_profile), mock.patch.object(runtime, "EXP445_PROFILE_ANTIGRAVITY", patched_profile), mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity.EXP445_PROFILE_ANTIGRAVITY", patched_profile), mock.patch.object(runtime, "current_clean_head_antigravity", return_value=_COMMIT_ANTIGRAVITY), mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(99103, "exp445-test")), mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)), mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=complete), mock.patch.object(runtime, "_verify_exp445_predecessor_antigravity", return_value=authority), mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.provenance_antigravity.verify_exp444_provenance_antigravity"),
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


if __name__ == "__main__":
    unittest.main()
