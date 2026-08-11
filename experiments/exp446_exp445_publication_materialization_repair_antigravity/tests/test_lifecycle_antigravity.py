"""Single-use Exp446 replay boundaries on deterministic values-only fixtures."""
from __future__ import annotations

import hashlib
import json
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
                for index, item in enumerate(active):
                    progress(stage="active", sequence=index)
                    commit(index, item, _result_antigravity(item))
                return tuple(_result_antigravity(item) for item in active)
            with (
                mock.patch.object(lifecycle_antigravity, "_EXP446_PROFILE_ANTIGRAVITY", profile),
                mock.patch.object(runtime, "EXP446_PROFILE_ANTIGRAVITY", profile),
                mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.readiness_antigravity.EXP446_PROFILE_ANTIGRAVITY", profile),
                mock.patch.object(runtime, "readiness_for_profile_antigravity"),
                mock.patch.object(runtime, "_validate_reviewed_commit_antigravity", return_value=_COMMIT_ANTIGRAVITY),
                mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(99446, "exp446-test")),
                mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, labels)),
                mock.patch.object(runtime, "production_worker_count_antigravity", return_value=workers),
                mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=ordered),
                mock.patch.object(runtime, "_verify_exp445_predecessor_antigravity", return_value=exp444),
                mock.patch.object(runtime, "_verify_exp446_predecessor_antigravity", side_effect=verify_exp445),
                mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.provenance_antigravity.verify_exp444_provenance_antigravity"),
            ):
                try:
                    outcome = lifecycle_antigravity.run_new_antigravity(_COMMIT_ANTIGRAVITY)
                except BaseException as error:
                    outcome = error
            return root, outcome

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


if __name__ == "__main__":
    unittest.main()
