"""TDD coverage for Exp444's immutable label-free publication boundary."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments.exp444_train_only_stability_precision_gate_antigravity import reporting_antigravity as reporting
from experiments.exp444_train_only_stability_precision_gate_antigravity import runtime_antigravity as runtime
from experiments.exp444_train_only_stability_precision_gate_antigravity.checkpoint_antigravity import ProcessIdentityAntigravity
from experiments.exp444_train_only_stability_precision_gate_antigravity.contracts_antigravity import (
    CandidateInputAntigravity,
    CandidateStabilityResultAntigravity,
    DepthViewScoreAntigravity,
    OmissionMaskAntigravity,
    SourceCandidateAntigravity,
    StabilityViewAntigravity,
)
from experiments.exp440_train_only_low_frequency_precision_gate_antigravity.contracts_antigravity import (
    DepthScoreAntigravity as Exp440DepthScoreAntigravity,
    ProvisionalCandidateAntigravity,
)


class Exp444ReportingAntigravityTests(unittest.TestCase):
    def test_publication_module_exposes_the_frozen_review_interfaces_antigravity(self) -> None:
        """Removing the Task 8 boundary must fail before a label-capable stage exists."""
        from experiments.exp444_train_only_stability_precision_gate_antigravity.reporting_antigravity import (  # noqa: PLC0415
            FrozenPublicationAntigravity,
            inspect_publication_antigravity,
            publish_prediction_antigravity,
        )

        self.assertTrue(callable(publish_prediction_antigravity))
        self.assertTrue(callable(inspect_publication_antigravity))
        self.assertTrue(hasattr(FrozenPublicationAntigravity, "stable_count"))

    @staticmethod
    def _inputs_antigravity() -> tuple[CandidateInputAntigravity, ...]:
        rows: list[CandidateInputAntigravity] = []
        for index in range(30):
            candidate = SourceCandidateAntigravity(f"Dataset_{index:02d}", index, 8, (1, 2), index + 1)
            rows.append(CandidateInputAntigravity(candidate, (0, 1, 2, 3), np.arange(32, dtype="<f4").reshape(4, 8), np.arange(8, dtype="<f4")))
        return tuple(rows)

    @staticmethod
    def _result_antigravity(item: CandidateInputAntigravity) -> CandidateStabilityResultAntigravity:
        stable = item.candidate.original_rank % 2 == 1
        score = (
            DepthViewScoreAntigravity(1, 2.0 if stable else 0.5, 1.0, stable, 2.0 if stable else 0.5, 0.5),
            DepthViewScoreAntigravity(2, 2.0 if stable else 0.5, 1.0, stable, 2.0 if stable else 0.5, 0.5),
        )
        view = StabilityViewAntigravity(
            OmissionMaskAntigravity(0, (0,), hashlib.sha256(str(item.candidate.original_rank).encode()).hexdigest()),
            (1, 2, 3), score, (1, 2) if stable else (), stable,
        )
        return CandidateStabilityResultAntigravity(item.candidate, (view,), int(stable), 1, "stable_review" if stable else "unstable_review")

    def _complete_attempt_antigravity(self, root: Path) -> tuple[Path, tuple[CandidateInputAntigravity, ...]]:
        inputs = self._inputs_antigravity()

        def completed_map(active, *, commit, progress, **_kwargs):
            values = tuple(self._result_antigravity(item) for item in active)
            for sequence, (item, result) in enumerate(zip(active, values, strict=True)):
                progress(stage="active", sequence=sequence)
                commit(sequence, item, result)
            return values

        with (
            mock.patch.object(runtime, "current_process_identity_antigravity", return_value=ProcessIdentityAntigravity(9_900_001, "reporting-A")),
            mock.patch.object(runtime, "_load_source_inputs_antigravity", return_value=(inputs, 0)),
            mock.patch.object(runtime, "ordered_parallel_map_antigravity", side_effect=completed_map),
        ):
            outcome = runtime.run_new_antigravity("reviewed", run_root=root)
        self.assertEqual((outcome.status, outcome.completed_count), ("complete", 30))
        return root / "attempts_antigravity/attempt_0000_antigravity", inputs

    def _published_antigravity(self, directory: str):
        attempt, inputs = self._complete_attempt_antigravity(Path(directory) / "run_antigravity")
        authority = reporting._PublishSourceAuthorityAntigravity(tuple(item.candidate for item in inputs), tuple({"dataset": f"Summary_{index:04d}"} for index in range(1117)), {"database": "a" * 64}, self._full_controls_antigravity(inputs))
        with mock.patch.object(reporting, "_publish_source_authority_antigravity", return_value=authority):
            return reporting.publish_prediction_antigravity(attempt)

    @staticmethod
    def _full_controls_antigravity(inputs: tuple[CandidateInputAntigravity, ...]) -> tuple[dict[str, object], ...]:
        rows = []
        for item in inputs:
            scores = (Exp440DepthScoreAntigravity(1, 2.0, 1.0, True, 2.0, 0.5), Exp440DepthScoreAntigravity(2, 2.0, 1.0, True, 2.0, 0.5))
            candidate = ProvisionalCandidateAntigravity(item.candidate.dataset, item.candidate.instance_index, item.candidate.raw_length, (1, 2), 2, 2.0, 0.5, scores)
            rows.append({"source": item.candidate.to_payload(), "depth_scores": [score.to_payload() for score in scores], "replayed_depths": [1, 2], "replayed_candidate": candidate.to_payload(), "matches_source": True, "replayed_rank": item.candidate.original_rank})
        return tuple(rows)

    @staticmethod
    def _regenerate_publication_hashes_antigravity(frozen, leaf_name: str, leaf_payload: bytes) -> None:
        payloads = {name: (frozen.publication_root / name).read_bytes() for name in reporting._LEAF_FILES_ANTIGRAVITY}
        payloads[leaf_name] = leaf_payload
        expected = reporting._final_payloads_antigravity(payloads, frozen.selection)
        for name, value in expected.items():
            (frozen.publication_root / name).write_bytes(value)

    def test_publish_and_read_only_inspect_conserve_every_source_coordinate_antigravity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp444_reporting_antigravity_") as directory:
            attempt, inputs = self._complete_attempt_antigravity(Path(directory) / "run_antigravity")
            authority = reporting._PublishSourceAuthorityAntigravity(
                tuple(item.candidate for item in inputs),
                tuple({"dataset": f"Summary_{index:04d}", "candidate_count": 0} for index in range(1117)),
                {"database": "a" * 64, "selection": "b" * 64},
                self._full_controls_antigravity(inputs),
            )
            with mock.patch.object(reporting, "_publish_source_authority_antigravity", return_value=authority):
                frozen = reporting.publish_prediction_antigravity(attempt)
                repeated = reporting.publish_prediction_antigravity(attempt)
            publication = attempt / reporting.PUBLICATION_DIRECTORY_ANTIGRAVITY
            before = {path.name: path.read_bytes() for path in publication.iterdir()}
            inspected = reporting.inspect_publication_antigravity(publication)
            after = {path.name: path.read_bytes() for path in publication.iterdir()}
            self.assertEqual(before, after)
            self.assertEqual((frozen.stable_count, frozen.unstable_count), (15, 15))
            self.assertEqual(repeated.prediction_sha256, frozen.prediction_sha256)
            self.assertTrue(frozen.stable_coordinates.isdisjoint(frozen.unstable_coordinates))
            self.assertEqual(frozen.stable_coordinates | frozen.unstable_coordinates, frozen.source_coordinates)
            self.assertEqual((inspected.stable_count + inspected.unstable_count, inspected.label_access_count), (30, 0))
            self.assertEqual(set(before), set(reporting._PUBLICATION_FILES_ANTIGRAVITY))
            artifact_map = __import__("json").loads(before[reporting.ARTIFACT_SHA256_FILE_ANTIGRAVITY])
            self.assertNotIn(reporting.ARTIFACT_SHA256_FILE_ANTIGRAVITY, artifact_map["files"])

    def test_inspector_detects_tampering_without_opening_publish_source_antigravity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp444_reporting_tamper_antigravity_") as directory:
            attempt, inputs = self._complete_attempt_antigravity(Path(directory) / "run_antigravity")
            authority = reporting._PublishSourceAuthorityAntigravity(
                tuple(item.candidate for item in inputs),
                tuple({"dataset": f"Summary_{index:04d}"} for index in range(1117)),
                {"database": "a" * 64},
                self._full_controls_antigravity(inputs),
            )
            with mock.patch.object(reporting, "_publish_source_authority_antigravity", return_value=authority):
                frozen = reporting.publish_prediction_antigravity(attempt)
            leaf = frozen.publication_root / reporting.STABLE_REVIEW_FILE_ANTIGRAVITY
            leaf.write_bytes(leaf.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "canonical|hash|contract"):
                reporting.inspect_publication_antigravity(frozen.publication_root)

    def test_runtime_leaf_captures_every_terminal_authority_surface_antigravity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp444_reporting_runtime_antigravity_") as directory:
            attempt, inputs = self._complete_attempt_antigravity(Path(directory) / "run_antigravity")
            authority = reporting._PublishSourceAuthorityAntigravity(tuple(item.candidate for item in inputs), tuple({"dataset": f"Summary_{index:04d}"} for index in range(1117)), {"database": "a" * 64}, self._full_controls_antigravity(inputs))
            with mock.patch.object(reporting, "_publish_source_authority_antigravity", return_value=authority):
                frozen = reporting.publish_prediction_antigravity(attempt)
            runtime_leaf = __import__("json").loads((frozen.publication_root / reporting.RUNTIME_FILE_ANTIGRAVITY).read_text())
            self.assertEqual(set(runtime_leaf["authorities"]), {"context", "state", "progress", "pid", "events", "text"})

    def test_inspection_rejects_hardlinked_publication_leaf_antigravity(self) -> None:
        """A second name for a leaf must not become a mutable inspection authority."""
        with tempfile.TemporaryDirectory(prefix="exp444_reporting_hardlink_antigravity_") as directory:
            attempt, inputs = self._complete_attempt_antigravity(Path(directory) / "run_antigravity")
            authority = reporting._PublishSourceAuthorityAntigravity(tuple(item.candidate for item in inputs), tuple({"dataset": f"Summary_{index:04d}"} for index in range(1117)), {"database": "a" * 64}, self._full_controls_antigravity(inputs))
            with mock.patch.object(reporting, "_publish_source_authority_antigravity", return_value=authority):
                frozen = reporting.publish_prediction_antigravity(attempt)
            leaf = frozen.publication_root / reporting.RUNTIME_FILE_ANTIGRAVITY
            hardlink = frozen.publication_root.parent / "runtime_hardlink_antigravity.json"
            __import__("os").link(leaf, hardlink)
            with self.assertRaisesRegex(ValueError, "single-linked|hardlink|invalid"):
                reporting.inspect_publication_antigravity(frozen.publication_root)

    def test_existing_publication_reentry_does_not_open_authorities_antigravity(self) -> None:
        """A valid publication is inspected from its frozen leaves before any live authority."""
        with tempfile.TemporaryDirectory(prefix="exp444_reporting_reentry_antigravity_") as directory:
            attempt, inputs = self._complete_attempt_antigravity(Path(directory) / "run_antigravity")
            authority = reporting._PublishSourceAuthorityAntigravity(tuple(item.candidate for item in inputs), tuple({"dataset": f"Summary_{index:04d}"} for index in range(1117)), {"database": "a" * 64}, self._full_controls_antigravity(inputs))
            with mock.patch.object(reporting, "_publish_source_authority_antigravity", return_value=authority):
                reporting.publish_prediction_antigravity(attempt)
            with (
                mock.patch.object(reporting, "_publish_source_authority_antigravity", side_effect=AssertionError("source opened")),
                mock.patch.object(reporting, "_complete_checkpoint_results_antigravity", side_effect=AssertionError("checkpoint opened")),
                mock.patch("experiments.exp444_train_only_stability_precision_gate_antigravity.provenance_antigravity.verify_exp444_provenance_antigravity", side_effect=AssertionError("provenance opened")),
            ):
                self.assertEqual(reporting.publish_prediction_antigravity(attempt).stable_count, 15)

    def test_inspection_rejects_directory_entry_replacement_after_pin_antigravity(self) -> None:
        """Replacing a name after all leaf descriptors are pinned must invalidate inspection."""
        with tempfile.TemporaryDirectory(prefix="exp444_reporting_replace_antigravity_") as directory:
            attempt, inputs = self._complete_attempt_antigravity(Path(directory) / "run_antigravity")
            authority = reporting._PublishSourceAuthorityAntigravity(tuple(item.candidate for item in inputs), tuple({"dataset": f"Summary_{index:04d}"} for index in range(1117)), {"database": "a" * 64}, self._full_controls_antigravity(inputs))
            with mock.patch.object(reporting, "_publish_source_authority_antigravity", return_value=authority):
                frozen = reporting.publish_prediction_antigravity(attempt)
            def replace_leaf(_root: Path) -> None:
                leaf = frozen.publication_root / reporting.RUNTIME_FILE_ANTIGRAVITY
                replacement = leaf.with_name("runtime_replacement_antigravity.json")
                replacement.write_bytes(leaf.read_bytes())
                __import__("os").replace(replacement, leaf)
            with mock.patch.object(reporting, "_PINNED_PUBLICATION_HOOK_ANTIGRAVITY", replace_leaf, create=True):
                with self.assertRaisesRegex(ValueError, "entry changed|inventory changed"):
                    reporting.inspect_publication_antigravity(frozen.publication_root)

    def test_inspection_rejects_substituted_backend_wrapper_with_regenerated_hashes_antigravity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp444_reporting_backend_antigravity_") as directory:
            frozen = self._published_antigravity(directory)
            payload = __import__("json").loads((frozen.publication_root / reporting.BACKEND_DECISION_FILE_ANTIGRAVITY).read_text())
            payload["authority"]["task5_log_utf8"] += "substitution\n"
            self._regenerate_publication_hashes_antigravity(frozen, reporting.BACKEND_DECISION_FILE_ANTIGRAVITY, reporting.canonical_json_bytes_antigravity(payload))
            with self.assertRaisesRegex(ValueError, "backend authority"):
                reporting.inspect_publication_antigravity(frozen.publication_root)

    def test_inspection_rejects_rewritten_runtime_surface_with_regenerated_hashes_antigravity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp444_reporting_runtime_rewrite_antigravity_") as directory:
            frozen = self._published_antigravity(directory)
            payload = __import__("json").loads((frozen.publication_root / reporting.RUNTIME_FILE_ANTIGRAVITY).read_text())
            surface = payload["authorities"]["text"]
            surface["utf8"] = "종료 stage=interrupted\n"
            raw = surface["utf8"].encode()
            surface["byte_size"] = len(raw)
            surface["sha256"] = hashlib.sha256(raw).hexdigest()
            self._regenerate_publication_hashes_antigravity(frozen, reporting.RUNTIME_FILE_ANTIGRAVITY, reporting.canonical_json_bytes_antigravity(payload))
            with self.assertRaisesRegex(ValueError, "terminal text"):
                reporting.inspect_publication_antigravity(frozen.publication_root)

    def test_inspection_rejects_substituted_control_with_regenerated_hashes_antigravity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp444_reporting_control_rewrite_antigravity_") as directory:
            frozen = self._published_antigravity(directory)
            payload = __import__("json").loads((frozen.publication_root / reporting.CONTROL_REPLAY_FILE_ANTIGRAVITY).read_text())
            payload["rows"][0]["original_rank"] = 30
            self._regenerate_publication_hashes_antigravity(frozen, reporting.CONTROL_REPLAY_FILE_ANTIGRAVITY, reporting.canonical_json_bytes_antigravity(payload))
            with self.assertRaisesRegex(ValueError, "original-control"):
                reporting.inspect_publication_antigravity(frozen.publication_root)

    def test_inspection_rejects_top_level_control_score_contradiction_antigravity(self) -> None:
        """Top-level replay evidence cannot disagree with the nested Exp440 candidate."""
        with tempfile.TemporaryDirectory(prefix="exp444_reporting_control_score_antigravity_") as directory:
            frozen = self._published_antigravity(directory)
            payload = __import__("json").loads((frozen.publication_root / reporting.CONTROL_REPLAY_FILE_ANTIGRAVITY).read_text())
            payload["rows"][0]["depth_scores"][0]["score"] = 100.0
            payload["rows"][0]["depth_scores"][0]["excess_ratio"] = 100.0
            self._regenerate_publication_hashes_antigravity(frozen, reporting.CONTROL_REPLAY_FILE_ANTIGRAVITY, reporting.canonical_json_bytes_antigravity(payload))
            with self.assertRaisesRegex(ValueError, "original-control"):
                reporting.inspect_publication_antigravity(frozen.publication_root)

    def test_attempt_capture_rejects_hardlinked_progress_antigravity(self) -> None:
        """A progress authority with another namespace link cannot be captured."""
        with tempfile.TemporaryDirectory(prefix="exp444_attempt_hardlink_antigravity_") as directory:
            attempt, _ = self._complete_attempt_antigravity(Path(directory) / "run_antigravity")
            __import__("os").link(attempt / "progress_antigravity.json", attempt / "progress_link_antigravity.json")
            with self.assertRaisesRegex(ValueError, "single-linked|attempt"):
                reporting._complete_checkpoint_results_antigravity(attempt)

    def test_attempt_capture_rejects_post_open_state_replacement_antigravity(self) -> None:
        """A state pathname replacement after descriptor capture is never accepted."""
        with tempfile.TemporaryDirectory(prefix="exp444_attempt_state_replace_antigravity_") as directory:
            attempt, _ = self._complete_attempt_antigravity(Path(directory) / "run_antigravity")
            def replace_state(root: Path) -> None:
                state = root / "checkpoint_antigravity/state_antigravity.json"
                replacement = state.with_name("state_replacement_antigravity.json")
                replacement.write_bytes(state.read_bytes())
                __import__("os").replace(replacement, state)
            with mock.patch.object(reporting, "_ATTEMPT_CAPTURE_HOOK_ANTIGRAVITY", replace_state):
                with self.assertRaisesRegex(ValueError, "entry changed"):
                    reporting._complete_checkpoint_results_antigravity(attempt)

    def test_attempt_capture_rejects_post_open_record_replacement_antigravity(self) -> None:
        """A record pathname replacement after descriptor capture is never accepted."""
        with tempfile.TemporaryDirectory(prefix="exp444_attempt_record_replace_antigravity_") as directory:
            attempt, _ = self._complete_attempt_antigravity(Path(directory) / "run_antigravity")
            def replace_record(root: Path) -> None:
                record = root / "checkpoint_antigravity/records_antigravity/0000_record_antigravity.json"
                replacement = record.with_name("record_replacement_antigravity.json")
                replacement.write_bytes(record.read_bytes())
                __import__("os").replace(replacement, record)
            with mock.patch.object(reporting, "_ATTEMPT_CAPTURE_HOOK_ANTIGRAVITY", replace_record):
                with self.assertRaisesRegex(ValueError, "entry changed"):
                    reporting._complete_checkpoint_results_antigravity(attempt)

    def test_invalid_terminal_context_never_creates_a_publication_antigravity(self) -> None:
        """Terminal validation must finish before the create-only destination becomes visible."""
        with tempfile.TemporaryDirectory(prefix="exp444_reporting_terminal_preflight_antigravity_") as directory:
            attempt, inputs = self._complete_attempt_antigravity(Path(directory) / "run_antigravity")
            state_path = attempt / "checkpoint_antigravity/state_antigravity.json"
            state = __import__("json").loads(state_path.read_text())
            state["context_sha256"] = "0" * 64
            state_path.write_bytes(reporting.canonical_json_bytes_antigravity(state))
            authority = reporting._PublishSourceAuthorityAntigravity(tuple(item.candidate for item in inputs), tuple({"dataset": f"Summary_{index:04d}"} for index in range(1117)), {"database": "a" * 64}, self._full_controls_antigravity(inputs))
            with mock.patch.object(reporting, "_publish_source_authority_antigravity", return_value=authority):
                with self.assertRaisesRegex(ValueError, "terminal checkpoint state"):
                    reporting.publish_prediction_antigravity(attempt)
                self.assertFalse((attempt / reporting.PUBLICATION_DIRECTORY_ANTIGRAVITY).exists())
                state["context_sha256"] = hashlib.sha256((attempt / "checkpoint_antigravity/context_antigravity.json").read_bytes()).hexdigest()
                state_path.write_bytes(reporting.canonical_json_bytes_antigravity(state))
                self.assertEqual(reporting.publish_prediction_antigravity(attempt).stable_count, 15)

    def test_invalid_terminal_process_binding_never_creates_a_publication_antigravity(self) -> None:
        """A detached state PID must fail before a create-only publication is visible."""
        with tempfile.TemporaryDirectory(prefix="exp444_reporting_terminal_process_antigravity_") as directory:
            attempt, inputs = self._complete_attempt_antigravity(Path(directory) / "run_antigravity")
            state_path = attempt / "checkpoint_antigravity/state_antigravity.json"
            state = __import__("json").loads(state_path.read_text())
            state["process"]["pid"] = 9_900_002
            state_path.write_bytes(reporting.canonical_json_bytes_antigravity(state))
            authority = reporting._PublishSourceAuthorityAntigravity(tuple(item.candidate for item in inputs), tuple({"dataset": f"Summary_{index:04d}"} for index in range(1117)), {"database": "a" * 64}, self._full_controls_antigravity(inputs))
            with mock.patch.object(reporting, "_publish_source_authority_antigravity", return_value=authority):
                with self.assertRaisesRegex(ValueError, "terminal process identity"):
                    reporting.publish_prediction_antigravity(attempt)
            self.assertFalse((attempt / reporting.PUBLICATION_DIRECTORY_ANTIGRAVITY).exists())

    def test_attempt_directory_replacement_after_capture_receives_no_publication_antigravity(self) -> None:
        """Publication must not move captured checkpoint evidence under a replacement attempt path."""
        with tempfile.TemporaryDirectory(prefix="exp444_reporting_attempt_replace_antigravity_") as directory:
            attempt, inputs = self._complete_attempt_antigravity(Path(directory) / "run_antigravity")
            moved = attempt.with_name("captured_attempt_authority_antigravity")
            authority = reporting._PublishSourceAuthorityAntigravity(tuple(item.candidate for item in inputs), tuple({"dataset": f"Summary_{index:04d}"} for index in range(1117)), {"database": "a" * 64}, self._full_controls_antigravity(inputs))

            def replace_attempt(root: Path) -> None:
                __import__("os").replace(root, moved)
                root.mkdir(mode=0o700)

            with (
                mock.patch.object(reporting, "_publish_source_authority_antigravity", return_value=authority),
                mock.patch.object(reporting, "_ATTEMPT_CAPTURE_HOOK_ANTIGRAVITY", replace_attempt),
            ):
                with self.assertRaisesRegex(ValueError, "attempt.*changed|attempt.*namespace|authority"):
                    reporting.publish_prediction_antigravity(attempt)
            self.assertFalse((attempt / reporting.PUBLICATION_DIRECTORY_ANTIGRAVITY).exists())
            self.assertEqual(len(reporting._complete_checkpoint_results_antigravity(moved)), 30)

    def test_staging_rejects_entry_replacement_before_rename_antigravity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp444_reporting_staging_entry_antigravity_") as directory:
            frozen = self._published_antigravity(directory)
            payloads = {name: (frozen.publication_root / name).read_bytes() for name in reporting._PUBLICATION_FILES_ANTIGRAVITY}
            destination = frozen.publication_root.parent / "replacement_publication_antigravity"
            def replace_entry(staging: Path) -> None:
                leaf = staging / reporting.RUNTIME_FILE_ANTIGRAVITY
                replacement = staging / "runtime_replacement_antigravity.json"
                replacement.write_bytes(leaf.read_bytes())
                __import__("os").replace(replacement, leaf)
            with mock.patch.object(reporting, "_STAGING_BEFORE_RENAME_HOOK_ANTIGRAVITY", replace_entry, create=True):
                with self.assertRaisesRegex(ValueError, "staging|changed"):
                    reporting._write_staging_antigravity(destination, payloads)
            self.assertFalse(destination.exists())

    def test_staging_rejects_directory_replacement_before_rename_antigravity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp444_reporting_staging_dir_antigravity_") as directory:
            frozen = self._published_antigravity(directory)
            payloads = {name: (frozen.publication_root / name).read_bytes() for name in reporting._PUBLICATION_FILES_ANTIGRAVITY}
            destination = frozen.publication_root.parent / "replacement_publication_antigravity"
            def replace_directory(staging: Path) -> None:
                moved = staging.with_name("moved_staging_antigravity")
                __import__("os").replace(staging, moved)
                staging.mkdir(mode=0o700)
            with mock.patch.object(reporting, "_STAGING_BEFORE_RENAME_HOOK_ANTIGRAVITY", replace_directory, create=True):
                with self.assertRaisesRegex(ValueError, "staging|changed|inventory"):
                    reporting._write_staging_antigravity(destination, payloads)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
