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

    def test_publish_and_read_only_inspect_conserve_every_source_coordinate_antigravity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exp444_reporting_antigravity_") as directory:
            attempt, inputs = self._complete_attempt_antigravity(Path(directory) / "run_antigravity")
            authority = reporting._PublishSourceAuthorityAntigravity(
                tuple(item.candidate for item in inputs),
                tuple({"dataset": f"Summary_{index:04d}", "candidate_count": 0} for index in range(1117)),
                {"database": "a" * 64, "selection": "b" * 64},
                tuple({"candidate": item.candidate.to_payload(), "original_depths": list(item.candidate.original_depths), "original_rank": item.candidate.original_rank} for item in inputs),
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
                tuple({"candidate": item.candidate.to_payload(), "original_depths": list(item.candidate.original_depths), "original_rank": item.candidate.original_rank} for item in inputs),
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
            authority = reporting._PublishSourceAuthorityAntigravity(tuple(item.candidate for item in inputs), tuple({"dataset": f"Summary_{index:04d}"} for index in range(1117)), {"database": "a" * 64}, tuple({"candidate": item.candidate.to_payload(), "original_depths": list(item.candidate.original_depths), "original_rank": item.candidate.original_rank} for item in inputs))
            with mock.patch.object(reporting, "_publish_source_authority_antigravity", return_value=authority):
                frozen = reporting.publish_prediction_antigravity(attempt)
            runtime_leaf = __import__("json").loads((frozen.publication_root / reporting.RUNTIME_FILE_ANTIGRAVITY).read_text())
            self.assertEqual(set(runtime_leaf["authorities"]), {"context", "state", "progress", "pid", "events", "text"})


if __name__ == "__main__":
    unittest.main()
