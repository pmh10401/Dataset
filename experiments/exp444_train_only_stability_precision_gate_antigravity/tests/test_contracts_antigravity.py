"""RED/GREEN tests for immutable Exp444 contracts."""
from __future__ import annotations

import unittest

from experiments.exp444_train_only_stability_precision_gate_antigravity.config_antigravity import FROZEN_CONFIG, scientific_config_sha256_antigravity
from experiments.exp444_train_only_stability_precision_gate_antigravity.contracts_antigravity import (
    BackendDecisionAntigravity,
    CandidateStabilityResultAntigravity,
    CheckpointContextAntigravity,
    CheckpointStateAntigravity,
    DepthViewScoreAntigravity,
    EvaluationSummaryAntigravity,
    OmissionMaskAntigravity,
    SourceCandidateAntigravity,
    StabilitySelectionAntigravity,
    StabilityViewAntigravity,
    validate_prediction_payload_antigravity,
)


_DIGEST = "a" * 64


def _source(dataset: str = "A", index: int = 0, rank: int = 1) -> SourceCandidateAntigravity:
    return SourceCandidateAntigravity(dataset, index, 32, (2, 3), rank)


def _view(candidate: SourceCandidateAntigravity, view_index: int = 0, survived: bool = True) -> StabilityViewAntigravity:
    score_a = DepthViewScoreAntigravity(2, 1.5, 1.0, True, 1.5, 0.99)
    score_b = DepthViewScoreAntigravity(3, 1.25, 1.0 if survived else 2.0, survived, 1.25 if survived else 0.625, 0.98)
    mask = OmissionMaskAntigravity(view_index, (view_index,), _DIGEST)
    return StabilityViewAntigravity(mask, (10, 11, 12, 13), (score_a, score_b), (2, 3), survived)


class Exp444ContractsAntigravityTests(unittest.TestCase):
    def test_source_and_mask_round_trip_and_exact_types(self) -> None:
        source = _source()
        self.assertEqual(SourceCandidateAntigravity.from_payload(source.to_payload()), source)
        mask = OmissionMaskAntigravity(0, (1, 2), _DIGEST)
        self.assertEqual(OmissionMaskAntigravity.from_payload(mask.to_payload()), mask)
        with self.assertRaises(ValueError):
            SourceCandidateAntigravity("A", True, 32, (2, 3), 1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            OmissionMaskAntigravity(0, (2, 1), _DIGEST)

    def test_depth_score_rejects_nonfinite_and_round_trips(self) -> None:
        score = DepthViewScoreAntigravity(2, 1.5, 1.0, True, 1.5, 0.99)
        self.assertEqual(DepthViewScoreAntigravity.from_payload(score.to_payload()), score)
        with self.assertRaises(ValueError):
            DepthViewScoreAntigravity(2, float("nan"), 1.0, True, 1.5, 0.99)
        with self.assertRaises(ValueError):
            DepthViewScoreAntigravity(2, 1.5, 1.0, False, 1.5, 0.99)

    def test_stability_result_round_trip_and_lane_invariant(self) -> None:
        source = _source()
        views = tuple(_view(source, i, i < 7) for i in range(8))
        result = CandidateStabilityResultAntigravity(source, views, 7, 7, "stable_review")
        self.assertEqual(result.source_count, 1)
        self.assertEqual(CandidateStabilityResultAntigravity.from_payload(result.to_payload()), result)
        with self.assertRaises(ValueError):
            CandidateStabilityResultAntigravity(source, views, 7, 7, "stable")
        with self.assertRaises(ValueError):
            CandidateStabilityResultAntigravity(source, views, 6, 7, "stable_review")

    def test_selection_conserves_two_review_lanes_and_rejects_duplicate_coordinates(self) -> None:
        source = (_source("A", rank=1), _source("B", rank=2))
        a = CandidateStabilityResultAntigravity(source[0], tuple(_view(source[0], i) for i in range(8)), 8, 7, "stable_review")
        b = CandidateStabilityResultAntigravity(source[1], tuple(_view(source[1], i, False) for i in range(8)), 0, 7, "unstable_review")
        result = StabilitySelectionAntigravity(source=source, stable=(a,), unstable=(b,))
        self.assertEqual(result.source_count, result.stable_count + result.unstable_count)
        self.assertEqual(result.source_count, 2)
        self.assertEqual(StabilitySelectionAntigravity.from_payload(result.to_payload()), result)
        with self.assertRaises(ValueError):
            StabilitySelectionAntigravity(source=source, stable=(a,), unstable=(a,))

    def test_backend_checkpoint_and_evaluation_round_trip(self) -> None:
        backend = BackendDecisionAntigravity("cpu", 11, (), "MLX/Numba parity or speed gate not met")
        self.assertEqual(BackendDecisionAntigravity.from_payload(backend.to_payload()), backend)
        context = CheckpointContextAntigravity(
            FROZEN_CONFIG.experiment_id,
            FROZEN_CONFIG.run_identity,
            scientific_config_sha256_antigravity(),
            FROZEN_CONFIG.environment_lock_sha256,
            30,
            11,
            1,
        )
        self.assertEqual(CheckpointContextAntigravity.from_payload(context.to_payload()), context)
        state = CheckpointStateAntigravity("running", 2, 30, 0, "A", None, _DIGEST)
        self.assertEqual(CheckpointStateAntigravity.from_payload(state.to_payload()), state)
        summary = EvaluationSummaryAntigravity("descriptive_complete_no_promotion", 30, 1, 29, 0, 0, 0.0, 0.0, _DIGEST)
        self.assertEqual(EvaluationSummaryAntigravity.from_payload(summary.to_payload()), summary)
        with self.assertRaises(ValueError):
            CheckpointStateAntigravity("running", 2.0, 30, 0, "A", None, _DIGEST)  # type: ignore[arg-type]

    def test_recursive_prediction_firewall_and_immutable_arrays(self) -> None:
        validate_prediction_payload_antigravity({"candidate": {"scores": [1.0, 2.0]}})
        for forbidden in ("label", "labels_blob", "evaluation", "precision", "f1", "anomaly_positions"):
            with self.subTest(forbidden=forbidden), self.assertRaises(ValueError):
                validate_prediction_payload_antigravity({"evidence": [{forbidden: 1}]})
        caller_depth_array = [2, 3]
        direct = SourceCandidateAntigravity("CBF", 4, 128, caller_depth_array, 1)  # type: ignore[arg-type]
        caller_depth_array.append(9)
        self.assertEqual(direct.original_depths, (2, 3))
        source_payload = _source().to_payload()
        original_depths = source_payload["original_depths"]
        restored = SourceCandidateAntigravity.from_payload(source_payload)
        original_depths.append(9)
        self.assertEqual(restored.original_depths, (2, 3))
        detached_payload = restored.to_payload()
        detached_payload["original_depths"].append(10)
        self.assertEqual(restored.original_depths, (2, 3))
        with self.assertRaises(AttributeError):
            restored.original_depths.append(11)  # type: ignore[attr-defined]

        view = _view(restored)
        view_payload = view.to_payload()
        restored_view = StabilityViewAntigravity.from_payload(view_payload)
        view_payload["retained_train_indices"].append(99)
        view_payload["depth_scores"][0]["score"] = 999.0
        self.assertEqual(restored_view.retained_train_indices, (10, 11, 12, 13))
        self.assertEqual(restored_view.depth_scores[0].score, 1.5)

    def test_every_contract_rejects_missing_and_extra_payload_keys(self) -> None:
        source = _source()
        views = tuple(_view(source, index) for index in range(2))
        result = CandidateStabilityResultAntigravity(source, views, 2, 2, "stable_review")
        selection = StabilitySelectionAntigravity((source,), (result,), ())
        backend = BackendDecisionAntigravity("cpu", 11, (), "fixture")
        context = CheckpointContextAntigravity(
            FROZEN_CONFIG.experiment_id, FROZEN_CONFIG.run_identity,
            scientific_config_sha256_antigravity(), FROZEN_CONFIG.environment_lock_sha256,
            30, 11, 1,
        )
        state = CheckpointStateAntigravity("running", 0, 30, 0, None, None, _DIGEST)
        summary = EvaluationSummaryAntigravity(
            "descriptive_complete_no_promotion", 30, 1, 29, 0, 0, 0.0, 0.0, _DIGEST,
        )
        values = (
            (source, SourceCandidateAntigravity),
            (views[0].mask, OmissionMaskAntigravity),
            (views[0].depth_scores[0], DepthViewScoreAntigravity),
            (views[0], StabilityViewAntigravity),
            (result, CandidateStabilityResultAntigravity),
            (selection, StabilitySelectionAntigravity),
            (backend, BackendDecisionAntigravity),
            (context, CheckpointContextAntigravity),
            (state, CheckpointStateAntigravity),
            (summary, EvaluationSummaryAntigravity),
        )
        for value, contract_type in values:
            payload = value.to_payload()
            missing = dict(payload)
            missing.pop(next(iter(missing)))
            extra = dict(payload)
            extra["extra_antigravity"] = None
            with self.subTest(contract=contract_type.__name__, mutation="missing"), self.assertRaises(ValueError):
                contract_type.from_payload(missing)
            with self.subTest(contract=contract_type.__name__, mutation="extra"), self.assertRaises(ValueError):
                contract_type.from_payload(extra)


if __name__ == "__main__":
    unittest.main()
