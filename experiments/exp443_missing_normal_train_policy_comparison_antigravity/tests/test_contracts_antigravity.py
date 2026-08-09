"""RED tests for strict Exp443 in-memory and serialized contracts."""
from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
import unittest

import numpy as np

from experiments.exp443_missing_normal_train_policy_comparison_antigravity.contracts_antigravity import (
    CheckpointContextAntigravity,
    CheckpointStateAntigravity,
    EvaluationEvidenceAntigravity,
    KernelSeedScoresAntigravity,
    PolicyDatasetResultAntigravity,
    PredictionRowAntigravity,
    RawSeriesAntigravity,
    RunPolicyAuditAntigravity,
    validate_prediction_payload_antigravity,
)


_DIGEST = "a" * 64


def _raw(split: str = "TRAIN", index: int = 0) -> RawSeriesAntigravity:
    values = np.asarray([1.0, np.nan, 3.0], dtype="<f4")
    return RawSeriesAntigravity("Demo", split, index, _DIGEST, values, (1,))


class Exp443ContractsAntigravityTests(unittest.TestCase):
    def test_raw_series_copies_array_read_only_and_round_trips_explicit_dtype_shape(self) -> None:
        source = np.asarray([1.0, np.nan, 3.0], dtype="<f4")
        item = RawSeriesAntigravity("Demo", "TRAIN", 0, _DIGEST, source, (1,))
        source[0] = 99.0
        self.assertEqual(item.values[0], 1.0)
        self.assertFalse(item.values.flags.writeable)
        payload = item.to_payload()
        self.assertEqual(payload["dtype"], "<f4")
        self.assertEqual(payload["shape"], [3])
        self.assertEqual(RawSeriesAntigravity.from_payload(payload), item)

    def test_raw_series_rejects_wrong_dtype_shape_missing_positions_and_digest(self) -> None:
        with self.assertRaises(ValueError):
            RawSeriesAntigravity("Demo", "TRAIN", 0, _DIGEST, np.asarray([1, 2, 3], dtype="<i4"), (1,))
        with self.assertRaises(ValueError):
            RawSeriesAntigravity("Demo", "TRAIN", 0, _DIGEST, np.asarray([[1, 2]], dtype="<f4"), (1,))
        with self.assertRaises(ValueError):
            RawSeriesAntigravity("Demo", "TRAIN", 0, _DIGEST, np.asarray([1, np.nan, 3], dtype="<f4"), ())
        with self.assertRaises(ValueError):
            RawSeriesAntigravity("Demo", "TRAIN", 0, "A" * 64, np.asarray([1, 2, 3], dtype="<f4"), (1,))

    def test_policy_audit_and_score_arrays_copy_round_trip_and_reject_nonfinite_scores(self) -> None:
        audit = RunPolicyAuditAntigravity("Demo", "exclude_nonfinite_row", 0, 3, (1,), _DIGEST, "excluded_nonfinite_row")
        self.assertEqual(RunPolicyAuditAntigravity.from_payload(audit.to_payload()), audit)
        oof = np.asarray([0.1, 0.2], dtype="<f8")
        test = np.asarray([0.3], dtype="<f8")
        score = KernelSeedScoresAntigravity(1024, 20260721, oof, test, (0,))
        oof[0] = 9.0
        self.assertEqual(score.oof_scores[0], 0.1)
        self.assertFalse(score.oof_scores.flags.writeable)
        self.assertEqual(KernelSeedScoresAntigravity.from_payload(score.to_payload()), score)
        with self.assertRaises(ValueError):
            KernelSeedScoresAntigravity(1024, 20260721, np.asarray([np.nan], dtype="<f8"), test, ())

    def test_policy_dataset_result_has_consistent_counts_and_detached_mapping(self) -> None:
        score = KernelSeedScoresAntigravity(1024, 20260721, np.asarray([0.1], dtype="<f8"), np.asarray([0.2], dtype="<f8"), (0,))
        mapping = {1024: (0,)}
        result = PolicyDatasetResultAntigravity(
            "linear_edge_nearest_control", "Demo", 3, 4, 1, 3, 1, 1,
            (score,), mapping, (), "scored",
        )
        mapping[1024] = (9,)
        self.assertEqual(result.candidates_by_kernel[1024], (0,))
        self.assertEqual(PolicyDatasetResultAntigravity.from_payload(result.to_payload()), result)
        with self.assertRaises(ValueError):
            replace(result, train_used=4, train_excluded=1)

    def test_prediction_recursively_rejects_label_and_evaluation_keys(self) -> None:
        row = PredictionRowAntigravity("Demo", "linear_edge_nearest_control", (1, 4), (0.2, 0.1))
        self.assertEqual(PredictionRowAntigravity.from_payload(row.to_payload()), row)
        for forbidden in ("label", "labels_blob", "ground_truth", "anomaly_positions", "confusion", "f1", "evaluation", "precision"):
            payload = row.to_payload()
            payload["evidence"] = {"nested": [{forbidden: 1}]}
            with self.subTest(forbidden=forbidden):
                with self.assertRaises(ValueError):
                    validate_prediction_payload_antigravity(payload)

    def test_checkpoint_context_state_and_evaluation_round_trip_and_wrong_types(self) -> None:
        context = CheckpointContextAntigravity("Exp443", "run", _DIGEST, _DIGEST, 57, 11, 1)
        self.assertEqual(CheckpointContextAntigravity.from_payload(context.to_payload()), context)
        state = CheckpointStateAntigravity("running", 2, 57, 0, None, _DIGEST)
        self.assertEqual(CheckpointStateAntigravity.from_payload(state.to_payload()), state)
        evidence = EvaluationEvidenceAntigravity("linear_edge_nearest_control", 1915, 303, 2003, 191679, 0.6241851368970013, True)
        self.assertEqual(EvaluationEvidenceAntigravity.from_payload(evidence.to_payload()), evidence)
        with self.assertRaises(ValueError):
            CheckpointStateAntigravity("running", 2.0, 57, 0, None, _DIGEST)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            EvaluationEvidenceAntigravity("linear_edge_nearest_control", 1915, 303, 2003, 191679, float("nan"), True)


if __name__ == "__main__":
    unittest.main()
