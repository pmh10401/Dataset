"""RED tests for strict Exp443 in-memory and serialized contracts."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
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
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.config_antigravity import (
    FROZEN_CONFIG,
    scientific_config_sha256_antigravity,
)


_DIGEST = "a" * 64


def _values_digest(values: np.ndarray) -> str:
    return sha256(np.asarray(values, dtype="<f4").tobytes(order="C")).hexdigest()


def _missing_hash(positions: tuple[int, ...]) -> str:
    import json

    return sha256(json.dumps(list(positions), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def _raw(split: str = "TRAIN", index: int = 0) -> RawSeriesAntigravity:
    values = np.asarray([1.0, np.nan, 3.0], dtype="<f4")
    return RawSeriesAntigravity("Demo", split, index, _values_digest(values), values, (1,))


class Exp443ContractsAntigravityTests(unittest.TestCase):
    def test_raw_series_copies_array_read_only_and_round_trips_explicit_dtype_shape(self) -> None:
        source = np.asarray([1.0, np.nan, 3.0], dtype="<f4")
        item = RawSeriesAntigravity("Demo", "TRAIN", 0, _values_digest(source), source, (1,))
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
            RawSeriesAntigravity("Demo", "TRAIN", 0, _values_digest(np.asarray([1, np.nan, 3], dtype="<f4")), np.asarray([1, np.nan, 3], dtype="<f4"), ())
        with self.assertRaises(ValueError):
            RawSeriesAntigravity("Demo", "TRAIN", 0, "A" * 64, np.asarray([1, 2, 3], dtype="<f4"), ())

    def test_raw_series_recomputes_exact_little_endian_float32_digest(self) -> None:
        values = np.asarray([1.0, np.nan, 3.0], dtype="<f4")
        with self.assertRaisesRegex(ValueError, "raw_blob_sha256"):
            RawSeriesAntigravity("Demo", "TRAIN", 0, _DIGEST, values, (1,))

    def test_policy_audit_and_score_arrays_copy_round_trip_and_reject_nonfinite_scores(self) -> None:
        audit = RunPolicyAuditAntigravity("Demo", "exclude_nonfinite_row", 0, 3, (1,), _missing_hash((1,)), _DIGEST, "excluded_nonfinite_row")
        self.assertEqual(RunPolicyAuditAntigravity.from_payload(audit.to_payload()), audit)
        self.assertEqual(audit.to_payload()["missing_positions_sha256"], _missing_hash((1,)))
        oof = np.asarray([0.1, 0.2], dtype="<f8")
        test = np.asarray([0.3], dtype="<f8")
        score = KernelSeedScoresAntigravity(1024, 20260721, oof, test, (0,))
        oof[0] = 9.0
        self.assertEqual(score.oof_scores[0], 0.1)
        self.assertFalse(score.oof_scores.flags.writeable)
        self.assertEqual(KernelSeedScoresAntigravity.from_payload(score.to_payload()), score)
        with self.assertRaises(ValueError):
            KernelSeedScoresAntigravity(1024, 20260721, np.asarray([np.nan], dtype="<f8"), test, ())

    def test_policy_audit_freezes_action_vocabulary_and_missingness_semantics(self) -> None:
        invalid_cases = (
            ("linear_edge_nearest_control", (), "linear_and_edge_nearest"),
            ("linear_edge_nearest_control", (1,), "unchanged"),
            ("exclude_nonfinite_row", (1,), "unchanged"),
            ("exclude_nonfinite_row", (), "excluded_nonfinite_row"),
            ("leave_one_out_peer_median", (1,), "excluded_nonfinite_row"),
        )
        for policy, missing, action in invalid_cases:
            with self.subTest(policy=policy, missing=missing, action=action), self.assertRaises(ValueError):
                RunPolicyAuditAntigravity(
                    "Demo", policy, 0, 3, missing, _missing_hash(missing), _DIGEST, action,
                    (1,) if action == "leave_one_out_peer_median" else (),
                )
        valid = RunPolicyAuditAntigravity(
            "Demo", "leave_one_out_peer_median", 0, 3, (1,), _missing_hash((1,)), _DIGEST,
            "leave_one_out_peer_median", (1, 2), (),
        )
        self.assertEqual(valid.unsupported_positions, ())

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
        with self.assertRaises(ValueError):
            PolicyDatasetResultAntigravity(
                "linear_edge_nearest_control", "Demo", 3, 4, 1, 3, 0, 1,
                (score,), mapping, (), "scored",
            )
        with self.assertRaises(ValueError):
            PolicyDatasetResultAntigravity(
                "linear_edge_nearest_control", "Demo", 3, 4, 5, 0, 4, 1,
                (score,), mapping, (), "scored",
            )

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
        context = CheckpointContextAntigravity(
            FROZEN_CONFIG.experiment_id,
            FROZEN_CONFIG.run_identity,
            scientific_config_sha256_antigravity(),
            FROZEN_CONFIG.environment_lock_sha256,
            FROZEN_CONFIG.recompute_units,
            FROZEN_CONFIG.outer_workers,
            FROZEN_CONFIG.inner_threads,
        )
        self.assertTrue(context.validate_antigravity())
        self.assertEqual(CheckpointContextAntigravity.from_payload(context.to_payload()), context)
        state = CheckpointStateAntigravity("running", 2, 57, 0, None, _DIGEST)
        self.assertEqual(CheckpointStateAntigravity.from_payload(state.to_payload()), state)
        evidence = EvaluationEvidenceAntigravity("linear_edge_nearest_control", 1915, 303, 2003, 191679, 0.6241851368970013, True)
        self.assertEqual(EvaluationEvidenceAntigravity.from_payload(evidence.to_payload()), evidence)
        with self.assertRaises(ValueError):
            CheckpointStateAntigravity("running", 2.0, 57, 0, None, _DIGEST)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            EvaluationEvidenceAntigravity("linear_edge_nearest_control", 1915, 303, 2003, 191679, float("nan"), True)

    def test_checkpoint_context_rejects_each_identity_or_unit_mismatch(self) -> None:
        context = CheckpointContextAntigravity(
            FROZEN_CONFIG.experiment_id,
            FROZEN_CONFIG.run_identity,
            scientific_config_sha256_antigravity(),
            FROZEN_CONFIG.environment_lock_sha256,
            FROZEN_CONFIG.recompute_units,
            FROZEN_CONFIG.outer_workers,
            FROZEN_CONFIG.inner_threads,
        )
        for field_name, value in (
            ("experiment_id", "Exp442"),
            ("run_identity", "fabricated-run"),
            ("scientific_config_sha256", _DIGEST),
            ("environment_lock_sha256", _DIGEST),
            ("total_units", 56),
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError):
                    replace(context, **{field_name: value})

    def test_each_core_contract_rejects_wrong_types_or_invalid_values(self) -> None:
        score = KernelSeedScoresAntigravity(
            1024, 20260721, np.asarray([0.1], dtype="<f8"),
            np.asarray([0.2], dtype="<f8"), (0,),
        )
        invalid_cases = (
            ("raw_dataset", lambda: replace(_raw(), dataset=3)),
            ("raw_split", lambda: replace(_raw(), split="VAL")),
            ("raw_index", lambda: replace(_raw(), instance_index=1.0)),
            ("audit_policy", lambda: replace(RunPolicyAuditAntigravity("Demo", "exclude_nonfinite_row", 0, 3, (1,), _missing_hash((1,)), _DIGEST, "excluded_nonfinite_row"), policy="bad")),
            ("audit_position", lambda: RunPolicyAuditAntigravity("Demo", "exclude_nonfinite_row", 0, 3, (3,), _missing_hash((3,)), _DIGEST, "excluded_nonfinite_row")),
            ("score_kernel", lambda: replace(score, kernel=512)),
            ("score_seed", lambda: replace(score, seed=123456)),
            ("score_candidates", lambda: replace(score, candidate_indices=(1,))),
            ("result_status", lambda: PolicyDatasetResultAntigravity("linear_edge_nearest_control", "Demo", 3, 1, 0, 1, 0, 1, (score,), {1024: (0,)}, (), "unknown")),
            ("prediction_score", lambda: PredictionRowAntigravity("Demo", "linear_edge_nearest_control", (0,), (float("nan"),))),
            ("prediction_indices", lambda: PredictionRowAntigravity("Demo", "linear_edge_nearest_control", (1, 0), (0.1, 0.2))),
            ("state_status", lambda: CheckpointStateAntigravity("unknown", 0, 57, 0, None, _DIGEST)),
            ("state_digest", lambda: CheckpointStateAntigravity("running", 0, 57, 0, None, "not-a-digest")),
            ("evaluation_policy", lambda: EvaluationEvidenceAntigravity("bad", 0, 0, 0, 0, 0.0, False)),
            ("evaluation_gate", lambda: EvaluationEvidenceAntigravity("linear_edge_nearest_control", 0, 0, 0, 0, 0.0, 1)),
        )
        for name, factory in invalid_cases:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    factory()


if __name__ == "__main__":
    unittest.main()
