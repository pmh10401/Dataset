"""RED tests for the frozen Exp443 TRAIN policies and TEST preprocessing.

These tests intentionally exercise real preprocessing functions against literal
float32 rows.  The real 19-dataset test is opt-in because it opens the pinned
database/source layer and is run separately during Task 4 evidence collection.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from experiments.exp443_missing_normal_train_policy_comparison_antigravity.contracts_antigravity import RawSeriesAntigravity
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.database_antigravity import (
    load_raw_dataset_antigravity,
    pin_database_snapshot_antigravity,
)
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.config_antigravity import FROZEN_CONFIG
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.sources_antigravity import (
    verify_frozen_sources_antigravity,
)
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.preprocessing_antigravity import (
    POLICY_A_ANTIGRAVITY,
    POLICY_B_ANTIGRAVITY,
    POLICY_C_ANTIGRAVITY,
    PreparedDatasetAntigravity,
    PreparedTrainAntigravity,
    apply_train_policy_antigravity,
    original_target_length_antigravity,
    prepare_dataset_antigravity,
    repair_linear_edge_nearest_antigravity,
)


_DIGEST = "a" * 64


def _position_hash(positions: tuple[int, ...] | list[int]) -> str:
    payload = json.dumps(list(positions), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return sha256(payload).hexdigest()


def _raw(
    values: list[float],
    index: int,
    *,
    dataset: str = "Demo",
    split: str = "TRAIN",
    digest: str = _DIGEST,
) -> RawSeriesAntigravity:
    array = np.asarray(values, dtype="<f4")
    positions = tuple(int(item) for item in np.flatnonzero(~np.isfinite(array)))
    return RawSeriesAntigravity(dataset, split, index, digest, array, positions)


@dataclass(frozen=True)
class _UnequalRaw:
    """A literal raw adapter fixture that permits unequal original lengths."""

    dataset: str
    train: tuple[RawSeriesAntigravity, ...]
    test: tuple[RawSeriesAntigravity, ...]


def _fixture_raw() -> _UnequalRaw:
    train = (
        _raw([1.0, 2.0, 3.0, 4.0], 0),
        _raw([1.0, np.nan, 3.0, 4.0], 1),
        _raw([2.0, 4.0, 6.0, 8.0], 2),
        _raw([5.0, 6.0, 7.0, 8.0], 3),
        _raw([9.0, 10.0, np.inf, 12.0], 4),
        _raw([13.0, 14.0, 15.0, 16.0], 5),
    )
    test = (
        _raw([0.0, np.nan, 2.0, np.inf], 0, split="TEST"),
        _raw([2.0, 4.0, 6.0, 8.0], 1, split="TEST"),
    )
    return _UnequalRaw("Demo", train, test)


class Exp443PreprocessingAntigravityTests(unittest.TestCase):
    def test_linear_repair_matches_exp226_for_internal_and_edge_nonfinite(self) -> None:
        source = np.asarray([np.nan, 1.0, np.nan, 5.0, np.inf], dtype="<f4")
        actual = repair_linear_edge_nearest_antigravity(source)
        expected = np.asarray([1.0, 1.0, 3.0, 5.0, 5.0], dtype=np.float32)
        self.assertEqual(actual.dtype, np.dtype("<f4"))
        self.assertTrue(np.array_equal(actual, expected))
        from experiments.exp226_235_hard_alert_quality.preprocessing import sanitize_series

        self.assertEqual(actual.tobytes(), sanitize_series(source).astype("<f4").tobytes())

    def test_linear_repair_handles_negative_infinity_one_finite_and_all_finite(self) -> None:
        self.assertTrue(
            np.array_equal(
                repair_linear_edge_nearest_antigravity(np.asarray([-np.inf, 2.0, np.nan, 8.0], dtype="<f4")),
                np.asarray([2.0, 2.0, 5.0, 8.0], dtype="<f4"),
            )
        )
        self.assertTrue(
            np.array_equal(
                repair_linear_edge_nearest_antigravity(np.asarray([np.nan, 4.0, np.inf], dtype="<f4")),
                np.asarray([4.0, 4.0, 4.0], dtype="<f4"),
            )
        )
        finite = np.asarray([1.0, 2.0, 3.0], dtype="<f4")
        repaired = repair_linear_edge_nearest_antigravity(finite)
        self.assertTrue(np.array_equal(repaired, finite))
        self.assertEqual(repaired.dtype, np.dtype("<f4"))

    def test_linear_repair_fails_closed_for_empty_and_all_missing(self) -> None:
        with self.assertRaises(ValueError):
            repair_linear_edge_nearest_antigravity(np.asarray([], dtype="<f4"))
        with self.assertRaises(ValueError):
            repair_linear_edge_nearest_antigravity(np.asarray([np.nan, np.inf, -np.inf], dtype="<f4"))

    def test_target_length_uses_original_train_lengths_before_any_policy(self) -> None:
        rows = (_raw([1.0] * 7, 0), _raw([1.0] * 11, 1), _raw([1.0] * 12, 2))
        self.assertEqual(original_target_length_antigravity(rows), 11)
        # Filtering a short row must not change the target computed from the
        # original population.
        filtered = tuple(row for row in rows if row.instance_index != 2)
        self.assertNotEqual(original_target_length_antigravity(rows), original_target_length_antigravity(filtered))

    def test_policy_a_repairs_rows_and_records_exact_audit_identity(self) -> None:
        raw = _fixture_raw()
        prepared = apply_train_policy_antigravity(raw, POLICY_A_ANTIGRAVITY, 4)
        self.assertIsInstance(prepared, PreparedTrainAntigravity)
        self.assertEqual(prepared.instance_indices, (0, 1, 2, 3, 4, 5))
        self.assertEqual(prepared.train_total, 6)
        self.assertEqual(prepared.train_missing, 2)
        self.assertEqual(prepared.train_used, 6)
        self.assertEqual(prepared.train_excluded, 0)
        self.assertEqual(prepared.status, "scored")
        self.assertTrue(np.isfinite(prepared.train_values).all())
        self.assertEqual(prepared.train_values.dtype, np.dtype("<f4"))
        self.assertFalse(prepared.train_values.flags.writeable)
        by_index = {audit.instance_index: audit for audit in prepared.audits}
        self.assertEqual(by_index[1].action, "linear_and_edge_nearest")
        self.assertEqual(by_index[1].missing_positions, (1,))
        self.assertEqual(by_index[1].missing_positions_sha256, _position_hash((1,)))
        self.assertEqual(by_index[1].raw_blob_sha256, _DIGEST)
        self.assertEqual(by_index[4].missing_positions, (2,))
        self.assertEqual(by_index[4].action, "linear_and_edge_nearest")

    def test_policy_b_excludes_complete_nonfinite_rows_and_abstains_below_five(self) -> None:
        raw = _fixture_raw()
        prepared = apply_train_policy_antigravity(raw, POLICY_B_ANTIGRAVITY, 4)
        self.assertEqual(prepared.instance_indices, (0, 2, 3, 5))
        self.assertEqual(prepared.train_total, 6)
        self.assertEqual(prepared.train_missing, 2)
        self.assertEqual(prepared.train_used, 4)
        self.assertEqual(prepared.train_excluded, 2)
        self.assertEqual(prepared.status, "abstained")
        # Abstention is a status, not synthetic zero filling: finite rows stay
        # available in the prepared matrix for the runtime to reject cleanly.
        self.assertEqual(prepared.train_values.shape, (4, 4))
        self.assertTrue(np.isfinite(prepared.train_values).all())
        excluded = [audit for audit in prepared.audits if audit.action == "excluded_nonfinite_row"]
        self.assertEqual([audit.instance_index for audit in excluded], [1, 4])
        self.assertEqual(excluded[0].original_length, 4)
        self.assertEqual(excluded[0].missing_positions, (1,))
        self.assertEqual(excluded[0].missing_positions_sha256, _position_hash((1,)))
        self.assertEqual(excluded[0].raw_blob_sha256, _DIGEST)
        self.assertEqual(excluded[0].action, "excluded_nonfinite_row")

    def test_policy_c_uses_same_length_leave_one_out_peers_and_even_median(self) -> None:
        train = (
            _raw([1.0, np.nan, 3.0, 4.0], 0),
            _raw([2.0, 10.0, 4.0, 5.0], 1),
            _raw([3.0, 20.0, 5.0, 6.0], 2),
            _raw([4.0, 30.0, 6.0, 7.0], 3),
            # Different original length: it must not contribute to index 1.
            _raw([100.0, 999.0, 100.0, 100.0, 100.0], 4),
            # Missing peer at the repaired coordinate is ignored.
            _raw([5.0, np.nan, 7.0, 8.0], 5),
        )
        raw = _UnequalRaw("Demo", train, (_raw([0.0, 1.0, 2.0, 3.0], 0, split="TEST"),))
        prepared = apply_train_policy_antigravity(raw, POLICY_C_ANTIGRAVITY, 4)
        self.assertEqual(prepared.instance_indices, (0, 1, 2, 3, 4, 5))
        repaired_row = prepared.train_values[0]
        # Same-length finite peers at coordinate 1 are [10, 20, 30]; the
        # different-length 999 and self NaN are not used.  Prepared rows are
        # z-normalized after repair, exactly like Exp249.
        expected_repaired = np.asarray([1.0, 20.0, 3.0, 4.0], dtype="<f4")
        expected_center = expected_repaired.mean(dtype=np.float32)
        expected_scale = expected_repaired.std(dtype=np.float32)
        expected_normalized = ((expected_repaired - expected_center) / expected_scale).astype("<f4")
        self.assertEqual(repaired_row.tobytes(), expected_normalized.tobytes())
        audit = next(audit for audit in prepared.audits if audit.instance_index == 0)
        self.assertEqual(audit.reference_instance_indices, (1, 2, 3, 5))
        self.assertEqual(audit.action, "leave_one_out_peer_median")
        # The median of an even peer set is the ordinary NumPy average of the
        # two central values, not a lower/upper selection.
        self.assertEqual(
            np.median(np.asarray([10.0, 20.0, 30.0, 40.0], dtype="<f4")),
            np.float32(25.0),
        )

    def test_policy_c_excludes_row_when_any_coordinate_has_no_finite_peer(self) -> None:
        train = (
            # A singleton original-length group has no peer for position 1.
            _raw([1.0, np.nan, 3.0], 0),
            _raw([2.0, 10.0, 5.0, 9.0], 1),
            _raw([3.0, 11.0, 6.0, 9.0], 2),
            _raw([4.0, 12.0, 7.0, 9.0], 3),
            _raw([5.0, 13.0, 8.0, 9.0], 4),
        )
        raw = _UnequalRaw("Demo", train, (_raw([0.0, 1.0, 2.0], 0, split="TEST"),))
        prepared = apply_train_policy_antigravity(raw, POLICY_C_ANTIGRAVITY, 3)
        self.assertEqual(prepared.instance_indices, (1, 2, 3, 4))
        self.assertEqual(prepared.train_used, 4)
        self.assertEqual(prepared.train_excluded, 1)
        audit = next(audit for audit in prepared.audits if audit.instance_index == 0)
        self.assertEqual(audit.action, "excluded_unsupported_peer")
        self.assertEqual(audit.unsupported_positions, (1,))
        self.assertEqual(audit.reference_instance_indices, ())
        self.assertEqual(prepared.status, "abstained")

    def test_finite_rows_keep_exact_bytes_and_original_order_across_policies(self) -> None:
        raw = _fixture_raw()
        target = original_target_length_antigravity(raw.train)
        prepared = {
            policy: apply_train_policy_antigravity(raw, policy, target)
            for policy in (POLICY_A_ANTIGRAVITY, POLICY_B_ANTIGRAVITY, POLICY_C_ANTIGRAVITY)
        }
        for index in (0, 2, 3, 5):
            rows = [prepared[policy] for policy in prepared]
            bytes_by_policy = {
                policy: rows[position].train_values[rows[position].instance_indices.index(index)].tobytes()
                for position, policy in enumerate(prepared)
                if index in rows[position].instance_indices
            }
            self.assertEqual(len(bytes_by_policy), 3)
            self.assertEqual(len(set(bytes_by_policy.values())), 1)
        self.assertEqual(prepared[POLICY_A_ANTIGRAVITY].instance_indices, (0, 1, 2, 3, 4, 5))
        self.assertEqual(prepared[POLICY_B_ANTIGRAVITY].instance_indices, (0, 2, 3, 5))

    def test_prepare_dataset_keeps_fixed_test_preprocessing_byte_identical(self) -> None:
        raw = _fixture_raw()
        outputs = {
            policy: prepare_dataset_antigravity(raw, policy)
            for policy in (POLICY_A_ANTIGRAVITY, POLICY_B_ANTIGRAVITY, POLICY_C_ANTIGRAVITY)
        }
        self.assertTrue(all(isinstance(value, PreparedDatasetAntigravity) for value in outputs.values()))
        self.assertEqual({value.target_length for value in outputs.values()}, {8})
        self.assertEqual({value.test_values.dtype for value in outputs.values()}, {np.dtype("<f4")})
        test_bytes = {policy: value.test_values.tobytes() for policy, value in outputs.items()}
        self.assertEqual(len(set(test_bytes.values())), 1)
        self.assertTrue(all(np.isfinite(value.test_values).all() for value in outputs.values()))
        self.assertTrue(all(not value.test_values.flags.writeable for value in outputs.values()))

    def test_prepared_arrays_have_immutable_byte_ownership_and_no_mask_feature(self) -> None:
        prepared = prepare_dataset_antigravity(_fixture_raw(), POLICY_A_ANTIGRAVITY)
        self.assertIsNone(getattr(prepared, "missing_mask", None))
        self.assertFalse(prepared.train_values.flags.writeable)
        self.assertFalse(prepared.test_values.flags.writeable)
        with self.assertRaises(ValueError):
            prepared.train_values.setflags(write=True)

    @unittest.skipUnless(os.environ.get("EXP443_REAL_TASK4_ANTIGRAVITY") == "1", "opt-in real pinned source evidence")
    def test_real_19_dataset_policy_invariants_are_label_free(self) -> None:
        """Run once with the pinned source layer; no TEST labels are imported."""
        snapshot = verify_frozen_sources_antigravity()
        attempt_root = Path(os.environ.get("EXP443_TASK4_ATTEMPT_ROOT", "/tmp/exp443_task4_attempt_antigravity"))
        real_results: dict[str, dict[str, PreparedDatasetAntigravity]] = {}
        raw_by_dataset = {}
        pinned = pin_database_snapshot_antigravity(
            Path(FROZEN_CONFIG.database_path), attempt_root, expected_sha256=FROZEN_CONFIG.database_sha256
        )
        try:
            for dataset in snapshot.affected_datasets:
                raw = load_raw_dataset_antigravity(pinned, dataset)
                raw_by_dataset[dataset] = raw
                real_results[dataset] = {
                    policy: prepare_dataset_antigravity(raw, policy)
                    for policy in (POLICY_A_ANTIGRAVITY, POLICY_B_ANTIGRAVITY, POLICY_C_ANTIGRAVITY)
                }
            self.assertEqual(pinned.label_access_count, 0)
        finally:
            pinned.close(terminal_evidence_durable=True)
        self.assertEqual(len(real_results), 19)
        self.assertEqual(sum(value[POLICY_C_ANTIGRAVITY].train_excluded for value in real_results.values()), 2)
        self.assertEqual(sum(value[POLICY_C_ANTIGRAVITY].train_missing - value[POLICY_C_ANTIGRAVITY].train_excluded for value in real_results.values()), 162)
        self.assertEqual(snapshot.label_access_count, 0)
        self.assertTrue(all(np.isfinite(value.train_values).all() and np.isfinite(value.test_values).all() for rows in real_results.values() for value in rows.values()))
        self.assertTrue(all(value[POLICY_B_ANTIGRAVITY].train_used >= 15 for value in real_results.values()))
        unsupported = tuple(
            (dataset, audit.instance_index, audit.unsupported_positions)
            for dataset in sorted(real_results)
            for audit in real_results[dataset][POLICY_C_ANTIGRAVITY].audits
            if audit.action == "excluded_unsupported_peer"
        )
        self.assertEqual(
            unsupported,
            (
                ("MelbournePedestrian_normal_8", 1, (2, 3)),
                ("MelbournePedestrian_normal_9", 242, (4, 5)),
            ),
        )
        self.assertEqual(sum(len(positions) for _, _, positions in unsupported), 4)

        # Every originally finite TRAIN row is byte-identical across A/B/C;
        # TEST is fixed byte-identical across all policies.
        from experiments.exp239_248_diverse_score_sources.foundation import load_bundle

        for dataset, raw in raw_by_dataset.items():
            outputs = real_results[dataset]
            self.assertEqual(outputs[POLICY_A_ANTIGRAVITY].target_length, snapshot.target_length_by_dataset[dataset])
            for row in raw.train:
                if row.missing_positions:
                    continue
                row_bytes = {
                    policy: outputs[policy].train_values[outputs[policy].train_instance_indices.index(row.instance_index)].tobytes()
                    for policy in (POLICY_A_ANTIGRAVITY, POLICY_B_ANTIGRAVITY, POLICY_C_ANTIGRAVITY)
                }
                self.assertEqual(len(set(row_bytes.values())), 1, dataset)
            test_bytes = {policy: outputs[policy].test_values.tobytes() for policy in outputs}
            self.assertEqual(len(set(test_bytes.values())), 1, dataset)

            # Policy A is an exact Exp249 foundation replay for every affected
            # dataset, including ragged TRAIN/TEST row counts.
            foundation = load_bundle(dataset, Path(FROZEN_CONFIG.database_path))
            self.assertEqual(outputs[POLICY_A_ANTIGRAVITY].train_values.tobytes(), np.asarray(foundation.train_normal, dtype="<f4").tobytes(), dataset)
            self.assertEqual(outputs[POLICY_A_ANTIGRAVITY].test_values.tobytes(), np.asarray(foundation.test_values, dtype="<f4").tobytes(), dataset)


if __name__ == "__main__":
    unittest.main()
