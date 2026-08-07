from hashlib import sha256
import unittest

import numpy as np

from experiments.exp414_missing_aware_near_repeat.repair import (
    build_dataset_repair_audit,
    repair_train_blob,
)


class RepairTests(unittest.TestCase):
    def test_repairs_internal_and_edges_as_little_endian_float32(self) -> None:
        raw = np.asarray([np.nan, 1.0, np.nan, 5.0, np.inf], dtype="<f4").tobytes()
        run, audit = repair_train_blob("Alpha_normal_0", 7, raw)
        self.assertIsNotNone(run)
        assert run is not None
        np.testing.assert_array_equal(
            np.asarray(run.values, dtype="<f4"),
            np.asarray([1.0, 1.0, 3.0, 5.0, 5.0], dtype="<f4"),
        )
        self.assertEqual(audit.repair_method, "linear_and_edge_nearest")
        self.assertEqual((audit.nan_count, audit.posinf_count, audit.neginf_count), (2, 1, 0))

    def test_finite_blob_is_byte_identical(self) -> None:
        raw = np.asarray([0.25, 1.5, -2.0], dtype="<f4").tobytes()
        run, audit = repair_train_blob("Alpha_normal_0", 0, raw)
        self.assertIsNotNone(run)
        assert run is not None
        self.assertEqual(audit.raw_values_sha256, audit.repaired_values_sha256)
        self.assertEqual(np.asarray(run.values, dtype="<f4").tobytes(), raw)
        self.assertEqual(audit.repair_method, "unchanged")

    def test_all_missing_returns_no_run(self) -> None:
        raw = np.asarray([np.nan, np.inf, -np.inf], dtype="<f4").tobytes()
        run, audit = repair_train_blob("Alpha_normal_0", 0, raw)
        self.assertIsNone(run)
        self.assertIsNone(audit.repaired_values_sha256)
        self.assertEqual(audit.repair_method, "all_missing_abstain")

    def test_dataset_audit_preserves_raw_and_repaired_identities(self) -> None:
        first = np.asarray([0.0, 1.0], dtype="<f4").tobytes()
        second = np.asarray([np.nan, 2.0], dtype="<f4").tobytes()
        first_run, first_audit = repair_train_blob("Alpha_normal_0", 0, first)
        second_run, second_audit = repair_train_blob("Alpha_normal_0", 1, second)
        assert first_run is not None
        assert second_run is not None

        dataset_audit = build_dataset_repair_audit(
            "Alpha_normal_0",
            (first_run, second_run),
            (first_audit, second_audit),
        )

        self.assertEqual(dataset_audit.status, "ready")
        self.assertEqual(dataset_audit.runs, (first_audit, second_audit))
        self.assertNotEqual(
            dataset_audit.raw_train_identity_sha256,
            dataset_audit.repaired_train_identity_sha256,
        )
        self.assertEqual(first_audit.raw_values_sha256, sha256(first).hexdigest())

    def test_dataset_audit_abstains_when_any_run_has_no_finite_support(self) -> None:
        raw = np.asarray([np.nan, np.inf], dtype="<f4").tobytes()
        run, audit = repair_train_blob("Alpha_normal_0", 0, raw)
        self.assertIsNone(run)

        dataset_audit = build_dataset_repair_audit("Alpha_normal_0", (), (audit,))

        self.assertEqual(dataset_audit.status, "missing_support_abstain")
        self.assertIsNone(dataset_audit.repaired_train_identity_sha256)


if __name__ == "__main__":
    unittest.main()
