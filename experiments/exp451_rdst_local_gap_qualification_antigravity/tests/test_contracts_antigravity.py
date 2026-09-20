"""Exact, mutation-safe records for later Exp451 phases."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.exp451_rdst_local_gap_qualification_antigravity.contracts_antigravity import (
    CandidateScoresAntigravity,
    DatasetBundleAntigravity,
    FrozenPredictionAuthorityAntigravity,
)


class ContractsAntigravityTest(unittest.TestCase):
    def test_dataset_bundle_copies_caller_arrays_without_labels_antigravity(self) -> None:
        train = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        test = np.array([[5.0, 6.0]], dtype=np.float64)
        bundle = DatasetBundleAntigravity(0, "Crop", "Crop_001", train, test, 1)
        train[0, 0] = 99.0
        test[0, 0] = 99.0
        self.assertEqual(bundle.dataset_identity, (0, "Crop", "Crop_001"))
        np.testing.assert_array_equal(bundle.train_normal, np.array([[1.0, 2.0], [3.0, 4.0]]))
        self.assertFalse(bundle.train_normal.flags.writeable)
        self.assertFalse(hasattr(bundle, "labels"))
        with self.assertRaises(ValueError):
            DatasetBundleAntigravity(True, "Crop", "Crop_001", train, test, 1)
        with self.assertRaises(ValueError):
            DatasetBundleAntigravity(0, "Crop", "Crop_001", train, np.array([[1.0, 2.0, 3.0]]), 1)

    def test_candidate_scores_enforce_rdst_width_and_snapshot_arrays_antigravity(self) -> None:
        train = np.array([0.1, 0.2], dtype=np.float64)
        test = np.array([0.3], dtype=np.float64)
        scores = CandidateScoresAntigravity(train, test, 2, 1, 6, 2, 20260721)
        train[0] = 9.0
        self.assertEqual(scores.train_oof[0], 0.1)
        self.assertFalse(scores.train_oof.flags.writeable)
        self.assertFalse(hasattr(scores, "__dict__"))
        with self.assertRaises(ValueError):
            CandidateScoresAntigravity(np.array([0.1]), np.array([0.2]), 1, 1, 5, 2, 20260721)
        with self.assertRaises(ValueError):
            CandidateScoresAntigravity(np.array([0.1]), np.array([0.2]), 1, 1, 3, True, 20260721)

    def test_candidate_scores_bind_expected_train_and_test_lengths_antigravity(self) -> None:
        train = np.arange(4, dtype=np.float64)
        test = np.arange(3, dtype=np.float64)
        scores = CandidateScoresAntigravity(train, test, 4, 3, 9, 3, 20260721)
        self.assertEqual((scores.expected_train_length, scores.expected_test_length), (4, 3))
        for expected_train, expected_test in ((1, 3), (4, 500), (True, 3), (4, True)):
            with self.subTest(expected_train=expected_train, expected_test=expected_test), self.assertRaises(ValueError):
                CandidateScoresAntigravity(train, test, expected_train, expected_test, 9, 3, 20260721)

    def test_prediction_authority_has_canonical_hash_and_rejects_unordered_or_malformed_input_antigravity(self) -> None:
        authority = FrozenPredictionAuthorityAntigravity(
            "a" * 40,
            ((0, "Crop", "Crop_001"), (1, "Crop", "Crop_002")),
            (3, 2),
            ((0, 2), ()),
        )
        payload = authority.to_payload_antigravity()
        restored = FrozenPredictionAuthorityAntigravity.from_payload_antigravity(payload)
        self.assertEqual(restored.canonical_bytes_antigravity, authority.canonical_bytes_antigravity)
        self.assertEqual(restored.canonical_sha256_antigravity, authority.canonical_sha256_antigravity)
        self.assertEqual(len(authority.canonical_sha256_antigravity), 64)
        with self.assertRaises(ValueError):
            FrozenPredictionAuthorityAntigravity("a" * 40, ((1, "Crop", "Crop_002"), (0, "Crop", "Crop_001")), (2, 3), ((), ()))
        malformed = dict(payload)
        malformed["canonical_sha256"] = "bad"
        with self.assertRaises(ValueError):
            FrozenPredictionAuthorityAntigravity.from_payload_antigravity(malformed)
        malformed = dict(payload)
        malformed["extra"] = 1
        with self.assertRaises(ValueError):
            FrozenPredictionAuthorityAntigravity.from_payload_antigravity(malformed)


if __name__ == "__main__":
    unittest.main()
