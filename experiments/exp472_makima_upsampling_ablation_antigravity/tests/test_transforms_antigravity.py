"""Behavioral tests for the isolated Exp472 Makima alignment boundary."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import unittest
from unittest.mock import patch

import numpy as np
import scipy.interpolate

from experiments.exp450_resampling_image_ablation_antigravity.transforms_antigravity import (
    resample_series_exp450_antigravity,
)
from paper.equipment_anomaly_detection.contracts_paper import RawDataset

from experiments.exp472_makima_upsampling_ablation_antigravity import (
    transforms_antigravity as transforms,
)


def _raw_fixture_antigravity() -> RawDataset:
    """Return a label-free fixture with both upsampled and unchanged rows."""
    train = (
        np.asarray([0.0, 2.0, 0.0], dtype=np.float32),
        np.arange(6, dtype=np.float32),
    )
    test = (np.asarray([1.0], dtype=np.float32),)
    return RawDataset(
        dataset="fixture",
        metadata_length=6,
        train_indices=(0, 1),
        train_blobs=tuple(row.astype("<f4").tobytes() for row in train),
        train_rows=train,
        test_indices=(0,),
        test_blobs=tuple(row.astype("<f4").tobytes() for row in test),
        test_rows=test,
    )


def _nonfinite_raw_fixture_antigravity() -> RawDataset:
    """Return an identity-valid dataset that Exp450 abstains before alignment."""
    train = (np.asarray([0.0, np.nan, 1.0], dtype=np.float32),)
    test = (np.asarray([2.0], dtype=np.float32),)
    return RawDataset(
        dataset="nonfinite_fixture",
        metadata_length=6,
        train_indices=(0,),
        train_blobs=tuple(row.astype("<f4").tobytes() for row in train),
        train_rows=train,
        test_indices=(0,),
        test_blobs=tuple(row.astype("<f4").tobytes() for row in test),
        test_rows=test,
    )


class Exp472TransformTests(unittest.TestCase):
    def test_makima_is_used_only_for_true_upsampling(self) -> None:
        """Routing equal or shorter targets to Makima would change frozen D parity."""
        row = np.asarray([0.0, 1.0, -0.5, 2.0], dtype=np.float32)
        with patch.object(
            transforms.scipy.interpolate,
            "Akima1DInterpolator",
            wraps=scipy.interpolate.Akima1DInterpolator,
        ) as makima:
            up = transforms.resample_series_exp472_antigravity(row, 13)
            same = transforms.resample_series_exp472_antigravity(row, 4)
            down = transforms.resample_series_exp472_antigravity(row, 2)
        self.assertEqual(makima.call_count, 1)
        self.assertEqual(
            makima.call_args.kwargs,
            {"method": "makima", "extrapolate": False},
        )
        np.testing.assert_array_equal(same, row)
        self.assertEqual((up.shape, down.shape), ((13,), (2,)))

    def test_equal_singleton_and_downsampling_equal_exp450_d(self) -> None:
        """Changing any non-upsampling route must fail the frozen D control parity."""
        cases = (
            (np.asarray([2.5], dtype=np.float32), 9),
            (np.arange(8, dtype=np.float32), 8),
            (np.arange(12, dtype=np.float32), 5),
        )
        for row, target in cases:
            with self.subTest(source_length=row.size, target=target):
                actual = transforms.resample_series_exp472_antigravity(row, target)
                expected = resample_series_exp450_antigravity(
                    row, target, mode="polyphase_pchip"
                )
                np.testing.assert_array_equal(actual, expected)

    def test_two_point_makima_is_finite_deterministic_and_endpoint_exact(self) -> None:
        """A nondeterministic or endpoint-shifting short Makima row is invalid."""
        row = np.asarray([-2.0, 3.0], dtype=np.float32)
        first = transforms.resample_series_exp472_antigravity(row, 17)
        second = transforms.resample_series_exp472_antigravity(row, 17)
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertEqual(first.dtype, np.dtype("float32"))
        self.assertTrue(np.isfinite(first).all())
        self.assertEqual((float(first[0]), float(first[-1])), (-2.0, 3.0))

    def test_shape_diagnostics_cover_preregistered_fixture_families(self) -> None:
        """Dropping shape evidence would hide overshoot or extrema changes."""
        fixtures = {
            "constant": np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            "monotone": np.asarray([0.0, 0.3, 1.0], dtype=np.float32),
            "peaked": np.asarray([0.0, 2.0, 0.0], dtype=np.float32),
            "oscillatory": np.asarray([0.0, 1.0, -1.0, 1.0, 0.0], dtype=np.float32),
        }
        for name, row in fixtures.items():
            with self.subTest(name=name):
                output = transforms.resample_series_exp472_antigravity(row, 31)
                diagnostic = transforms.shape_diagnostic_exp472_antigravity(row, output)
                self.assertGreaterEqual(diagnostic["overshoot"], 0.0)
                self.assertGreaterEqual(diagnostic["undershoot"], 0.0)
                self.assertGreaterEqual(diagnostic["output_extrema"], 0)
                self.assertEqual(len(diagnostic["source_sha256"]), 64)
                self.assertEqual(len(diagnostic["output_sha256"]), 64)

    def test_align_rows_records_each_shape_without_changing_row_order(self) -> None:
        """A reordered batch or missing per-row diagnostics invalidates score provenance."""
        rows = (
            np.asarray([0.0, 2.0, 0.0], dtype=np.float32),
            np.asarray([1.0], dtype=np.float32),
        )
        diagnostics: list[dict[str, object]] = []
        actual = transforms.align_rows_exp472_antigravity(rows, 7, diagnostics)
        expected = np.stack(
            [transforms.resample_series_exp472_antigravity(row, 7) for row in rows]
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.dtype("float32"))
        self.assertEqual(len(diagnostics), 2)
        self.assertEqual(
            diagnostics[0]["source_sha256"], sha256(rows[0].tobytes()).hexdigest()
        )

    def test_dataset_diagnostics_count_true_upsampling_without_labels(self) -> None:
        """Counting singleton/equal rows as Makima rows would corrupt the subset audit."""
        values = _raw_fixture_antigravity()
        diagnostic = transforms.dataset_shape_diagnostic_exp472_antigravity(values)
        self.assertEqual(
            transforms.count_upsampled_rows_exp472_antigravity(values), 1
        )
        self.assertEqual(diagnostic["dataset"], "fixture")
        self.assertEqual(diagnostic["target_len"], 6)
        self.assertEqual(diagnostic["upsampled_rows"], 1)
        self.assertEqual(diagnostic["row_count"], 3)
        self.assertEqual(len(diagnostic["diagnostic_sha256"]), 64)

    def test_nonfinite_dataset_records_not_invoked_shape_evidence_without_resampling(self) -> None:
        """Exp450's dataset-wide nonfinite abstention must never enter Makima diagnostics."""
        values = _nonfinite_raw_fixture_antigravity()
        with patch.object(
            transforms.scipy.interpolate,
            "Akima1DInterpolator",
            wraps=scipy.interpolate.Akima1DInterpolator,
        ) as makima:
            diagnostic = transforms.dataset_shape_diagnostic_exp472_antigravity(values)
            count = transforms.count_upsampled_rows_exp472_antigravity(values)
        self.assertEqual(makima.call_count, 0)
        self.assertEqual(count, 0)
        self.assertEqual(diagnostic["schema"], "exp472-shape-diagnostic-antigravity.v2")
        self.assertEqual(diagnostic["alignment_status"], "not_invoked")
        self.assertEqual(diagnostic["alignment_reason"], "nonfinite_input")
        self.assertEqual((diagnostic["row_count"], diagnostic["aligned_row_count"], diagnostic["upsampled_rows"]), (2, 0, 0))
        self.assertEqual(diagnostic["train"]["row_count"], 1)
        self.assertEqual(diagnostic["train"]["aligned_row_count"], 0)
        self.assertEqual(diagnostic["train"]["upsampled_rows"], 0)
        self.assertEqual(diagnostic["train"]["overshoot"], 0.0)
        self.assertEqual(diagnostic["train"]["added_extrema"], 0)
        self.assertEqual(diagnostic["test"]["row_count"], 1)
        self.assertEqual(diagnostic["test"]["aligned_row_count"], 0)
        self.assertEqual(len(diagnostic["diagnostic_sha256"]), 64)
        with self.assertRaisesRegex(ValueError, "finite nonempty input"):
            transforms.resample_series_exp472_antigravity(values.train_rows[0], 6)

    def test_test_identity_is_deterministic_and_binds_test_values_only(self) -> None:
        """A test identity that omits blob order cannot bind a score vector safely."""
        values = _raw_fixture_antigravity()
        first = transforms.test_identity_exp472_antigravity(values)
        second = transforms.test_identity_exp472_antigravity(values)
        self.assertEqual(first, second)
        self.assertEqual(first["dataset"], "fixture")
        self.assertEqual(first["split"], "TEST")
        self.assertEqual(first["indices"], [0])
        self.assertEqual(
            first["values_blob_sha256"],
            [sha256(values.test_blobs[0]).hexdigest()],
        )
        self.assertEqual(len(first["row_digest"]), 64)

    def test_test_identity_rejects_a_blob_that_does_not_match_its_test_row(self) -> None:
        """A mismatched blob would bind score provenance to different values."""
        values = _raw_fixture_antigravity()
        mismatched = replace(values, test_blobs=(b"\x00\x00\x00\x00",))
        with self.assertRaisesRegex(ValueError, "raw dataset identity differs"):
            transforms.test_identity_exp472_antigravity(mismatched)

    def test_empty_nonfinite_and_nonpositive_inputs_are_rejected(self) -> None:
        """Accepting invalid inputs would allow a nonfinite score path to start."""
        for row, target in (
            (np.asarray([], dtype=np.float32), 3),
            (np.asarray([np.nan], dtype=np.float32), 3),
            (np.asarray([1.0], dtype=np.float32), 0),
        ):
            with self.subTest(row=row, target=target):
                with self.assertRaisesRegex(ValueError, "finite nonempty input and positive target"):
                    transforms.resample_series_exp472_antigravity(row, target)


if __name__ == "__main__":
    unittest.main()
