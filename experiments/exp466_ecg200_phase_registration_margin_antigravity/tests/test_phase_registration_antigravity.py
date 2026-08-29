from __future__ import annotations

import dataclasses
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp466_ecg200_phase_registration_margin_antigravity.phase_registration_antigravity import (
    PhaseScoresAntigravity,
    phase_registration_margin_antigravity,
    score_phase_registration_antigravity,
)


class PhaseRegistrationAntigravityTest(unittest.TestCase):
    @staticmethod
    def _base_antigravity() -> np.ndarray:
        x = np.arange(140, dtype=np.float64)
        return (
            0.01 * x
            + np.sin(2.0 * np.pi * x / 21.0)
            + 0.3 * np.cos(2.0 * np.pi * x / 9.0)
        )

    def test_fixed_inverse_warp_margin_and_affine_invariance_antigravity(self) -> None:
        """Break caught: P9 changes its fixed 56/84/1.25/67 formula or normalization."""
        template = self._base_antigravity()
        normal = template.copy()
        warped = template.copy()
        warped[56:] = np.interp(
            np.arange(84, dtype=np.float64) / 1.25,
            np.arange(84, dtype=np.float64),
            template[56:],
        )

        normal_margin = phase_registration_margin_antigravity(normal, template)
        warped_margin = phase_registration_margin_antigravity(warped, template)
        self.assertLessEqual(normal_margin, 0.0)
        self.assertGreater(warped_margin, 0.0)

        transformed_margin = phase_registration_margin_antigravity(
            7.25 + 2.5 * warped,
            -3.5 + 0.75 * template,
        )
        self.assertAlmostEqual(warped_margin, transformed_margin, delta=1e-12)

        query_suffix = warped[56:]
        query_median = np.median(query_suffix)
        query_scale = 1.4826 * np.median(np.abs(query_suffix - query_median))
        query_standardized = (query_suffix - query_median) / query_scale
        template_suffix = template[56:]
        template_median = np.median(template_suffix)
        template_scale = 1.4826 * np.median(
            np.abs(template_suffix - template_median),
        )
        template_standardized = (template_suffix - template_median) / template_scale
        shared = np.arange(67, dtype=np.float64)
        identity_error = float(
            np.mean((query_standardized[:67] - template_standardized[:67]) ** 2),
        )
        undone = np.interp(
            1.25 * shared,
            np.arange(84, dtype=np.float64),
            query_standardized,
        )
        undo_error = float(
            np.mean((undone - template_standardized[:67]) ** 2),
        )
        self.assertAlmostEqual(
            warped_margin,
            identity_error - undo_error,
            delta=1e-15,
        )

    def test_score_uses_loo_fit_full_fit_query_strict_confirmation_and_exact_rank_antigravity(
        self,
    ) -> None:
        """Break caught: calibration leaks a fit row or changes rank/strict confirmation."""
        x = np.arange(140, dtype=np.float64)
        base = self._base_antigravity()
        fit_rows = np.stack(
            (
                base,
                base + 0.03 * np.sin(2.0 * np.pi * x / 13.0),
                base + 0.04 * np.cos(2.0 * np.pi * x / 17.0),
                base + 0.02 * np.sin(2.0 * np.pi * x / 7.0),
            ),
        )
        normal = base + 0.015 * np.cos(2.0 * np.pi * x / 11.0)
        warped = base.copy()
        warped[56:] = np.interp(
            np.arange(84, dtype=np.float64) / 1.25,
            np.arange(84, dtype=np.float64),
            base[56:],
        )
        query_rows = np.stack((normal, warped))

        scores = score_phase_registration_antigravity(fit_rows, query_rows)

        np.testing.assert_allclose(
            scores.fit_margins,
            np.array(
                (
                    -1.1803652644198817,
                    -1.1902612785807918,
                    -1.1691030154695568,
                    -1.1866388604308464,
                ),
                dtype=np.float64,
            ),
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            scores.query_margins,
            np.array((-1.197118344225737, 1.2835243752805767), dtype=np.float64),
            rtol=0.0,
            atol=1e-15,
        )
        self.assertEqual(scores.fit_normal_max_margin, -1.1691030154695568)
        np.testing.assert_array_equal(
            scores.ranks,
            np.array((0.2, 1.0), dtype=np.float64),
        )
        np.testing.assert_array_equal(
            scores.confirmed,
            np.array((False, True), dtype=np.bool_),
        )

    def test_margin_rejects_wrong_geometry_dtype_nonfinite_and_zero_mad_antigravity(
        self,
    ) -> None:
        """Break caught: malformed or scale-free suffixes enter the phase formula."""
        valid = self._base_antigravity()
        invalid_arrays = (
            valid.astype(np.float32),
            valid.astype(np.int64),
            valid[:-1],
            valid.reshape(1, 140),
            np.concatenate((valid[:-1], np.array((np.nan,), dtype=np.float64))),
            np.ones(140, dtype=np.float64),
        )
        for invalid in invalid_arrays:
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype):
                with self.assertRaises(ValueError):
                    phase_registration_margin_antigravity(invalid, valid)
        with self.assertRaises(ValueError):
            phase_registration_margin_antigravity(valid, np.ones(140, dtype=np.float64))
        with self.assertRaises(TypeError):
            phase_registration_margin_antigravity(list(valid), valid)  # type: ignore[arg-type]

    def test_score_rejects_invalid_matrices_and_returns_owned_read_only_outputs_antigravity(
        self,
    ) -> None:
        """Break caught: score matrices bypass boundaries or outputs alias mutable storage."""
        base = self._base_antigravity()
        fit_rows = np.stack((base, base + 0.02 * np.sin(np.arange(140))))
        query_rows = np.stack((base, base.copy()))
        scores = score_phase_registration_antigravity(fit_rows, query_rows)

        fields = (
            (scores.fit_margins, np.dtype(np.float64), len(fit_rows)),
            (scores.query_margins, np.dtype(np.float64), len(query_rows)),
            (scores.ranks, np.dtype(np.float64), len(query_rows)),
            (scores.confirmed, np.dtype(np.bool_), len(query_rows)),
        )
        for values, dtype, length in fields:
            self.assertEqual(values.dtype, dtype)
            self.assertEqual(values.shape, (length,))
            self.assertTrue(values.flags.owndata)
            self.assertTrue(values.flags.c_contiguous)
            self.assertFalse(values.flags.writeable)
            self.assertFalse(np.shares_memory(values, fit_rows))
            self.assertFalse(np.shares_memory(values, query_rows))
            with self.assertRaises(ValueError):
                values[0] = values[0]

        invalid_pairs = (
            (fit_rows.astype(np.float32), query_rows),
            (fit_rows[:, :-1], query_rows),
            (fit_rows.reshape(2, 1, 140), query_rows),
            (fit_rows[:1], query_rows),
            (fit_rows, query_rows.astype(np.int64)),
            (fit_rows, query_rows[:, :-1]),
            (fit_rows, query_rows[:0]),
        )
        for bad_fit, bad_query in invalid_pairs:
            with self.subTest(fit_shape=bad_fit.shape, query_shape=bad_query.shape):
                with self.assertRaises(ValueError):
                    score_phase_registration_antigravity(bad_fit, bad_query)
        nonfinite = fit_rows.copy()
        nonfinite[0, 56] = np.inf
        with self.assertRaises(ValueError):
            score_phase_registration_antigravity(nonfinite, query_rows)
        with self.assertRaises(TypeError):
            score_phase_registration_antigravity(fit_rows.tolist(), query_rows)  # type: ignore[arg-type]

    def test_phase_scores_freezes_inputs_and_enforces_derived_fields_antigravity(self) -> None:
        """Break caught: the result contract accepts aliases or forged rank/threshold state."""
        fit = np.array((0.4, 0.5), dtype=np.float64)
        query = np.array((0.25, 0.5, 0.75, -1.0), dtype=np.float64)
        ranks = np.array((1.0 / 3.0, 1.0, 1.0, 1.0 / 3.0), dtype=np.float64)
        confirmed = np.array((False, False, True, False), dtype=np.bool_)
        scores = PhaseScoresAntigravity(
            fit_margins=fit,
            query_margins=query,
            ranks=ranks,
            fit_normal_max_margin=0.5,
            confirmed=confirmed,
        )

        for source, frozen in (
            (fit, scores.fit_margins),
            (query, scores.query_margins),
            (ranks, scores.ranks),
            (confirmed, scores.confirmed),
        ):
            self.assertFalse(np.shares_memory(source, frozen))
            self.assertTrue(frozen.flags.owndata)
            self.assertFalse(frozen.flags.writeable)

        with self.assertRaises(ValueError):
            dataclasses.replace(scores, fit_normal_max_margin=0.4)
        forged = scores.confirmed.copy()
        forged.setflags(write=True)
        forged[1] = True
        with self.assertRaises(ValueError):
            dataclasses.replace(scores, confirmed=forged)
        with self.assertRaises(ValueError):
            dataclasses.replace(scores, ranks=np.zeros(4, dtype=np.float64))
        with self.assertRaises(ValueError):
            dataclasses.replace(scores, fit_margins=fit.astype(np.float32))
        with self.assertRaises(ValueError):
            dataclasses.replace(
                scores,
                query_margins=np.array((0.25, np.nan, 0.75, -1.0)),
            )

    def test_zero_margin_is_not_confirmed_above_negative_fit_max_antigravity(
        self,
    ) -> None:
        """Break caught: deleting the positive-margin conjunct confirms exact zero."""
        base = self._base_antigravity()
        fit_rows = np.stack((base, base + 0.01 * np.sin(np.arange(140))))
        query_rows = np.array(base[None, :], dtype=np.float64, copy=True)
        margin_source = (
            "experiments.exp466_ecg200_phase_registration_margin_antigravity."
            "phase_registration_antigravity.phase_registration_margin_antigravity"
        )

        with patch(margin_source, side_effect=(-2.0, -1.0, 0.0)):
            scores = score_phase_registration_antigravity(fit_rows, query_rows)

        self.assertEqual(scores.query_margins[0], 0.0)
        self.assertEqual(scores.fit_normal_max_margin, -1.0)
        self.assertGreater(0.0, scores.fit_normal_max_margin)
        self.assertFalse(scores.confirmed[0])


if __name__ == "__main__":
    unittest.main()
