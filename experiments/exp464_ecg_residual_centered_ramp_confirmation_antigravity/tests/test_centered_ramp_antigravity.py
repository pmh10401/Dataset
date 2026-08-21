from __future__ import annotations

import unittest

import numpy as np


def independent_centered_ramp_reference_antigravity(
    values: np.ndarray,
    template: np.ndarray,
) -> tuple[float, float]:
    residual = np.asarray(values, dtype=np.float64) - np.asarray(
        template,
        dtype=np.float64,
    )
    suffix = residual[56:]
    centered = suffix - float(np.mean(suffix))
    t = np.linspace(0.0, 1.0, 84, dtype=np.float64)
    basis = t - float(np.mean(t))
    basis /= float(np.linalg.norm(basis))
    amplitude = float(np.dot(centered, basis))
    norm = float(np.linalg.norm(centered))
    cosine = 0.0 if norm == 0.0 else amplitude / norm
    return amplitude, cosine


def independent_centered_suffix_basis_antigravity() -> np.ndarray:
    t = np.linspace(0.0, 1.0, 84, dtype=np.float64)
    basis = t - float(np.mean(t))
    basis /= float(np.linalg.norm(basis))
    return basis


class CenteredRampAntigravityTest(unittest.TestCase):
    def test_centered_ramp_matches_exact_numpy_reference_and_basis_contract_antigravity(
        self,
    ) -> None:
        """Break caught: residual, centering, onset, basis, or cosine drift."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.centered_ramp_antigravity import (
            _scoring_basis_antigravity,
            centered_ramp_evidence_antigravity,
            centered_suffix_basis_antigravity,
        )

        time = np.linspace(-1.5, 2.0, 140, dtype=np.float64)
        values = (
            0.3 * np.sin(2.2 * time)
            + 0.05 * time * time
            + np.where(np.arange(140) >= 56, 0.7 * (np.arange(140) - 56) / 83.0, 0.0)
            + 0.2
        ).astype(np.float32)
        template = (
            0.1 * np.cos(1.7 * time)
            + 0.02 * time
            + np.where(np.arange(140) >= 56, 0.15 * (np.arange(140) - 56) / 83.0, 0.0)
            - 0.05
        ).astype(np.float32)
        original_values = np.array(values, copy=True)
        original_template = np.array(template, copy=True)
        expected_amplitude, expected_cosine = independent_centered_ramp_reference_antigravity(
            values,
            template,
        )

        actual = centered_ramp_evidence_antigravity(values, template)
        basis = centered_suffix_basis_antigravity()
        expected_basis = independent_centered_suffix_basis_antigravity()

        self.assertEqual(actual.amplitude, expected_amplitude)
        self.assertEqual(actual.cosine, expected_cosine)
        np.testing.assert_array_equal(basis, expected_basis)
        np.testing.assert_array_equal(_scoring_basis_antigravity(), expected_basis)
        self.assertEqual(float(np.sum(basis)), float(np.sum(expected_basis)))
        self.assertEqual(basis.shape, (84,))
        self.assertFalse(basis.flags.writeable)
        self.assertTrue(basis.flags.owndata)
        np.testing.assert_array_equal(values, original_values)
        np.testing.assert_array_equal(template, original_template)

    def test_centered_ramp_constant_height_shift_returns_zero_evidence_antigravity(self) -> None:
        """Break caught: non-centered suffix leaks constant shifts into the score."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.centered_ramp_antigravity import (
            centered_ramp_evidence_antigravity,
        )

        template = np.linspace(-0.5, 0.5, 140, dtype=np.float64)
        values = template + 3.25

        actual = centered_ramp_evidence_antigravity(values, template)

        self.assertEqual(actual.amplitude, 0.0)
        self.assertEqual(actual.cosine, 0.0)

    def test_centered_ramp_perfect_positive_ramp_keeps_raw_formula_cosine_antigravity(self) -> None:
        """Break caught: positive-ramp cosine is snapped instead of matching the frozen formula."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.centered_ramp_antigravity import (
            centered_ramp_evidence_antigravity,
        )

        template = np.zeros(140, dtype=np.float64)
        values = np.zeros(140, dtype=np.float64)
        values[56:] = np.linspace(-2.0, 4.0, 84, dtype=np.float64)

        actual = centered_ramp_evidence_antigravity(values, template)
        expected_amplitude, expected_cosine = independent_centered_ramp_reference_antigravity(
            values,
            template,
        )

        self.assertEqual(actual.amplitude, expected_amplitude)
        self.assertEqual(actual.cosine, expected_cosine)
        self.assertEqual(expected_cosine, 0.9999999999999998)

    def test_centered_ramp_descending_ramp_keeps_raw_formula_cosine_antigravity(self) -> None:
        """Break caught: descending-ramp cosine is snapped instead of matching the frozen formula."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.centered_ramp_antigravity import (
            centered_ramp_evidence_antigravity,
        )

        template = np.zeros(140, dtype=np.float64)
        values = np.zeros(140, dtype=np.float64)
        values[56:] = np.linspace(4.0, -2.0, 84, dtype=np.float64)

        actual = centered_ramp_evidence_antigravity(values, template)
        expected_amplitude, expected_cosine = independent_centered_ramp_reference_antigravity(
            values,
            template,
        )

        self.assertEqual(actual.amplitude, expected_amplitude)
        self.assertEqual(actual.cosine, expected_cosine)

    def test_centered_ramp_arbitrary_probe_uses_public_basis_exactly_antigravity(self) -> None:
        """Break caught: public basis bytes drift from the basis used for scoring."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.centered_ramp_antigravity import (
            centered_ramp_evidence_antigravity,
            centered_suffix_basis_antigravity,
        )

        prefix = np.linspace(-0.2, 0.2, 56, dtype=np.float64)
        suffix = (
            0.3 * np.sin(np.linspace(0.0, 4.0 * np.pi, 84, dtype=np.float64))
            + np.linspace(-0.4, 0.6, 84, dtype=np.float64)
        )
        values = np.concatenate((prefix, suffix))
        template = np.linspace(0.1, -0.1, 140, dtype=np.float64)
        residual = values - template
        centered = residual[56:] - float(np.mean(residual[56:]))
        public_basis = centered_suffix_basis_antigravity()

        actual = centered_ramp_evidence_antigravity(values, template)

        self.assertEqual(actual.amplitude, float(np.dot(centered, public_basis)))
        self.assertEqual(
            actual.cosine,
            independent_centered_ramp_reference_antigravity(values, template)[1],
        )

    def test_centered_ramp_zero_vector_has_zero_cosine_antigravity(self) -> None:
        """Break caught: zero-energy suffix rows produce NaN or an arbitrary sign."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.centered_ramp_antigravity import (
            centered_ramp_evidence_antigravity,
        )

        values = np.zeros(140, dtype=np.float64)
        template = np.zeros(140, dtype=np.float64)

        actual = centered_ramp_evidence_antigravity(values, template)

        self.assertEqual(actual.amplitude, 0.0)
        self.assertEqual(actual.cosine, 0.0)


if __name__ == "__main__":
    unittest.main()
