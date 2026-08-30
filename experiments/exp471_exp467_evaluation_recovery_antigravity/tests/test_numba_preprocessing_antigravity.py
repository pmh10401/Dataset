"""Parity tests for candidate Exp471 Numba preprocessing kernels."""

from __future__ import annotations

import unittest

import numpy as np

from paper.equipment_anomaly_detection.sources_paper.low_frequency_paper import (
    _aspe_paper,
)

try:
    from experiments.exp471_exp467_evaluation_recovery_antigravity.numba_preprocessing_antigravity import (
        create_aspe_variants_antigravity,
        instrument_numba_aspe_antigravity,
    )
except ImportError:
    create_aspe_variants_antigravity = None
    instrument_numba_aspe_antigravity = None


class NumbaPreprocessingTestsAntigravity(unittest.TestCase):
    def test_all_variants_preserve_aspe_with_deterministic_parallel_results(self) -> None:
        """A fast variant must not move ASPE beyond the approved 1e-12 bound."""

        if create_aspe_variants_antigravity is None:
            self.fail("Exp471 Numba ASPE variants are missing")
        variants = create_aspe_variants_antigravity()
        self.assertEqual(
            tuple(variants),
            ("serial_safe", "parallel_safe", "parallel_fast"),
        )
        signals = (
            np.asarray(
                [0.0, 1.0, 1.0, 0.25, 0.5, 2.0, 0.0, 1.5, 0.75],
                dtype=np.float64,
            ),
            np.linspace(-3.0, 5.0, 257, dtype=np.float64),
            np.sin(np.arange(513, dtype=np.float64) / 7.0),
            np.ones(64, dtype=np.float64),
        )
        for signal in signals:
            for delay in (1, 2, 3, 4):
                expected = _aspe_paper(signal, 3, delay)
                for name, variant in variants.items():
                    observed = [variant(signal, 3, delay) for _ in range(3)]
                    with self.subTest(name=name, size=signal.size, delay=delay):
                        self.assertTrue(all(value == observed[0] for value in observed))
                        self.assertAlmostEqual(observed[0], expected, delta=1e-12)

    def test_instrumentation_restores_captured_aspe_after_failure(self) -> None:
        """A failed accelerated run must restore the frozen ASPE callable."""

        if instrument_numba_aspe_antigravity is None:
            self.fail("Exp471 Numba ASPE instrumentation is missing")
        from paper.equipment_anomaly_detection.sources_paper import (
            low_frequency_paper,
        )

        original = low_frequency_paper.score_low_frequency.__globals__["_aspe_paper"]
        variant = create_aspe_variants_antigravity()["serial_safe"]
        with self.assertRaisesRegex(RuntimeError, "public interruption"):
            with instrument_numba_aspe_antigravity(
                low_frequency_paper.score_low_frequency, variant
            ):
                self.assertIs(
                    low_frequency_paper.score_low_frequency.__globals__["_aspe_paper"],
                    variant,
                )
                raise RuntimeError("public interruption")
        self.assertIs(
            low_frequency_paper.score_low_frequency.__globals__["_aspe_paper"],
            original,
        )


if __name__ == "__main__":
    unittest.main()
