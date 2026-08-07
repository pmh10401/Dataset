import unittest

from experiments.exp438_frequency_complexity_evidence_antigravity.aspe_antigravity import (
    amplitude_sensitive_permutation_entropy_antigravity,
)
from experiments.exp438_frequency_complexity_evidence_antigravity.numba_antigravity import (
    NumbaAspeEngineAntigravity,
)


class Exp438NumbaAntigravityTests(unittest.TestCase):
    def test_injected_numba_engine_preserves_aspe_value_and_provenance(self) -> None:
        def kernel(values, embedding_dimension, delay):
            result = amplitude_sensitive_permutation_entropy_antigravity(
                values, embedding_dimension, delay
            )
            return result.value, result.flat_or_zero_weight

        engine = NumbaAspeEngineAntigravity(kernel, numba_version="test-numba")
        result = engine.compute([0.0, 2.0, 0.0, 2.0, 0.0], 3, 1)
        expected = amplitude_sensitive_permutation_entropy_antigravity(
            [0.0, 2.0, 0.0, 2.0, 0.0], 3, 1
        )
        self.assertEqual(result, expected)
        self.assertEqual(engine.backend, "numba")
        self.assertEqual(engine.numba_version, "test-numba")

    def test_nonfinite_numba_output_fails_closed(self) -> None:
        engine = NumbaAspeEngineAntigravity(
            lambda values, dimension, delay: (float("nan"), False),
            numba_version="test-numba",
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            engine.compute([0.0, 1.0, 0.0], 3, 1)


if __name__ == "__main__":
    unittest.main()
