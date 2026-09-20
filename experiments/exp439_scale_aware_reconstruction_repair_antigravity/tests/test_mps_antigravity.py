import unittest

import numpy as np

from experiments.exp439_scale_aware_reconstruction_repair_antigravity.mps_antigravity import (
    MpsDenseEngineAntigravity,
)


class Exp439MpsAntigravityTests(unittest.TestCase):
    def test_injected_engine_validates_shape_finiteness_and_metadata(self) -> None:
        engine = MpsDenseEngineAntigravity(
            lambda queries, references, k, batch_size: np.arange(len(queries), dtype=np.float32),
            pytorch_version="2.0",
            device="mps",
        )
        result = engine.score_batch(np.ones((3, 2)), np.ones((4, 2)), 3, batch_size=16)
        np.testing.assert_array_equal(result, [0.0, 1.0, 2.0])
        self.assertEqual(engine.backend, "mps")
        self.assertEqual(engine.device, "mps")
        self.assertFalse(result.flags.writeable)

    def test_bad_output_finiteness_fails_closed(self) -> None:
        engine = MpsDenseEngineAntigravity(
            lambda *args, **kwargs: [0.0, float("inf")], "2", "mps"
        )
        with self.assertRaisesRegex(ValueError, "output"):
            engine.score_batch(np.ones((2, 2)), np.ones((3, 2)), 3, batch_size=16)


if __name__ == "__main__":
    unittest.main()
