import unittest

import numpy as np

from experiments.exp439_scale_aware_reconstruction_repair_antigravity.mlx_antigravity import (
    MlxDenseEngineAntigravity,
)


class Exp439MlxAntigravityTests(unittest.TestCase):
    def test_injected_engine_validates_shape_finiteness_and_metadata(self) -> None:
        engine = MlxDenseEngineAntigravity(
            lambda queries, references, k, batch_size: np.arange(len(queries), dtype=np.float32),
            mlx_version="1.0",
            mlx_metal_version="2.0",
            device="metal-gpu",
        )
        result = engine.score_batch(np.ones((3, 2)), np.ones((4, 2)), 3, batch_size=16)
        np.testing.assert_array_equal(result, [0.0, 1.0, 2.0])
        self.assertEqual(engine.backend, "mlx")
        self.assertEqual(engine.device, "metal-gpu")
        self.assertFalse(result.flags.writeable)

    def test_bad_output_shape_fails_closed(self) -> None:
        engine = MlxDenseEngineAntigravity(
            lambda *args, **kwargs: [1.0], "1", "2", "metal-gpu"
        )
        with self.assertRaisesRegex(ValueError, "output"):
            engine.score_batch(np.ones((2, 2)), np.ones((3, 2)), 3, batch_size=16)


if __name__ == "__main__":
    unittest.main()
