from __future__ import annotations

import inspect
import unittest

import numpy as np


class SourcesAntigravityTest(unittest.TestCase):
    def test_loader_exports_only_the_train_entrypoint_antigravity(self) -> None:
        """Break caught: Exp464 source module grows TEST or label entrypoints."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import (
            sources_antigravity as sources,
        )

        self.assertEqual(tuple(sources.__all__), ("load_normal2_train_antigravity",))
        self.assertEqual(
            tuple(inspect.signature(sources.load_normal2_train_antigravity).parameters),
            (),
        )
        self.assertFalse(hasattr(sources, "load_ecg_test_antigravity"))
        self.assertFalse(hasattr(sources, "load_labels_antigravity"))

    def test_loader_reuses_exp463_train_matrix_without_aliasing_antigravity(self) -> None:
        """Break caught: Exp464 source boundary changes bytes, dtype, ownership, or mutability."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity.sources_antigravity import (
            load_normal2_train_antigravity as load_exp463_train_antigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import (
            sources_antigravity,
        )

        first = sources_antigravity.load_normal2_train_antigravity()
        second = sources_antigravity.load_normal2_train_antigravity()
        exp463 = load_exp463_train_antigravity()

        np.testing.assert_array_equal(first, exp463)
        np.testing.assert_array_equal(second, exp463)
        self.assertEqual(first.shape, (1413, 140))
        self.assertEqual(first.dtype, np.dtype(np.float32))
        self.assertTrue(first.flags.owndata)
        self.assertTrue(first.flags.c_contiguous)
        self.assertFalse(first.flags.writeable)
        self.assertTrue(np.isfinite(first).all())
        self.assertFalse(np.shares_memory(first, second))
        self.assertFalse(np.shares_memory(first, exp463))
        self.assertFalse(np.shares_memory(second, exp463))
        self.assertEqual(sources_antigravity.TRAIN_LABEL_QUERY_COUNT_ANTIGRAVITY, 0)

    def test_loader_source_stays_train_only_and_label_free_antigravity(self) -> None:
        """Break caught: Exp464 loader directly queries private roots, TEST, or labels."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity import (
            sources_antigravity,
        )

        source = inspect.getsource(sources_antigravity)
        self.assertIn("exp463_ecg_prefix_ramp_gated_qualification_antigravity", source)
        for forbidden in (
            "sqlite3",
            "values_blob",
            "split = 'TEST'",
            "load_labels_antigravity",
            "official_result_root",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
