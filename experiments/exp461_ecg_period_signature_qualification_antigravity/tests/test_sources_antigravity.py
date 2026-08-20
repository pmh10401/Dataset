from __future__ import annotations

import inspect
import unittest

import numpy as np


class SourcesAntigravityTest(unittest.TestCase):
    def test_loader_exports_only_the_exact_train_entrypoint_antigravity(self) -> None:
        from experiments.exp461_ecg_period_signature_qualification_antigravity import (
            sources_antigravity as sources,
        )

        self.assertEqual(tuple(sources.__all__), ("load_normal2_train_antigravity",))
        self.assertEqual(
            tuple(inspect.signature(sources.load_normal2_train_antigravity).parameters),
            (),
        )
        self.assertFalse(hasattr(sources, "load_ecg_test_antigravity"))
        self.assertFalse(hasattr(sources, "load_labels_antigravity"))

    def test_loader_replays_the_exact_exp459_train_matrix_without_aliasing_antigravity(
        self,
    ) -> None:
        from experiments.exp459_ecg_normal2_blocked_validation_antigravity.sources_antigravity import (
            load_normal2_train_antigravity as load_exp459_train_antigravity,
        )
        from experiments.exp461_ecg_period_signature_qualification_antigravity.sources_antigravity import (
            load_normal2_train_antigravity,
        )

        first = load_normal2_train_antigravity()
        second = load_normal2_train_antigravity()
        exp459 = load_exp459_train_antigravity()

        np.testing.assert_array_equal(first, exp459)
        np.testing.assert_array_equal(second, exp459)
        self.assertEqual(first.shape, (1413, 140))
        self.assertEqual(first.dtype, np.dtype(np.float32))
        self.assertTrue(first.flags.owndata)
        self.assertTrue(first.flags.c_contiguous)
        self.assertFalse(first.flags.writeable)
        self.assertTrue(np.isfinite(first).all())
        self.assertFalse(np.shares_memory(first, second))
        self.assertFalse(np.shares_memory(first, exp459))
        self.assertFalse(np.shares_memory(second, exp459))

    def test_loader_source_stays_train_only_and_label_free_antigravity(self) -> None:
        from experiments.exp461_ecg_period_signature_qualification_antigravity import (
            sources_antigravity,
        )

        source = inspect.getsource(sources_antigravity)
        self.assertIn("exp459_ecg_normal2_blocked_validation_antigravity", source)
        for forbidden in (
            "sqlite3",
            "values_blob",
            "split = 'TEST'",
            "label",
            "labels",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
