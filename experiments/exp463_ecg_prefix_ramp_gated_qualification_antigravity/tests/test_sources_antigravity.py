from __future__ import annotations

import inspect
import unittest

import numpy as np


class SourcesAntigravityTest(unittest.TestCase):
    def test_loader_exports_only_the_train_entrypoint_antigravity(self) -> None:
        """Break caught: source module grows private-data or label entrypoints."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity import sources_antigravity as sources

        self.assertEqual(tuple(sources.__all__), ("load_normal2_train_antigravity",))
        self.assertEqual(tuple(inspect.signature(sources.load_normal2_train_antigravity).parameters), ())
        self.assertFalse(hasattr(sources, "load_ecg_test_antigravity"))
        self.assertFalse(hasattr(sources, "load_labels_antigravity"))

    def test_loader_returns_owned_finite_read_only_train_with_label_count_zero_antigravity(self) -> None:
        """Break caught: source boundary changes shape, bytes, dtype, ownership, or mutability."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity import sources_antigravity

        first = sources_antigravity.load_normal2_train_antigravity()
        second = sources_antigravity.load_normal2_train_antigravity()
        self.assertEqual(first.shape, (1413, 140))
        self.assertEqual(first.dtype, np.dtype(np.float32))
        self.assertTrue(first.flags.owndata)
        self.assertTrue(first.flags.c_contiguous)
        self.assertFalse(first.flags.writeable)
        self.assertTrue(np.isfinite(first).all())
        self.assertFalse(np.shares_memory(first, second))
        self.assertEqual(sources_antigravity.TRAIN_LABEL_QUERY_COUNT_ANTIGRAVITY, 0)

    def test_loader_source_stays_train_only_and_label_free_antigravity(self) -> None:
        """Break caught: loader directly queries SQLite, private split, or labels."""
        from experiments.exp463_ecg_prefix_ramp_gated_qualification_antigravity import sources_antigravity

        source = inspect.getsource(sources_antigravity)
        self.assertIn("exp462_ecg_gradual_signature_qualification_antigravity", source)
        for forbidden in ("sqlite3", "values_blob", "split = 'TEST'", "label", "labels"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
