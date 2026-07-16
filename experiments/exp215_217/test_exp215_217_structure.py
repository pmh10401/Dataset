from __future__ import annotations

import inspect
import unittest

from experiments.exp215_217 import core


class ShapeletStabilityStructureTests(unittest.TestCase):
    def test_shape_scoring_does_not_accept_test_labels(self) -> None:
        self.assertNotIn("y_test", inspect.signature(core.fit_shape_source).parameters)

    def test_fixed_dictionary_has_no_input_data_parameter(self) -> None:
        self.assertEqual(list(inspect.signature(core.fixed_kernel_dictionary).parameters), [])


if __name__ == "__main__":
    unittest.main()
