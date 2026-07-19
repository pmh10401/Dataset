from __future__ import annotations

import inspect
import unittest

import numpy as np

from experiments.exp226_235_hard_alert_quality.exp230_seed_stability import (
    stability_support,
)
from experiments.exp226_235_hard_alert_quality.exp231_normal_mode_calibration import (
    fit_normal_modes,
    local_empirical_pvalues,
)
from experiments.exp226_235_hard_alert_quality.exp232_calibration_reliability_gate import (
    reliability_state,
)
from experiments.exp226_235_hard_alert_quality.exp233_conformal_fdr import bh_indices


class Exp230Tests(unittest.TestCase):
    def test_stability_support_is_input_order_invariant(self) -> None:
        first = stability_support({"s2": {1, 2}, "s1": {1, 3}, "s3": {1}})
        second = stability_support({"s3": {1}, "s1": {1, 3}, "s2": {1, 2}})
        self.assertEqual(first, second)
        self.assertEqual(first, {1: 3, 2: 1, 3: 1})


class Exp231Tests(unittest.TestCase):
    def test_normal_mode_fit_is_deterministic(self) -> None:
        values = np.asarray([[0.0], [0.1], [9.9], [10.0]])
        one = fit_normal_modes(values, clusters=2, seed=17)
        two = fit_normal_modes(values, clusters=2, seed=17)
        np.testing.assert_allclose(one.centers, two.centers)
        np.testing.assert_array_equal(one.assignments, two.assignments)

    def test_local_pvalues_use_assigned_train_mode(self) -> None:
        train = np.asarray([0.0, 0.1, 10.0, 10.1])
        assignments = np.asarray([0, 0, 1, 1])
        query = np.asarray([0.2, 10.2])
        query_modes = np.asarray([0, 1])
        actual = local_empirical_pvalues(train, assignments, query, query_modes)
        np.testing.assert_allclose(actual, [1.0 / 3.0, 1.0 / 3.0])


class Exp232Tests(unittest.TestCase):
    def test_reliability_api_has_no_test_or_family_inputs(self) -> None:
        parameters = set(inspect.signature(reliability_state).parameters)
        self.assertTrue({"n_test", "family", "dataset", "test_length"}.isdisjoint(parameters))

    def test_reliability_states(self) -> None:
        self.assertEqual(
            reliability_state(
                n_train=100,
                attainable_p=0.01,
                risk_upper=0.002,
                source_stability=0.9,
                train_drift=0.05,
            ),
            "eligible",
        )
        self.assertEqual(
            reliability_state(
                n_train=20,
                attainable_p=0.05,
                risk_upper=0.02,
                source_stability=0.4,
                train_drift=0.5,
            ),
            "review_only",
        )
        self.assertEqual(
            reliability_state(
                n_train=4,
                attainable_p=0.2,
                risk_upper=0.2,
                source_stability=0.0,
                train_drift=2.0,
            ),
            "abstain",
        )


class Exp233Tests(unittest.TestCase):
    def test_bh_rejections_are_monotone(self) -> None:
        pvalues = np.asarray([0.001, 0.01, 0.2, 0.8])
        strict = bh_indices(pvalues, q=0.05)
        relaxed = bh_indices(pvalues, q=0.10)
        self.assertEqual(strict, {0, 1})
        self.assertGreaterEqual(relaxed, strict)


if __name__ == "__main__":
    unittest.main()
