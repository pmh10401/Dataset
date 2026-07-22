from __future__ import annotations

import unittest

import numpy as np

from experiments.exp303_expanded_train_only_gates.gates import (
    choose_calibration_pair,
    choose_diversity_pair,
    representative_train_score,
)


def audit(risk: float, attainable: float, drift: float, stability: float) -> dict[str, object]:
    return {
        "available": True,
        "provenance_verified": True,
        "risk_upper_95": risk,
        "attainable_p": attainable,
        "train_drift": drift,
        "stability": stability,
    }


class ExpandedGateTests(unittest.TestCase):
    def test_representative_score_is_rank_based_and_has_train_length(self) -> None:
        score = representative_train_score(
            {
                "v1": np.asarray([10.0, 30.0, 20.0]),
                "v2": np.asarray([100.0, 300.0, 200.0]),
            }
        )
        np.testing.assert_allclose(score, [1.0, 3.0, 2.0])

    def test_diversity_gate_selects_lowest_absolute_train_correlation_pair(self) -> None:
        train = {
            "statistical": {"v": np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])},
            "structural": {"v": np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])},
            "change_point": {"v": np.asarray([3.0, 1.0, 5.0, 2.0, 4.0])},
            "official_multirocket": {"v": np.asarray([5.0, 4.0, 3.0, 2.0, 1.0])},
        }
        audits = {
            name: audit(0.01 + index * 0.01, 0.01, 0.1, 0.8)
            for index, name in enumerate(train)
        }

        selected = choose_diversity_pair(train, audits)

        self.assertEqual(selected, ["change_point", "statistical"])

    def test_calibration_gate_uses_all_four_train_audit_ranks(self) -> None:
        audits = {
            "statistical": audit(0.01, 0.02, 0.1, 0.9),
            "structural": audit(0.02, 0.01, 0.2, 0.8),
            "change_point": audit(0.30, 0.05, 0.8, 0.2),
            "official_multirocket": audit(0.20, 0.04, 0.7, 0.3),
        }

        selected = choose_calibration_pair(audits)

        self.assertEqual(selected, ["statistical", "structural"])

    def test_both_gates_abstain_when_fewer_than_two_experts_are_available(self) -> None:
        audits = {
            "statistical": audit(0.01, 0.01, 0.1, 0.9),
            "structural": {**audit(0.02, 0.01, 0.1, 0.8), "available": False},
        }
        train = {"statistical": {"v": np.asarray([1.0, 2.0, 3.0])}}

        self.assertEqual(choose_diversity_pair(train, audits), [])
        self.assertEqual(choose_calibration_pair(audits), [])


if __name__ == "__main__":
    unittest.main()
