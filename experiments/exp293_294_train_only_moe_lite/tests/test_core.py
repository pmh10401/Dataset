from __future__ import annotations

import inspect
import unittest
import warnings

import numpy as np

from experiments.exp293_294_train_only_moe_lite.core import (
    audit_expert,
    family_majority_loo_false_count,
)
from experiments.exp293_294_train_only_moe_lite.routing import (
    gate_experts,
    rank_standard_candidates,
)


class TrainOnlyAuditTests(unittest.TestCase):
    def test_family_loo_calibration_excludes_query_score(self) -> None:
        variants = {
            "a": np.asarray([0.0, 1.0, 2.0, 100.0]),
            "b": np.asarray([0.0, 1.0, 2.0, 100.0]),
        }
        self.assertEqual(family_majority_loo_false_count(variants, alpha=0.25), 1)

    def test_audit_uses_train_arrays_only(self) -> None:
        parameters = set(inspect.signature(audit_expert).parameters)
        self.assertNotIn("test_scores", parameters)
        self.assertNotIn("y_test", parameters)
        row = audit_expert(
            dataset="demo",
            expert="statistical",
            train_variants={"a": np.arange(10.0), "b": np.arange(10.0)},
            provenance_verified=True,
            alpha=0.005,
        )
        self.assertEqual(row["n_train"], 10)
        self.assertEqual(row["normal_false_count"], 0)
        self.assertTrue(row["contains_test_values"] is False)

    def test_constant_train_variants_do_not_emit_correlation_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            row = audit_expert(
                dataset="constant",
                expert="structural",
                train_variants={"a": np.ones(10), "b": np.ones(10)},
                provenance_verified=True,
                alpha=0.005,
            )
        self.assertEqual(row["stability"], 0.0)


class SparseGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.audit = {
            "statistical": {"available": True, "provenance_verified": True, "risk_upper_95": 0.02, "stability": 0.9, "attainable_p": 0.01},
            "structural": {"available": True, "provenance_verified": True, "risk_upper_95": 0.01, "stability": 0.5, "attainable_p": 0.01},
            "change_point": {"available": True, "provenance_verified": True, "risk_upper_95": 0.03, "stability": 0.8, "attainable_p": 0.01},
            "official_multirocket": {"available": True, "provenance_verified": True, "risk_upper_95": 0.04, "stability": 0.95, "attainable_p": 0.01},
            "masked_reconstruction": {"available": True, "provenance_verified": False, "risk_upper_95": 0.0, "stability": 1.0, "attainable_p": 0.001},
        }

    def test_gate_has_no_test_or_family_policy_inputs(self) -> None:
        parameters = set(inspect.signature(gate_experts).parameters)
        self.assertNotIn("n_test", parameters)
        self.assertNotIn("family", parameters)
        self.assertNotIn("y_test", parameters)

    def test_gate_excludes_unverified_expert_from_primary_variants(self) -> None:
        ordered = gate_experts(self.audit, variant="risk_first_top2")
        self.assertEqual(ordered[:2], ["structural", "statistical"])
        self.assertNotIn("masked_reconstruction", ordered)

    def test_candidate_ranking_preserves_standard_candidate_set(self) -> None:
        ranked = rank_standard_candidates(
            standard_indices=[1, 2, 3],
            support_by_index={"1": ["structural"], "2": ["statistical", "change_point"], "3": []},
            ordered_experts=["structural", "statistical", "change_point", "official_multirocket"],
        )
        self.assertEqual(set(ranked), {1, 2, 3})
        self.assertEqual(ranked[0], 2)


if __name__ == "__main__":
    unittest.main()
