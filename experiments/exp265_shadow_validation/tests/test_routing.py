from __future__ import annotations

import unittest

from experiments.exp265_shadow_validation.config import SOURCE_NAMES
from experiments.exp265_shadow_validation.contracts import validate_run_input
from experiments.exp265_shadow_validation.routing import route_shadow_run
from experiments.exp265_shadow_validation.tests.test_contracts import valid_run_input


def routed(**changes: object) -> dict[str, object]:
    raw = valid_run_input()
    raw.update(changes)
    return route_shadow_run(validate_run_input(raw))


class RoutingTest(unittest.TestCase):
    def test_hard_is_preserved_and_gets_confidence_grade(self) -> None:
        result = routed(
            baseline_hard_indices=[0, 2, 4],
            baseline_standard_indices=[1, 3],
            source_evidence={
                "statistical": {"candidate_indices": [0, 2], "scores": {"0": 3.0, "2": 2.5}, "threshold": 2.0},
                "structural": {"candidate_indices": [0], "scores": {"0": 4.0}, "threshold": 3.0},
                "change_point": {"candidate_indices": [0], "scores": {"0": 1.2}, "threshold": 1.0},
                "reconstruction": {"candidate_indices": [], "scores": {}, "threshold": 0.5},
            },
        )
        self.assertEqual(result["hard_indices"], [0, 2, 4])
        self.assertEqual(result["baseline_hard_indices"], result["hard_indices"])
        self.assertEqual(result["confidence"], {"0": "A", "2": "B", "4": "C"})

    def test_shadow_never_enters_hard(self) -> None:
        evidence = {
            name: {"candidate_indices": [1], "scores": {"1": 2.0}, "threshold": 1.0}
            for name in SOURCE_NAMES
        }
        result = routed(baseline_hard_indices=[], baseline_standard_indices=[1], source_evidence=evidence)
        self.assertEqual(result["shadow_indices"], [1])
        self.assertEqual(result["hard_indices"], [])

    def test_small_train_three_source_candidate_is_priority(self) -> None:
        evidence = {
            name: {"candidate_indices": [1] if name != "reconstruction" else [], "scores": {"1": 2.0} if name != "reconstruction" else {}, "threshold": 1.0}
            for name in SOURCE_NAMES
        }
        result = routed(n_train=99, baseline_standard_indices=[1], source_evidence=evidence)
        self.assertEqual(result["priority_indices"], [1])
        self.assertEqual(result["shadow_indices"], [])

    def test_quality_blocked_candidate_is_priority(self) -> None:
        evidence = {
            name: {"candidate_indices": [1], "scores": {"1": 2.0}, "threshold": 1.0}
            for name in SOURCE_NAMES
        }
        result = routed(baseline_standard_indices=[1], source_evidence=evidence, quality_flag_indices=[1])
        self.assertEqual(result["priority_indices"], [1])
        self.assertEqual(result["shadow_indices"], [])

    def test_zero_and_one_source_candidates_remain_standard(self) -> None:
        result = routed()
        self.assertEqual(result["standard_indices"], [2, 3])
        self.assertEqual(result["shadow_indices"], [1])

    def test_candidate_free_run_is_valid(self) -> None:
        evidence = {name: {"candidate_indices": [], "scores": {}, "threshold": 1.0} for name in SOURCE_NAMES}
        result = routed(
            baseline_hard_indices=[], baseline_standard_indices=[], baseline_priority_indices=[],
            source_evidence=evidence, series_preview={"indices": [0, 1, 2, 3, 4], "values": [0, 0, 0, 0, 0]},
        )
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["candidate_count"], 0)


if __name__ == "__main__":
    unittest.main()
