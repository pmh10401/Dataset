from __future__ import annotations

import unittest

from experiments.exp265_shadow_validation.contracts import (
    ContractError,
    prediction_sha256,
    validate_run_input,
)


def valid_run_input() -> dict[str, object]:
    return {
        "schema_version": 1,
        "collection_mode": "prospective_shadow",
        "policy_version": "exp265-shadow-v1",
        "run_id": "run-001",
        "source_id": "equipment-a",
        "observed_at": "2026-07-21T00:00:00+00:00",
        "dataset_key": "recipe-a/sensor-1/step-2",
        "equipment_id": "eq-a",
        "recipe_id": "recipe-a",
        "sensor_id": "sensor-1",
        "step_id": "step-2",
        "n_train": 120,
        "n_test": 5,
        "baseline_hard_indices": [0],
        "baseline_standard_indices": [1, 2, 3],
        "baseline_priority_indices": [],
        "source_evidence": {
            "statistical": {"candidate_indices": [0, 1], "scores": {"0": 3.2, "1": 2.8}, "threshold": 2.0},
            "structural": {"candidate_indices": [0, 1], "scores": {"0": 4.2, "1": 3.8}, "threshold": 3.0},
            "change_point": {"candidate_indices": [0, 1], "scores": {"0": 1.2, "1": 1.1}, "threshold": 1.0},
            "reconstruction": {"candidate_indices": [], "scores": {}, "threshold": 0.5},
        },
        "quality_flag_indices": [],
        "has_missing": False,
        "missing_quality_status": "not_applicable",
        "config_sha256": "a" * 64,
        "series_preview": {"indices": [0, 1, 2, 3, 4], "values": [0.0, 1.0, 0.5, -0.2, 0.1]},
    }


class ContractTest(unittest.TestCase):
    def test_prediction_contract_rejects_evaluation_fields(self) -> None:
        raw = valid_run_input()
        raw["y_test"] = [0, 1]
        with self.assertRaisesRegex(ContractError, "forbidden field"):
            validate_run_input(raw)

    def test_prediction_contract_rejects_nested_evaluation_fields(self) -> None:
        raw = valid_run_input()
        raw["source_evidence"]["statistical"]["tp"] = 1  # type: ignore[index]
        with self.assertRaisesRegex(ContractError, "forbidden field"):
            validate_run_input(raw)

    def test_prediction_hash_is_order_independent(self) -> None:
        raw = valid_run_input()
        left = dict(raw)
        right = dict(reversed(list(raw.items())))
        self.assertEqual(prediction_sha256(left), prediction_sha256(right))

    def test_rejects_candidate_outside_range(self) -> None:
        raw = valid_run_input()
        raw["baseline_standard_indices"] = [5]
        with self.assertRaisesRegex(ContractError, "outside run range"):
            validate_run_input(raw)

    def test_rejects_lane_overlap(self) -> None:
        raw = valid_run_input()
        raw["baseline_priority_indices"] = [1]
        with self.assertRaisesRegex(ContractError, "lane overlap"):
            validate_run_input(raw)

    def test_preview_must_include_every_candidate(self) -> None:
        raw = valid_run_input()
        raw["series_preview"] = {"indices": [0, 2, 3, 4], "values": [0.0, 0.5, -0.2, 0.1]}
        with self.assertRaisesRegex(ContractError, "preview omits candidate"):
            validate_run_input(raw)


if __name__ == "__main__":
    unittest.main()
