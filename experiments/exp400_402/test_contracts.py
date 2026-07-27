from __future__ import annotations

import copy
import unittest

from experiments.exp400_402.contracts import (
    ALPHAS,
    prediction_sha256,
    validate_shadow_row,
)


def valid_candidate(index: int = 2) -> dict[str, object]:
    return {
        "candidate_id": f"replay-exp397-demo:{index}",
        "candidate_index": index,
        "exp292_tier": 1,
        "exp292_rank": 4,
        "tier_preserving_rank": {
            str(alpha): position
            for alpha, position in zip(ALPHAS, (3, 2, 1, 4), strict=True)
        },
        "control_rank": {
            "additive_grammar_family": {
                str(alpha): position
                for alpha, position in zip(ALPHAS, (4, 3, 2, 1), strict=True)
            },
            "grammar_only_control": {
                str(alpha): position
                for alpha, position in zip(ALPHAS, (2, 3, 4, 1), strict=True)
            },
        },
        "grammar_support_count": {str(alpha): 1 for alpha in ALPHAS},
        "p_values": {
            "transition": 0.002,
            "dwell": 0.4,
            "recovery": 0.3,
            "composite": 0.01,
        },
    }


def valid_dataset_row() -> dict[str, object]:
    unsigned = {
        "schema_version": 1,
        "collection_mode": "retrospective_replay",
        "policy_version": "exp397-review-ranking-replay-v1",
        "run_id": "replay-exp397-demo",
        "dataset": "Demo",
        "n_test": 8,
        "automatic_hard_count": 2,
        "priority_review_count": 1,
        "standard_review_count": 1,
        "candidates": [valid_candidate()],
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "automatic_policy_changed": False,
        "retrospective": True,
        "prospective": False,
    }
    return {**unsigned, "prediction_sha256": prediction_sha256(unsigned)}


class ShadowContractTests(unittest.TestCase):
    def test_accepts_canonical_row(self) -> None:
        row = validate_shadow_row(valid_dataset_row())
        self.assertEqual(row["dataset"], "Demo")

    def test_prediction_rejects_test_metrics(self) -> None:
        row = valid_dataset_row()
        row["f1"] = 0.9
        with self.assertRaisesRegex(ValueError, "forbidden|unknown"):
            validate_shadow_row(row)

    def test_prediction_rejects_nested_test_label(self) -> None:
        row = valid_dataset_row()
        candidate = row["candidates"][0]
        candidate["true_label"] = 1
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_shadow_row(row)

    def test_dataset_without_candidates_is_valid(self) -> None:
        row = valid_dataset_row()
        row["candidates"] = []
        row["standard_review_count"] = 0
        unsigned = dict(row)
        unsigned.pop("prediction_sha256")
        row["prediction_sha256"] = prediction_sha256(unsigned)
        self.assertEqual(validate_shadow_row(row)["candidates"], [])

    def test_rejects_incorrect_alpha_coverage(self) -> None:
        row = valid_dataset_row()
        candidate = row["candidates"][0]
        candidate["tier_preserving_rank"].pop("0.05")
        unsigned = dict(row)
        unsigned.pop("prediction_sha256")
        row["prediction_sha256"] = prediction_sha256(unsigned)
        with self.assertRaisesRegex(ValueError, "alpha"):
            validate_shadow_row(row)

    def test_rejects_out_of_range_candidate(self) -> None:
        row = valid_dataset_row()
        candidate = row["candidates"][0]
        candidate["candidate_index"] = 9
        candidate["candidate_id"] = "replay-exp397-demo:9"
        unsigned = dict(row)
        unsigned.pop("prediction_sha256")
        row["prediction_sha256"] = prediction_sha256(unsigned)
        with self.assertRaisesRegex(ValueError, "range"):
            validate_shadow_row(row)

    def test_rejects_hash_tampering(self) -> None:
        row = copy.deepcopy(valid_dataset_row())
        row["automatic_hard_count"] = 3
        with self.assertRaisesRegex(ValueError, "SHA256"):
            validate_shadow_row(row)


if __name__ == "__main__":
    unittest.main()
