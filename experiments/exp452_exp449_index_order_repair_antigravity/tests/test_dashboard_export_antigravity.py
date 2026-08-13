from __future__ import annotations

from copy import deepcopy
import unittest

from experiments.exp452_exp449_index_order_repair_antigravity.config_antigravity import CANDIDATE_MATRIX_ANTIGRAVITY
from experiments.exp452_exp449_index_order_repair_antigravity.dashboard_export_antigravity import public_result_antigravity


def _valid_result_antigravity() -> dict[str, object]:
    global_f1 = 0.62664286873276
    candidates = [
        {
            "family": family,
            "method": method,
            "target_fn_delta": -1,
            "gates": {"global_fn": True, "global_fp": True, "global_f1": True, "target_fn_delta": True},
            "qualified": True,
            "global": {"tp": 1931, "fp": 314, "fn": 1987, "tn": 191668, "f1": global_f1},
        }
        for family, methods in CANDIDATE_MATRIX_ANTIGRAVITY
        for method in methods
    ]
    return {
        "schema": "exp452.public-method-qualification.v1",
        "experiment": "Exp452",
        "baseline": {"tp": 1930, "fp": 314, "fn": 1988, "tn": 191668, "f1": 0.6264199935086011},
        "candidates": candidates,
        "families": [family for family, _methods in CANDIDATE_MATRIX_ANTIGRAVITY],
        "status": "synthetic_complete",
        "repeatable": True,
        "qualified_candidate_count": 16,
        "routing_research_eligible": True,
        "runtime": {"label_sessions": 1, "label_queries": 48, "label_rows": 72300},
    }


class DashboardExportAntigravityTest(unittest.TestCase):
    def test_public_result_returns_only_the_complete_aggregate_allowlist_antigravity(self) -> None:
        """Break caught: dropping an aggregate or adding an operational/detail field."""
        payload = _valid_result_antigravity()
        self.assertEqual(public_result_antigravity(payload), payload)

    def test_public_result_rejects_forbidden_material_at_any_depth_antigravity(self) -> None:
        """Break caught: leaking labels, indices, dataset rows, paths, hashes, or logs."""
        mutations: tuple[tuple[str, object], ...] = (
            ("raw labels", {"raw_labels": [0, 1]}),
            ("indices", {"indices": [1, 3, 5]}),
            ("dataset name", {"dataset_name": "Crop_normal_1"}),
            ("per-dataset rows", {"per_dataset_rows": [{"dataset": "Crop_normal_1"}]}),
            ("local path", {"local_path": "/Users/mac/Documents/Dataset/private"}),
            ("hash", {"sha256": "a" * 64}),
            ("logs", {"logs": ["private"]}),
        )
        for label, nested in mutations:
            payload = _valid_result_antigravity()
            payload["runtime"] = {"label_sessions": 1, "label_queries": 48, "label_rows": 72300, "nested": {"deeper": nested}}
            with self.subTest(label=label), self.assertRaises(ValueError):
                public_result_antigravity(payload)

    def test_public_result_rejects_forbidden_nested_string_values_antigravity(self) -> None:
        """Break caught: hiding a path, digest, or raw dataset identity under an allowed-looking key."""
        for value in (
            "/Users/mac/Documents/Dataset/private",
            "a" * 64,
            "Crop_normal_1",
        ):
            payload = _valid_result_antigravity()
            payload["status"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                public_result_antigravity(payload)

    def test_public_result_recomputes_counts_f1_gates_and_summary_antigravity(self) -> None:
        """Break caught: accepting self-reported aggregate math without recomputation."""
        mutations = []
        wrong_count = _valid_result_antigravity()
        wrong_count["baseline"] = {"tp": True, "fp": 314, "fn": 1988, "tn": 191668, "f1": 0.6264199935086011}
        mutations.append(wrong_count)
        wrong_f1 = _valid_result_antigravity()
        wrong_f1["baseline"] = {**wrong_f1["baseline"], "f1": 0.5}  # type: ignore[arg-type]
        mutations.append(wrong_f1)
        wrong_gate = _valid_result_antigravity()
        wrong_gate["candidates"] = deepcopy(wrong_gate["candidates"])
        wrong_gate["candidates"][0]["gates"]["global_fn"] = False  # type: ignore[index]
        mutations.append(wrong_gate)
        wrong_summary = _valid_result_antigravity()
        wrong_summary["qualified_candidate_count"] = 15
        mutations.append(wrong_summary)
        for payload in mutations:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                public_result_antigravity(payload)


if __name__ == "__main__":
    unittest.main()
