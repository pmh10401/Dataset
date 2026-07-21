from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.exp266_fn_recovery_shadow.config import EVIDENCE_GROUPS
from experiments.exp266_fn_recovery_shadow.contracts import FrozenDatasetInputs
from experiments.exp266_fn_recovery_shadow.inputs import sha256_file
from experiments.exp266_fn_recovery_shadow.prediction import freeze_rows, predict_all
from experiments.exp266_fn_recovery_shadow.routing import route_recovery


def prediction_row(dataset: str) -> dict[str, object]:
    return {
        "dataset": dataset,
        "hard_indices": [0],
        "shadow_indices": [],
        "priority_indices": [],
        "standard_indices": [],
        "recovery_shadow_indices": [1],
        "recovery_priority_indices": [],
        "data_quality_review_indices": [],
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }


def example_inputs_with_n_test(n_test: int) -> FrozenDatasetInputs:
    return FrozenDatasetInputs(
        dataset="demo",
        n_train=100,
        n_test=n_test,
        hard=frozenset({0}),
        shadow=frozenset(),
        priority=frozenset(),
        standard=frozenset(),
        evidence={
            "kernel": frozenset({1}),
            "statistical": frozenset({1}),
            "structural": frozenset({1}),
            "change_point": frozenset(),
            "reconstruction": frozenset(),
        },
        available_groups=frozenset(EVIDENCE_GROUPS),
        quality_flags=frozenset(),
    )


class PredictionBoundaryTests(unittest.TestCase):
    def test_prediction_has_no_evaluation_fields(self) -> None:
        with TemporaryDirectory() as directory:
            path = freeze_rows(Path(directory), [prediction_row("a")])
            row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            forbidden = {"y_test", "tp", "fp", "fn", "precision", "recall", "f1"}
            self.assertFalse(forbidden.intersection(key.lower() for key in row))
            self.assertIs(row["contains_test_labels"], False)

    def test_hash_is_input_order_invariant(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            left = freeze_rows(root / "left", [prediction_row("b"), prediction_row("a")])
            right = freeze_rows(root / "right", [prediction_row("a"), prediction_row("b")])
            self.assertEqual(sha256_file(left), sha256_file(right))

    def test_appending_unselected_test_rows_keeps_prefix_routing(self) -> None:
        short = route_recovery(example_inputs_with_n_test(8))
        long = route_recovery(example_inputs_with_n_test(12))
        self.assertEqual(short.recovery_shadow, long.recovery_shadow)

    def test_prediction_api_has_no_label_parameter(self) -> None:
        self.assertNotIn("y_test", inspect.signature(predict_all).parameters)
