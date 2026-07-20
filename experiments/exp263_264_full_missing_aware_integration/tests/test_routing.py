from __future__ import annotations

import inspect
import unittest

from experiments.exp263_264_full_missing_aware_integration.routing import (
    RoutingInputs,
    route_dataset,
)


def make_inputs(
    *,
    n_train: int = 100,
    quality_flags: set[int] | None = None,
    **changes: object,
) -> RoutingInputs:
    values: dict[str, object] = {
        "dataset": "sample",
        "n_train": n_train,
        "n_test": 5,
        "baseline_hard": {0},
        "baseline_standard": {1, 2, 3, 4},
        "baseline_priority": set(),
        "sources": {
            "statistical": {1, 2},
            "structural": {1, 2},
            "change_point": {1},
            "reconstruction": {3},
        },
        "quality_flags": set() if quality_flags is None else set(quality_flags),
    }
    values.update(changes)
    return RoutingInputs(**values)  # type: ignore[arg-type]


class RoutingTest(unittest.TestCase):
    def test_three_of_four_promotes_large_train_quality_pass(self) -> None:
        row = route_dataset(make_inputs())

        self.assertEqual(row["promoted_indices"], [1])
        self.assertEqual(row["final_hard_indices"], [0, 1])

    def test_three_of_four_quality_failure_moves_to_priority(self) -> None:
        row = route_dataset(make_inputs(quality_flags={1}))

        self.assertEqual(row["promoted_indices"], [])
        self.assertIn(1, row["final_priority_indices"])

    def test_two_sources_move_standard_to_priority(self) -> None:
        row = route_dataset(make_inputs())

        self.assertIn(2, row["final_priority_indices"])

    def test_one_source_keeps_standard(self) -> None:
        row = route_dataset(make_inputs())

        self.assertIn(3, row["final_standard_indices"])

    def test_small_train_never_promotes_to_hard(self) -> None:
        row = route_dataset(make_inputs(n_train=99))

        self.assertEqual(row["promoted_indices"], [])
        self.assertNotIn(1, row["final_hard_indices"])

    def test_existing_hard_is_never_removed(self) -> None:
        row = route_dataset(make_inputs())

        self.assertIn(0, row["final_hard_indices"])

    def test_lanes_are_mutually_exclusive(self) -> None:
        row = route_dataset(make_inputs())
        lanes = [
            set(row[key])
            for key in (
                "final_hard_indices",
                "final_standard_indices",
                "final_priority_indices",
            )
        ]

        self.assertTrue(
            all(
                not lanes[left] & lanes[right]
                for left in range(len(lanes))
                for right in range(left + 1, len(lanes))
            )
        )

    def test_rejects_out_of_bounds_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside TEST range"):
            route_dataset(
                make_inputs(
                    sources={
                        "statistical": {5},
                        "structural": set(),
                        "change_point": set(),
                        "reconstruction": set(),
                    }
                )
            )

    def test_rejects_overlapping_baseline_lanes(self) -> None:
        with self.assertRaisesRegex(ValueError, "baseline lane overlap"):
            route_dataset(make_inputs(baseline_priority={0}))

    def test_rejects_missing_baseline_hard_from_final_lanes(self) -> None:
        inputs = make_inputs()
        object.__setattr__(inputs, "baseline_hard", {0})
        row = route_dataset(inputs)
        self.assertTrue(set(inputs.baseline_hard) <= set(row["final_hard_indices"]))

    def test_route_interface_and_output_are_label_free(self) -> None:
        parameters = inspect.signature(route_dataset).parameters
        self.assertFalse({"labels", "y_test", "test_labels"} & set(parameters))

        row = route_dataset(make_inputs())
        self.assertFalse(
            {"labels", "y_test", "test_labels", "f1", "precision", "recall"}
            & set(row)
        )


if __name__ == "__main__":
    unittest.main()
