"""Pure, label-free Exp443 routing behavior tests."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from experiments.exp443_missing_normal_train_policy_comparison_antigravity.contracts_antigravity import (
    KernelSeedScoresAntigravity,
    PolicyDatasetResultAntigravity,
)


POLICY = "linear_edge_nearest_control"


def _score(kernel: int, seed: int, candidates: tuple[int, ...], n_test: int) -> KernelSeedScoresAntigravity:
    oof = np.arange(1000, dtype="<f8")
    test = np.zeros(n_test, dtype="<f8")
    test[list(candidates)] = 2000.0
    return KernelSeedScoresAntigravity(kernel, seed, oof, test, candidates)


def _result(dataset: str, *, candidates_1024: tuple[int, ...], candidates_4096: tuple[int, ...], n_test: int = 6) -> PolicyDatasetResultAntigravity:
    scores = tuple(
        _score(kernel, seed, candidates, n_test)
        for kernel, candidates in ((1024, candidates_1024), (4096, candidates_4096))
        for seed in (20260721, 20261721)
    )
    return PolicyDatasetResultAntigravity(
        POLICY,
        dataset,
        8,
        1000,
        0,
        1000,
        0,
        n_test,
        scores,
        {1024: candidates_1024, 4096: candidates_4096},
        (),
        "scored",
    )


def _snapshot() -> SimpleNamespace:
    return SimpleNamespace(
        datasets=("A", "B", "C"),
        affected_datasets=("A",),
        test_row_count_by_dataset={"A": 6, "B": 4, "C": 3},
        exp236_lanes={
            "A": {"hard": (0,), "standard": (1, 2), "priority": (4,), "promotion": ()},
            "B": {"hard": (1,), "standard": (0, 2), "priority": (3,), "promotion": ()},
            "C": {"hard": (), "standard": (0,), "priority": (), "promotion": ()},
        },
        exp249_candidate_maps={
            "A": {1024: (1, 3), 4096: (2, 3)},
            "B": {1024: (0,), 4096: (2,)},
            "C": {1024: (0,), 4096: ()},
        },
        hard_coordinates_exp291=frozenset({("A", 0), ("B", 1)}),
        hard_coordinates_exp404=frozenset({("A", 0), ("B", 1)}),
    )


class RoutingAntigravityTests(unittest.TestCase):
    def test_routing_intersects_standard_promotes_union_and_removes_exact_normal(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.routing_antigravity import route_policy_antigravity

        explicit_snapshot = _snapshot()
        explicit_snapshot.exp236_lanes = dict(explicit_snapshot.exp236_lanes)
        explicit_snapshot.exp236_lanes["A"] = {
            **explicit_snapshot.exp236_lanes["A"],
            "promotion": (3,),
        }
        explicit_snapshot.hard_coordinates_exp291 = frozenset({("B", 1)})
        explicit_snapshot.hard_coordinates_exp404 = frozenset({("B", 1)})
        prediction = route_policy_antigravity(
            explicit_snapshot,
            {"A": _result("A", candidates_1024=(1, 3), candidates_4096=(2, 3))},
            POLICY,
            exact_normal_coordinates=frozenset({("A", 0), ("A", 3)}),
        )
        row_a = next(row for row in prediction.rows if row["dataset"] == "A")
        self.assertEqual(row_a["promotion"], (1, 2))
        self.assertEqual(row_a["pre_exact"], (0, 1, 2))
        self.assertEqual(row_a["exact_normal_removed"], (0,))
        self.assertEqual(row_a["final_hard"], (1, 2))
        self.assertEqual(row_a["inherited_priority"], (4,))
        self.assertEqual(row_a["outside_standard"], (3,))
        self.assertEqual(
            prediction.final_hard_coordinates,
            frozenset({("A", 1), ("A", 2), ("B", 0), ("B", 1), ("B", 2), ("C", 0)}),
        )

    def test_unaffected_datasets_import_exp249_candidates_and_audit_only_standard(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.routing_antigravity import route_policy_antigravity

        prediction = route_policy_antigravity(_snapshot(), {"A": _result("A", candidates_1024=(1, 3), candidates_4096=(2, 3))}, POLICY)
        row_b = next(row for row in prediction.rows if row["dataset"] == "B")
        self.assertEqual(row_b["mini1024"], (0,))
        self.assertEqual(row_b["mini4096"], (2,))
        self.assertEqual(row_b["promotion"], (0, 2))
        self.assertEqual(row_b["final_hard"], (0, 1, 2))
        row_c = next(row for row in prediction.rows if row["dataset"] == "C")
        self.assertEqual(row_c["promotion"], (0,))
        self.assertEqual(row_c["final_hard"], (0,))
        self.assertNotIn(2, row_c["promotion"])

    def test_rows_are_deterministic_and_canonical_bytes_are_label_free(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.routing_antigravity import canonical_prediction_bytes_antigravity, route_policy_antigravity

        first = route_policy_antigravity(_snapshot(), {"A": _result("A", candidates_1024=(1, 3), candidates_4096=(2, 3))}, POLICY)
        second = route_policy_antigravity(_snapshot(), {"A": _result("A", candidates_1024=(1, 3), candidates_4096=(2, 3))}, POLICY)
        self.assertEqual(canonical_prediction_bytes_antigravity(first), canonical_prediction_bytes_antigravity(second))
        self.assertEqual(len(first.rows), 3)
        self.assertFalse(first.contains_test_labels)
        self.assertFalse(first.contains_evaluation_metrics)
        self.assertTrue(all(row["contains_test_labels"] is False for row in first.rows))
        self.assertTrue(all(row["contains_evaluation_metrics"] is False for row in first.rows))
        self.assertNotIn('"contains_test_labels":true', canonical_prediction_bytes_antigravity(first).decode())
        self.assertNotIn('"contains_evaluation_metrics":true', canonical_prediction_bytes_antigravity(first).decode())

    def test_public_prediction_row_schema_and_flags_are_exact(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.routing_antigravity import route_policy_antigravity

        prediction = route_policy_antigravity(_snapshot(), {_snapshot().affected_datasets[0]: _result("A", candidates_1024=(1, 3), candidates_4096=(2, 3))}, POLICY)
        expected = {
            "dataset", "policy", "n_test", "mini1024", "mini4096",
            "inherited_hard", "inherited_standard", "inherited_priority",
            "outside_standard", "promotion", "pre_exact", "exact_normal_removed",
            "final_hard", "contains_test_labels", "contains_evaluation_metrics",
            "provenance",
        }
        for row in prediction.rows:
            self.assertEqual(set(row), expected)
            self.assertIs(row["contains_test_labels"], False)
            self.assertIs(row["contains_evaluation_metrics"], False)

    def test_public_prediction_rejects_true_non_bool_and_recursive_forbidden_fields(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.routing_antigravity import (
            PolicyPredictionAntigravity,
            route_policy_antigravity,
        )

        prediction = route_policy_antigravity(_snapshot(), {"A": _result("A", candidates_1024=(1, 3), candidates_4096=(2, 3))}, POLICY)
        row = dict(prediction.rows[0])
        for key, value in (("contains_test_labels", True), ("contains_evaluation_metrics", 1), ("labels", {"x": 1})):
            forged = dict(row)
            forged[key] = value
            with self.assertRaises(ValueError):
                PolicyPredictionAntigravity(POLICY, (forged,), frozenset(), frozenset(), frozenset(), frozenset())
        forged = dict(row)
        forged["provenance"] = {"candidate_source": "x", "routing_rule": "x", "labels": {"x": 1}}
        with self.assertRaises(ValueError):
            PolicyPredictionAntigravity(POLICY, (forged,), frozenset(), frozenset(), frozenset(), frozenset())

    def test_prediction_deep_copy_blocks_source_mutation_and_canonical_revalidates(self) -> None:
        from types import MappingProxyType

        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.routing_antigravity import (
            PolicyPredictionAntigravity,
            canonical_prediction_bytes_antigravity,
        )

        provenance = {
            "candidate_source": "recomputed_policy",
            "routing_rule": "standard_intersection_union_then_exact_normal_subtraction",
        }
        row = {
            "dataset": "A", "policy": POLICY, "n_test": 1,
            "mini1024": (), "mini4096": (), "inherited_hard": (),
            "inherited_standard": (), "inherited_priority": (), "outside_standard": (),
            "promotion": (), "pre_exact": (), "exact_normal_removed": (), "final_hard": (),
            "contains_test_labels": False, "contains_evaluation_metrics": False,
            "provenance": provenance,
        }
        prediction = PolicyPredictionAntigravity(POLICY, (row,), frozenset(), frozenset(), frozenset(), frozenset())
        before = canonical_prediction_bytes_antigravity(prediction)
        provenance["candidate_source"] = "mutated"
        row["provenance"]["routing_rule"] = "mutated"
        self.assertEqual(canonical_prediction_bytes_antigravity(prediction), before)

        forged_row = dict(prediction.rows[0])
        forged_row["provenance"] = {
            "candidate_source": "recomputed_policy",
            "routing_rule": "standard_intersection_union_then_exact_normal_subtraction",
            "labels": {"secret": 1},
        }
        object.__setattr__(prediction, "rows", (MappingProxyType(forged_row),))
        with self.assertRaises(ValueError):
            canonical_prediction_bytes_antigravity(prediction)

    def test_routing_rejects_out_of_range_duplicates_and_candidates_outside_standard(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.routing_antigravity import route_policy_antigravity

        forged = SimpleNamespace(**vars(_snapshot()))
        forged.exp236_lanes = dict(forged.exp236_lanes)
        forged.exp236_lanes["A"] = {**forged.exp236_lanes["A"], "standard": (1, 1)}
        with self.assertRaisesRegex(ValueError, "not sorted and unique"):
            route_policy_antigravity(
                forged,
                {"A": _result("A", candidates_1024=(1, 3), candidates_4096=(2, 3))},
                POLICY,
            )
        forged_result = _result("A", candidates_1024=(1, 3), candidates_4096=())
        object.__setattr__(forged_result, "candidates_by_kernel", {1024: (6,), 4096: ()})
        with self.assertRaises(ValueError):
            route_policy_antigravity(_snapshot(), {"A": forged_result}, POLICY)

    def test_explicit_exact_normal_path_validates_inherited_promotion_and_frozen_relation(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.routing_antigravity import route_policy_antigravity

        duplicate_promotion = SimpleNamespace(**vars(_snapshot()))
        duplicate_promotion.exp236_lanes = dict(duplicate_promotion.exp236_lanes)
        duplicate_promotion.exp236_lanes["A"] = {
            **duplicate_promotion.exp236_lanes["A"],
            "promotion": (3, 3),
        }
        with self.assertRaisesRegex(ValueError, "not sorted and unique"):
            route_policy_antigravity(
                duplicate_promotion,
                {"A": _result("A", candidates_1024=(1, 3), candidates_4096=(2, 3))},
                POLICY,
                exact_normal_coordinates=frozenset(),
            )

        out_of_range_promotion = SimpleNamespace(**vars(_snapshot()))
        out_of_range_promotion.exp236_lanes = dict(out_of_range_promotion.exp236_lanes)
        out_of_range_promotion.exp236_lanes["A"] = {
            **out_of_range_promotion.exp236_lanes["A"],
            "promotion": (6,),
        }
        with self.assertRaisesRegex(ValueError, "outside TEST identity"):
            route_policy_antigravity(
                out_of_range_promotion,
                {"A": _result("A", candidates_1024=(1, 3), candidates_4096=(2, 3))},
                POLICY,
                exact_normal_coordinates=frozenset(),
            )

        with self.assertRaisesRegex(ValueError, "exact-normal coordinates disagree"):
            route_policy_antigravity(
                _snapshot(),
                {"A": _result("A", candidates_1024=(1, 3), candidates_4096=(2, 3))},
                POLICY,
                exact_normal_coordinates=frozenset({("A", 0)}),
            )

    def test_canonical_replay_rejects_row_policy_and_provenance_tampering(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.routing_antigravity import (
            canonical_prediction_bytes_antigravity,
            route_policy_antigravity,
        )

        for update in (
            {"policy": "exclude_nonfinite_row"},
            {"provenance": {"candidate_source": "forged", "routing_rule": "standard_intersection_union_then_exact_normal_subtraction"}},
            {"provenance": {"candidate_source": "recomputed_policy", "routing_rule": "forged"}},
        ):
            prediction = route_policy_antigravity(
                _snapshot(),
                {"A": _result("A", candidates_1024=(1, 3), candidates_4096=(2, 3))},
                POLICY,
            )
            row = dict(prediction.rows[0])
            row.update(update)
            object.__setattr__(prediction, "rows", (row, *prediction.rows[1:]))
            with self.assertRaises(ValueError):
                canonical_prediction_bytes_antigravity(prediction)

        prediction = route_policy_antigravity(
            _snapshot(),
            {"A": _result("A", candidates_1024=(1, 3), candidates_4096=(2, 3))},
            POLICY,
        )
        object.__setattr__(prediction, "policy", "exclude_nonfinite_row")
        with self.assertRaises(ValueError):
            canonical_prediction_bytes_antigravity(prediction)

    def test_canonical_replay_recomputes_row_routing_algebra(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.routing_antigravity import (
            canonical_prediction_bytes_antigravity,
            route_policy_antigravity,
        )

        for field, value in (
            ("promotion", (1,)),
            ("pre_exact", (0, 1)),
            ("exact_normal_removed", (5,)),
            ("final_hard", (0, 1, 2, 5)),
        ):
            prediction = route_policy_antigravity(
                _snapshot(),
                {"A": _result("A", candidates_1024=(1, 3), candidates_4096=(2, 3))},
                POLICY,
            )
            row = dict(prediction.rows[0])
            row[field] = value
            object.__setattr__(prediction, "rows", (row, *prediction.rows[1:]))
            with self.assertRaises(ValueError):
                canonical_prediction_bytes_antigravity(prediction)

    def test_canonical_replay_requires_aggregate_coordinates_to_equal_row_union(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.routing_antigravity import (
            canonical_prediction_bytes_antigravity,
            route_policy_antigravity,
        )

        for field in ("promotions", "pre_exact_coordinates", "exact_normal_removals", "final_hard_coordinates"):
            prediction = route_policy_antigravity(
                _snapshot(),
                {"A": _result("A", candidates_1024=(1, 3), candidates_4096=(2, 3))},
                POLICY,
            )
            value = getattr(prediction, field)
            object.__setattr__(prediction, field, frozenset(set(value) | {("A", 5)}))
            with self.assertRaises(ValueError):
                canonical_prediction_bytes_antigravity(prediction)

    def test_canonical_replay_rejects_tampered_lane_disjointness_or_exact_evidence_range(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.routing_antigravity import (
            canonical_prediction_bytes_antigravity,
            route_policy_antigravity,
        )

        prediction = route_policy_antigravity(
            _snapshot(),
            {"A": _result("A", candidates_1024=(1, 3), candidates_4096=(2, 3))},
            POLICY,
        )
        row = dict(prediction.rows[0])
        row["inherited_standard"] = (0, 1, 2)
        object.__setattr__(prediction, "rows", (row, *prediction.rows[1:]))
        with self.assertRaises(ValueError):
            canonical_prediction_bytes_antigravity(prediction)

        prediction = route_policy_antigravity(
            _snapshot(),
            {"A": _result("A", candidates_1024=(1, 3), candidates_4096=(2, 3))},
            POLICY,
        )
        object.__setattr__(prediction, "exact_normal_coordinates", frozenset({("A", 6)}))
        with self.assertRaises(ValueError):
            canonical_prediction_bytes_antigravity(prediction)

    def test_affected_result_requires_scored_exact_four_seed_kernel_scores_and_recomputed_candidates(self) -> None:
        from types import MappingProxyType

        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.routing_antigravity import route_policy_antigravity

        for status in ("abstained", "failed"):
            result = _result("A", candidates_1024=(1, 3), candidates_4096=(2, 3))
            object.__setattr__(result, "status", status)
            with self.assertRaises(ValueError):
                route_policy_antigravity(_snapshot(), {"A": result}, POLICY)

        result = _result("A", candidates_1024=(1, 3), candidates_4096=(2, 3))
        object.__setattr__(result, "kernel_scores", ())
        with self.assertRaises(ValueError):
            route_policy_antigravity(_snapshot(), {"A": result}, POLICY)

        result = _result("A", candidates_1024=(1, 3), candidates_4096=(2, 3))
        object.__setattr__(result, "candidates_by_kernel", MappingProxyType({1024: (1,), 4096: (2, 3)}))
        with self.assertRaises(ValueError):
            route_policy_antigravity(_snapshot(), {"A": result}, POLICY)

    def test_routing_rejects_overlapping_hard_standard_priority_lanes(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.routing_antigravity import route_policy_antigravity

        for left, right in (("hard", "standard"), ("hard", "priority"), ("standard", "priority")):
            forged = SimpleNamespace(**vars(_snapshot()))
            forged.exp236_lanes = dict(forged.exp236_lanes)
            lanes = dict(forged.exp236_lanes["A"])
            lanes[left] = (0,)
            lanes[right] = (0,)
            forged.exp236_lanes["A"] = lanes
            with self.assertRaises(ValueError):
                route_policy_antigravity(forged, {"A": _result("A", candidates_1024=(1, 3), candidates_4096=(2, 3))}, POLICY)

    def test_real_source_backed_affected_replay_uses_pinned_cache_arrays(self) -> None:
        from experiments.exp239_248_diverse_score_sources.calibration import empirical_candidates
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.config_antigravity import FROZEN_CONFIG
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.minirocket_antigravity import seed_intersection_candidates_antigravity
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.routing_antigravity import (
            canonical_prediction_bytes_antigravity,
            route_policy_antigravity,
        )
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.sources_antigravity import verify_frozen_sources_antigravity

        snapshot = verify_frozen_sources_antigravity()
        results: dict[str, PolicyDatasetResultAntigravity] = {}
        for dataset in snapshot.affected_datasets:
            authority = snapshot.cache_inventory[dataset]
            arrays = authority["arrays"]
            scores: list[KernelSeedScoresAntigravity] = []
            candidates_by_kernel: dict[int, tuple[int, ...]] = {}
            for kernel in FROZEN_CONFIG.kernel_requests:
                train_by_seed = arrays[f"kernels_{kernel}__train_by_seed"]
                test_by_seed = arrays[f"kernels_{kernel}__test_by_seed"]
                pair: list[KernelSeedScoresAntigravity] = []
                for seed_index, seed in enumerate(FROZEN_CONFIG.seeds):
                    oof = np.asarray(train_by_seed[seed_index], dtype="<f8")
                    test_scores = np.asarray(test_by_seed[seed_index], dtype="<f8")
                    empirical = empirical_candidates(oof, test_scores, FROZEN_CONFIG.alpha)
                    candidates = tuple(int(index) for index in empirical.indices)
                    pair.append(KernelSeedScoresAntigravity(kernel, seed, oof, test_scores, candidates))
                scores.extend(pair)
                candidates_by_kernel[kernel] = seed_intersection_candidates_antigravity(
                    pair, alpha=FROZEN_CONFIG.alpha
                )
                self.assertEqual(candidates_by_kernel[kernel], snapshot.exp249_candidate_maps[dataset][kernel])
            results[dataset] = PolicyDatasetResultAntigravity(
                POLICY,
                dataset,
                authority["target_length"],
                authority["n_train"],
                0,
                authority["n_train"],
                0,
                authority["n_test"],
                tuple(scores),
                candidates_by_kernel,
                (),
                "scored",
            )
        prediction = route_policy_antigravity(snapshot, results, POLICY)
        self.assertEqual(set(results), set(snapshot.affected_datasets))
        self.assertEqual(len(prediction.rows), 1117)
        self.assertEqual(len(prediction.promotions), 137)
        self.assertEqual(len(prediction.pre_exact_coordinates), 2253)
        self.assertEqual(len(prediction.exact_normal_removals), 35)
        self.assertEqual(len(prediction.final_hard_coordinates), 2218)
        self.assertEqual(prediction.final_hard_coordinates, snapshot.hard_coordinates_exp291)
        self.assertEqual(prediction.final_hard_coordinates, snapshot.hard_coordinates_exp404)
        self.assertEqual(len(canonical_prediction_bytes_antigravity(prediction)), 728425)


if __name__ == "__main__":
    unittest.main()
