"""Task5/Task6 policy identity contract: two distinct Hard-route identities.

Candidate C keeps Task6 hard_lanes equal to exp253.hard and EvidenceSurface
alerts/policies equal to automatic_hard. The two sets differ when
exact_normal_removed is nonempty.
"""
from __future__ import annotations

import unittest

import numpy as np

from paper.equipment_anomaly_detection.contracts_paper import (
    LowFrequencyResult,
    LowFrequencyRow,
    RawDataset,
)
from paper.equipment_anomaly_detection.hard_route_paper import HardRouteResult, RouteLanes, SourceExpansion
from paper.equipment_anomaly_detection.routing_paper import EvidenceSurface, build_policies
from paper.equipment_anomaly_detection.sources_paper.stability_paper import StabilityPartition


DATASET = "fixture"
EXP253_HARD = frozenset({0, 1})
EXACT_NORMAL_REMOVED = frozenset({0})
AUTOMATIC_HARD = frozenset({1})


def _blob(values: tuple[float, ...]) -> bytes:
    return np.asarray(values, dtype="<f4").tobytes()


def _divergent_route_authority() -> HardRouteResult:
    empty = RouteLanes(frozenset(), frozenset(), frozenset())
    exp253 = RouteLanes(EXP253_HARD, frozenset(), frozenset())
    return HardRouteResult(
        empty,
        frozenset(),
        frozenset(),
        frozenset(),
        empty,
        SourceExpansion(frozenset(), frozenset()),
        exp253,
        frozenset(),
        exp253,
        EXACT_NORMAL_REMOVED,
        AUTOMATIC_HARD,
        2,
    )


def _divergent_fixture() -> tuple[EvidenceSurface, LowFrequencyResult, StabilityPartition, dict[str, RawDataset], dict[str, HardRouteResult]]:
    rows = (np.asarray((0.0, 1.0), dtype="<f4"), np.asarray((2.0, 3.0), dtype="<f4"))
    blobs = (_blob((0.0, 1.0)), _blob((2.0, 3.0)))
    raw = RawDataset(DATASET, 2, (0, 1), blobs, rows, (0, 1), blobs, rows)
    route = _divergent_route_authority()
    low_rows = (
        LowFrequencyRow(DATASET, 0, 2, (), False, None, None, False, True, "no_group"),
        LowFrequencyRow(DATASET, 1, 2, (), False, None, None, False, True, "no_group"),
    )
    low = LowFrequencyResult((DATASET,), low_rows, (), (), 0, {DATASET: frozenset(route.exp253.hard)})
    stability = StabilityPartition((), (), (), {})
    surface = EvidenceSurface(
        {DATASET: frozenset(route.automatic_hard)},
        {DATASET: np.asarray((0.1, 0.2))},
        {DATASET: np.asarray((False, False))},
    )
    return surface, low, stability, {DATASET: raw}, {DATASET: route}


class Task5Task6PolicyIdentityTests(unittest.TestCase):
    def test_divergent_exp253_hard_and_automatic_hard_are_independently_valid(self) -> None:
        surface, low, stability, raw, routes = _divergent_fixture()
        self.assertEqual(low.hard_lanes[DATASET], EXP253_HARD)
        self.assertEqual(surface.alerts[DATASET], AUTOMATIC_HARD)
        self.assertNotEqual(low.hard_lanes[DATASET], surface.alerts[DATASET])
        self.assertEqual(routes[DATASET].exp253.hard, EXP253_HARD)
        self.assertEqual(routes[DATASET].exact_normal_removed, EXACT_NORMAL_REMOVED)
        self.assertEqual(routes[DATASET].automatic_hard, AUTOMATIC_HARD)
        self.assertEqual(routes[DATASET].automatic_hard, routes[DATASET].exp253.hard - routes[DATASET].exact_normal_removed)
        policies = build_policies(surface, low, stability, raw, routes)
        self.assertEqual(policies.by_name["hard_only"].alerts[DATASET], AUTOMATIC_HARD)
        self.assertEqual(low.hard_lanes[DATASET], EXP253_HARD)

    def test_wrong_task6_lanes_fail_closed(self) -> None:
        surface, low, stability, raw, routes = _divergent_fixture()
        wrong_low = LowFrequencyResult(low.datasets, low.rows, low.provisional_candidates, low.selected_candidates, low.excluded_train_rows, {DATASET: frozenset(AUTOMATIC_HARD)})
        with self.assertRaisesRegex(ValueError, "Task 6 eligibility lanes"):
            build_policies(surface, wrong_low, stability, raw, routes)

    def test_wrong_surface_alerts_fail_closed(self) -> None:
        surface, low, stability, raw, routes = _divergent_fixture()
        wrong_surface = EvidenceSurface(
            {DATASET: frozenset(EXP253_HARD)},
            {DATASET: np.asarray(surface.strength[DATASET])},
            {DATASET: np.asarray(surface.evidence_missing[DATASET])},
        )
        with self.assertRaisesRegex(ValueError, "Hard evidence alerts"):
            build_policies(wrong_surface, low, stability, raw, routes)


if __name__ == "__main__":
    unittest.main()
