"""Synthetic, label-free regression contracts for the guarded Hard route."""
from __future__ import annotations

import inspect
import unittest
from unittest.mock import patch

import numpy as np

from paper.equipment_anomaly_detection.contracts_paper import RawDataset
from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import (
    EXPECTED_HARD_SOURCES,
    HardSourceBundle,
    HardSourceResult,
)
from paper.equipment_anomaly_detection.sources_paper.legacy_route_sources_paper import (
    ConfirmationBlocks,
    FullFitScoreView,
    LegacyRouteEvidence,
    WEAK_FAMILIES_PAPER,
)
from paper.equipment_anomaly_detection.sources_paper.minirocket_paper import MiniRocketResult
from paper.equipment_anomaly_detection.hard_route_paper import (
    RouteLanes,
    exact_normal_guard,
    minimum_source_expansion,
    replay_exp133_134_135_137,
    replay_exp74d_89_90_93,
    replay_hard_route,
)


def _values() -> RawDataset:
    train = tuple(np.asarray([0.0, 1.0, 2.0], dtype=np.float32) + index for index in range(12))
    test = tuple(np.asarray([0.0, 1.0, 2.0], dtype=np.float32) + index for index in range(6))
    return RawDataset(
        dataset="WeakFamily_normal_1",
        metadata_length=3,
        train_indices=tuple(range(len(train))), train_blobs=tuple(b"train" + bytes([index]) for index in range(len(train))), train_rows=train,
        test_indices=tuple(range(len(test))), test_blobs=(b"different", b"train\x02", b"different-2", b"different-3", b"different-4", b"different-5"), test_rows=test,
    )


def _view(name: str, test: tuple[float, ...], *, rate: float = .015) -> FullFitScoreView:
    train = np.linspace(0.0, 1.1, 12)
    return FullFitScoreView(name=name, train_score=train, test_score=np.asarray(test), rate=rate)


def _evidence() -> LegacyRouteEvidence:
    # Full-fit candidates deliberately disagree with the OOF evidence below.
    primary = {
        "rocket_exp40": _view("rocket_exp40", (0.1, 2.0, 1.8, 0.2, 0.3, 0.4)),
        "exp55_best": _view("exp55_best", (0.1, 1.9, 1.7, 0.2, 0.3, 0.4)),
        "exp56_best": _view("exp56_best", (0.1, 1.8, 1.6, 0.2, 0.3, 0.4)),
    }
    exp84 = _view("exp84", (0.1, 2.1, 1.5, 0.2, 0.3, 0.4), rate=.03)
    blocks = ConfirmationBlocks(
        block_a=_view("block_a", (0.0, 1.0, 2.0, 0.0, 0.0, 0.0), rate=.015),
        block_b=_view("block_b", (0.0, 2.0, 0.1, 0.0, 0.0, 0.0), rate=.015),
        block_c=_view("block_c", (0.0, 0.0, 2.0, 0.0, 0.0, 0.0), rate=.01),
    )
    return LegacyRouteEvidence(
        dataset="WeakFamily_normal_1", family="WeakFamily", n_train=12, n_test=6,
        test_median_length=3, primary=primary, exp84=exp84, confirmation_blocks=blocks,
    )


def _sources() -> HardSourceBundle:
    by_name = {}
    for name in EXPECTED_HARD_SOURCES:
        # p=1/201 for every OOF source except a two-source agreement at index 4.
        train_oof = np.arange(200, dtype=float)
        test = np.zeros(6, dtype=float)
        if name in {"rocket_exp40", "exp55_best"}:
            test[4] = 1000.0
        by_name[name] = HardSourceResult(
            name=name, origin=object(), train_oof_score=train_oof, test_score=test,
            test_pvalue=np.ones(6), threshold=.01, test_native_pass=np.zeros(6, dtype=bool),
            eligible=True, eligibility_reason="eligible",
        )
    return HardSourceBundle(EXPECTED_HARD_SOURCES, by_name)


class HardRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.values = _values()
        self.evidence = _evidence()
        self.sources = _sources()
        self.mini = MiniRocketResult(k1024=frozenset({3}), k4096=frozenset({3}), by_kernel={})

    def test_fullfit_evidence_drives_exp74d_not_task4_oof(self) -> None:
        """Changing Task4 OOF values cannot change the frozen Exp74d selection."""
        exp74d, exp89, exp90, exp93 = replay_exp74d_89_90_93(self.evidence)
        self.assertIn(1, exp74d.hard | exp74d.standard | exp74d.priority)
        self.assertEqual(exp89, exp90)
        self.assertIsInstance(exp93, frozenset)

    def test_exp84_family_guard_uses_the_frozen_weak_family_set(self) -> None:
        self.assertIn("ChlorineConcentration", WEAK_FAMILIES_PAPER)
        self.assertNotIn("InlineSkate", WEAK_FAMILIES_PAPER)
        guarded = self.evidence.exp84_family_guard
        self.assertEqual(guarded.rate, .02)

    def test_exp234_uses_only_five_task4_oof_sources(self) -> None:
        lanes = RouteLanes(frozenset({1}), frozenset({2, 3}), frozenset())
        result = minimum_source_expansion(lanes, self.sources, alpha=.01)
        self.assertEqual(result.minimum_two, frozenset({4}))
        self.assertEqual(result.minimum_three, frozenset())
        self.sources.by_name["rocket_block_b"].test_score[4] = 1000.0
        self.assertEqual(minimum_source_expansion(lanes, self.sources, alpha=.01).minimum_two, frozenset({4}))

    def test_confirmation_route_keeps_lanes_disjoint(self) -> None:
        exp93 = frozenset({1, 2})
        lanes = replay_exp133_134_135_137(exp93, self.evidence.confirmation_blocks)
        self.assertFalse(lanes.hard & lanes.standard)
        self.assertFalse(lanes.hard & lanes.priority)
        self.assertFalse(lanes.standard & lanes.priority)

    def test_route_preserves_stages_and_exact_blob_guard(self) -> None:
        with patch("paper.equipment_anomaly_detection.hard_route_paper.build_legacy_route_evidence", return_value=self.evidence):
            result = replay_hard_route(self.values, self.sources, self.mini)
        self.assertEqual(result.exact_normal_removed, frozenset({1}))
        self.assertNotIn(1, result.automatic_hard)
        self.assertTrue(result.exp253.hard.issuperset(result.automatic_hard))

    def test_exact_normal_guard_is_byte_exact_not_value_equivalent(self) -> None:
        automatic, removed = exact_normal_guard(frozenset({0, 1}), (b"x",), (b"x", b"x\x00"))
        self.assertEqual(automatic, frozenset({1}))
        self.assertEqual(removed, frozenset({0}))

    def test_route_function_cannot_accept_oracle_coordinates(self) -> None:
        self.assertNotIn("oracle_path", inspect.signature(replay_hard_route).parameters)
        self.assertNotIn("frozen_coordinates", inspect.signature(replay_hard_route).parameters)


if __name__ == "__main__":
    unittest.main()
