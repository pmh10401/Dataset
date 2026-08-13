"""Bounded RED/GREEN contracts for the three paper r4 smoke failure families."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from paper.equipment_anomaly_detection.contracts_paper import FrozenTask8BackendDecision, RawDataset
from paper.equipment_anomaly_detection.hard_route_paper import HardRouteResult, replay_hard_route
from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import (
    EXPECTED_HARD_SOURCES,
    HardSourceBundle,
    HardSourceResult,
    score_hard_sources,
)
from paper.equipment_anomaly_detection.sources_paper.legacy_route_sources_paper import (
    ConfirmationBlocks,
    FullFitScoreView,
    build_legacy_route_evidence,
)
from paper.equipment_anomaly_detection.sources_paper.minirocket_paper import (
    ConfigurationScoresPaper,
    MiniRocketKernelResult,
    MiniRocketResult,
    SEEDS_PAPER,
)
from paper.equipment_anomaly_detection.sources_paper.preprocessing_paper import DependencyUnavailablePaper


FAILED_SMOKE_ROOT = Path(
    "/Users/mac/Documents/Dataset/paper_proposal_r4_prelaunch_smoke_antigravity_20260813"
)
FAILED_SMOKE_OPS_ROOT = Path(
    "/Users/mac/Documents/Dataset/paper_proposal_r4_prelaunch_smoke_antigravity_20260813_ops_antigravity"
)
VERIFIED_V2_SMOKE_ROOT = Path(
    "/Users/mac/Documents/Dataset/paper_proposal_r4_prelaunch_smoke_v2_antigravity_20260813"
)
VERIFIED_V2_SMOKE_OPS_ROOT = Path(str(VERIFIED_V2_SMOKE_ROOT) + "_ops_antigravity")
NEW_SMOKE_ROOT = Path(
    "/Users/mac/Documents/Dataset/paper_proposal_r4_prelaunch_smoke_v3_antigravity_20260814"
)
NEW_SMOKE_OPS_ROOT = Path(str(NEW_SMOKE_ROOT) + "_ops_antigravity")
NEW_INTERPRETER = Path(
    "/Users/mac/Documents/Dataset/venvs/paper_r4_torch_antigravity_py314_v2/bin/python"
)


def _row(offset: float, *, missing_at: int | None = None) -> np.ndarray:
    values = np.arange(8, dtype="<f4") + np.float32(offset)
    if missing_at is not None:
        values = values.copy()
        values[missing_at] = np.nan
    return values


def _raw(name: str, train: tuple[np.ndarray, ...], test: tuple[np.ndarray, ...]) -> RawDataset:
    return RawDataset(
        name, 8,
        tuple(range(len(train))), tuple(row.tobytes() for row in train), train,
        tuple(range(len(test))), tuple(b"t" + bytes([index]) for index in range(len(test))), test,
    )


def _sources(n_test: int) -> HardSourceBundle:
    by_name = {}
    for name in EXPECTED_HARD_SOURCES:
        by_name[name] = HardSourceResult(
            name, object(), np.arange(8, dtype=float), np.zeros(n_test, dtype=float),
            np.ones(n_test), .01, np.zeros(n_test, dtype=bool), True, "eligible",
        )
    return HardSourceBundle(EXPECTED_HARD_SOURCES, by_name)


def _mini(n_test: int) -> MiniRocketResult:
    def kernel(requested: int, dimension: int) -> MiniRocketKernelResult:
        scores = ConfigurationScoresPaper(np.arange(8, dtype=float), np.zeros(n_test, dtype=float), dimension)
        return MiniRocketKernelResult(requested, dimension, {seed: scores for seed in SEEDS_PAPER}, frozenset())
    k1024, k4096 = kernel(1024, 1008), kernel(4096, 4032)
    return MiniRocketResult(frozenset(), frozenset(), {1024: k1024, 4096: k4096})


class PaperR4SmokeRemediationTests(unittest.TestCase):
    def test_bounded_e2e_covers_all_three_old_failure_families_without_full_population(self) -> None:
        """Production mutation caught: the three smoke families crashed the wave instead of repairing, empty-routing, or failing fast."""
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        from paper.equipment_anomaly_detection.proposal_score_paper import _collect_task4_task5

        repaired = _raw(
            "DodgerLoopDay_normal_1",
            (_row(0.0, missing_at=1), _row(1.0), _row(2.0), _row(3.0), _row(4.0)),
            (_row(10.0),),
        )
        undersized = _raw("Phoneme_normal_39", (_row(0.0),), (_row(1.0), _row(2.0)))
        torch_missing = _raw(
            "ChlorineConcentration_normal_3",
            (_row(0.0), _row(1.0), _row(2.0), _row(3.0), _row(4.0)),
            (_row(10.0),),
        )

        blocks = ConfirmationBlocks(*(
            FullFitScoreView(name, np.arange(5, dtype=float), np.zeros(1), rate)
            for name, rate in (("a", .015), ("b", .015), ("b", .01), ("c", .01))
        ))
        with patch(
            "paper.equipment_anomaly_detection.sources_paper.legacy_route_sources_paper._fullfit_primary",
            side_effect=lambda name, train, test, target: FullFitScoreView(
                name, np.arange(len(train), dtype=float), np.arange(len(test), dtype=float), .015
            ),
        ), patch(
            "paper.equipment_anomaly_detection.sources_paper.legacy_route_sources_paper._confirmation_blocks",
            return_value=blocks,
        ):
            evidence = build_legacy_route_evidence(repaired)
        self.assertEqual(evidence.n_train, 5)
        self.assertTrue(np.isnan(repaired.train_rows[0][1]))
        task4 = score_hard_sources(repaired)
        self.assertTrue(all(item.eligibility_reason == "nonfinite_input" for item in task4.by_name.values()))

        empty = replay_hard_route(undersized, _sources(2), _mini(2))
        self.assertIsInstance(empty, HardRouteResult)
        self.assertEqual(empty.n_test, 2)
        self.assertEqual(empty.automatic_hard, frozenset())

        scored: list[str] = []
        backend = FrozenTask8BackendDecision.from_acceleration_manifest(
            build_acceleration_manifest("f" * 64, logical_cpu_count=2)
        )
        with patch(
            "paper.equipment_anomaly_detection.proposal_score_paper.verify_exp84_environment_paper",
            side_effect=DependencyUnavailablePaper("dependency_unavailable:torch"),
        ), patch(
            "paper.equipment_anomaly_detection.proposal_score_paper._score_dataset_task4_task5",
            side_effect=lambda values: scored.append(values.dataset),
        ):
            with self.assertRaisesRegex(DependencyUnavailablePaper, "torch"):
                _collect_task4_task5({"ChlorineConcentration_normal_3": torch_missing}, backend)
        self.assertEqual(scored, [])

    def test_old_failed_and_verified_v2_roots_are_preserved_and_new_v3_roots_are_absent(self) -> None:
        """Production mutation caught: repaired launch must not reuse v2 smoke or the failed official ops root."""
        self.assertTrue(FAILED_SMOKE_ROOT.is_dir())
        self.assertTrue(FAILED_SMOKE_OPS_ROOT.is_dir())
        self.assertFalse(FAILED_SMOKE_ROOT.is_symlink())
        self.assertFalse(FAILED_SMOKE_OPS_ROOT.is_symlink())
        self.assertTrue(VERIFIED_V2_SMOKE_ROOT.is_dir())
        self.assertTrue(VERIFIED_V2_SMOKE_OPS_ROOT.is_dir())
        self.assertFalse(VERIFIED_V2_SMOKE_ROOT.is_symlink())
        self.assertFalse(VERIFIED_V2_SMOKE_OPS_ROOT.is_symlink())
        self.assertFalse(NEW_SMOKE_ROOT.exists())
        self.assertFalse(NEW_SMOKE_OPS_ROOT.exists())
        self.assertFalse(NEW_SMOKE_ROOT.is_symlink())
        self.assertFalse(NEW_SMOKE_OPS_ROOT.is_symlink())

    def test_union_inventory_contract_is_distinct_from_train_authority(self) -> None:
        """Production mutation caught: readiness only bound TRAIN 19/164/983/0 and missed the official TRAIN-or-TEST union of 20."""
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity.paper_r4_readiness_antigravity import (
            EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY,
            EXPECTED_UNION_DATASETS_ANTIGRAVITY,
            EXPECTED_UNION_DATASET_COUNT_ANTIGRAVITY,
            EXPECTED_UNION_TEST_ONLY_DATASET_ANTIGRAVITY,
            EXPECTED_UNION_TEST_ONLY_ROWS_ANTIGRAVITY,
            EXPECTED_UNION_TEST_ONLY_SCALARS_ANTIGRAVITY,
            UnionInventoryGateAntigravity,
        )

        self.assertEqual(len(EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY), 19)
        self.assertEqual(EXPECTED_UNION_DATASET_COUNT_ANTIGRAVITY, 20)
        self.assertEqual(len(EXPECTED_UNION_DATASETS_ANTIGRAVITY), 20)
        self.assertEqual(EXPECTED_UNION_TEST_ONLY_DATASET_ANTIGRAVITY, "MelbournePedestrian_normal_10")
        self.assertIn("MelbournePedestrian_normal_10", EXPECTED_UNION_DATASETS_ANTIGRAVITY)
        self.assertNotIn("MelbournePedestrian_normal_10", EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY)
        self.assertEqual(EXPECTED_UNION_TEST_ONLY_ROWS_ANTIGRAVITY, 3)
        self.assertEqual(EXPECTED_UNION_TEST_ONLY_SCALARS_ANTIGRAVITY, 3)
        gate = UnionInventoryGateAntigravity(
            dataset_count=20,
            datasets=EXPECTED_UNION_DATASETS_ANTIGRAVITY,
            test_only_dataset="MelbournePedestrian_normal_10",
            test_only_rows=3,
            test_only_scalars=3,
            label_access_count=0,
        )
        self.assertEqual(gate.dataset_count, 20)
        with self.assertRaises(ValueError):
            UnionInventoryGateAntigravity(
                dataset_count=19,
                datasets=EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY,
                test_only_dataset="MelbournePedestrian_normal_10",
                test_only_rows=3,
                test_only_scalars=3,
                label_access_count=0,
            )


if __name__ == "__main__":
    unittest.main()
