"""Synthetic Task 5 lifecycle tests; no database or official root is touched."""
from __future__ import annotations

from dataclasses import dataclass
import tempfile
import unittest
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class _BaselineAntigravity:
    family: str


@dataclass(frozen=True)
class _Exp448Antigravity:
    target_baseline_rows: tuple[_BaselineAntigravity, ...]


class PredictionAntigravityTests(unittest.TestCase):
    def test_real_eleven_worker_scheduler_commits_reversed_completion_in_sequence_antigravity(self) -> None:
        """Break caught: allowing ProcessPool completion order to determine leaf order."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import (  # noqa: PLC0415
            schedule_synthetic_units_antigravity,
        )

        ordered, completed, provenance = schedule_synthetic_units_antigravity(tuple(range(96)))
        self.assertEqual(ordered, tuple(range(96)))
        self.assertEqual(len(completed), 96)
        self.assertNotEqual(completed, tuple(range(96)))
        self.assertEqual(provenance, {"logical_cpus": 12, "workers": 11})

    def test_prediction_inventory_commits_ordered_96_label_free_units_antigravity(self) -> None:
        """Break caught: unordered, duplicate, non-96, or label-bearing candidate commitments."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import (  # noqa: PLC0415
            PredictionDependenciesAntigravity,
            predict_with_dependencies_antigravity,
        )

        families = ("Crop", "ElectricDevices", "FordA", "FordB", "Yoga", "ECG5000", "ChlorineConcentration", "StarLightCurves")
        rows = tuple(_BaselineAntigravity(family) for family, count in zip(families, (24, 7, 2, 2, 2, 5, 3, 3), strict=True) for _ in range(count))
        def load(sequence: int, family: str):
            return family + "_" + str(sequence), np.arange(10.0).reshape(5, 2), np.arange(6.0).reshape(3, 2)
        def score(train, test, candidate, seed):
            return np.arange(len(train), dtype=float) + seed / 10_000_000, np.arange(len(test), dtype=float) + seed / 10_000_000
        with tempfile.TemporaryDirectory(prefix="exp449_prediction_antigravity_") as temporary:
            dependencies = PredictionDependenciesAntigravity(
                root=Path(temporary), exp448_authority=_Exp448Antigravity(rows), load_bundle=load,
                score=score, method_audit=lambda: (), mlx_decision=lambda: {"backend": "cpu"}, executor_factory=None,
            )
            authority = predict_with_dependencies_antigravity(dependencies, reviewed_commit="b" * 40)
            self.assertEqual(authority.family_method_pairs, 16)
            self.assertEqual(authority.dataset_units, 96)
            self.assertEqual(authority.label_access_count, 0)
            self.assertEqual(len(authority.candidate_names), 96)
            self.assertEqual(len(set(authority.candidate_names)), 96)
            self.assertEqual(authority.candidate_names[0], "Crop_0__minirocket_local_gap")
            self.assertEqual(authority.candidate_names[-1], "StarLightCurves_47__hydra_local_gap")

    def test_validator_fails_closed_without_official_root_antigravity(self) -> None:
        """Break caught: Task 3 can open a database before descriptor validation fails."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.prediction_antigravity import (  # noqa: PLC0415
            FrozenPredictionAuthorityAntigravity,
            validate_frozen_prediction_authority_antigravity,
        )

        authority = FrozenPredictionAuthorityAntigravity("c" * 40, 16, 96, (), (), (), "d" * 64, 0)
        with self.assertRaises(ValueError):
            validate_frozen_prediction_authority_antigravity(authority)
