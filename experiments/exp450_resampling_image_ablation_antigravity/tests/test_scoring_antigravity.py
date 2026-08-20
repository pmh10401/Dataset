"""Behavioral contracts for Exp450's dataset-local A--D scoring boundary."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import pickle
import sys
import unittest
from unittest.mock import patch

import numpy as np

from paper.equipment_anomaly_detection.config_paper import FROZEN_CONFIG
from paper.equipment_anomaly_detection.contracts_paper import RawDataset
from paper.equipment_anomaly_detection.data_paper import load_dataset_values
from paper.equipment_anomaly_detection.hard_route_paper import replay_hard_route
from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import (
    score_hard_sources,
)
from paper.equipment_anomaly_detection.sources_paper.minirocket_paper import (
    MiniRocketKernelResult,
    MiniRocketResult,
)

import experiments.exp450_resampling_image_ablation_antigravity.scoring_antigravity as scoring_antigravity
from experiments.exp450_resampling_image_ablation_antigravity.contracts_antigravity import (
    Exp450BackendDecision,
    Exp450Condition,
)


BACKEND_CPU = Exp450BackendDecision("cpu", "test", False, False, 1, 1)
CONDITION_A = Exp450Condition("A", "linear", "linear")
CONDITION_B = Exp450Condition("B", "polyphase_pchip", "linear")
CONDITION_C = Exp450Condition("C", "linear", "area")
CONDITION_D = Exp450Condition("D", "polyphase_pchip", "area")


def _raw_fixture() -> RawDataset:
    grid = np.linspace(0.0, 2.0 * np.pi, 16, dtype=np.float32)
    train = tuple(
        (np.sin(grid + index / 9.0) + index / 50.0).astype(np.float32)
        for index in range(5)
    )
    test = tuple(
        (np.cos(grid + index / 7.0) + index / 40.0).astype(np.float32)
        for index in range(3)
    )
    return RawDataset(
        "fixture",
        16,
        tuple(range(len(train))),
        tuple(row.astype("<f4").tobytes() for row in train),
        train,
        tuple(range(len(test))),
        tuple(row.astype("<f4").tobytes() for row in test),
        test,
    )


def _pickle_digest(value: object) -> str:
    return sha256(pickle.dumps(value, protocol=5)).hexdigest()


class Exp450DatasetScoringTests(unittest.TestCase):
    def test_condition_a_delegates_to_exact_exp449_science(self) -> None:
        """Replacing A with an Exp450 copy must break its frozen object identity."""
        expected_sources = object()
        expected_route = object()
        with (
            patch.object(scoring_antigravity, "score_hard_sources", return_value=expected_sources),
            patch.object(scoring_antigravity, "score_minirocket_confirmations", return_value=object()),
            patch.object(scoring_antigravity, "replay_hard_route", return_value=expected_route),
        ):
            result = scoring_antigravity.score_dataset_condition_exp450_antigravity(
                _raw_fixture(), CONDITION_A, BACKEND_CPU
            )
        self.assertIs(result.sources, expected_sources)
        self.assertIs(result.route, expected_route)

    def test_condition_b_changes_alignment_only(self) -> None:
        """Using linear alignment or area images for B changes a declared factor."""
        with (
            patch.object(
                scoring_antigravity,
                "score_minirocket_confirmations_exp450_antigravity",
                return_value=object(),
            ),
            patch.object(
                scoring_antigravity,
                "replay_hard_route_exp450_antigravity",
                return_value=object(),
            ),
            patch.object(
                scoring_antigravity,
                "align_rows_exp450_antigravity",
                wraps=scoring_antigravity.align_rows_exp450_antigravity,
            ) as align,
            patch.object(
                scoring_antigravity,
                "imaging_score_pair_exp450_antigravity",
                wraps=scoring_antigravity.imaging_score_pair_exp450_antigravity,
            ) as image,
        ):
            scoring_antigravity.score_dataset_condition_exp450_antigravity(
                _raw_fixture(), CONDITION_B, BACKEND_CPU
            )
        self.assertTrue(align.call_args_list)
        self.assertTrue(image.call_args_list)
        self.assertTrue(
            all(call.kwargs["mode"] == "polyphase_pchip" for call in align.call_args_list)
        )
        self.assertTrue(
            all(call.kwargs["image_mode"] == "linear" for call in image.call_args_list)
        )

    def test_conditions_c_and_d_pass_the_declared_factor_pairs(self) -> None:
        """Swapping C/D factor values must be observable at the source boundary."""
        for condition, length_mode, image_mode in (
            (CONDITION_C, "linear", "area"),
            (CONDITION_D, "polyphase_pchip", "area"),
        ):
            with (
                patch.object(
                    scoring_antigravity,
                    "score_minirocket_confirmations_exp450_antigravity",
                    return_value=object(),
                ),
                patch.object(
                    scoring_antigravity,
                    "replay_hard_route_exp450_antigravity",
                    return_value=object(),
                ),
                patch.object(
                    scoring_antigravity,
                    "align_rows_exp450_antigravity",
                    wraps=scoring_antigravity.align_rows_exp450_antigravity,
                ) as align,
                patch.object(
                    scoring_antigravity,
                    "imaging_score_pair_exp450_antigravity",
                    wraps=scoring_antigravity.imaging_score_pair_exp450_antigravity,
                ) as image,
            ):
                scoring_antigravity.score_dataset_condition_exp450_antigravity(
                    _raw_fixture(), condition, BACKEND_CPU
                )
            self.assertTrue(
                all(call.kwargs["mode"] == length_mode for call in align.call_args_list)
            )
            self.assertTrue(
                all(call.kwargs["image_mode"] == image_mode for call in image.call_args_list)
            )

    def test_scoring_boundary_never_imports_test_label_access(self) -> None:
        """Adding TEST-label access to the dataset scorer violates the score firewall."""
        sys.modules.pop("paper.equipment_anomaly_detection.label_access_paper", None)
        with (
            patch.object(scoring_antigravity, "score_hard_sources", return_value=object()),
            patch.object(scoring_antigravity, "score_minirocket_confirmations", return_value=object()),
            patch.object(scoring_antigravity, "replay_hard_route", return_value=object()),
        ):
            scoring_antigravity.score_dataset_condition_exp450_antigravity(
                _raw_fixture(), CONDITION_A, BACKEND_CPU
            )
        self.assertNotIn("paper.equipment_anomaly_detection.label_access_paper", sys.modules)

    def test_acsf1_values_only_control_matches_frozen_and_variants_are_finite(self) -> None:
        """A control drift or incomplete B--D TEST vector must fail this bounded real check."""
        observed_sql: list[str] = []
        values = load_dataset_values(
            Path(FROZEN_CONFIG.database_path),
            "ACSF1_normal_0",
            sql_observer=observed_sql.append,
        )
        self.assertFalse(any("split='TEST' AND i.label" in query for query in observed_sql))
        expected_sources = score_hard_sources(values)
        expected_mini = MiniRocketResult(
            frozenset(),
            frozenset(),
            {
                requested: MiniRocketKernelResult(
                    requested, dimension, {}, frozenset()
                )
                for requested, dimension in ((1024, 1008), (4096, 4032))
            },
        )
        expected_route = replay_hard_route(values, expected_sources, expected_mini)
        with (
            patch.object(
                scoring_antigravity, "score_hard_sources", return_value=expected_sources
            ),
            patch.object(
                scoring_antigravity,
                "score_minirocket_confirmations",
                return_value=expected_mini,
            ),
            patch.object(
                scoring_antigravity, "replay_hard_route", return_value=expected_route
            ),
        ):
            actual = scoring_antigravity.score_dataset_condition_exp450_antigravity(
                values, CONDITION_A, BACKEND_CPU
            )
        self.assertEqual(_pickle_digest(actual.sources), _pickle_digest(expected_sources))
        self.assertEqual(_pickle_digest(actual.route), _pickle_digest(expected_route))
        for condition in (CONDITION_B, CONDITION_C, CONDITION_D):
            target = scoring_antigravity.source_metadata_paper(values).train_target_len
            train = scoring_antigravity.align_rows_exp450_antigravity(
                values.train_rows, target, mode=condition.length_mode
            )
            test = scoring_antigravity.align_rows_exp450_antigravity(
                values.test_rows, target, mode=condition.length_mode
            )
            for source_name in ("exp55_best", "exp56_best"):
                _, test_score = scoring_antigravity._historical_pair_exp450_antigravity(
                    source_name,
                    train,
                    test,
                    target,
                    condition=condition,
                    backend=BACKEND_CPU,
                )
                self.assertEqual(test_score.shape, (len(values.test_rows),))
                self.assertTrue(np.isfinite(test_score).all())


if __name__ == "__main__":
    unittest.main()
