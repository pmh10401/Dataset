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
    SEEDS_PAPER,
    score_minirocket_confirmations,
)

import experiments.exp450_resampling_image_ablation_antigravity.scoring_antigravity as scoring_antigravity
from experiments.exp450_resampling_image_ablation_antigravity.contracts_antigravity import (
    Exp450BackendDecision,
    Exp450Condition,
    Exp450DatasetConditionResult,
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

    def test_acsf1_unmocked_control_matches_independent_frozen_outputs(self) -> None:
        """Replacing A with mocked or altered science must break independent bytes."""
        observed_sql: list[str] = []
        values = load_dataset_values(
            Path(FROZEN_CONFIG.database_path),
            "ACSF1_normal_0",
            sql_observer=observed_sql.append,
        )
        self.assertFalse(any("split='TEST' AND i.label" in query for query in observed_sql))
        expected_sources = score_hard_sources(values)
        expected_mini = score_minirocket_confirmations(values)
        expected_route = replay_hard_route(values, expected_sources, expected_mini)
        actual = scoring_antigravity.score_dataset_condition_exp450_antigravity(
            values, CONDITION_A, BACKEND_CPU
        )
        self.assertIsInstance(actual, Exp450DatasetConditionResult)
        self.assertEqual(actual.dataset, values.dataset)
        self.assertEqual(actual.condition_id, "A")
        self.assertEqual(_pickle_digest(actual.sources), _pickle_digest(expected_sources))
        self.assertEqual(_pickle_digest(actual.route), _pickle_digest(expected_route))

    def test_acsf1_unmocked_bcd_full_paths_return_complete_dataset_results(self) -> None:
        """Bypassing B--D's public route integration must fail these final invariants."""
        values = load_dataset_values(
            Path(FROZEN_CONFIG.database_path),
            "ACSF1_normal_0",
            sql_observer=lambda query: self.assertNotIn("split='TEST' AND i.label", query),
        )
        n_test = len(values.test_rows)
        for condition in (CONDITION_B, CONDITION_C, CONDITION_D):
            captured_mini = []
            real_minirocket = (
                scoring_antigravity.score_minirocket_confirmations_exp450_antigravity
            )

            def capture_minirocket(*arguments, **keywords):
                mini = real_minirocket(*arguments, **keywords)
                captured_mini.append(mini)
                return mini

            with patch.object(
                scoring_antigravity,
                "score_minirocket_confirmations_exp450_antigravity",
                side_effect=capture_minirocket,
            ):
                result = scoring_antigravity.score_dataset_condition_exp450_antigravity(
                    values, condition, BACKEND_CPU
                )
            self.assertEqual(len(captured_mini), 1)
            mini = captured_mini[0]
            self.assertEqual(set(mini.by_kernel), {1024, 4096})
            for requested, expected_dimension, candidates in (
                (1024, 1008, mini.k1024),
                (4096, 4032, mini.k4096),
            ):
                kernel = mini.by_kernel[requested]
                self.assertEqual(kernel.requested_kernels, requested)
                self.assertEqual(kernel.expected_dimension, expected_dimension)
                self.assertEqual(kernel.candidate, candidates)
                self.assertEqual(set(kernel.by_seed), set(SEEDS_PAPER))
                self.assertLessEqual(len(candidates), n_test)
                self.assertTrue(all(0 <= index < n_test for index in candidates))
                for score in kernel.by_seed.values():
                    self.assertEqual(score.train_oof.shape, (len(values.train_rows),))
                    self.assertEqual(score.test_score.shape, (n_test,))
                    self.assertTrue(np.isfinite(score.train_oof).all())
                    self.assertTrue(np.isfinite(score.test_score).all())
            self.assertIsInstance(result, Exp450DatasetConditionResult)
            self.assertEqual((result.dataset, result.condition_id), (values.dataset, condition.condition_id))
            self.assertEqual(result.sources.source_names, (
                "rocket_exp40",
                "exp55_best",
                "exp56_best",
                "exp84",
                "rocket_block_b",
                "canonical_normal_state",
            ))
            for source in result.sources.by_name.values():
                if source.eligible:
                    self.assertEqual(source.test_score.shape, (n_test,))
                    self.assertTrue(np.isfinite(source.test_score).all())
            route = result.route
            self.assertEqual(route.n_test, n_test)
            self.assertEqual(
                route.promoted,
                frozenset(
                    set(route.exp236.standard) & (set(mini.k1024) | set(mini.k4096))
                ),
            )
            self.assertEqual(route.exp253.hard, route.exp236.hard | route.promoted)
            self.assertEqual(route.exp253.standard, route.exp236.standard - route.promoted)
            self.assertEqual(route.exp253.priority, route.exp236.priority)
            for indices in (
                route.exp253.hard,
                route.exp253.standard,
                route.exp253.priority,
                route.promoted,
                route.automatic_hard,
                route.exact_normal_removed,
            ):
                self.assertLessEqual(len(indices), n_test)
                self.assertTrue(all(0 <= index < n_test for index in indices))


if __name__ == "__main__":
    unittest.main()
