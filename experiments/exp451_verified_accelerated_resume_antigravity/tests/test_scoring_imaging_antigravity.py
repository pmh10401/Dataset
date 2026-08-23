"""RED/GREEN and mutation guard tests for Exp451 scientific scoring and imaging closure."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from paper.equipment_anomaly_detection.contracts_paper import RawDataset
from paper.equipment_anomaly_detection.hard_route_paper import (
    HardRouteResult,
    replay_hard_route,
)
from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import (
    EXPECTED_HARD_SOURCES,
    HardSourceBundle,
    score_hard_sources,
)
from paper.equipment_anomaly_detection.sources_paper.imaging_paper import (
    EXP55_CONFIG_PAPER,
    EXP56_CONFIG_PAPER,
)
from paper.equipment_anomaly_detection.sources_paper.legacy_route_sources_paper import (
    ConfirmationBlocks,
    FullFitScoreView,
)
from paper.equipment_anomaly_detection.sources_paper.minirocket_paper import (
    SEEDS_PAPER,
    ConfigurationScoresPaper,
    score_minirocket_confirmations,
)
from paper.equipment_anomaly_detection.sources_paper.preprocessing_paper import (
    DependencyUnavailablePaper,
)

import experiments.exp451_verified_accelerated_resume_antigravity.imaging_antigravity as imaging_antigravity
from experiments.exp451_verified_accelerated_resume_antigravity.imaging_antigravity import (
    FROZEN_IMAGING_SOURCE_ANTIGRAVITY,
    PROVENANCE_ANTIGRAVITY,
    imaging_score_pair_exp451_antigravity,
    recurrence_images_exp451_antigravity,
    spectrogram_images_exp451_antigravity,
)
import experiments.exp451_verified_accelerated_resume_antigravity.scoring_antigravity as scoring_antigravity
from experiments.exp451_verified_accelerated_resume_antigravity.contracts_antigravity import (
    Exp451BackendDecision,
    Exp451Condition,
    Exp451DatasetConditionResult,
)
from experiments.exp451_verified_accelerated_resume_antigravity.transforms_antigravity import (
    adaptive_average_1d_antigravity,
    adaptive_average_2d_antigravity,
)


BACKEND_CPU = Exp451BackendDecision(
    tensor_backend="cpu",
    torch_version="2.13.0",
    mps_available=False,
    mlx_available=False,
    logical_cpu_count=8,
    cpu_workers=7,
)

CONDITION_A = Exp451Condition("A", "linear", "linear")
CONDITION_B = Exp451Condition("B", "polyphase_pchip", "linear")
CONDITION_C = Exp451Condition("C", "linear", "area")
CONDITION_D = Exp451Condition("D", "polyphase_pchip", "area")

TRAIN_MATRIX = np.asarray(
    [
        [0.0, 1.0, -1.0, 2.0, 0.5, -0.5, 3.0, -2.0, 1.5],
        [2.0, -1.0, 0.0, 1.0, -2.0, 0.5, 1.5, -0.5, 0.0],
        [-1.5, 0.0, 2.0, -0.5, 1.0, 0.5, -1.0, 2.5, 0.0],
        [0.5, 1.5, -0.5, -1.0, 2.0, 0.0, 1.0, -2.0, 0.5],
    ],
    dtype=np.float32,
)
TEST_MATRIX = np.asarray(
    [[1.0, 0.0, -1.0, 2.0, 0.0, -0.5, 1.0, 0.5, -1.5]], dtype=np.float32
)


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


class Exp451ImagingTests(unittest.TestCase):
    def test_linear_imaging_delegates_to_frozen_exp449(self) -> None:
        """Changing the linear control to a copied pipeline must fail."""
        expected = (np.asarray([1.0]), np.asarray([2.0]))
        with patch.object(
            imaging_antigravity, "imaging_score_pair_paper", return_value=expected
        ) as frozen:
            actual = imaging_score_pair_exp451_antigravity(
                TRAIN_MATRIX, TEST_MATRIX, EXP55_CONFIG_PAPER, image_mode="linear", device="cpu"
            )
        frozen.assert_called_once_with(TRAIN_MATRIX, TEST_MATRIX, EXP55_CONFIG_PAPER)
        self.assertEqual(actual[1].tolist(), [2.0])

    def test_area_recurrence_recomputes_symmetric_image_after_exp451_pooling(self) -> None:
        """Area recurrence must invoke Exp451 adaptive pooling and produce a symmetric image."""
        size = 4
        with patch(
            "experiments.exp451_verified_accelerated_resume_antigravity.imaging_antigravity.adaptive_average_1d_antigravity",
            wraps=adaptive_average_1d_antigravity,
        ) as pool_spy:
            images = recurrence_images_exp451_antigravity(TRAIN_MATRIX, size, device="cpu")
        pool_spy.assert_called_once()
        fixed = adaptive_average_1d_antigravity(TRAIN_MATRIX, size, device="cpu")
        distance = np.abs(fixed[:, :, None] - fixed[:, None, :])
        scale = np.quantile(distance.reshape(len(TRAIN_MATRIX), -1), 0.75, axis=1)
        expected = np.exp(-distance / np.maximum(scale, 1e-6)[:, None, None]).astype(
            np.float32
        )
        self.assertEqual(images.shape, (len(TRAIN_MATRIX), size, size))
        self.assertEqual(images.dtype, np.dtype(np.float32))
        np.testing.assert_allclose(images, expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(images, images.transpose(0, 2, 1))

    def test_area_spectrogram_uses_exp451_2d_pooling_and_no_linear_resize(self) -> None:
        """Spectrogram area mode must invoke Exp451 2D pooling without np.interp linear resize."""
        with (
            patch(
                "experiments.exp451_verified_accelerated_resume_antigravity.imaging_antigravity.adaptive_average_2d_antigravity",
                wraps=adaptive_average_2d_antigravity,
            ) as pool_spy,
            patch.object(np, "interp", side_effect=AssertionError("linear resize called")),
        ):
            images = spectrogram_images_exp451_antigravity(TRAIN_MATRIX, 32, device="cpu")
        pool_spy.assert_called_once()
        self.assertEqual(images.shape, (len(TRAIN_MATRIX), 32, 32))
        self.assertEqual(images.dtype, np.dtype(np.float32))
        self.assertTrue(np.isfinite(images).all())

    def test_area_score_pair_preserves_frozen_glcm_pca_scaling_and_knn_closure(self) -> None:
        """Area mode must retain the frozen Exp56 estimator stages after imaging changes."""
        train_scores, test_scores = imaging_score_pair_exp451_antigravity(
            TRAIN_MATRIX, TEST_MATRIX, EXP56_CONFIG_PAPER, image_mode="area", device="cpu"
        )
        self.assertEqual(train_scores.shape, (len(TRAIN_MATRIX),))
        self.assertEqual(test_scores.shape, (len(TEST_MATRIX),))
        self.assertTrue(np.isfinite(train_scores).all())
        self.assertTrue(np.isfinite(test_scores).all())

    def test_frozen_imaging_source_provenance_is_recorded(self) -> None:
        """Changing the copied frozen closure without its source identity must fail."""
        self.assertEqual(
            FROZEN_IMAGING_SOURCE_ANTIGRAVITY,
            {
                "path": "paper/equipment_anomaly_detection/sources_paper/imaging_paper.py",
                "sha256": "1c18b6c6e3f70c922f9bba9689d26b99d2eb6320864e9c6425727af6e95d61e2",
            },
        )
        self.assertEqual(
            PROVENANCE_ANTIGRAVITY,
            {"frozen_imaging_source": FROZEN_IMAGING_SOURCE_ANTIGRAVITY},
        )

    def test_mps_area_images_complete_without_cpu_parity_when_available(self) -> None:
        """MPS completion needs finite area images, not equality with CPU output."""
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable")
        recurrence = recurrence_images_exp451_antigravity(TRAIN_MATRIX, 32, device="mps")
        spectrogram = spectrogram_images_exp451_antigravity(TRAIN_MATRIX, 32, device="mps")
        self.assertEqual(recurrence.shape, (len(TRAIN_MATRIX), 32, 32))
        self.assertEqual(spectrogram.shape, (len(TRAIN_MATRIX), 32, 32))
        self.assertTrue(np.isfinite(recurrence).all())
        self.assertTrue(np.isfinite(spectrogram).all())
        np.testing.assert_array_equal(recurrence, recurrence.transpose(0, 2, 1))

    def test_mps_area_public_c_d_image_paths_handle_mixed_axis_short_inputs(self) -> None:
        """C/D's frozen Exp55/56 area paths must finish on MPS for 64-sample rows."""
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable")
        grid = np.linspace(0.0, 2.0 * np.pi, 64, dtype=np.float32)
        train = np.stack([np.sin(grid + offset) for offset in range(4)]).astype(np.float32)
        test = np.cos(grid, dtype=np.float32).reshape(1, -1)
        for config in (EXP55_CONFIG_PAPER, EXP56_CONFIG_PAPER):
            with self.subTest(config=config["name"]):
                train_score, test_score = imaging_score_pair_exp451_antigravity(
                    train, test, config, image_mode="area", device="mps"
                )
                self.assertEqual(train_score.shape, (len(train),))
                self.assertEqual(test_score.shape, (len(test),))
                self.assertTrue(np.isfinite(train_score).all())
                self.assertTrue(np.isfinite(test_score).all())

    def test_invalid_image_mode_raises_value_error(self) -> None:
        """Unknown image mode raises ValueError."""
        with self.assertRaises(ValueError):
            imaging_score_pair_exp451_antigravity(
                TRAIN_MATRIX, TEST_MATRIX, EXP55_CONFIG_PAPER, image_mode="invalid", device="cpu"
            )


class Exp451ScoringFactorPreservationTests(unittest.TestCase):
    def test_condition_factor_mapping_is_immutable(self) -> None:
        """Condition factorial mapping must strictly adhere to A-D definitions."""
        self.assertEqual((CONDITION_A.length_mode, CONDITION_A.image_mode), ("linear", "linear"))
        self.assertEqual((CONDITION_B.length_mode, CONDITION_B.image_mode), ("polyphase_pchip", "linear"))
        self.assertEqual((CONDITION_C.length_mode, CONDITION_C.image_mode), ("linear", "area"))
        self.assertEqual((CONDITION_D.length_mode, CONDITION_D.image_mode), ("polyphase_pchip", "area"))

    def test_condition_a_delegates_to_exact_exp449_science(self) -> None:
        """Condition A must call the frozen paper closures directly."""
        expected_sources = object()
        expected_route = object()
        with (
            patch.object(scoring_antigravity, "score_hard_sources", return_value=expected_sources),
            patch.object(scoring_antigravity, "score_minirocket_confirmations", return_value=object()),
            patch.object(scoring_antigravity, "replay_hard_route", return_value=expected_route),
        ):
            result = scoring_antigravity.score_dataset_condition_exp451_antigravity(
                _raw_fixture(), CONDITION_A, BACKEND_CPU
            )
        self.assertIs(result.sources, expected_sources)
        self.assertIs(result.route, expected_route)
        self.assertEqual(result.dataset, "fixture")
        self.assertEqual(result.condition_id, "A")

    def test_condition_b_changes_alignment_only_and_retains_linear_image_control(self) -> None:
        """Condition B must use polyphase_pchip alignment and paper linear imaging control."""
        with (
            patch.object(
                scoring_antigravity,
                "score_minirocket_confirmations_exp451_antigravity",
                return_value=object(),
            ),
            patch.object(
                scoring_antigravity,
                "replay_hard_route_exp451_antigravity",
                return_value=object(),
            ),
            patch.object(
                scoring_antigravity,
                "_canonical_source_paper",
                return_value=(np.zeros(5), np.zeros(3)),
            ),
            patch.object(
                scoring_antigravity,
                "align_rows_exp451_antigravity",
                wraps=scoring_antigravity.align_rows_exp451_antigravity,
            ) as align,
            patch.object(
                scoring_antigravity,
                "imaging_score_pair_exp451_antigravity",
                wraps=scoring_antigravity.imaging_score_pair_exp451_antigravity,
            ) as image,
            patch.object(
                imaging_antigravity,
                "imaging_score_pair_paper",
                side_effect=lambda train, test, config: (np.zeros(len(train)), np.zeros(len(test))),
            ) as paper_imaging,
        ):
            scoring_antigravity.score_dataset_condition_exp451_antigravity(
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
        self.assertTrue(paper_imaging.called)

    def test_conditions_c_and_d_pass_the_declared_factor_pairs_and_call_area_pooling(self) -> None:
        """Conditions C and D must pass declared factor pairs and execute Exp451 area pooling."""
        for condition, length_mode, image_mode in (
            (CONDITION_C, "linear", "area"),
            (CONDITION_D, "polyphase_pchip", "area"),
        ):
            with (
                patch.object(
                    scoring_antigravity,
                    "score_minirocket_confirmations_exp451_antigravity",
                    return_value=object(),
                ),
                patch.object(
                    scoring_antigravity,
                    "replay_hard_route_exp451_antigravity",
                    return_value=object(),
                ),
                patch.object(
                    scoring_antigravity,
                    "_canonical_source_paper",
                    return_value=(np.zeros(5), np.zeros(3)),
                ),
                patch.object(
                    scoring_antigravity,
                    "align_rows_exp451_antigravity",
                    wraps=scoring_antigravity.align_rows_exp451_antigravity,
                ) as align,
                patch.object(
                    scoring_antigravity,
                    "imaging_score_pair_exp451_antigravity",
                    wraps=scoring_antigravity.imaging_score_pair_exp451_antigravity,
                ) as image,
                patch(
                    "experiments.exp451_verified_accelerated_resume_antigravity.imaging_antigravity.adaptive_average_2d_antigravity",
                    wraps=adaptive_average_2d_antigravity,
                ) as pool2d_spy,
                patch(
                    "experiments.exp451_verified_accelerated_resume_antigravity.imaging_antigravity.adaptive_average_1d_antigravity",
                    wraps=adaptive_average_1d_antigravity,
                ) as pool1d_spy,
            ):
                scoring_antigravity.score_dataset_condition_exp451_antigravity(
                    _raw_fixture(), condition, BACKEND_CPU
                )
            self.assertTrue(
                all(call.kwargs["mode"] == length_mode for call in align.call_args_list)
            )
            self.assertTrue(
                all(call.kwargs["image_mode"] == image_mode for call in image.call_args_list)
            )
            # Area mode in Exp55 (spectrogram) and Exp56 (RP) must call Exp451 adaptive pooling
            self.assertTrue(pool2d_spy.called)
            self.assertTrue(pool1d_spy.called)

    def test_scoring_boundary_never_imports_test_label_access(self) -> None:
        """Scoring boundary must remain strictly label-free (firewall check)."""
        sys.modules.pop("paper.equipment_anomaly_detection.label_access_paper", None)
        with (
            patch.object(scoring_antigravity, "score_hard_sources", return_value=object()),
            patch.object(scoring_antigravity, "score_minirocket_confirmations", return_value=object()),
            patch.object(scoring_antigravity, "replay_hard_route", return_value=object()),
        ):
            scoring_antigravity.score_dataset_condition_exp451_antigravity(
                _raw_fixture(), CONDITION_A, BACKEND_CPU
            )
        self.assertNotIn("paper.equipment_anomaly_detection.label_access_paper", sys.modules)

    def test_score_minirocket_exp451_alignment_and_factors(self) -> None:
        """score_minirocket_exp451 delegates A to paper and applies factor alignment for B-D."""
        fixture = _raw_fixture()
        # Condition A delegates
        with patch.object(
            scoring_antigravity, "score_minirocket_confirmations", return_value=MagicMock()
        ) as mock_mini:
            scoring_antigravity.score_minirocket_exp451_antigravity(fixture, CONDITION_A, BACKEND_CPU)
            mock_mini.assert_called_once_with(fixture)

        # Condition B uses polyphase_pchip alignment
        with patch(
            "experiments.exp451_verified_accelerated_resume_antigravity.scoring_antigravity.score_configuration_paper",
            side_effect=lambda train, test, config, requested_kernels, seed, expected_dimension: ConfigurationScoresPaper(
                np.zeros(len(train)), np.zeros(len(test)), expected_dimension
            ),
        ), patch.object(
            scoring_antigravity,
            "align_rows_exp451_antigravity",
            wraps=scoring_antigravity.align_rows_exp451_antigravity,
        ) as align_spy:
            res_b = scoring_antigravity.score_minirocket_exp451_antigravity(fixture, CONDITION_B, BACKEND_CPU)
            self.assertTrue(all(call.kwargs["mode"] == "polyphase_pchip" for call in align_spy.call_args_list))
            self.assertEqual(set(res_b.by_kernel), {1024, 4096})

        # Train rows < 5 returns empty result
        short_raw = RawDataset("short", 16, (0, 1), (b"", b""), (np.zeros(16), np.zeros(16)), (), (), ())
        empty_mini = scoring_antigravity.score_minirocket_exp451_antigravity(short_raw, CONDITION_B, BACKEND_CPU)
        self.assertEqual(empty_mini.k1024, frozenset())
        self.assertEqual(empty_mini.k4096, frozenset())

    def test_score_hard_sources_exp451_guards_small_and_nonfinite(self) -> None:
        """score_hard_sources_exp451 handles < 5 train rows and non-finite values safely."""
        short_raw = RawDataset("short", 16, (0, 1), (b"", b""), (np.zeros(16), np.zeros(16)), (0,), (b"",), (np.zeros(16),))
        bundle = scoring_antigravity.score_hard_sources_exp451_antigravity(short_raw, CONDITION_C, BACKEND_CPU)
        self.assertEqual(bundle.source_names, EXPECTED_HARD_SOURCES)
        self.assertTrue(all(not res.eligible and res.eligibility_reason == "unsupported_n_train_lt_5" for res in bundle.by_name.values()))

        nan_rows = (np.full(16, np.nan, dtype=np.float32),) * 5
        nan_raw = RawDataset("nan", 16, tuple(range(5)), tuple(r.tobytes() for r in nan_rows), nan_rows, (0,), (b"",), (np.zeros(16),))
        nan_bundle = scoring_antigravity.score_hard_sources_exp451_antigravity(nan_raw, CONDITION_C, BACKEND_CPU)
        self.assertTrue(all(not res.eligible and res.eligibility_reason == "nonfinite_input" for res in nan_bundle.by_name.values()))

    def test_build_legacy_route_evidence_exp451_bounds_and_primary_sources(self) -> None:
        """build_legacy_route_evidence_exp451 verifies input bounds and builds primary source views."""
        short_raw = RawDataset("short", 16, (0,), (b"",), (np.zeros(16),), (), (), ())
        with self.assertRaises(ValueError):
            scoring_antigravity.build_legacy_route_evidence_exp451_antigravity(short_raw, CONDITION_C, BACKEND_CPU)

        mock_primary = {
            "rocket_exp40": FullFitScoreView("rocket_exp40", np.zeros(5), np.zeros(3), 0.015),
            "exp55_best": FullFitScoreView("exp55_best", np.zeros(5), np.zeros(3), 0.015),
            "exp56_best": FullFitScoreView("exp56_best", np.zeros(5), np.zeros(3), 0.015),
        }
        mock_exp84 = FullFitScoreView("exp84", np.zeros(5), np.zeros(3), 0.015)
        mock_blocks = ConfirmationBlocks(
            FullFitScoreView("block_a", np.zeros(5), np.zeros(3), 0.015),
            FullFitScoreView("block_b", np.zeros(5), np.zeros(3), 0.015),
            FullFitScoreView("block_b", np.zeros(5), np.zeros(3), 0.01),
            FullFitScoreView("block_c", np.zeros(5), np.zeros(3), 0.01),
        )
        with patch.object(
            scoring_antigravity,
            "_fullfit_primary_exp451_antigravity",
            side_effect=lambda name, *args, **kw: mock_primary.get(name, mock_exp84),
        ), patch.object(
            scoring_antigravity,
            "_confirmation_blocks_exp451_antigravity",
            return_value=mock_blocks,
        ):
            evidence = scoring_antigravity.build_legacy_route_evidence_exp451_antigravity(_raw_fixture(), CONDITION_C, BACKEND_CPU)
            self.assertEqual(evidence.dataset, "fixture")
            self.assertEqual(evidence.n_train, 5)
            self.assertEqual(evidence.n_test, 3)

    def test_fullfit_primary_exp451_delegates_to_imaging_with_condition_and_device(self) -> None:
        """Fullfit primary routes must delegate exp55 and exp56 to imaging_score_pair_exp451_antigravity."""
        train = np.ones((4, 16), dtype=np.float32)
        test = np.ones((2, 16), dtype=np.float32)
        with patch.object(
            scoring_antigravity,
            "imaging_score_pair_exp451_antigravity",
            return_value=(np.zeros(4), np.zeros(2)),
        ) as mock_img:
            view = scoring_antigravity._fullfit_primary_exp451_antigravity(
                "exp55_best", train, test, 16, condition=CONDITION_C, backend=BACKEND_CPU
            )
            mock_img.assert_called_once_with(
                train, test, EXP55_CONFIG_PAPER, image_mode="area", device="cpu"
            )
            self.assertEqual(view.name, "exp55_best")
            self.assertEqual(view.train_score.shape, (4,))
            self.assertEqual(view.test_score.shape, (2,))

    def test_replay_hard_route_empty_on_insufficient_rows(self) -> None:
        """Replay hard route returns empty result when rows are insufficient."""
        small_raw = RawDataset("small", 8, (0,), (b"",), (np.zeros(8),), (), (), ())
        res = scoring_antigravity.replay_hard_route_exp451_antigravity(
            small_raw, MagicMock(), MagicMock(), CONDITION_B, BACKEND_CPU
        )
        self.assertEqual(res.n_test, 0)

    def test_score_dataset_condition_exp451_returns_typed_result(self) -> None:
        """score_dataset_condition_exp451 produces typed Exp451DatasetConditionResult."""
        mock_sources = MagicMock(spec=HardSourceBundle)
        mock_route = MagicMock(spec=HardRouteResult)
        with patch.object(
            scoring_antigravity, "score_hard_sources_exp451_antigravity", return_value=mock_sources
        ), patch.object(
            scoring_antigravity, "score_minirocket_confirmations_exp451_antigravity", return_value=MagicMock()
        ), patch.object(
            scoring_antigravity, "replay_hard_route_exp451_antigravity", return_value=mock_route
        ):
            res = scoring_antigravity.score_dataset_condition_exp451_antigravity(
                _raw_fixture(), CONDITION_C, BACKEND_CPU
            )
            self.assertIsInstance(res, Exp451DatasetConditionResult)
            self.assertEqual(res.dataset, "fixture")
            self.assertEqual(res.condition_id, "C")
            self.assertIs(res.sources, mock_sources)
            self.assertIs(res.route, mock_route)

    def test_mutation_guard_invalid_condition_raises_error(self) -> None:
        """Mutating factorial parameters on Condition raises ValueError."""
        with self.assertRaises(ValueError):
            Exp451Condition("A", "polyphase_pchip", "linear")
        with self.assertRaises(ValueError):
            Exp451Condition("B", "linear", "linear")
        with self.assertRaises(ValueError):
            Exp451Condition("C", "linear", "linear")
        with self.assertRaises(ValueError):
            Exp451Condition("D", "polyphase_pchip", "linear")

    def test_public_closure_is_exp451_only(self) -> None:
        """The new closure must not expose Exp450-named aliases as its own API."""
        self.assertTrue(
            all("exp450" not in name for name in imaging_antigravity.__all__)
        )
        self.assertTrue(
            all("exp450" not in name for name in scoring_antigravity.__all__)
        )


if __name__ == "__main__":
    unittest.main()
