"""Behavioral contracts for Exp472's isolated condition-E score boundary."""

from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from paper.equipment_anomaly_detection.contracts_paper import RawDataset
from paper.equipment_anomaly_detection.routing_paper import (
    build_hard_evidence_surface,
    build_policies,
)
from paper.equipment_anomaly_detection.sources_paper.low_frequency_paper import (
    score_low_frequency,
)
from paper.equipment_anomaly_detection.sources_paper.stability_paper import (
    classify_stability,
)

import experiments.exp450_resampling_image_ablation_antigravity.scoring_antigravity as exp450_scoring
from experiments.exp450_resampling_image_ablation_antigravity.artifacts_antigravity import (
    canonical_result_bytes_antigravity,
)
from experiments.exp450_resampling_image_ablation_antigravity.contracts_antigravity import (
    Exp450Condition,
)
from experiments.exp450_resampling_image_ablation_antigravity.scoring_antigravity import (
    score_dataset_condition_exp450_antigravity,
    score_dataset_condition_with_alignment_exp450_antigravity,
)
from experiments.exp450_resampling_image_ablation_antigravity.imaging_antigravity import (
    imaging_score_pair_exp450_antigravity,
)
from experiments.exp450_resampling_image_ablation_antigravity.transforms_antigravity import (
    align_rows_exp450_antigravity,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.contracts_antigravity import (
    Exp472BackendDecisionAntigravity,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.transforms_antigravity import (
    align_rows_exp472_antigravity,
)
import experiments.exp472_makima_upsampling_ablation_antigravity.scoring_antigravity as scoring


BACKEND_CPU = Exp472BackendDecisionAntigravity(
    pytorch_available=False,
    pytorch_version="unavailable",
    scipy_available=True,
    scipy_version="test",
    mlx_available=False,
    mlx_version="unavailable",
    mlx_metal_available=False,
    mlx_metal_version="unavailable",
    mps_available=False,
    tensor_backend="cpu",
    logical_cpu_count=1,
    cpu_workers=1,
)
CONDITION_D = Exp450Condition("D", "polyphase_pchip", "area")


def _raw_fixture(dataset: str, row_length: int) -> RawDataset:
    train = tuple(
        (
            np.sin(np.linspace(0.0, 3.0 * np.pi, row_length, dtype=np.float32) + index / 7.0)
            + index / 20.0
        ).astype("<f4")
        for index in range(5)
    )
    test = tuple(
        (
            np.cos(np.linspace(0.0, 3.0 * np.pi, row_length, dtype=np.float32) + index / 5.0)
            + index / 25.0
        ).astype("<f4")
        for index in range(3)
    )
    return RawDataset(
        dataset,
        16,
        tuple(range(len(train))),
        tuple(row.tobytes() for row in train),
        train,
        tuple(range(len(test))),
        tuple(row.tobytes() for row in test),
        test,
    )


RAW_FIXTURE = _raw_fixture("exp472_upsample_fixture", 12)
DOWNSAMPLE_FIXTURE = _raw_fixture("exp472_downsample_fixture", 24)


class Exp472ScoringTests(unittest.TestCase):
    def test_e_routes_all_alignment_through_exp472_and_keeps_area_images(self) -> None:
        """Falling back to Exp450 alignment or linear images changes E factors."""
        with (
            patch.object(
                scoring,
                "align_rows_exp472_antigravity",
                wraps=align_rows_exp472_antigravity,
            ) as align,
            patch.object(
                exp450_scoring,
                "imaging_score_pair_exp450_antigravity",
                wraps=imaging_score_pair_exp450_antigravity,
            ) as image,
        ):
            scoring.score_dataset_condition_e_exp472_antigravity(
                RAW_FIXTURE, BACKEND_CPU
            )
        self.assertGreater(align.call_count, 0)
        self.assertGreater(image.call_count, 0)
        self.assertTrue(
            all(call.kwargs["image_mode"] == "area" for call in image.call_args_list)
        )

    def test_scoring_module_has_no_label_import(self) -> None:
        """A TEST-label import before evaluation violates the score firewall."""
        source = Path(scoring.__file__).read_text(encoding="utf-8")
        self.assertNotIn("label_access", source)
        self.assertNotIn("load_dataset_labels", source)

    def test_downsample_only_e_equals_exp450_d(self) -> None:
        """Routing a downsample-only input to Makima would break D control parity."""
        e_result = score_dataset_condition_with_alignment_exp450_antigravity(
            DOWNSAMPLE_FIXTURE,
            CONDITION_D,
            BACKEND_CPU,
            align_rows_override=align_rows_exp472_antigravity,
        )
        d_result = score_dataset_condition_exp450_antigravity(
            DOWNSAMPLE_FIXTURE, CONDITION_D, BACKEND_CPU
        )
        self.assertEqual(
            canonical_result_bytes_antigravity(e_result),
            canonical_result_bytes_antigravity(d_result),
        )

    def test_only_true_upsampling_changes_e_from_d(self) -> None:
        """Applying Makima to an upsample must differ from D before source scoring."""
        makima = align_rows_exp472_antigravity(RAW_FIXTURE.train_rows, 16)
        pchip = align_rows_exp450_antigravity(
            RAW_FIXTURE.train_rows, 16, mode="polyphase_pchip"
        )
        self.assertFalse(np.array_equal(makima, pchip))

    def test_per_dataset_composition_equals_explicit_full_map_composition(self) -> None:
        """Dropping another dataset from policy construction must alter its global route."""
        other = _raw_fixture("exp472_second_upsample_fixture", 12)
        first_result = scoring.score_dataset_condition_e_exp472_antigravity(
            RAW_FIXTURE, BACKEND_CPU
        )
        second_result = scoring.score_dataset_condition_e_exp472_antigravity(
            other, BACKEND_CPU
        )
        raw = {RAW_FIXTURE.dataset: RAW_FIXTURE, other.dataset: other}
        routes = {
            RAW_FIXTURE.dataset: first_result.route,
            other.dataset: second_result.route,
        }
        sources = {
            RAW_FIXTURE.dataset: first_result.sources,
            other.dataset: second_result.sources,
        }
        hard = {dataset: route.exp253.hard for dataset, route in routes.items()}
        low = score_low_frequency(raw, hard)
        stability = classify_stability(low, raw)
        surface = build_hard_evidence_surface(sources, routes, raw)
        policies = build_policies(surface, low, stability, raw, routes)
        expected = next(
            policy for policy in policies.policies if policy.name == "hard_plus_stable"
        )
        actual = scoring.compose_final_score_exp472_antigravity(
            RAW_FIXTURE, first_result, BACKEND_CPU
        )
        np.testing.assert_array_equal(
            actual.scores, np.asarray(expected.scores[RAW_FIXTURE.dataset], dtype=np.float64)
        )
        self.assertEqual(actual.alerts, expected.alerts[RAW_FIXTURE.dataset])


if __name__ == "__main__":
    unittest.main()
