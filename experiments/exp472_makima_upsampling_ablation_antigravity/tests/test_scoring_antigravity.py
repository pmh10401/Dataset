"""Behavioral contracts for Exp472's isolated condition-E score boundary."""

from __future__ import annotations

import json
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
    canonical_json_bytes_exp472_antigravity,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.transforms_antigravity import (
    align_rows_exp472_antigravity,
    count_upsampled_rows_exp472_antigravity,
    dataset_shape_diagnostic_exp472_antigravity,
    test_identity_exp472_antigravity,
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


def _mixed_length_fixture(dataset: str) -> RawDataset:
    train_source = _raw_fixture(f"{dataset}_train", 12)
    test_source = _raw_fixture(f"{dataset}_test", 4)
    return RawDataset(
        dataset,
        16,
        train_source.train_indices,
        train_source.train_blobs,
        train_source.train_rows,
        test_source.test_indices,
        test_source.test_blobs,
        test_source.test_rows,
    )


def _science_bytes(result: object) -> bytes:
    payload = json.loads(canonical_result_bytes_antigravity(result))
    return (
        json.dumps(
            payload["payload"],
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
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
            patch.object(
                exp450_scoring,
                "align_rows_exp450_antigravity",
                wraps=align_rows_exp450_antigravity,
            ) as default_align,
        ):
            scoring.score_dataset_condition_e_exp472_antigravity(
                RAW_FIXTURE, BACKEND_CPU
            )
        self.assertGreater(align.call_count, 0)
        self.assertEqual(default_align.call_count, 0)
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
        e_result = scoring.score_dataset_condition_e_exp472_antigravity(
            DOWNSAMPLE_FIXTURE, BACKEND_CPU
        )
        d_result = score_dataset_condition_exp450_antigravity(
            DOWNSAMPLE_FIXTURE, CONDITION_D, BACKEND_CPU
        )
        self.assertEqual(e_result.condition_id, "E")
        self.assertEqual(d_result.condition_id, "D")
        self.assertEqual(_science_bytes(e_result), _science_bytes(d_result))

    def test_only_true_upsampling_changes_e_from_d(self) -> None:
        """E must equal D-plus-Makima and differ from D on scored short TEST rows."""
        values = _mixed_length_fixture("exp472_scoring_upsample_fixture")
        e_result = scoring.score_dataset_condition_e_exp472_antigravity(
            values, BACKEND_CPU
        )
        makima_result = score_dataset_condition_with_alignment_exp450_antigravity(
            values,
            CONDITION_D,
            BACKEND_CPU,
            align_rows_override=align_rows_exp472_antigravity,
        )
        d_result = score_dataset_condition_exp450_antigravity(
            values, CONDITION_D, BACKEND_CPU
        )
        self.assertEqual(_science_bytes(e_result), _science_bytes(makima_result))
        self.assertNotEqual(_science_bytes(makima_result), _science_bytes(d_result))

    def test_per_dataset_composition_equals_explicit_full_map_composition(self) -> None:
        """Local composition must preserve this fixture's full-map policy output."""
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
        for values, result in ((RAW_FIXTURE, first_result), (other, second_result)):
            with self.subTest(dataset=values.dataset):
                actual = scoring.compose_final_score_exp472_antigravity(
                    values, result, BACKEND_CPU
                )
                np.testing.assert_array_equal(
                    actual.scores,
                    np.asarray(expected.scores[values.dataset], dtype=np.float64),
                )
                self.assertEqual(actual.dataset, values.dataset)
                self.assertEqual(actual.alerts, expected.alerts[values.dataset])
                self.assertEqual(actual.scores.dtype, np.dtype("float64"))
                self.assertTrue(np.isfinite(actual.scores).all())
                self.assertEqual(
                    canonical_json_bytes_exp472_antigravity(actual.test_identity),
                    canonical_json_bytes_exp472_antigravity(
                        test_identity_exp472_antigravity(values)
                    ),
                )
                self.assertEqual(
                    actual.upsampled_rows,
                    count_upsampled_rows_exp472_antigravity(values),
                )
                self.assertEqual(
                    canonical_json_bytes_exp472_antigravity(actual.diagnostic),
                    canonical_json_bytes_exp472_antigravity(
                        dataset_shape_diagnostic_exp472_antigravity(values)
                    ),
                )


if __name__ == "__main__":
    unittest.main()
