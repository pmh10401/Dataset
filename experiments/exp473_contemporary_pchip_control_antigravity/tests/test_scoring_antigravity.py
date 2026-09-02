"""Exp473 same-environment PCHIP control scoring contracts."""

from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from paper.equipment_anomaly_detection.contracts_paper import RawDataset

from experiments.exp450_resampling_image_ablation_antigravity.artifacts_antigravity import (
    canonical_result_bytes_antigravity,
)
from experiments.exp450_resampling_image_ablation_antigravity.contracts_antigravity import (
    Exp450Condition,
)
from experiments.exp450_resampling_image_ablation_antigravity.scoring_antigravity import (
    score_dataset_condition_exp450_antigravity,
)
from experiments.exp450_resampling_image_ablation_antigravity.transforms_antigravity import (
    align_rows_exp450_antigravity,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.contracts_antigravity import (
    Exp472BackendDecisionAntigravity,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.scoring_antigravity import (
    score_dataset_condition_e_exp472_antigravity,
)
import experiments.exp473_contemporary_pchip_control_antigravity.scoring_antigravity as scoring


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


def _raw_fixture(dataset: str, train_length: int, test_length: int) -> RawDataset:
    train = tuple(
        np.sin(np.linspace(0.0, 2.0 * np.pi, train_length, dtype=np.float32) + index / 7.0)
        .astype("<f4")
        for index in range(5)
    )
    test = tuple(
        np.cos(np.linspace(0.0, 2.0 * np.pi, test_length, dtype=np.float32) + index / 5.0)
        .astype("<f4")
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


class Exp473ScoringTestsAntigravity(unittest.TestCase):
    def test_d_prime_routes_through_explicit_current_pchip_aligner(self) -> None:
        """D-prime must recompute PCHIP now, not read historical Exp471 shards."""
        values = _raw_fixture("exp473_upsample_fixture", 12, 4)
        with (
            patch.object(
                scoring,
                "align_rows_exp450_antigravity",
                wraps=align_rows_exp450_antigravity,
            ) as align,
            patch.object(
                scoring,
                "score_dataset_condition_with_alignment_exp450_antigravity",
                wraps=scoring.score_dataset_condition_with_alignment_exp450_antigravity,
            ) as score,
        ):
            result = scoring.score_dataset_condition_d_prime_exp473_antigravity(
                values, BACKEND_CPU
            )
        self.assertEqual(result.condition_id, "D")
        self.assertGreater(align.call_count, 0)
        self.assertIs(
            score.call_args.kwargs["align_rows_override"],
            scoring.align_rows_d_prime_exp473_antigravity,
        )
        self.assertTrue(
            all(call.kwargs["mode"] == "polyphase_pchip" for call in align.call_args_list)
        )

    def test_no_upsampling_exposure_is_scientifically_identical_to_e(self) -> None:
        """With no short rows, Makima and PCHIP cannot affect any score value."""
        values = _raw_fixture("exp473_no_exposure_fixture", 24, 24)
        d_prime = scoring.score_dataset_condition_d_prime_exp473_antigravity(
            values, BACKEND_CPU
        )
        condition_e = score_dataset_condition_e_exp472_antigravity(
            values, BACKEND_CPU
        )
        self.assertEqual(_science_bytes(d_prime), _science_bytes(condition_e))

    def test_d_prime_is_current_exp450_d_and_composes_a_durable_score(self) -> None:
        values = _raw_fixture("exp473_exposed_fixture", 12, 4)
        result = scoring.score_dataset_condition_d_prime_exp473_antigravity(
            values, BACKEND_CPU
        )
        expected = score_dataset_condition_exp450_antigravity(
            values,
            Exp450Condition("D", "polyphase_pchip", "area"),
            BACKEND_CPU,
        )
        self.assertEqual(_science_bytes(result), _science_bytes(expected))
        shard = scoring.compose_final_score_exp473_antigravity(values, result)
        self.assertEqual(shard.dataset, values.dataset)
        self.assertEqual(shard.scores.dtype, np.dtype("float64"))
        self.assertTrue(np.isfinite(shard.scores).all())
        self.assertEqual(
            shard.alerts,
            frozenset(np.flatnonzero(shard.scores >= 0.5).tolist()),
        )

    def test_scoring_boundary_is_label_free(self) -> None:
        source = Path(scoring.__file__).read_text(encoding="utf-8")
        self.assertNotIn("label_access", source)
        self.assertNotIn("load_dataset_labels", source)


if __name__ == "__main__":
    unittest.main()
