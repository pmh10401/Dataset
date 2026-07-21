from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from experiments.exp266_fn_recovery_shadow.config import (
    ALPHA,
    EVIDENCE_GROUPS,
    FULL_RUN_WORKERS,
    MINI_VARIANTS,
)
from experiments.exp266_fn_recovery_shadow.inputs import (
    InputContractError,
    _select_mini_rows,
    validate_cache_archive_boundary,
    group_kernel_candidates,
    load_frozen_inputs,
    validate_prediction_boundary,
    validate_reconstruction_archive,
    validate_indices,
    validate_population,
)


class FrozenInputTests(unittest.TestCase):
    def test_exact_variants_are_frozen(self) -> None:
        self.assertEqual(ALPHA, 0.005)
        self.assertEqual(FULL_RUN_WORKERS, 7)
        self.assertEqual(
            MINI_VARIANTS,
            (
                "kernels_1024_seed_intersection",
                "kernels_4096_seed_intersection",
            ),
        )
        self.assertEqual(
            EVIDENCE_GROUPS,
            (
                "kernel",
                "statistical",
                "structural",
                "change_point",
                "reconstruction",
            ),
        )

    def test_partial_population_is_rejected(self) -> None:
        with self.assertRaisesRegex(InputContractError, "1,117"):
            validate_population(("demo",))

    def test_kernel_candidates_require_matching_variant_population(self) -> None:
        with self.assertRaisesRegex(InputContractError, "MiniROCKET population mismatch"):
            group_kernel_candidates(
                {"first": {"candidate_indices": [0]}},
                {"second": {"candidate_indices": [0]}},
            )

    def test_kernel_candidates_union_both_frozen_variants(self) -> None:
        self.assertEqual(
            group_kernel_candidates(
                {"demo": {"candidate_indices": [0]}},
                {"demo": {"candidate_indices": [1]}},
            ),
            {"demo": frozenset({0, 1})},
        )

    def test_indices_reject_outside_test_range(self) -> None:
        with self.assertRaisesRegex(InputContractError, "outside TEST range"):
            validate_indices("demo", 2, {2})

    def test_prediction_boundary_rejects_label_and_evaluation_fields(self) -> None:
        for field in ("label", "anomaly_positions", "auc"):
            with self.subTest(field=field), self.assertRaisesRegex(
                InputContractError, "forbidden prediction field"
            ):
                validate_prediction_boundary({"dataset": "demo", field: []}, "unit")

    def test_prediction_boundary_rejects_true_anomaly_position_presence_flags(self) -> None:
        for field in (
            "has_anomaly_positions",
            "anomalyPositionsPresent",
            "contains-test-anomaly-index",
            "has_anomalies_indexes",
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                InputContractError, "anomaly-position presence flag"
            ):
                validate_prediction_boundary({"dataset": "demo", field: True}, "unit")

    def test_mini_selection_uses_frozen_alpha_constant(self) -> None:
        rows = [
            {
                "dataset": f"dataset_{index}",
                "variant": MINI_VARIANTS[0],
                "alpha": 0.01,
            }
            for index in range(1117)
        ]
        with patch("experiments.exp266_fn_recovery_shadow.inputs.ALPHA", 0.01):
            selected = _select_mini_rows(rows, MINI_VARIANTS[0])
        self.assertEqual(len(selected), 1117)

    def test_reconstruction_archive_requires_mask_025_oof_content(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            np.savez(path, n_test=np.array([2]))

            with self.assertRaisesRegex(InputContractError, "reconstruction archive missing key"):
                validate_reconstruction_archive(path, "demo", n_train=3, n_test=2)

    def test_reconstruction_archive_rejects_forbidden_key_and_wrong_hash(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            np.savez(
                path,
                mask_025__train_oof=np.ones(3),
                mask_025__test_scores=np.ones(2),
                mask_025__train_by_seed=np.ones((1, 3)),
                mask_025__test_by_seed=np.ones((1, 2)),
                n_test=np.array([2]),
                anomaly_positions=np.array([1]),
            )

            with self.assertRaisesRegex(InputContractError, "forbidden cache field"):
                validate_reconstruction_archive(path, "demo", n_train=3, n_test=2)

            path = Path(directory) / "clean-cache.npz"
            np.savez(
                path,
                mask_025__train_oof=np.ones(3),
                mask_025__test_scores=np.ones(2),
                mask_025__train_by_seed=np.ones((1, 3)),
                mask_025__test_by_seed=np.ones((1, 2)),
                n_test=np.array([2]),
            )
            with self.assertRaisesRegex(InputContractError, "reconstruction archive SHA256 mismatch"):
                validate_reconstruction_archive(
                    path, "demo", n_train=3, n_test=2, expected_sha256="0" * 64
                )

    def test_cache_archive_boundary_rejects_test_labels(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            np.savez(path, test_labels=np.array([0, 1]))

            with self.assertRaisesRegex(InputContractError, "forbidden cache field"):
                validate_cache_archive_boundary(path, "unit", "demo")

    def test_loads_the_complete_frozen_population_and_writes_audit(self) -> None:
        with TemporaryDirectory() as directory:
            recovery = load_frozen_inputs(output_root=Path(directory))

            self.assertEqual(len(recovery.datasets), 1117)
            self.assertEqual(recovery.lane_totals, {
                "hard": 2253,
                "shadow": 2,
                "priority": 31,
                "standard": 806,
                "union": 3092,
            })
            self.assertEqual(
                set(recovery.datasets["ACSF1_normal_0"].evidence),
                {"kernel", "statistical", "structural", "change_point", "reconstruction"},
            )
            self.assertNotIn(
                "reconstruction",
                recovery.datasets["ACSF1_normal_0"].available_groups,
            )
            self.assertTrue((Path(directory) / "00_contract_audit.json").is_file())
            self.assertTrue((Path(directory) / "01_source_manifest.csv").is_file())


if __name__ == "__main__":
    unittest.main()
