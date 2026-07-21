from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.exp266_fn_recovery_shadow.config import (
    ALPHA,
    EVIDENCE_GROUPS,
    MINI_VARIANTS,
)
from experiments.exp266_fn_recovery_shadow.inputs import (
    InputContractError,
    group_kernel_candidates,
    load_frozen_inputs,
    validate_indices,
    validate_population,
)


class FrozenInputTests(unittest.TestCase):
    def test_exact_variants_are_frozen(self) -> None:
        self.assertEqual(ALPHA, 0.005)
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
