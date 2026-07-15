from __future__ import annotations

import unittest
from unittest.mock import patch
import numpy as np

from experiments.exp169_170.pool_core import (
    build_descriptor,
    build_pool_manifest,
    pooled_p_values,
    fixed_candidate_rows,
    SOURCES,
    ALPHAS,
    SOURCE_CONFIGS,
    POLICIES,
)


class TestExp169170PoolCore(unittest.TestCase):
    def synthetic_packs(self) -> dict[str, dict]:
        # Create a few synthetic datasets
        packs = {}
        names = ["A", "B", "C", "D", "E"]
        for idx, name in enumerate(names):
            packs[name] = {
                "n_train": 10,
                "target_length": 100 + idx * 50,
                "oob_scores": {
                    source: np.arange(10, dtype=float) + (idx * 2)
                    for source in SOURCES
                }
            }
        return packs

    def test_every_pool_excludes_target_and_meets_reference_contract(self):
        packs = self.synthetic_packs()
        pools = build_pool_manifest(dataset="A", packs=packs, minimum_reference=10, donor_cap=5)
        for policy, pool in pools.items():
            self.assertNotIn("A", pool.donors)
            for source in SOURCES:
                # 4 donor datasets (B, C, D, E). Each contributes min(10, 5) = 5.
                # Total references: 4 * 5 = 20.
                self.assertGreaterEqual(pool.references[source].size, 10)
                self.assertLessEqual(max(pool.per_donor_counts[source].values()), 5)

    def test_pool_minimum_uses_actual_source_score_count(self):
        packs = self.synthetic_packs()
        for pack in packs.values():
            pack["oob_scores"] = {
                source: np.asarray([1.0]) for source in SOURCES
            }

        with self.assertRaisesRegex(ValueError, "minimum reference"):
            build_pool_manifest(
                dataset="A",
                packs=packs,
                minimum_reference=8,
                donor_cap=5,
            )

    def test_similarity_distance_ties_do_not_duplicate_donors(self):
        packs = self.synthetic_packs()
        for pack in packs.values():
            pack["oob_scores"] = {
                source: np.asarray([1.0, 2.0]) for source in SOURCES
            }
        descriptors = {
            "A": np.zeros(23),
            "B": np.r_[1.0, np.zeros(22)],
            "C": np.r_[1.0 + 5e-10, np.zeros(22)],
            "D": np.r_[10.0, np.zeros(22)],
            "E": np.r_[20.0, np.zeros(22)],
        }

        with patch(
            "experiments.exp169_170.pool_core.build_descriptors",
            return_value=descriptors,
        ):
            pools = build_pool_manifest(
                dataset="A",
                packs=packs,
                minimum_reference=6,
                donor_cap=2,
            )

        donors = pools["similarity_pool"].donors
        self.assertEqual(len(donors), len(set(donors)))

    def test_pooled_p_values_use_plus_one_finite_sample_correction(self):
        reference = np.array([1.0, 2.0, 3.0])
        query = np.array([0.0, 2.5, 4.0])
        np.testing.assert_allclose(pooled_p_values(query, reference), [1.0, 0.5, 0.25])

    def test_descriptor_rejects_non_finite_oob_scores(self):
        scores = {source: np.arange(5, dtype=float) for source in SOURCES}
        scores["spectral"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            build_descriptor(5, 100, scores)

    def test_fixed_grid_has_three_pools_four_alphas_five_configs(self):
        n_test = 3
        # synthetic p-values: policy -> source -> test_p_values
        p_values = {
            policy: {
                source: np.array([0.01, 0.1, 0.5])
                for source in SOURCES
            }
            for policy in POLICIES
        }

        rows = fixed_candidate_rows(
            dataset="demo",
            n_test=n_test,
            p_values=p_values,
            config_hash="x",
            n_train=10,
        )
        self.assertEqual(len(rows), 60)

        # Verify the unique values
        policies_found = {row["pool_policy"] for row in rows}
        alphas_found = {row["alpha"] for row in rows}
        configs_found = {row["source_config"] for row in rows}

        self.assertEqual(policies_found, set(POLICIES))
        self.assertEqual(alphas_found, set(ALPHAS))
        self.assertEqual(configs_found, set(SOURCE_CONFIGS))


if __name__ == "__main__":
    unittest.main()
