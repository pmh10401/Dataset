from __future__ import annotations

import unittest

from experiments.exp309_316_fn_recovery.evidence_fusion import (
    FULL_SOURCE_GROUPS, independent_group_count,
)


class FullFusionTests(unittest.TestCase):
    def test_correlated_representation_sources_count_once(self):
        support = {"contrastive_normal_memory", "deep_svdd_normal_boundary", "wavelet_phase_coherence"}
        self.assertEqual(independent_group_count(support, FULL_SOURCE_GROUPS), 2)

    def test_all_distinct_sources_count_separately(self):
        support = {"multiscale_patch_discord", "multiwindow_change_point", "predictive_residual"}
        self.assertEqual(independent_group_count(support, FULL_SOURCE_GROUPS), 3)


if __name__ == "__main__": unittest.main()
