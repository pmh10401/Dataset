from __future__ import annotations

import unittest

from experiments.exp309_316_fn_recovery.evidence_fusion import fuse_indices


class EvidenceFusionTests(unittest.TestCase):
    def test_two_independent_groups_create_shadow_but_exact_normal_suppresses_it(self):
        evidence = {
            "multiscale_patch_discord": {4, 5},
            "multiwindow_change_point": {4, 5},
            "wavelet_phase_coherence": set(),
        }
        self.assertEqual(fuse_indices(evidence, exact_normal=set(), minimum_groups=2), {4, 5})
        self.assertEqual(fuse_indices(evidence, exact_normal={4}, minimum_groups=2), {5})

    def test_existing_hard_is_not_duplicated_in_shadow(self):
        evidence = {
            "multiscale_patch_discord": {2},
            "wavelet_phase_coherence": {2},
        }
        self.assertEqual(fuse_indices(evidence, exact_normal=set(), excluded={2}, minimum_groups=2), set())


if __name__ == "__main__": unittest.main()
