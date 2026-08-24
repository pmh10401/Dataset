from __future__ import annotations

import runpy
import unittest


class Exp470PredecessorTestAntigravity(unittest.TestCase):
    def test_exports_are_exact_exp468_authority_objects_antigravity(self) -> None:
        """Break caught: Exp470 replaces a reviewed Exp468 authority object with a wrapper."""
        from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.interpolation_antigravity import (
            interpolate_to_140_antigravity as exp468_interpolate,
        )
        from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.predecessor_antigravity import (
            capture_exp465_authority_antigravity as exp468_capture,
            score_c0_fold_antigravity as exp468_score_c0,
        )
        from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.sources_antigravity import (
            load_ecg200_normal_train_antigravity as exp468_load,
        )
        from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.synthetic_antigravity import (
            primary_rows_antigravity as exp468_primary_rows,
        )
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.interpolation_antigravity import (
            interpolate_to_140_antigravity,
        )
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.predecessor_antigravity import (
            capture_exp465_authority_antigravity,
            score_c0_fold_antigravity,
        )
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.sources_antigravity import (
            load_ecg200_normal_train_antigravity,
        )
        from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.synthetic_antigravity import (
            primary_rows_antigravity,
        )

        self.assertIs(load_ecg200_normal_train_antigravity, exp468_load)
        self.assertIs(interpolate_to_140_antigravity, exp468_interpolate)
        self.assertIs(primary_rows_antigravity, exp468_primary_rows)
        self.assertIs(capture_exp465_authority_antigravity, exp468_capture)
        self.assertIs(score_c0_fold_antigravity, exp468_score_c0)

    def test_codemap_has_complete_exp468_authority_edges_antigravity(self) -> None:
        """Break caught: the versioned map omits its triplet or points authority edges at phantom nodes."""
        builder = runpy.run_path("scratch/build_codemap_exp470_antigravity.py")
        graph = builder["build_graph_antigravity"]()
        node_by_id = {node["id"]: node for node in graph["nodes"]}
        self.assertEqual(
            tuple(path for path in builder["TRIPLET_ANTIGRAVITY"]),
            tuple(path for path in graph["represented_paths"] if path in builder["TRIPLET_ANTIGRAVITY"]),
        )
        for edge in graph["edges"]:
            self.assertIn(edge["from"], node_by_id)
            self.assertIn(edge["to"], node_by_id)
        authority_edges = [edge for edge in graph["edges"] if edge["type"] == "authority"]
        self.assertEqual(
            {
                (
                    edge["from"],
                    edge["to"],
                    node_by_id[edge["to"]]["path"],
                    node_by_id[edge["to"]]["evidence"]["symbol"],
                )
                for edge in authority_edges
            },
            {
                (
                    "sources",
                    "exp468_load_ecg200_normal_train",
                    "experiments/exp468_ecg200_local_stft_spectral_redistribution_antigravity/sources_antigravity.py",
                    "load_ecg200_normal_train_antigravity",
                ),
                (
                    "interpolation",
                    "exp468_interpolate_to_140",
                    "experiments/exp468_ecg200_local_stft_spectral_redistribution_antigravity/interpolation_antigravity.py",
                    "interpolate_to_140_antigravity",
                ),
                (
                    "synthetic",
                    "exp468_primary_rows",
                    "experiments/exp468_ecg200_local_stft_spectral_redistribution_antigravity/synthetic_antigravity.py",
                    "primary_rows_antigravity",
                ),
                (
                    "predecessor",
                    "exp468_capture_exp465_authority",
                    "experiments/exp468_ecg200_local_stft_spectral_redistribution_antigravity/predecessor_antigravity.py",
                    "capture_exp465_authority_antigravity",
                ),
                (
                    "predecessor",
                    "exp468_score_c0_fold",
                    "experiments/exp468_ecg200_local_stft_spectral_redistribution_antigravity/predecessor_antigravity.py",
                    "score_c0_fold_antigravity",
                ),
            },
        )


if __name__ == "__main__":
    unittest.main()
