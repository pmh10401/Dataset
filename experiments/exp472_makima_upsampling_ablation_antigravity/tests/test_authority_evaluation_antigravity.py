"""Task 5's label-gated Exp471 reuse and Exp472 evaluation contract."""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from experiments.exp472_makima_upsampling_ablation_antigravity.authority_antigravity import (
    Exp472ReuseAuthorityAntigravity,
    bind_e_to_exp471_exp472_antigravity,
    verify_exp471_reuse_exp472_antigravity,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.evaluation_antigravity import (
    evaluate_exp472_antigravity,
    paired_e_minus_d_exp472_antigravity,
)


_DIGEST = "a" * 64
_METHODS = (
    "A", "B", "C", "D", "E", "knn_distance", "lof_novelty",
    "ocsvm_rbf", "isolation_forest", "deep_svdd_1d", "drocc_1d",
)


def _authority_fixture_antigravity() -> dict[str, object]:
    return {
        "exp471_source_authority_sha256": _DIGEST,
        "exp471_cache_manifest_sha256": "b" * 64,
        "exp471_evaluation_sha256": "c" * 64,
        "exp471_report_manifest_sha256": "d" * 64,
        "database_sha256": "e" * 64,
        "coordinate_inventory_sha256": "f" * 64,
        "method_order": _METHODS[0:4] + _METHODS[5:],
        "total": 11170, "scored": 10775, "abstained": 395, "failed": 0,
        "dataset_count": 1117, "common_support": 922,
        "coordinates": {"one": {"count": 3, "indices": [2, 5, 7], "blob_sha256": _DIGEST, "row_sha256": "b" * 64}},
    }


def _e_fixture_antigravity() -> dict[str, object]:
    return {
        "database_sha256": "e" * 64,
        "dataset_order": ["one"],
        "descriptors": [{"dataset": "one", "test_identity": {"count": 3, "indices": [2, 5, 7], "blob_sha256": _DIGEST, "row_sha256": "b" * 64}}],
        "scores": {"one": [0.1, 0.9, 0.8]},
        "upsampled_rows": {"one": 2},
    }


def _d_fixture_antigravity() -> dict[str, object]:
    scores = {method: {"one": [0.2, 0.8, 0.7]} for method in _METHODS if method != "E"}
    return {"authority": _authority_fixture_antigravity(), "scores": scores}


class Exp472AuthorityEvaluationTestsAntigravity(unittest.TestCase):
    def test_reuse_requires_exact_exp471_terminal_counts(self) -> None:
        """Changing a terminal partition must reject supposedly reusable evidence."""
        authority = verify_exp471_reuse_exp472_antigravity(_authority_fixture_antigravity())
        self.assertEqual((authority.total, authority.scored, authority.abstained, authority.failed), (11170, 10775, 395, 0))
        self.assertEqual((authority.dataset_count, authority.common_support), (1117, 922))
        wrong = _authority_fixture_antigravity(); wrong["failed"] = 1; wrong["scored"] = 10774
        with self.assertRaisesRegex(ValueError, "terminal counts"):
            verify_exp471_reuse_exp472_antigravity(wrong)

    def test_e_and_d_test_coordinates_must_match_exactly(self) -> None:
        """A reordered TEST coordinate must fail before any label loader can run."""
        authority = verify_exp471_reuse_exp472_antigravity(_authority_fixture_antigravity())
        mismatch = _e_fixture_antigravity(); mismatch["descriptors"][0]["test_identity"]["indices"] = [5, 2, 7]
        with self.assertRaisesRegex(ValueError, "TEST coordinate"):
            bind_e_to_exp471_exp472_antigravity(mismatch, authority)
        with self.assertRaisesRegex(ValueError, "TEST coordinate"):
            bind_e_to_exp471_exp472_antigravity(
                _e_fixture_antigravity(), replace(authority, coordinate_sha256="0" * 64)
            )

    def test_curve_axes_method_order_and_label_gate_are_fixed(self) -> None:
        """Opening labels before evidence binding or changing curve axes is a contract bug."""
        calls: list[tuple[str, ...]] = []
        def labels(datasets: tuple[str, ...]) -> dict[str, list[int]]:
            calls.append(datasets)
            return {"one": [0, 1, 1]}
        evaluation = evaluate_exp472_antigravity(_e_fixture_antigravity(), _d_fixture_antigravity(), labels)
        self.assertEqual(calls, [("one",)])
        self.assertEqual(evaluation.method_order, _METHODS)
        self.assertEqual(evaluation.curves["roc"]["x_name"], "false_positive_rate")
        self.assertEqual(evaluation.curves["pr"]["x_name"], "recall")
        self.assertEqual(evaluation.denominators["upsampled_rows_gt_zero"]["dataset_count"], 1)

    def test_bootstrap_is_seeded_and_uses_tie_tolerance(self) -> None:
        """Changing draw seed or treating a 1e-12 tie as a win corrupts paired evidence."""
        result = paired_e_minus_d_exp472_antigravity([1.0, 1.0 + 5e-13], [1.0, 1.0])
        self.assertEqual((result["wins"], result["ties"], result["losses"]), (0, 2, 0))
        self.assertEqual(result["draw_count"], 20_000)

    def test_nonfinite_score_is_rejected_before_curve_publication(self) -> None:
        """Letting NaN enter a curve would invalidate its monotonicity and bounds."""
        with self.assertRaisesRegex(ValueError, "TEST scores"):
            evaluate_exp472_antigravity(
                {**_e_fixture_antigravity(), "scores": {"one": np.asarray([0.1, np.nan, 0.8])}},
                _d_fixture_antigravity(), {"one": [0, 1, 1]},
            )
