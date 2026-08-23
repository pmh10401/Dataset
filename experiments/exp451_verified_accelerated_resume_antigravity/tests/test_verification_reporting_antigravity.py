"""Post-completion gates and frozen Exp451 reporting definitions."""

from __future__ import annotations

from types import MappingProxyType
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


def _verification_antigravity(*, total: int = 4468, manifests: tuple[str, ...] = ("A", "B", "C", "D")):
    from experiments.exp451_verified_accelerated_resume_antigravity.verifier_antigravity import (
        Exp451VerificationAntigravity,
    )

    condition_manifests = {
        condition: {"condition": {"condition_id": condition}}
        for condition in manifests
    }
    return Exp451VerificationAntigravity(
        total_successes=total,
        imported_by_condition={"A": 1117, "B": 1117, "C": 1000, "D": 1000},
        native_by_condition={"A": 0, "B": 0, "C": 117, "D": 117},
        condition_manifests=MappingProxyType(condition_manifests),
    )


class VerificationGateTests(unittest.TestCase):
    def test_evaluation_refuses_4467_successes_before_label_loader(self) -> None:
        """A partial coordinate set cannot reach the TEST-label boundary."""
        from experiments.exp451_verified_accelerated_resume_antigravity.evaluation_antigravity import (
            require_exp451_evaluation_ready_antigravity,
        )

        with self.assertRaisesRegex(ValueError, "4468"):
            require_exp451_evaluation_ready_antigravity(_verification_antigravity(total=4467))

    def test_evaluation_refuses_missing_condition_manifest(self) -> None:
        """All four independently frozen condition manifests are required."""
        from experiments.exp451_verified_accelerated_resume_antigravity.evaluation_antigravity import (
            require_exp451_evaluation_ready_antigravity,
        )

        with self.assertRaisesRegex(ValueError, "four"):
            require_exp451_evaluation_ready_antigravity(
                _verification_antigravity(manifests=("A", "B", "C"))
            )

    def test_provenance_keeps_imported_and_native_counts_separate(self) -> None:
        """The evidence summary never collapses the two execution origins."""
        from experiments.exp451_verified_accelerated_resume_antigravity.verifier_antigravity import (
            provenance_summary_antigravity,
        )

        summary = provenance_summary_antigravity(_verification_antigravity())
        self.assertEqual(summary["imported_by_condition"], {"A": 1117, "B": 1117, "C": 1000, "D": 1000})
        self.assertEqual(summary["native_by_condition"], {"A": 0, "B": 0, "C": 117, "D": 117})
        self.assertEqual(summary["imported_total"], 4234)
        self.assertEqual(summary["native_total"], 234)

    def test_curve_axis_contracts_keep_roc_and_pr_coordinates_correct(self) -> None:
        """Paper figure coordinates are delegated to the frozen Exp450 definitions."""
        from experiments.exp451_verified_accelerated_resume_antigravity.evaluation_antigravity import (
            curve_tables_exp451_antigravity,
        )

        methods = ("A", "B", "C", "D", "knn_distance", "lof_novelty", "ocsvm_rbf", "isolation_forest", "deep_svdd_1d", "drocc_1d")
        scores = {method: {"fixture_antigravity": (0.1, 0.9)} for method in methods}
        curves = curve_tables_exp451_antigravity(
            {"fixture_antigravity": (0, 1)}, scores
        )
        self.assertEqual(curves["roc"]["x_name"], "false_positive_rate")
        self.assertEqual(curves["roc"]["y_name"], "true_positive_rate")
        self.assertEqual(curves["precision_recall"]["x_name"], "recall")
        self.assertEqual(curves["precision_recall"]["y_name"], "precision")
        self.assertEqual(
            {row["method"] for row in curves["roc"]["rows"]}, set(methods)
        )

    def test_four_verified_condition_manifests_publish_create_only(self) -> None:
        """The label gate is backed by four durable, byte-stable manifest leaves."""
        from experiments.exp451_verified_accelerated_resume_antigravity.verifier_antigravity import (
            write_exp451_condition_manifests_antigravity,
        )

        with TemporaryDirectory(prefix="exp451_manifest_fixture_antigravity_") as temporary:
            root = Path(temporary).resolve()
            first = write_exp451_condition_manifests_antigravity(root, _verification_antigravity())
            second = write_exp451_condition_manifests_antigravity(root, _verification_antigravity())
        self.assertEqual(tuple(first), ("A", "B", "C", "D"))
        self.assertEqual(first, second)
