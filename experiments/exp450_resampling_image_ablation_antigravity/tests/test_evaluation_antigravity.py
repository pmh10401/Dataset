"""Fixture-only contracts for Exp450 continuous-score evaluation."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import unittest

import numpy as np

from experiments.exp450_resampling_image_ablation_antigravity.evaluation_antigravity import (
    METHOD_ORDER,
    curve_tables_exp450_antigravity,
    evaluate_exp450_antigravity,
    metric_tables_exp450_antigravity,
)


DB_SHA256 = "a" * 64
DATASETS = ("mixed_antigravity", "normal_antigravity")
LABELS = {
    "mixed_antigravity": (0, 1, 0),
    "normal_antigravity": (0, 0),
}


def _identity(dataset: str, count: int) -> dict[str, object]:
    return {"dataset": dataset, "split": "TEST", "count": count, "indices": list(range(count))}


def _manifest(condition: str, values: dict[str, tuple[float, ...]]) -> dict[str, object]:
    return {
        "condition": {"condition_id": condition},
        "database_sha256": DB_SHA256,
        "datasets": [
            {"dataset": dataset, "test": _identity(dataset, len(values[dataset]))}
            for dataset in DATASETS
        ],
        "score_rows": values,
    }


EXP_SCORES = {
    "A": {"mixed_antigravity": (0.1, 0.8, 0.2), "normal_antigravity": (0.5, 0.4)},
    "B": {"mixed_antigravity": (0.1, 0.9, 0.2), "normal_antigravity": (0.5, 0.4)},
    "C": {"mixed_antigravity": (0.8, 0.4, 0.2), "normal_antigravity": (0.5, 0.4)},
    "D": {"mixed_antigravity": (0.2, 0.7, 0.1), "normal_antigravity": (0.5, 0.4)},
}
MANIFESTS = {condition: _manifest(condition, scores) for condition, scores in EXP_SCORES.items()}


def _baseline_manifest(methods: tuple[str, ...]) -> dict[str, object]:
    return {
        "database_sha256": DB_SHA256,
        "datasets": [
            {"dataset": dataset, "test": _identity(dataset, len(LABELS[dataset]))}
            for dataset in DATASETS
        ],
        "methods": {
            method: {
                dataset: {
                    "status": "complete",
                    "test_indices": list(range(len(LABELS[dataset]))),
                    "train_oof_scores": [0.05, 0.15, 0.25, 0.35],
                    "test_scores": list(EXP_SCORES["A"][dataset]),
                }
                for dataset in DATASETS
            }
            for method in methods
        },
    }


BASELINES = {
    "classical": _baseline_manifest(("knn_distance", "lof_novelty", "ocsvm_rbf", "isolation_forest")),
    "deep_svdd": _baseline_manifest(("deep_svdd_1d",)),
    "drocc": _baseline_manifest(("drocc_1d",)),
}
ALL_SCORES = {**EXP_SCORES, **{method: EXP_SCORES["A"] for method in METHOD_ORDER[4:]}}


class Exp450EvaluationTests(unittest.TestCase):
    def test_labels_are_loaded_only_after_all_four_verifiers_return(self) -> None:
        """Loading labels before D verifies could leak unverified results into evaluation."""
        events: list[str] = []

        def verify(condition: str) -> dict[str, object]:
            events.append(f"verify-{condition}")
            return MANIFESTS[condition]

        def labels(datasets: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
            events.append("labels")
            self.assertEqual(datasets, DATASETS)
            return LABELS

        evaluate_exp450_antigravity(
            Path("/fixture/root_antigravity"), baseline_roots=BASELINES,
            verifier=verify, label_loader=labels,
        )
        self.assertEqual(events, ["verify-A", "verify-B", "verify-C", "verify-D", "labels"])

    def test_missing_baseline_blocks_the_label_loader(self) -> None:
        """A missing frozen public root must fail before TEST labels become accessible."""
        labels_called = False

        def labels(_: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
            nonlocal labels_called
            labels_called = True
            return LABELS

        incomplete = dict(BASELINES)
        incomplete.pop("drocc")
        with self.assertRaisesRegex(ValueError, "baseline roots"):
            evaluate_exp450_antigravity(
                Path("/fixture/root_antigravity"), baseline_roots=incomplete,
                verifier=lambda condition: MANIFESTS[condition], label_loader=labels,
            )
        self.assertFalse(labels_called)

    def test_identity_mismatch_blocks_labels_after_the_full_verification_pass(self) -> None:
        """A mismatched D TEST population must not reach the lazy label capability."""
        labels_called = False
        mismatched = deepcopy(MANIFESTS)
        mismatched["D"]["datasets"][1]["test"]["count"] = 99

        def labels(_: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
            nonlocal labels_called
            labels_called = True
            return LABELS

        with self.assertRaisesRegex(ValueError, "TEST identity"):
            evaluate_exp450_antigravity(
                Path("/fixture/root_antigravity"), baseline_roots=BASELINES,
                verifier=lambda condition: mismatched[condition], label_loader=labels,
            )
        self.assertFalse(labels_called)

    def test_curve_axes_are_not_swapped(self) -> None:
        """Swapping ROC or PR coordinate columns makes the paper curves false."""
        curves = curve_tables_exp450_antigravity(LABELS, ALL_SCORES)
        self.assertEqual(curves["roc"]["x_name"], "false_positive_rate")
        self.assertEqual(curves["roc"]["y_name"], "true_positive_rate")
        self.assertEqual(curves["precision_recall"]["x_name"], "recall")
        self.assertEqual(curves["precision_recall"]["y_name"], "precision")
        first_roc = next(row for row in curves["roc"]["rows"] if row["method"] == "A")
        first_pr = next(row for row in curves["precision_recall"]["rows"] if row["method"] == "A")
        self.assertEqual((first_roc["false_positive_rate"], first_roc["true_positive_rate"]), (0.0, 0.0))
        self.assertEqual((first_pr["recall"], first_pr["precision"]), (1.0, 1.0 / 5.0))

    def test_all_four_variants_and_six_baselines_are_present(self) -> None:
        """Omitting a method changes the preregistered comparison population."""
        result = metric_tables_exp450_antigravity(LABELS, ALL_SCORES)
        self.assertEqual(result.method_order, METHOD_ORDER)
        self.assertEqual({row["method"] for row in result.per_dataset}, set(METHOD_ORDER))

    def test_single_class_denominators_are_explicit(self) -> None:
        """Silently treating normal-only data as valid AP/AUROC biases macro results."""
        result = metric_tables_exp450_antigravity(LABELS, ALL_SCORES)
        normal = next(
            row for row in result.per_dataset
            if row["method"] == "A" and row["dataset"] == "normal_antigravity"
        )
        summary = result.macro["A"]
        self.assertIsNone(normal["average_precision"])
        self.assertEqual(normal["average_precision_exclusion"], "no_positive_test_labels")
        self.assertIsNone(normal["auroc"])
        self.assertEqual(normal["auroc_exclusion"], "single_class_test_labels")
        self.assertEqual(summary["average_precision_denominator"], 1)
        self.assertEqual(summary["auroc_denominator"], 1)

    def test_nonfinite_scores_are_rejected(self) -> None:
        """NaN rankings would invalidate every AP/AUROC comparison."""
        invalid = deepcopy(ALL_SCORES)
        invalid["A"] = dict(invalid["A"])
        invalid["A"]["mixed_antigravity"] = (0.1, float("nan"), 0.2)
        with self.assertRaisesRegex(ValueError, "finite"):
            metric_tables_exp450_antigravity(LABELS, invalid)

    def test_paired_bootstrap_uses_shared_seed_and_fixed_order(self) -> None:
        """Changing seed or pairing order changes the preregistered confidence intervals."""
        labels = {"d1_antigravity": (0, 1), "d2_antigravity": (0, 1)}
        scores = {
            method: {
                "d1_antigravity": (0.1, 0.9),
                "d2_antigravity": (0.2, 0.8),
            }
            for method in METHOD_ORDER
        }
        scores["B"] = {"d1_antigravity": (0.1, 0.95), "d2_antigravity": (0.7, 0.3)}
        result = metric_tables_exp450_antigravity(labels, scores)
        statistic = next(
            row for row in result.paired_statistics
            if row["comparison"] == "B-A" and row["metric"] == "average_precision"
        )
        draws = np.random.default_rng(45020260820).integers(0, 2, size=(20_000, 2), dtype=np.int64)
        self.assertEqual(statistic["seed"], 45020260820)
        self.assertEqual(statistic["draw_count"], 20_000)
        self.assertEqual(statistic["dataset_order"], ("d1_antigravity", "d2_antigravity"))
        self.assertEqual(statistic["draw_sha256"], hashlib.sha256(draws.tobytes(order="C")).hexdigest())
        self.assertEqual((statistic["wins"], statistic["ties"], statistic["losses"]), (0, 1, 1))


if __name__ == "__main__":
    unittest.main()
