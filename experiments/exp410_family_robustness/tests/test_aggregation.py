from dataclasses import replace
import unittest

from experiments.exp410_family_robustness.aggregation import (
    aggregate_family_robustness,
)
from experiments.exp410_family_robustness.config import FROZEN_CONFIG
from experiments.exp410_family_robustness.contracts import (
    DatasetMetric,
    DatasetProfile,
)


def aggregation_fixture() -> tuple[
    tuple[DatasetMetric, ...],
    tuple[DatasetProfile, ...],
]:
    metrics = (
        DatasetMetric("Large_normal_0", "Large", 1, 1, 0, 0, 1.0, 1.0, 1.0),
        DatasetMetric("Large_normal_1", "Large", 5, 5, 0, 0, 1.0, 1.0, 1.0),
        DatasetMetric("Large_normal_2", "Large", 12, 12, 0, 0, 1.0, 1.0, 1.0),
        DatasetMetric("Other_normal_0", "Other", 0, 0, 0, 11, None, 0.0, 0.0),
        DatasetMetric("Small_normal_0", "Small", 1, 0, 1, 1, 0.0, 0.0, 0.0),
    )
    profiles = (
        DatasetProfile("Large_normal_0", "Large", 10, 5, 1, 4, 1, 0.25),
        DatasetProfile("Large_normal_1", "Large", 30, 15, 5, 10, 7, 0.7),
        DatasetProfile("Large_normal_2", "Large", 60, 32, 12, 20, 19, 0.95),
        DatasetProfile("Other_normal_0", "Other", 50, 21, 11, 10, 9, 0.9),
        DatasetProfile("Small_normal_0", "Small", 25, 5, 1, 4, 1, 0.25),
    )
    return metrics, profiles


def fixture_config():
    return replace(
        FROZEN_CONFIG,
        expected_datasets=5,
        expected_families=3,
        expected_family_size_buckets=(2, 1, 0, 0),
        bootstrap_iterations=50,
    )


class AggregationTests(unittest.TestCase):
    def test_family_macro_caps_and_influence_are_exact(self) -> None:
        metrics, profiles = aggregation_fixture()
        result = aggregate_family_robustness(
            metrics,
            profiles,
            config=fixture_config(),
        )
        sensitivity = {
            row["cap"]: row["mean_f1"]
            for row in result.weight_sensitivity
        }
        family_scores = {"Large": 1.0, "Small": 0.0, "Other": 0.0}
        self.assertAlmostEqual(
            sensitivity["1"],
            sum(family_scores.values()) / 3,
        )
        self.assertAlmostEqual(sensitivity["unlimited"], 3.0 / 5.0)
        self.assertEqual(
            {row.family for row in result.family_metrics},
            set(family_scores),
        )
        self.assertEqual(len(result.family_influence), 3)
        influence = {
            row["family"]: row
            for row in result.family_influence
        }
        self.assertAlmostEqual(
            influence["Large"]["dataset_macro_f1_delta"],
            -3.0 / 5.0,
        )

    def test_bootstrap_is_family_clustered_and_seeded(self) -> None:
        metrics, profiles = aggregation_fixture()
        config = fixture_config()
        one = aggregate_family_robustness(metrics, profiles, config=config)
        two = aggregate_family_robustness(metrics, profiles, config=config)
        self.assertEqual(one.summary["bootstrap"], two.summary["bootstrap"])
        self.assertEqual(one.summary["bootstrap"]["unit"], "family")
        self.assertEqual(one.summary["bootstrap"]["seed"], 410)
        self.assertAlmostEqual(
            one.summary["bootstrap"]["point_delta_f1"],
            1.0 / 3.0 - 3.0 / 5.0,
        )
        self.assertEqual(
            one.summary["bootstrap"]["classification"],
            "severe_family_imbalance",
        )

    def test_all_preregistered_strata_and_missing_difficulty_are_published(self) -> None:
        metrics, profiles = aggregation_fixture()
        result = aggregate_family_robustness(
            metrics,
            profiles,
            config=fixture_config(),
        )
        buckets = {}
        for row in result.stratified_metrics:
            buckets.setdefault(row["dimension"], []).append(row["bucket"])
        self.assertEqual(buckets["family_size"], ["1", "2-5", "6-20", "21+"])
        self.assertEqual(buckets["normal_train_count"], ["<20", "20-49", "50+"])
        self.assertEqual(
            buckets["exact_normal_overlap"],
            ["<50%", "50-<90%", ">=90%"],
        )
        self.assertEqual(
            buckets["test_anomaly_count"],
            ["1", "2-10", "11+"],
        )
        self.assertEqual(
            result.summary["difficulty_stratification"],
            {
                "status": "not_evaluated",
                "reason": "source_artifact_unavailable",
            },
        )


if __name__ == "__main__":
    unittest.main()
