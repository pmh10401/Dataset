import tempfile
from pathlib import Path
import unittest

from services.time_series_anomaly_diagnosis.model_resources import load_model_resources
from services.time_series_anomaly_diagnosis.multi_evidence_review import (
    DatasetReviewEvidence,
    load_review_evidence,
    rank_review_candidates,
)

RESOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "resources"


class MultiEvidenceReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_resources = load_model_resources(RESOURCE_DIRECTORY)
        cls.evidence = cls.model_resources.review_evidence

    def test_frozen_evidence_coverage_and_counts(self) -> None:
        self.assertEqual(len(self.evidence), 1117)
        self.assertIn("Phoneme_normal_39", self.evidence)
        self.assertNotIn("Wafer_normal_1", self.evidence)

        total_p1 = sum(len(item.priority_1_indices) for item in self.evidence.values())
        total_p2 = sum(len(item.priority_2_indices) for item in self.evidence.values())
        total_p3 = sum(len(item.priority_3_indices) for item in self.evidence.values())

        self.assertEqual(total_p1, 17)
        self.assertEqual(total_p2, 57)
        self.assertEqual(total_p3, 732)

    def test_routing_and_evidence_catalog_intersection(self) -> None:
        evidence_keys = set(self.evidence.keys())
        route_keys = set(self.model_resources.route_by_dataset.keys())

        common = evidence_keys & route_keys
        self.assertEqual(len(common), 1116)
        self.assertEqual(route_keys - evidence_keys, {"Wafer_normal_1"})
        self.assertEqual(evidence_keys - route_keys, {"Phoneme_normal_39"})

    def test_sample_dataset_structure_and_no_label_leakage(self) -> None:
        sample = self.evidence["ChlorineConcentration_normal_3"]
        self.assertEqual(sample.priority_1_indices, (2326,))
        self.assertEqual(
            sample.support_by_index[2326],
            ("official_multirocket", "structural"),
        )
        self.assertFalse(hasattr(sample, "labels"))
        self.assertFalse(hasattr(sample, "contains_test_labels"))
        self.assertFalse(hasattr(sample, "contains_evaluation_metrics"))
        self.assertFalse(hasattr(sample, "auc"))

    def test_packaged_review_evidence_has_no_automatic_or_label_data(self) -> None:
        trace = self.evidence["Trace_normal_2"]

        self.assertEqual(trace.automatic_hard_indices, ())
        self.assertFalse(hasattr(trace, "labels"))
        self.assertFalse(hasattr(trace, "contains_test_labels"))

    def test_manifest_validation_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            evidence_file = temp_path / "evidence.jsonl"
            manifest_file = temp_path / "manifest.json"

            evidence_file.write_text("invalid content\n", encoding="utf-8")
            manifest_file.write_text('{"prediction_sha256": "fake"}', encoding="utf-8")

            with self.assertRaises(ValueError):
                load_review_evidence(evidence_file, manifest_file)

    def test_synthetic_deduplication_and_priority_sorting(self) -> None:
        synthetic_evidence = {
            "test_ds": DatasetReviewEvidence(
                dataset_name="test_ds",
                test_count=10,
                priority_1_indices=(10, 20),
                priority_2_indices=(30, 40),
                priority_3_indices=(50,),
                support_by_index={
                    10: ("family_a", "family_b"),
                    20: ("family_c",),
                    30: ("family_d",),
                    40: ("family_e", "family_f", "family_g"),
                    50: ("family_h",),
                },
            )
        }

        # Index 10 is automatically detected -> should be excluded from review candidates
        detection_rows = (
            {"series_index": 10, "score": 10.0, "p_value": 0.0001, "automatic_anomaly": True},
            {"series_index": 20, "score": 8.0, "p_value": 0.01, "automatic_anomaly": False},
            {"series_index": 30, "score": 7.0, "p_value": 0.02, "automatic_anomaly": False},
            {"series_index": 40, "score": 6.0, "p_value": 0.03, "automatic_anomaly": False},
            {"series_index": 50, "score": 5.0, "p_value": 0.04, "automatic_anomaly": False},
        )

        ranked = rank_review_candidates("test_ds", detection_rows, synthetic_evidence)
        self.assertTrue(ranked["evidence_available"])
        self.assertFalse(ranked["automatic_promotion"])
        self.assertEqual(ranked["unavailable_candidate_indices"], [])

        candidate_indices = [row["series_index"] for row in ranked["candidates"]]
        # 10 is excluded. Remaining: 20 (P1), 40 (P2, support 3), 30 (P2, support 1), 50 (P3)
        self.assertEqual(candidate_indices, [20, 40, 30, 50])

        priorities = [row["priority"] for row in ranked["candidates"]]
        self.assertEqual(priorities, ["P1", "P2", "P2", "P3"])

    def test_within_tier_multikey_sorting(self) -> None:
        synthetic_evidence = {
            "sort_ds": DatasetReviewEvidence(
                dataset_name="sort_ds",
                test_count=20,
                priority_1_indices=(1, 2, 3, 4),
                priority_2_indices=(),
                priority_3_indices=(),
                support_by_index={
                    1: ("f1", "f2"),
                    2: ("f1",),
                    3: ("f1", "f2"),
                    4: ("f1", "f2"),
                },
            )
        }

        # 1: support 2, score 5.0, p 0.05, index 1
        # 2: support 1, score 9.0, p 0.01, index 2
        # 3: support 2, score 5.0, p 0.02, index 3 (better p-value than 1)
        # 4: support 2, score 5.0, p 0.05, index 4 (same support, score, p as 1, tie-breaker: index 1 < 4)
        detection_rows = (
            {"series_index": 1, "score": 5.0, "p_value": 0.05, "automatic_anomaly": False},
            {"series_index": 2, "score": 9.0, "p_value": 0.01, "automatic_anomaly": False},
            {"series_index": 3, "score": 5.0, "p_value": 0.02, "automatic_anomaly": False},
            {"series_index": 4, "score": 5.0, "p_value": 0.05, "automatic_anomaly": False},
        )

        ranked = rank_review_candidates("sort_ds", detection_rows, synthetic_evidence)
        # Ordering:
        # Tier P1 for all.
        # support_count 2 (indices 3, 1, 4) before support_count 1 (index 2).
        # Among 3, 1, 4 (all score 5.0):
        # index 3 has p_value 0.02 < 0.05 -> index 3 first
        # between 1 and 4 (both p_value 0.05): index 1 < index 4
        # Then index 2 (support_count 1)
        ordered_indices = [row["series_index"] for row in ranked["candidates"]]
        self.assertEqual(ordered_indices, [3, 1, 4, 2])

        # Verify candidate row dictionary schema
        first_row = ranked["candidates"][0]
        self.assertEqual(
            set(first_row.keys()),
            {"series_index", "priority", "support_count", "source_families", "score", "p_value"},
        )

    def test_missing_evidence_dataset(self) -> None:
        ranked = rank_review_candidates("Wafer_normal_1", (), self.evidence)
        self.assertFalse(ranked["evidence_available"])
        self.assertFalse(ranked["automatic_promotion"])
        self.assertEqual(ranked["candidates"], [])
        self.assertEqual(ranked["unavailable_candidate_indices"], [])

    def test_unavailable_candidate_indices_reported_and_sorted(self) -> None:
        synthetic_evidence = {
            "missing_score_ds": DatasetReviewEvidence(
                dataset_name="missing_score_ds",
                test_count=5,
                priority_1_indices=(100, 50),
                priority_2_indices=(200,),
                priority_3_indices=(),
                support_by_index={
                    100: ("f1",),
                    50: ("f2",),
                    200: ("f3",),
                },
            )
        }
        # Provide score only for index 50
        detection_rows = (
            {"series_index": 50, "score": 8.0, "p_value": 0.01, "automatic_anomaly": False},
        )
        ranked = rank_review_candidates("missing_score_ds", detection_rows, synthetic_evidence)
        self.assertTrue(ranked["evidence_available"])
        self.assertEqual([row["series_index"] for row in ranked["candidates"]], [50])
        self.assertEqual(ranked["unavailable_candidate_indices"], [100, 200])


if __name__ == "__main__":
    unittest.main()
