#!/usr/bin/env python3
"""Regression tests for the Exp479 derived-dataset report."""

from __future__ import annotations

import sqlite3
import unittest

import build_exp479_class_count_report_antigravity as report


class Exp479DerivedDatasetReportTests(unittest.TestCase):
    def test_every_derived_dataset_has_one_normal_train_class(self) -> None:
        connection = sqlite3.connect(
            f"file:{report.DATABASE_PATH_ANTIGRAVITY}?mode=ro", uri=True
        )
        try:
            rows = connection.execute(
                "SELECT d.name, "
                "COUNT(DISTINCT CASE WHEN i.split = 'TRAIN' THEN i.label END), "
                "GROUP_CONCAT(DISTINCT CASE WHEN i.split = 'TRAIN' THEN i.label END), "
                "COUNT(DISTINCT i.label) "
                "FROM datasets AS d "
                "JOIN instances AS i ON i.dataset_id = d.id "
                "WHERE d.name LIKE '%_normal_%' "
                "GROUP BY d.id ORDER BY d.name"
            ).fetchall()
        finally:
            connection.close()

        self.assertEqual(1118, len(rows))
        self.assertEqual([], [row for row in rows if tuple(row[1:]) != (1, "0", 2)])

    def test_report_separates_normal_class_from_source_label_variants(self) -> None:
        artifact = report.build_artifact_antigravity()
        datasets = artifact["snapshot"]["datasets"]
        summary = datasets["summary_antigravity"][0]
        families = {
            row["family"]: row
            for row in datasets["family_source_label_variants_antigravity"]
        }

        self.assertEqual(1, summary["normal_class_count_per_derived_dataset"])
        self.assertEqual(0, summary["derived_dataset_label_violations"])
        self.assertEqual("train_normal_count", summary["candidate_ranking_basis"])
        self.assertEqual(69, summary["unique_source_label_candidate_families"])
        self.assertEqual(59, summary["tied_source_label_candidate_families"])
        self.assertEqual(674, summary["source_label_candidate_variants"])

        ecg5000 = families["ECG5000"]
        self.assertEqual(5, ecg5000["source_label_variant_count"])
        self.assertEqual(1, ecg5000["normal_class_count_per_variant"])
        self.assertEqual("1", ecg5000["source_label_candidates"])
        self.assertEqual(2335, ecg5000["candidate_train_normal_count"])
        self.assertEqual("단일 후보", ecg5000["candidate_status"])

        self.assertTrue(
            all(
                "class_count" not in row
                and "maximum_labels" not in row
                and "maximum_derived_count" not in row
                for row in families.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
