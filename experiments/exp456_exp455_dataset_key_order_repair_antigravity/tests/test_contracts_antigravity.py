from __future__ import annotations

from dataclasses import fields
import unittest


class ContractsAntigravityTest(unittest.TestCase):
    def test_contracts_preserve_the_exp455_data_boundary_antigravity(self) -> None:
        """Break caught: a field shape changes before the dataset-key repair is introduced."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.contracts_antigravity import (
            DatasetRoutingAuthorityAntigravity, DatasetRoutingFeaturesAntigravity, FrozenRoutingAuthorityAntigravity,
        )

        self.assertEqual(tuple(field.name for field in fields(DatasetRoutingFeaturesAntigravity)), (
            "dataset", "family", "target_length", "baseline_count", "candidate_count", "overlap_count",
            "union_count", "added_count", "removed_count", "overlap_ppm",
        ))
        self.assertEqual(tuple(field.name for field in fields(DatasetRoutingAuthorityAntigravity)), (
            "dataset", "family", "target_length", "method", "baseline_indices", "candidate_indices",
            "baseline_confusion", "candidate_confusion",
        ))
        self.assertEqual(tuple(field.name for field in fields(FrozenRoutingAuthorityAntigravity)), (
            "rows", "exp454_leaf_sha256", "label_query_count",
        ))
