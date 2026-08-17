from __future__ import annotations

import unittest


class FeaturesAntigravityTest(unittest.TestCase):
    def test_features_normalize_chlorine_first_authority_to_frozen_public_order_antigravity(self) -> None:
        """Break caught: feature output follows predecessor source order instead of frozen public order."""
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.contracts_antigravity import (
            ConfusionAntigravity, DatasetRoutingAuthorityAntigravity, FrozenRoutingAuthorityAntigravity,
        )
        from experiments.exp456_exp455_dataset_key_order_repair_antigravity.features_antigravity import build_label_free_features_antigravity

        def row(dataset: str, family: str, method: str) -> DatasetRoutingAuthorityAntigravity:
            return DatasetRoutingAuthorityAntigravity(dataset, family, 10, method, frozenset({1, 2}), frozenset({2, 3}), ConfusionAntigravity(0, 0, 0, 0), ConfusionAntigravity(0, 0, 0, 0))

        ecg = tuple(row(f"ECG5000__{index}", "ECG5000", "hydra_local_gap") for index in range(5))
        chlorine = tuple(row(f"ChlorineConcentration__{index}", "ChlorineConcentration", "multirocket_local_gap") for index in range(3))
        features = build_label_free_features_antigravity(FrozenRoutingAuthorityAntigravity(chlorine + ecg, (), 0))
        self.assertEqual(tuple(item.dataset for item in features), tuple(item.dataset for item in ecg + chlorine))
        self.assertEqual((features[0].overlap_count, features[0].union_count, features[0].overlap_ppm), (1, 3, 333333))
