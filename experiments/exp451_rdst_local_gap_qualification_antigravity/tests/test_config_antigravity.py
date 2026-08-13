"""Frozen Exp451 scientific configuration contract."""

from __future__ import annotations

import unittest

import experiments.exp451_rdst_local_gap_qualification_antigravity.config_antigravity as config_module

from experiments.exp451_rdst_local_gap_qualification_antigravity.config_antigravity import (
    FROZEN_CONFIG_ANTIGRAVITY,
    FrozenConfigAuthorityAntigravity,
    decode_frozen_config_antigravity,
)


class ConfigAntigravityTest(unittest.TestCase):
    def test_frozen_config_binds_the_preregistered_rdst_contract_antigravity(self) -> None:
        config = FROZEN_CONFIG_ANTIGRAVITY
        self.assertEqual(config["experiment"], "Exp451")
        self.assertEqual(config["schema"], "exp451")
        self.assertEqual(config["rdst_public_class_path"], "aeon.transformations.collection.shapelet_based.RandomDilatedShapeletTransform")
        self.assertEqual(config["rdst_implementation_class_path"], "aeon.transformations.collection.shapelet_based._dilated_shapelet_transform.RandomDilatedShapeletTransform")
        self.assertEqual(config["prediction_root"], "/Users/mac/Documents/Dataset/exp451_rdst_local_gap_qualification_antigravity_20260813")
        self.assertEqual(config["evaluation_root"], "/Users/mac/Documents/Dataset/exp451_rdst_local_gap_qualification_evaluation_antigravity_20260813")
        self.assertEqual(config["max_shapelets"], 1024)
        self.assertEqual(config["seeds"], (20260721, 20261721))
        self.assertEqual(config["alpha"], 0.005)
        self.assertEqual(config["source_sha256"], "900397653dd89bb4f04b33182d87e0f682555568f7f50690ef9a78ba946334f2")
        self.assertEqual(config["crossfit_folds"], 5)
        self.assertEqual(config["knn_neighbors"], 3)
        self.assertEqual(config["max_train_reference"], 1000)
        self.assertEqual(config["query_chunk_size"], 128)
        self.assertEqual((config["target_family_count"], config["target_dataset_count"], config["target_test_rows"]), (8, 48, 72300))
        self.assertEqual(config["worker_rule"], "max(1, logical_cpus-1)")
        self.assertEqual(config["rdst_n_jobs"], 1)
        with self.assertRaises(TypeError):
            config["alpha"] = 1.0  # type: ignore[index]

    def test_public_class_path_is_frozen_separately_from_observed_implementation_antigravity(self) -> None:
        raw = dict(FROZEN_CONFIG_ANTIGRAVITY)
        raw["rdst_public_class_path"] = raw["rdst_implementation_class_path"]
        with self.assertRaises(ValueError):
            decode_frozen_config_antigravity(raw)

    def test_typed_config_decode_rejects_nonexact_shape_and_bool_integers_antigravity(self) -> None:
        raw = dict(FROZEN_CONFIG_ANTIGRAVITY)
        authority = decode_frozen_config_antigravity(raw)
        self.assertIs(type(authority), FrozenConfigAuthorityAntigravity)
        self.assertEqual(authority.max_shapelets, 1024)
        for key, value in (("max_shapelets", True), ("alpha", 5), ("seeds", [20260721, 20261721])):
            malformed = dict(raw)
            malformed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                decode_frozen_config_antigravity(malformed)
        with self.assertRaises(ValueError):
            decode_frozen_config_antigravity({key: value for key, value in raw.items() if key != "alpha"})
        extra = dict(raw)
        extra["unexpected"] = 1
        with self.assertRaises(ValueError):
            decode_frozen_config_antigravity(extra)

    def test_public_config_has_no_mutable_backing_alias_antigravity(self) -> None:
        before = dict(FROZEN_CONFIG_ANTIGRAVITY)
        self.assertFalse(hasattr(config_module, "_FROZEN_CONFIG_DATA_ANTIGRAVITY"))
        self.assertFalse(hasattr(config_module, "_FROZEN_CONFIG_VALUES_ANTIGRAVITY"))
        public_copy = dict(FROZEN_CONFIG_ANTIGRAVITY)
        public_copy["max_shapelets"] = 1
        self.assertEqual(FROZEN_CONFIG_ANTIGRAVITY["max_shapelets"], 1024)
        self.assertEqual(config_module.FROZEN_CONFIG_AUTHORITY_ANTIGRAVITY.max_shapelets, 1024)
        self.assertEqual(dict(FROZEN_CONFIG_ANTIGRAVITY), before)


if __name__ == "__main__":
    unittest.main()
