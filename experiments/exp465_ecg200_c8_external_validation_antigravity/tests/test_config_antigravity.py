from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType
import unittest


class Exp465ConfigTestAntigravity(unittest.TestCase):
    def test_frozen_config_exact_values(self) -> None:
        from experiments.exp465_ecg200_c8_external_validation_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        config = FROZEN_CONFIG_ANTIGRAVITY
        self.assertEqual((config.train_rows, config.width, config.normal_rows), (100, 96, 69))
        self.assertEqual(config.method_order, ("C0", "C6", "C8"))
        self.assertEqual((config.primary_candidate, config.onset, config.cosine_min), ("C8", 56, 0.95))

    def test_frozen_contract_preserves_all_exp465_truth(self) -> None:
        """Break caught: external-validation identity, data, methods, or gates drift."""
        from experiments.exp465_ecg200_c8_external_validation_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        config = FROZEN_CONFIG_ANTIGRAVITY
        self.assertEqual(config.experiment, "Exp465")
        self.assertEqual(config.kind, "ecg200_c8_external_train_qualification")
        self.assertEqual(
            config.run_identity,
            "exp465-ecg200-c8-external-validation-antigravity-20260822",
        )
        self.assertEqual(config.dataset, "ECG200")
        self.assertEqual(config.data_split, "TRAIN_only")
        self.assertEqual(config.train_shape, (100, 96))
        self.assertEqual((config.normal_rows, config.infarct_rows), (69, 31))
        self.assertEqual(config.train_md5, "808875aeac1c358a90c630fe369327c4")
        self.assertEqual(config.interpolated_width, 140)
        self.assertEqual(config.seeds, (20260721, 20261721))
        self.assertEqual(config.folds, 5)
        self.assertEqual(config.stability_record_count, 10)
        self.assertEqual((config.onset, config.suffix_length), (56, 84))
        self.assertEqual(config.cosine_min, 0.95)
        self.assertEqual(config.method_order, ("C0", "C6", "C8"))
        self.assertIsInstance(config.method_definitions, MappingProxyType)
        self.assertEqual(
            dict(config.method_definitions),
            {
                "C0": "Exp464 C0 exact replay control",
                "C6": "Exp464 C6 exact diagnostic contrast only",
                "C8": "Exp464 C8 exact sole primary candidate",
            },
        )
        self.assertEqual(config.per_type_auc_min, 0.85)
        self.assertEqual(config.mean_auc_min, 0.90)
        self.assertEqual(config.required_stability_records, 10)
        self.assertEqual(config.normal_trials_per_seed, 69)
        self.assertEqual(config.normal_confirmation_count_must_remain, 0)
        self.assertEqual(config.label_query_count_must_remain, 0)
        self.assertEqual(config.official_evaluate_count_must_remain, 0)
        self.assertEqual(config.official_inspect_count_must_remain, 0)
        self.assertEqual(
            config.approved_design_sha256,
            "a5f5f43e1787ad2942c883761ee95478e8cf5fcdcc958722a22fe0e1ad98d904",
        )
        self.assertEqual(config.predecessor_experiment, "Exp464")
        self.assertEqual(
            config.predecessor_terminal_closeout_commit,
            "1fdc8422996fe7e85325987f7543e00db79e22d9",
        )
        with self.assertRaises(FrozenInstanceError):
            config.primary_candidate = "C6"  # type: ignore[misc]
        with self.assertRaises(TypeError):
            config.method_definitions["C6"] = "changed"  # type: ignore[index]

    def test_config_rejects_aliases_mutation_reordered_mapping_and_gate_drift(self) -> None:
        """Break caught: strict config accepts bool aliases, mutable containers, or altered gates."""
        from experiments.exp465_ecg200_c8_external_validation_antigravity.config_antigravity import (
            FROZEN_CONFIG_ANTIGRAVITY,
        )

        config = FROZEN_CONFIG_ANTIGRAVITY
        mutations = (
            ("train_rows", True),
            ("normal_rows", True),
            ("folds", True),
            ("onset", False),
            ("seeds", list(config.seeds)),
            ("method_order", list(config.method_order)),
            (
                "method_definitions",
                {
                    "C6": dict(config.method_definitions)["C6"],
                    "C0": dict(config.method_definitions)["C0"],
                    "C8": dict(config.method_definitions)["C8"],
                },
            ),
            ("primary_candidate", "C6"),
            ("per_type_auc_min", 0.86),
            ("mean_auc_min", 0.91),
            ("required_stability_records", 9),
            ("normal_confirmation_count_must_remain", 1),
            ("label_query_count_must_remain", 1),
            ("official_evaluate_count_must_remain", 1),
        )
        for field_name, value in mutations:
            with self.subTest(field_name=field_name), self.assertRaises(ValueError):
                replace(config, **{field_name: value})


if __name__ == "__main__":
    unittest.main()
