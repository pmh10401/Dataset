from dataclasses import replace
import unittest

from experiments.exp416_exp404_hard_test_confusion_audit.config import (
    FROZEN_CONFIG,
)


class ConfigTests(unittest.TestCase):
    def test_frozen_values_match_preregistration(self) -> None:
        self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp416")
        self.assertEqual(FROZEN_CONFIG.expected_datasets, 1117)
        self.assertEqual(FROZEN_CONFIG.expected_prediction_datasets, 888)
        self.assertEqual(FROZEN_CONFIG.expected_prediction_coordinates, 2218)
        self.assertEqual(FROZEN_CONFIG.expected_label_rows, 195900)
        self.assertEqual(
            FROZEN_CONFIG.prediction_sha256,
            "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70",
        )
        FROZEN_CONFIG.validate()

    def test_mutated_frozen_config_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen"):
            replace(FROZEN_CONFIG, expected_prediction_coordinates=2217).validate()

    def test_wrong_frozen_field_type_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected_datasets"):
            replace(FROZEN_CONFIG, expected_datasets=True).validate()
