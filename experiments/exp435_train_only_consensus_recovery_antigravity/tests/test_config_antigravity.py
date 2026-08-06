"""TDD tests for the frozen Exp435 configuration contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
import os
import unittest

from experiments.exp435_train_only_consensus_recovery_antigravity.config_antigravity import (
    FROZEN_CONFIG,
    Exp435FrozenConfig,
)


class Exp435ConfigTests(unittest.TestCase):
    def test_frozen_config_has_preregistered_values(self) -> None:
        self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp435")
        self.assertEqual(
            FROZEN_CONFIG.run_identity,
            "exp435-train-only-consensus-recovery-antigravity-20260806",
        )
        self.assertEqual(FROZEN_CONFIG.expected_datasets, 1117)
        self.assertEqual(FROZEN_CONFIG.neighbor_count, 3)
        self.assertEqual(FROZEN_CONFIG.candidate_alpha, 0.01)
        self.assertEqual(FROZEN_CONFIG.train_quantile, 0.99)
        self.assertEqual(FROZEN_CONFIG.global_candidate_budget, 100)
        self.assertEqual(FROZEN_CONFIG.mlx_smoke_seed, 43501)
        self.assertEqual(
            FROZEN_CONFIG.cpu_workers,
            max(1, (os.cpu_count() or 1) - 1),
        )
        self.assertEqual(FROZEN_CONFIG.bootstrap_repetitions, 20_000)
        self.assertEqual(FROZEN_CONFIG.bootstrap_seed, 43_520_260_806)
        self.assertEqual(FROZEN_CONFIG.rescued_dataset_minimum, 10)
        self.assertEqual(FROZEN_CONFIG.added_false_positive_maximum, 30)
        FROZEN_CONFIG.validate()

    def test_reference_hashes_are_frozen(self) -> None:
        self.assertEqual(
            FROZEN_CONFIG.database_sha256,
            "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c",
        )
        self.assertEqual(
            FROZEN_CONFIG.exp291_prediction_sha256,
            "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848",
        )
        self.assertEqual(
            FROZEN_CONFIG.exp291_manifest_sha256,
            "761466253cccbb7b91fe36cdbd6d298de5d141c90e01e9fe646a8ae242ca9c3e",
        )
        self.assertEqual(
            FROZEN_CONFIG.exp292_prediction_sha256,
            "f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b",
        )
        self.assertEqual(
            FROZEN_CONFIG.exp292_manifest_sha256,
            "0672b2f84a3b2883628bac3c12ad8a25ebfb48efd929e4c0a448c5418267b0d0",
        )
        self.assertEqual(
            FROZEN_CONFIG.hard_prediction_sha256,
            "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70",
        )

    def test_mutated_config_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Exp435 frozen config mismatch"):
            replace(FROZEN_CONFIG, neighbor_count=2).validate()

    def test_configuration_is_frozen_and_complete(self) -> None:
        self.assertIsInstance(FROZEN_CONFIG, Exp435FrozenConfig)
        with self.assertRaises(FrozenInstanceError):
            FROZEN_CONFIG.neighbor_count = 2  # type: ignore[misc]
        # validate() must check every field, including newly added reference fields.
        for item in fields(FROZEN_CONFIG):
            current = getattr(FROZEN_CONFIG, item.name)
            if type(current) is bool:
                replacement: object = not current
            elif type(current) is int:
                replacement = current + 1
            elif type(current) is float:
                replacement = current + 0.01
            elif type(current) is str:
                replacement = current + "-changed"
            elif type(current) is tuple:
                replacement = current + ("changed",)
            else:
                self.fail(f"unhandled config field type for {item.name}")
            with self.subTest(field=item.name), self.assertRaisesRegex(
                ValueError, "Exp435 frozen config mismatch"
            ):
                replace(FROZEN_CONFIG, **{item.name: replacement}).validate()


if __name__ == "__main__":
    unittest.main()
