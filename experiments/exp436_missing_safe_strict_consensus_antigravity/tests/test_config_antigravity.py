"""Frozen configuration tests for the Exp436 contract boundary."""

from dataclasses import replace
import os
import unittest

from experiments.exp436_missing_safe_strict_consensus_antigravity.config_antigravity import (
    FROZEN_CONFIG,
    cpu_worker_count_antigravity,
)


class Exp436ConfigTests(unittest.TestCase):
    def test_frozen_config_and_cpu_policy(self) -> None:
        self.assertEqual(FROZEN_CONFIG.experiment_id, "Exp436")
        self.assertEqual(
            FROZEN_CONFIG.run_identity,
            "exp436-missing-safe-strict-consensus-antigravity-20260806",
        )
        self.assertEqual(FROZEN_CONFIG.expected_datasets, 1117)
        self.assertEqual(cpu_worker_count_antigravity(), max(1, (os.cpu_count() or 1) - 1))
        self.assertEqual(FROZEN_CONFIG.cpu_workers, cpu_worker_count_antigravity())
        self.assertEqual(FROZEN_CONFIG.backend_batch_sizes, (16, 32, 64, 128, 256))
        self.assertEqual(FROZEN_CONFIG.backend_timed_runs, 5)
        self.assertEqual(FROZEN_CONFIG.database_sha256, "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c")
        self.assertEqual(FROZEN_CONFIG.exp291_prediction_sha256, "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848")
        self.assertEqual(FROZEN_CONFIG.exp291_manifest_sha256, "761466253cccbb7b91fe36cdbd6d298de5d141c90e01e9fe646a8ae242ca9c3e")
        self.assertEqual(FROZEN_CONFIG.exp292_prediction_sha256, "f02259ae77fdcbccb8922fb9f90d5918401020d5d7d902212c2e424b878d0d3b")
        self.assertEqual(FROZEN_CONFIG.exp292_manifest_sha256, "0672b2f84a3b2883628bac3c12ad8a25ebfb48efd929e4c0a448c5418267b0d0")
        self.assertEqual(FROZEN_CONFIG.hard_prediction_sha256, "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70")
        FROZEN_CONFIG.validate()

    def test_mutated_value_or_type_fails_closed(self) -> None:
        for changed in (
            replace(FROZEN_CONFIG, neighbor_count=2),
            replace(FROZEN_CONFIG, candidate_alpha="0.01"),
        ):
            with self.subTest(changed=changed), self.assertRaisesRegex(
                ValueError, "Exp436 frozen config mismatch"
            ):
                changed.validate()
