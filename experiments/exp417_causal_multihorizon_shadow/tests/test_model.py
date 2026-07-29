"""Synthetic CPU tests for Exp417 causal multi-horizon model fitting."""

from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import torch

from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG
from experiments.exp417_causal_multihorizon_shadow.contracts import RawSeries, RawTrainDataset
from experiments.exp417_causal_multihorizon_shadow.preprocessing import (
    build_horizons,
    canonical_waveform_sha256,
    preprocess_length_groups,
)
from experiments.exp417_causal_multihorizon_shadow.model import (
    CausalMultiHorizonCNN,
    fit_final_seed_models,
    predict_residuals,
    train_oof_models,
)


CPU = torch.device("cpu")


def synthetic_group(count: int = 100, length: int = 64):
    """Create finite, exact-length normal TRAIN records with unique identities."""
    positions = np.arange(length, dtype=np.float64)
    series = tuple(
        RawSeries(
            instance_index=index,
            values=tuple(
                0.03 * index
                + np.sin(positions / (3.0 + (index % 5)))
                + 0.01 * np.cos(positions * (index + 1) / 17.0)
            ),
        )
        for index in range(count)
    )
    group = preprocess_length_groups(RawTrainDataset(dataset="synthetic", series=series))[0]
    if group.status not in {"diagnostic_only", "candidate_eligible"}:
        raise AssertionError(f"unexpected synthetic group status: {group.status}")
    return group


def model() -> CausalMultiHorizonCNN:
    torch.manual_seed(41701)
    return CausalMultiHorizonCNN(build_horizons(128)).eval()


class CausalArchitectureTests(unittest.TestCase):
    def test_frozen_three_block_architecture_and_aligned_heads(self) -> None:
        network = model()
        self.assertEqual(len(network.blocks), 3)
        self.assertEqual(network.hidden_channels, 32)
        self.assertEqual(tuple(block.dilation for block in network.blocks), (1, 2, 4))
        self.assertTrue(all(block.kernel_size == 5 for block in network.blocks))
        self.assertEqual(tuple(network.horizons), (1, 3, 13))
        self.assertEqual(len(network.heads), 3)
        output = network(torch.zeros(2, 2, 128))
        self.assertEqual(tuple(output), (1, 3, 13))
        self.assertEqual({horizon: values.shape for horizon, values in output.items()}, {
            1: (2, 1, 127), 3: (2, 1, 125), 13: (2, 1, 115)
        })

    def test_future_mutation_cannot_change_earlier_prediction(self) -> None:
        network = model()
        original = torch.zeros(1, 2, 128)
        changed = original.clone()
        changed[:, :, 96:] = 1000.0
        before = network(original)
        after = network(changed)
        for horizon in before:
            torch.testing.assert_close(before[horizon][..., :80], after[horizon][..., :80])

    def test_model_rejects_wrong_horizons_and_input_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "three distinct"):
            CausalMultiHorizonCNN((1, 1, 3))
        network = model()
        with self.assertRaisesRegex(ValueError, "two input channels"):
            network(torch.zeros(1, 1, 128))
        with self.assertRaisesRegex(ValueError, "finite"):
            network(torch.full((1, 2, 128), float("nan")))


class DeterministicTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.group = synthetic_group()

    def test_final_cpu_fits_are_byte_deterministic_and_input_detached(self) -> None:
        first = fit_final_seed_models(self.group, device=CPU)
        mutable = np.array(self.group.channels[0], copy=True)
        mutable[:] = 123.0
        second = fit_final_seed_models(self.group, device=CPU)
        self.assertEqual(tuple(item.seed for item in first), FROZEN_CONFIG.model_seeds)
        self.assertEqual(tuple(item.model_sha256 for item in first), tuple(item.model_sha256 for item in second))
        self.assertEqual(tuple(item.stop_epoch for item in first), tuple(item.stop_epoch for item in second))
        self.assertEqual(tuple(item.training_losses for item in first), tuple(item.training_losses for item in second))
        self.assertEqual(tuple(item.validation_losses for item in first), tuple(item.validation_losses for item in second))
        expected_early_validation = tuple(
            item.instance_index
            for item in self.group.group.series
            if int(canonical_waveform_sha256(item), 16) % 10 == 0
        )
        for trained in first:
            self.assertEqual(trained.validation_indices, ())
            self.assertEqual(trained.early_stop_validation_indices, expected_early_validation)
            self.assertFalse(
                {canonical_waveform_sha256(item) for item in self.group.group.series if item.instance_index in trained.early_stop_fit_indices}
                & {canonical_waveform_sha256(item) for item in self.group.group.series if item.instance_index in trained.early_stop_validation_indices}
            )
            residuals = predict_residuals(trained, self.group.channels[0])
            repeated = predict_residuals(trained, self.group.channels[0])
            self.assertEqual(tuple(residuals), build_horizons(self.group.group.length))
            for horizon in residuals:
                self.assertFalse(residuals[horizon].flags.writeable)
                np.testing.assert_array_equal(residuals[horizon], repeated[horizon])
                with self.assertRaises(ValueError):
                    residuals[horizon][0] = 0.0

    def test_oof_models_keep_validation_identities_out_of_all_fit_sets(self) -> None:
        bundle = train_oof_models(self.group, device=CPU)
        self.assertEqual(len(bundle.models), 15)
        self.assertEqual(tuple(bundle.folds), bundle.folds)
        identities = {
            item.instance_index: canonical_waveform_sha256(item)
            for item in self.group.group.series
        }
        for trained in bundle.models:
            self.assertFalse(set(trained.validation_indices) & set(trained.fit_indices))
            self.assertFalse(set(trained.validation_indices) & set(trained.early_stop_fit_indices))
            self.assertFalse(set(trained.validation_indices) & set(trained.early_stop_validation_indices))
            self.assertFalse(
                {identities[index] for index in trained.validation_indices}
                & {identities[index] for index in trained.fit_indices}
            )
        for seed in FROZEN_CONFIG.model_seeds:
            expected = {item.instance_index for item in self.group.group.series}
            self.assertEqual(set(bundle.oof_residuals[seed]), expected)
            for residual_by_horizon in bundle.oof_residuals[seed].values():
                for residual in residual_by_horizon.values():
                    self.assertFalse(residual.flags.writeable)

    def test_training_rejects_unusable_channels_and_non_cpu_fallback(self) -> None:
        bad_channels = dict(self.group.channels)
        bad_channels[0] = np.zeros((1, self.group.group.length), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "two input channels"):
            model()(torch.from_numpy(bad_channels[0][None, ...]))
        with self.assertRaisesRegex(ValueError, "support"):
            fit_final_seed_models(self.group, device=torch.device("cuda"))
        with self.assertRaisesRegex(ValueError, "frozen"):
            fit_final_seed_models(self.group, device=CPU, config=replace(FROZEN_CONFIG, candidate_alpha=0.02))


if __name__ == "__main__":
    unittest.main()
