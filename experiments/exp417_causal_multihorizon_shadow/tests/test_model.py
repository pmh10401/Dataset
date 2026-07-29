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
    OOFModelBundle,
    TrainedSeedModel,
    _canonical_state_sha256,
    _serialize_state,
    fit_final_seed_models,
    predict_residuals,
    train_oof_models,
)
from experiments.exp417_causal_multihorizon_shadow.preprocessing import Fold


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


def synthetic_folds(universe: tuple[int, ...]) -> tuple[Fold, ...]:
    """Build direct, synthetic five-fold attestations without training models."""
    per_fold = len(universe) // 5
    folds = []
    for offset in range(0, len(universe), per_fold):
        validation = universe[offset : offset + per_fold]
        fit = tuple(index for index in universe if index not in set(validation))
        folds.append(
            Fold(
                fit_indices=fit,
                validation_indices=validation,
                early_stop_fit_indices=fit[5:],
                early_stop_validation_indices=fit[:5],
            )
        )
    return tuple(folds)


def synthetic_trained_model(seed: int, fold: Fold, horizons: tuple[int, int, int]) -> TrainedSeedModel:
    state = _serialize_state(CausalMultiHorizonCNN(horizons).state_dict())
    return TrainedSeedModel(
        seed=seed,
        horizons=horizons,
        model_sha256=_canonical_state_sha256(state),
        state=state,
        stop_epoch=1,
        training_losses=(0.25,),
        validation_losses=(0.5,),
        fit_indices=fold.fit_indices,
        validation_indices=fold.validation_indices,
        early_stop_fit_indices=fold.early_stop_fit_indices,
        early_stop_validation_indices=fold.early_stop_validation_indices,
        refit_indices=fold.fit_indices,
    )


def synthetic_oof_bundle() -> OOFModelBundle:
    length = 64
    horizons = build_horizons(length)
    universe = tuple(range(50))
    folds = synthetic_folds(universe)
    models = tuple(
        synthetic_trained_model(seed, fold, horizons)
        for seed in FROZEN_CONFIG.model_seeds
        for fold in folds
    )
    residuals = {
        seed: {
            index: {
                horizon: np.zeros(length - horizon, dtype=np.float64)
                for horizon in horizons
            }
            for index in universe
        }
        for seed in FROZEN_CONFIG.model_seeds
    }
    return OOFModelBundle(
        length=length,
        universe_indices=universe,
        horizons=horizons,
        folds=folds,
        models=models,
        oof_residuals=residuals,
    )


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
            self.assertTrue(set(trained.early_stop_validation_indices).issubset(trained.fit_indices))
            self.assertEqual(trained.refit_indices, trained.fit_indices)
            self.assertEqual(set(trained.refit_indices), {item.instance_index for item in self.group.group.series})
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


class ModelEvidenceBindingTests(unittest.TestCase):
    def test_trained_model_rejects_nonfinite_loss_parameter_hash_and_epoch_corruption(self) -> None:
        bundle = synthetic_oof_bundle()
        trained = bundle.models[0]
        with self.assertRaisesRegex(ValueError, "finite evidence"):
            replace(trained, training_losses=(float("nan"),))
        with self.assertRaisesRegex(ValueError, "100-epoch"):
            replace(trained, stop_epoch=101)
        with self.assertRaisesRegex(ValueError, "SHA"):
            replace(trained, model_sha256="0" * 64)
        corrupted = dict(trained.state)
        name, (dtype, shape, _payload) = next(iter(corrupted.items()))
        corrupted[name] = (dtype, shape, np.full(shape, np.nan, dtype=np.float32).tobytes())
        with self.assertRaisesRegex(ValueError, "non-finite"):
            replace(trained, state=corrupted, model_sha256=_canonical_state_sha256(corrupted))
        with self.assertRaisesRegex(ValueError, "five identity"):
            replace(trained, early_stop_validation_indices=trained.early_stop_validation_indices[:4])

    def test_oof_bundle_rejects_duplicate_overlap_missing_and_wrong_length_evidence(self) -> None:
        bundle = synthetic_oof_bundle()
        duplicate_folds = (bundle.folds[0], bundle.folds[0]) + bundle.folds[2:]
        with self.assertRaisesRegex(ValueError, "partitions"):
            replace(bundle, folds=duplicate_folds)
        with self.assertRaisesRegex(ValueError, "universe"):
            replace(bundle, universe_indices=bundle.universe_indices[:-1])
        duplicate_models = (bundle.models[0],) + bundle.models[1:-1] + (bundle.models[0],)
        with self.assertRaisesRegex(ValueError, "seed/fold"):
            replace(bundle, models=duplicate_models)
        missing = {seed: dict(by_index) for seed, by_index in bundle.oof_residuals.items()}
        del missing[FROZEN_CONFIG.model_seeds[0]][bundle.universe_indices[0]]
        with self.assertRaisesRegex(ValueError, "coverage"):
            replace(bundle, oof_residuals=missing)
        wrong_length = {seed: dict(by_index) for seed, by_index in bundle.oof_residuals.items()}
        index = bundle.universe_indices[0]
        wrong_horizons = dict(wrong_length[FROZEN_CONFIG.model_seeds[0]][index])
        wrong_horizons[bundle.horizons[0]] = np.zeros(1, dtype=np.float64)
        wrong_length[FROZEN_CONFIG.model_seeds[0]][index] = wrong_horizons
        with self.assertRaisesRegex(ValueError, "length"):
            replace(bundle, oof_residuals=wrong_length)

    def test_oof_bundle_rejects_nonfinite_residual_evidence(self) -> None:
        bundle = synthetic_oof_bundle()
        seed = FROZEN_CONFIG.model_seeds[0]
        index = bundle.universe_indices[0]
        horizon = bundle.horizons[0]
        for value in (float("nan"), float("inf")):
            corrupted = {item_seed: dict(by_index) for item_seed, by_index in bundle.oof_residuals.items()}
            horizon_residuals = dict(corrupted[seed][index])
            residual = np.array(horizon_residuals[horizon], copy=True)
            residual[0] = value
            horizon_residuals[horizon] = residual
            corrupted[seed][index] = horizon_residuals
            with self.assertRaisesRegex(ValueError, "residuals must be finite"):
                replace(bundle, oof_residuals=corrupted)


if __name__ == "__main__":
    unittest.main()
