"""CPU-reference parity checks for the Exp421 frozen-model batch adapter."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from experiments.exp417_causal_multihorizon_shadow.model import (
    CausalMultiHorizonCNN,
    TrainedSeedModel,
    _canonical_state_sha256,
    _fit_seed_model_from_state,
    _serialize_state,
    predict_residuals,
)
from experiments.exp421_mps_batched_acceleration.batch_inference import FrozenModelAdapter


CPU = torch.device("cpu")
HORIZONS = (1, 3, 7)
LENGTH = 40
IDENTITIES = tuple(range(10))
EARLY_FIT = IDENTITIES[:5]
EARLY_VALIDATION = IDENTITIES[5:]


def _state() -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    torch.manual_seed(41701)
    return dict(_serialize_state(CausalMultiHorizonCNN(HORIZONS).state_dict()))


def _trained_model() -> TrainedSeedModel:
    state = _state()
    return TrainedSeedModel(
        seed=41701,
        horizons=HORIZONS,
        model_sha256=_canonical_state_sha256(state),
        state=state,
        stop_epoch=1,
        training_losses=(0.0,),
        validation_losses=(0.0,),
        fit_indices=IDENTITIES,
        validation_indices=(),
        early_stop_fit_indices=EARLY_FIT,
        early_stop_validation_indices=EARLY_VALIDATION,
        refit_indices=IDENTITIES,
    )


def _channels() -> np.ndarray:
    positions = np.arange(LENGTH, dtype=np.float32)
    return np.ascontiguousarray(
        np.stack((np.sin(positions / 4.0), np.cos(positions / 6.0))).astype(np.float32)
    )


def _fit_channels() -> dict[int, np.ndarray]:
    return {
        identity: np.ascontiguousarray(
            _channels() + np.float32(identity) * np.float32(0.01), dtype=np.float32
        )
        for identity in IDENTITIES
    }


class FrozenModelAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = _trained_model()
        cls.channels = _channels()
        cls.adapter = FrozenModelAdapter.from_trained_seed(cls.model, device=CPU)

    def test_predict_batch_matches_single_series_reference_for_every_batch_size(self) -> None:
        """Catches a wrong batch-axis alignment or residual target subtraction."""
        for batch_size in (1, 4, 8, 16, 32):
            with self.subTest(batch_size=batch_size):
                batch = np.repeat(self.channels[None, ...], batch_size, axis=0)
                actual = self.adapter.predict_batch(batch)
                expected = {
                    horizon: np.stack(
                        [predict_residuals(self.model, row)[horizon] for row in batch]
                    )
                    for horizon in self.model.horizons
                }
                self.assertEqual(tuple(actual), self.model.horizons)
                for horizon in self.model.horizons:
                    self.assertEqual(actual[horizon].shape, (batch_size, LENGTH - horizon))
                    self.assertTrue(np.isfinite(actual[horizon]).all())
                    self.assertFalse(actual[horizon].flags.writeable)
                    np.testing.assert_allclose(actual[horizon], expected[horizon], rtol=0, atol=1e-5)

    def test_predict_batch_rejects_invalid_shapes_lengths_and_nonfinite_values(self) -> None:
        """Catches validation that permits inputs the frozen reference rejects."""
        with self.assertRaisesRegex(ValueError, "shape"):
            self.adapter.predict_batch(self.channels)
        with self.assertRaisesRegex(ValueError, "two input channels"):
            self.adapter.predict_batch(np.zeros((1, 1, LENGTH), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "aligned horizon"):
            self.adapter.predict_batch(np.zeros((1, 2, max(HORIZONS)), dtype=np.float32))
        nonfinite = np.repeat(self.channels[None, ...], 1, axis=0)
        nonfinite[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            self.adapter.predict_batch(nonfinite)

    def test_fit_batch_matches_frozen_cpu_fit_from_the_same_initial_state(self) -> None:
        """Catches changed ordering, optimizer constants, or state serialization in fitting."""
        channels = _fit_channels()
        expected = _fit_seed_model_from_state(
            channels,
            HORIZONS,
            seed=self.model.seed,
            initial_state=self.model.state,
            device=CPU,
            fit_indices=IDENTITIES,
            validation_indices=(),
            early_stop_fit_indices=EARLY_FIT,
            early_stop_validation_indices=EARLY_VALIDATION,
        )
        actual = self.adapter.fit_batch(
            channels,
            fit_indices=IDENTITIES,
            validation_indices=(),
            early_stop_fit_indices=EARLY_FIT,
            early_stop_validation_indices=EARLY_VALIDATION,
        )
        self.assertEqual(actual.stop_epoch, expected.stop_epoch)
        self.assertEqual(actual.model_sha256, expected.model_sha256)
        np.testing.assert_allclose(actual.training_losses, expected.training_losses, rtol=0, atol=1e-12)
        np.testing.assert_allclose(actual.validation_losses, expected.validation_losses, rtol=0, atol=1e-12)
        residuals = FrozenModelAdapter.from_trained_seed(actual, device=CPU).predict_batch(
            np.stack([channels[index] for index in IDENTITIES])
        )
        for row, index in enumerate(IDENTITIES):
            expected_residuals = predict_residuals(expected, channels[index])
            for horizon in HORIZONS:
                np.testing.assert_allclose(residuals[horizon][row], expected_residuals[horizon], rtol=0, atol=1e-5)

    def test_from_trained_seed_fails_closed_for_unavailable_mps(self) -> None:
        """Catches an implicit MPS-to-CPU device fallback."""
        if torch.backends.mps.is_available():
            self.skipTest("MPS is available; unavailable-device failure is inapplicable")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            FrozenModelAdapter.from_trained_seed(self.model, device=torch.device("mps"))


@unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable on this host")
class FrozenModelAdapterMpsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = _trained_model()
        cls.channels = _channels()

    def test_mps_batch_matches_cpu_and_rejects_model_device_fallback(self) -> None:
        """Catches MPS fallback or a backend-specific residual alignment change."""
        try:
            adapter = FrozenModelAdapter.from_trained_seed(self.model, device=torch.device("mps"))
        except ValueError as error:
            if "device fallback" in str(error):
                self.skipTest("MPS materialization falls back to CPU on this host")
            raise
        batch = np.repeat(self.channels[None, ...], 4, axis=0)
        actual = adapter.predict_batch(batch)
        expected = FrozenModelAdapter.from_trained_seed(self.model, device=CPU).predict_batch(batch)
        for horizon in HORIZONS:
            np.testing.assert_allclose(actual[horizon], expected[horizon], rtol=0, atol=1e-4)

        adapter._network.to(CPU)
        with self.assertRaisesRegex(ValueError, "device fallback"):
            adapter.predict_batch(batch)


if __name__ == "__main__":
    unittest.main()
