"""Fail-closed batched inference for immutable Exp417 seed models.

The adapter deliberately uses the Exp417 network and its canonical CPU state
serialization.  It only changes the inference batch axis; horizon alignment,
targets, fitting operations, and model evidence remain the frozen reference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import torch

from experiments.exp417_causal_multihorizon_shadow.model import (
    CausalMultiHorizonCNN,
    TrainedSeedModel,
    _canonical_state_sha256,
    _fit_seed_model_from_state,
    _freeze_state,
    _materialize_network_from_state,
    prepare_seed_initial_states,
)
from experiments.exp417_causal_multihorizon_shadow.preprocessing import build_horizons


_INPUT_CHANNELS = 2
_MAX_INPUT_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class FrozenModelAdapter:
    """One selected-device materialization of immutable Exp417 model evidence."""

    model: TrainedSeedModel
    device: torch.device
    _network: CausalMultiHorizonCNN

    @classmethod
    def from_trained_seed(
        cls, model: TrainedSeedModel, *, device: torch.device | None = None
    ) -> "FrozenModelAdapter":
        """Materialize one canonical CPU state on the explicitly selected device."""
        if not isinstance(model, TrainedSeedModel):
            raise ValueError("model must be a TrainedSeedModel")
        _validate_frozen_horizons(model.horizons)
        selected = torch.device(model.device) if device is None else device
        _validate_device(selected)
        network = _materialize_network_from_state(model.state, model.horizons, selected)
        _assert_model_device(network, selected)
        return cls(model=model, device=selected, _network=network)

    def predict_batch(self, channels: np.ndarray) -> Mapping[int, np.ndarray]:
        """Return read-only float64 aligned residuals for a finite two-channel batch."""
        values = _validated_batch(channels, self.model.horizons)
        # Never expose a caller-owned array to PyTorch, including immutable views.
        input_copy = np.array(values, dtype=np.float32, order="C", copy=True)
        inputs = torch.from_numpy(input_copy).to(self.device)
        _assert_tensor_device(inputs, self.device)
        _assert_model_device(self._network, self.device)
        self._network.eval()
        with torch.no_grad():
            predictions = self._network(inputs)
            if tuple(predictions) != self.model.horizons:
                raise ValueError("network predictions do not cover frozen horizons")
            for tensor in predictions.values():
                _assert_tensor_device(tensor, self.device)
                if not torch.isfinite(tensor).all():
                    raise ValueError("network predictions must be finite")
            # Concatenating then copying transfers all horizon predictions in one
            # device-to-CPU operation while preserving each head's float32 values.
            packed = torch.cat(
                [predictions[horizon].reshape(-1) for horizon in self.model.horizons]
            )
            _assert_tensor_device(packed, self.device)
            if self.device.type == "mps":
                torch.mps.synchronize()
            packed_cpu = packed.detach().cpu().numpy()

        residuals: dict[int, np.ndarray] = {}
        offset = 0
        for horizon in self.model.horizons:
            expected_length = values.shape[2] - horizon
            count = values.shape[0] * expected_length
            prediction = packed_cpu[offset : offset + count].reshape(values.shape[0], expected_length)
            offset += count
            residual = np.abs(prediction - values[:, 0, horizon:]).astype(np.float64, copy=False)
            if not np.isfinite(residual).all():
                raise ValueError("model residuals must be finite")
            residuals[horizon] = _readonly_float64(residual)
        if offset != packed_cpu.size:
            raise ValueError("packed predictions have an unexpected length")
        return MappingProxyType(residuals)

    def fit_batch(
        self,
        channels: Mapping[int, np.ndarray],
        *,
        fit_indices: Sequence[int],
        validation_indices: Sequence[int],
        early_stop_fit_indices: Sequence[int],
        early_stop_validation_indices: Sequence[int],
        seed: int,
        initial_state: Mapping[str, tuple[str, tuple[int, ...], bytes]],
    ) -> TrainedSeedModel:
        """Run the exact frozen fit pipeline from canonical initial CPU state.

        The Exp417 private reference owns the one-worker DataLoader order, Huber
        loss, AdamW settings, finite checks, early stopping, and serialization.
        Calling it directly is intentional: CPU is byte-for-byte the reference,
        while MPS executes the same frozen tensor operations on its selected device.
        """
        _validate_channel_mapping(channels, self.model.horizons)
        frozen_initial_state = _freeze_state(initial_state)
        expected_initial_state = prepare_seed_initial_states(
            self.model.horizons, self.device
        ).get(seed)
        if (
            expected_initial_state is None
            or _canonical_state_sha256(frozen_initial_state)
            != _canonical_state_sha256(expected_initial_state)
        ):
            raise ValueError("initial_state must match the canonical frozen seed initialization")
        trained = _fit_seed_model_from_state(
            channels,
            self.model.horizons,
            seed=seed,
            initial_state=frozen_initial_state,
            device=self.device,
            fit_indices=tuple(fit_indices),
            validation_indices=tuple(validation_indices),
            early_stop_fit_indices=tuple(early_stop_fit_indices),
            early_stop_validation_indices=tuple(early_stop_validation_indices),
        )
        if not isinstance(trained, TrainedSeedModel) or trained.device != self.device.type:
            raise ValueError("frozen fitting returned invalid device evidence")
        return trained


def _validated_batch(channels: np.ndarray, horizons: tuple[int, int, int]) -> np.ndarray:
    if not isinstance(channels, np.ndarray) or channels.dtype != np.dtype("float32"):
        raise ValueError("batch channels must be float32 arrays")
    if channels.ndim != 3:
        raise ValueError("batch channels must have shape (batch, two input channels, length)")
    if channels.shape[0] < 1:
        raise ValueError("batch channels must contain at least one series")
    if channels.shape[1] != _INPUT_CHANNELS:
        raise ValueError("batch channels must have exactly two input channels")
    if channels.shape[2] <= max(horizons):
        raise ValueError("batch channels leave no aligned horizon targets")
    if build_horizons(int(channels.shape[2])) != horizons:
        raise ValueError("batch horizons must match the frozen horizon definition for its length")
    if channels.nbytes > _MAX_INPUT_BYTES:
        raise ValueError("batch channels exceed the bounded allocation limit")
    if not np.isfinite(channels).all():
        raise ValueError("batch channels must be finite")
    return channels


def _validate_channel_mapping(
    channels: Mapping[int, np.ndarray], horizons: tuple[int, int, int]
) -> None:
    if not isinstance(channels, Mapping) or not channels:
        raise ValueError("channels must be a non-empty mapping")
    expected_length: int | None = None
    for index, values in channels.items():
        if type(index) is not int or index < 0:
            raise ValueError("channel indices must be non-negative integers")
        batch = _validated_batch(
            values[None, ...] if isinstance(values, np.ndarray) else values, horizons
        )
        if expected_length is None:
            expected_length = int(batch.shape[2])
        elif batch.shape[2] != expected_length:
            raise ValueError("channel mapping must preserve one exact frozen length")


def _validate_frozen_horizons(horizons: tuple[int, int, int]) -> None:
    """Require that a serialized model's triple is reachable by Exp417's formula."""
    if not isinstance(horizons, tuple) or len(horizons) != 3:
        raise ValueError("model horizons must match the frozen horizon definition")
    final_horizon = horizons[-1]
    if type(final_horizon) is not int or final_horizon < 1:
        raise ValueError("model horizons must match the frozen horizon definition")
    candidate_lengths = range(10 * (final_horizon - 1) + 1, 10 * final_horizon + 1)
    if not any(build_horizons(length) == horizons for length in candidate_lengths):
        raise ValueError("model horizons must match the frozen horizon definition")


def _validate_device(device: torch.device) -> None:
    if not isinstance(device, torch.device) or device.type not in {"cpu", "mps"}:
        raise ValueError("adapter device must be cpu or mps")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("requested MPS device is unavailable; CPU fallback is forbidden")


def _assert_tensor_device(tensor: torch.Tensor, device: torch.device) -> None:
    if tensor.device != device:
        raise ValueError("unexpected tensor device fallback")


def _assert_model_device(model: CausalMultiHorizonCNN, device: torch.device) -> None:
    if any(parameter.device != device for parameter in model.parameters()):
        raise ValueError("unexpected model device fallback")


def _readonly_float64(values: np.ndarray) -> np.ndarray:
    frozen = np.frombuffer(np.ascontiguousarray(values, dtype=np.float64).tobytes(), dtype=np.float64)
    return frozen.reshape(values.shape)


__all__ = ["FrozenModelAdapter"]
