"""Deterministic CPU causal multi-horizon TRAIN-only models for Exp417."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import random
from types import MappingProxyType

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as functional
from torch.utils.data import DataLoader, TensorDataset

from .config import Exp417FrozenConfig, FROZEN_CONFIG
from .contracts import ProcessedGroup
from .preprocessing import (
    Fold,
    build_horizons,
    canonical_waveform_sha256,
    identity_grouped_folds,
)


_INPUT_CHANNELS = 2
_HIDDEN_CHANNELS = 32
_KERNEL_SIZE = 5
_DILATIONS = (1, 2, 4)
_MAX_EPOCHS = 100
_PATIENCE = 10
_LEARNING_RATE = 1e-3
_WEIGHT_DECAY = 1e-4
_BATCH_LIMIT = 32


class CausalConv1d(nn.Conv1d):
    """A Conv1d with left-only padding so its output has no future access."""

    def forward(self, values: Tensor) -> Tensor:
        left = self.dilation[0] * (self.kernel_size[0] - 1)
        return super().forward(functional.pad(values, (left, 0)))


class _CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.dilation = int(dilation)
        self.kernel_size = _KERNEL_SIZE
        self.causal = CausalConv1d(
            channels, channels, kernel_size=_KERNEL_SIZE, dilation=self.dilation, padding=0
        )
        self.activation = nn.GELU()
        self.projection: nn.Module = nn.Identity()

    def forward(self, values: Tensor) -> Tensor:
        return self.projection(values) + self.activation(self.causal(values))


class CausalMultiHorizonCNN(nn.Module):
    """Frozen two-channel, three-block causal CNN with three raw-value heads."""

    input_channels = _INPUT_CHANNELS
    hidden_channels = _HIDDEN_CHANNELS

    def __init__(self, horizons: Sequence[int]) -> None:
        super().__init__()
        frozen_horizons = _validate_horizons(horizons)
        self.horizons = frozen_horizons
        self.input_projection = nn.Conv1d(_INPUT_CHANNELS, _HIDDEN_CHANNELS, kernel_size=1)
        self.blocks = nn.ModuleList(
            _CausalResidualBlock(_HIDDEN_CHANNELS, dilation) for dilation in _DILATIONS
        )
        self.heads = nn.ModuleDict(
            {str(horizon): nn.Conv1d(_HIDDEN_CHANNELS, 1, kernel_size=1) for horizon in frozen_horizons}
        )

    def forward(self, values: Tensor) -> dict[int, Tensor]:
        if not isinstance(values, Tensor) or values.ndim != 3:
            raise ValueError("model inputs must have shape (batch, two input channels, length)")
        if values.shape[1] != _INPUT_CHANNELS:
            raise ValueError("model requires exactly two input channels")
        if values.shape[2] <= max(self.horizons):
            raise ValueError("model input length leaves no aligned horizon targets")
        if not torch.isfinite(values).all():
            raise ValueError("model inputs must be finite")
        hidden = self.input_projection(values)
        for block in self.blocks:
            hidden = block(hidden)
        return {
            horizon: self.heads[str(horizon)](hidden)[..., : values.shape[2] - horizon]
            for horizon in self.horizons
        }


@dataclass(frozen=True)
class TrainedSeedModel:
    """Immutable CPU state and evidence for one seed/fold fit."""

    seed: int
    horizons: tuple[int, int, int]
    model_sha256: str
    state: Mapping[str, tuple[str, tuple[int, ...], bytes]]
    stop_epoch: int
    training_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    fit_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    early_stop_fit_indices: tuple[int, ...]
    early_stop_validation_indices: tuple[int, ...]
    refit_indices: tuple[int, ...]
    device: str = "cpu"

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed not in FROZEN_CONFIG.model_seeds:
            raise ValueError("trained model seed must be one of the frozen model seeds")
        if self.device not in {"cpu", "mps"}:
            raise ValueError("trained model device must be cpu or mps")
        object.__setattr__(self, "horizons", _validate_horizons(self.horizons))
        if not isinstance(self.model_sha256, str) or len(self.model_sha256) != 64:
            raise ValueError("model_sha256 must be a SHA256 hex digest")
        if type(self.stop_epoch) is not int or not 1 <= self.stop_epoch <= _MAX_EPOCHS:
            raise ValueError("stop epoch must be within the frozen 100-epoch limit")
        for name in ("training_losses", "validation_losses"):
            values = tuple(getattr(self, name))
            valid_length = (
                len(values) == self.stop_epoch
                if name == "training_losses"
                else self.stop_epoch <= len(values) <= _MAX_EPOCHS
            )
            if not valid_length or not all(np.isfinite(value) for value in values):
                raise ValueError(f"{name} must contain finite evidence for every completed epoch")
            object.__setattr__(self, name, tuple(float(value) for value in values))
        state = _freeze_state(self.state)
        if _canonical_state_sha256(state) != self.model_sha256:
            raise ValueError("model SHA does not match canonical CPU state")
        object.__setattr__(self, "state", state)
        for name in (
            "fit_indices",
            "validation_indices",
            "early_stop_fit_indices",
            "early_stop_validation_indices",
            "refit_indices",
        ):
            indices = tuple(getattr(self, name))
            if any(type(index) is not int or index < 0 for index in indices):
                raise ValueError(f"{name} must contain non-negative integer indices")
            if indices != tuple(sorted(indices)) or len(indices) != len(set(indices)):
                raise ValueError(f"{name} must be sorted and unique")
            object.__setattr__(self, name, indices)
        fit = set(self.fit_indices)
        validation = set(self.validation_indices)
        early_fit = set(self.early_stop_fit_indices)
        early_validation = set(self.early_stop_validation_indices)
        refit = set(self.refit_indices)
        if fit & validation or validation & early_fit or validation & early_validation:
            raise ValueError("OOF validation identities cannot appear in model fit or early stopping")
        if not early_fit.issubset(fit) or not early_validation.issubset(fit):
            raise ValueError("early-stop identities must be drawn from the OOF fit side")
        if len(early_fit) < 5 or len(early_validation) < 5:
            raise ValueError("early stopping requires at least five identity groups on both sides")
        if refit != fit:
            raise ValueError("final refit must use every identity in the declared fit partition")


@dataclass(frozen=True)
class OOFModelBundle:
    """All five OOF folds for each frozen seed and immutable residual evidence."""

    length: int
    universe_indices: tuple[int, ...]
    horizons: tuple[int, int, int]
    folds: tuple[Fold, ...]
    models: tuple[TrainedSeedModel, ...]
    oof_residuals: Mapping[int, Mapping[int, Mapping[int, np.ndarray]]]

    def __post_init__(self) -> None:
        if type(self.length) is not int or self.length <= 0:
            raise ValueError("OOF length must be a positive integer")
        universe = tuple(self.universe_indices)
        if not universe or any(type(index) is not int or index < 0 for index in universe):
            raise ValueError("OOF universe must contain non-negative integer identities")
        if universe != tuple(sorted(universe)) or len(universe) != len(set(universe)):
            raise ValueError("OOF universe must be sorted and unique")
        horizons = _validate_horizons(self.horizons)
        if horizons != build_horizons(self.length):
            raise ValueError("OOF horizons must match the exact group length")
        folds = tuple(self.folds)
        if len(folds) != 5 or any(not isinstance(fold, Fold) for fold in folds):
            raise ValueError("OOF bundle requires exactly five identity folds")
        universe_set = set(universe)
        covered: set[int] = set()
        for fold in folds:
            validation = set(fold.validation_indices)
            if not validation or not validation.issubset(universe_set):
                raise ValueError("OOF validation partitions must be non-empty within the universe")
            if covered & validation:
                raise ValueError("OOF validation partitions must be pairwise disjoint")
            if set(fold.fit_indices) | validation != universe_set:
                raise ValueError("OOF fold partitions must cover the exact universe")
            covered.update(validation)
        if covered != universe_set:
            raise ValueError("OOF validation partitions must cover the exact universe")
        models = tuple(self.models)
        if len(models) != len(FROZEN_CONFIG.model_seeds) * len(folds):
            raise ValueError("OOF bundle requires every frozen seed in every fold")
        if any(not isinstance(model, TrainedSeedModel) or model.horizons != horizons for model in models):
            raise ValueError("OOF models must use the frozen horizons")
        expected_bindings = {
            (seed, fold.fit_indices, fold.validation_indices, fold.early_stop_fit_indices, fold.early_stop_validation_indices)
            for seed in FROZEN_CONFIG.model_seeds
            for fold in folds
        }
        actual_bindings = [
            (model.seed, model.fit_indices, model.validation_indices, model.early_stop_fit_indices, model.early_stop_validation_indices)
            for model in models
        ]
        if len(actual_bindings) != len(set(actual_bindings)) or set(actual_bindings) != expected_bindings:
            raise ValueError("OOF models must attest exactly one complete seed/fold partition")
        frozen_residuals = _freeze_oof_residuals(self.oof_residuals, horizons, universe, self.length)
        object.__setattr__(self, "length", self.length)
        object.__setattr__(self, "universe_indices", universe)
        object.__setattr__(self, "horizons", horizons)
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "oof_residuals", frozen_residuals)


@dataclass(frozen=True)
class SeedOOFTraining:
    """One frozen seed's five OOF fits before Exp419 reassembles all seeds."""

    seed: int
    length: int
    horizons: tuple[int, int, int]
    folds: tuple[Fold, ...]
    models: tuple[TrainedSeedModel, ...]
    oof_residuals: Mapping[int, Mapping[int, np.ndarray]]

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed not in FROZEN_CONFIG.model_seeds:
            raise ValueError("seed OOF training seed must be frozen")
        if type(self.length) is not int or self.length <= 0:
            raise ValueError("seed OOF training length must be positive")
        horizons = _validate_horizons(self.horizons)
        if horizons != build_horizons(self.length):
            raise ValueError("seed OOF training horizons must match length")
        folds = tuple(self.folds)
        models = tuple(self.models)
        if len(folds) != 5 or len(models) != len(folds):
            raise ValueError("seed OOF training requires one model for each fold")
        if any(not isinstance(fold, Fold) for fold in folds) or any(
            not isinstance(model, TrainedSeedModel) for model in models
        ):
            raise ValueError("seed OOF training contains invalid evidence")
        for fold, model in zip(folds, models, strict=True):
            if (
                model.seed != self.seed
                or model.horizons != horizons
                or model.fit_indices != fold.fit_indices
                or model.validation_indices != fold.validation_indices
                or model.early_stop_fit_indices != fold.early_stop_fit_indices
                or model.early_stop_validation_indices != fold.early_stop_validation_indices
            ):
                raise ValueError("seed OOF model does not match its fold")
        expected_indices = tuple(sorted(index for fold in folds for index in fold.validation_indices))
        if len(expected_indices) != len(set(expected_indices)):
            raise ValueError("seed OOF validation identities overlap")
        frozen = _freeze_seed_oof_residuals(
            self.oof_residuals, horizons, expected_indices, self.length
        )
        object.__setattr__(self, "horizons", horizons)
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "oof_residuals", frozen)


def set_deterministic_seed(seed: int, device: torch.device) -> None:
    """Set all fitting RNGs without silently changing the requested device."""
    _validate_seed(seed)
    _validate_device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "mps":
        torch.mps.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _validate_seed(seed: int) -> None:
    if type(seed) is not int or seed not in FROZEN_CONFIG.model_seeds:
        raise ValueError("seed must be a frozen Exp417 model seed")


def train_oof_models(
    group: ProcessedGroup, *, device: torch.device, config: Exp417FrozenConfig = FROZEN_CONFIG
) -> OOFModelBundle:
    """Fit every frozen seed on five identity-disjoint OOF folds on the requested device."""
    channels, horizons = _validated_group_channels(group, config)
    _validate_device(device)
    folds = identity_grouped_folds(group.group.series)
    models: list[TrainedSeedModel] = []
    mutable_residuals: dict[int, dict[int, dict[int, np.ndarray]]] = {
        seed: {} for seed in config.model_seeds
    }
    for fold in folds:
        for seed in config.model_seeds:
            trained = _fit_seed_model(
                channels,
                horizons,
                seed=seed,
                device=device,
                fit_indices=fold.fit_indices,
                validation_indices=fold.validation_indices,
                early_stop_fit_indices=fold.early_stop_fit_indices,
                early_stop_validation_indices=fold.early_stop_validation_indices,
            )
            models.append(trained)
            for index in fold.validation_indices:
                mutable_residuals[seed][index] = dict(predict_residuals(trained, channels[index]))
    return OOFModelBundle(
        length=group.group.length,
        universe_indices=tuple(item.instance_index for item in group.group.series),
        horizons=horizons,
        folds=folds,
        models=tuple(models),
        oof_residuals=mutable_residuals,
    )


def fit_final_seed_models(
    group: ProcessedGroup, *, device: torch.device, config: Exp417FrozenConfig = FROZEN_CONFIG
) -> tuple[TrainedSeedModel, ...]:
    """Fit the three frozen seeds on all normal TRAIN identities with early stopping."""
    channels, horizons = _validated_group_channels(group, config)
    _validate_device(device)
    folds = identity_grouped_folds(group.group.series)
    identities = tuple(sorted(channels))
    identity_by_index = {
        item.instance_index: canonical_waveform_sha256(item) for item in group.group.series
    }
    early_validation = tuple(
        index for index in identities if int(identity_by_index[index], 16) % 10 == 0
    )
    early_fit = tuple(index for index in identities if index not in set(early_validation))
    if len({identity_by_index[index] for index in early_fit}) < 5 or len(
        {identity_by_index[index] for index in early_validation}
    ) < 5:
        raise ValueError("final fit requires five identity groups on both early-stop sides")
    return tuple(
        _fit_seed_model(
            channels,
            horizons,
            seed=seed,
            device=device,
            fit_indices=identities,
            validation_indices=(),
            early_stop_fit_indices=early_fit,
            early_stop_validation_indices=early_validation,
        )
        for seed in config.model_seeds
    )


def prepare_seed_initial_states(
    horizons: tuple[int, int, int], device: torch.device
) -> Mapping[int, Mapping[str, tuple[str, tuple[int, ...], bytes]]]:
    """Create immutable seed states serially before concurrent fitting begins."""
    frozen_horizons = _validate_horizons(horizons)
    _validate_device(device)
    return MappingProxyType(
        {
            seed: _initial_state(seed, frozen_horizons, device)
            for seed in FROZEN_CONFIG.model_seeds
        }
    )


def fit_seed_oof_from_state(
    group: ProcessedGroup,
    *,
    seed: int,
    initial_state: Mapping[str, tuple[str, tuple[int, ...], bytes]],
    device: torch.device,
) -> SeedOOFTraining:
    """Fit a single OOF seed from a state prepared before task concurrency."""
    channels, horizons = _validated_group_channels(group, FROZEN_CONFIG)
    _validate_device(device)
    _validate_seed(seed)
    frozen_state = _freeze_state(initial_state)
    folds = identity_grouped_folds(group.group.series)
    models: list[TrainedSeedModel] = []
    residuals: dict[int, dict[int, np.ndarray]] = {}
    for fold in folds:
        trained = _fit_seed_model_from_state(
            channels,
            horizons,
            seed=seed,
            initial_state=frozen_state,
            device=device,
            fit_indices=fold.fit_indices,
            validation_indices=fold.validation_indices,
            early_stop_fit_indices=fold.early_stop_fit_indices,
            early_stop_validation_indices=fold.early_stop_validation_indices,
        )
        models.append(trained)
        for index in fold.validation_indices:
            residuals[index] = dict(predict_residuals(trained, channels[index]))
    return SeedOOFTraining(
        seed=seed,
        length=group.group.length,
        horizons=horizons,
        folds=folds,
        models=tuple(models),
        oof_residuals=residuals,
    )


def fit_seed_final_from_state(
    group: ProcessedGroup,
    *,
    seed: int,
    initial_state: Mapping[str, tuple[str, tuple[int, ...], bytes]],
    device: torch.device,
) -> TrainedSeedModel:
    """Fit one final seed from a state prepared before task concurrency."""
    channels, horizons = _validated_group_channels(group, FROZEN_CONFIG)
    _validate_device(device)
    _validate_seed(seed)
    identities = tuple(sorted(channels))
    identity_by_index = {
        item.instance_index: canonical_waveform_sha256(item) for item in group.group.series
    }
    early_validation = tuple(
        index for index in identities if int(identity_by_index[index], 16) % 10 == 0
    )
    early_fit = tuple(index for index in identities if index not in set(early_validation))
    if len({identity_by_index[index] for index in early_fit}) < 5 or len(
        {identity_by_index[index] for index in early_validation}
    ) < 5:
        raise ValueError("final fit requires five identity groups on both early-stop sides")
    return _fit_seed_model_from_state(
        channels,
        horizons,
        seed=seed,
        initial_state=_freeze_state(initial_state),
        device=device,
        fit_indices=identities,
        validation_indices=(),
        early_stop_fit_indices=early_fit,
        early_stop_validation_indices=early_validation,
    )


def predict_residuals(model: TrainedSeedModel, channels: np.ndarray) -> Mapping[int, np.ndarray]:
    """Return immutable aligned absolute residuals for one two-channel series."""
    return predict_residuals_batch(model, {0: channels}, batch_size=1)[0]


def predict_residuals_batch(
    model: TrainedSeedModel,
    channels_by_index: Mapping[int, np.ndarray],
    *,
    batch_size: int = _BATCH_LIMIT,
) -> Mapping[int, Mapping[int, np.ndarray]]:
    """Return immutable residuals while materializing one network per batch call."""
    if not isinstance(model, TrainedSeedModel):
        raise ValueError("model must be a TrainedSeedModel")
    if not isinstance(channels_by_index, Mapping) or not channels_by_index:
        raise ValueError("channels_by_index must be a non-empty mapping")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    indices = tuple(channels_by_index)
    if (
        any(type(index) is not int or index < 0 for index in indices)
        or indices != tuple(sorted(indices))
        or len(indices) != len(set(indices))
    ):
        raise ValueError("channels_by_index must use sorted unique non-negative indices")
    values_by_index = {
        index: _validated_channels_array(channels_by_index[index], expected_length=None)
        for index in indices
    }
    lengths = {values.shape[1] for values in values_by_index.values()}
    if len(lengths) != 1:
        raise ValueError("channels_by_index must preserve one exact series length")
    length = next(iter(lengths))
    if length <= max(model.horizons):
        raise ValueError("channels leave no aligned horizon targets")
    device = torch.device(model.device)
    _validate_device(device)
    network = _materialize_model(model, device)
    residuals_by_index: dict[int, Mapping[int, np.ndarray]] = {}
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_values = np.stack(
                [values_by_index[index] for index in batch_indices], axis=0
            )
            # A fresh CPU copy avoids PyTorch's non-writeable-array behavior and
            # prevents later caller mutation from affecting inference.
            inputs = torch.from_numpy(
                np.array(batch_values, dtype=np.float32, copy=True, order="C")
            ).to(device)
            _assert_tensor_device(inputs, device)
            predictions = network(inputs)
            for position, index in enumerate(batch_indices):
                residuals = {
                    horizon: np.abs(
                        predictions[horizon][position, 0].detach().cpu().numpy()
                        - values_by_index[index][0, horizon:]
                    ).astype(np.float64, copy=False)
                    for horizon in model.horizons
                }
                residuals_by_index[index] = _freeze_residual_mapping(
                    residuals, model.horizons, length
                )
    return MappingProxyType(dict(sorted(residuals_by_index.items())))


def canonical_model_sha256(model: nn.Module) -> str:
    """Hash named tensors by name, dtype, shape, and CPU-contiguous bytes."""
    if not isinstance(model, nn.Module):
        raise ValueError("model must be a torch module")
    return _canonical_state_sha256(_serialize_state(model.state_dict()))


def _fit_seed_model(
    channels: Mapping[int, np.ndarray],
    horizons: tuple[int, int, int],
    *,
    seed: int,
    device: torch.device,
    fit_indices: tuple[int, ...],
    validation_indices: tuple[int, ...],
    early_stop_fit_indices: tuple[int, ...],
    early_stop_validation_indices: tuple[int, ...],
) -> TrainedSeedModel:
    if len(early_stop_fit_indices) < 5 or len(early_stop_validation_indices) < 5:
        raise ValueError("model fitting requires five identity groups on both early-stop sides")
    selected_epoch, selection_validation_losses = _select_stop_epoch(
        channels,
        horizons,
        seed=seed,
        device=device,
        fit_indices=early_stop_fit_indices,
        validation_indices=early_stop_validation_indices,
    )
    final_state, final_training_losses = _refit_all_identities(
        channels,
        horizons,
        seed=seed,
        device=device,
        fit_indices=fit_indices,
        epochs=selected_epoch,
    )
    return TrainedSeedModel(
        seed=seed,
        horizons=horizons,
        model_sha256=_canonical_state_sha256(final_state),
        state=final_state,
        stop_epoch=selected_epoch,
        training_losses=final_training_losses,
        validation_losses=selection_validation_losses,
        fit_indices=fit_indices,
        validation_indices=validation_indices,
        early_stop_fit_indices=early_stop_fit_indices,
        early_stop_validation_indices=early_stop_validation_indices,
        refit_indices=fit_indices,
        device=device.type,
    )


def _initial_state(
    seed: int, horizons: tuple[int, int, int], device: torch.device
) -> Mapping[str, tuple[str, tuple[int, ...], bytes]]:
    _validate_seed(seed)
    set_deterministic_seed(seed, device)
    network = CausalMultiHorizonCNN(horizons).to(device)
    _assert_model_device(network, device)
    return _serialize_state(network.state_dict())


def _fit_seed_model_from_state(
    channels: Mapping[int, np.ndarray],
    horizons: tuple[int, int, int],
    *,
    seed: int,
    initial_state: Mapping[str, tuple[str, tuple[int, ...], bytes]],
    device: torch.device,
    fit_indices: tuple[int, ...],
    validation_indices: tuple[int, ...],
    early_stop_fit_indices: tuple[int, ...],
    early_stop_validation_indices: tuple[int, ...],
) -> TrainedSeedModel:
    if len(early_stop_fit_indices) < 5 or len(early_stop_validation_indices) < 5:
        raise ValueError("model fitting requires five identity groups on both early-stop sides")
    selected_epoch, selection_validation_losses = _select_stop_epoch_from_state(
        channels,
        horizons,
        initial_state=_freeze_state(initial_state),
        device=device,
        fit_indices=early_stop_fit_indices,
        validation_indices=early_stop_validation_indices,
    )
    final_state, final_training_losses = _refit_all_identities_from_state(
        channels,
        horizons,
        initial_state=_freeze_state(initial_state),
        device=device,
        fit_indices=fit_indices,
        epochs=selected_epoch,
    )
    return TrainedSeedModel(
        seed=seed,
        horizons=horizons,
        model_sha256=_canonical_state_sha256(final_state),
        state=final_state,
        stop_epoch=selected_epoch,
        training_losses=final_training_losses,
        validation_losses=selection_validation_losses,
        fit_indices=fit_indices,
        validation_indices=validation_indices,
        early_stop_fit_indices=early_stop_fit_indices,
        early_stop_validation_indices=early_stop_validation_indices,
        refit_indices=fit_indices,
        device=device.type,
    )


def _select_stop_epoch(
    channels: Mapping[int, np.ndarray],
    horizons: tuple[int, int, int],
    *,
    seed: int,
    device: torch.device,
    fit_indices: tuple[int, ...],
    validation_indices: tuple[int, ...],
) -> tuple[int, tuple[float, ...]]:
    """Use an identity-disjoint TRAIN subset only to choose the stop epoch."""
    set_deterministic_seed(seed, device)
    network = CausalMultiHorizonCNN(horizons).to(device)
    _assert_model_device(network, device)
    return _select_stop_epoch_with_network(
        network,
        channels,
        horizons,
        device=device,
        fit_indices=fit_indices,
        validation_indices=validation_indices,
    )


def _select_stop_epoch_from_state(
    channels: Mapping[int, np.ndarray],
    horizons: tuple[int, int, int],
    *,
    initial_state: Mapping[str, tuple[str, tuple[int, ...], bytes]],
    device: torch.device,
    fit_indices: tuple[int, ...],
    validation_indices: tuple[int, ...],
) -> tuple[int, tuple[float, ...]]:
    network = _materialize_network_from_state(initial_state, horizons, device)
    return _select_stop_epoch_with_network(
        network,
        channels,
        horizons,
        device=device,
        fit_indices=fit_indices,
        validation_indices=validation_indices,
    )


def _select_stop_epoch_with_network(
    network: CausalMultiHorizonCNN,
    channels: Mapping[int, np.ndarray],
    horizons: tuple[int, int, int],
    *,
    device: torch.device,
    fit_indices: tuple[int, ...],
    validation_indices: tuple[int, ...],
) -> tuple[int, tuple[float, ...]]:
    _assert_model_device(network, device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=_LEARNING_RATE, weight_decay=_WEIGHT_DECAY)
    loss_function = nn.HuberLoss()
    training = _channel_batch(channels, fit_indices)
    validation = _channel_batch(channels, validation_indices)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(training)),
        batch_size=min(_BATCH_LIMIT, len(training)),
        shuffle=False,
        num_workers=0,
    )
    best_validation = float("inf")
    validation_losses: list[float] = []
    best_epoch: int | None = None
    stale_epochs = 0
    for epoch in range(_MAX_EPOCHS):
        network.train()
        total_loss = 0.0
        total_rows = 0
        for (batch_cpu,) in loader:
            batch = batch_cpu.to(device)
            _assert_tensor_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = _multi_horizon_huber(network, batch, horizons, loss_function)
            if not torch.isfinite(loss):
                raise ValueError("non-finite TRAIN loss")
            loss.backward()
            optimizer.step()
            _assert_finite_parameters(network)
            total_loss += float(loss.detach().cpu()) * len(batch_cpu)
            total_rows += len(batch_cpu)
        train_loss = total_loss / total_rows
        validation_loss = _evaluate_loss(network, validation, horizons, loss_function, device)
        if not np.isfinite(train_loss) or not np.isfinite(validation_loss):
            raise ValueError("non-finite model loss")
        validation_losses.append(float(validation_loss))
        if validation_loss < best_validation - 1e-12:
            best_validation = validation_loss
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= _PATIENCE:
                break
    if best_epoch is None:
        raise ValueError("early stopping did not produce a finite selected epoch")
    return best_epoch, tuple(validation_losses)


def _refit_all_identities(
    channels: Mapping[int, np.ndarray],
    horizons: tuple[int, int, int],
    *,
    seed: int,
    device: torch.device,
    fit_indices: tuple[int, ...],
    epochs: int,
) -> tuple[Mapping[str, tuple[str, tuple[int, ...], bytes]], tuple[float, ...]]:
    """Reinitialize and fit every declared normal-TRAIN identity exactly once per epoch."""
    if type(epochs) is not int or not 1 <= epochs <= _MAX_EPOCHS:
        raise ValueError("selected stop epoch must be within the frozen limit")
    set_deterministic_seed(seed, device)
    network = CausalMultiHorizonCNN(horizons).to(device)
    _assert_model_device(network, device)
    return _refit_all_identities_with_network(
        network,
        channels,
        horizons,
        device=device,
        fit_indices=fit_indices,
        epochs=epochs,
    )


def _refit_all_identities_from_state(
    channels: Mapping[int, np.ndarray],
    horizons: tuple[int, int, int],
    *,
    initial_state: Mapping[str, tuple[str, tuple[int, ...], bytes]],
    device: torch.device,
    fit_indices: tuple[int, ...],
    epochs: int,
) -> tuple[Mapping[str, tuple[str, tuple[int, ...], bytes]], tuple[float, ...]]:
    if type(epochs) is not int or not 1 <= epochs <= _MAX_EPOCHS:
        raise ValueError("selected stop epoch must be within the frozen limit")
    network = _materialize_network_from_state(initial_state, horizons, device)
    return _refit_all_identities_with_network(
        network,
        channels,
        horizons,
        device=device,
        fit_indices=fit_indices,
        epochs=epochs,
    )


def _refit_all_identities_with_network(
    network: CausalMultiHorizonCNN,
    channels: Mapping[int, np.ndarray],
    horizons: tuple[int, int, int],
    *,
    device: torch.device,
    fit_indices: tuple[int, ...],
    epochs: int,
) -> tuple[Mapping[str, tuple[str, tuple[int, ...], bytes]], tuple[float, ...]]:
    _assert_model_device(network, device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=_LEARNING_RATE, weight_decay=_WEIGHT_DECAY)
    loss_function = nn.HuberLoss()
    training = _channel_batch(channels, fit_indices)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(training)),
        batch_size=min(_BATCH_LIMIT, len(training)),
        shuffle=False,
        num_workers=0,
    )
    losses: list[float] = []
    for _epoch in range(epochs):
        network.train()
        total_loss = 0.0
        total_rows = 0
        for (batch_cpu,) in loader:
            batch = batch_cpu.to(device)
            _assert_tensor_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = _multi_horizon_huber(network, batch, horizons, loss_function)
            if not torch.isfinite(loss):
                raise ValueError("non-finite TRAIN loss")
            loss.backward()
            optimizer.step()
            _assert_finite_parameters(network)
            total_loss += float(loss.detach().cpu()) * len(batch_cpu)
            total_rows += len(batch_cpu)
        current_loss = total_loss / total_rows
        if not np.isfinite(current_loss):
            raise ValueError("non-finite final-refit loss")
        losses.append(float(current_loss))
    return _serialize_state(network.state_dict()), tuple(losses)


def _multi_horizon_huber(
    network: CausalMultiHorizonCNN,
    channels: Tensor,
    horizons: tuple[int, int, int],
    loss_function: nn.HuberLoss,
) -> Tensor:
    predictions = network(channels)
    losses = [
        loss_function(predictions[horizon][:, 0], channels[:, 0, horizon:])
        for horizon in horizons
    ]
    loss = torch.stack(losses).mean()
    if not torch.isfinite(loss):
        raise ValueError("non-finite multi-horizon Huber loss")
    return loss


def _evaluate_loss(
    network: CausalMultiHorizonCNN,
    validation: np.ndarray,
    horizons: tuple[int, int, int],
    loss_function: nn.HuberLoss,
    device: torch.device,
) -> float:
    network.eval()
    with torch.no_grad():
        batch = torch.from_numpy(validation).to(device)
        _assert_tensor_device(batch, device)
        loss = _multi_horizon_huber(network, batch, horizons, loss_function)
    return float(loss.detach().cpu())


def _validated_group_channels(
    group: ProcessedGroup, config: Exp417FrozenConfig) -> tuple[Mapping[int, np.ndarray], tuple[int, int, int]]:
    if not isinstance(config, Exp417FrozenConfig):
        raise ValueError("config must be an Exp417FrozenConfig")
    config.validate()
    if not isinstance(group, ProcessedGroup):
        raise ValueError("group must be a ProcessedGroup")
    if group.status not in {"diagnostic_only", "candidate_eligible"}:
        raise ValueError("cannot train an unusable or abstained processed group")
    if len(group.group.series) < config.min_train_model:
        raise ValueError("processed group has insufficient frozen TRAIN series")
    channels = group.channels
    if not isinstance(channels, Mapping) or not channels:
        raise ValueError("processed group requires complete channels")
    expected = tuple(item.instance_index for item in group.group.series)
    if tuple(sorted(channels)) != expected:
        raise ValueError("processed group channels are incomplete")
    for values in channels.values():
        _validated_channels_array(values, expected_length=group.group.length)
    return channels, build_horizons(group.group.length)


def _validated_channels_array(channels: np.ndarray, *, expected_length: int | None) -> np.ndarray:
    if not isinstance(channels, np.ndarray) or channels.dtype != np.dtype("float32"):
        raise ValueError("channels must be float32 arrays")
    if channels.ndim != 2 or channels.shape[0] != _INPUT_CHANNELS:
        raise ValueError("channels must have exactly two-channel shape")
    if expected_length is not None and channels.shape != (_INPUT_CHANNELS, expected_length):
        raise ValueError("channels must match the exact group length")
    if not np.isfinite(channels).all():
        raise ValueError("channels must be finite")
    return np.ascontiguousarray(channels, dtype=np.float32)


def _channel_batch(channels: Mapping[int, np.ndarray], indices: Sequence[int]) -> np.ndarray:
    batch = np.stack([_validated_channels_array(channels[index], expected_length=None) for index in indices])
    if not np.isfinite(batch).all():
        raise ValueError("channel batch must be finite")
    return np.ascontiguousarray(batch, dtype=np.float32)


def _validate_horizons(horizons: Sequence[int]) -> tuple[int, int, int]:
    if isinstance(horizons, (str, bytes)):
        raise ValueError("horizons must be a sequence")
    values = tuple(horizons)
    if len(values) != 3 or any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("horizons must contain three distinct positive integers")
    if values != tuple(sorted(values)) or len(set(values)) != 3:
        raise ValueError("horizons must contain three distinct sorted integers")
    return values  # type: ignore[return-value]


def _validate_device(device: torch.device) -> None:
    if not isinstance(device, torch.device):
        raise ValueError("device must be a torch.device")
    if device.type not in {"cpu", "mps"}:
        raise ValueError("Exp417 supports only cpu or mps devices")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("requested MPS device is unavailable; CPU fallback is forbidden")


def _assert_tensor_device(tensor: Tensor, device: torch.device) -> None:
    if not _device_matches(tensor.device, device):
        raise ValueError("unexpected tensor device fallback")


def _assert_model_device(model: nn.Module, device: torch.device) -> None:
    if any(not _device_matches(parameter.device, device) for parameter in model.parameters()):
        raise ValueError("unexpected model device fallback")


def _device_matches(actual: torch.device, requested: torch.device) -> bool:
    """Treat the canonical ``mps`` device and its ``mps:0`` alias equally."""
    if not isinstance(actual, torch.device) or not isinstance(requested, torch.device):
        return False
    if actual.type != requested.type:
        return False
    if requested.type == "mps":
        return actual.index in {None, 0} and requested.index in {None, 0}
    return actual == requested


def _assert_finite_parameters(model: nn.Module) -> None:
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("non-finite model parameters")


def _serialize_state(state_dict: Mapping[str, Tensor]) -> Mapping[str, tuple[str, tuple[int, ...], bytes]]:
    serialized: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    for name, tensor in state_dict.items():
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise ValueError("model state must contain named tensors")
        cpu = tensor.detach().to(device="cpu").contiguous()
        if not torch.isfinite(cpu).all():
            raise ValueError("model state contains non-finite parameters")
        serialized[name] = (str(cpu.dtype), tuple(cpu.shape), cpu.numpy().tobytes(order="C"))
    return MappingProxyType(dict(sorted(serialized.items())))


def _freeze_state(state: Mapping[str, tuple[str, tuple[int, ...], bytes]]) -> Mapping[str, tuple[str, tuple[int, ...], bytes]]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("model state must be a non-empty mapping")
    copied: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    for name, record in state.items():
        if not isinstance(name, str) or not isinstance(record, tuple) or len(record) != 3:
            raise ValueError("model state record is malformed")
        dtype_name, shape, payload = record
        if not isinstance(dtype_name, str) or not isinstance(shape, tuple) or not isinstance(payload, bytes):
            raise ValueError("model state record is malformed")
        if any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise ValueError("model state shape is malformed")
        dtype = _torch_dtype(dtype_name)
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * torch.empty((), dtype=dtype).element_size()
        if len(payload) != expected_bytes:
            raise ValueError("model state byte length is malformed")
        tensor = torch.frombuffer(bytearray(payload), dtype=dtype).reshape(shape)
        if not torch.isfinite(tensor).all():
            raise ValueError("model state contains non-finite parameters")
        copied[name] = (dtype_name, shape, bytes(payload))
    return MappingProxyType(dict(sorted(copied.items())))


def _canonical_state_sha256(state: Mapping[str, tuple[str, tuple[int, ...], bytes]]) -> str:
    digest = sha256()
    for name in sorted(state):
        dtype_name, shape, payload = state[name]
        for value in (name.encode("utf-8"), dtype_name.encode("ascii")):
            digest.update(len(value).to_bytes(8, "little"))
            digest.update(value)
        digest.update(len(shape).to_bytes(8, "little"))
        for dimension in shape:
            digest.update(dimension.to_bytes(8, "little", signed=False))
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _materialize_model(trained: TrainedSeedModel, device: torch.device) -> CausalMultiHorizonCNN:
    return _materialize_network_from_state(trained.state, trained.horizons, device)


def _materialize_network_from_state(
    state: Mapping[str, tuple[str, tuple[int, ...], bytes]],
    horizons: tuple[int, int, int],
    device: torch.device,
) -> CausalMultiHorizonCNN:
    _validate_device(device)
    frozen_horizons = _validate_horizons(horizons)
    frozen_state = _freeze_state(state)
    state: dict[str, Tensor] = {}
    for name, (dtype_name, shape, payload) in frozen_state.items():
        tensor = torch.frombuffer(bytearray(payload), dtype=_torch_dtype(dtype_name)).clone().reshape(shape)
        state[name] = tensor
    network = CausalMultiHorizonCNN(frozen_horizons).to(device)
    network.load_state_dict(state, strict=True)
    _assert_model_device(network, device)
    _assert_finite_parameters(network)
    network.eval()
    return network


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "torch.float32": torch.float32,
        "torch.float64": torch.float64,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.int64": torch.int64,
        "torch.int32": torch.int32,
        "torch.int16": torch.int16,
        "torch.int8": torch.int8,
        "torch.uint8": torch.uint8,
        "torch.bool": torch.bool,
    }
    if name not in mapping:
        raise ValueError("unsupported CPU state dtype")
    return mapping[name]


def _freeze_residual_mapping(
    values: Mapping[int, np.ndarray], horizons: tuple[int, int, int], length: int
) -> Mapping[int, np.ndarray]:
    if tuple(sorted(values)) != horizons:
        raise ValueError("residuals must cover the frozen horizons")
    frozen: dict[int, np.ndarray] = {}
    for horizon in horizons:
        residual = np.asarray(values[horizon], dtype=np.float64)
        if residual.ndim != 1 or len(residual) != length - horizon:
            raise ValueError("residual length must match the exact aligned horizon region")
        if not np.isfinite(residual).all():
            raise ValueError("residuals must be finite")
        frozen[horizon] = np.frombuffer(np.ascontiguousarray(residual).tobytes(), dtype=np.float64)
    return MappingProxyType(frozen)


def _freeze_oof_residuals(
    values: Mapping[int, Mapping[int, Mapping[int, np.ndarray]]],
    horizons: tuple[int, int, int],
    universe: tuple[int, ...],
    length: int,
) -> Mapping[int, Mapping[int, Mapping[int, np.ndarray]]]:
    if not isinstance(values, Mapping) or tuple(sorted(values)) != FROZEN_CONFIG.model_seeds:
        raise ValueError("OOF residuals must contain every frozen seed")
    frozen: dict[int, Mapping[int, Mapping[int, np.ndarray]]] = {}
    for seed in FROZEN_CONFIG.model_seeds:
        by_index = values[seed]
        if not isinstance(by_index, Mapping):
            raise ValueError("OOF residuals must map seed to identity evidence")
        if set(by_index) != set(universe):
            raise ValueError("OOF residual identity coverage must match the exact universe")
        frozen[seed] = MappingProxyType(
            {
                index: _freeze_residual_mapping(residuals, horizons, length)
                for index, residuals in sorted(by_index.items())
                if type(index) is int and index >= 0
            }
        )
        if len(frozen[seed]) != len(by_index):
            raise ValueError("OOF residual identity is invalid")
    return MappingProxyType(frozen)


def _freeze_seed_oof_residuals(
    values: Mapping[int, Mapping[int, np.ndarray]],
    horizons: tuple[int, int, int],
    universe: tuple[int, ...],
    length: int,
) -> Mapping[int, Mapping[int, np.ndarray]]:
    if not isinstance(values, Mapping) or tuple(sorted(values)) != universe:
        raise ValueError("seed OOF residual identities must match validation universe")
    frozen: dict[int, Mapping[int, np.ndarray]] = {}
    for index in universe:
        residuals = values[index]
        if not isinstance(residuals, Mapping):
            raise ValueError("seed OOF residual mapping is invalid")
        frozen[index] = _freeze_residual_mapping(residuals, horizons, length)
    return MappingProxyType(frozen)


__all__ = [
    "CausalConv1d",
    "CausalMultiHorizonCNN",
    "OOFModelBundle",
    "SeedOOFTraining",
    "TrainedSeedModel",
    "canonical_model_sha256",
    "fit_final_seed_models",
    "fit_seed_final_from_state",
    "fit_seed_oof_from_state",
    "prepare_seed_initial_states",
    "predict_residuals",
    "predict_residuals_batch",
    "set_deterministic_seed",
    "train_oof_models",
]
