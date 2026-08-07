"""TRAIN-only exact-length preprocessing for Exp417."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import ceil

import numpy as np

from experiments.exp414_missing_aware_near_repeat.repair import repair_train_blob

from .config import Exp417FrozenConfig, FROZEN_CONFIG
from .contracts import LengthGroup, ProcessedGroup, RawSeries, RawTrainDataset


_MAD_TO_NORMAL_SCALE = 1.4826
_EARLY_STOP_MODULUS = 10


@dataclass(frozen=True)
class Fold:
    """One outer OOF fold and its TRAIN-only identity-disjoint early split."""

    fit_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    early_stop_fit_indices: tuple[int, ...]
    early_stop_validation_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "fit_indices",
            "validation_indices",
            "early_stop_fit_indices",
            "early_stop_validation_indices",
        ):
            values = getattr(self, name)
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise ValueError(f"{name} must be a sequence")
            frozen = tuple(values)
            if any(type(index) is not int or index < 0 for index in frozen):
                raise ValueError(f"{name} must contain non-negative integer indices")
            if frozen != tuple(sorted(frozen)) or len(frozen) != len(set(frozen)):
                raise ValueError(f"{name} must be sorted and unique")
            object.__setattr__(self, name, frozen)
        fit = set(self.fit_indices)
        validation = set(self.validation_indices)
        early_fit = set(self.early_stop_fit_indices)
        early_validation = set(self.early_stop_validation_indices)
        if fit & validation:
            raise ValueError("outer fit and validation indices overlap")
        if not early_fit.issubset(fit) or not early_validation.issubset(fit):
            raise ValueError("early-stop indices must be within outer fit indices")
        if early_fit & early_validation:
            raise ValueError("early-stop fit and validation indices overlap")


def build_horizons(length: int) -> tuple[int, int, int]:
    """Return the three frozen, usable causal forecast offsets for ``length``."""
    if type(length) is not int or length <= 0:
        raise ValueError("length must be a positive integer")
    horizons = (1, ceil(0.02 * length), ceil(0.10 * length))
    if len(set(horizons)) != 3:
        raise ValueError("horizons must be distinct")
    if any(horizon >= length for horizon in horizons):
        raise ValueError("horizons must leave a causal prediction region")
    return horizons


def canonical_waveform_sha256(series: RawSeries) -> str:
    """Hash exact length plus canonical little-endian float32 waveform bytes."""
    if not isinstance(series, RawSeries):
        raise ValueError("series must be a RawSeries")
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.asarray(series.values, dtype="<f4")
    if values.ndim != 1 or len(values) != len(series.values):
        raise ValueError("series values must be one-dimensional")
    payload = len(values).to_bytes(8, byteorder="little", signed=False) + values.tobytes(order="C")
    return sha256(payload).hexdigest()


def identity_grouped_folds(
    series: Sequence[RawSeries], *, folds: int = 5
) -> tuple[Fold, ...]:
    """Build five deterministic OOF folds without waveform-identity leakage."""
    if type(folds) is not int or folds != 5:
        raise ValueError("Exp417 requires exactly five folds")
    if isinstance(series, (str, bytes)) or not isinstance(series, Sequence) or not series:
        raise ValueError("series must be a non-empty sequence")
    ordered = tuple(sorted(series, key=lambda item: item.instance_index))
    if any(not isinstance(item, RawSeries) for item in ordered):
        raise ValueError("series must contain RawSeries records")
    indices = tuple(item.instance_index for item in ordered)
    if len(indices) != len(set(indices)):
        raise ValueError("series indices must be unique")

    identity_groups: dict[str, tuple[int, ...]] = {}
    mutable_groups: dict[str, list[int]] = {}
    for item in ordered:
        mutable_groups.setdefault(canonical_waveform_sha256(item), []).append(item.instance_index)
    for identity, members in mutable_groups.items():
        identity_groups[identity] = tuple(sorted(members))

    assigned: list[list[str]] = [[] for _ in range(folds)]
    sizes = [0] * folds
    for identity in sorted(identity_groups):
        target = min(range(folds), key=lambda fold_index: (sizes[fold_index], fold_index))
        assigned[target].append(identity)
        sizes[target] += len(identity_groups[identity])

    all_indices = set(indices)
    result: list[Fold] = []
    for validation_identities in assigned:
        validation = tuple(
            sorted(index for identity in validation_identities for index in identity_groups[identity])
        )
        validation_set = set(validation)
        fit = tuple(sorted(all_indices - validation_set))
        fit_identities = [
            identity
            for identity in sorted(identity_groups)
            if identity not in set(validation_identities)
        ]
        early_validation_identities = [
            identity
            for identity in fit_identities
            if int(identity, 16) % _EARLY_STOP_MODULUS == 0
        ]
        early_fit_identities = [
            identity for identity in fit_identities if identity not in set(early_validation_identities)
        ]
        if len(early_validation_identities) < 5 or len(early_fit_identities) < 5:
            raise ValueError("early stopping requires at least five identities on both sides")
        early_validation = tuple(
            sorted(index for identity in early_validation_identities for index in identity_groups[identity])
        )
        early_fit = tuple(
            sorted(index for identity in early_fit_identities for index in identity_groups[identity])
        )
        result.append(
            Fold(
                fit_indices=fit,
                validation_indices=validation,
                early_stop_fit_indices=early_fit,
                early_stop_validation_indices=early_validation,
            )
        )
    return tuple(result)


def preprocess_length_groups(
    dataset: RawTrainDataset, *, config: Exp417FrozenConfig = FROZEN_CONFIG
) -> tuple[ProcessedGroup, ...]:
    """Repair then scale normal TRAIN values independently by their exact length."""
    if not isinstance(config, Exp417FrozenConfig):
        raise ValueError("config must be an Exp417FrozenConfig")
    config.validate()
    if not isinstance(dataset, RawTrainDataset):
        raise ValueError("dataset must be a RawTrainDataset")
    by_length: dict[int, list[RawSeries]] = {}
    for item in dataset.series:
        by_length.setdefault(len(item.values), []).append(item)
    return tuple(
        _preprocess_group(dataset.dataset, length, tuple(series), config)
        for length, series in sorted(by_length.items())
    )


def _preprocess_group(
    dataset: str,
    length: int,
    series: tuple[RawSeries, ...],
    config: Exp417FrozenConfig,
) -> ProcessedGroup:
    repaired: list[RawSeries] = []
    for item in series:
        payload = _little_endian_float32_bytes(item.values)
        repaired_run, _audit = repair_train_blob(dataset, item.instance_index, payload)
        if repaired_run is None:
            return _abstained_group(length, series, "nonfinite_abstain")
        if len(repaired_run.values) != length:
            return _abstained_group(length, series, "nonfinite_abstain")
        repaired_values = np.asarray(repaired_run.values, dtype=np.float64)
        if not np.isfinite(repaired_values).all():
            return _abstained_group(length, series, "nonfinite_abstain")
        repaired.append(RawSeries(instance_index=item.instance_index, values=tuple(repaired_run.values)))

    group = LengthGroup(length=length, series=tuple(repaired))
    raw = np.asarray([item.values for item in group.series], dtype=np.float64)
    differences = np.diff(raw, axis=1, prepend=raw[:, :1])
    if not np.isfinite(raw).all() or not np.isfinite(differences).all():
        return _abstained_group(length, tuple(repaired), "nonfinite_abstain")
    raw_center, raw_scale = _robust_center_scale(raw)
    difference_center, difference_scale = _robust_center_scale(differences)
    if not all(
        np.isfinite(value)
        for value in (raw_center, raw_scale, difference_center, difference_scale)
    ):
        return ProcessedGroup(
            group=group,
            raw_center=0.0,
            raw_scale=0.0,
            difference_center=0.0,
            difference_scale=0.0,
            status="nonfinite_abstain",
        )
    if raw_scale <= 1e-12 or difference_scale <= 1e-12:
        return ProcessedGroup(
            group=group,
            raw_center=raw_center,
            raw_scale=raw_scale,
            difference_center=difference_center,
            difference_scale=difference_scale,
            status="zero_scale_abstain",
        )
    try:
        build_horizons(length)
    except ValueError:
        return ProcessedGroup(
            group=group,
            raw_center=raw_center,
            raw_scale=raw_scale,
            difference_center=difference_center,
            difference_scale=difference_scale,
            status="horizon_abstain",
        )
    if len(group.series) < config.min_train_model:
        return ProcessedGroup(
            group=group,
            raw_center=raw_center,
            raw_scale=raw_scale,
            difference_center=difference_center,
            difference_scale=difference_scale,
            status="insufficient_train_abstain",
        )

    channels = _standardized_channels(
        group.series,
        raw,
        differences,
        raw_center,
        raw_scale,
        difference_center,
        difference_scale,
    )
    if channels is None:
        return ProcessedGroup(
            group=group,
            raw_center=raw_center,
            raw_scale=raw_scale,
            difference_center=difference_center,
            difference_scale=difference_scale,
            status="nonfinite_abstain",
        )
    status = (
        "candidate_eligible"
        if len(group.series) >= config.min_train_candidate
        else "diagnostic_only"
    )
    return ProcessedGroup(
        group=group,
        raw_center=raw_center,
        raw_scale=raw_scale,
        difference_center=difference_center,
        difference_scale=difference_scale,
        status=status,
        channels=channels,
    )


def _little_endian_float32_bytes(values: Sequence[float]) -> bytes:
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(values, dtype="<f4").tobytes(order="C")


def _robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    scale = float(_MAD_TO_NORMAL_SCALE * np.median(np.abs(values - center)))
    return center, scale


def _standardized_channels(
    series: Sequence[RawSeries],
    raw: np.ndarray,
    differences: np.ndarray,
    raw_center: float,
    raw_scale: float,
    difference_center: float,
    difference_scale: float,
) -> Mapping[int, np.ndarray] | None:
    raw_channel = (raw - raw_center) / raw_scale
    difference_channel = (differences - difference_center) / difference_scale
    standardized = np.stack((raw_channel, difference_channel), axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        output = np.asarray(standardized, dtype=np.float32)
    if not np.isfinite(output).all():
        return None
    return {
        item.instance_index: np.array(output[position], dtype=np.float32, copy=True, order="C")
        for position, item in enumerate(series)
    }


def _abstained_group(
    length: int, series: Sequence[RawSeries], status: str
) -> ProcessedGroup:
    return ProcessedGroup(
        group=LengthGroup(length=length, series=tuple(series)),
        raw_center=0.0,
        raw_scale=0.0,
        difference_center=0.0,
        difference_scale=0.0,
        status=status,
    )


__all__ = [
    "Fold",
    "build_horizons",
    "canonical_waveform_sha256",
    "identity_grouped_folds",
    "preprocess_length_groups",
]
