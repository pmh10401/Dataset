"""Deterministic normal-TRAIN omission masks for Exp444.

The mask generator is intentionally independent of the numerical scorer.  It
normalizes TRAIN instance identities first, derives every identity from a
canonical JSON payload, and uses the frozen Exp444 seed for the bounded large
combination case.  No random global state is touched.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import combinations
import json
import math
from typing import Iterable, Sequence

from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import OmissionMaskAntigravity


def _canonical_bytes_antigravity(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _validated_inputs_antigravity(
    dataset: str,
    raw_length: int,
    train_indices: Iterable[int],
) -> tuple[str, int, tuple[int, ...]]:
    if type(dataset) is not str or not dataset:
        raise ValueError("dataset must be a non-empty string")
    if type(raw_length) is not int or raw_length < 1:
        raise ValueError("raw_length must be a positive exact integer")
    if isinstance(train_indices, (str, bytes)):
        raise ValueError("train_indices must be a sequence of integers")
    try:
        values = tuple(train_indices)
    except TypeError as error:
        raise ValueError("train_indices must be iterable") from error
    if any(type(item) is not int or item < 0 for item in values):
        raise ValueError("train_indices must contain nonnegative exact integers")
    normalized = tuple(sorted(values))
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError("train_indices must be non-empty and unique")
    omit_count = omission_count_antigravity(len(normalized))
    if len(normalized) - omit_count < FROZEN_CONFIG.minimum_train_rows:
        raise ValueError("omission would retain fewer than four TRAIN rows")
    return dataset, raw_length, normalized


def omission_count_antigravity(train_count: int) -> int:
    """Return ``max(1, floor(0.10*n))`` without floating-point rounding."""
    if type(train_count) is not int or train_count < 1:
        raise ValueError("train_count must be a positive exact integer")
    return max(1, train_count // 10)


def required_survival_count_antigravity(view_count: int) -> int:
    """Return the exact ``ceil(0.875 * view_count)`` requirement."""
    if type(view_count) is not int or view_count < 1:
        raise ValueError("view_count must be a positive exact integer")
    # 0.875 is exactly 7/8 in the frozen configuration.  Integer arithmetic
    # avoids a platform-dependent boundary when view_count is large.
    return (7 * view_count + 7) // 8


def _identity_payload_antigravity(
    dataset: str,
    raw_length: int,
    train_indices: tuple[int, ...],
    excluded_train_indices: tuple[int, ...],
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "excluded_train_indices": list(excluded_train_indices),
        "mask_seed": FROZEN_CONFIG.mask_seed,
        "raw_length": raw_length,
        "train_indices": list(train_indices),
    }


def canonical_mask_identity_sha256_antigravity(
    dataset: str,
    raw_length: int,
    train_indices: Sequence[int],
    excluded_train_indices: Sequence[int],
) -> str:
    """Return the exact SHA256 identity for one validated omission mask."""
    dataset, raw_length, train = _validated_inputs_antigravity(dataset, raw_length, train_indices)
    excluded = tuple(excluded_train_indices)
    if isinstance(excluded_train_indices, (str, bytes)) or any(type(item) is not int for item in excluded):
        raise ValueError("excluded_train_indices must contain exact integers")
    if excluded != tuple(sorted(set(excluded))) or not excluded:
        raise ValueError("excluded_train_indices must be sorted, unique, and non-empty")
    if any(item not in train for item in excluded):
        raise ValueError("excluded_train_indices must be drawn from train_indices")
    if len(train) - len(excluded) < FROZEN_CONFIG.minimum_train_rows:
        raise ValueError("omission would retain fewer than four TRAIN rows")
    return sha256(_canonical_bytes_antigravity(_identity_payload_antigravity(dataset, raw_length, train, excluded))).hexdigest()


# A short alias keeps callers from having to know whether the hash is called
# ``canonical`` or ``mask`` in an audit record.
mask_identity_sha256_antigravity = canonical_mask_identity_sha256_antigravity


def _counter_payload_antigravity(
    dataset: str,
    raw_length: int,
    train_indices: tuple[int, ...],
    view_counter: int,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "mask_seed": FROZEN_CONFIG.mask_seed,
        "raw_length": raw_length,
        "train_indices": list(train_indices),
        "view_counter": view_counter,
    }


def _counter_mask_antigravity(
    dataset: str,
    raw_length: int,
    train_indices: tuple[int, ...],
    omit_count: int,
    view_counter: int,
) -> tuple[int, ...]:
    """Derive one candidate subset from a counter-specific SHA ordering."""
    base = _canonical_bytes_antigravity(_counter_payload_antigravity(dataset, raw_length, train_indices, view_counter))
    ranked = sorted(
        train_indices,
        key=lambda index: sha256(base + b"\x00" + str(index).encode("ascii")).digest(),
    )
    return tuple(sorted(ranked[:omit_count]))


def generate_omission_masks_antigravity(
    dataset: str,
    raw_length: int,
    train_indices: tuple[int, ...],
) -> tuple[OmissionMaskAntigravity, ...]:
    """Generate all small-group or sixteen bounded deterministic masks.

    For groups with at most sixteen possible combinations, every combination
    is ranked by its canonical identity.  Larger groups use monotonically
    increasing counter payloads to derive unique combinations; those entries
    are then ordered by their canonical identity before assigning view indices.
    """
    dataset, raw_length, train = _validated_inputs_antigravity(dataset, raw_length, train_indices)
    omit_count = omission_count_antigravity(len(train))
    possible_count = math.comb(len(train), omit_count)
    target = min(FROZEN_CONFIG.maximum_stability_views, possible_count)

    candidates: dict[tuple[int, ...], tuple[str, int]] = {}
    if possible_count <= FROZEN_CONFIG.maximum_stability_views:
        for excluded in combinations(train, omit_count):
            identity = canonical_mask_identity_sha256_antigravity(dataset, raw_length, train, excluded)
            candidates[excluded] = (identity, -1)
    else:
        # SHA256 ordering of a counter-specific canonical payload gives a
        # stable pseudo-random subset without depending on Python's hash seed.
        counter = 0
        # The birthday bound for 256-bit identities makes this loop tiny; the
        # explicit guard prevents a malformed implementation from spinning.
        while len(candidates) < target and counter < target * 128:
            excluded = _counter_mask_antigravity(dataset, raw_length, train, omit_count, counter)
            if excluded not in candidates:
                identity = canonical_mask_identity_sha256_antigravity(dataset, raw_length, train, excluded)
                candidates[excluded] = (identity, counter)
            counter += 1
        if len(candidates) != target:
            raise RuntimeError("deterministic omission mask generation did not reach target")

    ordered = sorted(candidates.items(), key=lambda item: (item[1][0], item[0]))
    return tuple(
        OmissionMaskAntigravity(view_index, excluded, identity)
        for view_index, (excluded, (identity, _counter)) in enumerate(ordered)
    )


__all__ = [
    "canonical_mask_identity_sha256_antigravity",
    "mask_identity_sha256_antigravity",
    "omission_count_antigravity",
    "required_survival_count_antigravity",
    "generate_omission_masks_antigravity",
]
