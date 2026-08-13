"""Exact immutable label-free records shared by future Exp451 phases."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass

import numpy as np


def _exact_string_antigravity(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field} must be an exact non-empty string")
    return value


def _exact_int_antigravity(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an exact integer at least {minimum}")
    return value


def _frozen_array_antigravity(value: object, field: str, *, dimensions: int) -> np.ndarray:
    if type(value) is not np.ndarray or value.ndim != dimensions or not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"{field} must be an exact {dimensions}D numeric ndarray")
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"{field} must be non-empty and finite")
    return np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True, slots=True)
class DatasetBundleAntigravity:
    dataset_order: int
    family: str
    dataset: str
    train_normal: np.ndarray
    test_values: np.ndarray
    target_length: int

    def __post_init__(self) -> None:
        _exact_int_antigravity(self.dataset_order, "dataset_order")
        _exact_string_antigravity(self.family, "family")
        _exact_string_antigravity(self.dataset, "dataset")
        _exact_int_antigravity(self.target_length, "target_length", minimum=1)
        train = _frozen_array_antigravity(self.train_normal, "train_normal", dimensions=2)
        test = _frozen_array_antigravity(self.test_values, "test_values", dimensions=2)
        if train.shape[1] != test.shape[1] or len(test) != self.target_length:
            raise ValueError("bundle dimensions or target length differ")
        object.__setattr__(self, "train_normal", train)
        object.__setattr__(self, "test_values", test)

    @property
    def dataset_identity(self) -> tuple[int, str, str]:
        return (self.dataset_order, self.family, self.dataset)


@dataclass(frozen=True, slots=True)
class CandidateScoresAntigravity:
    train_oof: np.ndarray
    test_scores: np.ndarray
    expected_train_length: int
    expected_test_length: int
    feature_dimension: int
    retained_shapelets: int
    seed: int

    def __post_init__(self) -> None:
        train = _frozen_array_antigravity(self.train_oof, "train_oof", dimensions=1)
        test = _frozen_array_antigravity(self.test_scores, "test_scores", dimensions=1)
        _exact_int_antigravity(self.expected_train_length, "expected_train_length", minimum=1)
        _exact_int_antigravity(self.expected_test_length, "expected_test_length", minimum=1)
        _exact_int_antigravity(self.feature_dimension, "feature_dimension", minimum=1)
        _exact_int_antigravity(self.retained_shapelets, "retained_shapelets", minimum=1)
        _exact_int_antigravity(self.seed, "seed")
        if self.feature_dimension != 3 * self.retained_shapelets:
            raise ValueError("feature_dimension must equal three times retained_shapelets")
        if len(train) != self.expected_train_length or len(test) != self.expected_test_length:
            raise ValueError("score lengths differ from their dataset authority")
        object.__setattr__(self, "train_oof", train)
        object.__setattr__(self, "test_scores", test)


def _canonical_bytes_antigravity(value: dict[str, object]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"


@dataclass(frozen=True, slots=True)
class FrozenPredictionAuthorityAntigravity:
    reviewed_commit: str
    dataset_identities: tuple[tuple[int, str, str], ...]
    target_lengths: tuple[int, ...]
    candidate_indices: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if type(self.reviewed_commit) is not str or not re.fullmatch(r"[0-9a-f]{40}", self.reviewed_commit):
            raise ValueError("reviewed_commit must be a lowercase commit SHA")
        if type(self.dataset_identities) is not tuple or type(self.target_lengths) is not tuple or type(self.candidate_indices) is not tuple:
            raise ValueError("prediction authority collections must be exact tuples")
        if not self.dataset_identities or len(self.dataset_identities) != len(self.target_lengths) or len(self.dataset_identities) != len(self.candidate_indices):
            raise ValueError("prediction authority collection lengths differ")
        identities: set[tuple[int, str, str]] = set()
        for order, identity in enumerate(self.dataset_identities):
            if type(identity) is not tuple or len(identity) != 3:
                raise ValueError("dataset identity is invalid")
            dataset_order, family, dataset = identity
            _exact_int_antigravity(dataset_order, "dataset_order")
            _exact_string_antigravity(family, "family")
            _exact_string_antigravity(dataset, "dataset")
            if dataset_order != order or identity in identities:
                raise ValueError("dataset identities must be unique and ordered")
            identities.add(identity)
        for target_length, indices in zip(self.target_lengths, self.candidate_indices, strict=True):
            _exact_int_antigravity(target_length, "target_length", minimum=1)
            if type(indices) is not tuple:
                raise ValueError("candidate indices must be an exact tuple")
            previous = -1
            for index in indices:
                _exact_int_antigravity(index, "candidate index")
                if index >= target_length or index <= previous:
                    raise ValueError("candidate indices must be sorted, unique, and in range")
                previous = index

    def _payload_without_hash_antigravity(self) -> dict[str, object]:
        return {
            "schema": "exp451.frozen_prediction_authority.v1",
            "reviewed_commit": self.reviewed_commit,
            "dataset_identities": [list(identity) for identity in self.dataset_identities],
            "target_lengths": list(self.target_lengths),
            "candidate_indices": [list(indices) for indices in self.candidate_indices],
        }

    @property
    def canonical_bytes_antigravity(self) -> bytes:
        return _canonical_bytes_antigravity(self._payload_without_hash_antigravity())

    @property
    def canonical_sha256_antigravity(self) -> str:
        return hashlib.sha256(self.canonical_bytes_antigravity).hexdigest()

    def to_payload_antigravity(self) -> dict[str, object]:
        return {**self._payload_without_hash_antigravity(), "canonical_sha256": self.canonical_sha256_antigravity}

    @classmethod
    def from_payload_antigravity(cls, value: object) -> "FrozenPredictionAuthorityAntigravity":
        if type(value) is not dict or set(value) != {"schema", "reviewed_commit", "dataset_identities", "target_lengths", "candidate_indices", "canonical_sha256"}:
            raise ValueError("frozen prediction payload keys differ")
        if value["schema"] != "exp451.frozen_prediction_authority.v1" or type(value["dataset_identities"]) is not list or type(value["target_lengths"]) is not list or type(value["candidate_indices"]) is not list:
            raise ValueError("frozen prediction payload is malformed")
        identities = tuple(tuple(item) if type(item) is list else item for item in value["dataset_identities"])
        target_lengths = tuple(value["target_lengths"])
        candidates = tuple(tuple(item) if type(item) is list else item for item in value["candidate_indices"])
        authority = cls(value["reviewed_commit"], identities, target_lengths, candidates)
        if type(value["canonical_sha256"]) is not str or value["canonical_sha256"] != authority.canonical_sha256_antigravity:
            raise ValueError("frozen prediction canonical SHA differs")
        return authority
