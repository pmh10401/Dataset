"""Typed, label-free durable contracts for Exp473 D-prime scores."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from types import MappingProxyType

import numpy as np


def canonical_json_bytes_exp473_antigravity(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Exp473 artifact data must be finite canonical JSON") from error


@dataclass(frozen=True, slots=True)
class Exp473ScoreShardAntigravity:
    dataset: str
    test_identity: Mapping[str, object]
    scores: np.ndarray
    alerts: frozenset[int]

    def __post_init__(self) -> None:
        if type(self.dataset) is not str or not self.dataset:
            raise ValueError("Exp473 score dataset is invalid")
        try:
            identity = json.loads(
                canonical_json_bytes_exp473_antigravity(dict(self.test_identity))
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("Exp473 TEST identity is invalid") from error
        if (
            not isinstance(identity, dict)
            or type(identity.get("count")) is not int
            or identity["count"] < 1
            or identity.get("indices") != list(range(identity["count"]))
        ):
            raise ValueError("Exp473 TEST identity differs")
        scores = self.scores
        if (
            type(scores) is not np.ndarray
            or scores.dtype != np.dtype("float64")
            or scores.ndim != 1
            or scores.size != identity["count"]
            or not np.isfinite(scores).all()
        ):
            raise ValueError("Exp473 score vector differs")
        alerts = frozenset(self.alerts)
        if any(type(index) is not int for index in alerts) or alerts != frozenset(
            np.flatnonzero(scores >= 0.5).tolist()
        ):
            raise ValueError("Exp473 alert threshold differs")
        copied = np.array(scores, dtype=np.float64, copy=True)
        copied.setflags(write=False)
        object.__setattr__(self, "test_identity", MappingProxyType(identity))
        object.__setattr__(self, "scores", copied)
        object.__setattr__(self, "alerts", alerts)


__all__ = [
    "Exp473ScoreShardAntigravity",
    "canonical_json_bytes_exp473_antigravity",
]
