"""Frozen, label-free Exp472 configuration and score artifact contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType
from typing import Literal

import numpy as np

from experiments.exp450_resampling_image_ablation_antigravity.config_antigravity import (
    FROZEN_EXP450_CONFIG,
)


MLX_FALLBACK_REASON_ANTIGRAVITY = (
    "MLX is assessed and recorded as unsuitable for the exact irregular 1-D "
    "SciPy interpolation path. It becomes eligible only in a separate "
    "implementation after exact index parity, finite output, numerical "
    "tolerance, and at least 1.1x end-to-end speed are demonstrated."
)

_SHA256_ANTIGRAVITY = re.compile(r"^[0-9a-f]{64}$")


def _run_root_by_mode_antigravity() -> Mapping[str, Path]:
    """Resolve the single frozen root authority without an import-time cycle."""
    from .config_antigravity import (
        EXP472_OFFICIAL_ROOT_ANTIGRAVITY,
        EXP472_SMOKE_ROOT_ANTIGRAVITY,
    )

    return MappingProxyType(
        {
            "smoke": EXP472_SMOKE_ROOT_ANTIGRAVITY,
            "full": EXP472_OFFICIAL_ROOT_ANTIGRAVITY,
        }
    )


def _sha256_antigravity(value: object, field_name: str) -> str:
    if type(value) is not str or _SHA256_ANTIGRAVITY.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256")
    return value


def _nonempty_string_antigravity(value: object, field_name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _contained_relative_path_antigravity(value: object, field_name: str) -> str:
    path = _nonempty_string_antigravity(value, field_name)
    parts = path.split("/")
    if (
        path.startswith("/")
        or "\\" in path
        or "\x00" in path
        or any(part in {"", ".", ".."} for part in parts)
        or PurePosixPath(path).is_absolute()
    ):
        raise ValueError(f"{field_name} must be a contained relative path")
    return path


def _plain_json_antigravity(value: object, active: frozenset[int]) -> object:
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite float")
        return value
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise ValueError("cyclic mapping")
        nested_active = active | {identity}
        result: dict[str, object] = {}
        for key, nested in value.items():
            if type(key) is not str or key in result:
                raise ValueError("mapping keys must be unique strings")
            result[key] = _plain_json_antigravity(nested, nested_active)
        return result
    if isinstance(value, (tuple, list)):
        identity = id(value)
        if identity in active:
            raise ValueError("cyclic sequence")
        nested_active = active | {identity}
        return [
            _plain_json_antigravity(nested, nested_active) for nested in value
        ]
    raise ValueError("unsupported canonical JSON value")


def canonical_json_bytes_exp472_antigravity(value: object) -> bytes:
    """Encode one finite Exp472 record in stable UTF-8 canonical JSON."""
    try:
        plain = _plain_json_antigravity(value, frozenset())
        return (
            json.dumps(
                plain,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Exp472 artifact data must be finite canonical JSON"
        ) from exc


def _freeze_json_antigravity(value: object, field_name: str) -> object:
    try:
        plain = _plain_json_antigravity(value, frozenset())
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must contain finite canonical JSON values"
        ) from exc

    def freeze(nested: object) -> object:
        if isinstance(nested, dict):
            return MappingProxyType(
                {key: freeze(item) for key, item in sorted(nested.items())}
            )
        if isinstance(nested, list):
            return tuple(freeze(item) for item in nested)
        return nested

    return freeze(plain)


def _frozen_digest_mapping_antigravity(
    value: object,
    field_name: str,
    *,
    paths: bool = False,
) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field_name} must be a non-empty digest mapping")
    copied: dict[str, str] = {}
    for key, digest in value.items():
        name = (
            _contained_relative_path_antigravity(key, field_name)
            if paths
            else _nonempty_string_antigravity(key, field_name)
        )
        if name in copied:
            raise ValueError(f"{field_name} contains a duplicate identity")
        copied[name] = _sha256_antigravity(digest, f"{field_name}.{name}")
    return MappingProxyType(dict(sorted(copied.items())))


@dataclass(frozen=True, slots=True)
class Exp472ConditionAntigravity:
    condition_id: str = "E"
    downsampling: str = "polyphase"
    upsampling: str = "makima"
    image_mode: str = "area"

    def __post_init__(self) -> None:
        if (
            self.condition_id,
            self.downsampling,
            self.upsampling,
            self.image_mode,
        ) != ("E", "polyphase", "makima", "area"):
            raise ValueError("Exp472 condition differs from the approved design")


@dataclass(frozen=True, slots=True)
class Exp472BackendDecisionAntigravity:
    pytorch_available: bool
    pytorch_version: str
    scipy_available: bool
    scipy_version: str
    mlx_available: bool
    mlx_version: str
    mlx_metal_available: bool
    mlx_metal_version: str
    mps_available: bool
    tensor_backend: str
    logical_cpu_count: int
    cpu_workers: int
    inner_numerical_threads: int = 1
    mlx_fallback_reason: str = MLX_FALLBACK_REASON_ANTIGRAVITY

    def __post_init__(self) -> None:
        availability = (
            self.pytorch_available,
            self.scipy_available,
            self.mlx_available,
            self.mlx_metal_available,
            self.mps_available,
        )
        if any(type(value) is not bool for value in availability):
            raise ValueError("Exp472 backend availability must be boolean")
        for available, version, name in (
            (self.pytorch_available, self.pytorch_version, "PyTorch"),
            (self.scipy_available, self.scipy_version, "SciPy"),
            (self.mlx_available, self.mlx_version, "MLX"),
            (self.mlx_metal_available, self.mlx_metal_version, "mlx-metal"),
        ):
            if type(version) is not str or not version:
                raise ValueError(f"Exp472 {name} version is invalid")
            if available == (version == "unavailable"):
                raise ValueError(
                    f"Exp472 {name} availability and version disagree"
                )
        if self.mps_available and not self.pytorch_available:
            raise ValueError("Exp472 MPS requires PyTorch availability")
        if self.mlx_metal_available and not self.mlx_available:
            raise ValueError("Exp472 mlx-metal requires MLX availability")
        selected = "mps" if self.mps_available else "cpu"
        if self.tensor_backend != selected:
            raise ValueError(
                "Exp472 tensor backend must use MPS when available and CPU otherwise"
            )
        if type(self.logical_cpu_count) is not int or self.logical_cpu_count < 1:
            raise ValueError("Exp472 logical CPU count is invalid")
        if self.cpu_workers != max(1, self.logical_cpu_count - 1):
            raise ValueError("Exp472 CPU worker policy is invalid")
        if self.inner_numerical_threads != 1:
            raise ValueError("Exp472 inner numerical threads must remain one")
        if self.mlx_fallback_reason != MLX_FALLBACK_REASON_ANTIGRAVITY:
            raise ValueError("Exp472 MLX fallback reason differs from the approved design")

    def artifact_record(self) -> dict[str, object]:
        return {
            "pytorch_available": self.pytorch_available,
            "pytorch_version": self.pytorch_version,
            "scipy_available": self.scipy_available,
            "scipy_version": self.scipy_version,
            "mlx_available": self.mlx_available,
            "mlx_version": self.mlx_version,
            "mlx_metal_available": self.mlx_metal_available,
            "mlx_metal_version": self.mlx_metal_version,
            "mps_available": self.mps_available,
            "tensor_backend": self.tensor_backend,
            "logical_cpu_count": self.logical_cpu_count,
            "cpu_workers": self.cpu_workers,
            "inner_numerical_threads": self.inner_numerical_threads,
            "mlx_fallback_reason": self.mlx_fallback_reason,
        }


@dataclass(frozen=True, slots=True)
class Exp472RunContextAntigravity:
    """Complete backend- and root-bound identity for one Exp472 score run."""

    mode: Literal["smoke", "full"]
    run_root: Path
    database_sha256: str
    catalog_sha256: str
    code_closure: Mapping[str, str]
    backend: Exp472BackendDecisionAntigravity
    config_sha256: str

    def __post_init__(self) -> None:
        run_roots = _run_root_by_mode_antigravity()
        if self.mode not in run_roots:
            raise ValueError("Exp472 run mode is invalid")
        root = Path(self.run_root)
        if root != run_roots[self.mode]:
            raise ValueError("Exp472 run root differs from the frozen mode root")
        for field_name in (
            "database_sha256",
            "catalog_sha256",
            "config_sha256",
        ):
            _sha256_antigravity(getattr(self, field_name), field_name)
        closure = _frozen_digest_mapping_antigravity(
            self.code_closure, "code_closure", paths=True
        )
        if not isinstance(self.backend, Exp472BackendDecisionAntigravity):
            raise ValueError("Exp472 run backend is invalid")
        if not self.backend.scipy_available:
            raise ValueError("Exp472 run requires the canonical SciPy backend")
        object.__setattr__(self, "run_root", root)
        object.__setattr__(self, "code_closure", closure)

    def artifact_record(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "run_root": str(self.run_root),
            "database_sha256": self.database_sha256,
            "catalog_sha256": self.catalog_sha256,
            "code_closure": dict(self.code_closure),
            "backend": self.backend.artifact_record(),
            "config_sha256": self.config_sha256,
        }

    @property
    def sha256(self) -> str:
        return sha256(
            canonical_json_bytes_exp472_antigravity(self.artifact_record())
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class Exp472ScoreShardAntigravity:
    dataset: str
    test_identity: Mapping[str, object]
    scores: np.ndarray
    alerts: frozenset[int]
    upsampled_rows: int
    diagnostic: Mapping[str, object]

    def __post_init__(self) -> None:
        _nonempty_string_antigravity(self.dataset, "dataset")
        values = np.asarray(self.scores)
        if (
            values.dtype != np.dtype("float64")
            or values.ndim != 1
            or values.size < 1
            or not np.isfinite(values).all()
        ):
            raise ValueError("Exp472 score shard differs")
        copied_scores = np.array(values, dtype=np.float64, copy=True, order="C")
        copied_scores.flags.writeable = False
        alerts = frozenset(self.alerts)
        if any(
            type(index) is not int or index < 0 or index >= copied_scores.size
            for index in alerts
        ):
            raise ValueError("Exp472 score threshold differs")
        if frozenset(np.flatnonzero(copied_scores >= 0.5).tolist()) != alerts:
            raise ValueError("Exp472 score threshold differs")
        if (
            type(self.upsampled_rows) is not int
            or self.upsampled_rows < 0
        ):
            raise ValueError("Exp472 upsampled row count differs")
        test_identity = _freeze_json_antigravity(
            self.test_identity, "test_identity"
        )
        diagnostic = _freeze_json_antigravity(self.diagnostic, "diagnostic")
        if not isinstance(test_identity, Mapping) or not test_identity:
            raise ValueError("Exp472 test identity must be a non-empty mapping")
        if not isinstance(diagnostic, Mapping) or not diagnostic:
            raise ValueError("Exp472 diagnostic must be a non-empty mapping")
        object.__setattr__(self, "scores", copied_scores)
        object.__setattr__(self, "alerts", alerts)
        object.__setattr__(self, "test_identity", test_identity)
        object.__setattr__(self, "diagnostic", diagnostic)


@dataclass(frozen=True, slots=True)
class Exp472ScoreManifestAntigravity:
    """Terminal ordered score-shard authority for smoke or full execution."""

    mode: Literal["smoke", "full"]
    context_sha256: str
    dataset_order: tuple[str, ...]
    shard_sha256: Mapping[str, str]
    manifest_sha256: str

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "full"}:
            raise ValueError("Exp472 score manifest mode is invalid")
        _sha256_antigravity(self.context_sha256, "context_sha256")
        _sha256_antigravity(self.manifest_sha256, "manifest_sha256")
        if not isinstance(self.dataset_order, (tuple, list)):
            raise ValueError("Exp472 manifest requires complete unique coverage")
        datasets = tuple(
            _nonempty_string_antigravity(dataset, "dataset_order dataset")
            for dataset in self.dataset_order
        )
        expected_count = 23 if self.mode == "smoke" else 1117
        if len(datasets) != expected_count or len(set(datasets)) != expected_count:
            raise ValueError("Exp472 manifest requires complete unique coverage")
        if (
            self.mode == "smoke"
            and datasets != FROZEN_EXP450_CONFIG.smoke_datasets
        ):
            raise ValueError(
                "Exp472 manifest differs from the frozen Exp450 smoke inventory"
            )
        shards = _frozen_digest_mapping_antigravity(
            self.shard_sha256, "shard_sha256"
        )
        if set(shards) != set(datasets):
            raise ValueError("Exp472 manifest requires complete unique coverage")
        object.__setattr__(self, "dataset_order", datasets)
        object.__setattr__(self, "shard_sha256", shards)


__all__ = [
    "Exp472BackendDecisionAntigravity",
    "Exp472ConditionAntigravity",
    "Exp472RunContextAntigravity",
    "Exp472ScoreManifestAntigravity",
    "Exp472ScoreShardAntigravity",
    "MLX_FALLBACK_REASON_ANTIGRAVITY",
    "canonical_json_bytes_exp472_antigravity",
]
