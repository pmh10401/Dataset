"""Exact provenance evidence for the isolated Exp443 CPU environment."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
from types import MappingProxyType

from .config_antigravity import FROZEN_CONFIG, cpu_worker_count_antigravity


_PACKAGE_NAMES = ("aeon", "numpy", "scikit-learn", "numba", "scipy", "pandas", "threadpoolctl")


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True)
class EnvironmentEvidenceAntigravity:
    python_version: str
    platform: str
    machine: str
    logical_cpu_count: int
    outer_workers: int
    inner_threads: int
    package_versions: Mapping[str, str | None]
    lock_sha256: str
    blas_thread_pools: tuple[Mapping[str, object], ...]
    acceleration_decision: str
    acceleration_reason: str
    pytorch_version: str | None = None
    mlx_version: str | None = None
    mps_version: str | None = None

    def __post_init__(self) -> None:
        if self.python_version != FROZEN_CONFIG.python_version:
            raise ValueError("Exp443 Python version differs from the frozen contract")
        if not self.machine or not self.platform:
            raise ValueError("platform evidence is incomplete")
        if type(self.logical_cpu_count) is not int or self.logical_cpu_count != FROZEN_CONFIG.logical_cpu_count:
            raise ValueError("logical CPU count differs from the frozen contract")
        if type(self.outer_workers) is not int or self.outer_workers != cpu_worker_count_antigravity(self.logical_cpu_count):
            raise ValueError("outer worker count differs from the frozen contract")
        if type(self.inner_threads) is not int or self.inner_threads != FROZEN_CONFIG.inner_threads:
            raise ValueError("inner numerical thread count differs from the frozen contract")
        if type(self.package_versions) is not MappingProxyType and not isinstance(self.package_versions, Mapping):
            raise ValueError("package versions must be a mapping")
        packages = dict(self.package_versions)
        required = set(_PACKAGE_NAMES)
        if not required <= set(packages):
            raise ValueError("package provenance is incomplete")
        for key, value in packages.items():
            if type(key) is not str or (value is not None and type(value) is not str):
                raise ValueError("package provenance has the wrong type")
        if type(self.lock_sha256) is not str or len(self.lock_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.lock_sha256) or self.lock_sha256 != FROZEN_CONFIG.environment_lock_sha256:
            raise ValueError("lock SHA256 is invalid")
        if type(self.blas_thread_pools) is not tuple:
            raise ValueError("BLAS thread-pool evidence must be a tuple")
        if self.acceleration_decision != "inapplicable_official_cpu_transformer":
            raise ValueError("Exp443 acceleration decision is invalid")
        if self.acceleration_reason != "official_aeon_minirocket_cpu_transformer_no_mlx_rewrite":
            raise ValueError("Exp443 acceleration reason is invalid")
        object.__setattr__(self, "package_versions", MappingProxyType(packages))

    def to_payload(self) -> dict[str, object]:
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "machine": self.machine,
            "logical_cpu_count": self.logical_cpu_count,
            "outer_workers": self.outer_workers,
            "inner_threads": self.inner_threads,
            "package_versions": dict(self.package_versions),
            "lock_sha256": self.lock_sha256,
            "blas_thread_pools": [dict(pool) for pool in self.blas_thread_pools],
            "acceleration_decision": self.acceleration_decision,
            "acceleration_reason": self.acceleration_reason,
            "pytorch_version": self.pytorch_version,
            "mlx_version": self.mlx_version,
            "mps_version": self.mps_version,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "EnvironmentEvidenceAntigravity":
        expected = set(cls.__dataclass_fields__)
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("environment evidence payload keys are invalid")
        if not isinstance(payload["package_versions"], Mapping) or not isinstance(payload["blas_thread_pools"], list):
            raise ValueError("environment evidence nested payload is invalid")
        pools: list[Mapping[str, object]] = []
        for pool in payload["blas_thread_pools"]:
            if not isinstance(pool, Mapping):
                raise ValueError("BLAS thread-pool payload is invalid")
            pools.append(dict(pool))
        return cls(
            payload["python_version"], payload["platform"], payload["machine"],
            payload["logical_cpu_count"], payload["outer_workers"], payload["inner_threads"],
            dict(payload["package_versions"]), payload["lock_sha256"], tuple(pools),
            payload["acceleration_decision"], payload["acceleration_reason"],
            payload["pytorch_version"], payload["mlx_version"], payload["mps_version"],
        )


def _thread_pools_antigravity() -> tuple[Mapping[str, object], ...]:
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return ()
    pools: list[Mapping[str, object]] = []
    for item in threadpool_info():
        pools.append({
            "user_api": item.get("user_api"),
            "internal_api": item.get("internal_api"),
            "num_threads": item.get("num_threads"),
            "prefix": item.get("prefix"),
        })
    return tuple(pools)


def verify_environment_antigravity() -> EnvironmentEvidenceAntigravity:
    """Verify interpreter, package pins, lock identity, CPU topology, and CPU-only routing."""
    python_version = platform.python_version()
    if python_version != FROZEN_CONFIG.python_version:
        raise ValueError(f"Exp443 requires Python {FROZEN_CONFIG.python_version}, got {python_version}")
    logical = os.cpu_count()
    if logical != FROZEN_CONFIG.logical_cpu_count:
        raise ValueError(f"Exp443 requires {FROZEN_CONFIG.logical_cpu_count} logical CPUs, got {logical}")
    lock_path = Path(FROZEN_CONFIG.environment_lock_path)
    if not lock_path.is_file() or _digest(lock_path) != FROZEN_CONFIG.environment_lock_sha256:
        raise ValueError("Exp443 environment lock SHA256 differs from the frozen contract")
    package_versions = {name: _version(name) for name in _PACKAGE_NAMES}
    expected_versions = {
        "aeon": FROZEN_CONFIG.aeon_version,
        "numpy": FROZEN_CONFIG.numpy_version,
        "scikit-learn": FROZEN_CONFIG.scikit_learn_version,
    }
    for name, expected in expected_versions.items():
        if package_versions.get(name) != expected:
            raise ValueError(f"{name} version differs from the frozen contract")
    return EnvironmentEvidenceAntigravity(
        python_version=python_version,
        platform=platform.platform(),
        machine=platform.machine(),
        logical_cpu_count=logical,
        outer_workers=cpu_worker_count_antigravity(logical),
        inner_threads=FROZEN_CONFIG.inner_threads,
        package_versions=package_versions,
        lock_sha256=FROZEN_CONFIG.environment_lock_sha256,
        blas_thread_pools=_thread_pools_antigravity(),
        acceleration_decision=FROZEN_CONFIG.acceleration_decision,
        acceleration_reason="official_aeon_minirocket_cpu_transformer_no_mlx_rewrite",
        pytorch_version=_version("torch"),
        mlx_version=_version("mlx"),
        mps_version=None,
    )


__all__ = ["EnvironmentEvidenceAntigravity", "verify_environment_antigravity"]
