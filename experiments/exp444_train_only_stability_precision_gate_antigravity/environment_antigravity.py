"""Exact provenance evidence for the isolated Exp444 environment."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
from types import MappingProxyType

from .config_antigravity import FROZEN_CONFIG, cpu_worker_count_antigravity, inner_thread_environment_names_antigravity


_PACKAGE_NAMES = ("mlx", "numba", "numpy", "pip")


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
    inner_thread_env: Mapping[str, str]
    blas_thread_pools: tuple[Mapping[str, object], ...]
    acceleration_decision: str
    acceleration_reason: str
    pytorch_version: str | None = None
    mlx_version: str | None = None
    numba_version: str | None = None
    mps_version: str | None = None
    mps_available: bool | None = None

    def __post_init__(self) -> None:
        if self.python_version != FROZEN_CONFIG.python_version:
            raise ValueError("Exp444 Python version differs from the frozen contract")
        if not self.platform or not self.machine:
            raise ValueError("platform evidence is incomplete")
        if type(self.logical_cpu_count) is not int or self.logical_cpu_count != FROZEN_CONFIG.logical_cpu_count:
            raise ValueError("logical CPU count differs from the frozen contract")
        if type(self.outer_workers) is not int or self.outer_workers != cpu_worker_count_antigravity(self.logical_cpu_count):
            raise ValueError("outer worker count differs from the frozen contract")
        if type(self.inner_threads) is not int or self.inner_threads != FROZEN_CONFIG.inner_threads:
            raise ValueError("inner numerical thread count differs from the frozen contract")
        if not isinstance(self.package_versions, Mapping):
            raise ValueError("package versions must be a mapping")
        packages = dict(self.package_versions)
        if not set(_PACKAGE_NAMES) <= set(packages):
            raise ValueError("package provenance is incomplete")
        for key, value in packages.items():
            if type(key) is not str or (value is not None and type(value) is not str):
                raise ValueError("package provenance has the wrong type")
        if type(self.lock_sha256) is not str or len(self.lock_sha256) != 64 or any(char not in "0123456789abcdef" for char in self.lock_sha256) or self.lock_sha256 != FROZEN_CONFIG.environment_lock_sha256:
            raise ValueError("lock SHA256 is invalid")
        if not isinstance(self.inner_thread_env, Mapping):
            raise ValueError("inner-thread environment evidence must be a mapping")
        thread_env = dict(self.inner_thread_env)
        expected_names = set(inner_thread_environment_names_antigravity())
        if set(thread_env) != expected_names or any(value != "1" for value in thread_env.values()):
            raise ValueError("all seven inner-thread controls must be exactly 1")
        if type(self.blas_thread_pools) is not tuple:
            raise ValueError("BLAS thread-pool evidence must be a tuple")
        for pool in self.blas_thread_pools:
            if not isinstance(pool, Mapping) or type(pool.get("num_threads")) is not int or pool["num_threads"] != 1:
                raise ValueError("every discovered numerical thread pool must use one thread")
        for name in ("pytorch_version", "mlx_version", "numba_version", "mps_version"):
            value = getattr(self, name)
            if value is not None and type(value) is not str:
                raise ValueError(f"{name} must be a string or None")
        if self.mps_available is not None and type(self.mps_available) is not bool:
            raise ValueError("mps_available must be a bool or None")
        if self.acceleration_decision != FROZEN_CONFIG.acceleration_decision:
            raise ValueError("Exp444 acceleration decision is invalid")
        if not self.acceleration_reason:
            raise ValueError("acceleration reason is required")
        object.__setattr__(self, "package_versions", MappingProxyType(packages))
        object.__setattr__(self, "inner_thread_env", MappingProxyType(thread_env))
        object.__setattr__(self, "blas_thread_pools", tuple(MappingProxyType(dict(pool)) for pool in self.blas_thread_pools))

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
            "inner_thread_env": dict(self.inner_thread_env),
            "blas_thread_pools": [dict(pool) for pool in self.blas_thread_pools],
            "acceleration_decision": self.acceleration_decision,
            "acceleration_reason": self.acceleration_reason,
            "pytorch_version": self.pytorch_version,
            "mlx_version": self.mlx_version,
            "numba_version": self.numba_version,
            "mps_version": self.mps_version,
            "mps_available": self.mps_available,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "EnvironmentEvidenceAntigravity":
        expected = set(cls.__dataclass_fields__)
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise ValueError("environment evidence payload keys are invalid")
        if not isinstance(payload["package_versions"], Mapping) or not isinstance(payload["inner_thread_env"], Mapping) or not isinstance(payload["blas_thread_pools"], list):
            raise ValueError("environment evidence nested payload is invalid")
        pools: list[Mapping[str, object]] = []
        for pool in payload["blas_thread_pools"]:
            if not isinstance(pool, Mapping):
                raise ValueError("BLAS thread-pool payload is invalid")
            pools.append(dict(pool))
        return cls(
            payload["python_version"], payload["platform"], payload["machine"],
            payload["logical_cpu_count"], payload["outer_workers"], payload["inner_threads"],
            dict(payload["package_versions"]), payload["lock_sha256"], dict(payload["inner_thread_env"]), tuple(pools),
            payload["acceleration_decision"], payload["acceleration_reason"],
            payload["pytorch_version"], payload["mlx_version"], payload["numba_version"], payload["mps_version"], payload["mps_available"],
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


def _inner_thread_env_antigravity() -> Mapping[str, str]:
    names = inner_thread_environment_names_antigravity()
    values = {name: os.environ.get(name) for name in names}
    if any(value != "1" for value in values.values()):
        wrong = ", ".join(f"{name}={value!r}" for name, value in values.items() if value != "1")
        raise ValueError(f"Exp444 requires one-thread environment controls: {wrong}")
    return values  # type: ignore[return-value]


def _optional_acceleration_evidence_antigravity() -> tuple[str | None, str | None, str | None, bool | None]:
    torch_version = _version("torch")
    mps_available: bool | None = None
    if torch_version is not None:
        try:
            import torch  # type: ignore[import-not-found]
        except Exception:
            mps_available = None
        else:
            torch_version = str(getattr(torch, "__version__", torch_version))
            backend = getattr(getattr(torch, "backends", None), "mps", None)
            if backend is None:
                mps_available = False
            else:
                try:
                    mps_available = bool(backend.is_available())
                except Exception:
                    mps_available = False
    mlx_version = _version("mlx")
    numba_version = _version("numba")
    return torch_version, mlx_version, numba_version, mps_available


def verify_environment_antigravity() -> EnvironmentEvidenceAntigravity:
    """Verify interpreter, exact lock, package pins, and seven thread controls."""
    thread_env = _inner_thread_env_antigravity()
    python_version = platform.python_version()
    if python_version != FROZEN_CONFIG.python_version:
        raise ValueError(f"Exp444 requires Python {FROZEN_CONFIG.python_version}, got {python_version}")
    logical = os.cpu_count()
    if logical != FROZEN_CONFIG.logical_cpu_count:
        raise ValueError(f"Exp444 requires {FROZEN_CONFIG.logical_cpu_count} logical CPUs, got {logical}")
    lock_path = Path(FROZEN_CONFIG.environment_lock_path)
    if not lock_path.is_file() or _digest(lock_path) != FROZEN_CONFIG.environment_lock_sha256:
        raise ValueError("Exp444 environment lock SHA256 differs from the frozen contract")
    package_versions = {name: _version(name) for name in _PACKAGE_NAMES}
    expected = {"mlx": FROZEN_CONFIG.mlx_version, "numba": FROZEN_CONFIG.numba_version, "numpy": FROZEN_CONFIG.numpy_version}
    for name, version in expected.items():
        if package_versions.get(name) != version:
            raise ValueError(f"{name} version differs from the frozen contract")
    pools = _thread_pools_antigravity()
    for pool in pools:
        if type(pool.get("num_threads")) is not int or pool["num_threads"] != FROZEN_CONFIG.inner_threads:
            raise ValueError("a discovered numerical thread pool exceeds the one-thread contract")
    torch_version, mlx_version, numba_version, mps_available = _optional_acceleration_evidence_antigravity()
    return EnvironmentEvidenceAntigravity(
        python_version=python_version,
        platform=platform.platform(),
        machine=platform.machine(),
        logical_cpu_count=logical,
        outer_workers=cpu_worker_count_antigravity(logical),
        inner_threads=FROZEN_CONFIG.inner_threads,
        package_versions=package_versions,
        lock_sha256=FROZEN_CONFIG.environment_lock_sha256,
        inner_thread_env=thread_env,
        blas_thread_pools=pools,
        acceleration_decision=FROZEN_CONFIG.acceleration_decision,
        acceleration_reason="MLX/Numba exact-stage parity and speed gate is evaluated before any full run",
        pytorch_version=torch_version,
        mlx_version=mlx_version,
        numba_version=numba_version,
        mps_version=torch_version if mps_available else None,
        mps_available=mps_available,
    )


__all__ = ["EnvironmentEvidenceAntigravity", "verify_environment_antigravity"]
