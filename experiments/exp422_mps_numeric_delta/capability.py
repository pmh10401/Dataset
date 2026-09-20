"""Host-local PyTorch MPS and MLX provenance for the Exp422 audit."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import platform


_UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class DeviceCapability:
    """Capability evidence, never an instruction to select an accelerator."""

    torch_version: str
    mps_built: bool
    mps_available: bool
    device_name: str
    mlx_version: str
    mlx_metal_version: str
    mlx_availability: str
    fallback_reason: str | None

    def __post_init__(self) -> None:
        for name in ("torch_version", "device_name", "mlx_version", "mlx_metal_version", "mlx_availability"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        if type(self.mps_built) is not bool or type(self.mps_available) is not bool:
            raise ValueError("MPS capability flags must be bools")
        if self.mps_available and not self.mps_built:
            raise ValueError("available MPS requires a built backend")
        if self.mlx_availability not in {"available", "unavailable"}:
            raise ValueError("mlx_availability must be available or unavailable")
        if self.mps_available:
            if self.fallback_reason is not None:
                raise ValueError("available MPS cannot have a fallback reason")
        elif self.fallback_reason != "mps_unavailable":
            raise ValueError("unavailable MPS requires mps_unavailable")


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except Exception:
        return _UNAVAILABLE


def detect_capability() -> DeviceCapability:
    """Read runtime availability without making a tensor or opening a database."""
    try:
        import torch
    except Exception:
        torch_version, mps_built, mps_available = _UNAVAILABLE, False, False
    else:
        torch_version = str(getattr(torch, "__version__", _UNAVAILABLE))
        try:
            mps_built = bool(torch.backends.mps.is_built())
        except Exception:
            mps_built = False
        try:
            mps_available = bool(torch.backends.mps.is_available())
        except Exception:
            mps_available = False
    mlx_version, mlx_metal_version = _version("mlx"), _version("mlx-metal")
    mlx_available = mlx_version != _UNAVAILABLE and mlx_metal_version != _UNAVAILABLE
    return DeviceCapability(
        torch_version=torch_version,
        mps_built=mps_built,
        mps_available=mps_available,
        device_name=f"{platform.system() or 'unknown'} {platform.machine() or 'unknown'} {'MPS' if mps_available else 'CPU'}",
        mlx_version=mlx_version if mlx_available else _UNAVAILABLE,
        mlx_metal_version=mlx_metal_version if mlx_available else _UNAVAILABLE,
        mlx_availability="available" if mlx_available else "unavailable",
        fallback_reason=None if mps_available else "mps_unavailable",
    )
