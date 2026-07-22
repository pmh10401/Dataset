from __future__ import annotations

import importlib.metadata

import torch


def dependency_report() -> dict[str, object]:
    """Report installed capabilities without reading labels or experiment rows."""

    return {
        "kymatio": importlib.metadata.version("kymatio"),
        "iisignature": importlib.metadata.version("iisignature"),
        "torch": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
    }
