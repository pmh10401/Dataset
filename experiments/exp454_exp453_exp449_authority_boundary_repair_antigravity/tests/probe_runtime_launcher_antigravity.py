"""Test-only probe launcher for Exp454 exact-runtime binding. Never opens labels."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[3]
_SCRATCH_ANTIGRAVITY = _REPOSITORY_ANTIGRAVITY / "scratch"
_EXP454_PACKAGE_PREFIX_ANTIGRAVITY = (
    "experiments.exp454_exp453_exp449_authority_boundary_repair_antigravity"
)


def _imported_exp454_antigravity() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if name == _EXP454_PACKAGE_PREFIX_ANTIGRAVITY
        or name.startswith(_EXP454_PACKAGE_PREFIX_ANTIGRAVITY + ".")
    )


if os.environ.get("EXP454_PROBE_BIND_ANTIGRAVITY") == "1":
    if str(_SCRATCH_ANTIGRAVITY) not in sys.path:
        sys.path.insert(0, str(_SCRATCH_ANTIGRAVITY))
    from exp454_runtime_binding_antigravity import ensure_fixed_runtime_antigravity

    probe = os.environ.get("EXP454_RUNTIME_METADATA_PROBE_ANTIGRAVITY")
    if probe:
        import importlib.metadata

        overrides = json.loads(probe)
        if type(overrides) is not dict or any(
            type(name) is not str or type(value) is not str for name, value in overrides.items()
        ):
            raise RuntimeError("Exp454 metadata probe is invalid")
        original = importlib.metadata.version

        def version_antigravity(name: str) -> str:
            if name in overrides:
                return overrides[name]
            return original(name)

        importlib.metadata.version = version_antigravity

    ensure_fixed_runtime_antigravity(Path(__file__), tuple(sys.argv[1:]))


print(
    json.dumps(
        {
            "executable": sys.executable,
            "prefix": sys.prefix,
            "argv": sys.argv,
            "reexec_marker": os.environ.get("EXP454_FIXED_RUNTIME_REEXEC_ANTIGRAVITY"),
            "exp454_imported": _imported_exp454_antigravity(),
        },
        sort_keys=True,
    )
)
