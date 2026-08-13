"""Test-only probe launcher for Exp453 exact-runtime binding. Never opens labels."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


_REPOSITORY_ANTIGRAVITY = Path(__file__).resolve().parents[3]
_SCRATCH_ANTIGRAVITY = _REPOSITORY_ANTIGRAVITY / "scratch"
_EXP453_PACKAGE_PREFIX_ANTIGRAVITY = (
    "experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity"
)


def _imported_exp453_antigravity() -> list[str]:
    return sorted(
        name
        for name in sys.modules
        if name == _EXP453_PACKAGE_PREFIX_ANTIGRAVITY
        or name.startswith(_EXP453_PACKAGE_PREFIX_ANTIGRAVITY + ".")
    )


if os.environ.get("EXP453_PROBE_BIND_ANTIGRAVITY") == "1":
    if str(_SCRATCH_ANTIGRAVITY) not in sys.path:
        sys.path.insert(0, str(_SCRATCH_ANTIGRAVITY))
    from exp453_runtime_binding_antigravity import ensure_fixed_runtime_antigravity

    ensure_fixed_runtime_antigravity(Path(__file__), tuple(sys.argv[1:]))


print(
    json.dumps(
        {
            "executable": sys.executable,
            "prefix": sys.prefix,
            "argv": sys.argv,
            "reexec_marker": os.environ.get("EXP453_FIXED_RUNTIME_REEXEC_ANTIGRAVITY"),
            "exp453_imported": _imported_exp453_antigravity(),
        },
        sort_keys=True,
    )
)
