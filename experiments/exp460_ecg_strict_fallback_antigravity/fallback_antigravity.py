"""Task 3 strict fallback routing for Exp460."""

from __future__ import annotations

from .contracts_antigravity import (
    Exp459FallbackAuthorityAntigravity,
    _task3_snapshot_from_authority_antigravity,
)


def strict_route_antigravity(authority: Exp459FallbackAuthorityAntigravity) -> str:
    """Return candidate only when all ten records pass, else baseline."""

    snapshot = _task3_snapshot_from_authority_antigravity(authority)
    all_records_passed = snapshot["all_records_passed"]
    assert type(all_records_passed) is bool
    return "candidate" if all_records_passed else "baseline"


__all__ = ("strict_route_antigravity",)
