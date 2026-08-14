"""Independent scorer-free verification of an already-published Exp449 proposal."""

from __future__ import annotations

from pathlib import Path

from .contracts_antigravity import Exp449Context, Exp449ProposalManifest
from .validation_antigravity import verify_exp449_proposal_root_antigravity


def verify_exp449_proposal_antigravity(
    root: Path, *, context: Exp449Context, catalog: dict[str, object]
) -> Exp449ProposalManifest:
    """Recompute every manifest and typed-shard binding without scoring or labels."""
    return verify_exp449_proposal_root_antigravity(root, context=context, catalog=catalog)


__all__ = ["verify_exp449_proposal_antigravity"]
