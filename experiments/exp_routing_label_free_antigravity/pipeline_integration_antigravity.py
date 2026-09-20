"""End-to-end pipeline integration for label-free routing.

Every dataset receives one method from the qualified Exp476 router. The
dataset name is used only to load its TRAIN rows from the database.
"""

from __future__ import annotations

from .official_router_antigravity import route_official_antigravity
_ALL_METHODS_ANTIGRAVITY = (
    "minirocket_local_gap",
    "multirocket_local_gap",
    "hydra_local_gap",
    "hydra_multirocket_local_gap",
)


def get_candidates_for_dataset_antigravity(dataset_name: str) -> tuple[str, ...]:
    """Return the single Exp476-qualified method for any dataset family."""
    return (route_official_antigravity(dataset_name),)
