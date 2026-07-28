from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from collections.abc import Sequence

from .config import FROZEN_CONFIG
from .contracts import IdentityFold, TrainRun


def group_crossfit_plan(
    runs: Sequence[TrainRun],
    *,
    seed: int,
) -> tuple[IdentityFold, ...]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, run in enumerate(runs):
        groups[str(run.values_sha256)].append(index)
    if len(groups) < FROZEN_CONFIG.minimum_identity_groups:
        return ()

    if len(groups) < FROZEN_CONFIG.five_fold_minimum_groups:
        query_groups = ((digest,) for digest in sorted(groups))
    else:
        ordered = sorted(
            groups,
            key=lambda digest: sha256(
                f"{int(seed)}:{digest}".encode("utf-8")
            ).digest(),
        )
        buckets = tuple(
            tuple(ordered[offset::5])
            for offset in range(5)
        )
        query_groups = (bucket for bucket in buckets if bucket)

    all_indices = frozenset(range(len(runs)))
    folds: list[IdentityFold] = []
    for query_identity in query_groups:
        query = tuple(
            sorted(
                index
                for digest in query_identity
                for index in groups[digest]
            )
        )
        fit = tuple(sorted(all_indices - frozenset(query)))
        folds.append(
            IdentityFold(
                fit_indices=fit,
                query_indices=query,
                query_identity_sha256=tuple(sorted(query_identity)),
            )
        )
    return tuple(folds)


__all__ = ["group_crossfit_plan"]
