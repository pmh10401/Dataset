"""Deterministic outside-Hard global candidate budget for Exp440."""
from __future__ import annotations
from collections.abc import Iterable
from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import ProvisionalCandidateAntigravity,RetainedCandidateAntigravity,validate_retained_candidates_antigravity


def select_global_candidates_antigravity(candidates:Iterable[ProvisionalCandidateAntigravity],hard_coordinates:frozenset[tuple[str,int]],*,budget:int=100)->tuple[RetainedCandidateAntigravity,...]:
    if budget!=FROZEN_CONFIG.outside_hard_budget:raise ValueError("Exp440 budget cannot be overridden")
    values=tuple(candidates)
    if any(type(x) is not ProvisionalCandidateAntigravity for x in values):raise ValueError("candidate type is invalid")
    coordinates=tuple(x.coordinate for x in values)
    if len(coordinates)!=len(set(coordinates)):raise ValueError("candidate coordinate is duplicated")
    outside=sorted((x for x in values if x.coordinate not in hard_coordinates),key=lambda x:x.ranking_key)
    retained=tuple(RetainedCandidateAntigravity.from_provisional(x,i+1) for i,x in enumerate(outside[:budget]))
    return validate_retained_candidates_antigravity(retained,hard_coordinates,budget=budget)


__all__=["select_global_candidates_antigravity"]
