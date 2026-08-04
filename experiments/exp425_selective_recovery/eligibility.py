"""Deterministic TRAIN-only identity eligibility preflight for Exp425."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from hashlib import sha256

from experiments.exp417_causal_multihorizon_shadow.contracts import ProcessedGroup, RawSeries
from experiments.exp417_causal_multihorizon_shadow.preprocessing import (
    canonical_waveform_sha256,
    identity_grouped_folds,
)

from .contracts import EligibilityAssessment, FoldIdentityCounts, GroupEligibilityEvidence


_EARLY_STOP_MODULUS = 10
_USABLE_PROCESSED_STATUSES = frozenset({"candidate_eligible", "diagnostic_only"})


def _identity_groups(series: Sequence[RawSeries]) -> dict[str, tuple[int, ...]]:
    """Return Exp417-canonical waveform identities and their sorted members."""
    members: dict[str, list[int]] = {}
    for item in sorted(series, key=lambda value: value.instance_index):
        members.setdefault(canonical_waveform_sha256(item), []).append(item.instance_index)
    return {identity: tuple(sorted(indices)) for identity, indices in members.items()}


def _train_identity_sha256(identity_groups: dict[str, tuple[int, ...]]) -> str:
    """Bind a group manifest to ordered waveform identities and member indices."""
    payload = "".join(
        f"{identity}:{member_index}\n"
        for identity in sorted(identity_groups)
        for member_index in identity_groups[identity]
    )
    return sha256(payload.encode("ascii")).hexdigest()


def _assigned_identities(
    identity_groups: dict[str, tuple[int, ...]]
) -> tuple[tuple[str, ...], ...]:
    """Reproduce Exp417's five-fold greedy allocation exactly."""
    assigned: list[list[str]] = [[] for _ in range(5)]
    sizes = [0] * 5
    for identity in sorted(identity_groups):
        target = min(range(5), key=lambda fold_index: (sizes[fold_index], fold_index))
        assigned[target].append(identity)
        sizes[target] += len(identity_groups[identity])
    return tuple(tuple(values) for values in assigned)


def _early_identity_counts(identities: Sequence[str]) -> tuple[int, int]:
    validation = sum(int(identity, 16) % _EARLY_STOP_MODULUS == 0 for identity in identities)
    return len(identities) - validation, validation


def _preflight_fold_counts(
    identity_groups: dict[str, tuple[int, ...]]
) -> tuple[FoldIdentityCounts, ...]:
    """Calculate identity cardinalities without calling Exp417's raising fold builder."""
    identities = tuple(sorted(identity_groups))
    assigned = _assigned_identities(identity_groups)
    result: list[FoldIdentityCounts] = []
    for validation in assigned:
        validation_set = set(validation)
        fit = tuple(identity for identity in identities if identity not in validation_set)
        early_fit, early_validation = _early_identity_counts(fit)
        result.append(
            FoldIdentityCounts(
                oof_fit=len(fit),
                oof_validation=len(validation),
                early_stop_fit=early_fit,
                early_stop_validation=early_validation,
            )
        )
    return tuple(result)


def _counts_from_exp417_folds(
    group: ProcessedGroup,
) -> tuple[FoldIdentityCounts, ...]:
    """Use frozen Exp417 folds once preflight establishes they cannot raise."""
    identity_by_index = {
        item.instance_index: canonical_waveform_sha256(item) for item in group.group.series
    }
    return tuple(
        FoldIdentityCounts(
            oof_fit=len({identity_by_index[index] for index in fold.fit_indices}),
            oof_validation=len({identity_by_index[index] for index in fold.validation_indices}),
            early_stop_fit=len({identity_by_index[index] for index in fold.early_stop_fit_indices}),
            early_stop_validation=len(
                {identity_by_index[index] for index in fold.early_stop_validation_indices}
            ),
        )
        for fold in identity_grouped_folds(group.group.series)
    )


def _splits_pass(
    folds: Sequence[FoldIdentityCounts], final_fit: int | None, final_validation: int | None
) -> bool:
    return bool(
        all(fold.early_stop_fit >= 5 and fold.early_stop_validation >= 5 for fold in folds)
        and (final_fit is None or final_fit >= 5)
        and (final_validation is None or final_validation >= 5)
    )


def evaluate_group_eligibility(
    dataset: str, group: ProcessedGroup, *, fold_contract_sha256: str
) -> GroupEligibilityEvidence:
    """Classify one already-preprocessed group using only TRAIN waveform identities."""
    if not isinstance(group, ProcessedGroup):
        raise ValueError("group must be a ProcessedGroup")
    identity_groups = _identity_groups(group.group.series)
    common = {
        "dataset": dataset,
        "length": group.group.length,
        "processed_status": group.status,
        "train_identity_sha256": _train_identity_sha256(identity_groups),
        "fold_contract_sha256": fold_contract_sha256,
        "unique_identity_count": len(identity_groups),
    }
    if group.status not in _USABLE_PROCESSED_STATUSES:
        return GroupEligibilityEvidence(
            **common,
            folds=(),
            final_fit=None,
            final_validation=None,
            status="preprocessing_abstained",
            reason=group.status,
        )

    folds = _preflight_fold_counts(identity_groups)
    final_fit: int | None = None
    final_validation: int | None = None
    if group.status == "candidate_eligible":
        final_fit, final_validation = _early_identity_counts(tuple(sorted(identity_groups)))

    oof_passes = all(
        fold.early_stop_fit >= 5 and fold.early_stop_validation >= 5 for fold in folds
    )
    if oof_passes:
        folds = _counts_from_exp417_folds(group)
    status = (
        "eligible"
        if _splits_pass(folds, final_fit, final_validation)
        else "oof_identity_insufficient"
    )
    return GroupEligibilityEvidence(
        **common,
        folds=folds,
        final_fit=final_fit,
        final_validation=final_validation,
        status=status,
        reason=None if status == "eligible" else "oof_identity_insufficient",
    )


def apply_group_eligibility(
    dataset: str,
    groups: Sequence[ProcessedGroup],
    *,
    fold_contract_sha256: str,
) -> EligibilityAssessment:
    """Return length-sorted groups, abstaining only OOF-identity-insufficient groups."""
    if isinstance(groups, (str, bytes)) or not isinstance(groups, Sequence):
        raise ValueError("groups must be a sequence")
    ordered = tuple(sorted(groups, key=lambda group: group.group.length))
    output: list[ProcessedGroup] = []
    evidence: dict[int, GroupEligibilityEvidence] = {}
    for group in ordered:
        item = evaluate_group_eligibility(
            dataset, group, fold_contract_sha256=fold_contract_sha256
        )
        output.append(
            group
            if item.status != "oof_identity_insufficient"
            else replace(group, status="insufficient_train_abstain", channels=None)
        )
        evidence[group.group.length] = item
    return EligibilityAssessment(groups=tuple(output), evidence=evidence)
