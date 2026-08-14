"""Independent scorer-free verification of Exp449 proposal and smoke roots."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
import stat

import numpy as np

from .config_antigravity import FROZEN_EXP449_CONFIG, FROZEN_EXP449_POLICY_KEYS
from .contracts_antigravity import Exp449Context, Exp449ProposalManifest, SmokeComparison
from .validation_antigravity import (
    VerifiedExp449ProposalRootSnapshotAntigravity,
    canonical_json_bytes_antigravity,
    verify_exp449_proposal_root_antigravity,
    verify_exp449_proposal_root_snapshot_antigravity,
)


_SMOKE_ROOT_PAIRS_ANTIGRAVITY = (
    (
        FROZEN_EXP449_CONFIG.smoke_a_scientific_root.resolve(strict=False),
        FROZEN_EXP449_CONFIG.smoke_b_scientific_root.resolve(strict=False),
    ),
)
_AFTER_SMOKE_SNAPSHOTS_ANTIGRAVITY = None


def verify_exp449_proposal_antigravity(
    root: Path, *, context: Exp449Context, catalog: dict[str, object]
) -> Exp449ProposalManifest:
    """Recompute every manifest and typed-shard binding without scoring or labels."""
    return verify_exp449_proposal_root_antigravity(root, context=context, catalog=catalog)


def _safe_resolved_root_antigravity(root: Path) -> Path:
    candidate = Path(root)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Exp449 smoke root authority differs")
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        try:
            details = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(details.st_mode):
            raise ValueError("Exp449 smoke symlinked root is forbidden")
    return candidate.resolve(strict=False)


def _smoke_inputs_antigravity(authority: object) -> tuple[Exp449Context, dict[str, object], Exp449Context, dict[str, object]]:
    required = {"smoke_a_context", "smoke_a_catalog", "smoke_b_context", "smoke_b_catalog"}
    if not isinstance(authority, Mapping) or set(authority) != required:
        raise ValueError("Exp449 smoke comparison authority differs")
    context_a = authority["smoke_a_context"]
    catalog_a = authority["smoke_a_catalog"]
    context_b = authority["smoke_b_context"]
    catalog_b = authority["smoke_b_catalog"]
    if not isinstance(context_a, Exp449Context) or not isinstance(context_b, Exp449Context) or not isinstance(catalog_a, dict) or not isinstance(catalog_b, dict):
        raise ValueError("Exp449 smoke comparison authority differs")
    return context_a, catalog_a, context_b, catalog_b


def _require_independent_smoke_authority_antigravity(
    root_a: Path, root_b: Path, authority: object,
) -> tuple[Path, Path, Exp449Context, dict[str, object], Exp449Context, dict[str, object]]:
    resolved_a = _safe_resolved_root_antigravity(root_a)
    resolved_b = _safe_resolved_root_antigravity(root_b)
    if resolved_a == resolved_b:
        raise ValueError("Exp449 smoke roots must be distinct")
    if resolved_a in resolved_b.parents or resolved_b in resolved_a.parents:
        raise ValueError("Exp449 smoke roots must be independent")
    if (resolved_a, resolved_b) not in _SMOKE_ROOT_PAIRS_ANTIGRAVITY:
        raise ValueError("Exp449 smoke root authority differs")
    context_a, catalog_a, context_b, catalog_b = _smoke_inputs_antigravity(authority)
    if context_a.mode != "smoke" or context_b.mode != "smoke" or context_a.run_root != resolved_a or context_b.run_root != resolved_b:
        raise ValueError("Exp449 smoke root declaration differs")
    return resolved_a, resolved_b, context_a, catalog_a, context_b, catalog_b


def _require_shared_scientific_closure_antigravity(
    context_a: Exp449Context, catalog_a: dict[str, object], context_b: Exp449Context, catalog_b: dict[str, object],
) -> None:
    if canonical_json_bytes_antigravity(catalog_a) != canonical_json_bytes_antigravity(catalog_b):
        raise ValueError("Exp449 smoke catalog/raw identity differs")
    fields = (
        "experiment_id", "mode", "database_sha256", "catalog_sha256", "code_closure",
        "lineage_sha256", "codemap_sha256", "interpreter", "environment", "backend_record",
        "seeds_and_constants", "historical_parity_status",
    )
    if any(getattr(context_a, field) != getattr(context_b, field) for field in fields):
        raise ValueError("Exp449 smoke scientific closure differs")


def _require_independent_files_antigravity(
    snapshot_a: VerifiedExp449ProposalRootSnapshotAntigravity,
    snapshot_b: VerifiedExp449ProposalRootSnapshotAntigravity,
) -> None:
    files_a = (snapshot_a.manifest_file, *snapshot_a.checkpoint_files, *(shard.file for shard in snapshot_a.shards))
    files_b = (snapshot_b.manifest_file, *snapshot_b.checkpoint_files, *(shard.file for shard in snapshot_b.shards))
    identities_a: set[tuple[int, int]] = set()
    for file in files_a:
        if file.link_count != 1:
            raise ValueError("Exp449 smoke scientific hard link differs")
        identities_a.add((file.device, file.inode))
    identities_b: set[tuple[int, int]] = set()
    for file in files_b:
        if file.link_count != 1:
            raise ValueError("Exp449 smoke scientific hard link differs")
        identities_b.add((file.device, file.inode))
    if identities_a & identities_b:
        raise ValueError("Exp449 smoke scientific inode differs")
    checkpoint_hashes_a = {sha256(file.raw_bytes).hexdigest() for file in snapshot_a.checkpoint_files}
    checkpoint_hashes_b = {sha256(file.raw_bytes).hexdigest() for file in snapshot_b.checkpoint_files}
    if checkpoint_hashes_a & checkpoint_hashes_b:
        raise ValueError("Exp449 smoke checkpoint reuse differs")


def _tie_groups_antigravity(scores: np.ndarray) -> tuple[tuple[int, ...], ...]:
    groups: dict[float, list[int]] = {}
    for index, score in enumerate(scores.tolist()):
        groups.setdefault(float(score), []).append(index)
    return tuple(sorted((tuple(indices) for indices in groups.values()), key=lambda item: item))


def _coordinate_digests_antigravity(manifest: Exp449ProposalManifest) -> dict[str, str]:
    return {
        policy: sha256(canonical_json_bytes_antigravity([
            [descriptor.dataset, descriptor.policy_coordinate_sha256[policy]]
            for descriptor in manifest.entries
        ])).hexdigest()
        for policy in FROZEN_EXP449_POLICY_KEYS
    }


def _smoke_comparison_evidence_antigravity(root_a: Path, root_b: Path, authority: object) -> tuple[SmokeComparison, dict[str, object]]:
    """Compare only the immutable bytes returned by two complete root snapshots."""
    root_a, root_b, context_a, catalog_a, context_b, catalog_b = _require_independent_smoke_authority_antigravity(root_a, root_b, authority)
    snapshot_a = verify_exp449_proposal_root_snapshot_antigravity(root_a, context=context_a, catalog=catalog_a)
    snapshot_b = verify_exp449_proposal_root_snapshot_antigravity(root_b, context=context_b, catalog=catalog_b)
    hook = _AFTER_SMOKE_SNAPSHOTS_ANTIGRAVITY
    if hook is not None:
        hook()
    _require_shared_scientific_closure_antigravity(context_a, catalog_a, context_b, catalog_b)
    _require_independent_files_antigravity(snapshot_a, snapshot_b)
    if tuple(shard.dataset for shard in snapshot_a.shards) != tuple(shard.dataset for shard in snapshot_b.shards):
        raise ValueError("Exp449 smoke catalog order differs")
    maximum_delta = 0.0
    for shard_a, shard_b in zip(snapshot_a.shards, snapshot_b.shards, strict=True):
        if shard_a.dataset != shard_b.dataset:
            raise ValueError("Exp449 smoke catalog order differs")
        policies_a = dict((policy, (alerts, scores)) for policy, alerts, scores in shard_a.policies)
        policies_b = dict((policy, (alerts, scores)) for policy, alerts, scores in shard_b.policies)
        if tuple(policies_a) != FROZEN_EXP449_POLICY_KEYS or tuple(policies_b) != FROZEN_EXP449_POLICY_KEYS or FROZEN_EXP449_CONFIG.policy_id not in policies_a or FROZEN_EXP449_CONFIG.policy_id not in policies_b:
            raise ValueError("Exp449 smoke policy closure differs")
        for policy in FROZEN_EXP449_POLICY_KEYS:
            alerts_a, values_a = policies_a[policy]
            alerts_b, values_b = policies_b[policy]
            if alerts_a != alerts_b:
                raise ValueError("Exp449 smoke policy coordinates differ")
            scores_a = np.asarray(values_a, dtype=np.dtype("<f8"))
            scores_b = np.asarray(values_b, dtype=np.dtype("<f8"))
            if scores_a.ndim != 1 or scores_b.ndim != 1 or scores_a.size != scores_b.size or not np.all(np.isfinite(scores_a)) or not np.all(np.isfinite(scores_b)):
                raise ValueError("Exp449 smoke policy score shape differs")
            if _tie_groups_antigravity(scores_a) != _tie_groups_antigravity(scores_b):
                raise ValueError("Exp449 smoke score tie relation differs")
            delta = float(np.max(np.abs(scores_a - scores_b))) if scores_a.size else 0.0
            maximum_delta = max(maximum_delta, delta)
    if maximum_delta > FROZEN_EXP449_CONFIG.cpu_repeat_tolerance:
        raise ValueError("Exp449 smoke score reproducibility differs")
    manifest_hash_a = sha256(snapshot_a.manifest_file.raw_bytes).hexdigest()
    manifest_hash_b = sha256(snapshot_b.manifest_file.raw_bytes).hexdigest()
    comparison = SmokeComparison(
        manifest_hash_a, manifest_hash_b, len(snapshot_a.shards), True, maximum_delta, True,
    )
    return comparison, {
        "schema": "exp449-smoke-comparison-antigravity.v1",
        "smoke_a_manifest_sha256": comparison.smoke_a_manifest_sha256,
        "smoke_b_manifest_sha256": comparison.smoke_b_manifest_sha256,
        "smoke_a_coordinate_sha256": _coordinate_digests_antigravity(snapshot_a.manifest),
        "smoke_b_coordinate_sha256": _coordinate_digests_antigravity(snapshot_b.manifest),
        "maximum_score_delta": comparison.maximum_score_delta,
        "passed": comparison.passed,
    }


def compare_exp449_smokes_antigravity(root_a: Path, root_b: Path, authority: object) -> SmokeComparison:
    """Certify independent smoke roots only after exact coordinate comparison."""
    comparison, _evidence = _smoke_comparison_evidence_antigravity(root_a, root_b, authority)
    return comparison


__all__ = ["compare_exp449_smokes_antigravity", "verify_exp449_proposal_antigravity"]
