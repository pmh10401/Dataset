"""Frozen, label-free Exp449 prediction lifecycle; official entry is intentionally guarded."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable

import numpy as np

from experiments.exp239_248_diverse_score_sources.calibration import empirical_candidates

from .config_antigravity import CANDIDATE_MATRIX_ANTIGRAVITY, FROZEN_CONFIG_ANTIGRAVITY
from .progress_antigravity import commit_candidate_leaf_antigravity


_OFFICIAL_ROOT_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/exp449_residual_fn_method_qualification_antigravity_20260812")


@dataclass(frozen=True, slots=True)
class FrozenPredictionAuthorityAntigravity:
    reviewed_commit: str
    family_method_pairs: int
    dataset_units: int
    candidate_names: tuple[str, ...]
    inventory: tuple[str, ...]
    leaf_sha256: tuple[tuple[str, str], ...]
    aggregate_sha256: str
    label_access_count: int


@dataclass(frozen=True, slots=True)
class PredictionDependenciesAntigravity:
    root: Path
    exp448_authority: object
    load_bundle: Callable[[int, str], tuple[str, np.ndarray, np.ndarray]]
    score: Callable[[np.ndarray, np.ndarray, str, int], tuple[np.ndarray, np.ndarray]]
    method_audit: Callable[[], object]
    mlx_decision: Callable[[], object]
    executor_factory: object | None = None


def _families_antigravity(authority: object) -> tuple[str, ...]:
    rows = getattr(authority, "target_baseline_rows", None)
    if type(rows) is not tuple or len(rows) != 48:
        raise ValueError("Exp448 target authority differs")
    families = tuple(getattr(row, "family", None) for row in rows)
    expected = tuple(family for family, _ in CANDIDATE_MATRIX_ANTIGRAVITY)
    if any(type(family) is not str for family in families) or tuple(sorted(set(families), key=expected.index)) != expected:
        raise ValueError("Exp448 family order differs")
    return families


def predict_with_dependencies_antigravity(dependencies: PredictionDependenciesAntigravity, *, reviewed_commit: str = "synthetic") -> FrozenPredictionAuthorityAntigravity:
    """Run synthetic/test dependencies in frozen sequence, committing only canonical leaves."""
    if type(dependencies) is not PredictionDependenciesAntigravity or type(reviewed_commit) is not str or not reviewed_commit:
        raise ValueError("prediction dependencies are invalid")
    families = _families_antigravity(dependencies.exp448_authority)
    candidates = dict(CANDIDATE_MATRIX_ANTIGRAVITY)
    root = dependencies.root
    leaves = root / "candidates_antigravity"
    names: list[str] = []
    inventory: list[str] = []
    digests: list[tuple[str, str]] = []
    for sequence, family in enumerate(families):
        dataset, train, test = dependencies.load_bundle(sequence, family)
        for method in candidates[family]:
            name = f"{dataset}__{method}"
            relative = f"{sequence:03d}_{name}.json"
            score_rows = [dependencies.score(train, test, method, seed) for seed in FROZEN_CONFIG_ANTIGRAVITY["seeds"]]
            train_median = np.median(np.stack([item[0] for item in score_rows]), axis=0)
            test_median = np.median(np.stack([item[1] for item in score_rows]), axis=0)
            if not np.isfinite(train_median).all() or not np.isfinite(test_median).all():
                raise ValueError("non-finite score")
            indices = tuple(int(item) for item in empirical_candidates(train_median, test_median, 0.005).indices)
            payload = {"sequence": sequence, "candidate_name": name, "family": family, "method": method, "indices": indices, "contains_test_labels": False, "target_length": int(len(test)), "train_count": int(len(train)), "test_count": int(len(test)), "seeds": list(FROZEN_CONFIG_ANTIGRAVITY["seeds"]), "alpha": 0.005}
            digest = commit_candidate_leaf_antigravity(leaves, relative, payload)
            names.append(name)
            inventory.append("candidates_antigravity/" + relative)
            digests.append(("candidates_antigravity/" + relative, digest))
    aggregate = hashlib.sha256("".join(digest for _, digest in digests).encode()).hexdigest()
    return FrozenPredictionAuthorityAntigravity(reviewed_commit, 16, 96, tuple(names), tuple(inventory), tuple(digests), aggregate, 0)


def predict_antigravity(reviewed_commit: str) -> FrozenPredictionAuthorityAntigravity:
    """Official lifecycle is unavailable until the sealed controller supplies real dependencies."""
    raise RuntimeError("official Exp449 prediction lifecycle requires the sealed controller")


def validate_frozen_prediction_authority_antigravity(authority: object) -> tuple[str, ...]:
    """Fail closed without opening a database; official root validation is intentionally descriptor-only."""
    if type(authority) is not FrozenPredictionAuthorityAntigravity:
        raise ValueError("prediction authority is invalid")
    if not _OFFICIAL_ROOT_ANTIGRAVITY.is_dir():
        raise ValueError("official prediction root is unavailable")
    if authority.family_method_pairs != 16 or authority.dataset_units != 96 or authority.label_access_count != 0:
        raise ValueError("prediction authority is invalid")
    datasets = tuple(name.split("__", 1)[0] for name in authority.candidate_names[::2])
    if len(datasets) != 48 or len(set(datasets)) != 48:
        raise ValueError("prediction authority is invalid")
    return datasets


__all__ = ["FrozenPredictionAuthorityAntigravity", "PredictionDependenciesAntigravity", "predict_antigravity", "predict_with_dependencies_antigravity", "validate_frozen_prediction_authority_antigravity"]
