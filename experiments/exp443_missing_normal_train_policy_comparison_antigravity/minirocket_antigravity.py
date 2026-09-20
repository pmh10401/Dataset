"""Strict, lazy adapter around the unchanged Exp249 MiniROCKET scorer.

The only numerical implementation used here is
``experiments.exp249_250_official_mini_multirocket.scoring``.  It is imported
inside the scoring call so source verification and routing tests do not import
aeon accidentally.  This module owns only the Exp443 policy/result boundary:
four frozen kernel/seed calls, empirical alpha=.005 calibration, and immutable
contract construction.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from typing import Any

import numpy as np

from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import (
    KernelSeedScoresAntigravity,
    PolicyDatasetResultAntigravity,
)
from .preprocessing_antigravity import PreparedDatasetAntigravity


# A callable may be installed by behavior tests at this seam.  Leaving the
# value as ``None`` is important: importing this module must not import aeon.
score_configuration: Any = None


def _resolve_score_configuration() -> Any:
    if score_configuration is not None:
        if not callable(score_configuration):
            raise ValueError("score_configuration seam is not callable")
        return score_configuration
    # Keep the official third-party scorer unchanged and lazy.  In particular,
    # do not import its transforms or aeon at module import time.
    from experiments.exp249_250_official_mini_multirocket.scoring import (
        score_configuration as official_score_configuration,
    )

    return official_score_configuration


def _finite_score_array(value: object, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy array")
    if value.dtype != np.dtype("<f8") or value.ndim != 1 or not value.flags.c_contiguous:
        raise ValueError(f"{name} must be contiguous little-endian float64")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite values")
    copied = np.array(value, dtype="<f8", copy=True, order="C")
    copied.setflags(write=False)
    return copied


def _score_output_field(output: object, name: str) -> object:
    if isinstance(output, Mapping):
        if name not in output:
            raise ValueError(f"official scorer output is missing {name}")
        return output[name]
    if not hasattr(output, name):
        raise ValueError(f"official scorer output is missing {name}")
    return getattr(output, name)


def _validated_empirical_candidates(
    train_oof: np.ndarray, test_scores: np.ndarray, *, alpha: float
) -> tuple[int, ...]:
    # Importing calibration is cheap and label-free.  Calling the existing
    # implementation preserves Exp249's exact attainable-p convention.
    from experiments.exp239_248_diverse_score_sources.calibration import (
        empirical_candidates,
    )

    result = empirical_candidates(train_oof, test_scores, alpha)
    values = tuple(int(index) for index in result.indices)
    if values != tuple(sorted(set(values))) or any(index < 0 or index >= len(test_scores) for index in values):
        raise ValueError("empirical candidate indices are malformed")
    return values


def seed_intersection_candidates_antigravity(
    kernel_scores: Iterable[KernelSeedScoresAntigravity], alpha: float = 0.005
) -> tuple[int, ...]:
    """Intersect the two preregistered seed candidate sets for one kernel.

    ``kernel_scores`` must contain exactly one score for each frozen seed and
    exactly one kernel.  Candidate indices are recalculated from the immutable
    OOF/TEST score arrays and must agree with the stored per-seed candidates;
    this prevents a forged result from silently changing routing.
    """

    if type(alpha) is not float or not math.isfinite(alpha) or alpha != FROZEN_CONFIG.alpha:
        raise ValueError("alpha is not the frozen 0.005 threshold")
    scores = tuple(kernel_scores)
    if len(scores) != len(FROZEN_CONFIG.seeds) or any(type(item) is not KernelSeedScoresAntigravity for item in scores):
        raise ValueError("seed intersection requires both frozen kernel scores")
    kernels = {item.kernel for item in scores}
    if len(kernels) != 1 or {item.seed for item in scores} != set(FROZEN_CONFIG.seeds):
        raise ValueError("seed intersection identity is incomplete or duplicated")
    first = scores[0]
    if any(len(item.oof_scores) != len(first.oof_scores) or len(item.test_scores) != len(first.test_scores) for item in scores[1:]):
        raise ValueError("seed score lengths disagree")
    candidate_sets: list[set[int]] = []
    for item in sorted(scores, key=lambda value: value.seed):
        recalculated = _validated_empirical_candidates(item.oof_scores, item.test_scores, alpha=alpha)
        if recalculated != item.candidate_indices:
            raise ValueError("stored seed candidates disagree with empirical calibration")
        candidate_sets.append(set(recalculated))
    return tuple(sorted(candidate_sets[0].intersection(*candidate_sets[1:])))


def _abstained_result(prepared: PreparedDatasetAntigravity) -> PolicyDatasetResultAntigravity:
    return PolicyDatasetResultAntigravity(
        prepared.policy,
        prepared.dataset,
        prepared.target_length,
        prepared.train_total,
        prepared.train_missing,
        prepared.train_used,
        prepared.train_excluded,
        int(prepared.test_values.shape[0]),
        (),
        {},
        prepared.train_result.contract_audits,
        "abstained",
    )


def score_prepared_dataset_antigravity(
    prepared: PreparedDatasetAntigravity,
) -> PolicyDatasetResultAntigravity:
    """Score one prepared policy dataset with exactly four Exp249 calls."""

    if type(prepared) is not PreparedDatasetAntigravity:
        raise ValueError("prepared dataset has the wrong contract type")
    train = prepared.train_values
    test = prepared.test_values
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != prepared.target_length or test.shape[1] != prepared.target_length:
        raise ValueError("prepared matrices do not match the target length")
    if train.dtype != np.dtype("<f4") or test.dtype != np.dtype("<f4") or not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("prepared matrices must be finite little-endian float32")
    # Exp249's official minimum is five finite TRAIN rows.  Preprocessing has
    # already made this decision; preserve it as an explicit abstention rather
    # than manufacturing a score or a candidate set.
    if prepared.status != "scored" or prepared.train_used < 5:
        return _abstained_result(prepared)

    from experiments.exp249_250_official_mini_multirocket.config import (
        RocketModelConfig,
    )
    from experiments.exp249_250_official_mini_multirocket.transforms import (
        expected_feature_dimension,
    )

    config = RocketModelConfig(
        experiment=249,
        source="official_minirocket",
        transformer="minirocket",
        requested_kernels=tuple(FROZEN_CONFIG.kernel_requests),
        max_dilations_per_kernel=FROZEN_CONFIG.max_dilations,
        n_features_per_kernel=1,
        normalise=False,
    )
    official_scorer = _resolve_score_configuration()
    results: list[KernelSeedScoresAntigravity] = []
    for kernel in FROZEN_CONFIG.kernel_requests:
        expected_dimension = expected_feature_dimension(config, int(kernel))
        frozen_dimension = FROZEN_CONFIG.kernel_dimensions[FROZEN_CONFIG.kernel_requests.index(kernel)]
        if expected_dimension != frozen_dimension:
            raise ValueError("official MiniROCKET dimension disagrees with frozen config")
        for seed in FROZEN_CONFIG.seeds:
            output = official_scorer(
                train,
                test,
                config=config,
                requested_kernels=int(kernel),
                seed=int(seed),
                expected_dimension=int(expected_dimension),
                max_train_reference=FROZEN_CONFIG.max_reference,
                query_chunk_size=FROZEN_CONFIG.query_chunk,
            )
            oof = _finite_score_array(_score_output_field(output, "train_oof"), "train_oof")
            test_scores = _finite_score_array(_score_output_field(output, "test_scores"), "test_scores")
            feature_dimension = _score_output_field(output, "feature_dimension")
            if type(feature_dimension) is not int or feature_dimension != expected_dimension:
                raise ValueError(
                    f"feature dimension mismatch: expected={expected_dimension} actual={feature_dimension}"
                )
            if len(oof) != train.shape[0] or len(test_scores) != test.shape[0]:
                raise ValueError("official scorer returned score arrays with wrong lengths")
            candidates = _validated_empirical_candidates(
                oof, test_scores, alpha=FROZEN_CONFIG.alpha
            )
            results.append(
                KernelSeedScoresAntigravity(
                    int(kernel), int(seed), oof, test_scores, candidates
                )
            )

    candidates_by_kernel: dict[int, tuple[int, ...]] = {}
    for kernel in FROZEN_CONFIG.kernel_requests:
        pair = tuple(item for item in results if item.kernel == kernel)
        if len(pair) != 2:
            raise ValueError("official scorer did not return both frozen seeds")
        candidates_by_kernel[int(kernel)] = seed_intersection_candidates_antigravity(
            pair, alpha=FROZEN_CONFIG.alpha
        )
    return PolicyDatasetResultAntigravity(
        prepared.policy,
        prepared.dataset,
        prepared.target_length,
        prepared.train_total,
        prepared.train_missing,
        prepared.train_used,
        prepared.train_excluded,
        int(test.shape[0]),
        tuple(results),
        candidates_by_kernel,
        prepared.train_result.contract_audits,
        "scored",
    )


__all__ = [
    "score_configuration",
    "score_prepared_dataset_antigravity",
    "seed_intersection_candidates_antigravity",
]
