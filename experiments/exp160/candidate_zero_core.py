"""Pure logic helpers for Exp160 Candidate-Zero Similarity Recovery.

This module is completely decoupled from database access and TEST label access.
It imports robust scaling, distance, and donor selection helpers from Exp158_159.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

# Add exp158_159 to sys.path to reuse similarity_pool_core
EXP158_159_DIR = REPO_ROOT / "experiments" / "exp158_159"
if str(EXP158_159_DIR) not in sys.path:
    sys.path.insert(0, str(EXP158_159_DIR))

import similarity_pool_core as pool_core

# Korean display names mapped from internal keys
DISPLAY_NAMES = {
    "rocket_exp40": "시간 패턴 점수",
    "exp55_best": "주파수 변화 점수",
    "exp56_best": "반복 구조 점수",
    "exp84": "다중 커널 점수",
}


def is_candidate_zero_target(n_train: int, baseline_candidate_count: int) -> bool:
    """Target gate: baseline candidate_count == 0 and n_train >= 5."""
    return int(n_train) >= 5 and int(baseline_candidate_count) == 0


def is_abstained(n_train: int) -> bool:
    """n_train < 5 is complete calibration abstention."""
    return int(n_train) < 5


def is_eligible_donor(n_train: int, descriptor_complete: bool) -> bool:
    """A donor must have enough TRAIN rows and a complete TRAIN descriptor."""
    return int(n_train) >= 5 and bool(descriptor_complete)


def prediction_policy_metadata(*, candidate_empty_gate_applied: bool) -> dict:
    """Return policy provenance shared by every label-free prediction row."""
    return {
        "source_display_names": dict(DISPLAY_NAMES),
        "uses_test_candidate_empty_gate": 1,
        "candidate_empty_gate_applied": int(bool(candidate_empty_gate_applied)),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "uses_family_name": 0,
        "uses_test_length_for_policy": 0,
        "uses_dataset_name_for_policy": 0,
    }


def integrate_candidate_zero_review(
    *,
    dataset_name: str,
    alpha: float,
    n_train: int,
    baseline_hard: Iterable[int],
    baseline_standard: Iterable[int],
    baseline_priority: Iterable[int],
    similarity_candidates: Iterable[int],
    n_test: int,
) -> dict[str, set[int]]:
    """Integrate similarity review candidates for target datasets only.

    For non-target datasets (e.g. n_train < 5 or baseline candidate_count > 0),
    integrated lanes must exactly match the baseline lanes.
    """
    hard = {int(idx) for idx in baseline_hard}
    standard = {int(idx) for idx in baseline_standard}
    priority = {int(idx) for idx in baseline_priority}

    if hard & standard or hard & priority or standard & priority:
        raise ValueError("baseline lane overlap")

    baseline_candidate = hard | standard | priority
    baseline_candidate_count = len(baseline_candidate)

    # Check if target
    if not is_candidate_zero_target(n_train, baseline_candidate_count):
        # Must exact match baseline. Verify range.
        invalid = sorted(idx for idx in baseline_candidate if idx < 0 or idx >= int(n_test))
        if invalid:
            raise ValueError(f"baseline index out of range: {invalid}")
        return {
            "hard": hard,
            "standard_review": standard,
            "priority_review": priority,
            "candidate": baseline_candidate,
        }

    # If it is a target, merge similarity_candidates into Standard review only
    sim_set = {int(idx) for idx in similarity_candidates}

    # Check range of all candidates
    all_indices = hard | standard | priority | sim_set
    invalid = sorted(idx for idx in all_indices if idx < 0 or idx >= int(n_test))
    if invalid:
        raise ValueError(f"candidate index out of range: {invalid}")

    # Standard review gets baseline standard plus new similarity candidates
    # that are not already in any of the baseline lanes.
    new_standard = standard | (sim_set - baseline_candidate)

    return {
        "hard": hard,
        "standard_review": new_standard,
        "priority_review": priority,
        "candidate": hard | new_standard | priority,
    }


def canonical_row_hash(row: dict) -> str:
    """Compute sha256 hash of a row after serializing keys deterministically."""
    cleaned = {}
    for k, v in row.items():
        if isinstance(v, (set, frozenset)):
            cleaned[k] = sorted(list(v))
        else:
            cleaned[k] = v
    serialized = json.dumps(cleaned, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
