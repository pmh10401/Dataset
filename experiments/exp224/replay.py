from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np

from experiments.exp224.artifacts import compute_profile_artifact, sha256_file
from experiments.exp224.core import ALPHAS, RISK_TARGET, select_largest_safe_alpha, train_fold_candidate_counts
from experiments.exp224.policy import load_verified_exp224_policy

FROZEN_PROFILE_COUNT = 639


def raw_replay_gate(
    profile_count: int,
    expected_profile_count: int,
    *,
    mismatch_count: int,
    risk_match: bool,
) -> str:
    if int(mismatch_count) > 0 or not bool(risk_match):
        return "blocked_raw_replay_mismatch"
    if (
        int(expected_profile_count) != FROZEN_PROFILE_COUNT
        or int(profile_count) != FROZEN_PROFILE_COUNT
    ):
        return "blocked_incomplete_raw_replay"
    return "passed"


def _compare_arrays(
    expected: Mapping[str, np.ndarray],
    actual: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    mismatches: list[dict[str, object]] = []
    for name in sorted(set(expected) | set(actual)):
        if name not in expected or name not in actual:
            mismatches.append({"array": name, "reason": "missing_array"})
            continue
        left = np.asarray(expected[name])
        right = np.asarray(actual[name])
        if left.dtype.kind in "US" or right.dtype.kind in "US":
            if not np.array_equal(left, right):
                mismatches.append({"array": name, "reason": "exact_mismatch"})
            continue
        if left.shape != right.shape:
            mismatches.append(
                {
                    "array": name,
                    "reason": "shape_mismatch",
                    "expected": list(left.shape),
                    "actual": list(right.shape),
                }
            )
            continue
        if not np.allclose(left, right, rtol=1e-6, atol=1e-8, equal_nan=False):
            delta = np.abs(left.astype(float) - right.astype(float))
            mismatches.append(
                {
                    "array": name,
                    "reason": "score_mismatch",
                    "max_abs_diff": float(np.max(delta)) if delta.size else 0.0,
                }
            )
    return mismatches


def _replay_one(arguments: tuple[dict[str, object], str]) -> dict[str, object]:
    profile, db_path = arguments
    artifact_path = Path(str(profile["artifact_path"]))
    with np.load(artifact_path, allow_pickle=False) as archive:
        expected = {name: archive[name] for name in archive.files}
    actual = compute_profile_artifact(str(profile["profile_id"]), Path(db_path))
    mismatches = _compare_arrays(expected, actual)
    counts, trials = train_fold_candidate_counts(
        actual["temporal_oof"], actual["canonical_oof"]
    )
    return {
        "profile_id": profile["profile_id"],
        "mismatches": mismatches,
        "false_counts": {str(alpha): counts[alpha] for alpha in ALPHAS},
        "trials": trials,
    }


def run_raw_replay(
    policy_path: Path,
    db_path: Path,
    output_path: Path,
    workers: int,
    profile_limit: int | None = None,
) -> dict[str, object]:
    policy = load_verified_exp224_policy(policy_path, verify_artifacts=True)
    database = Path(db_path).resolve()
    if sha256_file(database) != policy["database_sha256"]:
        raise ValueError("replay database SHA-256 mismatch")
    profiles = list(policy["profiles"])
    if profile_limit is not None:
        profiles = profiles[: max(0, int(profile_limit))]
    arguments = [(profile, str(database)) for profile in profiles]
    if int(workers) == 1:
        rows = [_replay_one(argument) for argument in arguments]
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            rows = list(executor.map(_replay_one, arguments))
    details = [
        {"profile_id": row["profile_id"], "mismatches": row["mismatches"]}
        for row in rows
        if row["mismatches"]
    ]
    aggregate_counts = {str(alpha): 0 for alpha in ALPHAS}
    trials = 0
    for row in rows:
        trials += int(row["trials"])
        for alpha in ALPHAS:
            aggregate_counts[str(alpha)] += int(row["false_counts"][str(alpha)])
    complete = len(rows) == int(policy["expected_profile_count"])
    selected_alpha = (
        select_largest_safe_alpha(
            {float(key): value for key, value in aggregate_counts.items()},
            trials,
            RISK_TARGET,
        )
        if trials
        else None
    )
    risk_match = bool(
        complete
        and aggregate_counts == policy["train_risk"]["false_counts"]
        and trials == int(policy["train_risk"]["trials"])
        and selected_alpha == policy["selected_alpha"]
    )
    gate = raw_replay_gate(
        len(rows),
        int(policy["expected_profile_count"]),
        mismatch_count=len(details),
        risk_match=(risk_match if complete else True),
    )
    report: dict[str, object] = {
        "schema": "exp224_raw_replay_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_hash": policy["policy_hash"],
        "database_sha256": policy["database_sha256"],
        "profile_count": len(rows),
        "expected_profile_count": int(policy["expected_profile_count"]),
        "mismatch_count": len(details),
        "mismatches": details,
        "train_risk": {
            "false_counts": aggregate_counts,
            "trials": trials,
            "selected_alpha": selected_alpha,
            "matches_policy": risk_match,
        },
        "shadow_execution_gate": gate,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report
