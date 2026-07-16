from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from experiments.exp186_200.artifacts import load_verified_predictions, verify_file_hash
from experiments.exp186_200.core import empirical_upper_tail_pvalues
from experiments.exp220_223.policy import sha256_file


def _indices(values: Iterable[int]) -> set[int]:
    return {int(value) for value in values}


def parse_indices(value: object) -> set[int]:
    text = str(value or "").replace(",", " ").strip()
    return {int(token) for token in text.split()} if text else set()


def replay_profile(
    *,
    profile_id: str,
    n_query: int,
    exp137_lanes: Mapping[str, Iterable[int]],
    conservative: Iterable[int],
    canonical_knn_p: np.ndarray,
    exp195_zero_train_guard: bool,
    expected_exp190: Mapping[str, Iterable[int]],
    expected_exp195: Iterable[int],
    expected_canonical_knn_p: np.ndarray | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> dict[str, object]:
    pvalues = np.asarray(canonical_knn_p, dtype=float)
    if pvalues.shape != (int(n_query),):
        raise ValueError(f"canonical score length mismatch: profile={profile_id}")
    if not np.all(np.isfinite(pvalues)):
        raise ValueError(f"non-finite canonical p-value: profile={profile_id}")

    lanes = {name: _indices(values) for name, values in exp137_lanes.items()}
    excluded = set().union(*lanes.values()) if lanes else set()
    if any(index < 0 or index >= n_query for index in excluded):
        raise ValueError(f"Exp137 index outside query range: profile={profile_id}")
    lane_names = sorted(lanes)
    for pos, left in enumerate(lane_names):
        for right in lane_names[pos + 1 :]:
            if lanes[left] & lanes[right]:
                raise ValueError(
                    f"Exp137 lanes overlap: profile={profile_id} lanes={left},{right}"
                )

    conservative_set = _indices(conservative) - excluded
    canonical_set = set(np.flatnonzero(pvalues <= 0.005).astype(int).tolist()) - excluded
    high = conservative_set & canonical_set
    standard = conservative_set - canonical_set
    shadow = set(high) if exp195_zero_train_guard else set()
    expected_high = _indices(expected_exp190.get("high", ()))
    expected_standard = _indices(expected_exp190.get("standard", ()))
    expected_shadow = _indices(expected_exp195)
    candidate_match = (
        high == expected_high
        and standard == expected_standard
        and shadow == expected_shadow
    )
    score_match = True
    if expected_canonical_knn_p is not None:
        expected_scores = np.asarray(expected_canonical_knn_p, dtype=float)
        score_match = bool(
            expected_scores.shape == pvalues.shape
            and np.allclose(pvalues, expected_scores, rtol=rtol, atol=atol)
        )

    return {
        "profile_id": profile_id,
        "passed": bool(candidate_match and score_match),
        "candidate_match": bool(candidate_match),
        "score_match": bool(score_match),
        "actual": {
            "exp190_high": sorted(high),
            "exp190_standard": sorted(standard),
            "exp195_shadow": sorted(shadow),
        },
        "expected": {
            "exp190_high": sorted(expected_high),
            "exp190_standard": sorted(expected_standard),
            "exp195_shadow": sorted(expected_shadow),
        },
    }


def _rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _verified_prediction_map(
    prediction_path: Path,
    manifest_path: Path,
    *,
    variants: Iterable[str],
) -> dict[tuple[str, str], dict[str, object]]:
    allowed = set(variants)
    rows = load_verified_predictions(
        prediction_path, manifest_path, expected_datasets=639
    )
    selected = {
        (str(row["dataset"]), str(row["variant"])): row
        for row in rows
        if str(row["variant"]) in allowed
    }
    if len(selected) != 639 * len(allowed):
        raise ValueError(
            f"prediction coverage mismatch: {prediction_path} rows={len(selected)}"
        )
    return selected


def run_cached_replay_audit(
    *,
    repo_root: Path,
    data_root: Path,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Recalibrate frozen score arrays and exactly replay Exp190/195 routing.

    This audit validates candidate logic and every consumed file hash. It does
    not claim that historical feature extractors were refit from raw series;
    callers must retain the explicit ``raw_feature_recompute=False`` marker.
    """

    repo_root = Path(repo_root)
    data_root = Path(data_root)
    exp177_root = data_root / "exp177_185_unattended"
    exp186_root = data_root / "exp186_200_unattended"
    baseline_path = repo_root / "results/exp137/experiment_137_operational_triage_results.csv"
    baseline_rows = {
        row["dataset_name"]: row
        for row in _rows(baseline_path)
        if int(row["train_normal_count"]) >= 20
    }
    if len(baseline_rows) != 639:
        raise ValueError(f"eligible profile mismatch: {len(baseline_rows)} != 639")

    exp178 = _verified_prediction_map(
        exp177_root / "exp178_frozen_predictions.jsonl",
        exp177_root / "exp178_prediction_manifest.json",
        variants=("risk_target_0.005",),
    )
    exp190 = _verified_prediction_map(
        exp186_root / "exp190_frozen_predictions.jsonl",
        exp186_root / "exp190_prediction_manifest.json",
        variants=("three_tier_high_confidence", "three_tier_standard"),
    )
    exp195 = _verified_prediction_map(
        exp186_root / "exp195_frozen_predictions.jsonl",
        exp186_root / "exp195_prediction_manifest.json",
        variants=("exp191_plus_zero_train_candidate_guard",),
    )

    manifest_path = exp177_root / "exp180_canonical_score_manifest.csv"
    canonical_manifest = {row["dataset_name"]: row for row in _rows(manifest_path)}
    if len(canonical_manifest) != 639:
        raise ValueError(f"canonical score coverage mismatch: {len(canonical_manifest)}")
    precompute_path = exp186_root / "train_guard_precompute.json"
    precompute = json.loads(precompute_path.read_text(encoding="utf-8"))["results"]
    if set(precompute) != set(baseline_rows):
        raise ValueError("TRAIN guard precompute profile mismatch")

    details: list[dict[str, object]] = []
    verified_score_files = 0
    for profile_id in sorted(baseline_rows):
        score_row = canonical_manifest[profile_id]
        score_path = exp177_root / "canonical_score_cache" / score_row["score_file"]
        verify_file_hash(score_path, score_row["score_sha256"], "canonical score cache")
        verified_score_files += 1
        with np.load(score_path, allow_pickle=False) as archive:
            train_scores = np.asarray(archive["knn_local_gap__train_oof"], dtype=float)
            query_scores = np.asarray(archive["knn_local_gap__test"], dtype=float)
        canonical_p = empirical_upper_tail_pvalues(train_scores, query_scores)
        baseline = baseline_rows[profile_id]
        guard = sum(int(value) for value in precompute[profile_id]["candidate_counts_195"]) == 0
        details.append(
            replay_profile(
                profile_id=profile_id,
                n_query=len(query_scores),
                exp137_lanes={
                    "hard": parse_indices(baseline["hard_alert_indices"]),
                    "standard": parse_indices(baseline["standard_review_indices"]),
                    "priority": parse_indices(baseline["priority_review_indices"]),
                },
                conservative=exp178[(profile_id, "risk_target_0.005")]["candidate_indices"],
                canonical_knn_p=canonical_p,
                exp195_zero_train_guard=guard,
                expected_exp190={
                    "high": exp190[(profile_id, "three_tier_high_confidence")]["candidate_indices"],
                    "standard": exp190[(profile_id, "three_tier_standard")]["candidate_indices"],
                },
                expected_exp195=exp195[
                    (profile_id, "exp191_plus_zero_train_candidate_guard")
                ]["candidate_indices"],
            )
        )

    mismatches = [row for row in details if not row["passed"]]
    report: dict[str, object] = {
        "experiment_id": "experiment_220_policy_replay_audit",
        "status": "passed" if not mismatches else "failed",
        "profile_count": len(details),
        "passed_profiles": len(details) - len(mismatches),
        "mismatch_count": len(mismatches),
        "verified_canonical_score_files": verified_score_files,
        "candidate_equality_required": True,
        "candidate_equality_passed": not mismatches,
        "score_tolerance": {"rtol": 1e-6, "atol": 1e-8},
        "score_scope": "recalibrated_from_verified_frozen_score_arrays",
        "raw_feature_recompute": False,
        "end_to_end_train_only_provenance_validated": False,
        "shadow_execution_gate": "blocked_pending_raw_feature_replay",
        "input_sha256": {
            "exp137": sha256_file(baseline_path),
            "exp178_manifest": sha256_file(exp177_root / "exp178_prediction_manifest.json"),
            "exp190_manifest": sha256_file(exp186_root / "exp190_prediction_manifest.json"),
            "exp195_manifest": sha256_file(exp186_root / "exp195_prediction_manifest.json"),
            "canonical_manifest": sha256_file(manifest_path),
            "train_guard_precompute": sha256_file(precompute_path),
        },
        "mismatches": mismatches,
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def build_raw_audit_report(
    *,
    details: list[dict[str, object]],
    expected_profiles: int,
    input_hashes: Mapping[str, str],
) -> dict[str, object]:
    failures = [row for row in details if not bool(row.get("passed"))]
    full_coverage = len(details) == int(expected_profiles)
    if failures:
        status = "failed"
        gate = "blocked_raw_replay_mismatch"
    elif not full_coverage:
        status = "partial"
        gate = "blocked_incomplete_raw_replay"
    else:
        status = "passed"
        gate = "passed"
    return {
        "experiment_id": "experiment_220_policy_raw_replay_audit",
        "status": status,
        "profile_count": len(details),
        "expected_profiles": int(expected_profiles),
        "passed_profiles": sum(bool(row.get("passed")) for row in details),
        "mismatch_count": len(failures),
        "candidate_equality_required": True,
        "score_tolerance": {"rtol": 1e-6, "atol": 1e-8},
        "raw_feature_recompute": True,
        "end_to_end_train_only_provenance_validated": status == "passed",
        "exp137_adapter_mode": "mandatory_frozen_upstream_lanes",
        "shadow_execution_gate": gate,
        "input_sha256": dict(input_hashes),
        "mismatches": failures,
    }


def _allclose_detail(actual: np.ndarray, expected: np.ndarray) -> tuple[bool, float | None]:
    left = np.asarray(actual, dtype=float)
    right = np.asarray(expected, dtype=float)
    if left.shape != right.shape:
        return False, None
    passed = bool(np.allclose(left, right, rtol=1e-6, atol=1e-8))
    difference = float(np.max(np.abs(left - right))) if left.size else 0.0
    return passed, difference


def _raw_replay_worker(args: tuple[object, ...]) -> dict[str, object]:
    (
        profile_id,
        database_path,
        source_cache_path,
        source_cache_sha,
        canonical_cache_path,
        canonical_cache_sha,
        conservative_alpha,
        exp137_lanes,
        expected_high,
        expected_standard,
        expected_shadow,
        expected_guard,
    ) = args
    from experiments.exp153_157 import exp153_157_bridge as bridge
    from experiments.exp177_185 import core as canonical_core
    from experiments.exp177_185.core import SOURCE_CACHE_KEYS
    from experiments.exp220_223.scoring import exp195_zero_train_guard, run_source_crossfit

    profile_id = str(profile_id)
    verify_file_hash(Path(source_cache_path), str(source_cache_sha), "Exp153 score cache")
    verify_file_hash(Path(canonical_cache_path), str(canonical_cache_sha), "canonical score cache")
    x_train, x_query, record = bridge.exp152.load_dataset_series_only(
        profile_id, Path(str(database_path))
    )
    source_train: dict[str, np.ndarray] = {}
    source_query: dict[str, np.ndarray] = {}
    score_checks: dict[str, dict[str, object]] = {}
    with np.load(Path(source_cache_path), allow_pickle=False) as frozen_source:
        for public_name, source_name in SOURCE_CACHE_KEYS.items():
            recomputed = run_source_crossfit(
                bridge, source_name, x_train, x_query, record
            )
            train = np.asarray(recomputed["train_oof_scores"], dtype=float)
            query = np.asarray(recomputed["test_scores"], dtype=float)
            source_train[public_name] = train
            source_query[public_name] = query
            train_ok, train_diff = _allclose_detail(
                train, frozen_source[f"{source_name}__train_oof"]
            )
            query_ok, query_diff = _allclose_detail(
                query, frozen_source[f"{source_name}__test"]
            )
            score_checks[public_name] = {
                "train_match": train_ok,
                "query_match": query_ok,
                "train_max_abs_diff": train_diff,
                "query_max_abs_diff": query_diff,
            }

    train_features = canonical_core.canonical_features(x_train)
    query_features = canonical_core.canonical_features(x_query)
    canonical_train, canonical_query = canonical_core.crossfit_scores(
        train_features,
        query_features,
        backend="knn_local_gap",
        seed=canonical_core.SEED,
    )
    with np.load(Path(canonical_cache_path), allow_pickle=False) as frozen_canonical:
        train_ok, train_diff = _allclose_detail(
            canonical_train, frozen_canonical["knn_local_gap__train_oof"]
        )
        query_ok, query_diff = _allclose_detail(
            canonical_query, frozen_canonical["knn_local_gap__test"]
        )
    score_checks["canonical_knn"] = {
        "train_match": train_ok,
        "query_match": query_ok,
        "train_max_abs_diff": train_diff,
        "query_max_abs_diff": query_diff,
    }
    score_match = all(
        bool(row["train_match"]) and bool(row["query_match"])
        for row in score_checks.values()
    )
    source_p = {
        name: empirical_upper_tail_pvalues(source_train[name], source_query[name])
        for name in source_train
    }
    support = np.sum(
        np.vstack([values <= float(conservative_alpha) for values in source_p.values()]),
        axis=0,
    )
    canonical_p = empirical_upper_tail_pvalues(canonical_train, canonical_query)
    excluded = set().union(*(_indices(values) for values in exp137_lanes.values()))
    conservative = set(np.flatnonzero(support >= 2).astype(int).tolist()) - excluded
    canonical = set(np.flatnonzero(canonical_p <= 0.005).astype(int).tolist()) - excluded
    high = conservative & canonical
    standard = conservative - canonical
    guard = exp195_zero_train_guard(
        source_train,
        canonical_train,
        conservative_alpha=float(conservative_alpha),
    )
    shadow = high if guard else set()
    candidate_match = (
        high == _indices(expected_high)
        and standard == _indices(expected_standard)
        and shadow == _indices(expected_shadow)
    )
    guard_match = guard == bool(expected_guard)
    return {
        "profile_id": profile_id,
        "passed": bool(score_match and candidate_match and guard_match),
        "score_match": score_match,
        "candidate_match": candidate_match,
        "guard_match": guard_match,
        "actual": {
            "exp190_high": sorted(high),
            "exp190_standard": sorted(standard),
            "exp195_shadow": sorted(shadow),
            "exp195_zero_train_guard": guard,
        },
        "expected": {
            "exp190_high": sorted(_indices(expected_high)),
            "exp190_standard": sorted(_indices(expected_standard)),
            "exp195_shadow": sorted(_indices(expected_shadow)),
            "exp195_zero_train_guard": bool(expected_guard),
        },
        "score_checks": score_checks if not score_match else {},
    }


def run_raw_replay_audit(
    *,
    repo_root: Path,
    data_root: Path,
    database_path: Path,
    policy_path: Path,
    output_path: Path,
    workers: int = 1,
    profile_limit: int | None = None,
) -> dict[str, object]:
    from experiments.exp220_223.policy import load_verified_policy

    repo_root = Path(repo_root)
    data_root = Path(data_root)
    policy = load_verified_policy(policy_path, verify_artifacts=True)
    exp177_root = data_root / "exp177_185_unattended"
    exp186_root = data_root / "exp186_200_unattended"
    baseline_path = repo_root / "results/exp137/experiment_137_operational_triage_results.csv"
    baseline = {
        row["dataset_name"]: row
        for row in _rows(baseline_path)
        if int(row["train_normal_count"]) >= 20
    }
    exp190 = _verified_prediction_map(
        exp186_root / "exp190_frozen_predictions.jsonl",
        exp186_root / "exp190_prediction_manifest.json",
        variants=("three_tier_high_confidence", "three_tier_standard"),
    )
    exp195 = _verified_prediction_map(
        exp186_root / "exp195_frozen_predictions.jsonl",
        exp186_root / "exp195_prediction_manifest.json",
        variants=("exp191_plus_zero_train_candidate_guard",),
    )
    source_manifest_path = repo_root / "outputs/exp153_157_train_only_improvements/01_score_cache_manifest.csv"
    source_manifest = {row["dataset_name"]: row for row in _rows(source_manifest_path)}
    canonical_manifest_path = exp177_root / "exp180_canonical_score_manifest.csv"
    canonical_manifest = {
        row["dataset_name"]: row for row in _rows(canonical_manifest_path)
    }
    source_cache_root = Path(
        "/Users/minho/Documents/timesries project/outputs/exp153_157_train_only_improvements/cache"
    )
    profile_ids = sorted(baseline)
    if profile_limit is not None:
        profile_ids = profile_ids[: int(profile_limit)]
    alpha = float(policy["metadata"]["conservative_alpha"])
    jobs = []
    for profile_id in profile_ids:
        baseline_row = baseline[profile_id]
        source_row = source_manifest[profile_id]
        canonical_row = canonical_manifest[profile_id]
        jobs.append(
            (
                profile_id,
                str(database_path),
                str(source_cache_root / source_row["cache_file"]),
                source_row["cache_sha256"],
                str(exp177_root / "canonical_score_cache" / canonical_row["score_file"]),
                canonical_row["score_sha256"],
                alpha,
                {
                    "hard": parse_indices(baseline_row["hard_alert_indices"]),
                    "standard": parse_indices(baseline_row["standard_review_indices"]),
                    "priority": parse_indices(baseline_row["priority_review_indices"]),
                },
                exp190[(profile_id, "three_tier_high_confidence")]["candidate_indices"],
                exp190[(profile_id, "three_tier_standard")]["candidate_indices"],
                exp195[(profile_id, "exp191_plus_zero_train_candidate_guard")]["candidate_indices"],
                policy["profiles"][profile_id]["exp195_zero_train_guard"],
            )
        )
    details: list[dict[str, object]] = []
    if int(workers) <= 1:
        for done, job in enumerate(jobs, 1):
            details.append(_raw_replay_worker(job))
            print(f"Raw replay: [{done}/{len(jobs)}] {job[0]}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = {executor.submit(_raw_replay_worker, job): str(job[0]) for job in jobs}
            for done, future in enumerate(as_completed(futures), 1):
                profile_id = futures[future]
                details.append(future.result())
                print(f"Raw replay: [{done}/{len(jobs)}] {profile_id}", flush=True)
    details.sort(key=lambda row: str(row["profile_id"]))
    report = build_raw_audit_report(
        details=details,
        expected_profiles=639,
        input_hashes={
            "policy": sha256_file(policy_path),
            "database": sha256_file(database_path),
            "source_manifest": sha256_file(source_manifest_path),
            "canonical_manifest": sha256_file(canonical_manifest_path),
        },
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
