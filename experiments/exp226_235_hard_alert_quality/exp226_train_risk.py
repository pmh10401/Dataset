from __future__ import annotations

from collections.abc import Iterable, Sequence
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from experiments.exp206_210.core import empirical_upper_pvalues
from experiments.exp206_210.core import fold_plan
from experiments.exp177_185.core import _fold_plan
from experiments.exp226_235_hard_alert_quality.database import sha256_file


FoldPlan = Sequence[tuple[np.ndarray, np.ndarray]]


def _validate_plan(plan: FoldPlan, n_rows: int) -> None:
    seen: list[int] = []
    universe = set(range(int(n_rows)))
    for fit_raw, query_raw in plan:
        fit = {int(value) for value in np.asarray(fit_raw).reshape(-1)}
        query = [int(value) for value in np.asarray(query_raw).reshape(-1)]
        if fit & set(query):
            raise ValueError("fit and query indices must be disjoint")
        if not fit <= universe or not set(query) <= universe:
            raise ValueError("fold index outside TRAIN range")
        seen.extend(query)
    if sorted(seen) != list(range(int(n_rows))):
        raise ValueError("fold query indices must cover every TRAIN row exactly once")


def restore_fold_concatenated(
    values: Iterable[float], plan: FoldPlan, n_rows: int
) -> np.ndarray:
    concatenated = np.asarray(list(values), dtype=float)
    _validate_plan(plan, n_rows)
    if len(concatenated) != int(n_rows):
        raise ValueError("fold-concatenated value count must equal TRAIN row count")
    restored = np.empty(int(n_rows), dtype=float)
    offset = 0
    for _, query_raw in plan:
        query = np.asarray(query_raw, dtype=int).reshape(-1)
        restored[query] = concatenated[offset : offset + len(query)]
        offset += len(query)
    if not np.isfinite(restored).all():
        raise ValueError("restored p-values must be finite")
    return restored


def fold_local_pvalues(scores: Iterable[float], plan: FoldPlan) -> np.ndarray:
    values = np.asarray(list(scores), dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("OOF scores must be a finite vector")
    _validate_plan(plan, len(values))
    output = np.empty(len(values), dtype=float)
    for fit_raw, query_raw in plan:
        fit = np.asarray(fit_raw, dtype=int).reshape(-1)
        query = np.asarray(query_raw, dtype=int).reshape(-1)
        output[query] = empirical_upper_pvalues(values[fit], values[query])
    return output


def conditional_risk_lanes(
    *,
    shape_p: Iterable[float],
    temporal_p: Iterable[float],
    canonical_p: Iterable[float],
    shape_alpha: float,
    confirmation_alpha: float,
) -> dict[str, set[int]]:
    shape = np.asarray(list(shape_p), dtype=float)
    temporal = np.asarray(list(temporal_p), dtype=float)
    canonical = np.asarray(list(canonical_p), dtype=float)
    if shape.ndim != 1 or temporal.shape != shape.shape or canonical.shape != shape.shape:
        raise ValueError("aligned p-value vectors must have the same one-dimensional shape")
    if not all(np.isfinite(array).all() for array in (shape, temporal, canonical)):
        raise ValueError("p-value vectors must be finite")
    if not 0.0 < float(shape_alpha) < 1.0 or not 0.0 < float(confirmation_alpha) < 1.0:
        raise ValueError("alpha must be between zero and one")
    candidates = set(np.flatnonzero(shape <= float(shape_alpha)).astype(int))
    temporal_hits = set(np.flatnonzero(temporal <= float(confirmation_alpha)).astype(int))
    canonical_hits = set(np.flatnonzero(canonical <= float(confirmation_alpha)).astype(int))
    dual = candidates & temporal_hits & canonical_hits
    single = candidates & (temporal_hits ^ canonical_hits)
    unconfirmed = candidates - dual - single
    if (dual & single) or (dual & unconfirmed) or (single & unconfirmed):
        raise AssertionError("conditional risk lanes overlap")
    if dual | single | unconfirmed != candidates:
        raise AssertionError("conditional risk lanes do not partition candidates")
    return {"dual": dual, "single": single, "unconfirmed": unconfirmed}


EXP219_ROOT = Path("/Users/minho/Documents/Dataset/exp215_217_shapelet_stability")
EXP224_ROOT = Path("/Users/minho/Documents/Dataset/exp224_prospective")
EXP215_SEED = 20260716
EXP224_SEED = 20260718


@lru_cache(maxsize=1)
def _frozen_contract() -> dict[str, object]:
    prediction_path = EXP219_ROOT / "exp219_frozen_predictions.jsonl"
    manifest_path = EXP219_ROOT / "exp219_prediction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(prediction_path) != manifest.get("prediction_sha256"):
        raise ValueError("Exp219 prediction SHA-256 mismatch")
    primary = {}
    for line in prediction_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("variant") == "fixed_dictionary_control_conservative_max__0.005":
            primary[str(row["dataset"])] = row
    if len(primary) != 251:
        raise ValueError(f"Exp219 primary coverage mismatch: {len(primary)}/251")

    policy_path = EXP224_ROOT / "policy_bundle.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if int(policy.get("seed", -1)) != EXP224_SEED:
        raise ValueError("Exp224 seed mismatch")
    profiles = {str(row["profile_id"]): row for row in policy.get("profiles", [])}
    if len(profiles) != 639:
        raise ValueError(f"Exp224 profile coverage mismatch: {len(profiles)}/639")
    if not set(primary) <= set(profiles):
        raise ValueError("Exp219 population is outside Exp224 coverage")
    return {
        "primary": primary,
        "profiles": profiles,
        "exp219_input_hashes": manifest.get("input_hashes", {}),
        "exp219_prediction_sha256": manifest["prediction_sha256"],
        "exp224_policy_sha256": sha256_file(policy_path),
    }


def audit_frozen_train_risk(dataset: str) -> dict[str, object]:
    contract = _frozen_contract()
    primary = contract["primary"]
    if dataset not in primary:
        return {"status": "not_in_frozen_exp219_population", "population": 251}
    n_train = int(primary[dataset]["n_train"])
    shape_path = EXP219_ROOT / "exp215_cache" / f"{dataset}.npz"
    expected_shape_hash = contract["exp219_input_hashes"].get(f"exp215_cache/{dataset}.npz")
    if not expected_shape_hash or sha256_file(shape_path) != expected_shape_hash:
        raise ValueError(f"Exp215 cache SHA-256 mismatch: {dataset}")
    profile = contract["profiles"][dataset]
    profile_path = Path(profile["artifact_path"])
    if sha256_file(profile_path) != profile["artifact_sha256"]:
        raise ValueError(f"Exp224 artifact SHA-256 mismatch: {dataset}")
    if n_train != int(profile["n_train"]):
        raise ValueError(f"Exp215/Exp224 TRAIN length mismatch: {dataset}")

    with np.load(shape_path, allow_pickle=False) as archive:
        shape_concatenated = np.asarray(archive["normal_pvalues"], dtype=float)
    with np.load(profile_path, allow_pickle=False) as archive:
        temporal_oof = np.asarray(archive["temporal_oof"], dtype=float)
        canonical_oof = np.asarray(archive["canonical_oof"], dtype=float)
    if len(shape_concatenated) != n_train or len(temporal_oof) != n_train or len(canonical_oof) != n_train:
        raise ValueError(f"Exp226 TRAIN vector length mismatch: {dataset}")

    shape_p = restore_fold_concatenated(
        shape_concatenated, fold_plan(n_train, 5, EXP215_SEED), n_train
    )
    confirmation_plan = _fold_plan(n_train, EXP224_SEED)
    temporal_p = fold_local_pvalues(temporal_oof, confirmation_plan)
    canonical_p = fold_local_pvalues(canonical_oof, confirmation_plan)
    lanes = conditional_risk_lanes(
        shape_p=shape_p,
        temporal_p=temporal_p,
        canonical_p=canonical_p,
        shape_alpha=0.005,
        confirmation_alpha=0.01,
    )
    candidate_count = sum(len(values) for values in lanes.values())
    return {
        "status": "completed_train_normal_risk_audit",
        "n_train_normal": n_train,
        "shape_alpha": 0.005,
        "confirmation_alpha": 0.01,
        "shape_candidate_count": candidate_count,
        "dual_count": len(lanes["dual"]),
        "single_count": len(lanes["single"]),
        "unconfirmed_count": len(lanes["unconfirmed"]),
        "workload_per_100_normal_runs": 100.0 * candidate_count / max(1, n_train),
        "contains_test_labels": False,
        "metric_scope": "train_normal_conditional_risk_audit",
        "exp219_prediction_sha256": contract["exp219_prediction_sha256"],
        "exp224_policy_sha256": contract["exp224_policy_sha256"],
    }
