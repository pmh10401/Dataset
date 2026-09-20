"""One-shot approved Exp472 zero-FP tail recovery diagnostic (observational)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import sqlite3
import subprocess
import sys

import numpy as np


_COMMIT_RE_ANTIGRAVITY = re.compile(r"[0-9a-f]{40}\Z")
_REPOSITORY_ROOT_ANTIGRAVITY = Path(__file__).resolve().parents[1]
_OUTPUT_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp472_zero_fp_tail_recovery_antigravity_20260825"
)
_DATABASE_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
_TEST_FILES_ANTIGRAVITY = ("ECG200_normal_1", "ECG200_normal_-1")
_SEEDS_ANTIGRAVITY = (20260721, 20261721)
_REQUIRED_TOKEN_ANTIGRAVITY = "EXP472_ZERO_FP_TAIL_RECOVERY_APPROVED"


def _canonical_json_bytes_antigravity(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True,
                   separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _head_commit_antigravity() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=_REPOSITORY_ROOT_ANTIGRAVITY,
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _require_clean_head_antigravity(reviewed: str) -> None:
    if _COMMIT_RE_ANTIGRAVITY.fullmatch(reviewed) is None:
        raise SystemExit("reviewed commit must be 40 hex characters")
    if _head_commit_antigravity() != reviewed:
        raise SystemExit("reviewed commit differs from current HEAD")
    dirty = subprocess.run(
        ("git", "status", "--porcelain"), cwd=_REPOSITORY_ROOT_ANTIGRAVITY,
        check=True, capture_output=True, text=True,
    ).stdout
    if dirty:
        raise SystemExit("diagnostic requires a clean worktree")


def _load_dataset_train_and_test_antigravity(
    connection: object, dataset: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load per-dataset TRAIN label-0 rows and TEST rows (values + anomaly_mask)."""
    did = connection.execute("SELECT id FROM datasets WHERE name = ?", (dataset,)).fetchone()[0]
    train_rows = connection.execute(
        "SELECT values_blob FROM instances WHERE dataset_id=? AND split='TRAIN' AND label='0' ORDER BY instance_index",
        (did,),
    ).fetchall()
    test_rows = connection.execute(
        "SELECT label, values_blob FROM instances WHERE dataset_id=? AND split='TEST' ORDER BY instance_index",
        (did,),
    ).fetchall()
    train_raw = np.stack([np.frombuffer(row[0], dtype="<f4") for row in train_rows]).astype(np.float64, order="C")
    test_raw = np.stack([np.frombuffer(row[1], dtype="<f4") for row in test_rows]).astype(np.float64, order="C")
    labels = [int(row[0]) for row in test_rows]
    anomaly_mask = np.asarray(labels, dtype=np.bool_)
    sys.path.insert(0, str(_REPOSITORY_ROOT_ANTIGRAVITY))
    from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.interpolation_antigravity import (
        interpolate_to_140_antigravity,
    )
    return interpolate_to_140_antigravity(train_raw), interpolate_to_140_antigravity(test_raw), anomaly_mask


def main_antigravity(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--approval-token", required=True)
    args = parser.parse_args(argv)
    if args.approval_token != _REQUIRED_TOKEN_ANTIGRAVITY:
        raise SystemExit("approval token rejected")

    reviewed = _head_commit_antigravity()
    _require_clean_head_antigravity(reviewed)
    if _OUTPUT_ROOT_ANTIGRAVITY.exists() or _OUTPUT_ROOT_ANTIGRAVITY.is_symlink():
        raise SystemExit("output root must be absent before the approved run")
    sys.path.insert(0, str(_REPOSITORY_ROOT_ANTIGRAVITY))

    from experiments.exp457_ecg_train_synthetic_confidence_routing_antigravity.qualification_antigravity import (
        _score_fold_antigravity as raw_local_gap_score_antigravity,
    )
    from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.rank_auc_antigravity import (
        mann_whitney_auc_antigravity,
    )
    from experiments.exp470_ecg200_spectral_entropy_numba_antigravity.spectral_entropy_antigravity import (
        score_spectral_entropy_numba_antigravity,
    )

    connection = sqlite3.connect(f"file:{_DATABASE_ANTIGRAVITY}?mode=ro", uri=True)
    datasets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    try:
        for name in _TEST_FILES_ANTIGRAVITY:
            datasets[name] = _load_dataset_train_and_test_antigravity(connection, name)
    finally:
        connection.close()

    row_payloads: list[dict[str, object]] = []
    summary: dict[str, object] = {"schema": "exp472_zero_fp_tail_recovery_v1", "seeds": {}}

    def auc_safe(normal: np.ndarray, anomalous: np.ndarray) -> float:
        return float(mann_whitney_auc_antigravity(normal, anomalous))

    def tp_at_thresholds(scores: np.ndarray, mask: np.ndarray) -> dict[str, object]:
        normal_max = float(scores[~mask].max())
        zero_fp = int(np.count_nonzero(scores[mask] > normal_max))
        budget = {}
        for fp in (3, 5, 10):
            sorted_norm = np.sort(scores[~mask])[::-1]
            threshold = float(sorted_norm[min(fp, len(sorted_norm)-1)])
            budget[str(fp)] = int(np.count_nonzero(scores[mask] >= threshold))
        return {"zero_fp_tp": zero_fp, "tp_at_budget": budget}

    for seed in _SEEDS_ANTIGRAVITY:
        per_seed: dict[str, object] = {}
        for name, (train140, test140, mask) in datasets.items():
            n_train = len(train140)
            fit_idx = np.arange(n_train, dtype=np.int64)
            query_idx = np.arange(n_train, dtype=np.int64)
            # Baseline C0 (ranked): use full chain scorer on combined matrix.
            from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.predecessor_antigravity import (
                capture_exp465_authority_antigravity,
                score_c0_fold_antigravity,
            )
            authority = capture_exp465_authority_antigravity()
            combined = np.vstack((train140, test140))
            c0_result = score_c0_fold_antigravity(
                combined,
                fit_idx,
                np.arange(n_train, n_train + len(test140), dtype=np.int64),
                seed=seed, fold_index=0, authority=authority,
            )
            baseline_c0 = np.asarray(c0_result.normal_scores, dtype=np.float64)

            # Approach A: raw LocalGap (unbounded).
            fold_a = raw_local_gap_score_antigravity(
                train140.astype(np.float32),
                fit_idx, query_idx, seed=seed, fold_index=0,
            )
            raw_gap_train = np.asarray(fold_a.original, dtype=np.float64)
            model_a = raw_local_gap_score_antigravity.__wrapped__ if hasattr(raw_local_gap_score_antigravity, "__wrapped__") else None
            # Score test rows using the same representation/model via direct call.
            from experiments.exp449_residual_fn_method_qualification_antigravity.representations_antigravity import (
                make_representation_antigravity,
            )
            from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import (
                LocalGapModelAntigravity,
            )
            rep = make_representation_antigravity("hydra_local_gap", seed)
            reference_indices = fit_idx  # all TRAIN rows are the reference
            ref_features = np.asarray(rep.fit_transform(train140[reference_indices].astype(np.float32)), dtype=np.float64)
            gap_model = LocalGapModelAntigravity(ref_features, neighbors=3)
            raw_gap_test = np.asarray(gap_model.score(np.asarray(rep.transform(test140.astype(np.float32)), dtype=np.float64)), dtype=np.float64)

            # Approach B: relaxed confirmation thresholds on ACF peak-lag.
            from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.acf_peak_lag_antigravity import (
                score_acf_peak_lag_numba_antigravity,
            )
            peaks = score_acf_peak_lag_numba_antigravity(train140, test140)
            e12_ranks = np.asarray(peaks.query_ranks, dtype=np.float64)
            confirmed_p95 = e12_ranks > 0.95  # proxy for percentile-based confirmation
            confirmed_p90 = e12_ranks > 0.90
            c12_p95 = np.where(confirmed_p95 & (e12_ranks > baseline_c0), e12_ranks, baseline_c0)
            c12_p90 = np.where(confirmed_p90 & (e12_ranks > baseline_c0), e12_ranks, baseline_c0)

            # Approach C: combine raw gap with spectral entropy.
            ent = score_spectral_entropy_numba_antigravity(train140, test140)
            entropy_scores = np.asarray(ent.query_ranks, dtype=np.float64)
            combined_ab = np.maximum(raw_gap_test / max(raw_gap_test[~mask].max(), 1e-9), entropy_scores)

            methods = {
                "baseline_c0": baseline_c0,
                "A_raw_gap": raw_gap_test,
                "B_c12_p90": c12_p90,
                "B_c12_p95": c12_p95,
                "C_gap_plus_entropy": combined_ab,
            }
            method_summary: dict[str, object] = {}
            for method_name, scores in methods.items():
                s = np.asarray(scores, dtype=np.float64)
                stats = tp_at_thresholds(s, mask)
                method_summary[method_name] = {
                    "auc": auc_safe(s[~mask], s[mask]),
                    **stats,
                }
            per_seed[name] = method_summary

            for index in range(len(test140)):
                row_payloads.append({
                    "dataset_file": name,
                    "instance_index": index,
                    "label_anomaly": bool(mask[index]),
                    "seed": seed,
                    "baseline_c0": float(baseline_c0[index]),
                    "raw_gap": float(raw_gap_test[index]),
                    "c12_p90": float(c12_p90[index]),
                    "c12_p95": float(c12_p95[index]),
                    "entropy_rank": float(entropy_scores[index]),
                })
        summary["seeds"][str(seed)] = per_seed

    _OUTPUT_ROOT_ANTIGRAVITY.mkdir(parents=True)
    rows_bytes = b"".join(_canonical_json_bytes_antigravity(row) for row in row_payloads)
    (_OUTPUT_ROOT_ANTIGRAVITY / "scores_antigravity.jsonl").write_bytes(rows_bytes)
    manifest = {
        "approval_token_name": _REQUIRED_TOKEN_ANTIGRAVITY,
        "label_query_for_routing_or_promotion": 0,
        "protocol": (
            "Observational Exp472 zero-FP tail recovery diagnostic. Per dataset "
            "majority-class TRAIN is used. Four scoring approaches compared "
            "against baseline ranked C0. Routing/champion/promotion unchanged."
        ),
        "reviewed_commit": reviewed,
        "runtime_python": platform.python_version(),
        "schema": "exp472_zero_fp_tail_recovery_manifest_v1",
    }
    (_OUTPUT_ROOT_ANTIGRAVITY / "manifest_antigravity.json").write_bytes(
        _canonical_json_bytes_antigravity(manifest)
    )
    summary_bytes = _canonical_json_bytes_antigravity(summary)
    (_OUTPUT_ROOT_ANTIGRAVITY / "summary_antigravity.json").write_bytes(summary_bytes)
    hashes = {name: hashlib.sha256((_OUTPUT_ROOT_ANTIGRAVITY / name).read_bytes()).hexdigest() for name in (
        "manifest_antigravity.json", "summary_antigravity.json", "scores_antigravity.jsonl")}
    (_OUTPUT_ROOT_ANTIGRAVITY / "artifact_sha256_antigravity.json").write_bytes(
        _canonical_json_bytes_antigravity({"hashes": hashes, "schema": "exp472_hashes_v1"})
    )
    sys.stdout.write(summary_bytes.decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
