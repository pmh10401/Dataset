"""One-shot approved Exp473 FN-focus extension diagnostic (observational)."""

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
    "/Users/mac/Documents/Dataset/exp473_fn_focus_extension_antigravity_20260825"
)
_DATABASE_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
_TARGET_DATASETS_ANTIGRAVITY = (
    "ElectricDevices_normal_5",
    "FordA_normal_-1",
    "FordB_normal_-1",
    "ElectricDevices_normal_4",
    "FordA_normal_1",
    "StarLightCurves_normal_3",
    "ChlorineConcentration_normal_3",
    "Yoga_normal_2",
    "Yoga_normal_1",
    "FordB_normal_1",
)
_SEEDS_ANTIGRAVITY = (20260721, 20261721)
_REQUIRED_TOKEN_ANTIGRAVITY = "EXP473_FN_FOCUS_EXTENSION_APPROVED"


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


def _load_dataset_antigravity(connection, dataset: str):
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
    return train_raw, test_raw, anomaly_mask


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

    from experiments.exp449_residual_fn_method_qualification_antigravity.representations_antigravity import (
        make_representation_antigravity,
    )
    from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import (
        LocalGapModelAntigravity,
    )
    from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.rank_auc_antigravity import (
        mann_whitney_auc_antigravity,
    )

    connection = sqlite3.connect(f"file:{_DATABASE_ANTIGRAVITY}?mode=ro", uri=True)
    datasets = {}
    try:
        for name in _TARGET_DATASETS_ANTIGRAVITY:
            datasets[name] = _load_dataset_antigravity(connection, name)
    finally:
        connection.close()

    row_payloads: list[dict[str, object]] = []
    summary: dict[str, object] = {"schema": "exp473_fn_focus_extension_v1", "seeds": {}}

    for seed in _SEEDS_ANTIGRAVITY:
        per_seed: dict[str, object] = {}
        for name, (train_raw, test_raw, mask) in datasets.items():
            rep = make_representation_antigravity("hydra_local_gap", seed)
            ref_features = np.asarray(rep.fit_transform(train_raw.astype(np.float32)), dtype=np.float64)
            gap_model = LocalGapModelAntigravity(ref_features, neighbors=3)

            train_scores = np.asarray(gap_model.score(ref_features), dtype=np.float64)
            test_scores = np.asarray(gap_model.score(np.asarray(rep.transform(test_raw.astype(np.float32)), dtype=np.float64)), dtype=np.float64)

            normal_max = float(test_scores[~mask].max())
            zero_fp_tp = int(np.count_nonzero(test_scores[mask] > normal_max))
            sorted_norm = np.sort(test_scores[~mask])[::-1]
            tp_budget = {}
            for fp_b in (3, 5, 10):
                threshold = float(sorted_norm[min(fp_b - 1, len(sorted_norm) - 1)])
                tp_budget[str(fp_b)] = int(np.count_nonzero(test_scores[mask] >= threshold))

            auc = float(mann_whitney_auc_antigravity(test_scores[~mask], test_scores[mask]))
            per_seed[name] = {
                "auc": auc,
                "zero_fp_tp": zero_fp_tp,
                "tp_at_budget": tp_budget,
                "anomaly_count": int(mask.sum()),
                "normal_count": int((~mask).sum()),
                "train_count": len(train_raw),
                "width": int(train_raw.shape[1]),
            }

            for index in range(len(test_raw)):
                row_payloads.append({
                    "dataset": name,
                    "instance_index": index,
                    "label_anomaly": bool(mask[index]),
                    "seed": seed,
                    "raw_gap": float(test_scores[index]),
                })
        summary["seeds"][str(seed)] = per_seed

    _OUTPUT_ROOT_ANTIGRAVITY.mkdir(parents=True)
    rows_bytes = b"".join(_canonical_json_bytes_antigravity(row) for row in row_payloads)
    (_OUTPUT_ROOT_ANTIGRAVITY / "scores_antigravity.jsonl").write_bytes(rows_bytes)
    manifest = {
        "approval_token_name": _REQUIRED_TOKEN_ANTIGRAVITY,
        "label_query_for_routing_or_promotion": 0,
        "protocol": (
            "Observational Exp473 FN-focus extension. Per-dataset majority-class "
            "TRAIN is used with raw LocalGap distance (no rank transform). "
            "Top-10 FN datasets from Exp448 authority are scored. Routing/"
            "champion/promotion unchanged."
        ),
        "reviewed_commit": reviewed,
        "runtime_python": platform.python_version(),
        "schema": "exp473_fn_focus_extension_manifest_v1",
    }
    (_OUTPUT_ROOT_ANTIGRAVITY / "manifest_antigravity.json").write_bytes(
        _canonical_json_bytes_antigravity(manifest)
    )
    summary_bytes = _canonical_json_bytes_antigravity(summary)
    (_OUTPUT_ROOT_ANTIGRAVITY / "summary_antigravity.json").write_bytes(summary_bytes)
    hashes = {name: hashlib.sha256((_OUTPUT_ROOT_ANTIGRAVITY / name).read_bytes()).hexdigest() for name in (
        "manifest_antigravity.json", "summary_antigravity.json", "scores_antigravity.jsonl")}
    (_OUTPUT_ROOT_ANTIGRAVITY / "artifact_sha256_antigravity.json").write_bytes(
        _canonical_json_bytes_antigravity({"hashes": hashes, "schema": "exp473_hashes_v1"})
    )
    sys.stdout.write(summary_bytes.decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_antigravity())
