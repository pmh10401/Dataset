"""One-shot approved Exp471 TEST/FN diagnostic (observational, no routing change)."""

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
_OFFICIAL_TRAIN_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "exp471_ecg200_acf_peak_lag_shift_numba_antigravity_20260825"
)
_OUTPUT_ROOT_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/"
    "exp471_ecg200_test_fn_diagnostic_antigravity_20260825"
)
_DATABASE_ANTIGRAVITY = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
_TEST_FILES_ANTIGRAVITY = ("ECG200_normal_1", "ECG200_normal_-1")
_EXPECTED_TEST_ROWS_ANTIGRAVITY = {"ECG200_normal_1": 100, "ECG200_normal_-1": 50}
_SEEDS_ANTIGRAVITY = (20260721, 20261721)
_TRAIN_PATH_ANTIGRAVITY = Path(
    "/Users/mac/Documents/Dataset/exp465_ecg200_train_antigravity/"
    "ECG200_TRAIN_antigravity.ts"
)
_REQUIRED_TOKEN_ANTIGRAVITY = "TASK6_TEST_FN_DIAGNOSTIC_APPROVED"


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


def _require_clean_head_antigravity(reviewed_commit: str) -> None:
    if _COMMIT_RE_ANTIGRAVITY.fullmatch(reviewed_commit) is None:
        raise SystemExit("reviewed commit must be 40 lowercase hex characters")
    if _head_commit_antigravity() != reviewed_commit:
        raise SystemExit("reviewed commit differs from current HEAD")
    dirty = subprocess.run(
        ("git", "status", "--porcelain"), cwd=_REPOSITORY_ROOT_ANTIGRAVITY,
        check=True, capture_output=True, text=True,
    ).stdout
    if dirty:
        raise SystemExit("diagnostic requires a clean worktree")


def _load_shared_train_antigravity() -> tuple[np.ndarray, str]:
    context_path = _OFFICIAL_TRAIN_ROOT_ANTIGRAVITY / "context_antigravity.json"
    payload = json.loads(context_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "exp471_context_v1":
        raise SystemExit("official context schema drifted")
    train = np.asarray(payload["shared_train140"], dtype=np.float64)
    if train.shape != (69, 140) or not np.isfinite(train).all():
        raise SystemExit("shared train140 has an invalid shape")

    sys.path.insert(0, str(_REPOSITORY_ROOT_ANTIGRAVITY))
    from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.sources_antigravity import (
        load_ecg200_normal_train_antigravity,
    )
    from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.interpolation_antigravity import (
        interpolate_to_140_antigravity,
    )

    rebuilt = interpolate_to_140_antigravity(
        load_ecg200_normal_train_antigravity(_TRAIN_PATH_ANTIGRAVITY)
    )
    if not np.array_equal(rebuilt, train):
        raise SystemExit("shared train140 differs from the frozen loader path")
    digest = hashlib.sha256(train.tobytes(order="C")).hexdigest()
    train.setflags(write=False)
    return train, digest


def _load_oof_antigravity() -> dict[int, np.ndarray]:
    context_path = _OFFICIAL_TRAIN_ROOT_ANTIGRAVITY / "context_antigravity.json"
    payload = json.loads(context_path.read_text(encoding="utf-8"))
    oof_by_seed: dict[int, np.ndarray] = {}
    for seed in _SEEDS_ANTIGRAVITY:
        parts = [
            np.asarray(record["c0_normal"], dtype=np.float64)
            for record in sorted(
                (row for row in payload["records"] if row["seed"] == seed),
                key=lambda row: row["fold_index"],
            )
        ]
        oof = np.concatenate(parts)
        if oof.shape != (69,) or not np.isfinite(oof).all():
            raise SystemExit(f"OOF pooling failed for seed {seed}")
        oof_by_seed[seed] = oof
    return oof_by_seed


def _load_test_rows_antigravity() -> dict[str, dict[str, object]]:
    sys.path.insert(0, str(_REPOSITORY_ROOT_ANTIGRAVITY))
    from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.interpolation_antigravity import (
        interpolate_to_140_antigravity,
    )

    connection = sqlite3.connect(f"file:{_DATABASE_ANTIGRAVITY}?mode=ro", uri=True)
    try:
        result: dict[str, dict[str, object]] = {}
        for name in _TEST_FILES_ANTIGRAVITY:
            rows = connection.execute(
                "SELECT i.instance_index, i.label, i.values_blob "
                "FROM instances AS i JOIN datasets AS d ON d.id = i.dataset_id "
                "WHERE d.name = ? AND i.split = 'TEST' ORDER BY i.instance_index",
                (name,),
            ).fetchall()
            if len(rows) != _EXPECTED_TEST_ROWS_ANTIGRAVITY[name]:
                raise SystemExit(f"{name}: unexpected TEST row count")
            if tuple(row[0] for row in rows) != tuple(range(len(rows))):
                raise SystemExit(f"{name}: TEST indices must be contiguous")
            labels = tuple(str(row[1]) for row in rows)
            if any(label not in {"0", "1"} for label in labels):
                raise SystemExit(f"{name}: TEST labels must be 0/1")
            decoded = [np.frombuffer(row[2], dtype="<f4") for row in rows]
            if any(values.size != 96 for values in decoded):
                raise SystemExit(f"{name}: TEST width must be 96")
            raw = np.stack(decoded).astype(np.float64, order="C")
            if not np.isfinite(raw).all():
                raise SystemExit(f"{name}: TEST rows must be finite")
            interpolated = interpolate_to_140_antigravity(raw)
            anomaly_mask = np.asarray([label == "1" for label in labels], dtype=np.bool_)
            result[name] = {
                "values": interpolated,
                "anomaly_mask": anomaly_mask,
            }
        return result
    finally:
        connection.close()


def main_antigravity(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--approval-token", required=True)
    args = parser.parse_args(argv)
    if args.approval_token != _REQUIRED_TOKEN_ANTIGRAVITY:
        raise SystemExit("diagnostic approval token rejected")

    reviewed = _head_commit_antigravity()
    _require_clean_head_antigravity(reviewed)
    if _OUTPUT_ROOT_ANTIGRAVITY.exists() or _OUTPUT_ROOT_ANTIGRAVITY.is_symlink():
        raise SystemExit("output root must be absent before the approved diagnostic")
    sys.path.insert(0, str(_REPOSITORY_ROOT_ANTIGRAVITY))

    from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import (
        LocalGapModelAntigravity,
    )
    from experiments.exp449_residual_fn_method_qualification_antigravity.representations_antigravity import (
        make_representation_antigravity,
    )
    from experiments.exp458_ecg_normal2_rank_confidence_routing_antigravity.rank_auc_antigravity import (
        mann_whitney_auc_antigravity,
    )
    from experiments.exp471_ecg200_acf_peak_lag_shift_numba_antigravity.acf_peak_lag_antigravity import (
        score_acf_peak_lag_numba_antigravity,
    )

    train140, train_sha = _load_shared_train_antigravity()
    test_data = _load_test_rows_antigravity()
    oof_by_seed = _load_oof_antigravity()

    names = list(_TEST_FILES_ANTIGRAVITY)
    row_payloads: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "schema": "exp471_test_fn_diagnostic_v1",
        "seeds": {},
    }
    for seed in _SEEDS_ANTIGRAVITY:
        representation = make_representation_antigravity("hydra_local_gap", seed)
        reference_features = representation.fit_transform(train140)
        model = LocalGapModelAntigravity(reference_features, neighbors=3)
        oof = oof_by_seed[seed]
        per_seed: dict[str, object] = {"oof_auc_c0_vs_perturbations": {}}
        for name in names:
            values = test_data[name]["values"]
            mask = test_data[name]["anomaly_mask"]
            features = representation.transform(values)
            c0 = np.asarray(model.score(features), dtype=np.float64)
            peaks = score_acf_peak_lag_numba_antigravity(train140, values)
            ranks = np.asarray(peaks.query_ranks, dtype=np.float64)
            confirmed = np.asarray(peaks.confirmation_mask, dtype=np.bool_)
            c12 = np.where(confirmed & (ranks > c0), ranks, c0)

            auc_c0 = mann_whitney_auc_antigravity(c0[~mask], c0[mask])
            auc_e12 = mann_whitney_auc_antigravity(ranks[~mask], ranks[mask])
            auc_c12 = mann_whitney_auc_antigravity(c12[~mask], c12[mask])
            tp0_c0 = int(np.count_nonzero(c0[mask] > c0[~mask].max()))
            tp0_e12 = int(np.count_nonzero(ranks[mask] > ranks[~mask].max()))
            tp0_c12 = int(np.count_nonzero(c12[mask] > c12[~mask].max()))

            per_seed[name] = {
                "auc_c0": auc_c0,
                "auc_e12_rank": auc_e12,
                "auc_c12": auc_c12,
                "tp_at_zero_fp_c0": tp0_c0,
                "tp_at_zero_fp_e12_rank": tp0_e12,
                "tp_at_zero_fp_c12": tp0_c12,
                "anomaly_count": int(mask.sum()),
                "normal_count": int((~mask).sum()),
                "confirmed_normal": int(np.count_nonzero(confirmed[~mask])),
                "confirmed_anomaly": int(np.count_nonzero(confirmed[mask])),
            }
            for index in range(len(values)):
                row_payloads.append({
                    "dataset_file": name,
                    "instance_index": index,
                    "label_anomaly": bool(mask[index]),
                    "seed": seed,
                    "c0": float(c0[index]),
                    "e12_rank": float(ranks[index]),
                    "confirmed": bool(confirmed[index]),
                    "c12": float(c12[index]),
                })
        summary["seeds"][str(seed)] = per_seed

    _OUTPUT_ROOT_ANTIGRAVITY.mkdir(parents=True)
    rows_bytes = b"".join(_canonical_json_bytes_antigravity(row) for row in row_payloads)
    (_OUTPUT_ROOT_ANTIGRAVITY / "test_scores_antigravity.jsonl").write_bytes(rows_bytes)
    manifest = {
        "approval_token_name": _REQUIRED_TOKEN_ANTIGRAVITY,
        "label_query_for_routing_or_promotion": 0,
        "official_train_root": str(_OFFICIAL_TRAIN_ROOT_ANTIGRAVITY),
        "protocol": (
            "Observational TEST/FN diagnostic. Frozen train140 reused from the "
            "approved official Exp471 root. Per frozen seed, hydra_local_gap is "
            "refitted on all 69 TRAIN normals (full-reference deployment "
            "convention) and scores the 150 interpolated ECG200 TEST rows. E12/C12 "
            "reuse the frozen peak-lag machinery with all 69 TRAIN rows as fit "
            "evidence. AUC and zero-FP true-positive counts are reported per file "
            "and seed. Routing, champion, promotion, and pending-candidate state "
            "remain unchanged."
        ),
        "reviewed_commit": reviewed,
        "runtime_python": platform.python_version(),
        "schema": "exp471_test_fn_diagnostic_manifest_v1",
        "train140_sha256": train_sha,
    }
    (_OUTPUT_ROOT_ANTIGRAVITY / "manifest_antigravity.json").write_bytes(
        _canonical_json_bytes_antigravity(manifest)
    )
    summary_bytes = _canonical_json_bytes_antigravity(summary)
    (_OUTPUT_ROOT_ANTIGRAVITY / "summary_antigravity.json").write_bytes(summary_bytes)
    hashes = {
        name: hashlib.sha256(
            (_OUTPUT_ROOT_ANTIGRAVITY / name).read_bytes()
        ).hexdigest()
        for name in (
            "manifest_antigravity.json",
            "summary_antigravity.json",
            "test_scores_antigravity.jsonl",
        )
    }
    (_OUTPUT_ROOT_ANTIGRAVITY / "artifact_sha256_antigravity.json").write_bytes(
        _canonical_json_bytes_antigravity(
            {"hashes": hashes, "schema": "exp471_test_fn_diagnostic_hashes_v1"}
        )
    )
    sys.stdout.write(summary_bytes.decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main_antigravity())

