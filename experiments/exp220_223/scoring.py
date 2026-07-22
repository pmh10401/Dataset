from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from experiments.exp177_185 import core as canonical_core
from experiments.exp186_200.core import empirical_upper_tail_pvalues
from experiments.exp220_223.prediction import EXP137_FIELDS, FORBIDDEN_PREDICTION_FIELDS
from experiments.exp220_223.policy import sha256_file


SOURCE_CONFIGS = {
    "temporal_random_kernel": "rocket_exp40",
    "spectral_image": "exp55_best",
    "texture_image": "exp56_best",
    "interval_statistical": "exp84",
}


def run_source_crossfit(bridge, source_name, x_train, x_query, record):
    """Call the scorer from its owning Exp152 module exposed by the bridge."""
    return bridge.exp152.crossfit_source(source_name, x_train, x_query, record)


def align_to_train_length(values: np.ndarray, target_length: int) -> np.ndarray:
    series = np.asarray(values, dtype=np.float32).reshape(-1)
    target_length = int(target_length)
    if target_length <= 0:
        raise ValueError("TRAIN target length must be positive")
    if len(series) == 0 or not np.all(np.isfinite(series)):
        raise ValueError("query values must be non-empty and finite")
    if len(series) == target_length:
        return series.copy()
    if len(series) == 1:
        return np.full(target_length, series[0], dtype=np.float32)
    source_x = np.linspace(0.0, 1.0, len(series), dtype=float)
    target_x = np.linspace(0.0, 1.0, target_length, dtype=float)
    return np.interp(target_x, source_x, series).astype(np.float32)


def exp195_zero_train_guard(
    source_train_oof: Mapping[str, np.ndarray],
    canonical_train_oof: np.ndarray,
    *,
    conservative_alpha: float,
    seed: int = 20260718,
) -> bool:
    if not source_train_oof:
        raise ValueError("source TRAIN OOF scores are required")
    lengths = {len(np.asarray(values)) for values in source_train_oof.values()}
    lengths.add(len(np.asarray(canonical_train_oof)))
    if len(lengths) != 1:
        raise ValueError("TRAIN OOF score lengths differ")
    n_train = lengths.pop()
    if n_train < 5:
        return False
    source_values = {
        name: np.asarray(values, dtype=float) for name, values in source_train_oof.items()
    }
    canonical = np.asarray(canonical_train_oof, dtype=float)
    candidate_count = 0
    for fit, query in canonical_core._fold_plan(n_train, seed=seed):
        for train_index in query:
            support = sum(
                empirical_upper_tail_pvalues(
                    scores[fit], np.asarray([scores[train_index]], dtype=float)
                )[0]
                <= conservative_alpha
                for scores in source_values.values()
            )
            canonical_p = empirical_upper_tail_pvalues(
                canonical[fit], np.asarray([canonical[train_index]], dtype=float)
            )[0]
            if support >= 2 and canonical_p <= 0.005:
                candidate_count += 1
    return candidate_count == 0


def _load_train_matrix(
    db_path: Path, profile_id: str, target_length: int
) -> tuple[np.ndarray, dict[str, object]]:
    with sqlite3.connect(Path(db_path)) as connection:
        rows = connection.execute(
            """
            SELECT i.values_blob
            FROM instances i
            JOIN datasets d ON i.dataset_id = d.id
            WHERE d.name = ? AND i.split = 'TRAIN'
            ORDER BY i.instance_index
            """,
            (profile_id,),
        ).fetchall()
    if not rows:
        raise ValueError(f"TRAIN profile not found: {profile_id}")
    raw = [np.frombuffer(row[0], dtype=np.float32).copy() for row in rows]
    train = np.vstack([align_to_train_length(values, target_length) for values in raw])
    record = {
        "dataset_name": "__prospective_profile__",
        "family": "__family_neutral__",
        "metadata_len": target_length,
        "actual_len_median": float(target_length),
        "actual_len_max": float(max(len(values) for values in raw)),
        "test_actual_len_median": float(target_length),
        "train_series": raw,
        "test_series": [],
        "train_target_len": target_length,
    }
    return train.astype(np.float32), record


def score_profile(
    *,
    train_matrix: np.ndarray,
    query_matrix: np.ndarray,
    scoring_record: Mapping[str, object],
    conservative_alpha: float,
) -> tuple[list[dict[str, object]], bool]:
    from experiments.exp153_157 import exp153_157_bridge as bridge

    source_query_p: dict[str, np.ndarray] = {}
    source_train: dict[str, np.ndarray] = {}
    for public_name, source_name in SOURCE_CONFIGS.items():
        result = run_source_crossfit(
            bridge,
            source_name,
            np.asarray(train_matrix, dtype=np.float32),
            np.asarray(query_matrix, dtype=np.float32),
            dict(scoring_record),
        )
        train_scores = np.asarray(result["train_oof_scores"], dtype=float)
        query_scores = np.asarray(result["test_scores"], dtype=float)
        source_train[public_name] = train_scores
        source_query_p[public_name] = empirical_upper_tail_pvalues(
            train_scores, query_scores
        )

    train_features = canonical_core.canonical_features(train_matrix)
    query_features = canonical_core.canonical_features(query_matrix)
    canonical_train, canonical_query = canonical_core.crossfit_scores(
        train_features,
        query_features,
        backend="knn_local_gap",
        seed=canonical_core.SEED,
    )
    canonical_p = empirical_upper_tail_pvalues(canonical_train, canonical_query)
    guard = exp195_zero_train_guard(
        source_train,
        canonical_train,
        conservative_alpha=conservative_alpha,
    )
    evidence = [
        {
            "source_pvalues": {
                name: float(source_query_p[name][index]) for name in SOURCE_CONFIGS
            },
            "canonical_knn_p": float(canonical_p[index]),
            "exp195_zero_train_guard": guard,
        }
        for index in range(len(query_matrix))
    ]
    return evidence, guard


def _read_exp137_upstream(path: Path, batch_id: str) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            forbidden = FORBIDDEN_PREDICTION_FIELDS & {
                str(key).lower() for key in row
            }
            if forbidden:
                raise ValueError(
                    f"evaluation fields in Exp137 upstream line={line_number}: {sorted(forbidden)}"
                )
            if row.get("batch_id") != batch_id:
                continue
            instance_id = str(row.get("instance_id", ""))
            if not instance_id or instance_id in output:
                raise ValueError(f"invalid/duplicate Exp137 instance_id: {instance_id}")
            if not all(field in row for field in EXP137_FIELDS):
                raise ValueError(f"Exp137 upstream lanes missing: instance_id={instance_id}")
            if sum(bool(row[field]) for field in EXP137_FIELDS) > 1:
                raise ValueError(f"Exp137 upstream lanes overlap: instance_id={instance_id}")
            output[instance_id] = row
    return output


def _verify_exp137_manifest(
    prediction_path: Path, manifest_path: Path, batch_id: str
) -> None:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("batch_id") != batch_id:
        raise ValueError("Exp137 upstream manifest batch mismatch")
    if manifest.get("contains_test_labels") is not False:
        raise ValueError("Exp137 upstream manifest must declare contains_test_labels=false")
    actual = sha256_file(prediction_path)
    if actual != manifest.get("prediction_sha256"):
        raise ValueError(
            "Exp137 upstream hash mismatch: "
            f"expected={manifest.get('prediction_sha256')} actual={actual}"
        )


class RawEvidenceProvider:
    """TRAIN-only raw scorer plus a mandatory frozen Exp137 upstream adapter."""

    def __init__(
        self,
        *,
        train_db_path: Path,
        exp137_predictions_path: Path,
        exp137_manifest_path: Path,
        batch_id: str,
    ) -> None:
        self.train_db_path = Path(train_db_path)
        self.exp137_predictions_path = Path(exp137_predictions_path)
        self.exp137_manifest_path = Path(exp137_manifest_path)
        self.batch_id = str(batch_id)

    def __call__(
        self,
        instances: list[dict[str, object]],
        policy: Mapping[str, object],
    ) -> dict[str, dict[str, object]]:
        _verify_exp137_manifest(
            self.exp137_predictions_path, self.exp137_manifest_path, self.batch_id
        )
        upstream = _read_exp137_upstream(
            self.exp137_predictions_path, self.batch_id
        )
        ids = {str(row["instance_id"]) for row in instances}
        if set(upstream) != ids:
            raise ValueError(
                "Exp137 upstream coverage does not match batch instances: "
                f"upstream={len(upstream)} batch={len(ids)}"
            )

        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for instance in instances:
            grouped[str(instance["profile_id"])].append(instance)
        alpha = float(policy.get("metadata", {}).get("conservative_alpha", 0.01))
        result: dict[str, dict[str, object]] = {}
        profiles = policy["profiles"]
        for profile_id in sorted(grouped):
            profile = profiles[profile_id]
            target_length = int(profile["train_target_len"])
            train_matrix, record = _load_train_matrix(
                self.train_db_path, profile_id, target_length
            )
            ordered = sorted(grouped[profile_id], key=lambda row: str(row["instance_id"]))
            query_matrix = np.vstack(
                [
                    align_to_train_length(
                        np.asarray(instance["values"], dtype=np.float32), target_length
                    )
                    for instance in ordered
                ]
            )
            evidence_rows, _ = score_profile(
                train_matrix=train_matrix,
                query_matrix=query_matrix,
                scoring_record=record,
                conservative_alpha=alpha,
            )
            for instance, evidence in zip(ordered, evidence_rows):
                instance_id = str(instance["instance_id"])
                upstream_row = upstream[instance_id]
                if str(upstream_row.get("profile_id")) != profile_id:
                    raise ValueError(
                        f"Exp137 upstream profile mismatch: instance_id={instance_id}"
                    )
                result[instance_id] = {
                    **evidence,
                    **{field: bool(upstream_row[field]) for field in EXP137_FIELDS},
                }
        return result
