from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from experiments.exp177_185.core import _score_fit_query, canonical_features
from experiments.exp186_200.core import empirical_upper_tail_pvalues
from experiments.exp220_223.prediction import EXP137_FIELDS, freeze_shadow_predictions
from experiments.exp220_223.scoring import (
    _read_exp137_upstream,
    _verify_exp137_manifest,
    align_to_train_length,
)
from experiments.exp220_223.storage import load_batch_instances, load_batch_metadata
from experiments.exp224.artifacts import _runtime_metadata, load_train_profile
from experiments.exp224.policy import load_verified_exp224_policy

ScoreBackend = Callable[
    [str, np.ndarray, np.ndarray, Mapping[str, object]],
    tuple[np.ndarray, np.ndarray],
]


def assert_shadow_gate(
    policy: Mapping[str, object],
    report: Mapping[str, object],
) -> None:
    if report.get("policy_hash") != policy.get("policy_hash"):
        raise ValueError("raw replay policy hash mismatch")
    if (
        int(report.get("profile_count", -1)) != 639
        or int(report.get("expected_profile_count", -1)) != 639
    ):
        raise ValueError("raw replay must cover exactly 639/639 profiles")
    if int(report.get("mismatch_count", -1)) != 0:
        raise ValueError("raw replay contains mismatches")
    if report.get("shadow_execution_gate") != "passed":
        raise ValueError("raw replay gate is not passed")


def _real_score_backend(
    profile_id: str,
    train: np.ndarray,
    query: np.ndarray,
    record: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    del profile_id
    from experiments.exp153_157 import exp153_157_bridge as bridge

    _, temporal_query, _ = bridge.exp152.source_score_pair(
        "rocket_exp40",
        np.asarray(train, dtype=np.float32),
        np.asarray(query, dtype=np.float32),
        np.asarray(query, dtype=np.float32),
        dict(record),
    )
    train_features = canonical_features(train)
    query_features = canonical_features(query)
    canonical_query = _score_fit_query(
        train_features,
        query_features,
        "knn_local_gap",
    )
    return (
        np.asarray(temporal_query, dtype=float),
        np.asarray(canonical_query, dtype=float),
    )


class Exp224EvidenceProvider:
    """Score future values from TRAIN plus a mandatory frozen Exp137 adapter."""

    def __init__(
        self,
        *,
        train_db_path: Path,
        exp137_predictions_path: Path,
        exp137_manifest_path: Path,
        batch_id: str,
        score_backend: ScoreBackend | None = None,
    ) -> None:
        self.train_db_path = Path(train_db_path)
        self.exp137_predictions_path = Path(exp137_predictions_path)
        self.exp137_manifest_path = Path(exp137_manifest_path)
        self.batch_id = str(batch_id)
        self.score_backend = score_backend or _real_score_backend

    def __call__(
        self,
        instances: list[dict[str, object]],
        policy: Mapping[str, object],
    ) -> dict[str, dict[str, object]]:
        if policy.get("runtime") != _runtime_metadata():
            raise ValueError("Exp224 runtime mismatch")
        _verify_exp137_manifest(
            self.exp137_predictions_path,
            self.exp137_manifest_path,
            self.batch_id,
        )
        upstream = _read_exp137_upstream(
            self.exp137_predictions_path,
            self.batch_id,
        )
        instance_ids = {str(instance["instance_id"]) for instance in instances}
        if len(instance_ids) != len(instances):
            raise ValueError("duplicate batch instance_id")
        if set(upstream) != instance_ids:
            raise ValueError("Exp137 upstream coverage does not match batch instances")

        profiles = {
            str(profile["profile_id"]): profile for profile in policy["profiles"]
        }
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for instance in instances:
            profile_id = str(instance["profile_id"])
            if profile_id not in profiles:
                raise ValueError(f"unregistered Exp224 profile: {profile_id}")
            grouped[profile_id].append(instance)

        result: dict[str, dict[str, object]] = {}
        for profile_id in sorted(grouped):
            profile = profiles[profile_id]
            target_length = int(profile["train_target_length"])
            train, record = load_train_profile(
                self.train_db_path,
                profile_id,
                target_length=target_length,
            )
            ordered = sorted(
                grouped[profile_id], key=lambda row: str(row["instance_id"])
            )
            query = np.vstack(
                [
                    align_to_train_length(
                        np.asarray(instance["values"], dtype=np.float32),
                        target_length,
                    )
                    for instance in ordered
                ]
            )
            temporal_scores, canonical_scores = self.score_backend(
                profile_id, train, query, record
            )
            temporal_scores = np.asarray(temporal_scores, dtype=float)
            canonical_scores = np.asarray(canonical_scores, dtype=float)
            if (
                len(temporal_scores) != len(ordered)
                or len(canonical_scores) != len(ordered)
                or not np.all(np.isfinite(temporal_scores))
                or not np.all(np.isfinite(canonical_scores))
            ):
                raise ValueError(f"invalid Exp224 query scores: {profile_id}")
            with np.load(Path(str(profile["artifact_path"])), allow_pickle=False) as archive:
                temporal_oof = np.asarray(archive["temporal_oof"], dtype=float)
                canonical_oof = np.asarray(archive["canonical_oof"], dtype=float)
            temporal_p = empirical_upper_tail_pvalues(temporal_oof, temporal_scores)
            canonical_p = empirical_upper_tail_pvalues(canonical_oof, canonical_scores)
            for index, instance in enumerate(ordered):
                instance_id = str(instance["instance_id"])
                upstream_row = upstream[instance_id]
                if str(upstream_row.get("profile_id")) != profile_id:
                    raise ValueError(
                        f"Exp137 upstream profile mismatch: instance_id={instance_id}"
                    )
                result[instance_id] = {
                    **{field: bool(upstream_row[field]) for field in EXP137_FIELDS},
                    "temporal_p": float(temporal_p[index]),
                    "canonical_p": float(canonical_p[index]),
                }
        return result


def build_exp224_rows(
    *,
    batch_id: str,
    instances: list[dict[str, object]],
    policy: Mapping[str, object],
    evidence: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    alpha_value = policy.get("selected_alpha")
    alpha = None if alpha_value is None else float(alpha_value)
    rows: list[dict[str, object]] = []
    for instance in sorted(instances, key=lambda row: str(row["instance_id"])):
        instance_id = str(instance["instance_id"])
        values = evidence[instance_id]
        exp137 = [bool(values[field]) for field in EXP137_FIELDS]
        if sum(exp137) > 1:
            raise ValueError(f"Exp137 lanes overlap: instance_id={instance_id}")
        temporal_p = float(values["temporal_p"])
        canonical_p = float(values["canonical_p"])
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in (temporal_p, canonical_p)
        ):
            raise ValueError(f"invalid Exp224 p-value: instance_id={instance_id}")
        excluded = any(exp137)
        selected = bool(
            alpha is not None
            and temporal_p <= alpha
            and canonical_p <= alpha
            and not excluded
        )
        rows.append(
            {
                "batch_id": str(batch_id),
                "instance_id": instance_id,
                "profile_id": str(instance["profile_id"]),
                "policy_hash": str(policy["policy_hash"]),
                **{field: exp137[index] for index, field in enumerate(EXP137_FIELDS)},
                "temporal_p": temporal_p,
                "canonical_p": canonical_p,
                "selected_alpha": alpha,
                "exp137_excluded": excluded,
                "exp224_shadow": selected,
                "autonomous_promotion": False,
            }
        )
    return rows


def predict_exp224_batch(
    *,
    values_db_path: Path,
    train_db_path: Path,
    batch_id: str,
    policy_path: Path,
    raw_report_path: Path,
    exp137_predictions_path: Path,
    exp137_manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    policy = load_verified_exp224_policy(policy_path, verify_artifacts=True)
    report = json.loads(Path(raw_report_path).read_text(encoding="utf-8"))
    assert_shadow_gate(policy, report)
    batch = load_batch_metadata(values_db_path, batch_id)
    if batch["policy_sha256"] != policy["policy_hash"]:
        raise ValueError("batch policy hash does not match frozen Exp224 policy")
    instances = load_batch_instances(values_db_path, batch_id)
    provider = Exp224EvidenceProvider(
        train_db_path=train_db_path,
        exp137_predictions_path=exp137_predictions_path,
        exp137_manifest_path=exp137_manifest_path,
        batch_id=batch_id,
    )
    evidence = provider(instances, policy)
    rows = build_exp224_rows(
        batch_id=batch_id,
        instances=instances,
        policy=policy,
        evidence=evidence,
    )
    return freeze_shadow_predictions(
        rows=rows,
        batch_id=batch_id,
        policy_sha256=str(policy["policy_hash"]),
        source_sha256=batch["source_sha256"],
        output_dir=output_dir,
    )

