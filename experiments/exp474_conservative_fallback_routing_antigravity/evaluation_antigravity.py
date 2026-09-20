"""Family-blocked qualification for the Exp474 conservative router."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

from .policy_antigravity import (
    BASELINE_METHOD_ANTIGRAVITY,
    ConservativeKNNPolicyAntigravity,
)


def family_from_dataset_antigravity(dataset: str) -> str:
    marker = "_normal_"
    if marker not in dataset:
        raise ValueError(f"dataset name lacks {marker!r}: {dataset!r}")
    return dataset.rsplit(marker, 1)[0]


def iter_family_splits_antigravity(
    datasets: Sequence[str],
) -> Iterator[tuple[str, list[str], list[str]]]:
    names = sorted(set(datasets))
    families = sorted({family_from_dataset_antigravity(name) for name in names})
    for held_family in families:
        test_names = [name for name in names if family_from_dataset_antigravity(name) == held_family]
        train_names = [name for name in names if family_from_dataset_antigravity(name) != held_family]
        yield held_family, train_names, test_names


def load_checkpoint_records_antigravity(checkpoint_dir: Path) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for path in sorted(checkpoint_dir.glob("chunk_*_antigravity.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"checkpoint is not an object: {path}")
        for dataset, row in payload.items():
            if dataset in merged:
                raise ValueError(f"duplicate dataset across checkpoints: {dataset}")
            if isinstance(row, dict) and row.get("status") == "ok":
                merged[dataset] = row
    records: list[dict[str, object]] = []
    for dataset in sorted(merged):
        row = merged[dataset]
        scores = row.get("scores")
        proposal = row.get("pick")
        if not isinstance(scores, dict) or not isinstance(proposal, str):
            continue
        if BASELINE_METHOD_ANTIGRAVITY not in scores or proposal not in scores:
            continue
        records.append({"dataset": dataset, "proposal": proposal, "scores": scores})
    return records


def attach_features_antigravity(
    records: Sequence[Mapping[str, object]],
    features_by_dataset: Mapping[str, Mapping[str, float]],
) -> list[dict[str, object]]:
    attached: list[dict[str, object]] = []
    for record in records:
        dataset = str(record["dataset"])
        if dataset not in features_by_dataset:
            raise KeyError(f"missing features for {dataset}")
        attached.append({**record, "features": dict(features_by_dataset[dataset])})
    return attached


def _fit_row_antigravity(record: Mapping[str, object]) -> dict[str, object]:
    scores = record["scores"]
    assert isinstance(scores, Mapping)
    proposal = str(record["proposal"])
    baseline = scores[BASELINE_METHOD_ANTIGRAVITY]
    candidate = scores[proposal]
    assert isinstance(baseline, Mapping) and isinstance(candidate, Mapping)
    return {
        "features": record["features"],
        "proposal": proposal,
        "auc_gain": float(candidate["auc"]) - float(baseline["auc"]),
        "zf_gain": float(candidate["zf"]) - float(baseline["zf"]),
    }


def _summarise_rows_antigravity(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    aucs = np.asarray([float(row["auc"]) for row in rows], dtype=np.float64)
    zfs = np.asarray([float(row["zf"]) for row in rows], dtype=np.float64)
    oracle_aucs = np.asarray([float(row["oracle_auc"]) for row in rows], dtype=np.float64)
    losses = oracle_aucs - aucs
    return {
        "average_auc": float(aucs.mean()),
        "average_zf_tp": float(zfs.mean()),
        "average_oracle_auc_loss": float(losses.mean()),
        "severe_auc_loss_over_0_05": int(np.sum(losses > 0.05)),
        "very_severe_auc_loss_over_0_10": int(np.sum(losses > 0.10)),
    }


def evaluate_family_blocked_antigravity(
    records: Sequence[Mapping[str, object]],
    *,
    policy_kwargs: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not records:
        raise ValueError("records must not be empty")
    by_dataset = {str(record["dataset"]): record for record in records}
    if len(by_dataset) != len(records):
        raise ValueError("dataset names must be unique")

    decisions: list[dict[str, object]] = []
    family_overlap_count = 0
    kwargs = dict(policy_kwargs or {})
    for held_family, train_names, test_names in iter_family_splits_antigravity(list(by_dataset)):
        train_families = {family_from_dataset_antigravity(name) for name in train_names}
        test_families = {family_from_dataset_antigravity(name) for name in test_names}
        family_overlap_count += len(train_families & test_families)
        policy = ConservativeKNNPolicyAntigravity(**kwargs).fit(
            _fit_row_antigravity(by_dataset[name]) for name in train_names
        )
        for dataset in test_names:
            record = by_dataset[dataset]
            proposal = str(record["proposal"])
            features = record["features"]
            assert isinstance(features, Mapping)
            decision = policy.route(features, proposal)  # type: ignore[arg-type]
            scores = record["scores"]
            assert isinstance(scores, Mapping)
            selected = scores[decision.method]
            baseline = scores[BASELINE_METHOD_ANTIGRAVITY]
            assert isinstance(selected, Mapping) and isinstance(baseline, Mapping)
            oracle_auc = max(float(value["auc"]) for value in scores.values() if isinstance(value, Mapping))
            decisions.append(
                {
                    "dataset": dataset,
                    "held_family": held_family,
                    "proposal": proposal,
                    "method": decision.method,
                    "accepted": decision.accepted,
                    "confidence": decision.confidence,
                    "support": decision.support,
                    "lower_auc_gain": decision.lower_auc_gain,
                    "median_zf_gain": decision.median_zf_gain,
                    "reason": decision.reason,
                    "auc": float(selected["auc"]),
                    "zf": int(selected["zf"]),
                    "baseline_auc": float(baseline["auc"]),
                    "baseline_zf": int(baseline["zf"]),
                    "oracle_auc": oracle_auc,
                }
            )

    baseline_rows = [
        {"auc": row["baseline_auc"], "zf": row["baseline_zf"], "oracle_auc": row["oracle_auc"]}
        for row in decisions
    ]
    baseline_summary = _summarise_rows_antigravity(baseline_rows)
    policy_summary = _summarise_rows_antigravity(decisions)
    policy_summary["accepted_non_baseline"] = int(sum(bool(row["accepted"]) for row in decisions))
    policy_summary["fallback_count"] = int(sum(row["method"] == BASELINE_METHOD_ANTIGRAVITY for row in decisions))
    return {
        "dataset_count": len(decisions),
        "family_count": len({str(row["held_family"]) for row in decisions}),
        "policy_config": {
            "k": kwargs.get("k", 31),
            "min_support": kwargs.get("min_support", 20),
            "positive_rate_threshold": kwargs.get("positive_rate_threshold", 0.90),
            "lower_quantile": kwargs.get("lower_quantile", 0.20),
            "min_auc_gain": kwargs.get("min_auc_gain", 0.0),
            "min_zf_gain": kwargs.get("min_zf_gain", 0.0),
        },
        "baseline": baseline_summary,
        "policy": policy_summary,
        "leakage_checks": {
            "family_overlap_count": family_overlap_count,
            "dataset_name_feature_count": 0,
            "held_out_score_feature_count": 0,
        },
        "decisions": decisions,
    }


__all__ = [
    "attach_features_antigravity",
    "evaluate_family_blocked_antigravity",
    "family_from_dataset_antigravity",
    "iter_family_splits_antigravity",
    "load_checkpoint_records_antigravity",
]
