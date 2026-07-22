from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from experiments.exp309_316_fn_recovery.contracts import verify_prediction_rows

from .config import (
    ALPHAS,
    EXP290_ROOT,
    EXP291_MANIFEST,
    EXP291_PREDICTION,
    EXPERIMENT_IDS,
    SHARE_THRESHOLDS,
    SOURCE_GROUPS,
)
from .contracts import freeze_jsonl, sha256_file, verify_frozen_jsonl
from .source_loader import load_verified_sources


@dataclass(frozen=True)
class WeightedGroupSupport:
    available_group_count: int
    supporting_group_count: int
    available_weight: float
    supporting_weight: float
    share: float
    supporting_groups: tuple[str, ...]


def weighted_group_support(
    weights: Mapping[str, float], supporting_sources: set[str], *,
    groups: Mapping[str, str] = SOURCE_GROUPS,
) -> WeightedGroupSupport:
    available_by_group: dict[str, float] = {}
    supporting_by_group: dict[str, float] = {}
    for source, raw_weight in weights.items():
        weight = float(raw_weight)
        if weight <= 0.0:
            continue
        group = str(groups[source])
        available_by_group[group] = max(available_by_group.get(group, 0.0), weight)
        if source in supporting_sources:
            supporting_by_group[group] = max(supporting_by_group.get(group, 0.0), weight)
    available_weight = float(sum(available_by_group.values()))
    supporting_weight = float(sum(supporting_by_group.values()))
    return WeightedGroupSupport(
        available_group_count=len(available_by_group),
        supporting_group_count=len(supporting_by_group),
        available_weight=available_weight,
        supporting_weight=supporting_weight,
        share=supporting_weight / available_weight if available_weight else 0.0,
        supporting_groups=tuple(sorted(supporting_by_group)),
    )


def qualifies(
    share: float, available_groups: int, supporting_groups: int, *, threshold: float,
) -> bool:
    return int(available_groups) >= 3 and int(supporting_groups) >= 2 and float(share) >= float(threshold)


def apply_candidate_guards(
    candidates: set[int], *, hard: set[int], exact_normal: set[int],
) -> set[int]:
    return set(candidates) - set(hard) - set(exact_normal)


def _alpha_key(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def _share_key(share: float) -> str:
    return f"{share:.2f}".replace(".", "p")


def _load_reliability(root: Path) -> tuple[dict[tuple[str, str, float], dict[str, object]], dict[str, object]]:
    rows_path = Path(root) / "exp317_frozen_reliability.jsonl"
    manifest_path = Path(root) / "exp317_reliability_manifest.json"
    rows = verify_frozen_jsonl(rows_path, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mapping = {
        (str(row["dataset"]), str(row["source"]), float(row["alpha"])): row
        for row in rows
    }
    if len(mapping) != len(rows):
        raise ValueError("duplicate Exp317 reliability rows")
    return mapping, manifest


def run_exp318_prediction(
    output_root: Path, reliability_root: Path, dataset_limit: int | None = None, *,
    source_roots: dict[str, Path] | None = None,
    progress_callback=None,
) -> dict[str, object]:
    sources, source_hashes, frozen_datasets = load_verified_sources(source_roots)
    reliability, reliability_manifest = _load_reliability(reliability_root)
    if reliability_manifest.get("source_prediction_sha256") != source_hashes:
        raise ValueError("Exp317 reliability source hashes do not match current sources")

    exp291_rows = verify_prediction_rows(EXP291_PREDICTION, EXP291_MANIFEST)
    exp291 = {str(row["dataset"]): row for row in exp291_rows}
    exp290_prediction = EXP290_ROOT / "exp290_frozen_predictions.jsonl"
    exp290_manifest = EXP290_ROOT / "exp290_prediction_manifest.json"
    exp290 = {
        str(row["dataset"]): row
        for row in verify_prediction_rows(exp290_prediction, exp290_manifest)
    }
    datasets = sorted(frozen_datasets)
    if dataset_limit is not None:
        datasets = datasets[: int(dataset_limit)]

    outputs: list[dict[str, object]] = []
    for position, dataset in enumerate(datasets, start=1):
        hard = set(map(int, exp291[dataset]["automatic_hard_indices"]))
        exact_normal = set(map(int, exp290[dataset]["train_raw_exact_normal_indices"]))
        row: dict[str, object] = {
            "experiment_id": EXPERIMENT_IDS[318],
            "dataset": dataset,
            "n_test": int(exp291[dataset]["n_test"]),
            "automatic_hard_indices": sorted(hard),
            "exact_normal_guard_indices": sorted(exact_normal),
            "contains_test_values": False,
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
            "automatic_policy_changed": False,
            "shadow_autonomous": False,
            "primary_policy_selected": False,
            "retrospective": True,
        }
        for alpha in ALPHAS:
            alpha_key = _alpha_key(alpha)
            weights: dict[str, float] = {}
            evidence: dict[str, set[int]] = {}
            for source in sorted(sources):
                reliability_row = reliability.get((dataset, source, float(alpha)))
                if reliability_row is None:
                    raise ValueError(f"missing Exp317 reliability row: {dataset}/{source}/{alpha}")
                weights[source] = float(reliability_row["weight"]) if reliability_row["available"] else 0.0
                evidence[source] = set(map(int, sources[source][dataset][f"alpha_{alpha_key}_indices"]))

            available = weighted_group_support(weights, set())
            universe = apply_candidate_guards(
                set().union(*evidence.values()), hard=hard, exact_normal=exact_normal,
            )
            diagnostics: dict[str, object] = {}
            selected_by_share = {share: set() for share in SHARE_THRESHOLDS}
            for index in sorted(universe):
                supporting_sources = {source for source, indices in evidence.items() if index in indices}
                support = weighted_group_support(weights, supporting_sources)
                diagnostics[str(index)] = {
                    "supporting_sources": sorted(supporting_sources),
                    "supporting_groups": list(support.supporting_groups),
                    "supporting_group_count": support.supporting_group_count,
                    "available_group_count": support.available_group_count,
                    "supporting_weight": support.supporting_weight,
                    "available_weight": support.available_weight,
                    "weighted_share": support.share,
                }
                for share in SHARE_THRESHOLDS:
                    if qualifies(
                        support.share,
                        support.available_group_count,
                        support.supporting_group_count,
                        threshold=share,
                    ):
                        selected_by_share[share].add(index)
            row[f"alpha_{alpha_key}_source_weights"] = weights
            row[f"alpha_{alpha_key}_available_group_count"] = available.available_group_count
            row[f"alpha_{alpha_key}_available_weight"] = available.available_weight
            row[f"alpha_{alpha_key}_support_by_index"] = diagnostics
            for share, selected in selected_by_share.items():
                row[f"alpha_{alpha_key}_share_{_share_key(share)}_indices"] = sorted(selected)
        outputs.append(row)
        if progress_callback is not None:
            progress_callback(position, len(datasets), dataset)

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    return freeze_jsonl(
        outputs,
        root / "exp318_frozen_predictions.jsonl",
        root / "exp318_prediction_manifest.json",
        experiment_id=EXPERIMENT_IDS[318],
        metadata={
            "datasets": len(outputs),
            "dataset_limit": dataset_limit,
            "alphas": list(ALPHAS),
            "share_thresholds": list(SHARE_THRESHOLDS),
            "minimum_available_groups": 3,
            "minimum_supporting_groups": 2,
            "source_groups": SOURCE_GROUPS,
            "source_prediction_sha256": source_hashes,
            "exp317_prediction_sha256": sha256_file(Path(reliability_root) / "exp317_frozen_reliability.jsonl"),
            "exp291_prediction_sha256": sha256_file(EXP291_PREDICTION),
            "exp290_prediction_sha256": sha256_file(exp290_prediction),
            "automatic_policy_changed": False,
            "research_shadow_only": True,
            "primary_policy_selected": False,
            "retrospective": True,
            "prospective_validated": False,
        },
    )
