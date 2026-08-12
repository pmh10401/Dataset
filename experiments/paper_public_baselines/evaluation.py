"""Post-freeze evaluation of public baselines against the frozen proposal."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .artifacts import (
    canonical_json_bytes,
    read_strict_json,
    sha256_file,
    write_create_only_bytes,
)
from .calibration import matched_top_k
from .config import ALPHAS, METHOD_IDS, SEED
from .runner import verify_score_root
from .statistics import bootstrap_draws, paired_bootstrap


LabelLoader = Callable[[tuple[str, ...]], Mapping[str, Sequence[object]]]
ProposalLoader = Callable[[Path], "ProposalAuthority"]


@dataclass(frozen=True, slots=True)
class ProposalAuthority:
    dataset_order: tuple[str, ...]
    alerts_by_dataset: Mapping[str, frozenset[int]]
    expected_confusion: tuple[int, int, int, int]
    author_ablation: tuple[Mapping[str, object], ...]
    reviewed_commit: str
    artifact_map_sha256: str

    def __post_init__(self) -> None:
        if not self.dataset_order or tuple(self.alerts_by_dataset) != self.dataset_order:
            raise ValueError("proposal dataset order is invalid")
        if len(self.expected_confusion) != 4 or any(
            type(value) is not int or value < 0 for value in self.expected_confusion
        ):
            raise ValueError("proposal confusion authority is invalid")
        object.__setattr__(
            self, "alerts_by_dataset", MappingProxyType(dict(self.alerts_by_dataset))
        )


@dataclass(frozen=True, slots=True)
class EvaluationBundle:
    metadata: Mapping[str, object]
    summaries: tuple[Mapping[str, object], ...]
    dataset_metrics: tuple[Mapping[str, object], ...]
    bootstrap: tuple[Mapping[str, object], ...]
    author_ablation: tuple[Mapping[str, object], ...]


def matched_budget_for_dataset(
    baseline_scores: np.ndarray,
    test_indices: tuple[int, ...],
    proposal_alerts: frozenset[int],
) -> tuple[int, ...]:
    """Match only the proposal alert count; never copy proposal coordinates."""
    return matched_top_k(baseline_scores, test_indices, len(proposal_alerts))


def evaluate_rows(
    predictions: Mapping[str, Sequence[int]],
    labels: Mapping[str, Sequence[int]],
    scores: Mapping[str, np.ndarray] | None = None,
) -> dict[str, object]:
    """Aggregate exact dataset-level confusion plus optional AP/AUROC."""
    if tuple(predictions) != tuple(labels):
        raise ValueError("prediction and label dataset order must match")
    dataset_rows: list[dict[str, object]] = []
    continuous_labels: list[np.ndarray] = []
    continuous_scores: list[np.ndarray] = []
    dataset_ap: list[float] = []
    dataset_auroc: list[float] = []
    for dataset, raw_labels in labels.items():
        label_values = np.asarray(tuple(raw_labels), dtype=np.int8)
        if label_values.ndim != 1 or not len(label_values) or not np.isin(label_values, (0, 1)).all():
            raise ValueError(f"labels are invalid for dataset={dataset}")
        predicted = tuple(int(index) for index in predictions[dataset])
        if len(predicted) != len(set(predicted)) or any(
            index < 0 or index >= len(label_values) for index in predicted
        ):
            raise ValueError(f"predictions are invalid for dataset={dataset}")
        selected = np.zeros(len(label_values), dtype=bool)
        selected[list(predicted)] = True
        confusion = _confusion(label_values, selected)
        ap: float | None = None
        auroc: float | None = None
        if scores is not None and dataset in scores:
            score_values = np.asarray(scores[dataset], dtype=np.float64)
            if score_values.shape != (len(label_values),) or not np.isfinite(score_values).all():
                raise ValueError(f"scores are invalid for dataset={dataset}")
            continuous_labels.append(label_values)
            continuous_scores.append(score_values)
            if len(np.unique(label_values)) == 2:
                ap = float(average_precision_score(label_values, score_values))
                auroc = float(roc_auc_score(label_values, score_values))
                dataset_ap.append(ap)
                dataset_auroc.append(auroc)
        dataset_rows.append(
            {
                "dataset": dataset,
                **confusion,
                "alert_count": len(predicted),
                "average_precision": ap,
                "auroc": auroc,
            }
        )
    micro_counts = {
        name: sum(int(row[name]) for row in dataset_rows)
        for name in ("tp", "fp", "fn", "tn")
    }
    micro = {**micro_counts, **_derived_metrics(**micro_counts)}
    if continuous_labels:
        joined_labels = np.concatenate(continuous_labels)
        joined_scores = np.concatenate(continuous_scores)
        micro["average_precision"] = float(
            average_precision_score(joined_labels, joined_scores)
        )
        micro["auroc"] = float(roc_auc_score(joined_labels, joined_scores))
    else:
        micro["average_precision"] = None
        micro["auroc"] = None
    macro = {
        name: float(np.mean([float(row[name]) for row in dataset_rows]))
        for name in ("precision", "recall", "f1", "false_positive_rate")
    }
    macro["average_precision"] = (
        float(np.mean(dataset_ap)) if dataset_ap else None
    )
    macro["auroc"] = float(np.mean(dataset_auroc)) if dataset_auroc else None
    return {
        "micro": micro,
        "macro": macro,
        "dataset_rows": tuple(dataset_rows),
        "continuous_dataset_count": len(dataset_ap),
        "alert_count": sum(int(row["alert_count"]) for row in dataset_rows),
    }


def evaluate_frozen_scores(
    run_root: Path,
    exp448_root: Path,
    *,
    label_loader: LabelLoader | None = None,
    proposal_loader: ProposalLoader | None = None,
) -> EvaluationBundle:
    """Verify all score bytes before the first possible TEST-label call."""
    root = Path(run_root)
    manifest = verify_score_root(root)
    run_config = read_strict_json(root / "run_config.json")
    datasets = _string_tuple(run_config.get("datasets"), "datasets")
    methods = _string_tuple(run_config.get("methods"), "methods")
    if methods != METHOD_IDS:
        raise ValueError("frozen score root does not cover all public methods")

    proposal = (
        proposal_loader(Path(exp448_root))
        if proposal_loader is not None
        else load_proposal_authority(Path(exp448_root))
    )
    if tuple(proposal.dataset_order[: len(datasets)]) != datasets:
        raise ValueError("score population differs from proposal population")

    if label_loader is None:
        from .label_access import load_test_labels

        active_label_loader = load_test_labels
    else:
        active_label_loader = label_loader
    labels = _normalize_labels(active_label_loader(datasets), datasets)
    proposal_alerts = {
        dataset: proposal.alerts_by_dataset[dataset] for dataset in datasets
    }
    for dataset in datasets:
        if any(index >= len(labels[dataset]) for index in proposal_alerts[dataset]):
            raise ValueError("proposal alert lies outside TEST population")

    shards = _load_shards(root, manifest.shards, methods, datasets, labels)
    common_datasets = tuple(
        dataset
        for dataset in datasets
        if all(shards[(method, dataset)]["status"] == "complete" for method in methods)
    )
    summary_rows: list[Mapping[str, object]] = []
    dataset_rows: list[Mapping[str, object]] = []

    for support, support_datasets in (
        ("full_population", datasets),
        ("common_support", common_datasets),
    ):
        if not support_datasets:
            continue
        proposal_eval = evaluate_rows(
            {dataset: tuple(sorted(proposal_alerts[dataset])) for dataset in support_datasets},
            {dataset: labels[dataset] for dataset in support_datasets},
        )
        proposal_summary = _summary_payload(
            method="proposed_equipment_anomaly_detection",
            external=False,
            protocol="proposal_frozen",
            alpha=None,
            support=support,
            support_datasets=support_datasets,
            completed_datasets=len(support_datasets),
            abstained_datasets=0,
            evaluated=proposal_eval,
        )
        summary_rows.append(proposal_summary)
        dataset_rows.extend(
            _decorate_dataset_rows(
                proposal_eval["dataset_rows"],
                "proposed_equipment_anomaly_detection",
                False,
                "proposal_frozen",
                None,
                support,
                {dataset: "complete" for dataset in support_datasets},
            )
        )

        for method in methods:
            for protocol, alpha in (
                *(("conformal", alpha) for alpha in ALPHAS),
                ("matched_alert_budget", None),
            ):
                predictions: dict[str, tuple[int, ...]] = {}
                score_rows: dict[str, np.ndarray] = {}
                status_by_dataset: dict[str, str] = {}
                completed = abstained = 0
                for dataset in support_datasets:
                    shard = shards[(method, dataset)]
                    status = str(shard["status"])
                    status_by_dataset[dataset] = status
                    if status == "complete":
                        completed += 1
                        scores = np.asarray(shard["test_scores"], dtype=np.float64)
                        score_rows[dataset] = scores
                        if protocol == "conformal":
                            predictions[dataset] = tuple(
                                int(value)
                                for value in shard["alerts_by_alpha"][format(float(alpha), ".12g")]
                            )
                        else:
                            predictions[dataset] = matched_budget_for_dataset(
                                scores,
                                tuple(int(value) for value in shard["test_indices"]),
                                proposal_alerts[dataset],
                            )
                    else:
                        abstained += 1
                        predictions[dataset] = ()
                evaluated = evaluate_rows(
                    predictions,
                    {dataset: labels[dataset] for dataset in support_datasets},
                    scores=score_rows,
                )
                summary_rows.append(
                    _summary_payload(
                        method=method,
                        external=True,
                        protocol=protocol,
                        alpha=alpha,
                        support=support,
                        support_datasets=support_datasets,
                        completed_datasets=completed,
                        abstained_datasets=abstained,
                        evaluated=evaluated,
                    )
                )
                dataset_rows.extend(
                    _decorate_dataset_rows(
                        evaluated["dataset_rows"],
                        method,
                        True,
                        protocol,
                        alpha,
                        support,
                        status_by_dataset,
                    )
                )

    if datasets == proposal.dataset_order:
        proposal_full = next(
            row
            for row in summary_rows
            if row["method"] == "proposed_equipment_anomaly_detection"
            and row["support"] == "full_population"
        )
        observed = tuple(int(proposal_full[name]) for name in ("tp", "fp", "fn", "tn"))
        if observed != proposal.expected_confusion:
            raise ValueError("reconstructed proposal confusion differs from Exp448")

    bootstrap_rows = _bootstrap_comparisons(dataset_rows, datasets, common_datasets)
    metadata = MappingProxyType(
        {
            "schema": "paper-public-baselines.evaluation.v1",
            "score_manifest_sha256": sha256_file(root / "frozen_score_manifest.json"),
            "database_sha256": manifest.database_sha256,
            "dataset_count": len(datasets),
            "common_support_count": len(common_datasets),
            "method_count": len(methods),
            "proposal_reviewed_commit": proposal.reviewed_commit,
            "proposal_artifact_map_sha256": proposal.artifact_map_sha256,
            "bootstrap_seed": SEED,
            "bootstrap_samples": 2_000,
            "evaluation_status": "retrospective_comparison_complete_no_promotion",
        }
    )
    return EvaluationBundle(
        metadata=metadata,
        summaries=tuple(MappingProxyType(dict(row)) for row in summary_rows),
        dataset_metrics=tuple(MappingProxyType(dict(row)) for row in dataset_rows),
        bootstrap=tuple(MappingProxyType(dict(row)) for row in bootstrap_rows),
        author_ablation=proposal.author_ablation,
    )


def load_proposal_authority(exp448_root: Path) -> ProposalAuthority:
    """Semantically replay Exp448 and reconstruct its final alert coordinates."""
    from experiments.exp448_exp447_text_label_boundary_repair_antigravity.publication_identity_antigravity import (
        capture_exp446_publication_antigravity,
    )
    from experiments.exp448_exp447_text_label_boundary_repair_antigravity.replay_antigravity import (
        inspect_replay_with_dependencies_antigravity,
    )
    from experiments.exp448_exp447_text_label_boundary_repair_antigravity.sources_antigravity import (
        capture_hard_predictions_antigravity,
    )

    class Dependencies:
        root = Path(exp448_root)

        @staticmethod
        def capture_after():
            return capture_exp446_publication_antigravity()

    replay = inspect_replay_with_dependencies_antigravity(Dependencies())
    identity = replay.identity
    hard = capture_hard_predictions_antigravity(identity)
    grouped: dict[str, set[int]] = {
        dataset: set() for dataset in identity.dataset_order
    }
    for dataset, index in hard.coordinates | frozenset(identity.stable_coordinates):
        grouped[dataset].add(index)
    primary = replay.metrics["hard_plus_stable_26"].confusion
    author_names = {
        "hard_baseline": "제안기술 기본 구성",
        "hard_plus_source_30": "후보 확장 적용 구성",
        "hard_plus_stable_26": "안정성 선별 적용 제안기술",
        "hard_plus_unstable_4": "불안정 후보 진단 구성",
    }
    ablation: list[Mapping[str, object]] = []
    for provenance, paper_name in author_names.items():
        summary = replay.metrics[provenance]
        confusion = summary.confusion
        ablation.append(
            MappingProxyType(
                {
                    "paper_name": paper_name,
                    "internal_provenance": provenance,
                    "tp": confusion.tp,
                    "fp": confusion.fp,
                    "fn": confusion.fn,
                    "tn": confusion.tn,
                    "f1": confusion.f1,
                    "macro_f1": summary.macro_f1,
                }
            )
        )
    return ProposalAuthority(
        dataset_order=tuple(identity.dataset_order),
        alerts_by_dataset=MappingProxyType(
            {dataset: frozenset(indices) for dataset, indices in grouped.items()}
        ),
        expected_confusion=(primary.tp, primary.fp, primary.fn, primary.tn),
        author_ablation=tuple(ablation),
        reviewed_commit=replay.reviewed_commit,
        artifact_map_sha256=sha256_file(
            Path(exp448_root) / "artifact_sha256_antigravity.json"
        ),
    )


def write_evaluation_bundle(run_root: Path, bundle: EvaluationBundle) -> str:
    """Materialize immutable evaluation leaves below the frozen score root."""
    destination = Path(run_root) / "evaluation_public_baselines"
    summary_payload = {
        "metadata": dict(bundle.metadata),
        "summaries": [dict(row) for row in bundle.summaries],
        "author_ablation": [dict(row) for row in bundle.author_ablation],
    }
    leaves = {
        "evaluation_summary.json": canonical_json_bytes(summary_payload),
        "evaluation_dataset_metrics.jsonl": b"".join(
            canonical_json_bytes(dict(row)) for row in bundle.dataset_metrics
        ),
        "evaluation_bootstrap.json": canonical_json_bytes(
            {"comparisons": [dict(row) for row in bundle.bootstrap]}
        ),
    }
    for name, payload in leaves.items():
        write_create_only_bytes(destination / name, payload)
    artifact_map = {
        name: hashlib.sha256(payload).hexdigest() for name, payload in leaves.items()
    }
    encoded_map = canonical_json_bytes(artifact_map)
    write_create_only_bytes(destination / "artifact_sha256.json", encoded_map)
    return hashlib.sha256(encoded_map).hexdigest()


def _normalize_labels(
    raw: Mapping[str, Sequence[object]], datasets: tuple[str, ...]
) -> dict[str, tuple[int, ...]]:
    if tuple(raw) != datasets:
        raise ValueError("label population order is invalid")
    normalized: dict[str, tuple[int, ...]] = {}
    for dataset in datasets:
        rows = tuple(raw[dataset])
        if rows and all(isinstance(row, tuple) and len(row) == 2 for row in rows):
            indices = tuple(row[0] for row in rows)
            values = tuple(row[1] for row in rows)
            if indices != tuple(range(len(rows))):
                raise ValueError("TEST label indices are not contiguous")
        else:
            values = rows
        if not values or any(type(value) is not int or value not in (0, 1) for value in values):
            raise ValueError("TEST labels must be exact binary integers")
        normalized[dataset] = tuple(values)
    return normalized


def _load_shards(
    root: Path,
    entries: Sequence[Mapping[str, object]],
    methods: tuple[str, ...],
    datasets: tuple[str, ...],
    labels: Mapping[str, tuple[int, ...]],
) -> dict[tuple[str, str], dict[str, object]]:
    by_pair: dict[tuple[str, str], dict[str, object]] = {}
    for entry in entries:
        method, dataset = str(entry["method"]), str(entry["dataset"])
        shard = read_strict_json(root / str(entry["path"]))
        if int(shard["n_test"]) != len(labels[dataset]):
            raise ValueError("score shard TEST population differs from labels")
        by_pair[(method, dataset)] = shard
    if set(by_pair) != {(method, dataset) for method in methods for dataset in datasets}:
        raise ValueError("score shard pair coverage differs during evaluation")
    return by_pair


def _summary_payload(
    *,
    method: str,
    external: bool,
    protocol: str,
    alpha: float | None,
    support: str,
    support_datasets: tuple[str, ...],
    completed_datasets: int,
    abstained_datasets: int,
    evaluated: Mapping[str, object],
) -> dict[str, object]:
    micro = evaluated["micro"]
    macro = evaluated["macro"]
    return {
        "method": method,
        "is_external_baseline": external,
        "protocol": protocol,
        "alpha": alpha,
        "support": support,
        "dataset_count": len(support_datasets),
        "completed_datasets": completed_datasets,
        "abstained_datasets": abstained_datasets,
        "coverage": completed_datasets / len(support_datasets),
        "alerts": evaluated["alert_count"],
        "tp": micro["tp"],
        "fp": micro["fp"],
        "fn": micro["fn"],
        "tn": micro["tn"],
        "precision": micro["precision"],
        "recall": micro["recall"],
        "f1": micro["f1"],
        "false_positive_rate": micro["false_positive_rate"],
        "average_precision": micro["average_precision"],
        "auroc": micro["auroc"],
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "macro_f1": macro["f1"],
        "macro_false_positive_rate": macro["false_positive_rate"],
        "macro_average_precision": macro["average_precision"],
        "macro_auroc": macro["auroc"],
        "continuous_dataset_count": evaluated["continuous_dataset_count"],
    }


def _decorate_dataset_rows(
    rows: Sequence[Mapping[str, object]],
    method: str,
    external: bool,
    protocol: str,
    alpha: float | None,
    support: str,
    status_by_dataset: Mapping[str, str],
) -> list[dict[str, object]]:
    return [
        {
            "method": method,
            "is_external_baseline": external,
            "protocol": protocol,
            "alpha": alpha,
            "support": support,
            "status": status_by_dataset[str(row["dataset"])],
            **dict(row),
        }
        for row in rows
    ]


def _bootstrap_comparisons(
    dataset_rows: Sequence[Mapping[str, object]],
    full_datasets: tuple[str, ...],
    common_datasets: tuple[str, ...],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for support, datasets in (
        ("full_population", full_datasets),
        ("common_support", common_datasets),
    ):
        if not datasets:
            continue
        draws, draw_sha = bootstrap_draws(
            len(datasets), seed=SEED, samples=2_000
        )
        proposal = {
            str(row["dataset"]): float(row["f1"])
            for row in dataset_rows
            if row["method"] == "proposed_equipment_anomaly_detection"
            and row["support"] == support
        }
        for method in METHOD_IDS:
            identities = sorted(
                {
                    (str(row["protocol"]), row["alpha"])
                    for row in dataset_rows
                    if row["method"] == method and row["support"] == support
                },
                key=lambda item: (item[0], -1.0 if item[1] is None else float(item[1])),
            )
            for protocol, alpha in identities:
                baseline = {
                    str(row["dataset"]): float(row["f1"])
                    for row in dataset_rows
                    if row["method"] == method
                    and row["support"] == support
                    and row["protocol"] == protocol
                    and row["alpha"] == alpha
                }
                comparison = paired_bootstrap(
                    np.asarray([baseline[dataset] for dataset in datasets]),
                    np.asarray([proposal[dataset] for dataset in datasets]),
                    draws=draws,
                    draw_sha256=draw_sha,
                    seed=SEED,
                )
                results.append(
                    {
                        "method": method,
                        "protocol": protocol,
                        "alpha": alpha,
                        "support": support,
                        "estimand": "dataset_macro_f1_baseline_minus_proposal",
                        **comparison,
                    }
                )
    return results


def _confusion(labels: np.ndarray, selected: np.ndarray) -> dict[str, object]:
    tp = int(np.count_nonzero(selected & (labels == 1)))
    fp = int(np.count_nonzero(selected & (labels == 0)))
    fn = int(np.count_nonzero(~selected & (labels == 1)))
    tn = int(np.count_nonzero(~selected & (labels == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, **_derived_metrics(tp, fp, fn, tn)}


def _derived_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
    }


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(type(item) is not str or not item for item in value):
        raise ValueError(f"run config {name} is invalid")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"run config {name} is not unique")
    return result


__all__ = [
    "EvaluationBundle",
    "ProposalAuthority",
    "evaluate_frozen_scores",
    "evaluate_rows",
    "load_proposal_authority",
    "matched_budget_for_dataset",
    "write_evaluation_bundle",
]
