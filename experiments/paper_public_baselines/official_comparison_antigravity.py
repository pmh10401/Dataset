"""Canonical six-method comparison over independently verified evaluations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .artifacts import (
    canonical_json_bytes,
    read_strict_json,
    sha256_file,
    write_create_only_bytes,
)
from .evaluation import EvaluationBundle, ProposalAuthority, evaluate_frozen_scores, evaluate_rows
from .reporting import DATASET_FIELDS, SUMMARY_FIELDS


EXPECTED_METHOD_ORDER_ANTIGRAVITY = (
    "knn_distance",
    "lof_novelty",
    "ocsvm_rbf",
    "isolation_forest",
    "deep_svdd_1d",
    "drocc_1d",
)
OFFICIAL_COMPARISON_SCHEMA_ANTIGRAVITY = (
    "paper-public-baselines.official-comparison-antigravity.v1"
)
OFFICIAL_CONTINUOUS_COMPARISON_SCHEMA_ANTIGRAVITY = (
    "paper-public-baselines.official-comparison-exp449-continuous-antigravity.v1"
)
EXP449_POLICY_ORDER_ANTIGRAVITY = (
    "hard_only",
    "hard_plus_source",
    "hard_plus_stable",
    "hard_plus_unstable",
)
EXP449_SELECTED_POLICY_ANTIGRAVITY = "hard_plus_stable"
EXP449_MANIFEST_RELATIVE_PATH_ANTIGRAVITY = Path(
    "stage_artifacts_antigravity/proposal_freeze_antigravity/"
    "proposal_manifest_antigravity.json"
)
OFFICIAL_COMPARISON_FILES_ANTIGRAVITY = (
    "official_comparison_summary_antigravity.json",
    "official_comparison_summary_antigravity.csv",
    "official_comparison_dataset_metrics_antigravity.csv",
    "official_comparison_bootstrap_antigravity.json",
)
OFFICIAL_COMPARISON_ARTIFACT_MAP_ANTIGRAVITY = (
    "official_comparison_artifact_sha256_antigravity.json"
)


@dataclass(frozen=True, slots=True)
class OfficialComparisonAntigravity:
    metadata: Mapping[str, object]
    public_methods: tuple[str, ...]
    summaries: tuple[Mapping[str, object], ...]
    dataset_metrics: tuple[Mapping[str, object], ...]
    bootstrap: tuple[Mapping[str, object], ...]
    author_ablation: tuple[Mapping[str, object], ...]

    @property
    def dataset_count(self) -> int:
        return int(self.metadata["dataset_count"])


@dataclass(frozen=True, slots=True)
class Exp449ContinuousProposalAntigravity:
    """Verified route-preserving Candidate-C scores and policy coordinates."""

    dataset_order: tuple[str, ...]
    alerts_by_dataset: Mapping[str, frozenset[int]]
    scores_by_dataset: Mapping[str, Sequence[float]]
    policy_alerts_by_dataset: Mapping[str, Mapping[str, frozenset[int]]]
    manifest_sha256: str
    context_sha256: str
    catalog_sha256: str
    test_row_count: int

    def __post_init__(self) -> None:
        if (
            not self.dataset_order
            or tuple(self.alerts_by_dataset) != self.dataset_order
            or tuple(self.scores_by_dataset) != self.dataset_order
            or tuple(self.policy_alerts_by_dataset) != self.dataset_order
        ):
            raise ValueError("Exp449 continuous proposal population differs")
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in (self.manifest_sha256, self.context_sha256, self.catalog_sha256)
        ):
            raise ValueError("Exp449 continuous proposal SHA256 differs")
        total = 0
        frozen_scores: dict[str, tuple[float, ...]] = {}
        frozen_policies: dict[str, Mapping[str, frozenset[int]]] = {}
        for dataset in self.dataset_order:
            scores = tuple(float(value) for value in self.scores_by_dataset[dataset])
            if not scores or not np.isfinite(np.asarray(scores, dtype=np.float64)).all():
                raise ValueError("Exp449 continuous proposal scores differ")
            alerts = frozenset(self.alerts_by_dataset[dataset])
            if alerts != frozenset(index for index, score in enumerate(scores) if score >= 0.5):
                raise ValueError("Exp449 continuous proposal alert boundary differs")
            policies = dict(self.policy_alerts_by_dataset[dataset])
            if tuple(policies) != EXP449_POLICY_ORDER_ANTIGRAVITY:
                raise ValueError("Exp449 continuous proposal policy closure differs")
            if frozenset(policies[EXP449_SELECTED_POLICY_ANTIGRAVITY]) != alerts:
                raise ValueError("Exp449 selected policy coordinates differ")
            frozen_scores[dataset] = scores
            frozen_policies[dataset] = MappingProxyType(
                {name: frozenset(indices) for name, indices in policies.items()}
            )
            total += len(scores)
        if type(self.test_row_count) is not int or total != self.test_row_count:
            raise ValueError("Exp449 continuous proposal TEST row count differs")
        object.__setattr__(self, "alerts_by_dataset", MappingProxyType(
            {dataset: frozenset(self.alerts_by_dataset[dataset]) for dataset in self.dataset_order}
        ))
        object.__setattr__(self, "scores_by_dataset", MappingProxyType(frozen_scores))
        object.__setattr__(self, "policy_alerts_by_dataset", MappingProxyType(frozen_policies))


def _validate_exp449_continuous_shard_antigravity(
    shard: object,
    *,
    dataset: str,
    catalog_sha256: str,
    backend_record: Mapping[str, object],
) -> tuple[frozenset[int], np.ndarray, Mapping[str, frozenset[int]]]:
    if (
        not isinstance(shard, dict)
        or shard.get("schema") != "exp449-proposal-shard-antigravity.v1"
        or shard.get("catalog_sha256") != catalog_sha256
        or shard.get("backend_record") != dict(backend_record)
        or not isinstance(shard.get("scientific_payload"), dict)
    ):
        raise ValueError("Exp449 continuous shard authority differs")
    scientific = shard["scientific_payload"]
    identity = scientific.get("test_identity")
    policies = scientific.get("policies")
    if (
        scientific.get("dataset") != dataset
        or not isinstance(identity, dict)
        or identity.get("dataset") != dataset
        or identity.get("split") != "TEST"
        or type(identity.get("count")) is not int
        or identity.get("indices") != list(range(identity["count"]))
        or not isinstance(policies, list)
        or [row.get("policy_id") if isinstance(row, dict) else None for row in policies]
        != list(EXP449_POLICY_ORDER_ANTIGRAVITY)
    ):
        raise ValueError("Exp449 continuous shard TEST identity differs")
    policy_alerts: dict[str, frozenset[int]] = {}
    selected_scores: np.ndarray | None = None
    for order, row in enumerate(policies):
        if (
            set(row) != {"policy_id", "order", "alerts", "scores", "score_mask"}
            or row["order"] != order
            or not isinstance(row["alerts"], list)
            or not isinstance(row["scores"], list)
            or not isinstance(row["score_mask"], list)
        ):
            raise ValueError("Exp449 continuous policy schema differs")
        scores = np.asarray(row["scores"], dtype=np.float64)
        count = identity["count"]
        if scores.shape != (count,) or not np.isfinite(scores).all():
            raise ValueError("Exp449 continuous policy score shape differs")
        alerts = frozenset(row["alerts"])
        if (
            len(alerts) != len(row["alerts"])
            or any(type(index) is not int or index < 0 or index >= count for index in alerts)
            or row["score_mask"] != [bool(value >= 0.5) for value in scores]
            or alerts != frozenset(int(index) for index in np.flatnonzero(scores >= 0.5))
        ):
            raise ValueError("Exp449 continuous policy alert boundary differs")
        policy_alerts[str(row["policy_id"])] = alerts
        if row["policy_id"] == EXP449_SELECTED_POLICY_ANTIGRAVITY:
            selected_scores = scores
    if selected_scores is None:
        raise ValueError("Exp449 selected continuous policy is absent")
    return (
        policy_alerts[EXP449_SELECTED_POLICY_ANTIGRAVITY],
        selected_scores,
        MappingProxyType(policy_alerts),
    )


def load_exp449_continuous_proposal_antigravity(
    root: Path,
) -> Exp449ContinuousProposalAntigravity:
    """Verify immutable Exp449 manifest/shard bytes and load Candidate-C scores."""
    proposal_root = Path(root)
    manifest_path = proposal_root / EXP449_MANIFEST_RELATIVE_PATH_ANTIGRAVITY
    manifest = read_strict_json(manifest_path)
    entries = manifest.get("entries") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {
            "schema", "context_sha256", "catalog_sha256", "backend_record",
            "entries", "historical_parity_status", "task45_checkpoints",
        }
        or manifest["schema"] != "exp449-proposal-manifest-antigravity.v1"
        or manifest["historical_parity_status"] != "not_used"
        or not isinstance(entries, list)
        or len(entries) != 1117
        or not isinstance(manifest["backend_record"], dict)
    ):
        raise ValueError("Exp449 continuous proposal manifest differs")
    datasets: list[str] = []
    alerts_by_dataset: dict[str, frozenset[int]] = {}
    scores_by_dataset: dict[str, tuple[float, ...]] = {}
    policies_by_dataset: dict[str, Mapping[str, frozenset[int]]] = {}
    previous: str | None = None
    total_rows = 0
    freeze_root = manifest_path.parent
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Exp449 continuous proposal descriptor differs")
        dataset = entry.get("dataset")
        relative_text = entry.get("relative_path")
        relative = PurePosixPath(relative_text) if isinstance(relative_text, str) else None
        if (
            type(dataset) is not str
            or not dataset
            or previous is not None and dataset <= previous
            or relative is None
            or relative.is_absolute()
            or ".." in relative.parts
            or entry.get("schema") != "exp449-proposal-shard-antigravity.v1"
            or type(entry.get("size")) is not int
            or type(entry.get("sha256")) is not str
        ):
            raise ValueError("Exp449 continuous proposal descriptor differs")
        shard_path = freeze_root.joinpath(*relative.parts)
        raw = shard_path.read_bytes()
        if len(raw) != entry["size"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Exp449 continuous proposal shard hash differs")
        shard = read_strict_json(shard_path)
        alerts, scores, policies = _validate_exp449_continuous_shard_antigravity(
            shard,
            dataset=dataset,
            catalog_sha256=manifest["catalog_sha256"],
            backend_record=manifest["backend_record"],
        )
        score_digest = hashlib.sha256(np.asarray(scores, dtype="<f8").tobytes()).hexdigest()
        if score_digest != entry.get("candidate_c_score_sha256"):
            raise ValueError("Exp449 Candidate-C score digest differs")
        datasets.append(dataset)
        alerts_by_dataset[dataset] = alerts
        scores_by_dataset[dataset] = tuple(float(value) for value in scores)
        policies_by_dataset[dataset] = policies
        total_rows += len(scores)
        previous = dataset
    if total_rows != 195900:
        raise ValueError("Exp449 continuous proposal TEST population differs")
    return Exp449ContinuousProposalAntigravity(
        dataset_order=tuple(datasets),
        alerts_by_dataset=alerts_by_dataset,
        scores_by_dataset=scores_by_dataset,
        policy_alerts_by_dataset=policies_by_dataset,
        manifest_sha256=sha256_file(manifest_path),
        context_sha256=str(manifest["context_sha256"]),
        catalog_sha256=str(manifest["catalog_sha256"]),
        test_row_count=total_rows,
    )


def attach_exp449_continuous_metrics_antigravity(
    bundle: EvaluationBundle,
    authority: Exp449ContinuousProposalAntigravity,
    labels: Mapping[str, Sequence[object]],
) -> EvaluationBundle:
    """Replace proposal-only rows with exact metrics from the frozen score vector."""
    normalized: dict[str, tuple[int, ...]] = {}
    for dataset in authority.dataset_order:
        raw = tuple(labels[dataset])
        values = tuple(row[1] for row in raw) if raw and isinstance(raw[0], tuple) else raw
        if any(type(value) is not int or value not in (0, 1) for value in values):
            raise ValueError("Exp449 proposal TEST labels differ")
        normalized[dataset] = tuple(values)
    summaries: list[Mapping[str, object]] = []
    metrics = [dict(row) for row in bundle.dataset_metrics]
    for summary in bundle.summaries:
        if summary.get("method") != "proposed_equipment_anomaly_detection":
            summaries.append(MappingProxyType(dict(summary)))
            continue
        support = str(summary["support"])
        datasets = tuple(
            str(row["dataset"])
            for row in metrics
            if row.get("method") == "proposed_equipment_anomaly_detection"
            and row.get("support") == support
        )
        evaluated = evaluate_rows(
            {dataset: tuple(sorted(authority.alerts_by_dataset[dataset])) for dataset in datasets},
            {dataset: normalized[dataset] for dataset in datasets},
            scores={dataset: np.asarray(authority.scores_by_dataset[dataset]) for dataset in datasets},
        )
        micro, macro = evaluated["micro"], evaluated["macro"]
        changed = dict(summary)
        changed.update(
            alerts=evaluated["alert_count"],
            tp=micro["tp"], fp=micro["fp"], fn=micro["fn"], tn=micro["tn"],
            precision=micro["precision"], recall=micro["recall"], f1=micro["f1"],
            false_positive_rate=micro["false_positive_rate"],
            average_precision=micro["average_precision"], auroc=micro["auroc"],
            macro_precision=macro["precision"], macro_recall=macro["recall"],
            macro_f1=macro["f1"], macro_false_positive_rate=macro["false_positive_rate"],
            macro_average_precision=macro["average_precision"], macro_auroc=macro["auroc"],
            continuous_dataset_count=evaluated["continuous_dataset_count"],
        )
        summaries.append(MappingProxyType(changed))
        by_dataset = {str(row["dataset"]): row for row in evaluated["dataset_rows"]}
        for index, row in enumerate(metrics):
            if row.get("method") == "proposed_equipment_anomaly_detection" and row.get("support") == support:
                metrics[index] = {**row, **dict(by_dataset[str(row["dataset"])])}
    metadata = {
        **dict(bundle.metadata),
        "schema": "paper-public-baselines.evaluation-exp449-continuous-antigravity.v1",
        "proposal_continuous_score_available": True,
        "proposal_score_semantics": "route_preserving_composite_anomaly_score",
        "proposal_score_is_calibrated_probability": False,
        "proposal_policy_id": EXP449_SELECTED_POLICY_ANTIGRAVITY,
        "proposal_manifest_sha256": authority.manifest_sha256,
        "proposal_context_sha256": authority.context_sha256,
        "proposal_catalog_sha256": authority.catalog_sha256,
    }
    return EvaluationBundle(
        metadata=MappingProxyType(metadata),
        summaries=tuple(summaries),
        dataset_metrics=tuple(MappingProxyType(dict(row)) for row in metrics),
        bootstrap=bundle.bootstrap,
        author_ablation=bundle.author_ablation,
    )


def evaluate_frozen_scores_exp449_continuous_antigravity(
    run_root: Path,
    proposal_root: Path,
    *,
    label_loader=None,
) -> EvaluationBundle:
    """Re-evaluate frozen public scores against Exp449 alerts and continuous score."""
    from .runner import verify_score_root

    verify_score_root(Path(run_root))
    authority = load_exp449_continuous_proposal_antigravity(Path(proposal_root))
    if label_loader is None:
        from .label_access import load_test_labels

        active_label_loader = load_test_labels
    else:
        active_label_loader = label_loader
    labels = active_label_loader(authority.dataset_order)
    final_eval = evaluate_rows(
        {dataset: tuple(sorted(authority.alerts_by_dataset[dataset])) for dataset in authority.dataset_order},
        {dataset: tuple(row[1] for row in labels[dataset]) for dataset in authority.dataset_order},
    )
    policy_names = {
        "hard_only": "제안기술 기본 구성",
        "hard_plus_source": "후보 확장 적용 구성",
        "hard_plus_stable": "안정성 선별 적용 제안기술",
        "hard_plus_unstable": "불안정 후보 진단 구성",
    }
    ablation = []
    for policy in EXP449_POLICY_ORDER_ANTIGRAVITY:
        evaluated = evaluate_rows(
            {
                dataset: tuple(sorted(authority.policy_alerts_by_dataset[dataset][policy]))
                for dataset in authority.dataset_order
            },
            {dataset: tuple(row[1] for row in labels[dataset]) for dataset in authority.dataset_order},
        )
        micro = evaluated["micro"]
        ablation.append(
            {
                "paper_name": policy_names[policy],
                "internal_provenance": policy,
                "tp": micro["tp"], "fp": micro["fp"], "fn": micro["fn"], "tn": micro["tn"],
                "f1": micro["f1"], "macro_f1": evaluated["macro"]["f1"],
            }
        )
    proposal = ProposalAuthority(
        dataset_order=authority.dataset_order,
        alerts_by_dataset=authority.alerts_by_dataset,
        expected_confusion=tuple(int(final_eval["micro"][name]) for name in ("tp", "fp", "fn", "tn")),
        author_ablation=tuple(ablation),
        reviewed_commit=authority.context_sha256[:40],
        artifact_map_sha256=authority.manifest_sha256,
    )
    base = evaluate_frozen_scores(
        Path(run_root),
        Path(proposal_root),
        label_loader=lambda _datasets: labels,
        proposal_loader=lambda _root: proposal,
    )
    return attach_exp449_continuous_metrics_antigravity(base, authority, labels)


def combine_official_evaluations_antigravity(
    classical: EvaluationBundle,
    deep_svdd: EvaluationBundle,
    drocc: EvaluationBundle,
    *,
    drocc_parity_status: str,
    require_proposal_continuous_antigravity: bool = False,
) -> OfficialComparisonAntigravity:
    """Combine exact full-population rows without inventing new scores."""
    if drocc_parity_status != "user_confirmed_no_difference":
        raise ValueError("DROCC MPS parity status is not accepted")
    bundles = (classical, deep_svdd, drocc)
    identity_fields = (
        "database_sha256",
        "proposal_reviewed_commit",
        "proposal_artifact_map_sha256",
        "dataset_count",
        "bootstrap_seed",
        "bootstrap_samples",
        "micro_rank_score",
        "macro_rank_score",
    )
    identities = {
        tuple(bundle.metadata.get(field) for field in identity_fields)
        for bundle in bundles
    }
    if len(identities) != 1 or next(iter(identities))[3] != 1117:
        raise ValueError("evaluation database/proposal identity differs")
    expected_evaluation_schema = (
        "paper-public-baselines.evaluation-exp449-continuous-antigravity.v1"
        if require_proposal_continuous_antigravity
        else "paper-public-baselines.evaluation.v1"
    )
    for bundle in bundles:
        if (
            bundle.metadata.get("schema")
            != expected_evaluation_schema
            or bundle.metadata.get("evaluation_status")
            != "retrospective_comparison_complete_no_promotion"
        ):
            raise ValueError("source evaluation status is invalid")

    expected_groups = (
        EXPECTED_METHOD_ORDER_ANTIGRAVITY[:4],
        ("deep_svdd_1d",),
        ("drocc_1d",),
    )
    proposal_rows: list[Mapping[str, object]] = []
    summary_rows: list[Mapping[str, object]] = []
    dataset_rows: list[Mapping[str, object]] = []
    bootstrap_rows: list[Mapping[str, object]] = []
    for bundle, expected_methods in zip(bundles, expected_groups, strict=True):
        full_summaries = tuple(
            row for row in bundle.summaries if row.get("support") == "full_population"
        )
        bundle_methods = tuple(
            dict.fromkeys(
                str(row["method"])
                for row in full_summaries
                if row.get("is_external_baseline") is True
            )
        )
        if bundle_methods != expected_methods:
            raise ValueError("source evaluation method ownership differs")
        for method in expected_methods:
            protocol_keys = {
                (row.get("protocol"), row.get("alpha"))
                for row in full_summaries
                if row.get("method") == method
            }
            expected_protocols = {
                ("conformal", 0.005),
                ("conformal", 0.01),
                ("conformal", 0.02),
                ("matched_alert_budget", None),
            }
            if protocol_keys != expected_protocols:
                raise ValueError("source evaluation protocol coverage differs")
        proposal = tuple(
            row
            for row in full_summaries
            if row.get("method") == "proposed_equipment_anomaly_detection"
            and row.get("protocol") == "proposal_frozen"
        )
        if len(proposal) != 1:
            raise ValueError("source proposal summary is not exact")
        proposal_rows.append(proposal[0])
        summary_rows.extend(
            row for row in full_summaries if row.get("is_external_baseline") is True
        )
        dataset_rows.extend(
            row
            for row in bundle.dataset_metrics
            if row.get("support") == "full_population"
            and row.get("is_external_baseline") is True
        )
        bootstrap_rows.extend(
            row for row in bundle.bootstrap if row.get("support") == "full_population"
        )
    if any(dict(row) != dict(proposal_rows[0]) for row in proposal_rows[1:]):
        raise ValueError("proposal summaries differ across evaluations")
    if any(
        tuple(dict(row) for row in bundle.author_ablation)
        != tuple(dict(row) for row in classical.author_ablation)
        for bundle in bundles[1:]
    ):
        raise ValueError("proposal ablation differs across evaluations")
    proposed = dict(proposal_rows[0])
    continuous_fields = (
        "average_precision", "auroc", "macro_average_precision", "macro_auroc",
    )
    if require_proposal_continuous_antigravity:
        if any(
            type(proposed.get(field)) not in (int, float)
            or not np.isfinite(float(proposed[field]))
            for field in continuous_fields
        ):
            raise ValueError("proposal continuous metrics are required")
        if any(
            bundle.metadata.get("proposal_continuous_score_available") is not True
            or bundle.metadata.get("proposal_score_semantics")
            != "route_preserving_composite_anomaly_score"
            for bundle in bundles
        ):
            raise ValueError("proposal continuous score authority differs")
    elif any(proposed.get(field) is not None for field in continuous_fields):
        raise ValueError("proposal continuous metrics must remain absent")

    ordered_summaries = (proposed, *tuple(dict(row) for row in summary_rows))
    observed_methods = tuple(
        dict.fromkeys(
            str(row["method"])
            for row in ordered_summaries
            if row.get("is_external_baseline") is True
        )
    )
    if observed_methods != EXPECTED_METHOD_ORDER_ANTIGRAVITY:
        raise ValueError("official public method order differs")

    status_by_method: dict[str, dict[str, str]] = {
        method: {} for method in EXPECTED_METHOD_ORDER_ANTIGRAVITY
    }
    for row in dataset_rows:
        if row.get("protocol") == "matched_alert_budget":
            status_by_method[str(row["method"])][str(row["dataset"])] = str(
                row["status"]
            )
    common_names = set.intersection(
        *(
            {dataset for dataset, status in statuses.items() if status == "complete"}
            for statuses in status_by_method.values()
        )
    ) if all(status_by_method.values()) else set()
    source_manifests = {
        group: str(bundle.metadata["score_manifest_sha256"])
        for group, bundle in zip(
            ("classical", "deep_svdd", "drocc_mps"), bundles, strict=True
        )
    }
    metadata = MappingProxyType(
        {
            "schema": (
                OFFICIAL_CONTINUOUS_COMPARISON_SCHEMA_ANTIGRAVITY
                if require_proposal_continuous_antigravity
                else OFFICIAL_COMPARISON_SCHEMA_ANTIGRAVITY
            ),
            "database_sha256": identities.pop()[0],
            "dataset_count": 1117,
            "test_row_count": 195900,
            "public_method_count": 6,
            "six_method_common_support_count": len(common_names),
            "primary_protocol": "matched_alert_budget",
            "conformal_alphas": [0.005, 0.01, 0.02],
            "proposal_continuous_score_available": require_proposal_continuous_antigravity,
            "proposal_score_semantics": (
                "route_preserving_composite_anomaly_score"
                if require_proposal_continuous_antigravity
                else None
            ),
            "drocc_backend": "mps",
            "drocc_mps_parity_status": drocc_parity_status,
            "drocc_mps_parity_evidence": "user_statement_no_machine_readable_artifact",
            "source_score_manifest_sha256": source_manifests,
            "evaluation_status": "retrospective_comparison_complete_no_promotion",
        }
    )
    return OfficialComparisonAntigravity(
        metadata=metadata,
        public_methods=EXPECTED_METHOD_ORDER_ANTIGRAVITY,
        summaries=tuple(MappingProxyType(dict(row)) for row in ordered_summaries),
        dataset_metrics=tuple(
            MappingProxyType(dict(row)) for row in dataset_rows
        ),
        bootstrap=tuple(MappingProxyType(dict(row)) for row in bootstrap_rows),
        author_ablation=tuple(
            MappingProxyType(dict(row)) for row in classical.author_ablation
        ),
    )


def write_official_comparison_antigravity(
    root: Path, comparison: OfficialComparisonAntigravity
) -> str:
    """Create the canonical combined comparison leaves and hash map."""
    destination = Path(root)
    summary = {
        "metadata": dict(comparison.metadata),
        "public_methods": list(comparison.public_methods),
        "summaries": [dict(row) for row in comparison.summaries],
        "author_ablation": [dict(row) for row in comparison.author_ablation],
        "dataset_metric_row_count": len(comparison.dataset_metrics),
        "bootstrap_row_count": len(comparison.bootstrap),
    }
    payloads = {
        OFFICIAL_COMPARISON_FILES_ANTIGRAVITY[0]: canonical_json_bytes(summary),
        OFFICIAL_COMPARISON_FILES_ANTIGRAVITY[1]: _csv_bytes_antigravity(
            SUMMARY_FIELDS, comparison.summaries
        ),
        OFFICIAL_COMPARISON_FILES_ANTIGRAVITY[2]: _csv_bytes_antigravity(
            DATASET_FIELDS, comparison.dataset_metrics
        ),
        OFFICIAL_COMPARISON_FILES_ANTIGRAVITY[3]: canonical_json_bytes(
            {
                "schema": "paper-public-baselines.official-bootstrap-antigravity.v1",
                "rows": [dict(row) for row in comparison.bootstrap],
            }
        ),
    }
    for name, payload in payloads.items():
        write_create_only_bytes(destination / name, payload)
    artifact_map = {
        "schema": "paper-public-baselines.official-comparison-artifacts-antigravity.v1",
        "artifacts": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(payloads.items())
        },
    }
    write_create_only_bytes(
        destination / OFFICIAL_COMPARISON_ARTIFACT_MAP_ANTIGRAVITY,
        canonical_json_bytes(artifact_map),
    )
    verify_official_comparison_antigravity(destination)
    return sha256_file(
        destination / OFFICIAL_COMPARISON_ARTIFACT_MAP_ANTIGRAVITY
    )


def verify_official_comparison_antigravity(
    root: Path,
) -> OfficialComparisonAntigravity:
    """Verify combined artifacts without importing scorers or TEST labels."""
    destination = Path(root)
    expected = {
        *OFFICIAL_COMPARISON_FILES_ANTIGRAVITY,
        OFFICIAL_COMPARISON_ARTIFACT_MAP_ANTIGRAVITY,
    }
    actual = {path.name for path in destination.iterdir() if path.is_file()}
    if actual != expected:
        raise ValueError("official comparison artifact inventory is not exact")
    artifact_map = read_strict_json(
        destination / OFFICIAL_COMPARISON_ARTIFACT_MAP_ANTIGRAVITY
    )
    if (
        set(artifact_map) != {"schema", "artifacts"}
        or artifact_map.get("schema")
        != "paper-public-baselines.official-comparison-artifacts-antigravity.v1"
        or not isinstance(artifact_map.get("artifacts"), dict)
        or set(artifact_map["artifacts"]) != set(OFFICIAL_COMPARISON_FILES_ANTIGRAVITY)
    ):
        raise ValueError("official comparison artifact map is invalid")
    for name, expected_sha in artifact_map["artifacts"].items():
        if sha256_file(destination / name) != expected_sha:
            raise ValueError("official comparison artifact SHA256 mismatch")
    summary = read_strict_json(destination / OFFICIAL_COMPARISON_FILES_ANTIGRAVITY[0])
    metadata = summary.get("metadata")
    methods = summary.get("public_methods")
    summaries = summary.get("summaries")
    ablation = summary.get("author_ablation")
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema") not in {
            OFFICIAL_COMPARISON_SCHEMA_ANTIGRAVITY,
            OFFICIAL_CONTINUOUS_COMPARISON_SCHEMA_ANTIGRAVITY,
        }
        or metadata.get("dataset_count") != 1117
        or metadata.get("public_method_count") != 6
        or metadata.get("proposal_continuous_score_available")
        is not (
            metadata.get("schema")
            == OFFICIAL_CONTINUOUS_COMPARISON_SCHEMA_ANTIGRAVITY
        )
        or metadata.get("drocc_mps_parity_status")
        != "user_confirmed_no_difference"
        or methods != list(EXPECTED_METHOD_ORDER_ANTIGRAVITY)
        or not isinstance(summaries, list)
        or not isinstance(ablation, list)
    ):
        raise ValueError("official comparison summary is invalid")
    bootstrap = read_strict_json(
        destination / OFFICIAL_COMPARISON_FILES_ANTIGRAVITY[3]
    )
    if (
        bootstrap.get("schema")
        != "paper-public-baselines.official-bootstrap-antigravity.v1"
        or not isinstance(bootstrap.get("rows"), list)
    ):
        raise ValueError("official comparison bootstrap is invalid")
    _require_csv_header_antigravity(
        destination / OFFICIAL_COMPARISON_FILES_ANTIGRAVITY[1], SUMMARY_FIELDS
    )
    _require_csv_header_antigravity(
        destination / OFFICIAL_COMPARISON_FILES_ANTIGRAVITY[2], DATASET_FIELDS
    )
    return OfficialComparisonAntigravity(
        metadata=MappingProxyType(dict(metadata)),
        public_methods=tuple(str(value) for value in methods),
        summaries=tuple(MappingProxyType(dict(row)) for row in summaries),
        dataset_metrics=(),
        bootstrap=tuple(
            MappingProxyType(dict(row)) for row in bootstrap["rows"]
        ),
        author_ablation=tuple(MappingProxyType(dict(row)) for row in ablation),
    )


def _csv_bytes_antigravity(
    fields: tuple[str, ...], rows: tuple[Mapping[str, object], ...]
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    return stream.getvalue().encode("utf-8")


def _require_csv_header_antigravity(path: Path, fields: tuple[str, ...]) -> None:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if tuple(next(reader, ())) != fields:
            raise ValueError("official comparison CSV header is invalid")


__all__ = [
    "EXPECTED_METHOD_ORDER_ANTIGRAVITY",
    "Exp449ContinuousProposalAntigravity",
    "OfficialComparisonAntigravity",
    "attach_exp449_continuous_metrics_antigravity",
    "combine_official_evaluations_antigravity",
    "evaluate_frozen_scores_exp449_continuous_antigravity",
    "load_exp449_continuous_proposal_antigravity",
    "verify_official_comparison_antigravity",
    "write_official_comparison_antigravity",
]
