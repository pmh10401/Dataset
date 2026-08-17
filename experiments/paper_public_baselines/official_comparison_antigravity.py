"""Canonical six-method comparison over independently verified evaluations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .artifacts import (
    canonical_json_bytes,
    read_strict_json,
    sha256_file,
    write_create_only_bytes,
)
from .evaluation import EvaluationBundle
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


def combine_official_evaluations_antigravity(
    classical: EvaluationBundle,
    deep_svdd: EvaluationBundle,
    drocc: EvaluationBundle,
    *,
    drocc_parity_status: str,
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
    for bundle in bundles:
        if (
            bundle.metadata.get("schema")
            != "paper-public-baselines.evaluation.v1"
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
    if any(
        proposed.get(field) is not None
        for field in (
            "average_precision",
            "auroc",
            "macro_average_precision",
            "macro_auroc",
        )
    ):
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
            "schema": OFFICIAL_COMPARISON_SCHEMA_ANTIGRAVITY,
            "database_sha256": identities.pop()[0],
            "dataset_count": 1117,
            "test_row_count": 195900,
            "public_method_count": 6,
            "six_method_common_support_count": len(common_names),
            "primary_protocol": "matched_alert_budget",
            "conformal_alphas": [0.005, 0.01, 0.02],
            "proposal_continuous_score_available": False,
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
        or metadata.get("schema") != OFFICIAL_COMPARISON_SCHEMA_ANTIGRAVITY
        or metadata.get("dataset_count") != 1117
        or metadata.get("public_method_count") != 6
        or metadata.get("proposal_continuous_score_available") is not False
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
    "OfficialComparisonAntigravity",
    "combine_official_evaluations_antigravity",
    "verify_official_comparison_antigravity",
    "write_official_comparison_antigravity",
]
