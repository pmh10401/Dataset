"""Deterministic Exp414 missing-aware qualification artifact rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict, is_dataclass
from io import StringIO
import json
import math

from experiments.exp411_near_repeat_qualification.contracts import (
    DatasetBackendAudit,
    DatasetQualification,
)

from .config import FROZEN_CONFIG, Exp414FrozenConfig
from .contracts import DatasetRepairAudit


CONFIG_FILE = "exp414_frozen_config.json"
REPAIR_FILE = "exp414_missing_repair_audit.csv"
DATASET_FILE = "exp414_dataset_qualification.csv"
QUANTILE_FILE = "exp414_quantile_summary.csv"
BACKEND_FILE = "exp414_backend_assessment.json"
RULE_FILE = "exp414_selected_rule.json"
SUMMARY_FILE = "exp414_summary.json"
REPORT_FILE = "EXP414_RESULT_REPORT_KO.md"
MANIFEST_FILE = "MANIFEST.json"

_PAYLOAD_FILES = {
    CONFIG_FILE,
    REPAIR_FILE,
    DATASET_FILE,
    QUANTILE_FILE,
    BACKEND_FILE,
    RULE_FILE,
    SUMMARY_FILE,
    REPORT_FILE,
}
_REPAIR_FIELDS = (
    "dataset",
    "instance_index",
    "raw_values_sha256",
    "repaired_values_sha256",
    "length",
    "nan_count",
    "posinf_count",
    "neginf_count",
    "missing_positions_sha256",
    "repair_method",
)
_DATASET_FIELDS = (
    "dataset",
    "family",
    "execution_source",
    "repair_status",
    "identity_group_count",
    "normal_total",
    "normal_abstained",
    "normal_near_json",
    "synthetic_total_json",
    "synthetic_near_json",
    "perturbation_total_json",
    "perturbation_near_json",
    "backend",
    "backend_sampled_pairs_json",
    "backend_max_abs_error",
    "backend_fallback_reason",
)
_QUANTILE_FIELDS = (
    "subgroup",
    "selection_authority",
    "status",
    "selected_quantile",
    "dataset_count",
    "family_count",
    "quantile",
    "normal_acceptance",
    "pooled_synthetic",
    "worst_perturbation",
    "bootstrap_upper",
    "passes_gate",
)


def render_payloads(
    records: Sequence[Mapping[str, object]],
    repair_audits: Sequence[DatasetRepairAudit],
    aggregate: Mapping[str, object],
    subgroups: Mapping[str, object],
    backend_assessment: Mapping[str, object],
    provenance: Mapping[str, object],
    *,
    full_qualification_executed: bool,
    config: Exp414FrozenConfig = FROZEN_CONFIG,
) -> Mapping[str, bytes]:
    """Render exactly eight deterministic, TRAIN-only Exp414 payloads."""
    config.validate()
    _require_train_only_provenance(provenance)
    status, selected = _selected_rule_values(aggregate)
    ordered_records = tuple(
        sorted(records, key=lambda record: _record_dataset(record))
    )
    ordered_audits = tuple(
        sorted(repair_audits, key=lambda audit: audit.dataset)
    )
    record_datasets = tuple(
        _record_dataset(record)
        for record in ordered_records
    )
    audit_datasets = tuple(audit.dataset for audit in ordered_audits)
    if (
        not record_datasets
        or len(set(record_datasets)) != len(record_datasets)
        or len(set(audit_datasets)) != len(audit_datasets)
        or record_datasets != audit_datasets
    ):
        raise ValueError("Exp414 reporting dataset coverage is invalid")

    audit_by_dataset = {
        audit.dataset: audit
        for audit in ordered_audits
    }
    dataset_rows = tuple(
        _dataset_row(
            record,
            repair_status=audit_by_dataset[_record_dataset(record)].status,
        )
        for record in ordered_records
    )
    repair_rows = _repair_rows(ordered_audits)
    quantile_rows = _quantile_rows(subgroups, config)
    plain_subgroups = _plain(subgroups)
    summary = {
        "experiment_id": config.experiment_id,
        "mode": (
            "full_qualification"
            if full_qualification_executed
            else "representative_smoke"
        ),
        "dataset_count": int(aggregate["dataset_count"]),
        "family_count": int(aggregate["family_count"]),
        "status": status,
        "selected_quantile": selected,
        "full_qualification_executed": bool(full_qualification_executed),
        "scope": "Exp414 TRAIN-only missing-aware qualification",
        "operational_policy_changed": False,
        "exp412_prediction_created": False,
        "exp413_audit_created": False,
        "test_values_used": False,
        "test_labels_used": False,
        "subgroups": plain_subgroups,
        "provenance": _plain(provenance),
    }
    selected_rule = {
        "status": status,
        "selected_quantile": selected,
        "scope": "Exp414 TRAIN-only missing-aware qualification",
        "selection_source": "all",
        "operational_policy_changed": False,
        "automatic_alert_promotion": False,
        "prediction_created": False,
        "audit_created": False,
    }
    payloads = {
        CONFIG_FILE: _json_bytes(config.to_payload()),
        REPAIR_FILE: _csv_bytes(repair_rows, _REPAIR_FIELDS),
        DATASET_FILE: _csv_bytes(dataset_rows, _DATASET_FIELDS),
        QUANTILE_FILE: _csv_bytes(quantile_rows, _QUANTILE_FIELDS),
        BACKEND_FILE: _json_bytes(_plain(backend_assessment)),
        RULE_FILE: _json_bytes(selected_rule),
        SUMMARY_FILE: _json_bytes(summary),
        REPORT_FILE: _korean_report(summary, quantile_rows).encode("utf-8"),
    }
    if set(payloads) != _PAYLOAD_FILES or len(payloads) != 8:
        raise ValueError("Exp414 rendered artifact coverage mismatch")
    return payloads


def _record_dataset(record: Mapping[str, object]) -> str:
    if not isinstance(record, Mapping):
        raise ValueError("Exp414 qualification record is invalid")
    qualification = record.get("qualification")
    if not isinstance(qualification, DatasetQualification):
        raise ValueError("Exp414 qualification record is invalid")
    return qualification.dataset


def _dataset_row(
    record: Mapping[str, object],
    *,
    repair_status: str,
) -> dict[str, object]:
    qualification = record.get("qualification")
    backend_audit = record.get("backend_audit")
    if not isinstance(qualification, DatasetQualification) or not isinstance(
        backend_audit,
        DatasetBackendAudit,
    ):
        raise ValueError("Exp414 qualification record is invalid")
    source = record.get("execution_source")
    if source not in {"exp414_computed", "migrated_exp411_verified"}:
        raise ValueError("Exp414 reporting execution source is invalid")
    if repair_status not in {"ready", "missing_support_abstain"}:
        raise ValueError("Exp414 reporting repair status is invalid")
    recorded_status = record.get("repair_status")
    if recorded_status is not None and recorded_status != repair_status:
        raise ValueError("Exp414 reporting repair status mismatch")
    maximum_error = float(backend_audit.max_abs_error)
    if not math.isfinite(maximum_error) or maximum_error < 0.0:
        raise ValueError("Exp414 reporting backend error must be finite")
    values = asdict(qualification)
    return {
        "dataset": values["dataset"],
        "family": values["family"],
        "execution_source": source,
        "repair_status": repair_status,
        "identity_group_count": values["identity_group_count"],
        "normal_total": values["normal_total"],
        "normal_abstained": values["normal_abstained"],
        "normal_near_json": _canonical_cell(values["normal_near"]),
        "synthetic_total_json": _canonical_cell(values["synthetic_total"]),
        "synthetic_near_json": _canonical_cell(values["synthetic_near"]),
        "perturbation_total_json": _canonical_cell(
            values["perturbation_total"]
        ),
        "perturbation_near_json": _canonical_cell(values["perturbation_near"]),
        "backend": backend_audit.backend,
        "backend_sampled_pairs_json": _canonical_cell(
            backend_audit.sampled_pairs
        ),
        "backend_max_abs_error": maximum_error,
        "backend_fallback_reason": backend_audit.fallback_reason,
    }


def _repair_rows(
    audits: Sequence[DatasetRepairAudit],
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    observed: set[tuple[str, int]] = set()
    for audit in audits:
        if not isinstance(audit, DatasetRepairAudit) or not audit.runs:
            raise ValueError("Exp414 repair audit is invalid")
        for run in sorted(
            audit.runs,
            key=lambda value: (value.dataset, value.instance_index),
        ):
            identity = (run.dataset, int(run.instance_index))
            if run.dataset != audit.dataset or identity in observed:
                raise ValueError("Exp414 repair audit coverage is invalid")
            observed.add(identity)
            rows.append(
                {
                    "dataset": run.dataset,
                    "instance_index": int(run.instance_index),
                    "raw_values_sha256": run.raw_values_sha256,
                    "repaired_values_sha256": run.repaired_values_sha256,
                    "length": int(run.length),
                    "nan_count": int(run.nan_count),
                    "posinf_count": int(run.posinf_count),
                    "neginf_count": int(run.neginf_count),
                    "missing_positions_sha256": (
                        run.missing_positions_sha256
                    ),
                    "repair_method": run.repair_method,
                }
            )
    return tuple(
        sorted(
            rows,
            key=lambda row: (str(row["dataset"]), int(row["instance_index"])),
        )
    )


def _quantile_rows(
    subgroups: Mapping[str, object],
    config: Exp414FrozenConfig,
) -> tuple[dict[str, object], ...]:
    if not isinstance(subgroups, Mapping) or "all" not in subgroups:
        raise ValueError("Exp414 subgroup summaries are invalid")
    rows: list[dict[str, object]] = []
    names = sorted(subgroups, key=lambda name: (str(name) != "all", str(name)))
    for raw_name in names:
        name = str(raw_name)
        summary = subgroups[raw_name]
        if not isinstance(summary, Mapping):
            raise ValueError("Exp414 subgroup summary is invalid")
        authority = summary.get("selection_authority")
        if not isinstance(authority, bool) or authority != (name == "all"):
            raise ValueError("Exp414 subgroup selection authority is invalid")
        subgroup_status = str(summary.get("status"))
        if subgroup_status not in {
            "qualified",
            "no_qualified_near_rule",
            "not_applicable",
        }:
            raise ValueError("Exp414 subgroup status is invalid")
        quantiles = summary.get("quantiles")
        if not isinstance(quantiles, Mapping):
            raise ValueError("Exp414 subgroup quantiles are invalid")
        common = {
            "subgroup": name,
            "selection_authority": authority,
            "status": subgroup_status,
            "selected_quantile": summary.get("selected_quantile"),
            "dataset_count": int(summary["dataset_count"]),
            "family_count": int(summary["family_count"]),
        }
        if not quantiles:
            rows.append(
                {
                    **common,
                    "quantile": None,
                    "normal_acceptance": None,
                    "pooled_synthetic": None,
                    "worst_perturbation": None,
                    "bootstrap_upper": None,
                    "passes_gate": False,
                }
            )
            continue
        for quantile, raw_metrics in sorted(
            quantiles.items(),
            key=lambda item: float(item[0]),
        ):
            if not isinstance(raw_metrics, Mapping):
                raise ValueError("Exp414 subgroup quantile metrics are invalid")
            metrics = {
                key: float(raw_metrics[key])
                for key in (
                    "normal_acceptance",
                    "pooled_synthetic",
                    "worst_perturbation",
                    "bootstrap_upper",
                )
            }
            if not all(math.isfinite(value) for value in metrics.values()):
                raise ValueError(
                    "Exp414 subgroup quantile metrics must be finite"
                )
            rows.append(
                {
                    **common,
                    "quantile": float(quantile),
                    **metrics,
                    "passes_gate": _passes_summary_gate(metrics, config),
                }
            )
    return tuple(rows)


def _selected_rule_values(
    aggregate: Mapping[str, object],
) -> tuple[str, object]:
    if not isinstance(aggregate, Mapping):
        raise ValueError("Exp414 aggregate is invalid")
    status = str(aggregate.get("status"))
    selected = aggregate.get("selected_quantile")
    if status not in {"qualified", "no_qualified_near_rule"}:
        raise ValueError("Exp414 selected rule status is invalid")
    if (status == "qualified") != (selected is not None):
        raise ValueError("Exp414 selected rule and status disagree")
    return status, selected


def _passes_summary_gate(
    row: Mapping[str, float],
    config: Exp414FrozenConfig,
) -> bool:
    return (
        row["normal_acceptance"] >= config.minimum_normal_acceptance
        and row["pooled_synthetic"]
        <= config.pooled_synthetic_false_normal_max
        and row["worst_perturbation"]
        <= config.per_perturbation_false_normal_max
        and row["bootstrap_upper"] <= config.bootstrap_upper_max
    )


def _canonical_cell(value: object) -> str:
    return json.dumps(
        _plain(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _csv_bytes(
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                name: "" if row.get(name) is None else row.get(name)
                for name in fieldnames
            }
        )
    return buffer.getvalue().encode("utf-8")


def _plain(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _plain(child)
            for key, child in sorted(
                value.items(),
                key=lambda item: str(item[0]),
            )
        }
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    return value


def _require_train_only_provenance(provenance: Mapping[str, object]) -> None:
    if not isinstance(provenance, Mapping):
        raise ValueError("Exp414 provenance is invalid")
    for name in ("test_values_used", "test_labels_used"):
        if name in provenance and provenance[name] is not False:
            raise ValueError("Exp414 provenance violates TRAIN-only scope")


def _korean_report(
    summary: Mapping[str, object],
    quantile_rows: Sequence[Mapping[str, object]],
) -> str:
    lines = [
        "# Exp414 TRAIN-only 결측 인지 near-repeat 자격 검증 결과",
        "",
        "## 핵심 결과",
        "",
        f"- 실행 모드: `{summary['mode']}`",
        (
            "- 데이터셋 / family 수: "
            f"`{summary['dataset_count']}` / `{summary['family_count']}`"
        ),
        f"- 전체 모집단 판정: `{summary['status']}`",
        f"- 선택 quantile: `{summary['selected_quantile']}`",
        "",
        "Exp414는 정상 TRAIN만 사용하는 결측 인지 자격 검증이다. TEST 값과",
        "TEST label을 사용하지 않았으며, 결측 복구 audit을 모든 원본 TRAIN",
        "run에 대해 기록한다.",
        "",
        "## 전체 모집단 gate",
        "",
        "| quantile | normal acceptance | pooled synthetic | worst perturbation | bootstrap upper | 통과 |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in quantile_rows:
        if row["subgroup"] != "all" or row["quantile"] is None:
            continue
        lines.append(
            "| {quantile} | {normal_acceptance:.6f} | "
            "{pooled_synthetic:.6f} | {worst_perturbation:.6f} | "
            "{bootstrap_upper:.6f} | {passes_gate} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## 진단 subgroup과 운영 경계",
            "",
            "- 결측 여부 및 family subgroup은 진단 전용이며 선택 권한이 없다.",
            "- 이 결과는 운영 정책을 변경하지 않는다.",
            "- Exp412 prediction과 Exp413 audit은 생성하지 않았다.",
            "- 후속 prediction과 독립 audit은 각각 Exp415와 Exp416의 별도 작업이다.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["MANIFEST_FILE", "render_payloads"]
