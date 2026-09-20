"""Deterministic Exp411 qualification artifact rendering."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict
from io import StringIO
import json

from experiments.exp403_408.artifact_io import canonical_json_bytes

from .config import FROZEN_CONFIG, Exp411FrozenConfig
from .contracts import DatasetQualification


CONFIG_FILE = "exp411_frozen_config.json"
DATASET_FILE = "exp411_dataset_qualification.csv"
QUANTILE_FILE = "exp411_quantile_summary.csv"
BACKEND_FILE = "exp411_backend_assessment.json"
RULE_FILE = "exp411_selected_rule.json"
SUMMARY_FILE = "exp411_summary.json"
REPORT_FILE = "EXP411_RESULT_REPORT_KO.md"
MANIFEST_FILE = "MANIFEST.json"


def render_payloads(
    rows: Sequence[DatasetQualification],
    aggregate: Mapping[str, object],
    backend_assessment: Mapping[str, object],
    provenance: Mapping[str, object],
    *,
    full_qualification_executed: bool,
    config: Exp411FrozenConfig = FROZEN_CONFIG,
) -> Mapping[str, bytes]:
    config.validate()
    ordered = tuple(sorted(rows, key=lambda row: row.dataset))
    status = str(aggregate.get("status"))
    selected = aggregate.get("selected_quantile")
    if status not in {"qualified", "no_qualified_near_rule"}:
        raise ValueError("Exp411 selected rule status is invalid")
    if (status == "qualified") != (selected is not None):
        raise ValueError("Exp411 selected rule and status disagree")

    dataset_rows = [_dataset_row(row) for row in ordered]
    quantile_rows = [
        {
            "quantile": float(quantile),
            **{
                str(key): float(value)
                for key, value in dict(metrics).items()
            },
            "passes_gate": _passes_summary_gate(dict(metrics), config),
        }
        for quantile, metrics in sorted(
            _mapping(aggregate["quantiles"], "quantiles").items(),
            key=lambda item: float(item[0]),
        )
    ]
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
        "operational_policy_changed": False,
        "exp412_prediction_created": False,
        "test_labels_used": False,
        "test_values_used": False,
        "provenance": _plain(provenance),
    }
    selected_rule = {
        "status": status,
        "selected_quantile": selected,
        "scope": "Exp411 TRAIN-only qualification",
        "operational_policy_changed": False,
        "automatic_alert_promotion": False,
    }
    payloads = {
        CONFIG_FILE: canonical_json_bytes(config.to_payload()) + b"\n",
        DATASET_FILE: _csv_bytes(
            dataset_rows,
            (
                "dataset",
                "family",
                "identity_group_count",
                "normal_total",
                "normal_abstained",
                "normal_near_json",
                "synthetic_total_json",
                "synthetic_near_json",
                "perturbation_total_json",
                "perturbation_near_json",
            ),
        ),
        QUANTILE_FILE: _csv_bytes(
            quantile_rows,
            (
                "quantile",
                "normal_acceptance",
                "pooled_synthetic",
                "worst_perturbation",
                "bootstrap_upper",
                "passes_gate",
            ),
        ),
        BACKEND_FILE: canonical_json_bytes(_plain(backend_assessment)) + b"\n",
        RULE_FILE: canonical_json_bytes(selected_rule) + b"\n",
        SUMMARY_FILE: canonical_json_bytes(summary) + b"\n",
        REPORT_FILE: _korean_report(summary, quantile_rows).encode("utf-8"),
    }
    expected = {
        CONFIG_FILE,
        DATASET_FILE,
        QUANTILE_FILE,
        BACKEND_FILE,
        RULE_FILE,
        SUMMARY_FILE,
        REPORT_FILE,
    }
    if set(payloads) != expected:
        raise ValueError("Exp411 rendered artifact coverage mismatch")
    return payloads


def _dataset_row(row: DatasetQualification) -> dict[str, object]:
    values = asdict(row)
    return {
        "dataset": values["dataset"],
        "family": values["family"],
        "identity_group_count": values["identity_group_count"],
        "normal_total": values["normal_total"],
        "normal_abstained": values["normal_abstained"],
        "normal_near_json": _canonical_cell(values["normal_near"]),
        "synthetic_total_json": _canonical_cell(values["synthetic_total"]),
        "synthetic_near_json": _canonical_cell(values["synthetic_near"]),
        "perturbation_total_json": _canonical_cell(values["perturbation_total"]),
        "perturbation_near_json": _canonical_cell(values["perturbation_near"]),
    }


def _canonical_cell(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _csv_bytes(
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        extrasaction="raise",
        lineterminator="\n",
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


def _korean_report(
    summary: Mapping[str, object],
    quantiles: Sequence[Mapping[str, object]],
) -> str:
    lines = [
        "# Exp411 TRAIN-only near-repeat 자격 검증 결과",
        "",
        "## 핵심 결과",
        "",
        f"- 실행 모드: `{summary['mode']}`",
        f"- 데이터셋 / family 수: `{summary['dataset_count']}` / `{summary['family_count']}`",
        f"- 판정: `{summary['status']}`",
        f"- 선택 quantile: `{summary['selected_quantile']}`",
        "",
        "이 결과는 정상 TRAIN 데이터만으로 exact/near/unfamiliar 구분 규칙을",
        "검증한다. TEST 값, TEST label, 실제 이상 위치는 사용하지 않았다.",
        "",
        "## 사전등록 gate",
        "",
        "| quantile | normal acceptance | pooled synthetic | worst perturbation | bootstrap upper | 통과 |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in quantiles:
        lines.append(
            "| {quantile} | {normal_acceptance:.6f} | {pooled_synthetic:.6f} | "
            "{worst_perturbation:.6f} | {bootstrap_upper:.6f} | {passes_gate} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## 운영 경계",
            "",
            "- 이 결과만으로 운영 정책을 바꾸지 않는다.",
            "- Exp412 prediction은 생성하지 않았다.",
            "- review/shadow 후보를 자동 경보로 승격하지 않는다.",
            "",
        ]
    )
    return "\n".join(lines)


def _passes_summary_gate(
    row: Mapping[str, object],
    config: Exp411FrozenConfig,
) -> bool:
    return (
        float(row["normal_acceptance"]) >= config.minimum_normal_acceptance
        and float(row["pooled_synthetic"])
        <= config.pooled_synthetic_false_normal_max
        and float(row["worst_perturbation"])
        <= config.per_perturbation_false_normal_max
        and float(row["bootstrap_upper"]) <= config.bootstrap_upper_max
    )


def _mapping(value: object, name: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Exp411 {name} must be a mapping")
    return value


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(child) for key, child in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_plain(child) for child in value]
    return value


__all__ = [
    "MANIFEST_FILE",
    "render_payloads",
]
