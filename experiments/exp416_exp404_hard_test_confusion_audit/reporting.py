"""Deterministic reporting and independent byte-level revalidation for Exp416."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import asdict
from io import StringIO
import json
import math
import re

from .config import Exp416FrozenConfig, FROZEN_CONFIG
from .contracts import (
    AuditResult,
    ConfusionCounts,
    ConfusionMetrics,
    DatasetAuditRow,
    FrozenLabels,
    FrozenPrediction,
    MacroMetric,
    metrics_from_counts,
)


SUMMARY_FILE = "exp416_confusion_matrix.json"
DATASET_FILE = "exp416_dataset_metrics.csv"
REPORT_FILE = "EXP416_RESULT_REPORT_KO.md"
MANIFEST_FILE = "MANIFEST.json"
_PAYLOAD_FILES = {SUMMARY_FILE, DATASET_FILE, REPORT_FILE}
_DATASET_FIELDS = (
    "dataset",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "specificity",
    "false_positive_rate",
    "false_negative_rate",
    "f1",
)
_COUNT_NAMES = ("tp", "fp", "fn", "tn")
_METRIC_NAMES = (
    "precision",
    "recall",
    "specificity",
    "false_positive_rate",
    "false_negative_rate",
    "f1",
)
_SUMMARY_FIELDS = {
    "experiment_id",
    "scope",
    "git_commit",
    "prediction_identity_sha256",
    "input_sha256",
    "dataset_count",
    "prediction_dataset_count",
    "prediction_coordinate_count",
    "frozen_label_row_count",
    "micro",
    "macro",
    "prediction_modified",
    "test_labels_loaded_after_prediction_freeze",
    "test_informed_reselection",
    "operational_policy_changed",
    "exp415_prediction_created",
}
_SCOPE = "Exp404 frozen Hard point-level TEST audit"
_HEX_40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")


def render_payloads(
    result: AuditResult,
    frozen_prediction: FrozenPrediction,
    frozen_labels: FrozenLabels,
    database_sha256: str,
    git_commit: str,
    *,
    config: Exp416FrozenConfig = FROZEN_CONFIG,
) -> Mapping[str, bytes]:
    """Render the exact three deterministic non-manifest Exp416 artifacts."""
    config.validate()
    _validate_render_inputs(
        result,
        frozen_prediction,
        frozen_labels,
        database_sha256,
        git_commit,
        config,
    )
    summary = {
        "experiment_id": "Exp416",
        "scope": _SCOPE,
        "git_commit": git_commit,
        "prediction_identity_sha256": frozen_prediction.identity_sha256,
        "input_sha256": {
            "database_reference": database_sha256,
            "prediction": frozen_prediction.prediction_sha256,
            "manifest": frozen_prediction.manifest_sha256,
            "frozen_test_labels": frozen_labels.labels_sha256,
        },
        "dataset_count": len(result.dataset_rows),
        "prediction_dataset_count": frozen_prediction.prediction_dataset_count,
        "prediction_coordinate_count": result.prediction_coordinates,
        "frozen_label_row_count": result.label_rows,
        "micro": {
            "counts": asdict(result.micro_counts),
            "metrics": asdict(result.micro_metrics),
        },
        "macro": {
            name: asdict(metric)
            for name, metric in sorted(result.macro_metrics.items())
        },
        "prediction_modified": False,
        "test_labels_loaded_after_prediction_freeze": True,
        "test_informed_reselection": False,
        "operational_policy_changed": False,
        "exp415_prediction_created": False,
    }
    rows = tuple(_dataset_row(row) for row in result.dataset_rows)
    payloads = {
        SUMMARY_FILE: _json_bytes(summary),
        DATASET_FILE: _csv_bytes(rows),
        REPORT_FILE: _korean_report(summary).encode("utf-8"),
    }
    if set(payloads) != _PAYLOAD_FILES or len(payloads) != 3:
        raise ValueError("Exp416 rendered payload coverage mismatch")
    verify_rendered_payloads(payloads, config=config)
    return payloads


def verify_rendered_payloads(
    payloads: Mapping[str, bytes],
    *,
    config: Exp416FrozenConfig = FROZEN_CONFIG,
) -> None:
    """Parse and independently recompute every count and metric in payloads."""
    config.validate()
    if (
        not isinstance(payloads, Mapping)
        or set(payloads) != _PAYLOAD_FILES
        or len(payloads) != 3
    ):
        raise ValueError("Exp416 rendered payload file coverage mismatch")
    if any(
        type(name) is not str or type(value) is not bytes
        for name, value in payloads.items()
    ):
        raise ValueError("Exp416 rendered payload names and values are invalid")

    summary = _parse_summary(payloads[SUMMARY_FILE], config)
    rows = _parse_dataset_csv(payloads[DATASET_FILE])
    dataset_count = _exact_int(summary["dataset_count"], "dataset_count")
    if dataset_count != config.expected_datasets or len(rows) != dataset_count:
        raise ValueError("Exp416 CSV dataset count mismatch")

    micro = _require_mapping(summary["micro"], "micro")
    micro_counts = _parse_counts(micro["counts"], "micro counts")
    micro_metrics = _parse_metrics(micro["metrics"], "micro metrics")
    if micro_metrics != metrics_from_counts(micro_counts):
        raise ValueError("Exp416 micro metrics do not match counts")
    frozen_label_rows = _exact_int(
        summary["frozen_label_row_count"],
        "frozen_label_row_count",
    )
    if (
        frozen_label_rows != config.expected_label_rows
        or micro_counts.total != frozen_label_rows
    ):
        raise ValueError("Exp416 micro total does not equal frozen label rows")

    prediction_coordinates = _exact_int(
        summary["prediction_coordinate_count"],
        "prediction_coordinate_count",
    )
    if (
        prediction_coordinates != config.expected_prediction_coordinates
        or prediction_coordinates != micro_counts.tp + micro_counts.fp
    ):
        raise ValueError("Exp416 prediction coordinate count mismatch")

    csv_counts = ConfusionCounts(
        tp=sum(row.counts.tp for row in rows),
        fp=sum(row.counts.fp for row in rows),
        fn=sum(row.counts.fn for row in rows),
        tn=sum(row.counts.tn for row in rows),
    )
    if csv_counts != micro_counts:
        raise ValueError("Exp416 CSV count sum does not match micro counts")

    macro = _require_mapping(summary["macro"], "macro")
    _require_exact_keys(macro, set(_METRIC_NAMES), "macro schema")
    for name in _METRIC_NAMES:
        observed = _parse_macro_metric(macro[name], name, dataset_count)
        values = [
            value
            for row in rows
            if (value := getattr(row.metrics, name)) is not None
        ]
        expected = MacroMetric(
            value=(sum(values) / len(values) if values else None),
            defined_dataset_count=len(values),
        )
        if observed != expected:
            raise ValueError(f"Exp416 macro metric mismatch: {name}")

    if payloads[REPORT_FILE] != _korean_report(summary).encode("utf-8"):
        raise ValueError("Exp416 Markdown report does not match verified summary")


def _validate_render_inputs(
    result: AuditResult,
    prediction: FrozenPrediction,
    labels: FrozenLabels,
    database_sha256: str,
    git_commit: str,
    config: Exp416FrozenConfig,
) -> None:
    if type(result) is not AuditResult:
        raise ValueError("Exp416 reporting requires AuditResult")
    if type(prediction) is not FrozenPrediction:
        raise ValueError("Exp416 reporting requires FrozenPrediction")
    if type(labels) is not FrozenLabels:
        raise ValueError("Exp416 reporting requires FrozenLabels")
    _require_digest(database_sha256, "database SHA256")
    _require_digest(prediction.identity_sha256, "prediction identity SHA256")
    _require_digest(prediction.prediction_sha256, "prediction SHA256")
    _require_digest(prediction.manifest_sha256, "manifest SHA256")
    _require_digest(labels.labels_sha256, "frozen label SHA256")
    if type(git_commit) is not str or _HEX_40.fullmatch(git_commit) is None:
        raise ValueError("Exp416 git commit must be 40 lowercase hex characters")
    if (
        database_sha256 != config.database_sha256
        or prediction.prediction_sha256 != config.prediction_sha256
        or prediction.manifest_sha256 != config.manifest_sha256
        or labels.labels_sha256 != config.labels_sha256
    ):
        raise ValueError("Exp416 reporting input SHA256 mismatch")
    result_datasets = tuple(row.dataset for row in result.dataset_rows)
    if (
        prediction.dataset_scope != labels.dataset_scope
        or result_datasets != labels.dataset_scope
        or len(result.dataset_rows) != config.expected_datasets
        or prediction.prediction_dataset_count
        != config.expected_prediction_datasets
        or result.prediction_coordinates
        != config.expected_prediction_coordinates
        or result.label_rows != config.expected_label_rows
        or len(labels.rows) != result.label_rows
        or len(prediction.coordinates) != result.prediction_coordinates
    ):
        raise ValueError("Exp416 reporting input coverage mismatch")


def _dataset_row(row: DatasetAuditRow) -> dict[str, object]:
    return {
        "dataset": row.dataset,
        **asdict(row.counts),
        **asdict(row.metrics),
    }


def _parse_summary(
    payload: bytes,
    config: Exp416FrozenConfig,
) -> Mapping[str, object]:
    try:
        text = payload.decode("utf-8")
        summary = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("Exp416 summary JSON is invalid") from error
    summary = _require_mapping(summary, "summary")
    _require_exact_keys(summary, _SUMMARY_FIELDS, "summary schema")
    if _json_bytes(summary) != payload:
        raise ValueError("Exp416 summary JSON is not canonical")
    if summary["experiment_id"] != config.experiment_id or type(
        summary["experiment_id"]
    ) is not str:
        raise ValueError("Exp416 summary experiment_id mismatch")
    if summary["scope"] != _SCOPE or type(summary["scope"]) is not str:
        raise ValueError("Exp416 summary scope mismatch")
    git_commit = summary["git_commit"]
    if type(git_commit) is not str or _HEX_40.fullmatch(git_commit) is None:
        raise ValueError("Exp416 summary git_commit type is invalid")
    _require_digest(
        summary["prediction_identity_sha256"],
        "prediction identity SHA256",
    )
    inputs = _require_mapping(summary["input_sha256"], "input_sha256")
    _require_exact_keys(
        inputs,
        {
            "database_reference",
            "prediction",
            "manifest",
            "frozen_test_labels",
        },
        "input SHA256 schema",
    )
    expected_hashes = {
        "database_reference": config.database_sha256,
        "prediction": config.prediction_sha256,
        "manifest": config.manifest_sha256,
        "frozen_test_labels": config.labels_sha256,
    }
    for name, expected in expected_hashes.items():
        _require_digest(inputs[name], f"{name} SHA256")
        if inputs[name] != expected:
            raise ValueError(f"Exp416 input SHA256 mismatch: {name}")
    if (
        _exact_int(
            summary["prediction_dataset_count"],
            "prediction_dataset_count",
        )
        != config.expected_prediction_datasets
    ):
        raise ValueError("Exp416 prediction dataset count mismatch")
    _require_exact_keys(
        _require_mapping(summary["micro"], "micro"),
        {"counts", "metrics"},
        "micro schema",
    )
    expected_flags = {
        "prediction_modified": False,
        "test_labels_loaded_after_prediction_freeze": True,
        "test_informed_reselection": False,
        "operational_policy_changed": False,
        "exp415_prediction_created": False,
    }
    for name, expected in expected_flags.items():
        if type(summary[name]) is not bool or summary[name] is not expected:
            raise ValueError(f"Exp416 governance flag mismatch: {name}")
    return summary


def _parse_dataset_csv(payload: bytes) -> tuple[DatasetAuditRow, ...]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Exp416 dataset CSV is not UTF-8") from error
    if not text.endswith("\n") or "\r" in text:
        raise ValueError("Exp416 dataset CSV line endings are invalid")
    try:
        raw_rows = tuple(csv.reader(StringIO(text, newline="")))
    except csv.Error as error:
        raise ValueError("Exp416 dataset CSV is invalid") from error
    if not raw_rows or tuple(raw_rows[0]) != _DATASET_FIELDS:
        raise ValueError("Exp416 dataset CSV header schema mismatch")
    parsed: list[DatasetAuditRow] = []
    canonical_rows: list[dict[str, object]] = []
    for row_number, cells in enumerate(raw_rows[1:], start=2):
        if len(cells) != len(_DATASET_FIELDS):
            raise ValueError(
                f"Exp416 dataset CSV row schema mismatch: {row_number}"
            )
        values = dict(zip(_DATASET_FIELDS, cells, strict=True))
        dataset = values["dataset"]
        if not dataset or dataset != dataset.strip():
            raise ValueError("Exp416 dataset CSV dataset is invalid")
        counts = ConfusionCounts(
            **{
                name: _csv_int(values[name], f"CSV {name}")
                for name in _COUNT_NAMES
            }
        )
        observed_metrics = ConfusionMetrics(
            **{
                name: _csv_metric(values[name], f"CSV {name}")
                for name in _METRIC_NAMES
            }
        )
        expected_metrics = metrics_from_counts(counts)
        if observed_metrics != expected_metrics:
            raise ValueError(
                f"Exp416 CSV metric mismatch for dataset: {dataset}"
            )
        parsed.append(DatasetAuditRow(dataset, counts, observed_metrics))
        canonical_rows.append(
            {
                "dataset": dataset,
                **asdict(counts),
                **asdict(observed_metrics),
            }
        )
    datasets = tuple(row.dataset for row in parsed)
    if datasets != tuple(sorted(datasets)) or len(set(datasets)) != len(datasets):
        raise ValueError("Exp416 CSV dataset order must be sorted and unique")
    if _csv_bytes(tuple(canonical_rows)) != payload:
        raise ValueError("Exp416 dataset CSV is not canonical")
    return tuple(parsed)


def _parse_counts(value: object, field_name: str) -> ConfusionCounts:
    mapping = _require_mapping(value, field_name)
    _require_exact_keys(mapping, set(_COUNT_NAMES), f"{field_name} schema")
    return ConfusionCounts(
        **{
            name: _exact_int(mapping[name], f"{field_name} {name}")
            for name in _COUNT_NAMES
        }
    )


def _parse_metrics(value: object, field_name: str) -> ConfusionMetrics:
    mapping = _require_mapping(value, field_name)
    _require_exact_keys(mapping, set(_METRIC_NAMES), f"{field_name} schema")
    return ConfusionMetrics(
        **{
            name: _json_metric(mapping[name], f"{field_name} {name}")
            for name in _METRIC_NAMES
        }
    )


def _parse_macro_metric(
    value: object,
    name: str,
    dataset_count: int,
) -> MacroMetric:
    mapping = _require_mapping(value, f"macro {name}")
    _require_exact_keys(
        mapping,
        {"value", "defined_dataset_count"},
        f"macro {name} schema",
    )
    defined = _exact_int(
        mapping["defined_dataset_count"],
        f"macro {name} defined_dataset_count",
    )
    if defined > dataset_count:
        raise ValueError(
            f"Exp416 macro {name} defined count exceeds dataset count"
        )
    return MacroMetric(
        value=_json_metric(mapping["value"], f"macro {name} value"),
        defined_dataset_count=defined,
    )


def _exact_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Exp416 {field_name} must be a nonnegative exact integer")
    return value


def _csv_int(value: str, field_name: str) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as error:
        raise ValueError(f"Exp416 {field_name} is not an integer") from error
    if parsed < 0 or str(parsed) != value:
        raise ValueError(f"Exp416 {field_name} is not a canonical integer")
    return parsed


def _json_metric(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"Exp416 {field_name} metric type must be finite float")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Exp416 {field_name} metric must be in [0, 1]")
    return value


def _csv_metric(value: str, field_name: str) -> float | None:
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"Exp416 {field_name} is not numeric") from error
    if (
        not math.isfinite(parsed)
        or not 0.0 <= parsed <= 1.0
        or str(parsed) != value
    ):
        raise ValueError(f"Exp416 {field_name} is not a canonical finite metric")
    return parsed


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Exp416 {field_name} must be an object")
    if any(type(key) is not str for key in value):
        raise ValueError(f"Exp416 {field_name} keys must be strings")
    return value


def _require_exact_keys(
    value: Mapping[str, object],
    expected: set[str],
    field_name: str,
) -> None:
    if set(value) != expected or len(value) != len(expected):
        raise ValueError(f"Exp416 {field_name} mismatch")


def _require_digest(value: object, field_name: str) -> None:
    if type(value) is not str or _HEX_64.fullmatch(value) is None:
        raise ValueError(f"Exp416 {field_name} must be 64 lowercase hex characters")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


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


def _csv_bytes(rows: tuple[Mapping[str, object], ...]) -> bytes:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=_DATASET_FIELDS,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                name: "" if row[name] is None else row[name]
                for name in _DATASET_FIELDS
            }
        )
    return buffer.getvalue().encode("utf-8")


def _format_metric(value: object) -> str:
    return "undefined" if value is None else str(value)


def _korean_report(summary: Mapping[str, object]) -> str:
    micro = _require_mapping(summary["micro"], "micro")
    counts = _require_mapping(micro["counts"], "micro counts")
    metrics = _require_mapping(micro["metrics"], "micro metrics")
    macro = _require_mapping(summary["macro"], "macro")
    lines = [
        "# Exp416 Exp404 Hard TEST 혼동행렬 감사",
        "",
        "## 정수 2×2 혼동행렬",
        "",
        "| 실제값 | 예측 양성 | 예측 음성 |",
        "|---|---:|---:|",
        f"| 실제 양성 | {counts['tp']} | {counts['fn']} |",
        f"| 실제 음성 | {counts['fp']} | {counts['tn']} |",
        "",
        "## Micro metric",
        "",
    ]
    lines.extend(
        f"- {name}: `{_format_metric(metrics[name])}`"
        for name in _METRIC_NAMES
    )
    lines.extend(
        [
            "",
            "## dataset-balanced macro 보조값",
            "",
        ]
    )
    for name in _METRIC_NAMES:
        item = _require_mapping(macro[name], f"macro {name}")
        lines.append(
            f"- {name}: `{_format_metric(item['value'])}` "
            f"(defined_dataset_count={item['defined_dataset_count']})"
        )
    lines.extend(
        [
            "",
            "## 해석 및 TEST 사용 경계",
            "",
            "- 이 결과는 동결된 Exp404 Hard prediction의 사후 TEST 감사다.",
            "- TEST 결과로 재선택하지 않는다: alpha, threshold, source, "
            "budget, lane 또는 family는 모두 동결 경계를 유지한다.",
            "- prediction을 수정하지 않고 운영 정책을 변경하지 않는다.",
            "- Exp415 prediction 또는 자동 승격을 만들지 않는다.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "DATASET_FILE",
    "MANIFEST_FILE",
    "REPORT_FILE",
    "SUMMARY_FILE",
    "render_payloads",
    "verify_rendered_payloads",
]
