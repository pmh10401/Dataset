"""Deterministic JSON/CSV/Korean reporting for Exp422-TEST-EVAL."""

from collections.abc import Mapping
import csv
from dataclasses import asdict
from hashlib import sha256
from io import StringIO
import json

from .config import Exp422TestEvalConfig, FROZEN_CONFIG
from .contracts import AuditResult, ConfusionMetrics, FrozenLabels, FrozenPrediction


SUMMARY_FILE = "exp422_test_eval_confusion_matrix.json"
DATASET_FILE = "exp422_test_eval_dataset_metrics.csv"
REPORT_FILE = "EXP422_TEST_EVAL_RESULT_REPORT_KO.md"
STATUS_FILE = "status.json"
MANIFEST_FILE = "MANIFEST.json"

_METRICS = (
    "precision", "recall", "specificity", "false_positive_rate",
    "false_negative_rate", "f1",
)
_CSV_FIELDS = (
    "dataset", "tn", "fp", "fn", "tp", "precision", "recall", "specificity",
    "false_positive_rate", "false_negative_rate", "f1",
)


def _metric_payload(metrics: ConfusionMetrics) -> dict[str, float | None]:
    return {name: getattr(metrics, name) for name in _METRICS}


def _counts_payload(counts: object) -> dict[str, int]:
    return {name: getattr(counts, name) for name in ("tn", "fp", "fn", "tp")}


def render_payloads(
    result: AuditResult,
    frozen_prediction: FrozenPrediction,
    frozen_labels: FrozenLabels,
    database_sha256: str,
    git_commit: str,
    *,
    config: Exp422TestEvalConfig = FROZEN_CONFIG,
) -> Mapping[str, bytes]:
    config.validate()
    if type(result) is not AuditResult or type(frozen_prediction) is not FrozenPrediction or type(frozen_labels) is not FrozenLabels:
        raise ValueError("Exp422-TEST-EVAL reporting inputs are invalid")
    summary = {
        "experiment_id": config.experiment_id,
        "source_experiment_id": config.source_experiment_id,
        "scope": "frozen predicted-positive coordinates versus TEST labels",
        "git_commit": git_commit,
        "input_sha256": {
            "database": database_sha256,
            "prediction": frozen_prediction.prediction_sha256,
            "prediction_manifest": frozen_prediction.manifest_sha256,
            "test_labels": frozen_labels.labels_sha256,
        },
        "prediction_identity_sha256": frozen_prediction.identity_sha256,
        "dataset_count": len(frozen_labels.dataset_scope),
        "prediction_dataset_count": frozen_prediction.prediction_dataset_count,
        "prediction_coordinate_count": result.prediction_coordinates,
        "frozen_label_row_count": result.label_rows,
        "micro": {
            "counts": _counts_payload(result.micro_counts),
            "matrix": result.micro_counts.as_nested(),
            "metrics": _metric_payload(result.micro_metrics),
        },
        "macro": {
            name: {
                "value": metric.value,
                "defined_dataset_count": metric.defined_dataset_count,
            }
            for name, metric in sorted(result.macro_metrics.items())
        },
        "test_labels_loaded_after_prediction_freeze": True,
        "test_informed_reselection": False,
        "operational_policy_changed": False,
    }
    summary_payload = json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in result.dataset_rows:
        values: dict[str, object] = {name: getattr(row.counts, name) for name in ("tn", "fp", "fn", "tp")}
        values.update({name: getattr(row.metrics, name) for name in _METRICS})
        writer.writerow({"dataset": row.dataset, **values})
    csv_payload = stream.getvalue().encode("utf-8")

    counts = result.micro_counts
    metrics = result.micro_metrics
    report = "\n".join(
        [
            "# Exp422-TEST-EVAL 결과",
            "",
            "이 문서는 동결된 예측을 TEST 정답과 대조한 사후 연구 평가다. 이 결과로 운영 정책을 다시 선택하지 않았다.",
            "",
            f"- 평가 dataset: {len(frozen_labels.dataset_scope):,}개",
            f"- TEST instance: {result.label_rows:,}개",
            f"- 예측 이상 좌표: {result.prediction_coordinates:,}개",
            "",
            "## 전체 혼동행렬",
            "",
            "행=실제, 열=예측이며 순서는 `[[TN, FP], [FN, TP]]`이다.",
            "",
            f"`{counts.as_nested()}`",
            "",
            f"- TN {counts.tn:,} / FP {counts.fp:,} / FN {counts.fn:,} / TP {counts.tp:,}",
            f"- precision {metrics.precision!r}, recall {metrics.recall!r}, specificity {metrics.specificity!r}, F1 {metrics.f1!r}",
            "",
            "TEST 라벨은 예측 해시를 먼저 동결한 뒤 읽었으며, alpha·threshold·source·budget·lane은 변경하지 않았다.",
            "",
        ]
    ).encode("utf-8")
    return {
        SUMMARY_FILE: summary_payload,
        DATASET_FILE: csv_payload,
        REPORT_FILE: report,
    }


def verify_rendered_payloads(payloads: Mapping[str, bytes]) -> None:
    if set(payloads) != {SUMMARY_FILE, DATASET_FILE, REPORT_FILE}:
        raise ValueError("Exp422-TEST-EVAL payload set is incomplete")
    try:
        summary = json.loads(payloads[SUMMARY_FILE])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("confusion summary is invalid JSON") from error
    if summary.get("experiment_id") != FROZEN_CONFIG.experiment_id:
        raise ValueError("confusion summary experiment id mismatch")
    if summary.get("test_labels_loaded_after_prediction_freeze") is not True:
        raise ValueError("confusion summary label ordering flag is missing")
    if summary.get("test_informed_reselection") is not False or summary.get("operational_policy_changed") is not False:
        raise ValueError("confusion summary governance flags are not frozen")
    if not payloads[DATASET_FILE].startswith(b"dataset,tn,fp,fn,tp,"):
        raise ValueError("dataset CSV header mismatch")
    if not payloads[REPORT_FILE].startswith(b"# Exp422-TEST-EVAL"):
        raise ValueError("Korean report header mismatch")


def artifact_hashes(payloads: Mapping[str, bytes]) -> dict[str, str]:
    verify_rendered_payloads(payloads)
    return {name: sha256(payload).hexdigest() for name, payload in sorted(payloads.items())}


def render_manifest(*, artifact_sha256: Mapping[str, str], provenance: Mapping[str, object]) -> bytes:
    body = {
        "experiment_id": FROZEN_CONFIG.experiment_id,
        "artifact_sha256": dict(sorted(artifact_sha256.items())),
        "provenance": dict(provenance),
    }
    return json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


__all__ = [
    "DATASET_FILE", "MANIFEST_FILE", "REPORT_FILE", "STATUS_FILE", "SUMMARY_FILE",
    "artifact_hashes", "render_manifest", "render_payloads", "verify_rendered_payloads",
]
