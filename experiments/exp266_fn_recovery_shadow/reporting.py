from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from .config import (
    DB_PATH,
    EXPECTED_DATASETS,
    EXP253_PREDICTION,
    EXP264_PREDICTION,
    FROZEN_DATABASE_SHA256,
)
from .independent_reaggregate import COMBINED_METRIC_FORMULA, write_independent_reaggregation
from .invariance import run_invariance_suite, write_invariance_report
from .prediction import CheckpointError, verify_prediction_checkpoint
from .train_risk import RECONSTRUCTION_PROVENANCE_BLOCKER, validate_train_risk_audit


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one_row(path: Path) -> dict[str, str]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"{Path(path).name} must contain exactly one row")
    return rows[0]


def _prediction_boundary(
    output_root: Path, *, require_full_population: bool
) -> tuple[str, list[dict[str, object]]]:
    if require_full_population:
        try:
            manifest = verify_prediction_checkpoint(output_root)
        except CheckpointError as exc:
            raise ValueError(f"strict checkpoint verification failed: {exc}") from exc
        if (
            manifest.get("complete_population") is not True
            or manifest.get("smoke") is not False
            or manifest.get("rows") != EXPECTED_DATASETS
            or manifest.get("datasets") != EXPECTED_DATASETS
        ):
            raise ValueError("reporting requires a complete 1,117-dataset checkpoint")
    prediction_path = output_root / "03_frozen_predictions.jsonl"
    manifest_path = output_root / "04_prediction_manifest.json"
    try:
        payload = prediction_path.read_bytes()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot verify report prediction boundary") from exc
    actual = hashlib.sha256(payload).hexdigest()
    if type(manifest) is not dict or manifest.get("prediction_sha256") != actual:
        raise ValueError("reporting prediction SHA256 mismatch")
    if any(type(row) is not dict for row in rows):
        raise ValueError("reporting prediction row is invalid")
    if require_full_population and (
        len(rows) != EXPECTED_DATASETS
        or len({str(row.get("dataset", "")) for row in rows}) != EXPECTED_DATASETS
    ):
        raise ValueError("reporting requires exactly 1,117 unique prediction rows")
    return actual, rows


def _invariance_boundary_checks(
    output_root: Path, prediction_sha256: str, rows: list[dict[str, object]]
) -> list[tuple[str, bool]]:
    summary = _one_row(output_root / "06_global_summary.csv")
    with (output_root / "05_dataset_results.csv").open(newline="", encoding="utf-8") as handle:
        result_rows = list(csv.DictReader(handle))
    no_label_fields = all(
        row.get("contains_test_labels") is False and "y_test" not in row for row in rows
    )
    no_metric_fields = all(
        row.get("contains_evaluation_metrics") is False
        and not {"tp", "fp", "fn", "precision", "recall", "f1"}.intersection(row)
        for row in rows
    )
    checks = [
        ("Frozen prediction SHA matches manifest", True),
        (
            "Evaluation rows bind to the verified prediction SHA",
            bool(result_rows)
            and all(row.get("prediction_sha256") == prediction_sha256 for row in result_rows),
        ),
        ("Global summary binds to the verified prediction SHA", summary.get("prediction_sha256") == prediction_sha256),
        ("Frozen prediction contains no TEST labels", no_label_fields),
        ("Frozen prediction contains no evaluation metrics", no_metric_fields),
        ("Autonomous Hard alert delta remains zero", summary.get("hard_alert_delta") == "0"),
        (
            "Exact pinned immutable-lane parity covers every dataset",
            all(
                summary.get(f"{lane}_parity_datasets") == summary.get("datasets")
                and summary.get(f"{lane}_parity_mismatches") == "0"
                and summary.get(f"{lane}_actual_indices_sha256")
                == summary.get(f"{lane}_expected_indices_sha256")
                for lane in ("hard", "shadow", "priority", "standard")
            ),
        ),
        ("Repeat read preserves the frozen prediction hash", _sha256(output_root / "03_frozen_predictions.jsonl") == prediction_sha256),
    ]
    if not all(passed for _, passed in checks):
        failed = [name for name, passed in checks if not passed]
        raise ValueError(f"invariance report check failed: {failed}")
    return checks


def _risk_summary(
    output_root: Path, *, required: bool
) -> dict[str, object] | None:
    path = output_root / "02_train_normal_risk_audit.csv"
    if not path.exists():
        if required:
            raise ValueError("reporting requires the strict TRAIN-risk audit")
        return None
    return validate_train_risk_audit(output_root)


def _write_smoke_invariance_report(
    output_root: Path,
    prediction_sha256: str,
    boundary_checks: list[tuple[str, bool]],
) -> Path:
    lines = [
        "# Exp266 Smoke Invariance Boundary Report",
        "",
        f"Prediction SHA-256: `{prediction_sha256}`",
        "",
        "| Check | Status |",
        "|---|---|",
        *[
            f"| {name} | {'PASS' if passed else 'FAIL'} |"
            for name, passed in boundary_checks
        ],
        "",
        "The six full-population mutation checks are generated only by the full report stage.",
    ]
    path = output_root / "08_invariance_report.md"
    _atomic_text(path, "\n".join(lines) + "\n")
    return path


def _write_result_report(output_root: Path, risk: dict[str, object] | None) -> Path:
    row = _one_row(output_root / "06_global_summary.csv")
    lines = [
        "# Exp266 Post-Freeze Retrospective Result",
        "",
        "This report evaluates the frozen label-free checkpoint after SHA-256 verification.",
        "It is retrospective, is not deployment evidence, and does not change any prediction.",
        "No threshold, policy, or source was selected from TEST results.",
        "",
        "## Autonomous Hard",
        "",
        "Human-assisted review candidates are excluded from every metric in this table.",
        "",
        "| Alerts | TP | FP | FN | Precision | Recall | Mean dataset F1 | Zero-F1 datasets | Delta |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {row['hard_alerts']} | {row['hard_tp']} | {row['hard_fp']} | {row['hard_fn']} | {float(row['hard_precision']):.6f} | {float(row['hard_recall']):.6f} | {float(row['hard_mean_dataset_f1']):.6f} | {row['hard_zero_f1_datasets']} | {row['hard_alert_delta']} |",
        "",
        "## Human-Assisted Review Only",
        "",
        "| Lane | Requests | TP | FP | Precision | Recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for lane, label in (
        ("shadow", "Shadow"),
        ("priority", "Priority"),
        ("standard", "Standard"),
        ("recovery_shadow", "RecoveryShadow"),
        ("recovery_priority", "RecoveryPriority"),
        ("data_quality_review", "DataQualityReview"),
    ):
        lines.append(
            f"| {label} | {row[f'{lane}_count']} | {row[f'{lane}_tp']} | "
            f"{row[f'{lane}_fp']} | {float(row[f'{lane}_precision']):.6f} | "
            f"{float(row[f'{lane}_recall']):.6f} | {float(row[f'{lane}_f1']):.6f} |"
        )
    lines.extend(
        [
            "",
            f"Unique recovered false negatives: **{row['recovered_unique_fn_count']}**",
            f"Datasets with added review candidates: **{row['datasets_with_added_candidates']}**",
            "Total existing plus Exp266 review-request rate per TEST run: "
            f"**{float(row['total_existing_plus_exp266_review_request_rate']):.6f}**",
            "Exp266 incremental review-request rate per TEST run: "
            f"**{float(row['exp266_incremental_review_request_rate']):.6f}**",
            "",
            "## Human-Assisted Combined Diagnostic",
            "",
            COMBINED_METRIC_FORMULA + ".",
            "This combined view is human-assisted and never replaces Autonomous Hard.",
            "",
            "| Combined accepted | TP | FP | FN | Precision | Recall | Mean combined F1 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            f"| {row['combined_count']} | {row['combined_tp']} | {row['combined_fp']} | "
            f"{row['combined_fn']} | {float(row['combined_precision']):.6f} | "
            f"{float(row['combined_recall']):.6f} | {float(row['mean_combined_f1']):.6f} |",
            "",
            "",
            "## Immutable Lane Parity",
            "",
            "| Lane | Pinned source | Matching datasets | Mismatches | Index-map SHA-256 |",
            "|---|---|---:|---:|---|",
        ]
    )
    for lane, source in (
        ("hard", "Exp253"),
        ("shadow", "Exp264"),
        ("priority", "Exp264"),
        ("standard", "Exp264"),
    ):
        lines.append(
            f"| {lane.title()} | {source} | {row[f'{lane}_parity_datasets']} | "
            f"{row[f'{lane}_parity_mismatches']} | "
            f"`{row[f'{lane}_actual_indices_sha256']}` |"
        )
    if risk is not None:
        lines.extend(
            [
                "",
                "## Certified TRAIN-Normal Risk Audit",
                "",
            "The certified audit uses exactly four provenance-verified sources. The",
            "five-source reconstruction result remains an unapproved sensitivity diagnostic.",
            "Exact five-source blocker: **"
            + str(risk["five_source_sensitivity_blocker"]).rstrip(".")
            + "**.",
            "The TRAIN-risk table below applies only to the Exp266 incremental "
            "RecoveryShadow and RecoveryPriority lanes, never to existing review lanes.",
                "",
                "| Lane | False reviews | Rate | One-sided Wilson U95 | Target <= 0.01 |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for lane, label in (
            ("recovery_shadow", "RecoveryShadow"),
            ("recovery_priority", "RecoveryPriority"),
            ("combined", "Combined review"),
        ):
            lines.append(
                f"| {label} | {risk[f'{lane}_false_reviews']} | "
                f"{float(risk[f'{lane}_false_review_rate']):.9f} | "
                f"{float(risk[f'{lane}_wilson_upper_95']):.9f} | "
                f"{'PASS' if risk[f'{lane}_meets_risk_target'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "RecoveryShadow, RecoveryPriority, DataQualityReview, and existing review lanes",
            "remain human-assisted and review-only. No review result is added to Autonomous Hard.",
            "All point matches are exact TEST instance-index matches with zero-denominator ratios",
            "recorded as 0.0.",
            "No threshold, policy, or source was selected from TEST results.",
            (
                "Five-source blocker: "
                + str(risk["five_source_sensitivity_blocker"]).rstrip(".")
                + "."
                if risk is not None
                else "Five-source risk was not evaluated in this smoke report."
            ),
        ]
    )
    path = output_root / "09_result_report.md"
    _atomic_text(path, "\n".join(lines) + "\n")
    return path


def _write_manifest(output_root: Path) -> Path:
    names = (
        "00_contract_audit.json",
        "01_source_manifest.csv",
        "02_train_normal_risk_audit.csv",
        "02_train_normal_risk_audit_manifest.json",
        "03_frozen_predictions.jsonl",
        "04_prediction_manifest.json",
        "05_dataset_results.csv",
        "06_global_summary.csv",
        "07_source_agreement_diagnostics.csv",
        "08_invariance_report.md",
        "09_result_report.md",
        "10_independent_reaggregation.json",
    )
    rows = []
    for name in names:
        path = output_root / name
        if path.exists():
            role = "frozen_input" if name.startswith(("00_", "01_", "02_", "03_", "04_")) else "retrospective_output"
            rows.append(
                {
                    "file": name,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                    "role": role,
                    "retrospective": True,
                    "deployment_eligible": False,
                    "database_sha256": FROZEN_DATABASE_SHA256,
                    "autonomous_scope": "Hard only; exact pinned Exp253 parity required",
                    "immutable_review_lane_scope": (
                        "Shadow, Priority, and Standard require exact pinned Exp264 parity"
                    ),
                    "human_assisted_scope": (
                        "Shadow, Priority, Standard, RecoveryShadow, RecoveryPriority, and "
                        "DataQualityReview; review only"
                    ),
                    "combined_metric_formula": COMBINED_METRIC_FORMULA,
                    "no_test_policy_selection": True,
                    "five_source_blocker": RECONSTRUCTION_PROVENANCE_BLOCKER,
                }
            )
    from io import StringIO

    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path = output_root / "MANIFEST.csv"
    _atomic_text(path, handle.getvalue())
    return path


def write_reports(
    output_root: Path,
    *,
    exp253_prediction_path: Path = EXP253_PREDICTION,
    exp264_prediction_path: Path = EXP264_PREDICTION,
    database_path: Path = DB_PATH,
    expected_database_sha256: str = FROZEN_DATABASE_SHA256,
    require_full_population: bool = True,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[Path, Path, Path, Path]:
    output_root = Path(output_root)
    prediction_sha256, rows = _prediction_boundary(
        output_root, require_full_population=require_full_population
    )
    risk = _risk_summary(output_root, required=require_full_population)
    reaggregation = write_independent_reaggregation(
        output_root,
        exp253_prediction_path=exp253_prediction_path,
        exp264_prediction_path=exp264_prediction_path,
        database_path=database_path,
        expected_database_sha256=expected_database_sha256,
        require_full_population=require_full_population,
        progress_callback=progress_callback,
    )
    if not reaggregation["passed"]:
        raise ValueError("independent reaggregation failed")
    boundary_checks = _invariance_boundary_checks(output_root, prediction_sha256, rows)
    if require_full_population:
        evidence = run_invariance_suite(
            baseline_root=output_root,
            source_database=database_path,
        )
        invariance_path = write_invariance_report(
            output_root, prediction_sha256, boundary_checks, evidence
        )
    else:
        invariance_path = _write_smoke_invariance_report(
            output_root, prediction_sha256, boundary_checks
        )
    result_path = _write_result_report(output_root, risk)
    reaggregation_path = output_root / "10_independent_reaggregation.json"
    manifest_path = _write_manifest(output_root)
    if _sha256(output_root / "03_frozen_predictions.jsonl") != prediction_sha256:
        raise ValueError("reporting mutated the frozen prediction artifact")
    return invariance_path, result_path, reaggregation_path, manifest_path
