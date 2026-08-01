"""Canonical, TRAIN-only Exp422 result and reviewed-source manifests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from types import MappingProxyType

from .capability import DeviceCapability
from .config import FROZEN_CONFIG, frozen_config_sha256
from .contracts import (
    BatchKey,
    CoreParityReport,
    StageResult,
    TensorDelta,
    TimingEvidence,
)
from .core_parity import CoreStageResult, _LENGTHS
from experiments.exp417_causal_multihorizon_shadow.preprocessing import build_horizons
from .score_parity import ScoreBatchReport, ScoreStageResult, validate_dataset_name


_ARTIFACT_KIND = "exp422_mps_numeric_delta_manifest"
_SOURCE_ARTIFACT_KIND = "exp422_reviewed_source_manifest"
_SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_RE = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_FORBIDDEN_VALUE_TOKEN = re.compile(
    r"(?:^|[^A-Za-z0-9])(?:test|eval|evaluation|label|labels|ground[_ -]?truth|location|locations)(?:$|[^A-Za-z0-9])",
    re.IGNORECASE,
)
_PROVENANCE_FIELDS = frozenset(
    {
        "attempt_root",
        "config_sha256",
        "database_sha256",
        "device_name",
        "fresh_no_clobber_attempt",
        "git_commit",
        "mlx_availability",
        "mlx_eligibility_reason",
        "mlx_metal_version",
        "mlx_version",
        "mps_available",
        "mps_built",
        "repeats",
        "source_manifest_sha256",
        "source_git_commit",
        "sqlite_quick_check",
        "torch_version",
        "train_only",
    }
)
_TOP_LEVEL_FIELDS = frozenset(
    {
        "artifact_kind",
        "schema_version",
        "experiment_id",
        "state",
        "candidate_batch_sizes",
        "core_stage",
        "score_stage",
        "fallback_reasons",
        "provenance",
    }
)

SOURCE_MANIFEST_RELATIVE = "experiments/exp422_mps_numeric_delta/SOURCE_MANIFEST.json"
_SOURCE_PATHS = (
    "docs/EXP422_PREREGISTRATION_KO.md",
    "docs/EXP422_RUNBOOK_KO.md",
    "docs/EXP422_IR_PREREGISTRATION_KO.md",
    "docs/EXP422_IR_RUNBOOK_KO.md",
    "docs/superpowers/plans/2026-08-02-exp422-mps-numeric-delta.md",
    "docs/superpowers/specs/2026-08-02-exp422-mps-numeric-delta-design.md",
    "docs/superpowers/specs/2026-08-02-exp422-interrupted419-design.md",
    "experiments/exp422_mps_numeric_delta/__init__.py",
    "experiments/exp422_mps_numeric_delta/capability.py",
    "experiments/exp422_mps_numeric_delta/config.py",
    "experiments/exp422_mps_numeric_delta/contracts.py",
    "experiments/exp422_mps_numeric_delta/core_parity.py",
    "experiments/exp422_mps_numeric_delta/manifest.py",
    "experiments/exp422_mps_numeric_delta/score_parity.py",
    "experiments/exp422_mps_numeric_delta/tests/__init__.py",
    "experiments/exp422_mps_numeric_delta/tests/test_capability.py",
    "experiments/exp422_mps_numeric_delta/tests/test_cli.py",
    "experiments/exp422_mps_numeric_delta/tests/test_config.py",
    "experiments/exp422_mps_numeric_delta/tests/test_contracts.py",
    "experiments/exp422_mps_numeric_delta/tests/test_core_parity.py",
    "experiments/exp422_mps_numeric_delta/tests/test_manifest.py",
    "experiments/exp422_mps_numeric_delta/tests/test_score_boundary.py",
    "experiments/exp422_mps_numeric_delta/tests/test_score_parity.py",
    "experiments/exp422_mps_numeric_delta/tests/test_interrupted_cli.py",
    "scratch/run_exp422_mps_numeric_delta.py",
    "scratch/run_exp422_interrupted419_diagnostic.py",
)

_CORE_LENGTHS = _LENGTHS


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest payload must be canonical JSON data") from exc


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(child) for child in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(child) for child in value]
    return value


def _sha256_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical lowercase SHA-256 digest")
    return value


def _stage_payload(stage: StageResult) -> dict[str, object]:
    return {"state": stage.state, "reason": stage.reason}


def _parse_stage(value: object, name: str) -> StageResult:
    if not isinstance(value, Mapping) or set(value) != {"state", "reason"}:
        raise ValueError(f"{name} stage schema is invalid")
    try:
        return StageResult(value["state"], value["reason"])  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} stage is invalid") from exc


def _timing_payload(timing: TimingEvidence) -> dict[str, object]:
    return {
        "backend": timing.backend,
        "transfer_seconds": timing.transfer_seconds,
        "compute_seconds": timing.compute_seconds,
        "synchronize_seconds": timing.synchronize_seconds,
        "total_seconds": timing.total_seconds,
    }


def _parse_timing(value: object, backend: str) -> TimingEvidence:
    fields = {
        "backend",
        "transfer_seconds",
        "compute_seconds",
        "synchronize_seconds",
        "total_seconds",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("timing schema is invalid")
    try:
        timing = TimingEvidence(**dict(value))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("timing evidence is invalid") from exc
    if timing.backend != backend:
        raise ValueError("timing backend is inconsistent")
    return timing


def _delta_payload(delta: TensorDelta) -> dict[str, object]:
    return {
        "name": delta.name,
        "finite": delta.finite,
        "max_absolute_error": delta.max_absolute_error,
        "mean_absolute_error": delta.mean_absolute_error,
        "max_normalized_error": delta.max_normalized_error,
        "max_ulp": delta.max_ulp,
        "cpu_sha256": delta.cpu_sha256,
        "mps_sha256": delta.mps_sha256,
        "failing_index": delta.failing_index,
    }


def _parse_delta(value: object) -> TensorDelta:
    fields = {
        "name",
        "finite",
        "max_absolute_error",
        "mean_absolute_error",
        "max_normalized_error",
        "max_ulp",
        "cpu_sha256",
        "mps_sha256",
        "failing_index",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("tensor delta schema is invalid")
    try:
        return TensorDelta(**dict(value))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("tensor delta is invalid") from exc


def _capability_payload(capability: DeviceCapability) -> dict[str, object]:
    return {
        "torch_version": capability.torch_version,
        "mps_built": capability.mps_built,
        "mps_available": capability.mps_available,
        "device_name": capability.device_name,
        "mlx_version": capability.mlx_version,
        "mlx_metal_version": capability.mlx_metal_version,
        "mlx_availability": capability.mlx_availability,
        "fallback_reason": capability.fallback_reason,
    }


def _parse_capability(value: object) -> DeviceCapability:
    fields = {
        "torch_version",
        "mps_built",
        "mps_available",
        "device_name",
        "mlx_version",
        "mlx_metal_version",
        "mlx_availability",
        "fallback_reason",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("capability schema is invalid")
    try:
        return DeviceCapability(**dict(value))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError("capability evidence is invalid") from exc


def _core_report_payload(report: CoreParityReport) -> dict[str, object]:
    return {
        "batch_key": {
            "batch_size": report.batch_key.batch_size,
            "model_seed": report.batch_key.model_seed,
            "perturbation_seed": report.batch_key.perturbation_seed,
        },
        "tensor_deltas": [_delta_payload(delta) for delta in report.tensor_deltas],
        "cpu_timing": _timing_payload(report.cpu_timing),
        "mps_timing": _timing_payload(report.mps_timing),
        "candidate_indices_match": report.candidate_indices_match,
        "spearman": report.spearman,
    }


def _parse_core_report(value: object) -> CoreParityReport:
    fields = {
        "batch_key",
        "tensor_deltas",
        "cpu_timing",
        "mps_timing",
        "candidate_indices_match",
        "spearman",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("core report schema is invalid")
    key = value["batch_key"]
    if not isinstance(key, Mapping) or set(key) != {"batch_size", "model_seed", "perturbation_seed"}:
        raise ValueError("core report batch key is invalid")
    deltas = value["tensor_deltas"]
    if isinstance(deltas, (str, bytes)) or not isinstance(deltas, Sequence):
        raise ValueError("core tensor deltas are invalid")
    try:
        return CoreParityReport(
            BatchKey(**dict(key)),  # type: ignore[arg-type]
            tuple(_parse_delta(delta) for delta in deltas),
            _parse_timing(value["cpu_timing"], "cpu"),
            _parse_timing(value["mps_timing"], "mps"),
            value["candidate_indices_match"],  # type: ignore[arg-type]
            value["spearman"],  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("core report is invalid") from exc


def _core_payload(core: CoreStageResult) -> dict[str, object]:
    return {
        "capability": _capability_payload(core.capability),
        "reports": [_core_report_payload(report) for report in core.reports],
        "stage": _stage_payload(core.stage),
    }


def _parse_core(value: object) -> CoreStageResult:
    if not isinstance(value, Mapping) or set(value) != {"capability", "reports", "stage"}:
        raise ValueError("core stage schema is invalid")
    raw_reports = value["reports"]
    if isinstance(raw_reports, (str, bytes)) or not isinstance(raw_reports, Sequence) or not raw_reports:
        raise ValueError("core reports must be a nonempty sequence")
    reports = tuple(_parse_core_report(report) for report in raw_reports)
    expected_order = tuple(
        (batch_size, model_seed, perturbation_seed, length)
        for batch_size in FROZEN_CONFIG.candidate_batch_sizes
        for model_seed, perturbation_seed in zip(
            FROZEN_CONFIG.model_seeds, FROZEN_CONFIG.perturbation_seeds, strict=True
        )
        for length in _CORE_LENGTHS
    )
    expected = set(expected_order)
    observed: list[tuple[int, int, int, int]] = []
    for report in reports:
        lengths: set[int] = set()
        for delta in report.tensor_deltas:
            match = re.fullmatch(r"length_(\d+)_(?:horizon_\d+|prediction)", delta.name)
            if match is None:
                raise ValueError("core report tensor names do not bind a canonical length")
            lengths.add(int(match.group(1)))
        if len(lengths) != 1:
            raise ValueError("core report must contain exactly one canonical length")
        length = next(iter(lengths))
        names = tuple(delta.name for delta in report.tensor_deltas)
        if len(set(names)) != len(names):
            raise ValueError("core report tensor names must be unique")
        horizon_names = {name for name in names if "_horizon_" in name}
        placeholder_names = {name for name in names if name.endswith("_prediction")}
        expected_horizon_names = {
            f"length_{length}_horizon_{horizon}" for horizon in build_horizons(length)
        }
        if horizon_names and horizon_names != expected_horizon_names:
            raise ValueError("core report horizons do not bind the canonical Exp417 heads")
        if placeholder_names and (placeholder_names != {f"length_{length}_prediction"} or report.passes):
            raise ValueError("core report placeholder is only valid for a failed row")
        if not horizon_names and not placeholder_names:
            raise ValueError("core report must contain canonical horizon or failure evidence")
        observed.append(
            (
                report.batch_key.batch_size,
                report.batch_key.model_seed,
                report.batch_key.perturbation_seed,
                length,
            )
        )
    if (
        tuple(observed) != expected_order
        or len(observed) != len(expected)
        or len(set(observed)) != len(observed)
        or set(observed) != expected
    ):
        raise ValueError("core reports must contain the complete exact 60-row evidence matrix")
    stage = _parse_stage(value["stage"], "core")
    if stage.state == "passed" and not all(report.passes for report in reports):
        raise ValueError("passed core stage contains a failed report")
    if stage.state == "failed" and all(report.passes for report in reports):
        raise ValueError("failed core stage is inconsistent with all-passing reports")
    return CoreStageResult(_parse_capability(value["capability"]), reports, stage)


def _score_report_payload(report: ScoreBatchReport) -> dict[str, object]:
    return {
        "batch_size": report.batch_size,
        "chunk_sizes": list(report.chunk_sizes),
        "tensor_deltas": [_delta_payload(delta) for delta in report.tensor_deltas],
        "cpu_forward_sha256": report.cpu_forward_sha256,
        "mps_forward_sha256": report.mps_forward_sha256,
        "cpu_residual_sha256": report.cpu_residual_sha256,
        "mps_residual_sha256": report.mps_residual_sha256,
        "cpu_score_sha256": report.cpu_score_sha256,
        "mps_score_sha256": report.mps_score_sha256,
        "cpu_candidate_coordinates_sha256": report.cpu_candidate_coordinates_sha256,
        "mps_candidate_coordinates_sha256": report.mps_candidate_coordinates_sha256,
        "candidate_indices_match": report.candidate_indices_match,
        "spearman": report.spearman,
        "qualification_match": report.qualification_match,
        "cpu_timing": _timing_payload(report.cpu_timing),
        "mps_timing": _timing_payload(report.mps_timing),
        "max_normalized_error": report.max_normalized_error,
        "status": report.status,
        "fallback_reason": report.fallback_reason,
    }


def _parse_score_report(value: object) -> ScoreBatchReport:
    fields = {
        "batch_size",
        "chunk_sizes",
        "tensor_deltas",
        "cpu_forward_sha256",
        "mps_forward_sha256",
        "cpu_residual_sha256",
        "mps_residual_sha256",
        "cpu_score_sha256",
        "mps_score_sha256",
        "cpu_candidate_coordinates_sha256",
        "mps_candidate_coordinates_sha256",
        "candidate_indices_match",
        "spearman",
        "qualification_match",
        "cpu_timing",
        "mps_timing",
        "max_normalized_error",
        "status",
        "fallback_reason",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("score report schema is invalid")
    try:
        return ScoreBatchReport(
            batch_size=value["batch_size"],  # type: ignore[arg-type]
            chunk_sizes=tuple(value["chunk_sizes"]),  # type: ignore[arg-type]
            tensor_deltas=tuple(_parse_delta(item) for item in value["tensor_deltas"]),  # type: ignore[arg-type]
            cpu_forward_sha256=value["cpu_forward_sha256"],  # type: ignore[arg-type]
            mps_forward_sha256=value["mps_forward_sha256"],  # type: ignore[arg-type]
            cpu_residual_sha256=value["cpu_residual_sha256"],  # type: ignore[arg-type]
            mps_residual_sha256=value["mps_residual_sha256"],  # type: ignore[arg-type]
            cpu_score_sha256=value["cpu_score_sha256"],  # type: ignore[arg-type]
            mps_score_sha256=value["mps_score_sha256"],  # type: ignore[arg-type]
            cpu_candidate_coordinates_sha256=value["cpu_candidate_coordinates_sha256"],  # type: ignore[arg-type]
            mps_candidate_coordinates_sha256=value["mps_candidate_coordinates_sha256"],  # type: ignore[arg-type]
            candidate_indices_match=value["candidate_indices_match"],  # type: ignore[arg-type]
            spearman=value["spearman"],  # type: ignore[arg-type]
            qualification_match=value["qualification_match"],  # type: ignore[arg-type]
            cpu_timing=_parse_timing(value["cpu_timing"], "cpu"),
            mps_timing=_parse_timing(value["mps_timing"], "mps"),
            max_normalized_error=value["max_normalized_error"],  # type: ignore[arg-type]
            status=value["status"],  # type: ignore[arg-type]
            fallback_reason=value["fallback_reason"],  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("score report is invalid") from exc


def _score_payload(score: ScoreStageResult) -> dict[str, object]:
    sample = score.sample
    return {
        "sample": {
            "dataset": sample.dataset,
            "length": sample.length,
            "train_count": sample.train_count,
            "raw_waveform_sha256": sample.raw_waveform_sha256,
            "bounded_series_count": sample.bounded_series_count,
            "indices": list(sample.indices),
        },
        "reports": [_score_report_payload(report) for report in score.reports],
        "stage": _stage_payload(score.stage),
        "model_state_sha256": score.model_state_sha256,
        "model_stop_sha256": score.model_stop_sha256,
        "loss_history_sha256": score.loss_history_sha256,
        "oof_residual_sha256": score.oof_residual_sha256,
        "calibration_sha256": score.calibration_sha256,
        "perturbation_sha256": score.perturbation_sha256,
        "cpu_qualification_sha256": score.cpu_qualification_sha256,
        "mps_qualification_sha256": score.mps_qualification_sha256,
    }


def _validate_score(value: object) -> tuple[StageResult, tuple[ScoreBatchReport, ...]]:
    fields = {
        "sample",
        "reports",
        "stage",
        "model_state_sha256",
        "model_stop_sha256",
        "loss_history_sha256",
        "oof_residual_sha256",
        "calibration_sha256",
        "perturbation_sha256",
        "cpu_qualification_sha256",
        "mps_qualification_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError("score stage schema is invalid")
    sample = value["sample"]
    sample_fields = {
        "dataset",
        "length",
        "train_count",
        "raw_waveform_sha256",
        "bounded_series_count",
        "indices",
    }
    if not isinstance(sample, Mapping) or set(sample) != sample_fields:
        raise ValueError("score sample schema is invalid")
    validate_dataset_name(sample["dataset"])
    if type(sample["length"]) is not int or sample["length"] < 1:
        raise ValueError("score sample length is invalid")
    if type(sample["train_count"]) is not int or sample["train_count"] < 1:
        raise ValueError("score TRAIN count is invalid")
    _sha256_digest(sample["raw_waveform_sha256"], "raw_waveform_sha256")
    indices = sample["indices"]
    if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes)):
        raise ValueError("score sample indices are invalid")
    frozen_indices = tuple(indices)
    if (
        not frozen_indices
        or any(type(index) is not int or index < 0 for index in frozen_indices)
        or frozen_indices != tuple(sorted(set(frozen_indices)))
        or sample["bounded_series_count"] != len(frozen_indices)
        or sample["train_count"] < len(frozen_indices)
    ):
        raise ValueError("score sample indices are inconsistent")
    reports_raw = value["reports"]
    if not isinstance(reports_raw, Sequence) or isinstance(reports_raw, (str, bytes)):
        raise ValueError("score reports are invalid")
    reports = tuple(_parse_score_report(report) for report in reports_raw)
    stage = _parse_stage(value["stage"], "score")
    if stage.state == "skipped":
        if reports:
            raise ValueError("skipped score stage cannot contain reports")
    elif tuple(report.batch_size for report in reports) != FROZEN_CONFIG.candidate_batch_sizes:
        raise ValueError("score reports must preserve the exact batch schedule")
    if stage.state == "passed" and not all(report.passes for report in reports):
        raise ValueError("passed score stage contains a failed report")
    if stage.state == "failed" and reports and all(report.passes for report in reports):
        raise ValueError("failed score stage is inconsistent with all-passing reports")
    digest_names = fields - {"sample", "reports", "stage"}
    for name in digest_names:
        digest = value[name]
        if stage.state == "skipped":
            if digest is not None:
                raise ValueError("skipped score stage cannot retain digests")
        else:
            _sha256_digest(digest, name)
    return stage, reports


def _fallback_reasons(
    core: CoreStageResult,
    score_stage: StageResult | None,
    score_reports: Sequence[ScoreBatchReport],
) -> tuple[str, ...]:
    ordered: list[str] = []
    for reason in (
        core.capability.fallback_reason,
        core.stage.reason,
        None if score_stage is None else score_stage.reason,
        *(report.fallback_reason for report in score_reports),
    ):
        if reason is not None and reason not in ordered:
            ordered.append(reason)
    return tuple(ordered)


def _validate_provenance(value: object, capability: DeviceCapability) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _PROVENANCE_FIELDS:
        raise ValueError("manifest provenance must contain the exact required fields")
    provenance = dict(value)
    if provenance["config_sha256"] != frozen_config_sha256():
        raise ValueError("config_sha256 must bind the frozen Exp422 config")
    if provenance["database_sha256"] != FROZEN_CONFIG.exp417_config.database_sha256:
        raise ValueError("database_sha256 must bind the frozen Exp417 database")
    _sha256_digest(provenance["source_manifest_sha256"], "source_manifest_sha256")
    if not isinstance(provenance["git_commit"], str) or _GIT_RE.fullmatch(provenance["git_commit"]) is None:
        raise ValueError("git_commit must be a canonical lowercase Git digest")
    if (
        not isinstance(provenance["source_git_commit"], str)
        or _GIT_RE.fullmatch(provenance["source_git_commit"]) is None
    ):
        raise ValueError("source_git_commit must be a canonical lowercase Git digest")
    if provenance["sqlite_quick_check"] != "ok":
        raise ValueError("SQLite quick-check must be ok")
    if type(provenance["repeats"]) is not int or not 1 <= provenance["repeats"] <= 7:
        raise ValueError("repeats must be between 1 and 7")
    if provenance["fresh_no_clobber_attempt"] is not True:
        raise ValueError("fresh_no_clobber_attempt must be true")
    if provenance["train_only"] is not True:
        raise ValueError("train_only must be true")
    capability_fields = {
        "torch_version": capability.torch_version,
        "mps_built": capability.mps_built,
        "mps_available": capability.mps_available,
        "device_name": capability.device_name,
        "mlx_version": capability.mlx_version,
        "mlx_metal_version": capability.mlx_metal_version,
        "mlx_availability": capability.mlx_availability,
    }
    for name, expected in capability_fields.items():
        if provenance[name] != expected:
            raise ValueError(f"provenance {name} must match core capability")
    expected_mlx_reason = (
        "algorithm_rewrite_required"
        if capability.mlx_availability == "available"
        else "package_absent"
    )
    if provenance["mlx_eligibility_reason"] != expected_mlx_reason:
        raise ValueError("MLX eligibility reason is inconsistent with availability")
    if not isinstance(provenance["attempt_root"], str) or not provenance["attempt_root"].strip():
        raise ValueError("attempt_root must be a nonempty path")
    return provenance


def _scan_forbidden_values(value: object) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            _scan_forbidden_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _scan_forbidden_values(child)
    elif isinstance(value, str) and _FORBIDDEN_VALUE_TOKEN.search(value):
        raise ValueError("manifest values must not contain TEST or evaluation tokens")


@dataclass(frozen=True)
class Exp422Manifest:
    """Validated immutable JSON-ready evidence for one bounded attempt."""

    state: str
    candidate_batch_sizes: tuple[int, ...]
    core_stage: Mapping[str, object]
    score_stage: Mapping[str, object] | None
    fallback_reasons: tuple[str, ...]
    provenance: Mapping[str, object]
    experiment_id: str = "Exp422"
    schema_version: int = _SCHEMA_VERSION
    artifact_kind: str = _ARTIFACT_KIND

    def __post_init__(self) -> None:
        if self.state not in {"passed", "failed"}:
            raise ValueError("manifest state must be passed or failed")
        if self.candidate_batch_sizes != FROZEN_CONFIG.candidate_batch_sizes:
            raise ValueError("manifest batch schedule is not frozen")
        if self.experiment_id != FROZEN_CONFIG.experiment_id:
            raise ValueError("manifest experiment_id is invalid")
        if self.schema_version != _SCHEMA_VERSION or self.artifact_kind != _ARTIFACT_KIND:
            raise ValueError("manifest schema identity is invalid")
        object.__setattr__(self, "core_stage", _freeze(self.core_stage))
        object.__setattr__(self, "score_stage", None if self.score_stage is None else _freeze(self.score_stage))
        object.__setattr__(self, "fallback_reasons", tuple(self.fallback_reasons))
        object.__setattr__(self, "provenance", _freeze(self.provenance))

    def to_payload(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "state": self.state,
            "candidate_batch_sizes": list(self.candidate_batch_sizes),
            "core_stage": _thaw(self.core_stage),
            "score_stage": _thaw(self.score_stage),
            "fallback_reasons": list(self.fallback_reasons),
            "provenance": _thaw(self.provenance),
        }


def build_manifest(
    core: CoreStageResult,
    score: ScoreStageResult | None,
    provenance: Mapping[str, object],
) -> Exp422Manifest:
    """Build and revalidate one complete manifest from typed stage evidence."""
    if not isinstance(core, CoreStageResult):
        raise ValueError("core must be a CoreStageResult")
    if score is not None and not isinstance(score, ScoreStageResult):
        raise ValueError("score must be a ScoreStageResult or None")
    if core.stage.state == "passed" and score is None:
        raise ValueError("passed core stage requires score evidence")
    if core.stage.state != "passed" and score is not None:
        raise ValueError("failed or skipped core stage forbids score evidence")
    score_stage = None if score is None else score.stage
    score_reports = () if score is None else score.reports
    state = "passed" if score_stage is not None and score_stage.state == "passed" else "failed"
    payload = {
        "artifact_kind": _ARTIFACT_KIND,
        "schema_version": _SCHEMA_VERSION,
        "experiment_id": FROZEN_CONFIG.experiment_id,
        "state": state,
        "candidate_batch_sizes": list(FROZEN_CONFIG.candidate_batch_sizes),
        "core_stage": _core_payload(core),
        "score_stage": None if score is None else _score_payload(score),
        "fallback_reasons": list(_fallback_reasons(core, score_stage, score_reports)),
        "provenance": dict(provenance),
    }
    return validate_manifest(payload)


def validate_manifest(payload: Mapping[str, object]) -> Exp422Manifest:
    """Reject noncanonical evidence, altered gates, or inconsistent stage ordering."""
    if not isinstance(payload, Mapping) or set(payload) != _TOP_LEVEL_FIELDS:
        raise ValueError("manifest top-level schema is invalid")
    plain = _thaw(payload)
    if not isinstance(plain, dict):
        raise ValueError("manifest payload is invalid")
    if plain["artifact_kind"] != _ARTIFACT_KIND or plain["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("manifest schema identity is invalid")
    if plain["experiment_id"] != FROZEN_CONFIG.experiment_id:
        raise ValueError("manifest experiment_id is invalid")
    if tuple(plain["candidate_batch_sizes"]) != FROZEN_CONFIG.candidate_batch_sizes:
        raise ValueError("manifest batch schedule is not frozen")
    core = _parse_core(plain["core_stage"])
    provenance = _validate_provenance(plain["provenance"], core.capability)
    raw_score = plain["score_stage"]
    score_stage: StageResult | None = None
    score_reports: tuple[ScoreBatchReport, ...] = ()
    if raw_score is not None:
        score_stage, score_reports = _validate_score(raw_score)
    if core.stage.state == "passed" and score_stage is None:
        raise ValueError("passed core stage requires score evidence")
    if core.stage.state != "passed" and score_stage is not None:
        raise ValueError("failed or skipped core stage forbids score evidence")
    expected_state = "passed" if score_stage is not None and score_stage.state == "passed" else "failed"
    if plain["state"] != expected_state:
        raise ValueError("manifest state is inconsistent with stage states")
    expected_reasons = _fallback_reasons(core, score_stage, score_reports)
    if tuple(plain["fallback_reasons"]) != expected_reasons:
        raise ValueError("manifest fallback reasons are inconsistent")
    _scan_forbidden_values(plain)
    _canonical_json_bytes(plain)
    return Exp422Manifest(
        state=expected_state,
        candidate_batch_sizes=FROZEN_CONFIG.candidate_batch_sizes,
        core_stage=plain["core_stage"],
        score_stage=raw_score,
        fallback_reasons=expected_reasons,
        provenance=provenance,
    )


def write_manifest(path: Path, manifest: Exp422Manifest) -> Path:
    """Write one canonical JSON line without replacing any prior evidence."""
    if not isinstance(manifest, Exp422Manifest):
        raise ValueError("manifest must be an Exp422Manifest")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(manifest.to_payload()) + b"\n"
    with target.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return target


def _read_regular_no_follow(path: Path, description: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{description} must be a readable regular file") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{description} must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _validated_source_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("source manifest path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or "\\" in value or value != path.as_posix() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("source manifest path traversal is forbidden")
    return value


def _validated_git_commit(value: object, name: str = "git_commit") -> str:
    if not isinstance(value, str) or _GIT_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical lowercase Git digest")
    return value


def _git_stdout(repo_root: Path, args: Sequence[str], description: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path(repo_root),
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"unable to verify {description}") from exc
    return result.stdout


def _git_commit_exists(repo_root: Path, commit: str) -> None:
    resolved = _git_stdout(
        repo_root,
        ["rev-parse", "--verify", f"{commit}^{{commit}}"],
        "source-manifest Git commit",
    ).decode("ascii", errors="strict").strip()
    if resolved != commit:
        raise ValueError("source-manifest Git commit is not an exact commit object")


def _git_head(repo_root: Path) -> str:
    head = _git_stdout(repo_root, ["rev-parse", "--verify", "HEAD"], "current Git HEAD")
    return _validated_git_commit(head.decode("ascii", errors="strict").strip(), "current Git HEAD")


def _git_worktree_is_clean(repo_root: Path) -> None:
    for args in (("diff", "--quiet", "HEAD", "--"), ("diff", "--cached", "--quiet", "HEAD", "--")):
        try:
            subprocess.run(["git", *args], cwd=Path(repo_root), check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            raise ValueError("source-manifest Git worktree has tracked changes") from exc
        except OSError as exc:
            raise ValueError("unable to verify source-manifest Git worktree") from exc


def _git_blob(repo_root: Path, commit: str, relative: str) -> bytes:
    return _git_stdout(repo_root, ["show", f"{commit}:{relative}"], f"Git source blob {relative}")


def _source_manifest_payload(raw: bytes) -> tuple[dict[str, object], str]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("source manifest is invalid JSON") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"artifact_kind", "schema_version", "git_commit", "sources"}
        or payload["artifact_kind"] != _SOURCE_ARTIFACT_KIND
        or payload["schema_version"] != _SCHEMA_VERSION
        or not isinstance(payload["sources"], list)
    ):
        raise ValueError("source manifest schema is invalid")
    if raw != _canonical_json_bytes(payload) + b"\n":
        raise ValueError("source manifest must use canonical JSON")
    return payload, _validated_git_commit(payload["git_commit"], "source-manifest git_commit")


def propose_source_manifest(repo_root: Path, *, git_commit: str) -> bytes:
    """Return canonical reviewed-source bytes bound to an already-committed source snapshot."""
    root = Path(repo_root)
    source_commit = _validated_git_commit(git_commit)
    _git_commit_exists(root, source_commit)
    sources = []
    for relative in _SOURCE_PATHS:
        data = _read_regular_no_follow(root / relative, f"source {relative}")
        sources.append({"path": relative, "sha256": sha256(data).hexdigest()})
    payload = {
        "artifact_kind": _SOURCE_ARTIFACT_KIND,
        "schema_version": _SCHEMA_VERSION,
        "git_commit": source_commit,
        "sources": sources,
    }
    return _canonical_json_bytes(payload) + b"\n"


def source_manifest_git_commit(
    repo_root: Path,
    manifest_path: Path | None = None,
) -> str:
    """Read the source snapshot commit after validating manifest syntax."""
    root = Path(repo_root)
    path = root / SOURCE_MANIFEST_RELATIVE if manifest_path is None else Path(manifest_path)
    raw = _read_regular_no_follow(path, "source manifest")
    _payload, commit = _source_manifest_payload(raw)
    return commit


def verify_source_manifest(
    repo_root: Path,
    manifest_path: Path | None = None,
) -> str:
    """Verify declared source bytes against both the working tree and a Git snapshot."""
    root = Path(repo_root)
    path = root / SOURCE_MANIFEST_RELATIVE if manifest_path is None else Path(manifest_path)
    raw = _read_regular_no_follow(path, "source manifest")
    payload, source_commit = _source_manifest_payload(raw)
    _git_commit_exists(root, source_commit)
    head = _git_head(root)
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, head],
            cwd=root,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError("source-manifest Git commit is not an ancestor of current HEAD") from exc
    except OSError as exc:
        raise ValueError("unable to verify source-manifest Git ancestry") from exc
    _git_worktree_is_clean(root)
    entries = payload["sources"]
    if any(not isinstance(entry, dict) or set(entry) != {"path", "sha256"} for entry in entries):
        raise ValueError("source manifest entries are invalid")
    paths = tuple(_validated_source_path(entry["path"]) for entry in entries)
    if paths != _SOURCE_PATHS or len(paths) != len(set(paths)):
        raise ValueError("source manifest source set is missing, extra, duplicated, or reordered")
    for entry, relative in zip(entries, paths, strict=True):
        expected = _sha256_digest(entry["sha256"], f"source digest {relative}")
        current = _read_regular_no_follow(root / relative, f"source {relative}")
        actual = sha256(current).hexdigest()
        committed = sha256(_git_blob(root, source_commit, relative)).hexdigest()
        if actual != expected or committed != expected:
            raise ValueError(f"source digest mismatch: {relative}")
    return sha256(raw).hexdigest()


__all__ = (
    "Exp422Manifest",
    "SOURCE_MANIFEST_RELATIVE",
    "build_manifest",
    "propose_source_manifest",
    "source_manifest_git_commit",
    "validate_manifest",
    "verify_source_manifest",
    "write_manifest",
)
