"""Deterministic, independently revalidated Exp417 Shadow reporting."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import io
import json
from math import isfinite
from types import MappingProxyType
from typing import Any

from experiments.exp403_408.artifact_io import canonical_json_bytes

from .checkpoints import (
    Exp417CheckpointProvenance,
    Exp417CheckpointStore,
    canonical_result_sha256,
)
from .backend import FROZEN_TASK7_BACKEND_AUDIT_SHA256
from .config import Exp417FrozenConfig, FROZEN_CONFIG
from .contracts import (
    BackendAudit,
    DatasetDecisionEvidence,
    DatasetResult,
    FrozenLaneSnapshot,
    FrozenRunContext,
    FrozenTrainContract,
    GroupQualification,
    HorizonScore,
    RunSummary,
    SeedDecision,
    ShadowCandidate,
    TestInstanceDecision,
)
PREDICTION_FILE = "exp417_shadow_predictions.jsonl"
DATASET_FILE = "exp417_dataset_qualification.csv"
OOF_FILE = "exp417_oof_normal_metrics.csv"
SYNTHETIC_FILE = "exp417_synthetic_safety.csv"
BACKEND_FILE = "exp417_backend_benchmark.json"
CONFIG_FILE = "exp417_frozen_config.json"
REPORT_FILE = "EXP417_RESULT_REPORT_KO.md"
MANIFEST_FILE = "MANIFEST.json"
NON_MANIFEST_FILES = frozenset(
    {
        PREDICTION_FILE,
        DATASET_FILE,
        OOF_FILE,
        SYNTHETIC_FILE,
        BACKEND_FILE,
        CONFIG_FILE,
        REPORT_FILE,
    }
)
OFFICIAL_FILES = NON_MANIFEST_FILES | {MANIFEST_FILE}

_DATASET_HEADER = (
    "dataset", "status", "test_size", "group_count",
    "observed_candidate_count", "published_candidate_count",
    "candidate_rate", "dataset_workload_passed",
    "terminal_stage", "terminal_reason",
)
_OOF_HEADER = (
    "dataset", "length", "series_count", "status",
    "oof_candidate_rate", "wilson_upper", "seed_agreement", "finite",
    "backend_fallback",
)
_SYNTHETIC_HEADER = (
    "dataset", "length", "pooled_synthetic_sensitivity",
    "amplitude_scaling", "gradual_drift", "level_shift", "period_change",
)
_GOVERNANCE = {
    "official_prediction": True,
    "publication_eligible": True,
    "contains_test_labels": False,
    "test_informed_reselection": False,
    "lane_promotion": False,
    "tail_replacement": False,
    "top_n_trimming": False,
    "mean_combined_f1_scope": "human_assisted_only",
}
_FORBIDDEN_KEYS = frozenset(
    {
        "ground_truth",
        "confusion_matrix",
        "precision",
        "recall",
        "f1",
        "mean_combined_f1",
        "actual_anomaly_location",
    }
)


@dataclass(frozen=True)
class TrustedCheckpointProvenance:
    """External result roots obtained from a verified complete checkpoint store."""

    checkpoint_provenance_sha256: str
    result_evidence_sha256: Mapping[str, str]
    checkpoint_payload_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        provenance_digest = _digest(
            self.checkpoint_provenance_sha256, "checkpoint provenance"
        )
        result_digests = _trusted_digest_mapping(
            self.result_evidence_sha256, "result payload"
        )
        checkpoint_digests = _trusted_digest_mapping(
            self.checkpoint_payload_sha256, "checkpoint record payload"
        )
        if tuple(result_digests) != tuple(checkpoint_digests):
            raise ValueError(
                "Exp417 trusted result and checkpoint maps must cover identical datasets"
            )
        object.__setattr__(self, "checkpoint_provenance_sha256", provenance_digest)
        object.__setattr__(
            self, "result_evidence_sha256", MappingProxyType(result_digests)
        )
        object.__setattr__(
            self,
            "checkpoint_payload_sha256",
            MappingProxyType(checkpoint_digests),
        )

    @classmethod
    def from_store(
        cls, store: Exp417CheckpointStore
    ) -> "TrustedCheckpointProvenance":
        if type(store) is not Exp417CheckpointStore:
            raise ValueError(
                "Exp417 trusted checkpoint root requires Exp417CheckpointStore"
            )
        completed, record_digests = store.require_complete_snapshot()
        return cls(
            checkpoint_provenance_sha256=sha256(
                canonical_json_bytes(store.provenance.to_payload())
            ).hexdigest(),
            result_evidence_sha256={
                dataset: canonical_result_sha256(completed[dataset])
                for dataset in store.expected_datasets
            },
            checkpoint_payload_sha256=record_digests,
        )


def render_exp417_payloads(
    summary: RunSummary,
    provenance: Exp417CheckpointProvenance,
    trusted_checkpoint: TrustedCheckpointProvenance,
    *,
    config: Exp417FrozenConfig = FROZEN_CONFIG,
) -> Mapping[str, bytes]:
    """Render the seven official non-manifest payloads from one frozen summary."""
    config.validate()
    if not isinstance(summary, RunSummary):
        raise ValueError("Exp417 reporting requires a RunSummary")
    if (
        summary.config if hasattr(summary, "config") else summary.context.config
    ) != config:
        raise ValueError("Exp417 reporting config does not match the summary")
    if summary.status != "complete" or not summary.publication_allowed:
        raise ValueError("Exp417 global publication gate forbids official rendering")
    if not isinstance(provenance, Exp417CheckpointProvenance):
        raise ValueError("Exp417 reporting requires checkpoint provenance")
    verified_provenance = provenance.to_payload()
    _verify_provenance_context(summary.context, verified_provenance)
    _verify_trusted_checkpoint(
        trusted_checkpoint, provenance, summary.results
    )

    predictions = _render_predictions(summary)
    dataset_csv = _render_dataset_csv(summary)
    oof_csv = _render_oof_csv(summary)
    synthetic_csv = _render_synthetic_csv(summary)
    backend_json = _render_backend(summary, provenance)
    rendered = {
        PREDICTION_FILE: predictions,
        DATASET_FILE: dataset_csv,
        OOF_FILE: oof_csv,
        SYNTHETIC_FILE: synthetic_csv,
        BACKEND_FILE: backend_json,
    }
    config_json = _render_config(
        summary, provenance, trusted_checkpoint, config
    )
    backend_counts = Counter(result.decision_evidence.frozen_train_contract.backend.backend
                             for result in summary.results)
    report = _render_report(
        dataset_count=len(summary.results),
        total_test=summary.total_test,
        observed_candidates=sum(result.observed_candidate_count for result in summary.results),
        published_candidates=sum(len(result.candidates) for result in summary.results),
        backend_counts=backend_counts,
    )
    payloads = MappingProxyType(
        {
            **rendered,
            CONFIG_FILE: config_json,
            REPORT_FILE: report,
        }
    )
    verify_exp417_payloads(
        payloads, trusted_checkpoint=trusted_checkpoint, config=config
    )
    return payloads


def verify_exp417_payloads(
    payloads: Mapping[str, bytes],
    *,
    trusted_checkpoint: TrustedCheckpointProvenance,
    config: Exp417FrozenConfig = FROZEN_CONFIG,
) -> None:
    """Reconstruct immutable contracts and cross-check all seven payloads."""
    config.validate()
    if (
        not isinstance(payloads, Mapping)
        or set(payloads) != NON_MANIFEST_FILES
        or len(payloads) != len(NON_MANIFEST_FILES)
        or any(type(name) is not str or type(value) is not bytes for name, value in payloads.items())
    ):
        raise ValueError("Exp417 reporting requires the exact seven non-manifest payloads")

    config_payload = _canonical_json_object(payloads[CONFIG_FILE], CONFIG_FILE)
    _exact_keys(
        config_payload,
        {
            "context", "experiment_id", "frozen_config",
            "frozen_config_sha256", "governance", "provenance",
            "result_evidence", "run",
        },
        CONFIG_FILE,
    )
    if config_payload["experiment_id"] != config.experiment_id:
        raise ValueError("Exp417 rendered experiment identity mismatch")
    frozen_config = _object(config_payload["frozen_config"], "frozen config")
    expected_frozen_config = json.loads(canonical_json_bytes(asdict(config)))
    if frozen_config != expected_frozen_config:
        raise ValueError("Exp417 rendered frozen config mismatch")
    frozen_digest = sha256(canonical_json_bytes(frozen_config)).hexdigest()
    if config_payload["frozen_config_sha256"] != frozen_digest:
        raise ValueError("Exp417 frozen config digest mismatch")
    governance = _object(config_payload["governance"], "governance")
    if governance != _GOVERNANCE:
        raise ValueError("Exp417 governance flags are not frozen")
    _reject_forbidden_keys(config_payload)

    provenance_payload = _object(config_payload["provenance"], "provenance")
    provenance = _reconstruct_provenance(provenance_payload)
    if not isinstance(trusted_checkpoint, TrustedCheckpointProvenance):
        raise ValueError("Exp417 verification requires trusted checkpoint provenance")
    expected_provenance_digest = sha256(
        canonical_json_bytes(provenance.to_payload())
    ).hexdigest()
    if (
        trusted_checkpoint.checkpoint_provenance_sha256
        != expected_provenance_digest
    ):
        raise ValueError("Exp417 checkpoint provenance root mismatch")
    if provenance.frozen_config_sha256 != frozen_digest:
        raise ValueError("Exp417 provenance frozen config digest mismatch")
    context = _reconstruct_context(
        _object(config_payload["context"], "context"), config
    )
    _verify_provenance_context(context, provenance.to_payload())

    candidates = _parse_predictions(payloads[PREDICTION_FILE], context)
    datasets = _parse_dataset_csv(payloads[DATASET_FILE], context)
    qualifications = _parse_group_csvs(
        payloads[OOF_FILE], payloads[SYNTHETIC_FILE], datasets
    )
    audits = _parse_backend(payloads[BACKEND_FILE], datasets, provenance)
    reconstructed = _reconstruct_results(
        _list(config_payload["result_evidence"], "result evidence"),
        context,
        datasets,
        qualifications,
        audits,
        trusted_checkpoint,
    )
    _verify_trusted_checkpoint(
        trusted_checkpoint, provenance, reconstructed
    )
    run = _object(config_payload["run"], "run")
    _verify_conservation(run, context, candidates, datasets, qualifications, audits)
    summary = RunSummary(
        tuple(reconstructed),
        _integer(run["total_test"], "run total_test"),
        True,
        "complete",
        context,
    )
    if _render_predictions(summary) != payloads[PREDICTION_FILE]:
        raise ValueError("Exp417 predictions do not match complete result evidence")

    backend_counts = Counter(audit.backend for audit in audits.values())
    expected_report = _render_report(
        dataset_count=len(datasets),
        total_test=_integer(run["total_test"], "run total_test"),
        observed_candidates=_integer(
            run["observed_candidate_count"], "run observed candidates"
        ),
        published_candidates=_integer(
            run["published_candidate_count"], "run published candidates"
        ),
        backend_counts=backend_counts,
    )
    if payloads[REPORT_FILE] != expected_report:
        raise ValueError("Exp417 Korean report does not match verified counts")


def _render_predictions(summary: RunSummary) -> bytes:
    rows: list[dict[str, object]] = []
    for result in summary.results:
        for candidate in result.candidates:
            body = {
                "dataset": candidate.dataset,
                "instance_index": candidate.instance_index,
                "length": candidate.length,
                "seed_decisions": _canonical_seed_decisions_payload(
                    candidate.seed_decisions
                ),
                "is_hard": False,
                "is_standard": False,
                "is_priority": False,
                "is_automatic_normal": False,
            }
            body["evidence_sha256"] = sha256(canonical_json_bytes(body)).hexdigest()
            rows.append(body)
    rows.sort(key=lambda row: (str(row["dataset"]), int(row["instance_index"])))
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _canonical_seed_decisions_payload(
    decisions: Sequence[SeedDecision],
) -> list[dict[str, object]]:
    by_seed = {decision.seed: decision for decision in decisions}
    if set(by_seed) != set(FROZEN_CONFIG.model_seeds) or len(by_seed) != len(decisions):
        raise ValueError("Exp417 seed decisions do not cover the frozen seeds")
    return [
        {
            "seed": seed,
            "candidate": by_seed[seed].candidate,
            "scores": [
                {
                    "horizon": score.horizon,
                    "score": score.score,
                    "p_value": score.p_value,
                    "persistence": score.persistence,
                }
                for score in by_seed[seed].scores.values()
            ],
        }
        for seed in FROZEN_CONFIG.model_seeds
    ]


def _result_evidence_payload(
    result: DatasetResult, trusted: TrustedCheckpointProvenance
) -> dict[str, object]:
    if not isinstance(result.decision_evidence, DatasetDecisionEvidence):
        raise ValueError("Exp417 official reporting requires complete decision evidence")
    evidence = result.decision_evidence
    return {
        "dataset": result.dataset,
        "train_identity_sha256": evidence.frozen_train_contract.train_identity_sha256,
        "decisions": [
            {
                "dataset": decision.dataset,
                "instance_index": decision.instance_index,
                "length": decision.length,
                "seed_decisions": _canonical_seed_decisions_payload(
                    decision.seed_decisions
                ),
            }
            for decision in evidence.decisions
        ],
        "checkpoint_result_payload_sha256": trusted.result_evidence_sha256[
            result.dataset
        ],
        "checkpoint_record_payload_sha256": trusted.checkpoint_payload_sha256[
            result.dataset
        ],
    }


def _render_dataset_csv(summary: RunSummary) -> bytes:
    rows = [_DATASET_HEADER]
    for result in summary.results:
        terminal = getattr(result.decision_evidence, "stage", "")
        reason = getattr(result.decision_evidence, "reason", "")
        rows.append(
            (
                result.dataset,
                result.status,
                str(result.test_size),
                str(result.group_count),
                str(result.observed_candidate_count),
                str(len(result.candidates)),
                _float(result.candidate_rate),
                _bool_text(result.candidate_rate <= FROZEN_CONFIG.dataset_workload_rate),
                terminal,
                reason,
            )
        )
    return _csv_bytes(rows)


def _render_oof_csv(summary: RunSummary) -> bytes:
    rows = [_OOF_HEADER]
    for result in summary.results:
        for group in result.groups:
            rows.append(
                (
                    result.dataset,
                    str(group.length),
                    str(group.series_count),
                    group.status,
                    _optional_float(group.oof_candidate_rate),
                    _optional_float(group.wilson_upper),
                    _optional_bool_text(group.seed_agreement),
                    _optional_bool_text(group.finite),
                    _optional_bool_text(group.backend_fallback),
                )
            )
    return _csv_bytes(rows)


def _render_synthetic_csv(summary: RunSummary) -> bytes:
    rows = [_SYNTHETIC_HEADER]
    for result in summary.results:
        for group in result.groups:
            sensitivity = group.per_type_synthetic_sensitivity
            rows.append(
                (
                    result.dataset,
                    str(group.length),
                    _optional_float(group.pooled_synthetic_sensitivity),
                    *(
                        _optional_float(sensitivity[name])
                        for name in _SYNTHETIC_HEADER[3:]
                    ),
                )
            )
    return _csv_bytes(rows)


def _render_backend(
    summary: RunSummary, provenance: Exp417CheckpointProvenance
) -> bytes:
    audits = []
    for result in summary.results:
        audit = result.decision_evidence.frozen_train_contract.backend
        audits.append({"dataset": result.dataset, **asdict(audit)})
    return _json_bytes(
        {
            "experiment_id": FROZEN_CONFIG.experiment_id,
            "backend_contract_sha256": provenance.backend_contract_sha256,
            "trusted_task7_backend_audit_sha256": FROZEN_TASK7_BACKEND_AUDIT_SHA256,
            "audits": audits,
        }
    )


def _render_config(
    summary: RunSummary,
    provenance: Exp417CheckpointProvenance,
    trusted: TrustedCheckpointProvenance,
    config: Exp417FrozenConfig,
) -> bytes:
    context = summary.context
    observed = sum(result.observed_candidate_count for result in summary.results)
    published = sum(len(result.candidates) for result in summary.results)
    return _json_bytes(
        {
            "experiment_id": config.experiment_id,
            "frozen_config": asdict(config),
            "frozen_config_sha256": provenance.frozen_config_sha256,
            "context": {
                "database_sha256": context.database_sha256,
                "datasets": list(context.datasets),
                "lanes": {
                    name: {dataset: list(indices) for dataset, indices in getattr(context.lanes, name).items()}
                    for name in ("hard", "standard", "priority", "automatic_normal")
                },
                "test_identity_manifest": {
                    dataset: [list(entry) for entry in entries]
                    for dataset, entries in context.test_identity_manifest.items()
                },
                "code_commit": context.code_commit,
                "runtime_versions": dict(context.runtime_versions),
            },
            "provenance": provenance.to_payload(),
            "governance": _GOVERNANCE,
            "run": {
                "status": summary.status,
                "publication_allowed": summary.publication_allowed,
                "dataset_count": len(summary.results),
                "total_test": summary.total_test,
                "observed_candidate_count": observed,
                "published_candidate_count": published,
                "global_candidate_rate": published / summary.total_test,
            },
            "result_evidence": [
                _result_evidence_payload(result, trusted)
                for result in summary.results
            ],
        }
    )


def _render_report(
    *,
    dataset_count: int,
    total_test: int,
    observed_candidates: int,
    published_candidates: int,
    backend_counts: Mapping[str, int],
) -> bytes:
    backend_text = ", ".join(
        f"{name} {count}" for name, count in sorted(backend_counts.items())
    )
    return (
        "# Exp417 Causal Multi-Horizon Shadow 결과 보고서\n\n"
        "## 거버넌스\n\n"
        "- 후보 생성과 안전성 검증은 정상 TRAIN만 사용했다.\n"
        "- TEST 라벨을 사용하지 않음.\n"
        "- 결과는 검토 전용 Shadow이며 자동 경보로 승격하지 않음.\n"
        "- 기존 Hard/Standard/Priority/automatic-normal lane은 변경하지 않았다.\n"
        "- tail replacement와 top-N trimming을 사용하지 않았다.\n\n"
        "## 보존 집계\n\n"
        f"- 데이터셋 수: {dataset_count}\n"
        f"- 전체 TEST 수: {total_test}\n"
        f"- workload gate 전 후보 수: {observed_candidates}\n"
        f"- 공식 Shadow 후보 수: {published_candidates}\n"
        f"- 백엔드: {backend_text}\n"
    ).encode("utf-8")


def _parse_predictions(
    payload: bytes, context: FrozenRunContext
) -> tuple[ShadowCandidate, ...]:
    candidates: list[ShadowCandidate] = []
    previous: tuple[str, int] | None = None
    for line_number, raw in enumerate(payload.splitlines(), 1):
        row = _json_line_object(raw, f"prediction line {line_number}")
        _exact_keys(
            row,
            {
                "dataset", "evidence_sha256", "instance_index", "is_automatic_normal",
                "is_hard", "is_priority", "is_standard", "length", "seed_decisions",
            },
            "prediction",
        )
        digest = row.pop("evidence_sha256")
        if digest != sha256(canonical_json_bytes(row)).hexdigest():
            raise ValueError("Exp417 candidate evidence digest mismatch")
        if any(
            row[name] is not False
            for name in ("is_hard", "is_standard", "is_priority", "is_automatic_normal")
        ):
            raise ValueError("Exp417 prediction governance flags must remain false")
        dataset = _string(row["dataset"], "prediction dataset")
        index = _integer(row["instance_index"], "prediction instance")
        coordinate = (dataset, index)
        if previous is not None and coordinate <= previous:
            raise ValueError("Exp417 predictions must be sorted and unique")
        previous = coordinate
        if dataset not in context.test_sizes or context.lanes.contains(dataset, index):
            raise ValueError("Exp417 prediction coordinate is not eligible")
        manifest = dict(context.test_identity_manifest[dataset])
        length = _integer(row["length"], "prediction length", minimum=1)
        if manifest.get(index) != length:
            raise ValueError("Exp417 prediction length differs from frozen TEST identity")
        decisions = tuple(
            _reconstruct_seed_decision(value)
            for value in _list(row["seed_decisions"], "seed decisions")
        )
        if tuple(decision.seed for decision in decisions) != FROZEN_CONFIG.model_seeds:
            raise ValueError("Exp417 seed decisions are not in frozen canonical order")
        candidates.append(ShadowCandidate(dataset, index, length, decisions))
    return tuple(candidates)


def _reconstruct_seed_decision(value: object) -> SeedDecision:
    row = _object(value, "seed decision")
    _exact_keys(row, {"candidate", "scores", "seed"}, "seed decision")
    scores: dict[int, HorizonScore] = {}
    for item in _list(row["scores"], "horizon scores"):
        score = _object(item, "horizon score")
        _exact_keys(
            score, {"horizon", "p_value", "persistence", "score"}, "horizon score"
        )
        record = HorizonScore(
            _integer(score["horizon"], "horizon", minimum=1),
            _number(score["score"], "score"),
            _number(score["p_value"], "p value"),
            _number(score["persistence"], "persistence"),
        )
        if record.horizon in scores:
            raise ValueError("Exp417 duplicate horizon score")
        scores[record.horizon] = record
    return SeedDecision(
        seed=_integer(row["seed"], "seed", minimum=1),
        scores=dict(sorted(scores.items())),
        candidate=_bool(row["candidate"], "seed candidate"),
    )


def _reconstruct_results(
    values: Sequence[object],
    context: FrozenRunContext,
    datasets: Mapping[str, Mapping[str, object]],
    qualifications: Mapping[tuple[str, int], GroupQualification],
    audits: Mapping[str, BackendAudit],
    trusted_checkpoint: TrustedCheckpointProvenance,
) -> tuple[DatasetResult, ...]:
    results: list[DatasetResult] = []
    observed_names: list[str] = []
    for value in values:
        row = _object(value, "result evidence")
        _exact_keys(
            row,
            {
                "checkpoint_record_payload_sha256",
                "checkpoint_result_payload_sha256",
                "dataset",
                "decisions",
                "train_identity_sha256",
            },
            "result evidence",
        )
        dataset = _string(row["dataset"], "result evidence dataset")
        observed_names.append(dataset)
        if dataset not in datasets:
            raise ValueError("Exp417 result evidence dataset is outside the context")
        groups = tuple(
            group
            for (group_dataset, _), group in qualifications.items()
            if group_dataset == dataset
        )
        contract = FrozenTrainContract(
            dataset=dataset,
            groups=groups,
            backend=audits[dataset],
            train_identity_sha256=_digest(
                row["train_identity_sha256"], "TRAIN identity"
            ),
        )
        decisions: list[TestInstanceDecision] = []
        for raw_decision in _list(row["decisions"], "instance decisions"):
            decision = _object(raw_decision, "instance decision")
            _exact_keys(
                decision,
                {"dataset", "instance_index", "length", "seed_decisions"},
                "instance decision",
            )
            seed_values = _list(decision["seed_decisions"], "seed decisions")
            seed_records = tuple(
                _reconstruct_seed_decision(item) for item in seed_values
            )
            if tuple(record.seed for record in seed_records) != FROZEN_CONFIG.model_seeds:
                raise ValueError("Exp417 seed decisions are not in frozen canonical order")
            decisions.append(
                TestInstanceDecision(
                    dataset=_string(decision["dataset"], "decision dataset"),
                    instance_index=_integer(
                        decision["instance_index"], "decision instance"
                    ),
                    length=_integer(decision["length"], "decision length", minimum=1),
                    seed_decisions=seed_records,
                )
            )
        evidence = DatasetDecisionEvidence(
            dataset=dataset,
            context=context,
            frozen_train_contract=contract,
            database_sha256=context.database_sha256,
            train_identity_sha256=contract.train_identity_sha256,
            decisions=tuple(decisions),
        )
        dataset_row = datasets[dataset]
        result = DatasetResult(
            dataset=dataset,
            test_size=int(dataset_row["test_size"]),
            group_count=int(dataset_row["group_count"]),
            groups=groups,
            candidates=(
                evidence.derived_candidates
                if dataset_row["status"] == "qualified"
                else ()
            ),
            decision_evidence=evidence,
            lanes=context.lanes,
            context=context,
            status=str(dataset_row["status"]),
        )
        if (
            row["checkpoint_result_payload_sha256"]
            != trusted_checkpoint.result_evidence_sha256.get(dataset)
            or row["checkpoint_record_payload_sha256"]
            != trusted_checkpoint.checkpoint_payload_sha256.get(dataset)
            or row["checkpoint_result_payload_sha256"]
            != canonical_result_sha256(result)
        ):
            raise ValueError("Exp417 checkpoint/result payload digest mismatch")
        results.append(result)
    if tuple(observed_names) != context.datasets:
        raise ValueError("Exp417 result evidence order or coverage mismatch")
    return tuple(results)


def _parse_dataset_csv(
    payload: bytes, context: FrozenRunContext
) -> dict[str, dict[str, object]]:
    rows = _csv_rows(payload, DATASET_FILE)
    if not rows or tuple(rows[0]) != _DATASET_HEADER:
        raise ValueError("Exp417 dataset CSV header mismatch")
    parsed: dict[str, dict[str, object]] = {}
    for row in rows[1:]:
        if len(row) != len(_DATASET_HEADER):
            raise ValueError("Exp417 dataset CSV width mismatch")
        values = dict(zip(_DATASET_HEADER, row, strict=True))
        dataset = _string(values["dataset"], "dataset CSV dataset")
        if dataset in parsed:
            raise ValueError("Exp417 duplicate dataset CSV row")
        status = values["status"]
        if status not in {"qualified", "abstained"}:
            raise ValueError(
                "Exp417 official complete payload contains an invalid dataset status"
            )
        test_size = _decimal(values["test_size"], "dataset TEST size", minimum=1)
        if context.test_sizes.get(dataset) != test_size:
            raise ValueError("Exp417 dataset TEST size mismatch")
        group_count = _decimal(values["group_count"], "group count")
        observed = _decimal(values["observed_candidate_count"], "observed candidates")
        published = _decimal(values["published_candidate_count"], "published candidates")
        rate = _decimal_float(values["candidate_rate"], "dataset candidate rate")
        if rate != observed / test_size:
            raise ValueError("Exp417 dataset candidate rate mismatch")
        workload = _text_bool(values["dataset_workload_passed"], "dataset workload")
        if workload != (rate <= FROZEN_CONFIG.dataset_workload_rate):
            raise ValueError("Exp417 dataset workload flag mismatch")
        if (
            published > observed
            or (status == "qualified" and (not workload or published != observed))
            or (status == "abstained" and published != 0)
            or (not workload and status != "abstained")
        ):
            raise ValueError("Exp417 dataset abstention accounting mismatch")
        if values["terminal_stage"] or values["terminal_reason"]:
            raise ValueError("Exp417 complete dataset must have empty terminal diagnostics")
        parsed[dataset] = {
            "status": status,
            "test_size": test_size,
            "group_count": group_count,
            "observed": observed,
            "published": published,
            "rate": rate,
            "workload": workload,
        }
    if tuple(parsed) != context.datasets:
        raise ValueError("Exp417 dataset CSV order or coverage mismatch")
    return parsed


def _parse_group_csvs(
    oof_payload: bytes,
    synthetic_payload: bytes,
    datasets: Mapping[str, Mapping[str, object]],
) -> dict[tuple[str, int], GroupQualification]:
    oof_rows = _csv_rows(oof_payload, OOF_FILE)
    synthetic_rows = _csv_rows(synthetic_payload, SYNTHETIC_FILE)
    if not oof_rows or tuple(oof_rows[0]) != _OOF_HEADER:
        raise ValueError("Exp417 OOF CSV header mismatch")
    if not synthetic_rows or tuple(synthetic_rows[0]) != _SYNTHETIC_HEADER:
        raise ValueError("Exp417 synthetic CSV header mismatch")
    synthetic: dict[tuple[str, int], dict[str, str]] = {}
    previous: tuple[str, int] | None = None
    for row in synthetic_rows[1:]:
        if len(row) != len(_SYNTHETIC_HEADER):
            raise ValueError("Exp417 synthetic CSV width mismatch")
        values = dict(zip(_SYNTHETIC_HEADER, row, strict=True))
        key = (
            _string(values["dataset"], "synthetic dataset"),
            _decimal(values["length"], "synthetic length", minimum=1),
        )
        if previous is not None and key <= previous:
            raise ValueError("Exp417 synthetic CSV order or duplicate mismatch")
        previous = key
        synthetic[key] = values
    result: dict[tuple[str, int], GroupQualification] = {}
    previous = None
    group_counts: Counter[str] = Counter()
    for row in oof_rows[1:]:
        if len(row) != len(_OOF_HEADER):
            raise ValueError("Exp417 OOF CSV width mismatch")
        values = dict(zip(_OOF_HEADER, row, strict=True))
        dataset = _string(values["dataset"], "OOF dataset")
        key = (dataset, _decimal(values["length"], "OOF length", minimum=1))
        if previous is not None and key <= previous:
            raise ValueError("Exp417 OOF CSV order or duplicate mismatch")
        previous = key
        safety = synthetic.get(key)
        if safety is None:
            raise ValueError("Exp417 OOF/synthetic group coverage mismatch")
        result[key] = GroupQualification(
            length=key[1],
            series_count=_decimal(values["series_count"], "series count"),
            status=_string(values["status"], "group status"),
            oof_candidate_rate=_optional_decimal_float(
                values["oof_candidate_rate"], "OOF rate"
            ),
            wilson_upper=_optional_decimal_float(
                values["wilson_upper"], "Wilson upper"
            ),
            pooled_synthetic_sensitivity=_optional_decimal_float(
                safety["pooled_synthetic_sensitivity"], "pooled sensitivity"
            ),
            per_type_synthetic_sensitivity={
                name: _optional_decimal_float(
                    safety[name], f"{name} sensitivity"
                )
                for name in _SYNTHETIC_HEADER[3:]
            },
            seed_agreement=_optional_text_bool(
                values["seed_agreement"], "seed agreement"
            ),
            finite=_optional_text_bool(values["finite"], "finite"),
            backend_fallback=_optional_text_bool(
                values["backend_fallback"], "backend fallback"
            ),
        )
        group_counts[dataset] += 1
    if set(result) != set(synthetic):
        raise ValueError("Exp417 OOF/synthetic group coverage mismatch")
    if set(group_counts) != set(datasets) or any(
        group_counts[name] != datasets[name]["group_count"] for name in datasets
    ):
        raise ValueError("Exp417 dataset/group accounting mismatch")
    return result


def _parse_backend(
    payload: bytes,
    datasets: Mapping[str, Mapping[str, object]],
    provenance: Exp417CheckpointProvenance,
) -> dict[str, BackendAudit]:
    root = _canonical_json_object(payload, BACKEND_FILE)
    _exact_keys(
        root,
        {
            "audits", "backend_contract_sha256", "experiment_id",
            "trusted_task7_backend_audit_sha256",
        },
        BACKEND_FILE,
    )
    if (
        root["experiment_id"] != FROZEN_CONFIG.experiment_id
        or root["backend_contract_sha256"] != provenance.backend_contract_sha256
        or provenance.backend_contract_sha256 != FROZEN_TASK7_BACKEND_AUDIT_SHA256
        or root["trusted_task7_backend_audit_sha256"]
        != FROZEN_TASK7_BACKEND_AUDIT_SHA256
    ):
        raise ValueError("Exp417 backend provenance mismatch")
    audits: dict[str, BackendAudit] = {}
    for value in _list(root["audits"], "backend audits"):
        row = _object(value, "backend audit")
        _exact_keys(
            row,
            {
                "backend", "candidate_identity", "dataset", "fallback",
                "fallback_reason", "max_normalized_error", "spearman", "speedup",
            },
            "backend audit",
        )
        dataset = _string(row["dataset"], "backend dataset")
        if dataset in audits:
            raise ValueError("Exp417 duplicate backend dataset")
        audit = BackendAudit(
            backend=_string(row["backend"], "backend"),
            candidate_identity=_bool(row["candidate_identity"], "candidate identity"),
            max_normalized_error=_number(row["max_normalized_error"], "backend error"),
            spearman=_number(row["spearman"], "backend Spearman"),
            speedup=_number(row["speedup"], "backend speedup"),
            fallback=_bool(row["fallback"], "backend fallback"),
            fallback_reason=(
                None
                if row["fallback_reason"] is None
                else _string(row["fallback_reason"], "fallback reason")
            ),
        )
        if audit != _trusted_task7_execution_audit():
            raise ValueError("Exp417 backend audit differs from pinned Task 7 evidence")
        audits[dataset] = audit
    if tuple(audits) != tuple(datasets):
        raise ValueError("Exp417 backend audit order or coverage mismatch")
    return audits


def _trusted_task7_execution_audit() -> BackendAudit:
    return BackendAudit(
        backend="cpu",
        candidate_identity=True,
        max_normalized_error=4.0948700075205124e-07,
        spearman=0.9999999999845686,
        speedup=0.6709759390840965,
        fallback=False,
        fallback_reason=None,
    )


def _verify_conservation(
    run: Mapping[str, object],
    context: FrozenRunContext,
    candidates: Sequence[ShadowCandidate],
    datasets: Mapping[str, Mapping[str, object]],
    qualifications: Mapping[tuple[str, int], GroupQualification],
    audits: Mapping[str, BackendAudit],
) -> None:
    _exact_keys(
        run,
        {
            "dataset_count", "global_candidate_rate", "observed_candidate_count",
            "publication_allowed", "published_candidate_count", "status", "total_test",
        },
        "run",
    )
    if run["status"] != "complete" or run["publication_allowed"] is not True:
        raise ValueError("Exp417 official run governance mismatch")
    dataset_count = _integer(run["dataset_count"], "dataset count")
    total_test = _integer(run["total_test"], "total TEST")
    observed = _integer(run["observed_candidate_count"], "observed candidates")
    published = _integer(run["published_candidate_count"], "published candidates")
    rate = _number(run["global_candidate_rate"], "global candidate rate")
    if (
        dataset_count != FROZEN_CONFIG.expected_datasets
        or dataset_count != len(datasets)
        or total_test != sum(context.test_sizes.values())
        or total_test != sum(int(row["test_size"]) for row in datasets.values())
        or observed != sum(int(row["observed"]) for row in datasets.values())
        or published != sum(int(row["published"]) for row in datasets.values())
        or published != len(candidates)
        or rate != published / total_test
        or rate > FROZEN_CONFIG.global_workload_rate
    ):
        raise ValueError("Exp417 run conservation or workload gate mismatch")
    by_dataset = Counter(candidate.dataset for candidate in candidates)
    if any(by_dataset[name] != datasets[name]["published"] for name in datasets):
        raise ValueError("Exp417 prediction/dataset candidate accounting mismatch")
    for (dataset, _), group in qualifications.items():
        if (
            group.backend_fallback is not None
            and group.backend_fallback != audits[dataset].fallback
        ):
            raise ValueError("Exp417 group/backend fallback mismatch")


def _reconstruct_provenance(
    payload: Mapping[str, object]
) -> Exp417CheckpointProvenance:
    expected = {
        "git_commit", "database_sha256", "exp291_prediction_sha256",
        "exp291_manifest_sha256", "exp292_prediction_sha256",
        "exp292_manifest_sha256", "frozen_config_sha256",
        "preprocessing_contract_sha256", "fold_contract_sha256",
        "model_contract_sha256", "model_seeds", "perturbation_seeds",
        "runtime_versions", "backend_contract_sha256",
    }
    _exact_keys(payload, expected, "provenance")
    return Exp417CheckpointProvenance(
        git_commit=_string(payload["git_commit"], "git commit"),
        database_sha256=_string(payload["database_sha256"], "database digest"),
        exp291_prediction_sha256=_string(payload["exp291_prediction_sha256"], "Exp291 prediction digest"),
        exp291_manifest_sha256=_string(payload["exp291_manifest_sha256"], "Exp291 manifest digest"),
        exp292_prediction_sha256=_string(payload["exp292_prediction_sha256"], "Exp292 prediction digest"),
        exp292_manifest_sha256=_string(payload["exp292_manifest_sha256"], "Exp292 manifest digest"),
        frozen_config_sha256=_string(payload["frozen_config_sha256"], "config digest"),
        preprocessing_contract_sha256=_string(payload["preprocessing_contract_sha256"], "preprocessing digest"),
        fold_contract_sha256=_string(payload["fold_contract_sha256"], "fold digest"),
        model_contract_sha256=_string(payload["model_contract_sha256"], "model digest"),
        model_seeds=tuple(_integer(value, "model seed", minimum=1) for value in _list(payload["model_seeds"], "model seeds")),
        perturbation_seeds=tuple(_integer(value, "perturbation seed", minimum=1) for value in _list(payload["perturbation_seeds"], "perturbation seeds")),
        runtime_versions=_object(payload["runtime_versions"], "runtime versions"),
        backend_contract_sha256=_string(payload["backend_contract_sha256"], "backend digest"),
    )


def _reconstruct_context(
    payload: Mapping[str, object], config: Exp417FrozenConfig
) -> FrozenRunContext:
    _exact_keys(
        payload,
        {
            "code_commit", "database_sha256", "datasets", "lanes",
            "runtime_versions", "test_identity_manifest",
        },
        "context",
    )
    lanes_payload = _object(payload["lanes"], "lanes")
    _exact_keys(
        lanes_payload, {"hard", "standard", "priority", "automatic_normal"}, "lanes"
    )
    lanes = FrozenLaneSnapshot(
        **{
            name: {
                dataset: tuple(_integer(index, "lane index") for index in _list(indices, "lane indices"))
                for dataset, indices in _object(lanes_payload[name], name).items()
            }
            for name in ("hard", "standard", "priority", "automatic_normal")
        }
    )
    manifest = {
        dataset: tuple(
            (
                _integer(_list(entry, "TEST identity")[0], "TEST index"),
                _integer(_list(entry, "TEST identity")[1], "TEST length", minimum=1),
            )
            for entry in _list(entries, "TEST identities")
        )
        for dataset, entries in _object(
            payload["test_identity_manifest"], "TEST identity manifest"
        ).items()
    }
    return FrozenRunContext(
        database_sha256=_string(payload["database_sha256"], "context database"),
        datasets=tuple(
            _string(value, "context dataset")
            for value in _list(payload["datasets"], "context datasets")
        ),
        lanes=lanes,
        test_identity_manifest=manifest,
        code_commit=_string(payload["code_commit"], "context commit"),
        runtime_versions=_object(payload["runtime_versions"], "context runtime versions"),
        config=config,
    )


def _verify_provenance_context(
    context: FrozenRunContext, provenance: Mapping[str, object]
) -> None:
    lanes = context.lanes
    expected = {
        "git_commit": context.code_commit,
        "database_sha256": context.database_sha256,
        "exp291_prediction_sha256": lanes.exp291_prediction_sha256,
        "exp291_manifest_sha256": lanes.exp291_manifest_sha256,
        "exp292_prediction_sha256": lanes.exp292_prediction_sha256,
        "exp292_manifest_sha256": lanes.exp292_manifest_sha256,
        "runtime_versions": dict(context.runtime_versions),
    }
    if any(provenance.get(name) != value for name, value in expected.items()):
        raise ValueError("Exp417 checkpoint provenance does not match frozen context")


def _verify_trusted_checkpoint(
    trusted: TrustedCheckpointProvenance,
    provenance: Exp417CheckpointProvenance,
    results: Sequence[DatasetResult],
) -> None:
    if not isinstance(trusted, TrustedCheckpointProvenance):
        raise ValueError("Exp417 trusted checkpoint provenance is required")
    expected_provenance = sha256(
        canonical_json_bytes(provenance.to_payload())
    ).hexdigest()
    if trusted.checkpoint_provenance_sha256 != expected_provenance:
        raise ValueError("Exp417 checkpoint provenance root mismatch")
    expected_results = {
        result.dataset: canonical_result_sha256(result) for result in results
    }
    if (
        tuple(trusted.result_evidence_sha256) != tuple(expected_results)
        or dict(trusted.result_evidence_sha256) != expected_results
    ):
        raise ValueError("Exp417 externally trusted result evidence mismatch")
    if tuple(trusted.checkpoint_payload_sha256) != tuple(expected_results):
        raise ValueError("Exp417 externally trusted checkpoint payload map mismatch")


def _trusted_digest_mapping(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Exp417 trusted {name} evidence must be a mapping")
    result: dict[str, str] = {}
    for dataset, digest in value.items():
        key = _string(dataset, "trusted checkpoint dataset")
        if key in result:
            raise ValueError("Exp417 trusted checkpoint dataset is duplicated")
        result[key] = _digest(digest, f"trusted {name}")
    if not result or tuple(result) != tuple(sorted(result)):
        raise ValueError("Exp417 trusted checkpoint map must be sorted and non-empty")
    return result


def _canonical_json_object(payload: bytes, name: str) -> dict[str, object]:
    value = _json_loads_unique(payload, name)
    result = _object(value, name)
    if payload != canonical_json_bytes(result) + b"\n":
        raise ValueError(f"Exp417 {name} is not canonical JSON")
    return result


def _json_line_object(payload: bytes, name: str) -> dict[str, object]:
    value = _json_loads_unique(payload, name)
    result = _object(value, name)
    if payload != canonical_json_bytes(result):
        raise ValueError(f"Exp417 {name} is not canonical JSON")
    return result


def _json_loads_unique(payload: bytes, name: str) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"Exp417 {name} contains a duplicate JSON key")
            result[key] = value
        return result

    try:
        return json.loads(payload, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Exp417 {name} is invalid JSON") from error


def _csv_rows(payload: bytes, name: str) -> list[list[str]]:
    try:
        text = payload.decode("utf-8")
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError(f"Exp417 {name} is invalid CSV") from error
    if payload != _csv_bytes(rows):
        raise ValueError(f"Exp417 {name} is not canonical CSV")
    return rows


def _csv_bytes(rows: Sequence[Sequence[object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _float(value: float) -> str:
    if not isfinite(value):
        raise ValueError("Exp417 rendered number must be finite")
    return repr(float(value))


def _optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return _float(value)


def _bool_text(value: bool) -> str:
    if type(value) is not bool:
        raise ValueError("Exp417 rendered flag must be bool")
    return "true" if value else "false"


def _optional_bool_text(value: bool | None) -> str:
    if value is None:
        return ""
    return _bool_text(value)


def _text_bool(value: object, name: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Exp417 {name} must be true or false")


def _optional_text_bool(value: object, name: str) -> bool | None:
    if value == "":
        return None
    return _text_bool(value, name)


def _decimal(value: object, name: str, minimum: int = 0) -> int:
    if not isinstance(value, str) or not value.isascii() or not value.isdecimal():
        raise ValueError(f"Exp417 {name} must be a canonical integer")
    number = int(value)
    if value != str(number) or number < minimum:
        raise ValueError(f"Exp417 {name} must be a canonical integer")
    return number


def _decimal_float(value: object, name: str) -> float:
    if not isinstance(value, str):
        raise ValueError(f"Exp417 {name} must be a canonical float")
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(f"Exp417 {name} must be a canonical float") from error
    if not isfinite(number) or value != repr(number):
        raise ValueError(f"Exp417 {name} must be a canonical float")
    return number


def _optional_decimal_float(value: object, name: str) -> float | None:
    if value == "":
        return None
    return _decimal_float(value, name)


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"Exp417 {name} must be an object")
    return value


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"Exp417 {name} must be a list")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Exp417 {name} must be a non-empty string")
    return value


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"Exp417 {name} must be an integer >= {minimum}")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"Exp417 {name} must be finite")
    return float(value)


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"Exp417 {name} must be a lowercase SHA-256 digest")
    return value


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"Exp417 {name} must be bool")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected or len(value) != len(expected):
        raise ValueError(f"Exp417 {name} schema mismatch")


def _reject_forbidden_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key.lower() in _FORBIDDEN_KEYS:
                raise ValueError("Exp417 rendered evaluation content is forbidden")
            _reject_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_keys(child)


__all__ = [
    "BACKEND_FILE",
    "CONFIG_FILE",
    "DATASET_FILE",
    "MANIFEST_FILE",
    "NON_MANIFEST_FILES",
    "OFFICIAL_FILES",
    "OOF_FILE",
    "PREDICTION_FILE",
    "REPORT_FILE",
    "SYNTHETIC_FILE",
    "TrustedCheckpointProvenance",
    "render_exp417_payloads",
    "verify_exp417_payloads",
]
