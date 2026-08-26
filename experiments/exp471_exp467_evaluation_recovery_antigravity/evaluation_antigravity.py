"""Label-gated, create-only persistence for the frozen Exp471 evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
from types import MappingProxyType

import numpy as np

from experiments.exp450_resampling_image_ablation_antigravity import (
    evaluation_antigravity as frozen_evaluation_antigravity,
)
from experiments.exp450_resampling_image_ablation_antigravity.evaluation_antigravity import (
    Exp450EvaluationAntigravity,
    MetricTablesExp450Antigravity,
    curve_tables_exp450_antigravity,
    metric_tables_exp450_antigravity,
)

from . import cache_artifacts_antigravity
from . import score_cache_antigravity
from . import source_authority_antigravity
from .config_antigravity import (
    EXP471_EVALUATION_ROOT_ANTIGRAVITY,
    EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY,
    METHOD_ORDER_ANTIGRAVITY,
)
from .contracts_antigravity import (
    Exp471EvaluationArtifactAntigravity,
    Exp471ScoreCacheManifestAntigravity,
    Exp471SourceAuthorityAntigravity,
    canonical_json_bytes_exp471_antigravity,
)
from .source_authority_antigravity import Exp471SourceSessionAntigravity
from experiments.paper_public_baselines.calibration import matched_top_k
from experiments.paper_public_baselines.config import ALPHAS


_EVALUATION_NAME_ANTIGRAVITY = "evaluation_antigravity.json"
_EVALUATION_SCHEMA_ANTIGRAVITY = "exp471-evaluation-antigravity.v2"
_SEMANTIC_NONE_SCHEMA_ANTIGRAVITY = "exp471-semantic-none-antigravity.v1"
_SHA256_ANTIGRAVITY = re.compile(r"^[0-9a-f]{64}$")
_CODE_CLOSURE_SCHEMA_ANTIGRAVITY = (
    "exp471-evaluation-code-closure-antigravity.v1"
)
_CODE_CLOSURE_PATHS_ANTIGRAVITY = (
    "experiments/exp450_resampling_image_ablation_antigravity/"
    "evaluation_antigravity.py",
    "experiments/exp471_exp467_evaluation_recovery_antigravity/"
    "cache_artifacts_antigravity.py",
    "experiments/exp471_exp467_evaluation_recovery_antigravity/"
    "config_antigravity.py",
    "experiments/exp471_exp467_evaluation_recovery_antigravity/"
    "contracts_antigravity.py",
    "experiments/exp471_exp467_evaluation_recovery_antigravity/"
    "evaluation_antigravity.py",
    "experiments/exp471_exp467_evaluation_recovery_antigravity/"
    "score_cache_antigravity.py",
    "experiments/exp471_exp467_evaluation_recovery_antigravity/"
    "source_authority_antigravity.py",
)


def artifact_view_exp471_antigravity(value: object) -> object:
    """Return the complete evaluation's strict-JSON publication view."""

    if isinstance(value, Mapping):
        return {
            str(key): artifact_view_exp471_antigravity(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [artifact_view_exp471_antigravity(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _repository_root_antigravity() -> Path:
    module_path = Path(__file__)
    if not module_path.is_absolute():
        module_path = Path.cwd() / module_path
    return module_path.parents[2]


def _closure_member_bytes_antigravity(root: Path, relative: str) -> bytes:
    parts = relative.split("/")
    if (
        not parts
        or any(part in {"", ".", ".."} for part in parts)
        or not relative.endswith("_antigravity.py")
    ):
        raise ValueError("Exp471 evaluation closure path is not contained")
    anchor = cache_artifacts_antigravity._open_absolute_directory_antigravity(
        Path(root), create=False
    )
    directories: list[tuple[int, str, os.stat_result, int]] = []
    try:
        parent = anchor.descriptor
        for part in parts[:-1]:
            descriptor, expected = (
                cache_artifacts_antigravity._open_child_directory_antigravity(
                    parent, part, create=False
                )
            )
            directories.append((parent, part, expected, descriptor))
            parent = descriptor
        raw = cache_artifacts_antigravity._read_leaf_at_antigravity(
            parent, parts[-1]
        )
        if raw is None:
            raise ValueError("Exp471 evaluation closure member is missing")
        for parent, name, expected, _descriptor in reversed(directories):
            cache_artifacts_antigravity._validate_child_directory_antigravity(
                parent, name, expected
            )
        anchor.validate()
        return raw
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Exp471 evaluation closure member is unavailable: {relative}"
        ) from error
    finally:
        for _parent, _name, _expected, descriptor in reversed(directories):
            os.close(descriptor)
        anchor.close()


def _code_closure_record_antigravity(
    repository: Path | None = None,
) -> dict[str, object]:
    root = _repository_root_antigravity() if repository is None else Path(repository)
    file_sha256 = {
        relative: sha256(
            _closure_member_bytes_antigravity(root, relative)
        ).hexdigest()
        for relative in _CODE_CLOSURE_PATHS_ANTIGRAVITY
    }
    bare = {
        "schema": _CODE_CLOSURE_SCHEMA_ANTIGRAVITY,
        "file_sha256": file_sha256,
    }
    return {
        **bare,
        "closure_sha256": sha256(
            canonical_json_bytes_exp471_antigravity(bare)
        ).hexdigest(),
    }


def _code_closure_sha256_antigravity() -> str:
    record = _code_closure_record_antigravity()
    digest = record.get("closure_sha256")
    if type(digest) is not str or _SHA256_ANTIGRAVITY.fullmatch(digest) is None:
        raise ValueError("Exp471 evaluation closure digest differs")
    return digest


def _dependency_version_antigravity(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _implementation_record_antigravity() -> dict[str, object]:
    code_closure = _code_closure_record_antigravity()
    return {
        "module": (
            "experiments.exp450_resampling_image_ablation_antigravity."
            "evaluation_antigravity"
        ),
        "metric_function": "metric_tables_exp450_antigravity",
        "curve_function": "curve_tables_exp450_antigravity",
        "code_closure_sha256": code_closure["closure_sha256"],
        "code_closure": code_closure,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "dependencies": {
            "numpy": _dependency_version_antigravity("numpy"),
            "scikit-learn": _dependency_version_antigravity("scikit-learn"),
            "scipy": _dependency_version_antigravity("scipy"),
        },
    }


def _require_source_session_antigravity(
    session: object,
) -> Exp471SourceSessionAntigravity:
    if (
        not isinstance(session, Exp471SourceSessionAntigravity)
        or not source_authority_antigravity._source_session_is_issued_antigravity(
            session
        )
    ):
        raise ValueError("Exp471 evaluation requires an issued source session")
    if not isinstance(session.authority, Exp471SourceAuthorityAntigravity):
        raise ValueError("Exp471 evaluation source authority has the wrong type")
    source_authority_antigravity._canonical_authority_bytes_antigravity(
        session.authority
    )
    return session


@dataclass(frozen=True, slots=True)
class _TypedCacheAntigravity:
    """Fully rebound typed cache evidence, before the label capability opens."""

    cache: Exp471ScoreCacheManifestAntigravity
    raw_scores: Mapping[str, Mapping[str, tuple[float, ...]]]
    micro_scores: Mapping[str, Mapping[str, tuple[float, ...]]]
    test_counts: Mapping[str, int]
    test_indices: Mapping[str, tuple[int, ...]]
    status: Mapping[str, Mapping[str, str]]
    reasons: Mapping[str, Mapping[str, str | None]]


def _verified_cache_shards_antigravity(
    session: Exp471SourceSessionAntigravity,
    cache: object,
    *,
    prepare_scores: bool,
) -> _TypedCacheAntigravity:
    session = _require_source_session_antigravity(session)
    if not isinstance(cache, Exp471ScoreCacheManifestAntigravity):
        raise ValueError("Exp471 evaluation cache authority has the wrong type")

    authorized = score_cache_antigravity.open_complete_score_cache_antigravity(
        session
    )
    if authorized != cache:
        raise ValueError("Exp471 evaluation caller cache differs from score authority")

    root = Path(EXP471_SCORE_CACHE_ROOT_ANTIGRAVITY)
    reopened = cache_artifacts_antigravity.read_cache_manifest_antigravity(root)
    if reopened != cache:
        raise ValueError("Exp471 evaluation caller cache differs from durable cache")
    if reopened.source_authority_sha256 != session.authority.authority_sha256:
        raise ValueError("Exp471 evaluation cache source authority differs")
    if reopened.method_order != METHOD_ORDER_ANTIGRAVITY:
        raise ValueError("Exp471 evaluation cache method order differs")
    if (
        reopened.total_shards
        != len(METHOD_ORDER_ANTIGRAVITY) * len(reopened.dataset_order)
    ):
        raise ValueError("Exp471 evaluation cache identity count differs")
    if reopened.failed_shards != 0:
        raise ValueError("Exp471 evaluation cache contains failed shards")

    is_production_population = len(reopened.dataset_order) == 1_117
    if is_production_population and (
        reopened.total_shards != 11_170
        or reopened.scored_shards != 10_775
        or reopened.abstained_shards != 395
    ):
        raise ValueError("Exp471 evaluation production cache partition differs")

    raw_scores: dict[str, dict[str, tuple[float, ...]]] = {}
    micro_scores: dict[str, dict[str, tuple[float, ...]]] = {}
    test_counts: dict[str, int] = {}
    test_indices: dict[str, tuple[int, ...]] = {}
    statuses: dict[str, dict[str, str]] = {}
    reasons: dict[str, dict[str, str | None]] = {}
    scored = 0
    abstained = 0
    for method in METHOD_ORDER_ANTIGRAVITY:
        raw_scores[method] = {}
        micro_scores[method] = {}
        statuses[method] = {}
        reasons[method] = {}
        for dataset in reopened.dataset_order:
            shard = cache_artifacts_antigravity.read_score_shard_antigravity(
                root, method, dataset
            )
            key = f"{method}:{dataset}"
            if (
                shard.source_authority_sha256
                != session.authority.authority_sha256
                or shard.cache_context_sha256 != reopened.cache_context_sha256
                or key not in reopened.shard_sha256
                or shard.envelope_sha256 != reopened.shard_sha256[key]
                or shard.method != method
                or shard.dataset != dataset
            ):
                raise ValueError("Exp471 evaluation score shard authority differs")
            prior_count = test_counts.setdefault(dataset, shard.test_count)
            if prior_count != shard.test_count:
                raise ValueError("Exp471 evaluation dataset row counts differ")
            prior_indices = test_indices.setdefault(dataset, shard.test_indices)
            if prior_indices != shard.test_indices:
                raise ValueError("Exp471 evaluation TEST indices differ by method")
            if shard.status == "scored":
                if shard.abstention_reason is not None:
                    raise ValueError("Exp471 evaluation scored shard reason differs")
                scored += 1
                if prepare_scores:
                    raw_scores[method][dataset] = shard.raw_scores
                    micro_scores[method][dataset] = shard.micro_scores
            elif shard.status == "abstained":
                if (
                    shard.abstention_reason != "insufficient_train_rows"
                    or shard.raw_scores
                    or shard.micro_scores
                ):
                    raise ValueError("Exp471 evaluation abstention shard differs")
                abstained += 1
            else:
                raise ValueError("Exp471 evaluation score shard status differs")
            statuses[method][dataset] = shard.status
            reasons[method][dataset] = shard.abstention_reason
    if (
        scored != reopened.scored_shards
        or abstained != reopened.abstained_shards
        or scored + abstained != reopened.total_shards
    ):
        raise ValueError("Exp471 evaluation cache status partition differs")
    return _TypedCacheAntigravity(
        reopened,
        raw_scores if prepare_scores else {},
        micro_scores if prepare_scores else {},
        MappingProxyType(test_counts),
        MappingProxyType(test_indices),
        MappingProxyType({method: MappingProxyType(statuses[method]) for method in METHOD_ORDER_ANTIGRAVITY}),
        MappingProxyType({method: MappingProxyType(reasons[method]) for method in METHOD_ORDER_ANTIGRAVITY}),
    )


def _verified_cache_scores_antigravity(
    session: Exp471SourceSessionAntigravity,
    cache: object,
) -> _TypedCacheAntigravity:
    return _verified_cache_shards_antigravity(session, cache, prepare_scores=True)


def _verified_cache_counts_antigravity(
    session: Exp471SourceSessionAntigravity,
    cache: object,
) -> _TypedCacheAntigravity:
    return _verified_cache_shards_antigravity(session, cache, prepare_scores=False)


def _normalize_labels_antigravity(
    value: object,
    datasets: tuple[str, ...],
    test_counts: Mapping[str, int],
    test_indices: Mapping[str, tuple[int, ...]] | None = None,
) -> dict[str, tuple[int, ...]]:
    if not isinstance(value, Mapping) or tuple(value) != datasets:
        raise ValueError("Exp471 evaluation label dataset order differs")
    normalized: dict[str, tuple[int, ...]] = {}
    for dataset in datasets:
        rows = tuple(value[dataset])
        if rows and all(isinstance(row, tuple) and len(row) == 2 for row in rows):
            indices, labels = zip(*rows, strict=True)
            expected = (
                tuple(range(test_counts[dataset]))
                if test_indices is None
                else test_indices[dataset]
            )
            if (
                len(indices) != len(expected)
                or any(type(index) is not int or index < 0 for index in indices)
                or len(set(indices)) != len(indices)
                or indices != expected
            ):
                raise ValueError("Exp471 evaluation TEST label identity differs")
        else:
            labels = rows
        if (
            len(labels) != test_counts[dataset]
            or any(type(label) is not int or label not in (0, 1) for label in labels)
        ):
            raise ValueError("Exp471 evaluation TEST labels differ from score rows")
        normalized[dataset] = tuple(labels)
    return normalized


def _load_labels_once_antigravity(
    session: Exp471SourceSessionAntigravity,
    datasets: tuple[str, ...],
    test_counts: Mapping[str, int],
    test_indices: Mapping[str, tuple[int, ...]],
    label_loader: Callable | None,
) -> dict[str, tuple[int, ...]]:
    if label_loader is not None:
        if not callable(label_loader):
            raise ValueError("Exp471 evaluation label loader must be callable")
        raw = label_loader(datasets)
    elif callable(session.label_callback_antigravity):
        raw = session.label_callback_antigravity()
    else:
        from experiments.paper_public_baselines.label_access import load_test_labels

        raw = load_test_labels(datasets)
    return _normalize_labels_antigravity(raw, datasets, test_counts, test_indices)


def _encode_semantic_none_antigravity(value: object) -> object:
    if value is None:
        return {"schema": _SEMANTIC_NONE_SCHEMA_ANTIGRAVITY}
    if isinstance(value, Mapping):
        return {
            str(key): _encode_semantic_none_antigravity(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_encode_semantic_none_antigravity(item) for item in value]
    return value


def _decode_semantic_none_antigravity(value: object) -> object:
    if isinstance(value, Mapping):
        if dict(value) == {"schema": _SEMANTIC_NONE_SCHEMA_ANTIGRAVITY}:
            return None
        return {
            str(key): _decode_semantic_none_antigravity(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_decode_semantic_none_antigravity(item) for item in value]
    return value


def _metric_record_antigravity(
    metrics: MetricTablesExp450Antigravity,
) -> dict[str, object]:
    if not isinstance(metrics, MetricTablesExp450Antigravity):
        raise ValueError("Exp471 frozen metric tables have the wrong type")
    return {
        "method_order": list(metrics.method_order),
        "per_dataset": list(metrics.per_dataset),
        "macro": metrics.macro,
        "micro": metrics.micro,
        "paired_statistics": list(metrics.paired_statistics),
    }


def _source_record_antigravity(
    authority: Exp471SourceAuthorityAntigravity,
) -> dict[str, object]:
    return {
        "authority_sha256": authority.authority_sha256,
        "source_commit": authority.source_commit,
        "source_context_sha256": authority.source_context_sha256,
        "inventory_sha256": authority.inventory_sha256,
    }


def _cache_record_antigravity(
    cache: Exp471ScoreCacheManifestAntigravity,
    test_counts: Mapping[str, int],
) -> dict[str, object]:
    if tuple(test_counts) != cache.dataset_order or any(
        type(test_counts[dataset]) is not int or test_counts[dataset] < 1
        for dataset in cache.dataset_order
    ):
        raise ValueError("Exp471 evaluation cache row-count authority differs")
    return {
        "cache_context_sha256": cache.cache_context_sha256,
        "manifest_sha256": cache.manifest_sha256,
        "total_shards": cache.total_shards,
        "scored_shards": cache.scored_shards,
        "abstained_shards": cache.abstained_shards,
        "failed_shards": cache.failed_shards,
        "dataset_test_count": dict(test_counts),
    }


def _coverage_rows_antigravity(
    typed: _TypedCacheAntigravity,
) -> list[dict[str, object]]:
    population = len(typed.cache.dataset_order)
    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER_ANTIGRAVITY:
        statuses = typed.status[method]
        scored = sum(status == "scored" for status in statuses.values())
        abstained = sum(status == "abstained" for status in statuses.values())
        reasons: dict[str, int] = {}
        for dataset in typed.cache.dataset_order:
            reason = typed.reasons[method][dataset]
            if reason is not None:
                reasons[reason] = reasons.get(reason, 0) + 1
        rows.append(
            {
                "method": method,
                "population": population,
                "scored": scored,
                "abstained": abstained,
                "failed": 0,
                "coverage": scored / population,
                "abstention_reasons": reasons,
            }
        )
    return rows


def _all_ten_common_support_antigravity(
    typed: _TypedCacheAntigravity,
) -> tuple[str, ...]:
    datasets = tuple(
        dataset
        for dataset in typed.cache.dataset_order
        if all(
            typed.status[method][dataset] == "scored"
            for method in METHOD_ORDER_ANTIGRAVITY
        )
    )
    if len(typed.cache.dataset_order) == 1_117 and len(datasets) != 922:
        raise ValueError("Exp471 evaluation all-ten production support differs")
    return datasets


def _support_authority_antigravity(
    datasets: tuple[str, ...],
) -> dict[str, object]:
    return {
        "all_ten_common_support_count": len(datasets),
        "all_ten_common_support_datasets": list(datasets),
        "all_ten_common_support_sha256": sha256(
            canonical_json_bytes_exp471_antigravity(list(datasets))
        ).hexdigest(),
    }


def _decision_alerts_antigravity(
    *,
    method: str,
    protocol: str,
    alpha: float | None,
    raw_scores: Sequence[float] | None,
    micro_scores: Sequence[float] | None,
    test_indices: tuple[int, ...],
    candidate_c_alert_count: int,
) -> tuple[int, ...]:
    """Return an immutable frozen-policy alert set; abstention is no-alert."""

    if raw_scores is None or micro_scores is None:
        return ()
    raw = tuple(float(value) for value in raw_scores)
    micro = tuple(float(value) for value in micro_scores)
    if len(raw) != len(micro) or len(raw) != len(test_indices):
        raise ValueError("Exp471 evaluation decision score identity differs")
    if protocol == "proposal_threshold":
        if method not in METHOD_ORDER_ANTIGRAVITY[:4] or alpha is not None:
            raise ValueError("Exp471 evaluation proposal decision protocol differs")
        selected = [index for index, score in zip(test_indices, raw, strict=True) if score >= 0.5]
    elif protocol == "conformal":
        if method in METHOD_ORDER_ANTIGRAVITY[:4] or alpha not in ALPHAS:
            raise ValueError("Exp471 evaluation conformal decision protocol differs")
        selected = [
            index
            for index, score in zip(test_indices, micro, strict=True)
            if score >= 1.0 - float(alpha)
        ]
    elif protocol == "matched_budget":
        if method in METHOD_ORDER_ANTIGRAVITY[:4] or alpha is not None:
            raise ValueError("Exp471 evaluation matched-budget protocol differs")
        selected = matched_top_k(
            np.asarray(raw, dtype=np.float64), test_indices, candidate_c_alert_count
        )
    else:
        raise ValueError("Exp471 evaluation decision protocol differs")
    return tuple(sorted(int(index) for index in selected))


def _confusion_row_antigravity(
    labels: tuple[int, ...], test_indices: tuple[int, ...], alerts: tuple[int, ...]
) -> dict[str, int]:
    by_index = dict(zip(test_indices, labels, strict=True))
    if len(by_index) != len(test_indices) or any(index not in by_index for index in alerts):
        raise ValueError("Exp471 evaluation decision TEST identity differs")
    alert_set = frozenset(alerts)
    tp = sum(label == 1 and index in alert_set for index, label in by_index.items())
    fp = sum(label == 0 and index in alert_set for index, label in by_index.items())
    fn = sum(label == 1 and index not in alert_set for index, label in by_index.items())
    tn = sum(label == 0 and index not in alert_set for index, label in by_index.items())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _derived_decision_metrics_antigravity(counts: Mapping[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = (int(counts[name]) for name in ("tp", "fp", "fn", "tn"))
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def _decision_tables_antigravity(
    typed: _TypedCacheAntigravity,
    labels: Mapping[str, tuple[int, ...]],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for method in METHOD_ORDER_ANTIGRAVITY:
        protocols = (
            (("proposal_threshold", None),)
            if method in METHOD_ORDER_ANTIGRAVITY[:4]
            else tuple(("conformal", alpha) for alpha in ALPHAS)
            + (("matched_budget", None),)
        )
        for protocol, alpha in protocols:
            method_rows: list[dict[str, object]] = []
            for dataset in typed.cache.dataset_order:
                status = typed.status[method][dataset]
                raw = typed.raw_scores[method].get(dataset)
                micro = typed.micro_scores[method].get(dataset)
                candidate_scores = typed.raw_scores["C"].get(dataset)
                candidate_alert_count = (
                    sum(score >= 0.5 for score in candidate_scores)
                    if candidate_scores is not None
                    else 0
                )
                alerts = _decision_alerts_antigravity(
                    method=method,
                    protocol=protocol,
                    alpha=alpha,
                    raw_scores=raw,
                    micro_scores=micro,
                    test_indices=typed.test_indices[dataset],
                    candidate_c_alert_count=candidate_alert_count,
                )
                counts = _confusion_row_antigravity(
                    labels[dataset], typed.test_indices[dataset], alerts
                )
                row = {
                    "method": method,
                    "dataset": dataset,
                    "protocol": protocol,
                    "alpha": alpha,
                    "status": status,
                    "abstention_reason": typed.reasons[method][dataset],
                    "alerts": list(alerts),
                    "alert_count": len(alerts),
                    **counts,
                    **_derived_decision_metrics_antigravity(counts),
                }
                rows.append(row)
                method_rows.append(row)
            counts = {
                name: sum(int(row[name]) for row in method_rows)
                for name in ("tp", "fp", "fn", "tn")
            }
            summaries.append(
                {
                    "method": method,
                    "protocol": protocol,
                    "alpha": alpha,
                    "support_count": len(method_rows),
                    "alert_count": sum(int(row["alert_count"]) for row in method_rows),
                    "coverage": sum(row["status"] == "scored" for row in method_rows)
                    / len(method_rows),
                    **counts,
                    **_derived_decision_metrics_antigravity(counts),
                }
            )
    return {"full_population": {"dataset_rows": rows, "summary_rows": summaries}}


def _continuous_table_record_antigravity(
    metrics: MetricTablesExp450Antigravity,
    typed: _TypedCacheAntigravity,
) -> dict[str, object]:
    """Keep Exp450 values byte-for-byte while naming typed score absence."""

    record = _metric_record_antigravity(metrics)
    rows: list[dict[str, object]] = []
    semantic_nulls: list[dict[str, str]] = []
    for original in record["per_dataset"]:
        row = dict(original)
        method = row["method"]
        dataset = row["dataset"]
        assert isinstance(method, str) and isinstance(dataset, str)
        reason = typed.reasons[method][dataset]
        if reason is not None:
            row["average_precision"] = None
            row["average_precision_exclusion"] = reason
            row["auroc"] = None
            row["auroc_exclusion"] = reason
            semantic_nulls.extend(
                {
                    "method": method,
                    "dataset": dataset,
                    "metric": metric,
                    "reason": reason,
                }
                for metric in ("average_precision", "auroc", "roc", "precision_recall")
            )
        rows.append(row)
    record["per_dataset"] = rows
    paired: list[dict[str, object]] = []
    for original in record["paired_statistics"]:
        row = dict(original)
        comparison = row["comparison"]
        assert isinstance(comparison, str)
        variant, reference = comparison.split("-", maxsplit=1)
        if reference != "A" or variant not in {"B", "C", "D"}:
            raise ValueError("Exp471 evaluation frozen pair identity differs")
        table_datasets = tuple(
            str(candidate["dataset"])
            for candidate in rows
            if candidate["method"] == "A"
        )
        pair_support = tuple(
            dataset
            for dataset in table_datasets
            if typed.status[reference][dataset] == "scored"
            and typed.status[variant][dataset] == "scored"
        )
        row["scored_intersection_datasets"] = list(pair_support)
        row["scored_intersection_count"] = len(pair_support)
        row["scored_intersection_sha256"] = sha256(
            canonical_json_bytes_exp471_antigravity(list(pair_support))
        ).hexdigest()
        paired.append(row)
    record["paired_statistics"] = paired
    record["semantic_nulls"] = semantic_nulls
    return record


def _curve_support_record_antigravity(
    curves: Mapping[str, Mapping[str, object]],
    typed: _TypedCacheAntigravity,
    datasets: tuple[str, ...],
) -> dict[str, object]:
    semantic_nulls = [
        {
            "method": method,
            "dataset": dataset,
            "metric": metric,
            "reason": typed.reasons[method][dataset],
        }
        for method in METHOD_ORDER_ANTIGRAVITY
        for dataset in datasets
        if typed.status[method][dataset] == "abstained"
        for metric in ("roc", "precision_recall")
    ]
    return {"curves": curves, "semantic_nulls": semantic_nulls}


def _payload_antigravity(
    authority: Exp471SourceAuthorityAntigravity,
    typed: _TypedCacheAntigravity,
    labels: Mapping[str, tuple[int, ...]],
    scored_metrics: MetricTablesExp450Antigravity,
    scored_curves: Mapping[str, Mapping[str, object]],
    common_metrics: MetricTablesExp450Antigravity,
    common_curves: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    common = _all_ten_common_support_antigravity(typed)
    return {
        "schema": _EVALUATION_SCHEMA_ANTIGRAVITY,
        "source_authority": _source_record_antigravity(authority),
        "score_cache": _cache_record_antigravity(typed.cache, typed.test_counts),
        "implementation": _implementation_record_antigravity(),
        "coverage": {"rows": _coverage_rows_antigravity(typed)},
        "decision_tables": _decision_tables_antigravity(typed, labels),
        "continuous_tables": {
            "scored_support": _continuous_table_record_antigravity(scored_metrics, typed),
            "all_ten_common_support": _continuous_table_record_antigravity(common_metrics, typed),
        },
        "curve_tables": {
            "scored_support": _curve_support_record_antigravity(
                scored_curves, typed, typed.cache.dataset_order
            ),
            "all_ten_common_support": _curve_support_record_antigravity(
                common_curves, typed, common
            ),
        },
        "support_authority": _support_authority_antigravity(common),
    }


def _expected_evaluation_payload_antigravity(
    authority: Exp471SourceAuthorityAntigravity,
    typed: _TypedCacheAntigravity,
    labels: Mapping[str, tuple[int, ...]],
) -> Mapping[str, object]:
    """Build the one canonical v2 payload from verified scores and labels only."""

    scored_metrics = metric_tables_exp450_antigravity(
        labels, typed.raw_scores, micro_scores=typed.micro_scores
    )
    scored_curves = curve_tables_exp450_antigravity(labels, typed.micro_scores)
    common = _all_ten_common_support_antigravity(typed)
    common_labels = {dataset: labels[dataset] for dataset in common}
    common_raw = {
        method: {
            dataset: typed.raw_scores[method][dataset]
            for dataset in common
        }
        for method in METHOD_ORDER_ANTIGRAVITY
    }
    common_micro = {
        method: {
            dataset: typed.micro_scores[method][dataset]
            for dataset in common
        }
        for method in METHOD_ORDER_ANTIGRAVITY
    }
    common_metrics = metric_tables_exp450_antigravity(
        common_labels, common_raw, micro_scores=common_micro
    )
    common_curves = curve_tables_exp450_antigravity(common_labels, common_micro)
    return _payload_antigravity(
        authority,
        typed,
        labels,
        scored_metrics,
        scored_curves,
        common_metrics,
        common_curves,
    )


def _artifact_record_antigravity(
    artifact: Exp471EvaluationArtifactAntigravity,
    *, include_digest: bool,
) -> dict[str, object]:
    record = {
        "source_authority_sha256": artifact.source_authority_sha256,
        "cache_manifest_sha256": artifact.cache_manifest_sha256,
        "code_closure_sha256": artifact.code_closure_sha256,
        "method_order": list(artifact.method_order),
        "dataset_order": list(artifact.dataset_order),
        "payload": artifact.payload,
    }
    if include_digest:
        record["evaluation_sha256"] = artifact.evaluation_sha256
    return record


def _expected_artifact_antigravity(
    authority: Exp471SourceAuthorityAntigravity,
    typed: _TypedCacheAntigravity,
    labels: Mapping[str, tuple[int, ...]],
) -> Exp471EvaluationArtifactAntigravity:
    """Materialize, but never publish, the deterministic cache-and-label artifact."""

    payload = _expected_evaluation_payload_antigravity(authority, typed, labels)
    raw_bare = {
        "source_authority_sha256": authority.authority_sha256,
        "cache_manifest_sha256": typed.cache.manifest_sha256,
        "code_closure_sha256": _code_closure_sha256_antigravity(),
        "method_order": list(typed.cache.method_order),
        "dataset_order": list(typed.cache.dataset_order),
        "payload": payload,
    }
    bare = artifact_view_exp471_antigravity(raw_bare)
    if not isinstance(bare, Mapping):
        raise ValueError("Exp471 complete expected artifact view is invalid")
    return Exp471EvaluationArtifactAntigravity(
        **bare,
        evaluation_sha256=sha256(
            canonical_json_bytes_exp471_antigravity(bare)
        ).hexdigest(),
    )


def _canonical_artifact_bytes_antigravity(
    artifact: Exp471EvaluationArtifactAntigravity,
) -> bytes:
    if not isinstance(artifact, Exp471EvaluationArtifactAntigravity):
        raise ValueError("Exp471 evaluation artifact has the wrong type")
    bare = _artifact_record_antigravity(artifact, include_digest=False)
    if artifact.evaluation_sha256 != sha256(
        canonical_json_bytes_exp471_antigravity(bare)
    ).hexdigest():
        raise ValueError("Exp471 evaluation artifact digest differs")
    return canonical_json_bytes_exp471_antigravity(
        _artifact_record_antigravity(artifact, include_digest=True)
    )


def _assert_no_null_antigravity(value: object, scope: str) -> None:
    if value is None:
        raise ValueError(f"Exp471 evaluation null is forbidden in {scope}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_null_antigravity(item, f"{scope}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_no_null_antigravity(item, f"{scope}[{index}]")


def _finite_number_antigravity(value: object, scope: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise ValueError(f"Exp471 evaluation {scope} must be finite")
    return float(value)


def _unit_number_antigravity(value: object, scope: str) -> float:
    number = _finite_number_antigravity(value, scope)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"Exp471 evaluation {scope} is outside [0, 1]")
    return number


def _bounded_number_antigravity(
    value: object, lower: float, upper: float, scope: str
) -> float:
    number = _finite_number_antigravity(value, scope)
    if number < lower or number > upper:
        raise ValueError(f"Exp471 evaluation {scope} is outside its bounds")
    return number


def _nonnegative_integer_antigravity(value: object, scope: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Exp471 evaluation {scope} must be a nonnegative integer")
    return value


def _method_blocks_antigravity(
    rows: Sequence[Mapping[str, object]], scope: str
) -> tuple[str, ...]:
    blocks: list[str] = []
    for row in rows:
        method = row.get("method")
        if method not in METHOD_ORDER_ANTIGRAVITY:
            raise ValueError(f"Exp471 evaluation {scope} method differs")
        if not blocks or blocks[-1] != method:
            if method in blocks:
                raise ValueError(f"Exp471 evaluation {scope} method blocks differ")
            blocks.append(method)
    return tuple(blocks)


def _curve_tables_antigravity(
    value: object,
) -> Mapping[str, Mapping[str, object]]:
    if not isinstance(value, Mapping) or set(value) != {"roc", "precision_recall"}:
        raise ValueError("Exp471 evaluation curve table schema differs")
    roc = value["roc"]
    precision_recall = value["precision_recall"]
    if (
        not isinstance(roc, Mapping)
        or set(roc) != {"x_name", "y_name", "rows"}
        or roc["x_name"] != "false_positive_rate"
        or roc["y_name"] != "true_positive_rate"
        or not isinstance(roc["rows"], (tuple, list))
    ):
        raise ValueError("Exp471 evaluation ROC schema differs")
    if (
        not isinstance(precision_recall, Mapping)
        or set(precision_recall) != {"x_name", "y_name", "rows"}
        or precision_recall["x_name"] != "recall"
        or precision_recall["y_name"] != "precision"
        or not isinstance(precision_recall["rows"], (tuple, list))
    ):
        raise ValueError("Exp471 evaluation precision-recall schema differs")

    roc_rows: list[Mapping[str, object]] = []
    seen_roc: set[str] = set()
    for raw_row in roc["rows"]:
        if not isinstance(raw_row, Mapping) or set(raw_row) != {
            "method",
            "false_positive_rate",
            "true_positive_rate",
            "threshold",
        }:
            raise ValueError("Exp471 evaluation ROC row schema differs")
        method = raw_row["method"]
        if method not in METHOD_ORDER_ANTIGRAVITY:
            raise ValueError("Exp471 evaluation ROC method differs")
        threshold = raw_row["threshold"]
        if method not in seen_roc:
            if threshold is not None:
                raise ValueError("Exp471 evaluation first ROC threshold must be null")
            threshold_value = math.inf
            seen_roc.add(method)
        else:
            if threshold is None:
                raise ValueError("Exp471 evaluation ROC null is misplaced")
            threshold_value = _finite_number_antigravity(
                threshold, "ROC threshold"
            )
        roc_rows.append(
            MappingProxyType(
                {
                    "method": method,
                    "false_positive_rate": _unit_number_antigravity(
                        raw_row["false_positive_rate"], "ROC false-positive rate"
                    ),
                    "true_positive_rate": _unit_number_antigravity(
                        raw_row["true_positive_rate"], "ROC true-positive rate"
                    ),
                    "threshold": threshold_value,
                }
            )
        )
    if _method_blocks_antigravity(roc_rows, "ROC") != METHOD_ORDER_ANTIGRAVITY:
        raise ValueError("Exp471 evaluation ROC method order differs")
    for method in METHOD_ORDER_ANTIGRAVITY:
        block = tuple(row for row in roc_rows if row["method"] == method)
        if (
            len(block) < 2
            or block[0]["false_positive_rate"] != 0.0
            or block[0]["true_positive_rate"] != 0.0
            or block[0]["threshold"] != math.inf
            or block[-1]["false_positive_rate"] != 1.0
            or block[-1]["true_positive_rate"] != 1.0
        ):
            raise ValueError("Exp471 evaluation ROC frozen endpoints differ")
        for prior, current in zip(block, block[1:]):
            if (
                current["false_positive_rate"] < prior["false_positive_rate"]
                or current["true_positive_rate"] < prior["true_positive_rate"]
                or current["threshold"] > prior["threshold"]
            ):
                raise ValueError("Exp471 evaluation ROC ordering differs")

    pr_rows: list[Mapping[str, object]] = []
    for raw_row in precision_recall["rows"]:
        if not isinstance(raw_row, Mapping) or set(raw_row) != {
            "method",
            "recall",
            "precision",
        }:
            raise ValueError("Exp471 evaluation precision-recall row schema differs")
        if raw_row["method"] not in METHOD_ORDER_ANTIGRAVITY:
            raise ValueError("Exp471 evaluation precision-recall method differs")
        _assert_no_null_antigravity(raw_row, "precision-recall row")
        pr_rows.append(
            MappingProxyType(
                {
                    "method": raw_row["method"],
                    "recall": _unit_number_antigravity(
                        raw_row["recall"], "precision-recall recall"
                    ),
                    "precision": _unit_number_antigravity(
                        raw_row["precision"], "precision-recall precision"
                    ),
                }
            )
        )
    if _method_blocks_antigravity(
        pr_rows, "precision-recall"
    ) != METHOD_ORDER_ANTIGRAVITY:
        raise ValueError("Exp471 evaluation precision-recall method order differs")
    for method in METHOD_ORDER_ANTIGRAVITY:
        block = tuple(row for row in pr_rows if row["method"] == method)
        if (
            len(block) < 2
            or block[0]["recall"] != 1.0
            or block[-1]["recall"] != 0.0
            or block[-1]["precision"] != 1.0
        ):
            raise ValueError(
                "Exp471 evaluation precision-recall frozen endpoints differ"
            )
        if any(
            current["recall"] > prior["recall"]
            for prior, current in zip(block, block[1:])
        ):
            raise ValueError("Exp471 evaluation precision-recall ordering differs")

    return MappingProxyType(
        {
            "roc": MappingProxyType(
                {
                    "x_name": "false_positive_rate",
                    "y_name": "true_positive_rate",
                    "rows": tuple(roc_rows),
                }
            ),
            "precision_recall": MappingProxyType(
                {
                    "x_name": "recall",
                    "y_name": "precision",
                    "rows": tuple(pr_rows),
                }
            ),
        }
    )


def _freeze_exp450_value_antigravity(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_exp450_value_antigravity(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_exp450_value_antigravity(item) for item in value)
    return value


def _freeze_mapping_rows_antigravity(value: object, scope: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"Exp471 evaluation {scope} must be a row sequence")
    rows: list[Mapping[str, object]] = []
    for row in value:
        if not isinstance(row, Mapping):
            raise ValueError(f"Exp471 evaluation {scope} row differs")
        frozen_row = _freeze_exp450_value_antigravity(row)
        assert isinstance(frozen_row, Mapping)
        rows.append(frozen_row)
    return tuple(rows)


def _freeze_method_mapping_antigravity(
    value: object, scope: str
) -> Mapping[str, Mapping[str, object]]:
    if not isinstance(value, Mapping) or set(value) != set(METHOD_ORDER_ANTIGRAVITY):
        raise ValueError(f"Exp471 evaluation {scope} method coverage differs")
    frozen: dict[str, Mapping[str, object]] = {}
    for method in METHOD_ORDER_ANTIGRAVITY:
        row = value[method]
        if not isinstance(row, Mapping):
            raise ValueError(f"Exp471 evaluation {scope} row differs")
        frozen_row = _freeze_exp450_value_antigravity(row)
        assert isinstance(frozen_row, Mapping)
        frozen[method] = frozen_row
    return MappingProxyType(frozen)


def _metric_value_pair_antigravity(
    row: Mapping[str, object],
    metric: str,
    missing_reason: str,
    scope: str,
) -> float | None:
    value = row[metric]
    exclusion = row[f"{metric}_exclusion"]
    if value is None:
        if exclusion != missing_reason:
            raise ValueError(f"Exp471 evaluation {scope} exclusion pairing differs")
        return None
    number = _unit_number_antigravity(value, scope)
    if exclusion is not None:
        raise ValueError(f"Exp471 evaluation {scope} exclusion pairing differs")
    return number


def _per_dataset_metrics_antigravity(
    value: object,
    datasets: tuple[str, ...],
) -> tuple[
    tuple[Mapping[str, object], ...],
    dict[tuple[str, str, str], float | None],
]:
    rows = _freeze_mapping_rows_antigravity(value, "per-dataset metrics")
    expected = tuple(
        (method, dataset)
        for method in METHOD_ORDER_ANTIGRAVITY
        for dataset in datasets
    )
    if len(rows) != len(expected):
        raise ValueError("Exp471 evaluation per-dataset metric coverage differs")
    values: dict[tuple[str, str, str], float | None] = {}
    required = {
        "method",
        "dataset",
        "average_precision",
        "average_precision_exclusion",
        "auroc",
        "auroc_exclusion",
    }
    for row, identity in zip(rows, expected, strict=True):
        if set(row) != required or (row["method"], row["dataset"]) != identity:
            raise ValueError("Exp471 evaluation per-dataset metric order differs")
        method, dataset = identity
        values[(method, dataset, "average_precision")] = (
            _metric_value_pair_antigravity(
                row,
                "average_precision",
                "no_positive_test_labels",
                f"{method}/{dataset} average precision",
            )
        )
        values[(method, dataset, "auroc")] = _metric_value_pair_antigravity(
            row,
            "auroc",
            "single_class_test_labels",
            f"{method}/{dataset} AUROC",
        )
    for dataset in datasets:
        for metric in ("average_precision", "auroc"):
            missing = {
                values[(method, dataset, metric)] is None
                for method in METHOD_ORDER_ANTIGRAVITY
            }
            if len(missing) != 1:
                raise ValueError(
                    "Exp471 evaluation per-dataset label exclusions differ by method"
                )
    return rows, values


def _macro_metrics_antigravity(
    value: object,
    datasets: tuple[str, ...],
    per_dataset: Mapping[tuple[str, str, str], float | None],
) -> Mapping[str, Mapping[str, object]]:
    rows = _freeze_method_mapping_antigravity(value, "macro metrics")
    required = {
        "average_precision",
        "auroc",
        "average_precision_denominator",
        "average_precision_excluded",
        "auroc_denominator",
        "auroc_excluded",
    }
    for method in METHOD_ORDER_ANTIGRAVITY:
        row = rows[method]
        if set(row) != required:
            raise ValueError("Exp471 evaluation macro metric schema differs")
        for metric in ("average_precision", "auroc"):
            available = tuple(
                per_dataset[(method, dataset, metric)]
                for dataset in datasets
                if per_dataset[(method, dataset, metric)] is not None
            )
            denominator = _nonnegative_integer_antigravity(
                row[f"{metric}_denominator"], f"macro {metric} denominator"
            )
            excluded = _nonnegative_integer_antigravity(
                row[f"{metric}_excluded"], f"macro {metric} excluded count"
            )
            if (
                denominator != len(available)
                or denominator + excluded != len(datasets)
            ):
                raise ValueError("Exp471 evaluation macro denominator relation differs")
            observed = row[metric]
            if not available:
                if observed is not None:
                    raise ValueError("Exp471 evaluation empty macro value differs")
            else:
                number = _unit_number_antigravity(observed, f"macro {metric}")
                expected = float(np.mean(np.asarray(available, dtype=np.float64)))
                if number != expected:
                    raise ValueError("Exp471 evaluation macro value relation differs")
    return rows


def _micro_metrics_antigravity(
    value: object,
    test_counts: Mapping[str, int],
) -> Mapping[str, Mapping[str, object]]:
    rows = _freeze_method_mapping_antigravity(value, "micro metrics")
    required = {
        "average_precision",
        "auroc",
        "row_denominator",
        "average_precision_exclusion",
        "auroc_exclusion",
    }
    total_rows = sum(test_counts.values())
    missing_patterns: dict[str, set[bool]] = {
        "average_precision": set(),
        "auroc": set(),
    }
    for method in METHOD_ORDER_ANTIGRAVITY:
        row = rows[method]
        if set(row) != required:
            raise ValueError("Exp471 evaluation micro metric schema differs")
        denominator = _nonnegative_integer_antigravity(
            row["row_denominator"], "micro row denominator"
        )
        if denominator != total_rows:
            raise ValueError("Exp471 evaluation micro row denominator differs")
        average_precision = _metric_value_pair_antigravity(
            row,
            "average_precision",
            "no_positive_test_rows",
            f"{method} micro average precision",
        )
        auroc = _metric_value_pair_antigravity(
            row,
            "auroc",
            "single_class_pooled_test_rows",
            f"{method} micro AUROC",
        )
        missing_patterns["average_precision"].add(average_precision is None)
        missing_patterns["auroc"].add(auroc is None)
    if any(len(pattern) != 1 for pattern in missing_patterns.values()):
        raise ValueError("Exp471 evaluation pooled label exclusions differ by method")
    return rows


def _paired_statistics_antigravity(
    value: object,
    datasets: tuple[str, ...],
    per_dataset: Mapping[tuple[str, str, str], float | None],
) -> tuple[Mapping[str, object], ...]:
    rows = _freeze_mapping_rows_antigravity(value, "paired statistics")
    expected = tuple(
        (f"{variant}-A", metric)
        for metric in ("average_precision", "auroc")
        for variant in ("B", "C", "D")
    )
    if len(rows) != len(expected):
        raise ValueError("Exp471 evaluation paired-statistic coverage differs")
    required = {
        "comparison",
        "metric",
        "dataset_order",
        "point_estimate",
        "lower",
        "upper",
        "wins",
        "ties",
        "losses",
        "seed",
        "draw_count",
        "draw_sha256",
    }
    by_identity: dict[tuple[str, str], Mapping[str, object]] = {}
    for row, identity in zip(rows, expected, strict=True):
        if (
            set(row) != required
            or (row["comparison"], row["metric"]) != identity
        ):
            raise ValueError("Exp471 evaluation paired-statistic order differs")
        by_identity[identity] = row

    draw_digest_by_common_count: dict[int, str] = {}
    for metric in ("average_precision", "auroc"):
        common = tuple(
            dataset
            for dataset in datasets
            if all(
                per_dataset[(method, dataset, metric)] is not None
                for method in METHOD_ORDER_ANTIGRAVITY[:4]
            )
        )
        reference = np.asarray(
            [per_dataset[("A", dataset, metric)] for dataset in common],
            dtype=np.float64,
        )
        differences = {
            variant: np.asarray(
                [per_dataset[(variant, dataset, metric)] for dataset in common],
                dtype=np.float64,
            )
            - reference
            for variant in ("B", "C", "D")
        }
        metric_draw_digest: str | None = None
        for variant in ("B", "C", "D"):
            row = by_identity[(f"{variant}-A", metric)]
            if tuple(row["dataset_order"]) != common:
                raise ValueError("Exp471 evaluation paired dataset order differs")
            if row["seed"] != frozen_evaluation_antigravity._BOOTSTRAP_SEED:
                raise ValueError("Exp471 evaluation paired seed differs")
            wins = _nonnegative_integer_antigravity(row["wins"], "paired wins")
            ties = _nonnegative_integer_antigravity(row["ties"], "paired ties")
            losses = _nonnegative_integer_antigravity(row["losses"], "paired losses")
            draw_count = _nonnegative_integer_antigravity(
                row["draw_count"], "paired draw count"
            )
            difference = differences[variant]
            expected_wins = int(
                np.count_nonzero(
                    difference > frozen_evaluation_antigravity._TIE_TOLERANCE
                )
            )
            expected_ties = int(
                np.count_nonzero(
                    np.abs(difference)
                    <= frozen_evaluation_antigravity._TIE_TOLERANCE
                )
            )
            expected_losses = int(
                np.count_nonzero(
                    difference < -frozen_evaluation_antigravity._TIE_TOLERANCE
                )
            )
            if (wins, ties, losses) != (
                expected_wins,
                expected_ties,
                expected_losses,
            ) or wins + ties + losses != len(common):
                raise ValueError("Exp471 evaluation paired counts differ")
            if not common:
                if (
                    row["point_estimate"] is not None
                    or row["lower"] is not None
                    or row["upper"] is not None
                    or draw_count != 0
                    or row["draw_sha256"] is not None
                ):
                    raise ValueError("Exp471 evaluation empty paired relation differs")
                continue
            point = _bounded_number_antigravity(
                row["point_estimate"], -1.0, 1.0, "paired point estimate"
            )
            lower = _bounded_number_antigravity(
                row["lower"], -1.0, 1.0, "paired lower bound"
            )
            upper = _bounded_number_antigravity(
                row["upper"], -1.0, 1.0, "paired upper bound"
            )
            draw_sha256 = row["draw_sha256"]
            if (
                point != float(difference.mean())
                or lower > upper
                or draw_count != frozen_evaluation_antigravity._BOOTSTRAP_SAMPLES
                or type(draw_sha256) is not str
                or _SHA256_ANTIGRAVITY.fullmatch(draw_sha256) is None
            ):
                raise ValueError("Exp471 evaluation paired relation differs")
            if metric_draw_digest is None:
                metric_draw_digest = draw_sha256
            elif draw_sha256 != metric_draw_digest:
                raise ValueError("Exp471 evaluation paired draw evidence differs")
        if common:
            assert metric_draw_digest is not None
            prior_digest = draw_digest_by_common_count.setdefault(
                len(common), metric_draw_digest
            )
            if metric_draw_digest != prior_digest:
                raise ValueError("Exp471 evaluation paired draw evidence differs")
    return rows


def _metric_tables_antigravity(
    value: object,
    datasets: tuple[str, ...],
    test_counts: Mapping[str, int],
) -> MetricTablesExp450Antigravity:
    _assert_no_null_antigravity(value, "metric tables")
    decoded = _decode_semantic_none_antigravity(value)
    if not isinstance(decoded, Mapping) or set(decoded) != {
        "method_order",
        "per_dataset",
        "macro",
        "micro",
        "paired_statistics",
    }:
        raise ValueError("Exp471 evaluation metric-table schema differs")
    if tuple(decoded["method_order"]) != METHOD_ORDER_ANTIGRAVITY:
        raise ValueError("Exp471 evaluation metric method order differs")
    per_dataset, values = _per_dataset_metrics_antigravity(
        decoded["per_dataset"], datasets
    )
    macro = _macro_metrics_antigravity(decoded["macro"], datasets, values)
    micro = _micro_metrics_antigravity(decoded["micro"], test_counts)
    paired = _paired_statistics_antigravity(
        decoded["paired_statistics"], datasets, values
    )
    return MetricTablesExp450Antigravity(
        method_order=METHOD_ORDER_ANTIGRAVITY,
        per_dataset=per_dataset,
        macro=macro,
        micro=micro,
        paired_statistics=paired,
    )


def _typed_metric_pair_antigravity(
    row: Mapping[str, object], metric: str, missing_reason: str, scope: str
) -> float | None:
    value = row[metric]
    exclusion = row[f"{metric}_exclusion"]
    if value is None:
        if exclusion != missing_reason:
            raise ValueError(f"Exp471 evaluation {scope} exclusion pairing differs")
        return None
    if exclusion is not None:
        raise ValueError(f"Exp471 evaluation {scope} exclusion pairing differs")
    return _unit_number_antigravity(value, scope)


def _typed_continuous_table_antigravity(
    value: object,
    datasets: tuple[str, ...],
    test_counts: Mapping[str, int],
) -> tuple[MetricTablesExp450Antigravity, Mapping[str, Mapping[str, str | None]]]:
    """Validate a v2 metric support table from its scientific relations alone."""

    if not isinstance(value, Mapping) or set(value) != {
        "method_order", "per_dataset", "macro", "micro", "paired_statistics", "semantic_nulls"
    } or tuple(value["method_order"]) != METHOD_ORDER_ANTIGRAVITY:
        raise ValueError("Exp471 evaluation typed continuous table schema differs")
    raw_rows = value["per_dataset"]
    if not isinstance(raw_rows, (tuple, list)):
        raise ValueError("Exp471 evaluation typed per-dataset rows differ")
    expected_identities = tuple(
        (method, dataset) for method in METHOD_ORDER_ANTIGRAVITY for dataset in datasets
    )
    if len(raw_rows) != len(expected_identities):
        raise ValueError("Exp471 evaluation typed per-dataset coverage differs")
    required_row = {
        "method", "dataset", "average_precision", "average_precision_exclusion",
        "auroc", "auroc_exclusion",
    }
    rows: list[Mapping[str, object]] = []
    values: dict[tuple[str, str, str], float | None] = {}
    statuses: dict[str, dict[str, str | None]] = {
        method: {} for method in METHOD_ORDER_ANTIGRAVITY
    }
    for raw_row, (method, dataset) in zip(raw_rows, expected_identities, strict=True):
        if (
            not isinstance(raw_row, Mapping)
            or set(raw_row) != required_row
            or (raw_row.get("method"), raw_row.get("dataset")) != (method, dataset)
        ):
            raise ValueError("Exp471 evaluation typed per-dataset order differs")
        ap_exclusion = raw_row["average_precision_exclusion"]
        auroc_exclusion = raw_row["auroc_exclusion"]
        abstained = (
            raw_row["average_precision"] is None
            and raw_row["auroc"] is None
            and ap_exclusion == "insufficient_train_rows"
            and auroc_exclusion == "insufficient_train_rows"
        )
        if abstained:
            statuses[method][dataset] = "insufficient_train_rows"
            values[(method, dataset, "average_precision")] = None
            values[(method, dataset, "auroc")] = None
        else:
            if ap_exclusion == "insufficient_train_rows" or auroc_exclusion == "insufficient_train_rows":
                raise ValueError("Exp471 evaluation typed abstention placement differs")
            statuses[method][dataset] = None
            values[(method, dataset, "average_precision")] = _typed_metric_pair_antigravity(
                raw_row, "average_precision", "no_positive_test_labels",
                f"{method}/{dataset} average precision",
            )
            values[(method, dataset, "auroc")] = _typed_metric_pair_antigravity(
                raw_row, "auroc", "single_class_test_labels", f"{method}/{dataset} AUROC"
            )
        rows.append(MappingProxyType(dict(raw_row)))

    raw_macro = value["macro"]
    raw_micro = value["micro"]
    if not isinstance(raw_macro, Mapping) or not isinstance(raw_micro, Mapping):
        raise ValueError("Exp471 evaluation typed aggregate schema differs")
    required_macro = {
        "average_precision", "auroc", "average_precision_denominator",
        "average_precision_excluded", "auroc_denominator", "auroc_excluded",
    }
    required_micro = {
        "average_precision", "auroc", "row_denominator",
        "average_precision_exclusion", "auroc_exclusion",
    }
    macro: dict[str, Mapping[str, object]] = {}
    micro: dict[str, Mapping[str, object]] = {}
    for method in METHOD_ORDER_ANTIGRAVITY:
        macro_row = raw_macro.get(method)
        micro_row = raw_micro.get(method)
        if (
            not isinstance(macro_row, Mapping) or set(macro_row) != required_macro
            or not isinstance(micro_row, Mapping) or set(micro_row) != required_micro
        ):
            raise ValueError("Exp471 evaluation typed aggregate method coverage differs")
        for metric in ("average_precision", "auroc"):
            available = tuple(
                values[(method, dataset, metric)] for dataset in datasets
                if values[(method, dataset, metric)] is not None
            )
            denominator = _nonnegative_integer_antigravity(
                macro_row[f"{metric}_denominator"], f"typed macro {metric} denominator"
            )
            excluded = _nonnegative_integer_antigravity(
                macro_row[f"{metric}_excluded"], f"typed macro {metric} excluded"
            )
            if denominator != len(available) or denominator + excluded != len(datasets):
                raise ValueError("Exp471 evaluation typed macro denominator relation differs")
            if not available:
                if macro_row[metric] is not None:
                    raise ValueError("Exp471 evaluation typed empty macro differs")
            elif _unit_number_antigravity(macro_row[metric], f"typed macro {metric}") != float(
                np.mean(np.asarray(available, dtype=np.float64))
            ):
                raise ValueError("Exp471 evaluation typed macro metric relation differs")
        scored_datasets = tuple(
            dataset for dataset in datasets if statuses[method][dataset] is None
        )
        if _nonnegative_integer_antigravity(
            micro_row["row_denominator"], "typed micro row denominator"
        ) != sum(test_counts[dataset] for dataset in scored_datasets):
            raise ValueError("Exp471 evaluation typed micro support differs")
        _typed_metric_pair_antigravity(
            micro_row, "average_precision", "no_positive_test_rows",
            f"{method} typed micro average precision",
        )
        _typed_metric_pair_antigravity(
            micro_row, "auroc", "single_class_pooled_test_rows", f"{method} typed micro AUROC"
        )
        macro[method] = MappingProxyType(dict(macro_row))
        micro[method] = MappingProxyType(dict(micro_row))

    raw_pairs = value["paired_statistics"]
    if not isinstance(raw_pairs, (tuple, list)):
        raise ValueError("Exp471 evaluation typed paired rows differ")
    expected_pairs = tuple(
        (f"{variant}-A", metric)
        for metric in ("average_precision", "auroc")
        for variant in ("B", "C", "D")
    )
    pair_fields = {
        "comparison", "metric", "dataset_order", "point_estimate", "lower", "upper",
        "wins", "ties", "losses", "seed", "draw_count", "draw_sha256",
        "scored_intersection_datasets", "scored_intersection_count",
        "scored_intersection_sha256",
    }
    if len(raw_pairs) != len(expected_pairs):
        raise ValueError("Exp471 evaluation typed pair coverage differs")
    pairs: list[Mapping[str, object]] = []
    for raw_pair, (comparison, metric) in zip(raw_pairs, expected_pairs, strict=True):
        if (
            not isinstance(raw_pair, Mapping) or set(raw_pair) != pair_fields
            or (raw_pair.get("comparison"), raw_pair.get("metric")) != (comparison, metric)
        ):
            raise ValueError("Exp471 evaluation typed pair order differs")
        variant = comparison[0]
        metric_support = tuple(
            dataset for dataset in datasets
            if all(values[(method, dataset, metric)] is not None for method in ("A", "B", "C", "D"))
        )
        pair_support = tuple(
            dataset for dataset in datasets
            if statuses["A"][dataset] is None and statuses[variant][dataset] is None
        )
        if (
            not isinstance(raw_pair["dataset_order"], (tuple, list))
            or tuple(raw_pair["dataset_order"]) != metric_support
            or not isinstance(raw_pair["scored_intersection_datasets"], (tuple, list))
            or tuple(raw_pair["scored_intersection_datasets"]) != pair_support
            or raw_pair["scored_intersection_count"] != len(pair_support)
            or raw_pair["scored_intersection_sha256"] != sha256(
                canonical_json_bytes_exp471_antigravity(list(pair_support))
            ).hexdigest()
            or raw_pair["seed"] != frozen_evaluation_antigravity._BOOTSTRAP_SEED
        ):
            raise ValueError("Exp471 evaluation typed pair support differs")
        difference = np.asarray(
            [values[(variant, dataset, metric)] - values[("A", dataset, metric)] for dataset in metric_support],
            dtype=np.float64,
        )
        wins = _nonnegative_integer_antigravity(raw_pair["wins"], "typed paired wins")
        ties = _nonnegative_integer_antigravity(raw_pair["ties"], "typed paired ties")
        losses = _nonnegative_integer_antigravity(raw_pair["losses"], "typed paired losses")
        expected_counts = (
            int(np.count_nonzero(difference > frozen_evaluation_antigravity._TIE_TOLERANCE)),
            int(np.count_nonzero(np.abs(difference) <= frozen_evaluation_antigravity._TIE_TOLERANCE)),
            int(np.count_nonzero(difference < -frozen_evaluation_antigravity._TIE_TOLERANCE)),
        )
        if (wins, ties, losses) != expected_counts or sum(expected_counts) != len(metric_support):
            raise ValueError("Exp471 evaluation typed pair count relation differs")
        draws = _nonnegative_integer_antigravity(raw_pair["draw_count"], "typed paired draws")
        if not metric_support:
            if any(raw_pair[name] is not None for name in ("point_estimate", "lower", "upper", "draw_sha256")) or draws != 0:
                raise ValueError("Exp471 evaluation typed empty pair differs")
        elif (
            _bounded_number_antigravity(raw_pair["point_estimate"], -1.0, 1.0, "typed pair point") != float(difference.mean())
            or _bounded_number_antigravity(raw_pair["lower"], -1.0, 1.0, "typed pair lower") > _bounded_number_antigravity(raw_pair["upper"], -1.0, 1.0, "typed pair upper")
            or draws != frozen_evaluation_antigravity._BOOTSTRAP_SAMPLES
            or type(raw_pair["draw_sha256"]) is not str
            or _SHA256_ANTIGRAVITY.fullmatch(raw_pair["draw_sha256"]) is None
        ):
            raise ValueError("Exp471 evaluation typed pair metric relation differs")
        pairs.append(MappingProxyType(dict(raw_pair)))

    raw_nulls = value["semantic_nulls"]
    if not isinstance(raw_nulls, (tuple, list)):
        raise ValueError("Exp471 evaluation typed semantic-null rows differ")
    expected_nulls = [
        {"method": method, "dataset": dataset, "metric": metric, "reason": "insufficient_train_rows"}
        for method in METHOD_ORDER_ANTIGRAVITY
        for dataset in datasets
        if statuses[method][dataset] == "insufficient_train_rows"
        for metric in ("average_precision", "auroc", "roc", "precision_recall")
    ]
    if list(raw_nulls) != expected_nulls:
        raise ValueError("Exp471 evaluation typed semantic-null placement differs")
    return (
        MetricTablesExp450Antigravity(
            METHOD_ORDER_ANTIGRAVITY, tuple(rows), MappingProxyType(macro),
            MappingProxyType(micro), tuple(
                MappingProxyType({
                    name: pair[name]
                    for name in (
                        "comparison", "metric", "dataset_order", "point_estimate",
                        "lower", "upper", "wins", "ties", "losses", "seed",
                        "draw_count", "draw_sha256",
                    )
                })
                for pair in pairs
            ),
        ),
        MappingProxyType({method: MappingProxyType(statuses[method]) for method in METHOD_ORDER_ANTIGRAVITY}),
    )


def _typed_curve_record_antigravity(
    value: object,
    datasets: tuple[str, ...],
    statuses: Mapping[str, Mapping[str, str | None]],
) -> Mapping[str, Mapping[str, object]]:
    if not isinstance(value, Mapping) or set(value) != {"curves", "semantic_nulls"}:
        raise ValueError("Exp471 evaluation typed curve schema differs")
    expected_nulls = [
        {"method": method, "dataset": dataset, "metric": metric, "reason": "insufficient_train_rows"}
        for method in METHOD_ORDER_ANTIGRAVITY
        for dataset in datasets
        if statuses[method][dataset] == "insufficient_train_rows"
        for metric in ("roc", "precision_recall")
    ]
    if not isinstance(value["semantic_nulls"], (tuple, list)) or list(value["semantic_nulls"]) != expected_nulls:
        raise ValueError("Exp471 evaluation curve semantic-null placement differs")
    curves = _curve_tables_antigravity(value["curves"])
    curve_methods = _method_blocks_antigravity(curves["roc"]["rows"], "ROC")
    if any(not any(statuses[method][dataset] is None for dataset in datasets) for method in curve_methods):
        raise ValueError("Exp471 evaluation curve includes abstained method")
    return curves


def _typed_decision_tables_antigravity(
    value: object,
    datasets: tuple[str, ...],
    test_counts: Mapping[str, int],
    statuses: Mapping[str, Mapping[str, str | None]],
    coverage: Mapping[str, Mapping[str, object]],
) -> None:
    if not isinstance(value, Mapping) or set(value) != {"full_population"}:
        raise ValueError("Exp471 evaluation decision-table schema differs")
    full = value["full_population"]
    if not isinstance(full, Mapping) or set(full) != {"dataset_rows", "summary_rows"}:
        raise ValueError("Exp471 evaluation full-population schema differs")
    rows, summaries = full["dataset_rows"], full["summary_rows"]
    if not isinstance(rows, (tuple, list)) or not isinstance(summaries, (tuple, list)):
        raise ValueError("Exp471 evaluation decision rows differ")
    protocols = lambda method: (("proposal_threshold", None),) if method in METHOD_ORDER_ANTIGRAVITY[:4] else tuple(("conformal", alpha) for alpha in ALPHAS) + (("matched_budget", None),)
    expected = tuple(
        (method, dataset, protocol, alpha)
        for method in METHOD_ORDER_ANTIGRAVITY
        for protocol, alpha in protocols(method)
        for dataset in datasets
    )
    required = {
        "method", "dataset", "protocol", "alpha", "status", "abstention_reason", "alerts", "alert_count",
        "tp", "fp", "fn", "tn", "precision", "recall", "f1", "false_positive_rate",
    }
    if len(rows) != len(expected):
        raise ValueError("Exp471 evaluation decision population coverage differs")
    grouped: dict[tuple[str, str, float | None], list[Mapping[str, object]]] = {}
    for row, identity in zip(rows, expected, strict=True):
        method, dataset, protocol, alpha = identity
        if not isinstance(row, Mapping) or set(row) != required or tuple(row.get(name) for name in ("method", "dataset", "protocol", "alpha")) != identity:
            raise ValueError("Exp471 evaluation decision order differs")
        abstention_reason = statuses[method][dataset]
        expected_status = "abstained" if abstention_reason is not None else "scored"
        if row["status"] != expected_status or row["abstention_reason"] != abstention_reason:
            raise ValueError("Exp471 evaluation decision status differs")
        if not isinstance(row["alerts"], (tuple, list)) or any(type(index) is not int or index < 0 for index in row["alerts"]) or list(row["alerts"]) != sorted(set(row["alerts"])):
            raise ValueError("Exp471 evaluation decision alerts differ")
        if _nonnegative_integer_antigravity(row["alert_count"], "decision alert count") != len(row["alerts"]):
            raise ValueError("Exp471 evaluation decision alert count differs")
        if expected_status == "abstained" and row["alerts"]:
            raise ValueError("Exp471 evaluation abstention must not alert")
        counts = {name: _nonnegative_integer_antigravity(row[name], f"decision {name}") for name in ("tp", "fp", "fn", "tn")}
        if sum(counts.values()) != test_counts[dataset]:
            raise ValueError("Exp471 evaluation decision confusion support differs")
        for name, expected_value in _derived_decision_metrics_antigravity(counts).items():
            if _unit_number_antigravity(row[name], f"decision {name}") != expected_value:
                raise ValueError("Exp471 evaluation decision derived metric differs")
        grouped.setdefault((method, protocol, alpha), []).append(row)
    expected_summary = tuple((method, protocol, alpha) for method in METHOD_ORDER_ANTIGRAVITY for protocol, alpha in protocols(method))
    summary_fields = {"method", "protocol", "alpha", "support_count", "alert_count", "coverage", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "false_positive_rate"}
    if len(summaries) != len(expected_summary):
        raise ValueError("Exp471 evaluation decision summary coverage differs")
    for summary, identity in zip(summaries, expected_summary, strict=True):
        if not isinstance(summary, Mapping) or set(summary) != summary_fields or tuple(summary.get(name) for name in ("method", "protocol", "alpha")) != identity:
            raise ValueError("Exp471 evaluation decision summary order differs")
        group = grouped[identity]
        if _nonnegative_integer_antigravity(summary["support_count"], "decision summary support") != len(group) or _nonnegative_integer_antigravity(summary["alert_count"], "decision summary alerts") != sum(int(row["alert_count"]) for row in group):
            raise ValueError("Exp471 evaluation decision summary support differs")
        expected_coverage = sum(row["status"] == "scored" for row in group) / len(group)
        if _unit_number_antigravity(summary["coverage"], "decision summary coverage") != expected_coverage:
            raise ValueError("Exp471 evaluation decision summary coverage differs")
        counts = {name: sum(int(row[name]) for row in group) for name in ("tp", "fp", "fn", "tn")}
        if any(_nonnegative_integer_antigravity(summary[name], f"decision summary {name}") != value for name, value in counts.items()):
            raise ValueError("Exp471 evaluation decision summary confusion differs")
        for name, expected_value in _derived_decision_metrics_antigravity(counts).items():
            if _unit_number_antigravity(summary[name], f"decision summary {name}") != expected_value:
                raise ValueError("Exp471 evaluation decision summary derived metric differs")
        method = identity[0]
        if summary["coverage"] != coverage[method]["coverage"]:
            raise ValueError("Exp471 evaluation decision/coverage relation differs")


def _validated_payload_antigravity(
    artifact: Exp471EvaluationArtifactAntigravity,
) -> tuple[MetricTablesExp450Antigravity, Mapping[str, Mapping[str, object]]]:
    payload = artifact.payload
    if set(payload) != {
        "schema",
        "source_authority",
        "score_cache",
        "implementation",
        "coverage",
        "decision_tables",
        "continuous_tables",
        "curve_tables",
        "support_authority",
    } or payload["schema"] != _EVALUATION_SCHEMA_ANTIGRAVITY:
        raise ValueError("Exp471 evaluation payload schema differs")
    for scope in ("source_authority", "score_cache", "implementation"):
        _assert_no_null_antigravity(payload[scope], scope)
    if payload["implementation"] != _implementation_record_antigravity():
        raise ValueError("Exp471 evaluation implementation provenance differs")
    source = payload["source_authority"]
    cache = payload["score_cache"]
    if (
        not isinstance(source, Mapping)
        or set(source)
        != {
            "authority_sha256",
            "source_commit",
            "source_context_sha256",
            "inventory_sha256",
        }
        or source.get("authority_sha256") != artifact.source_authority_sha256
        or type(source.get("source_commit")) is not str
        or re.fullmatch(r"^[0-9a-f]{40}$", source["source_commit"]) is None
        or type(source.get("source_context_sha256")) is not str
        or _SHA256_ANTIGRAVITY.fullmatch(source["source_context_sha256"])
        is None
        or type(source.get("inventory_sha256")) is not str
        or _SHA256_ANTIGRAVITY.fullmatch(source["inventory_sha256"]) is None
        or not isinstance(cache, Mapping)
        or cache.get("manifest_sha256") != artifact.cache_manifest_sha256
        or set(cache)
        != {
            "cache_context_sha256",
            "manifest_sha256",
            "total_shards",
            "scored_shards",
            "abstained_shards",
            "failed_shards",
            "dataset_test_count",
        }
        or not isinstance(cache.get("dataset_test_count"), Mapping)
        or type(cache.get("cache_context_sha256")) is not str
        or _SHA256_ANTIGRAVITY.fullmatch(cache["cache_context_sha256"]) is None
        or type(cache.get("total_shards")) is not int
        or cache["total_shards"]
        != len(METHOD_ORDER_ANTIGRAVITY) * len(artifact.dataset_order)
        or any(
            type(cache.get(name)) is not int or cache[name] < 0
            for name in ("scored_shards", "abstained_shards", "failed_shards")
        )
        or cache["scored_shards"] + cache["abstained_shards"] + cache["failed_shards"]
        != cache["total_shards"]
        or cache["failed_shards"] != 0
        or (
            len(artifact.dataset_order) == 1_117
            and (cache["total_shards"], cache["scored_shards"], cache["abstained_shards"])
            != (11_170, 10_775, 395)
        )
    ):
        raise ValueError("Exp471 evaluation payload authority binding differs")
    raw_counts = cache["dataset_test_count"]
    assert isinstance(raw_counts, Mapping)
    if set(raw_counts) != set(artifact.dataset_order):
        raise ValueError("Exp471 evaluation dataset row-count coverage differs")
    test_counts: dict[str, int] = {}
    for dataset in artifact.dataset_order:
        count = raw_counts[dataset]
        if type(count) is not int or count < 1:
            raise ValueError("Exp471 evaluation dataset row count differs")
        test_counts[dataset] = count

    coverage = payload["coverage"]
    if not isinstance(coverage, Mapping) or set(coverage) != {"rows"}:
        raise ValueError("Exp471 evaluation coverage schema differs")
    rows = coverage["rows"]
    if not isinstance(rows, (tuple, list)) or len(rows) != len(METHOD_ORDER_ANTIGRAVITY):
        raise ValueError("Exp471 evaluation coverage rows differ")
    coverage_by_method: dict[str, Mapping[str, object]] = {}
    for method, row in zip(METHOD_ORDER_ANTIGRAVITY, rows, strict=True):
        if not isinstance(row, Mapping) or set(row) != {
            "method", "population", "scored", "abstained", "failed", "coverage", "abstention_reasons"
        } or row["method"] != method:
            raise ValueError("Exp471 evaluation coverage order differs")
        if any(type(row[name]) is not int or row[name] < 0 for name in ("population", "scored", "abstained", "failed")) or (
            row["population"] != len(artifact.dataset_order)
            or row["scored"] + row["abstained"] + row["failed"] != row["population"]
            or row["failed"] != 0
            or not isinstance(row["abstention_reasons"], Mapping)
            or any(type(reason) is not str or type(count) is not int or count < 1 for reason, count in row["abstention_reasons"].items())
            or _finite_number_antigravity(row["coverage"], "coverage") != row["scored"] / row["population"]
        ):
            raise ValueError("Exp471 evaluation coverage relation differs")
        coverage_by_method[method] = row

    support = payload["support_authority"]
    if not isinstance(support, Mapping) or set(support) != {
        "all_ten_common_support_count", "all_ten_common_support_datasets", "all_ten_common_support_sha256"
    }:
        raise ValueError("Exp471 evaluation support authority schema differs")
    if not isinstance(support["all_ten_common_support_datasets"], (tuple, list)):
        raise ValueError("Exp471 evaluation support dataset list differs")
    common = tuple(support["all_ten_common_support_datasets"])
    if (
        type(support["all_ten_common_support_count"]) is not int
        or support["all_ten_common_support_count"] != len(common)
        or any(type(dataset) is not str for dataset in common)
        or common != tuple(dataset for dataset in artifact.dataset_order if dataset in set(common))
        or support["all_ten_common_support_sha256"] != sha256(canonical_json_bytes_exp471_antigravity(list(common))).hexdigest()
        or (len(artifact.dataset_order) == 1_117 and len(common) != 922)
    ):
        raise ValueError("Exp471 evaluation support authority differs")

    continuous = payload["continuous_tables"]
    curves_payload = payload["curve_tables"]
    if (
        not isinstance(continuous, Mapping)
        or set(continuous) != {"scored_support", "all_ten_common_support"}
        or not isinstance(curves_payload, Mapping)
        or set(curves_payload) != {"scored_support", "all_ten_common_support"}
    ):
        raise ValueError("Exp471 evaluation support table schema differs")
    decoded_metrics: dict[str, MetricTablesExp450Antigravity] = {}
    decoded_curves: dict[str, Mapping[str, Mapping[str, object]]] = {}
    scored_metrics, scored_statuses = _typed_continuous_table_antigravity(
        continuous["scored_support"], artifact.dataset_order, test_counts
    )
    derived_common = tuple(
        dataset for dataset in artifact.dataset_order
        if all(scored_statuses[method][dataset] is None for method in METHOD_ORDER_ANTIGRAVITY)
    )
    if common != derived_common:
        raise ValueError("Exp471 evaluation all-ten support relation differs")
    for method in METHOD_ORDER_ANTIGRAVITY:
        inferred_reasons: dict[str, int] = {}
        for dataset in artifact.dataset_order:
            reason = scored_statuses[method][dataset]
            if reason is not None:
                inferred_reasons[reason] = inferred_reasons.get(reason, 0) + 1
        row = coverage_by_method[method]
        if (
            row["scored"] != sum(scored_statuses[method][dataset] is None for dataset in artifact.dataset_order)
            or row["abstained"] != sum(scored_statuses[method][dataset] is not None for dataset in artifact.dataset_order)
            or row["abstention_reasons"] != inferred_reasons
        ):
            raise ValueError("Exp471 evaluation coverage/status relation differs")
    decoded_metrics["scored_support"] = scored_metrics
    decoded_curves["scored_support"] = _typed_curve_record_antigravity(
        curves_payload["scored_support"], artifact.dataset_order, scored_statuses
    )
    common_metrics, common_statuses = _typed_continuous_table_antigravity(
        continuous["all_ten_common_support"], common,
        {dataset: test_counts[dataset] for dataset in common},
    )
    if any(common_statuses[method][dataset] is not None for method in METHOD_ORDER_ANTIGRAVITY for dataset in common):
        raise ValueError("Exp471 evaluation common support contains abstention")
    decoded_metrics["all_ten_common_support"] = common_metrics
    decoded_curves["all_ten_common_support"] = _typed_curve_record_antigravity(
        curves_payload["all_ten_common_support"], common, common_statuses
    )
    _typed_decision_tables_antigravity(
        payload["decision_tables"], artifact.dataset_order, test_counts,
        scored_statuses, coverage_by_method,
    )
    metrics = decoded_metrics["scored_support"]
    curves = decoded_curves["scored_support"]
    return metrics, curves


def _validate_artifact_integrity_antigravity(
    artifact: Exp471EvaluationArtifactAntigravity,
) -> tuple[MetricTablesExp450Antigravity, Mapping[str, Mapping[str, object]]]:
    _canonical_artifact_bytes_antigravity(artifact)
    if (
        artifact.code_closure_sha256 != _code_closure_sha256_antigravity()
        or artifact.method_order != METHOD_ORDER_ANTIGRAVITY
    ):
        raise ValueError("Exp471 evaluation frozen implementation binding differs")
    return _validated_payload_antigravity(artifact)


def _validate_artifact_authority_antigravity(
    artifact: Exp471EvaluationArtifactAntigravity,
    session: Exp471SourceSessionAntigravity,
    typed: _TypedCacheAntigravity,
    *,
    expected: Exp471EvaluationArtifactAntigravity | None = None,
) -> None:
    _canonical_artifact_bytes_antigravity(artifact)
    if (
        artifact.code_closure_sha256 != _code_closure_sha256_antigravity()
        or artifact.method_order != METHOD_ORDER_ANTIGRAVITY
    ):
        raise ValueError("Exp471 evaluation frozen implementation binding differs")
    if expected is not None:
        if (
            artifact.source_authority_sha256 != session.authority.authority_sha256
            or artifact.cache_manifest_sha256 != typed.cache.manifest_sha256
            or artifact.method_order != typed.cache.method_order
            or artifact.dataset_order != typed.cache.dataset_order
        ):
            raise ValueError("Exp471 evaluation durable authority differs")
        if (
            canonical_json_bytes_exp471_antigravity(artifact.payload)
            != canonical_json_bytes_exp471_antigravity(expected.payload)
            or _canonical_artifact_bytes_antigravity(artifact)
            != _canonical_artifact_bytes_antigravity(expected)
        ):
            raise ValueError(
                "Exp471 evaluation payload differs from verified scores and labels"
            )
        return
    _validated_payload_antigravity(artifact)
    expected_coverage = {"rows": _coverage_rows_antigravity(typed)}
    expected_support = _support_authority_antigravity(
        _all_ten_common_support_antigravity(typed)
    )
    if (
        artifact.source_authority_sha256 != session.authority.authority_sha256
        or artifact.cache_manifest_sha256 != typed.cache.manifest_sha256
        or artifact.method_order != typed.cache.method_order
        or artifact.dataset_order != typed.cache.dataset_order
        or artifact.payload["source_authority"]
        != _source_record_antigravity(session.authority)
        or artifact.payload["score_cache"]
        != _cache_record_antigravity(typed.cache, typed.test_counts)
        or canonical_json_bytes_exp471_antigravity(artifact.payload["coverage"])
        != canonical_json_bytes_exp471_antigravity(expected_coverage)
        or canonical_json_bytes_exp471_antigravity(artifact.payload["support_authority"])
        != canonical_json_bytes_exp471_antigravity(expected_support)
    ):
        raise ValueError("Exp471 evaluation durable authority differs")


def _artifact_from_bytes_antigravity(
    raw: bytes,
) -> Exp471EvaluationArtifactAntigravity:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {constant}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("Exp471 evaluation JSON is malformed") from error
    required = {
        "source_authority_sha256",
        "cache_manifest_sha256",
        "code_closure_sha256",
        "method_order",
        "dataset_order",
        "payload",
        "evaluation_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Exp471 evaluation artifact schema differs")
    try:
        artifact = Exp471EvaluationArtifactAntigravity(**value)
    except (TypeError, ValueError) as error:
        raise ValueError("Exp471 evaluation artifact fields are invalid") from error
    if raw != _canonical_artifact_bytes_antigravity(artifact):
        raise ValueError("Exp471 evaluation artifact bytes are noncanonical")
    return artifact


def _publish_artifact_antigravity(
    root: Path, artifact: Exp471EvaluationArtifactAntigravity
) -> None:
    payload = _canonical_artifact_bytes_antigravity(artifact)
    anchor = cache_artifacts_antigravity._open_absolute_directory_antigravity(
        Path(root), create=True
    )
    try:
        cache_artifacts_antigravity._publish_bytes_at_antigravity(
            anchor.descriptor,
            _EVALUATION_NAME_ANTIGRAVITY,
            payload,
            temporary_scope="evaluation_artifact",
        )
        anchor.validate()
    finally:
        anchor.close()


def evaluate_exp471_antigravity(
    session: Exp471SourceSessionAntigravity,
    cache: Exp471ScoreCacheManifestAntigravity,
    *,
    label_loader: Callable | None = None,
) -> Exp471EvaluationArtifactAntigravity:
    """Verify complete score authority, open labels once, and publish metrics."""

    root = Path(EXP471_EVALUATION_ROOT_ANTIGRAVITY)
    if os.path.lexists(root / _EVALUATION_NAME_ANTIGRAVITY):
        return read_evaluation_exp471_antigravity(
            root, session, cache, label_loader=label_loader
        )

    typed = _verified_cache_scores_antigravity(session, cache)

    labels = _load_labels_once_antigravity(
        session,
        typed.cache.dataset_order,
        typed.test_counts,
        typed.test_indices,
        label_loader,
    )
    artifact = _expected_artifact_antigravity(session.authority, typed, labels)
    _validate_artifact_integrity_antigravity(artifact)
    _validate_artifact_authority_antigravity(
        artifact, session, typed, expected=artifact
    )
    _publish_artifact_antigravity(root, artifact)
    anchor = cache_artifacts_antigravity._open_absolute_directory_antigravity(
        root, create=False
    )
    try:
        raw = cache_artifacts_antigravity._read_leaf_at_antigravity(
            anchor.descriptor, _EVALUATION_NAME_ANTIGRAVITY
        )
        if raw is None or raw != _canonical_artifact_bytes_antigravity(artifact):
            raise ValueError("Exp471 evaluation published bytes differ")
        reopened = _artifact_from_bytes_antigravity(raw)
        anchor.validate()
    finally:
        anchor.close()
    _validate_artifact_integrity_antigravity(reopened)
    return reopened


def read_evaluation_exp471_antigravity(
    root: Path,
    session: Exp471SourceSessionAntigravity,
    cache: Exp471ScoreCacheManifestAntigravity,
    *,
    label_loader: Callable | None = None,
) -> Exp471EvaluationArtifactAntigravity:
    """Reopen by recomputing the expected artifact from cache scores and labels."""

    typed = _verified_cache_scores_antigravity(session, cache)
    labels = _load_labels_once_antigravity(
        session,
        typed.cache.dataset_order,
        typed.test_counts,
        typed.test_indices,
        label_loader,
    )
    expected = _expected_artifact_antigravity(session.authority, typed, labels)
    anchor = cache_artifacts_antigravity._open_absolute_directory_antigravity(
        Path(root), create=False
    )
    try:
        raw = cache_artifacts_antigravity._read_leaf_at_antigravity(
            anchor.descriptor, _EVALUATION_NAME_ANTIGRAVITY
        )
        if raw is None:
            raise ValueError("Exp471 evaluation artifact is missing")
        artifact = _artifact_from_bytes_antigravity(raw)
        anchor.validate()
    finally:
        anchor.close()
    _validate_artifact_authority_antigravity(
        artifact, session, typed, expected=expected
    )
    return artifact


def reconstruct_frozen_evaluation_exp471_antigravity(
    artifact: Exp471EvaluationArtifactAntigravity,
) -> Exp450EvaluationAntigravity:
    """Reconstruct the exact in-memory Exp450 metrics and curve pairings."""

    if not isinstance(artifact, Exp471EvaluationArtifactAntigravity):
        raise ValueError("Exp471 evaluation artifact has the wrong type")
    metrics, curves = _validate_artifact_integrity_antigravity(artifact)
    manifests = MappingProxyType(
        {
            "exp471_source_authority": artifact.payload["source_authority"],
            "exp471_score_cache": artifact.payload["score_cache"],
            "exp471_implementation": artifact.payload["implementation"],
        }
    )
    return Exp450EvaluationAntigravity(
        manifests=manifests,
        metric_tables=metrics,
        curve_tables=curves,
    )


__all__ = [
    "artifact_view_exp471_antigravity",
    "evaluate_exp471_antigravity",
    "read_evaluation_exp471_antigravity",
    "reconstruct_frozen_evaluation_exp471_antigravity",
]
