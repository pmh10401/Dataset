"""Label-gated, create-only persistence for the frozen Exp471 evaluation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
from types import MappingProxyType

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


_EVALUATION_NAME_ANTIGRAVITY = "evaluation_antigravity.json"
_EVALUATION_SCHEMA_ANTIGRAVITY = "exp471-evaluation-antigravity.v1"
_SEMANTIC_NONE_SCHEMA_ANTIGRAVITY = "exp471-semantic-none-antigravity.v1"


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


def _code_closure_sha256_antigravity() -> str:
    module_path = Path(frozen_evaluation_antigravity.__file__).resolve()
    return sha256(module_path.read_bytes()).hexdigest()


def _dependency_version_antigravity(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _implementation_record_antigravity() -> dict[str, object]:
    return {
        "module": (
            "experiments.exp450_resampling_image_ablation_antigravity."
            "evaluation_antigravity"
        ),
        "metric_function": "metric_tables_exp450_antigravity",
        "curve_function": "curve_tables_exp450_antigravity",
        "code_closure_sha256": _code_closure_sha256_antigravity(),
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


def _verified_cache_scores_antigravity(
    session: Exp471SourceSessionAntigravity,
    cache: object,
) -> tuple[
    Exp471ScoreCacheManifestAntigravity,
    dict[str, dict[str, tuple[float, ...]]],
    dict[str, dict[str, tuple[float, ...]]],
    dict[str, int],
]:
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

    raw_scores: dict[str, dict[str, tuple[float, ...]]] = {}
    micro_scores: dict[str, dict[str, tuple[float, ...]]] = {}
    test_counts: dict[str, int] = {}
    for method in METHOD_ORDER_ANTIGRAVITY:
        raw_scores[method] = {}
        micro_scores[method] = {}
        for dataset in reopened.dataset_order:
            shard = cache_artifacts_antigravity.read_score_shard_antigravity(
                root, method, dataset
            )
            key = f"{method}:{dataset}"
            if (
                shard.source_authority_sha256
                != session.authority.authority_sha256
                or shard.cache_context_sha256 != reopened.cache_context_sha256
                or shard.envelope_sha256 != reopened.shard_sha256[key]
                or shard.method != method
                or shard.dataset != dataset
            ):
                raise ValueError("Exp471 evaluation score shard authority differs")
            prior_count = test_counts.setdefault(dataset, shard.test_count)
            if prior_count != shard.test_count:
                raise ValueError("Exp471 evaluation dataset row counts differ")
            raw_scores[method][dataset] = shard.raw_scores
            micro_scores[method][dataset] = shard.micro_scores
    return reopened, raw_scores, micro_scores, test_counts


def _normalize_labels_antigravity(
    value: object,
    datasets: tuple[str, ...],
    test_counts: Mapping[str, int],
) -> dict[str, tuple[int, ...]]:
    if not isinstance(value, Mapping) or tuple(value) != datasets:
        raise ValueError("Exp471 evaluation label dataset order differs")
    normalized: dict[str, tuple[int, ...]] = {}
    for dataset in datasets:
        rows = tuple(value[dataset])
        if rows and all(isinstance(row, tuple) and len(row) == 2 for row in rows):
            indices, labels = zip(*rows, strict=True)
            if indices != tuple(range(test_counts[dataset])):
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
    return _normalize_labels_antigravity(raw, datasets, test_counts)


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
) -> dict[str, object]:
    return {
        "cache_context_sha256": cache.cache_context_sha256,
        "manifest_sha256": cache.manifest_sha256,
        "total_shards": cache.total_shards,
    }


def _payload_antigravity(
    authority: Exp471SourceAuthorityAntigravity,
    cache: Exp471ScoreCacheManifestAntigravity,
    metrics: MetricTablesExp450Antigravity,
    curves: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    return {
        "schema": _EVALUATION_SCHEMA_ANTIGRAVITY,
        "source_authority": _source_record_antigravity(authority),
        "score_cache": _cache_record_antigravity(cache),
        "implementation": _implementation_record_antigravity(),
        "metric_tables": _encode_semantic_none_antigravity(
            _metric_record_antigravity(metrics)
        ),
        "curve_tables": curves,
    }


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


def _new_artifact_antigravity(
    authority: Exp471SourceAuthorityAntigravity,
    cache: Exp471ScoreCacheManifestAntigravity,
    metrics: MetricTablesExp450Antigravity,
    curves: Mapping[str, Mapping[str, object]],
) -> Exp471EvaluationArtifactAntigravity:
    raw_bare = {
        "source_authority_sha256": authority.authority_sha256,
        "cache_manifest_sha256": cache.manifest_sha256,
        "code_closure_sha256": _code_closure_sha256_antigravity(),
        "method_order": list(cache.method_order),
        "dataset_order": list(cache.dataset_order),
        "payload": _payload_antigravity(authority, cache, metrics, curves),
    }
    bare = artifact_view_exp471_antigravity(raw_bare)
    if not isinstance(bare, Mapping):
        raise ValueError("Exp471 complete evaluation artifact view is invalid")
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
                    "false_positive_rate": _finite_number_antigravity(
                        raw_row["false_positive_rate"], "ROC false-positive rate"
                    ),
                    "true_positive_rate": _finite_number_antigravity(
                        raw_row["true_positive_rate"], "ROC true-positive rate"
                    ),
                    "threshold": threshold_value,
                }
            )
        )
    if _method_blocks_antigravity(roc_rows, "ROC") != METHOD_ORDER_ANTIGRAVITY:
        raise ValueError("Exp471 evaluation ROC method order differs")

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
                    "recall": _finite_number_antigravity(
                        raw_row["recall"], "precision-recall recall"
                    ),
                    "precision": _finite_number_antigravity(
                        raw_row["precision"], "precision-recall precision"
                    ),
                }
            )
        )
    if _method_blocks_antigravity(
        pr_rows, "precision-recall"
    ) != METHOD_ORDER_ANTIGRAVITY:
        raise ValueError("Exp471 evaluation precision-recall method order differs")

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


def _metric_tables_antigravity(value: object) -> MetricTablesExp450Antigravity:
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
    return MetricTablesExp450Antigravity(
        method_order=METHOD_ORDER_ANTIGRAVITY,
        per_dataset=_freeze_mapping_rows_antigravity(
            decoded["per_dataset"], "per-dataset metrics"
        ),
        macro=_freeze_method_mapping_antigravity(decoded["macro"], "macro metrics"),
        micro=_freeze_method_mapping_antigravity(decoded["micro"], "micro metrics"),
        paired_statistics=_freeze_mapping_rows_antigravity(
            decoded["paired_statistics"], "paired statistics"
        ),
    )


def _validated_payload_antigravity(
    artifact: Exp471EvaluationArtifactAntigravity,
) -> tuple[MetricTablesExp450Antigravity, Mapping[str, Mapping[str, object]]]:
    payload = artifact.payload
    if set(payload) != {
        "schema",
        "source_authority",
        "score_cache",
        "implementation",
        "metric_tables",
        "curve_tables",
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
        or source.get("authority_sha256") != artifact.source_authority_sha256
        or not isinstance(cache, Mapping)
        or cache.get("manifest_sha256") != artifact.cache_manifest_sha256
    ):
        raise ValueError("Exp471 evaluation payload authority binding differs")
    metrics = _metric_tables_antigravity(payload["metric_tables"])
    curves = _curve_tables_antigravity(payload["curve_tables"])
    if canonical_json_bytes_exp471_antigravity(
        _encode_semantic_none_antigravity(_metric_record_antigravity(metrics))
    ) != canonical_json_bytes_exp471_antigravity(payload["metric_tables"]):
        raise ValueError("Exp471 evaluation metric round trip differs")
    if canonical_json_bytes_exp471_antigravity(
        artifact_view_exp471_antigravity(curves)
    ) != canonical_json_bytes_exp471_antigravity(payload["curve_tables"]):
        raise ValueError("Exp471 evaluation curve coordinate round trip differs")
    return metrics, curves


def _validate_artifact_integrity_antigravity(
    artifact: Exp471EvaluationArtifactAntigravity,
) -> None:
    _canonical_artifact_bytes_antigravity(artifact)
    if (
        artifact.code_closure_sha256 != _code_closure_sha256_antigravity()
        or artifact.method_order != METHOD_ORDER_ANTIGRAVITY
    ):
        raise ValueError("Exp471 evaluation frozen implementation binding differs")
    _validated_payload_antigravity(artifact)


def _validate_artifact_authority_antigravity(
    artifact: Exp471EvaluationArtifactAntigravity,
    session: Exp471SourceSessionAntigravity,
    cache: Exp471ScoreCacheManifestAntigravity,
) -> None:
    _validate_artifact_integrity_antigravity(artifact)
    if (
        artifact.source_authority_sha256 != session.authority.authority_sha256
        or artifact.cache_manifest_sha256 != cache.manifest_sha256
        or artifact.method_order != cache.method_order
        or artifact.dataset_order != cache.dataset_order
        or artifact.payload["source_authority"]
        != _source_record_antigravity(session.authority)
        or artifact.payload["score_cache"] != _cache_record_antigravity(cache)
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

    reopened, raw_scores, micro_scores, test_counts = (
        _verified_cache_scores_antigravity(session, cache)
    )
    root = Path(EXP471_EVALUATION_ROOT_ANTIGRAVITY)
    if os.path.lexists(root / _EVALUATION_NAME_ANTIGRAVITY):
        return read_evaluation_exp471_antigravity(root, session, reopened)

    labels = _load_labels_once_antigravity(
        session,
        reopened.dataset_order,
        test_counts,
        label_loader,
    )
    metrics = metric_tables_exp450_antigravity(
        labels, raw_scores, micro_scores=micro_scores
    )
    curves = curve_tables_exp450_antigravity(labels, micro_scores)
    artifact = _new_artifact_antigravity(
        session.authority, reopened, metrics, curves
    )
    _validate_artifact_authority_antigravity(
        artifact, session, reopened
    )
    _publish_artifact_antigravity(root, artifact)
    return read_evaluation_exp471_antigravity(root, session, reopened)


def read_evaluation_exp471_antigravity(
    root: Path,
    session: Exp471SourceSessionAntigravity,
    cache: Exp471ScoreCacheManifestAntigravity,
) -> Exp471EvaluationArtifactAntigravity:
    """Reopen and fully validate one durable evaluation without recomputation."""

    reopened, _raw, _micro, _counts = _verified_cache_scores_antigravity(
        session, cache
    )
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
    _validate_artifact_authority_antigravity(artifact, session, reopened)
    return artifact


def reconstruct_frozen_evaluation_exp471_antigravity(
    artifact: Exp471EvaluationArtifactAntigravity,
) -> Exp450EvaluationAntigravity:
    """Reconstruct the exact in-memory Exp450 metrics and curve pairings."""

    if not isinstance(artifact, Exp471EvaluationArtifactAntigravity):
        raise ValueError("Exp471 evaluation artifact has the wrong type")
    _validate_artifact_integrity_antigravity(artifact)
    metrics, curves = _validated_payload_antigravity(artifact)
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
