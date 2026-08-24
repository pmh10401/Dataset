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


_EVALUATION_NAME_ANTIGRAVITY = "evaluation_antigravity.json"
_EVALUATION_SCHEMA_ANTIGRAVITY = "exp471-evaluation-antigravity.v1"
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


def _verified_cache_shards_antigravity(
    session: Exp471SourceSessionAntigravity,
    cache: object,
    *,
    prepare_scores: bool,
) -> tuple[
    Exp471ScoreCacheManifestAntigravity,
    dict[str, dict[str, tuple[float, ...]]] | None,
    dict[str, dict[str, tuple[float, ...]]] | None,
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

    raw_scores: dict[str, dict[str, tuple[float, ...]]] | None = (
        {} if prepare_scores else None
    )
    micro_scores: dict[str, dict[str, tuple[float, ...]]] | None = (
        {} if prepare_scores else None
    )
    test_counts: dict[str, int] = {}
    for method in METHOD_ORDER_ANTIGRAVITY:
        if prepare_scores:
            assert raw_scores is not None and micro_scores is not None
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
            if prepare_scores:
                assert raw_scores is not None and micro_scores is not None
                raw_scores[method][dataset] = shard.raw_scores
                micro_scores[method][dataset] = shard.micro_scores
    return reopened, raw_scores, micro_scores, test_counts


def _verified_cache_scores_antigravity(
    session: Exp471SourceSessionAntigravity,
    cache: object,
) -> tuple[
    Exp471ScoreCacheManifestAntigravity,
    dict[str, dict[str, tuple[float, ...]]],
    dict[str, dict[str, tuple[float, ...]]],
    dict[str, int],
]:
    reopened, raw_scores, micro_scores, test_counts = (
        _verified_cache_shards_antigravity(
            session, cache, prepare_scores=True
        )
    )
    assert raw_scores is not None and micro_scores is not None
    return reopened, raw_scores, micro_scores, test_counts


def _verified_cache_counts_antigravity(
    session: Exp471SourceSessionAntigravity,
    cache: object,
) -> tuple[Exp471ScoreCacheManifestAntigravity, dict[str, int]]:
    reopened, raw_scores, micro_scores, test_counts = (
        _verified_cache_shards_antigravity(
            session, cache, prepare_scores=False
        )
    )
    assert raw_scores is None and micro_scores is None
    return reopened, test_counts


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
        "dataset_test_count": dict(test_counts),
    }


def _payload_antigravity(
    authority: Exp471SourceAuthorityAntigravity,
    cache: Exp471ScoreCacheManifestAntigravity,
    test_counts: Mapping[str, int],
    metrics: MetricTablesExp450Antigravity,
    curves: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    return {
        "schema": _EVALUATION_SCHEMA_ANTIGRAVITY,
        "source_authority": _source_record_antigravity(authority),
        "score_cache": _cache_record_antigravity(cache, test_counts),
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
    test_counts: Mapping[str, int],
    metrics: MetricTablesExp450Antigravity,
    curves: Mapping[str, Mapping[str, object]],
) -> Exp471EvaluationArtifactAntigravity:
    raw_bare = {
        "source_authority_sha256": authority.authority_sha256,
        "cache_manifest_sha256": cache.manifest_sha256,
        "code_closure_sha256": _code_closure_sha256_antigravity(),
        "method_order": list(cache.method_order),
        "dataset_order": list(cache.dataset_order),
        "payload": _payload_antigravity(
            authority, cache, test_counts, metrics, curves
        ),
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
            "dataset_test_count",
        }
        or not isinstance(cache.get("dataset_test_count"), Mapping)
        or type(cache.get("cache_context_sha256")) is not str
        or _SHA256_ANTIGRAVITY.fullmatch(cache["cache_context_sha256"]) is None
        or type(cache.get("total_shards")) is not int
        or cache["total_shards"]
        != len(METHOD_ORDER_ANTIGRAVITY) * len(artifact.dataset_order)
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
    metrics = _metric_tables_antigravity(
        payload["metric_tables"], artifact.dataset_order, test_counts
    )
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
    cache: Exp471ScoreCacheManifestAntigravity,
    test_counts: Mapping[str, int],
) -> None:
    _validate_artifact_integrity_antigravity(artifact)
    if (
        artifact.source_authority_sha256 != session.authority.authority_sha256
        or artifact.cache_manifest_sha256 != cache.manifest_sha256
        or artifact.method_order != cache.method_order
        or artifact.dataset_order != cache.dataset_order
        or artifact.payload["source_authority"]
        != _source_record_antigravity(session.authority)
        or artifact.payload["score_cache"]
        != _cache_record_antigravity(cache, test_counts)
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
        return read_evaluation_exp471_antigravity(root, session, cache)

    reopened, raw_scores, micro_scores, test_counts = (
        _verified_cache_scores_antigravity(session, cache)
    )

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
        session.authority, reopened, test_counts, metrics, curves
    )
    _validate_artifact_authority_antigravity(
        artifact, session, reopened, test_counts
    )
    _publish_artifact_antigravity(root, artifact)
    return read_evaluation_exp471_antigravity(root, session, reopened)


def read_evaluation_exp471_antigravity(
    root: Path,
    session: Exp471SourceSessionAntigravity,
    cache: Exp471ScoreCacheManifestAntigravity,
) -> Exp471EvaluationArtifactAntigravity:
    """Reopen and fully validate one durable evaluation without recomputation."""

    reopened, counts = _verified_cache_counts_antigravity(session, cache)
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
        artifact, session, reopened, counts
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
