"""Immutable adapters for frozen Exp403 feature-family artifacts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from .artifact_io import sha256_file
from .contracts import SEED, DatasetInput, FeatureBlock, _is_forbidden_key


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
FEATURE_GROUPS = {
    "temporal_random_kernel",
    "normal_space_distance",
    "change_delta",
    "frequency_multiscale",
    "shape_complexity",
    "process_grammar",
}


@dataclass(frozen=True)
class FeatureSource:
    """The immutable identity and layout of one frozen feature source."""

    name: str
    group: str
    kind: str
    root: Path
    train_key: str
    test_key: str
    seed_values: tuple[tuple[str, object], ...] = ()
    reduction: str = "none"
    prediction_name: str = "exp393_predictions.jsonl"
    manifest_name: str = "exp393_prediction_manifest.json"

    def __post_init__(self) -> None:
        if self.group not in FEATURE_GROUPS:
            raise ValueError(f"unknown feature group: {self.group}")
        if self.kind not in {"cache", "grammar"}:
            raise ValueError(f"unknown feature source kind: {self.kind}")
        if self.reduction not in {"none", "median_axis0"}:
            raise ValueError(f"unknown feature reduction: {self.reduction}")


def _cache_source(
    name: str,
    group: str,
    root: str,
    train_key: str,
    test_key: str,
    *,
    seed_values: tuple[tuple[str, object], ...] = (),
    reduction: str = "none",
) -> FeatureSource:
    return FeatureSource(
        name=name,
        group=group,
        kind="cache",
        root=DATA_ROOT / root,
        train_key=train_key,
        test_key=test_key,
        seed_values=seed_values,
        reduction=reduction,
    )


FEATURE_SOURCES = (
    _cache_source(
        "rocket256",
        "temporal_random_kernel",
        "exp238_rocket_dimension_confidence",
        "rocket_256__train",
        "rocket_256__test",
        seed_values=(("base_seed", 20260706), ("stability_seed", 20261706)),
    ),
    _cache_source(
        "rocket512",
        "temporal_random_kernel",
        "exp238_rocket_dimension_confidence",
        "rocket_512__train",
        "rocket_512__test",
        seed_values=(("base_seed", 20260706), ("stability_seed", 20261706)),
    ),
    _cache_source(
        "rocket1024",
        "temporal_random_kernel",
        "exp238_rocket_dimension_confidence",
        "rocket_1024__train",
        "rocket_1024__test",
        seed_values=(("base_seed", 20260706), ("stability_seed", 20261706)),
    ),
    _cache_source(
        "minirocket1024",
        "temporal_random_kernel",
        "exp249_250_official_mini_multirocket/experiment_249_official_minirocket",
        "kernels_1024__train_by_seed",
        "kernels_1024__test_by_seed",
        seed_values=(("config.seeds", (20260721, 20261721)),),
        reduction="median_axis0",
    ),
    _cache_source(
        "normal_density",
        "normal_space_distance",
        "exp239_248_diverse_score_sources/experiment_248_nonlinear_normal_density",
        "train_oof",
        "test_scores",
        seed_values=(("config.seed", 20260720),),
    ),
    _cache_source(
        "change_point_combined",
        "change_delta",
        "exp255_change_point_features",
        "change_point_combined_33__train_oof",
        "change_point_combined_33__test_scores",
    ),
    _cache_source(
        "wavelet_multiscale",
        "frequency_multiscale",
        "exp239_248_diverse_score_sources/experiment_239_multiscale_wavelet_energy_deviation",
        "train_oof",
        "test_scores",
        seed_values=(("config.seed", 20260720),),
    ),
    _cache_source(
        "shape_complexity_combined",
        "shape_complexity",
        "exp252_sktime_statistical_features_safe_hurst",
        "combined_40__train_oof",
        "combined_40__test_scores",
        seed_values=(("config.seed", 20260722),),
    ),
    FeatureSource(
        name="grammar_composite",
        group="process_grammar",
        kind="grammar",
        root=DATA_ROOT / "exp393_396_normal_process_grammar",
        train_key="composite",
        test_key="composite",
        seed_values=(("config.random_seed", SEED),),
    ),
)

_SOURCE_BY_NAME = {source.name: source for source in FEATURE_SOURCES}
_LABEL_DERIVED_EXACT = frozenset(
    {
        "y_test",
        "test_label",
        "test_labels",
        "true_label",
        "true_anomaly_count",
        "actual_anomaly_positions",
        "evaluation_metrics",
        "hard_metrics",
    }
)
_LABEL_DERIVED_SUFFIXES = ("_tp", "_fp", "_fn", "_f1", "_precision", "_recall", "_auc")
_ALLOWED_DECLARATIONS = frozenset(
    {
        "contains_test_labels",
        "contains_evaluation_metrics",
        "test_metrics_select_policy",
    }
)
_TEST_EVALUATION_TERMS = frozenset(
    {"label", "labels", "accuracy", "auc", "precision", "recall", "f1", "metric", "metrics"}
)
_LABEL_METRIC_TOKENS = frozenset(
    {"label", "labels", "tp", "fp", "fn", "f1", "accuracy", "auc", "precision", "recall"}
)


def registry_for_names(names: Iterable[str]) -> tuple[FeatureBlock, ...]:
    """Return physical-group records without materializing source candidates."""
    blocks: list[FeatureBlock] = []
    for name in names:
        source = _SOURCE_BY_NAME.get(str(name))
        if source is None:
            raise ValueError(f"unknown feature source: {name}")
        blocks.append(_unavailable(source, "source_not_loaded"))
    return tuple(blocks)


def missing_feature_block(name: str) -> FeatureBlock:
    """Represent a missing artifact explicitly, never as empty score vectors."""
    source = _SOURCE_BY_NAME.get(str(name))
    if source is None:
        aliases = {
            "wavelet": "wavelet_multiscale",
            "grammar": "grammar_composite",
            "change_point": "change_point_combined",
        }
        source = _SOURCE_BY_NAME.get(aliases.get(str(name), ""))
    if source is None:
        raise ValueError(f"unknown feature source: {name}")
    return _unavailable(source, "source_artifact_missing")


def load_feature_blocks(
    dataset: str,
    bundle: DatasetInput,
    *,
    sources: Sequence[FeatureSource] = FEATURE_SOURCES,
) -> tuple[FeatureBlock, ...]:
    """Load only hash-verified, label-free frozen feature blocks for one dataset."""
    if str(dataset) != str(bundle.dataset):
        raise ValueError("dataset and DatasetInput.dataset must match")
    return tuple(_load_source(source, str(dataset), bundle) for source in sources)


def _load_source(source: FeatureSource, dataset: str, bundle: DatasetInput) -> FeatureBlock:
    try:
        if source.kind == "cache":
            return _load_cache_source(source, dataset, bundle)
        return _load_grammar_source(source, dataset, bundle)
    except FileNotFoundError:
        return _unavailable(source, "source_artifact_missing")
    except _UnavailableSource as error:
        return _unavailable(source, error.reason)
    except (OSError, ValueError, json.JSONDecodeError):
        return _unavailable(source, "source_artifact_invalid")


def _load_cache_source(
    source: FeatureSource, dataset: str, bundle: DatasetInput
) -> FeatureBlock:
    stem = hashlib.sha256(dataset.encode("utf-8")).hexdigest()[:20]
    archive_path = source.root / "cache" / f"{stem}.npz"
    metadata_path = source.root / "cache" / f"{stem}.json"
    if not archive_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(archive_path)
    metadata = _load_json(metadata_path)
    _verify_common_contract(metadata, dataset, bundle, source)
    expected_hash = metadata.get("cache_sha256")
    if not isinstance(expected_hash, str) or sha256_file(archive_path) != expected_hash:
        raise _UnavailableSource("source_hash_mismatch")
    if metadata.get("abstained") is True:
        raise _UnavailableSource("source_abstained")
    try:
        with archive_path.open("rb") as handle:
            with np.load(handle, allow_pickle=False) as archive:
                if any(
                    _is_forbidden_key(member) or _is_label_derived_key(member)
                    for member in archive.files
                ):
                    raise _UnavailableSource("source_label_derived")
                if source.train_key not in archive or source.test_key not in archive:
                    raise _UnavailableSource("source_feature_missing")
                train = _reduce(np.asarray(archive[source.train_key]), source.reduction)
                test = _reduce(np.asarray(archive[source.test_key]), source.reduction)
    except _UnavailableSource:
        raise
    except Exception as error:
        raise _UnavailableSource("source_artifact_invalid") from error
    return _available(source, expected_hash, train, test, bundle)


def _load_grammar_source(
    source: FeatureSource, dataset: str, bundle: DatasetInput
) -> FeatureBlock:
    prediction_path = source.root / source.prediction_name
    manifest_path = source.root / source.manifest_name
    if not prediction_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(prediction_path)
    manifest = _load_json(manifest_path)
    _verify_label_free(manifest)
    _verify_seeds(manifest, source.seed_values)
    expected_hash = manifest.get("prediction_sha256")
    if not isinstance(expected_hash, str) or sha256_file(prediction_path) != expected_hash:
        raise _UnavailableSource("source_hash_mismatch")
    row = _find_grammar_row(prediction_path, dataset)
    _verify_common_contract(
        row,
        dataset,
        bundle,
        source,
        train_field="n_train_normal",
        verify_seeds=False,
    )
    if row.get("abstained") is True:
        raise _UnavailableSource("source_abstained")
    train_scores = row.get("train_oof_scores")
    test_scores = row.get("scores")
    if not isinstance(train_scores, Mapping) or not isinstance(test_scores, Mapping):
        raise _UnavailableSource("source_feature_missing")
    if source.train_key not in train_scores or source.test_key not in test_scores:
        raise _UnavailableSource("source_feature_missing")
    return _available(
        source,
        expected_hash,
        np.asarray(train_scores[source.train_key]),
        np.asarray(test_scores[source.test_key]),
        bundle,
    )


def _find_grammar_row(prediction_path: Path, dataset: str) -> Mapping[str, Any]:
    found: Mapping[str, Any] | None = None
    for line in prediction_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("dataset", "")) != dataset:
            continue
        if found is not None:
            raise _UnavailableSource("source_dataset_duplicate")
        found = row
    if found is None:
        raise _UnavailableSource("source_dataset_missing")
    return found


def _verify_common_contract(
    metadata: Mapping[str, Any],
    dataset: str,
    bundle: DatasetInput,
    source: FeatureSource,
    *,
    train_field: str = "n_train",
    verify_seeds: bool = True,
) -> None:
    _verify_label_free(metadata)
    if metadata.get("dataset") != dataset:
        raise _UnavailableSource("source_dataset_mismatch")
    if _exact_count(metadata.get(train_field)) != len(bundle.train_normal):
        raise _UnavailableSource("source_train_count_mismatch")
    if _exact_count(metadata.get("n_test")) != len(bundle.test_values):
        raise _UnavailableSource("source_test_count_mismatch")
    if verify_seeds:
        _verify_seeds(metadata, source.seed_values)


def _verify_label_free(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in _ALLOWED_DECLARATIONS:
                if child is not False:
                    raise _UnavailableSource("source_label_derived")
            elif _is_label_derived_key(normalized):
                raise _UnavailableSource("source_label_derived")
            _verify_label_free(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _verify_label_free(child)


def _is_label_derived_key(key: str) -> bool:
    if key in _LABEL_DERIVED_EXACT or key.endswith(_LABEL_DERIVED_SUFFIXES):
        return True
    tokens = set(re.findall(r"[a-z0-9]+", key))
    return bool(tokens & _LABEL_METRIC_TOKENS) or (
        "test" in tokens and bool(tokens & _TEST_EVALUATION_TERMS)
    )


def _verify_seeds(
    metadata: Mapping[str, Any], seeds: Sequence[tuple[str, object]]
) -> None:
    for path, expected in seeds:
        current: object = metadata
        for key in path.split("."):
            if not isinstance(current, Mapping) or key not in current:
                raise _UnavailableSource("source_seed_mismatch")
            current = current[key]
        if isinstance(current, bool) or _canonical_seed_value(current) != _canonical_seed_value(expected):
            raise _UnavailableSource("source_seed_mismatch")


def _canonical_seed_value(value: object) -> object:
    if isinstance(value, list):
        return tuple(_canonical_seed_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_canonical_seed_value(item) for item in value)
    return value


def _available(
    source: FeatureSource,
    source_hash: str,
    train: np.ndarray,
    test: np.ndarray,
    bundle: DatasetInput,
) -> FeatureBlock:
    train_values = _readonly_vector(train, len(bundle.train_normal), "source_train_count_mismatch")
    test_values = _readonly_vector(test, len(bundle.test_values), "source_test_count_mismatch")
    if not np.isfinite(train_values).all() or not np.isfinite(test_values).all():
        raise _UnavailableSource("source_non_finite")
    return FeatureBlock(
        name=source.name,
        group=source.group,
        train_oof=train_values,
        test_values=test_values,
        source_hash=source_hash,
    )


def _reduce(values: np.ndarray, reduction: str) -> np.ndarray:
    if reduction == "median_axis0":
        if values.ndim != 2:
            raise _UnavailableSource("source_feature_shape_invalid")
        return np.median(values, axis=0)
    return values


def _readonly_vector(values: np.ndarray, size: int, reason: str) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0 or vector.size != size:
        raise _UnavailableSource(reason)
    vector = np.ascontiguousarray(vector)
    vector.setflags(write=False)
    return vector


def _exact_count(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if str(count) == str(value) or isinstance(value, int) else None


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise _UnavailableSource("source_metadata_invalid")
    return value


def _unavailable(source: FeatureSource, reason: str) -> FeatureBlock:
    identity = hashlib.sha256(
        f"unavailable:{source.name}:{source.group}:{reason}".encode("utf-8")
    ).hexdigest()
    return FeatureBlock(
        name=source.name,
        group=source.group,
        train_oof=None,
        test_values=None,
        source_hash=identity,
        available=False,
        reason=reason,
    )


class _UnavailableSource(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
