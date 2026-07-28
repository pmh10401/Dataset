"""Representative-smoke-only orchestration for Exp409 readiness."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import hashlib
from pathlib import Path
import shutil
import sqlite3
import uuid

from experiments.exp403_408.artifact_io import (
    canonical_json_bytes,
    sha256_file,
)
from experiments.exp403_408.consensus import build_consensus
from experiments.exp403_408.feature_registry import FEATURE_SOURCES
from experiments.exp403_408.foundation import (
    FrozenLaneUniverse,
    list_datasets_by_train_normal_count,
    load_dataset_input,
    load_frozen_lanes,
)
from experiments.exp403_408.paths import (
    DB_PATH,
    EXP291_MANIFEST_PATH,
    EXP291_PREDICTION_PATH,
    EXP292_MANIFEST_PATH,
    EXP292_PREDICTION_PATH,
)
from experiments.exp403_408.pipeline import (
    select_representative_smoke_datasets,
)
from experiments.exp403_408.synthetic_anomaly import MODEL_NAMES

from .config import FROZEN_CONFIG, Exp409FrozenConfig
from .contracts import (
    DatasetReadinessResult,
    ReadinessBatch,
    ShadowCandidate,
    validate_candidate,
)
from .guard import LabelBlindSqlTrace, load_values_bundle_label_blind
from .provenance import (
    build_canonical_provenance,
    build_runtime_envelope,
    dataset_values_sha256,
    require_clean_git,
)
from .publication import candidate_payload_sha256, publish_batch
from .sources import SourceBatch, build_source_batch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SMOKE_TIERS = frozenset(
    {"abstain", "robust_distance", "classical_committee", "learned_methods"}
)


def run_dataset(
    dataset: str,
    *,
    config: Exp409FrozenConfig = FROZEN_CONFIG,
) -> DatasetReadinessResult:
    """Run one dataset through the frozen, label-blind readiness path."""
    universe = load_frozen_lanes()
    result, _bundle = _run_dataset_with_bundle(
        str(dataset), universe=universe, config=config
    )
    return result


def _run_dataset_with_bundle(
    dataset: str,
    *,
    universe: FrozenLaneUniverse,
    config: Exp409FrozenConfig,
) -> tuple[DatasetReadinessResult, object]:
    if config != FROZEN_CONFIG:
        raise ValueError("Exp409 runner configuration is not frozen")
    sql_trace = LabelBlindSqlTrace()
    bundle = load_dataset_input(
        dataset,
        loader=partial(load_values_bundle_label_blind, trace=sql_trace),
        frozen_lane_universe=universe,
    )
    sql_trace.assert_clean()
    source_batch = build_source_batch(dataset, bundle, config)
    _validate_source_batch(source_batch)

    original_lanes = _bundle_lanes(bundle)
    if not source_batch.sources:
        if any(status.status != "abstained" for status in source_batch.statuses):
            raise ValueError("Exp409 empty source batch is inconsistent")
        return (
            DatasetReadinessResult(
                dataset=dataset,
                train_size=len(bundle.train_normal),
                test_size=len(bundle.test_values),
                hard_indices=original_lanes["hard"],
                standard_indices=original_lanes["standard"],
                priority_indices=original_lanes["priority"],
                source_statuses=source_batch.statuses,
                candidates=(),
                source_output_sha256=source_batch.output_sha256,
                status="abstained",
                abstention_reason="all_sources_abstained",
            ),
            bundle,
        )

    consensus = build_consensus(
        universe,
        source_batch.sources,
        threshold=config.independence_threshold,
        minimum_rows=config.minimum_oof_rows,
    )
    consensus_lanes = {
        "hard": tuple(sorted(int(value) for value in consensus.hard_indices)),
        "standard": tuple(
            sorted(int(value) for value in consensus.standard_indices)
        ),
        "priority": tuple(
            sorted(int(value) for value in consensus.priority_indices)
        ),
    }
    validate_lane_parity(original_lanes, consensus_lanes)
    try:
        frozen_rows = consensus.candidates_by_variant[config.variant][
            config.alpha
        ]
    except (KeyError, TypeError) as error:
        raise ValueError("Exp409 frozen consensus package is missing") from error

    candidates = tuple(
        ShadowCandidate(
            dataset=str(row.dataset),
            candidate_index=int(row.candidate_index),
            variant=str(row.variant),
            alpha=str(row.alpha),
            supporting_components=tuple(
                tuple(str(value) for value in component)
                for component in row.supporting_components
            ),
            supporting_source_markers=tuple(
                tuple(str(value) for value in component)
                for component in row.supporting_source_markers
            ),
        )
        for row in frozen_rows
    )
    for candidate in candidates:
        validate_candidate(
            candidate,
            n_test=len(bundle.test_values),
            hard_indices=original_lanes["hard"],
        )
    return (
        DatasetReadinessResult(
            dataset=dataset,
            train_size=len(bundle.train_normal),
            test_size=len(bundle.test_values),
            hard_indices=original_lanes["hard"],
            standard_indices=original_lanes["standard"],
            priority_indices=original_lanes["priority"],
            source_statuses=source_batch.statuses,
            candidates=candidates,
            source_output_sha256=source_batch.output_sha256,
            status="ready",
        ),
        bundle,
    )


def validate_lane_parity(
    original: Mapping[str, Sequence[int]],
    observed: Mapping[str, Sequence[int]],
) -> None:
    """Reject any change to Hard, Standard, or Priority membership."""
    expected_keys = {"hard", "standard", "priority"}
    if set(original) != expected_keys or set(observed) != expected_keys:
        raise ValueError("Exp409 lane parity mapping is invalid")
    normalized_original = {
        key: tuple(sorted({int(value) for value in original[key]}))
        for key in sorted(expected_keys)
    }
    normalized_observed = {
        key: tuple(sorted({int(value) for value in observed[key]}))
        for key in sorted(expected_keys)
    }
    if normalized_original != normalized_observed:
        raise ValueError("Exp409 lane parity gate failed")


def select_exp409_smoke_datasets(
    limit: int = 8,
) -> tuple[tuple[str, ...], Mapping[str, int]]:
    """Select only by canonical normal-TRAIN count and cover all four tiers."""
    size = int(limit)
    if isinstance(limit, bool) or not 4 <= size <= 20:
        raise ValueError("Exp409 representative smoke limit must be 4..20")
    datasets = tuple(list_datasets_by_train_normal_count())
    counts = _train_normal_counts()
    if set(datasets) - set(counts):
        raise ValueError("Exp409 TRAIN count coverage is incomplete")
    selected, coverage = select_representative_smoke_datasets(
        datasets,
        size,
        train_count_for_dataset=lambda dataset: counts[str(dataset)],
    )
    if set(coverage) != _SMOKE_TIERS or any(
        int(coverage[tier]) < 1 for tier in _SMOKE_TIERS
    ):
        raise ValueError("Exp409 representative smoke misses a TRAIN tier")
    return selected, dict(coverage)


def run_smoke(
    output_dir: Path,
    *,
    limit: int = 8,
) -> Mapping[str, str]:
    """Run the authorized representative smoke twice, then publish once."""
    output = Path(output_dir)
    if not output.parent.is_dir():
        raise FileNotFoundError("Exp409 output parent does not exist")
    if output.exists():
        raise FileExistsError("Exp409 output directory must not exist")
    git_commit = require_clean_git(
        _REPO_ROOT, git_executable=_resolve_git_executable()
    )
    datasets, _coverage = select_exp409_smoke_datasets(limit=limit)
    universe = load_frozen_lanes()

    first_batch, first_evidence = _assemble_smoke(datasets, universe)
    second_batch, second_evidence = _assemble_smoke(datasets, universe)
    _assert_deterministic(
        _determinism_payload(first_batch, first_evidence),
        _determinism_payload(second_batch, second_evidence),
    )

    source_output = _aggregate_source_output(first_batch)
    candidate_sha256 = candidate_payload_sha256(first_batch)
    canonical_provenance = build_canonical_provenance(
        git_commit=git_commit,
        code_sha256=_path_hashes(_code_paths()),
        database_snapshot_sha256={"univariate_ts.db": sha256_file(DB_PATH)},
        lane_artifact_sha256=_path_hashes(_lane_paths()),
        source_artifact_sha256=_flatten_source_hashes(first_batch),
        dataset_values_sha256=first_evidence["dataset_values_sha256"],
        source_output_sha256=source_output,
        upstream_artifact_sha256=_path_hashes(_upstream_paths()),
        candidate_sha256=candidate_sha256,
    )
    runtime_envelope = build_runtime_envelope(
        f"exp409-{uuid.uuid4().hex}",
        datetime.now(timezone.utc).isoformat(),
    )
    return publish_batch(
        first_batch, output, canonical_provenance, runtime_envelope
    )


def _assemble_smoke(
    datasets: Sequence[str], universe: FrozenLaneUniverse
) -> tuple[ReadinessBatch, dict[str, object]]:
    results: list[DatasetReadinessResult] = []
    values: dict[str, str] = {}
    for dataset in datasets:
        result, bundle = _run_dataset_with_bundle(
            str(dataset), universe=universe, config=FROZEN_CONFIG
        )
        results.append(result)
        values[str(dataset)] = dataset_values_sha256(bundle)
    return ReadinessBatch(tuple(results)), {
        "dataset_values_sha256": dict(sorted(values.items()))
    }


def _determinism_payload(
    batch: ReadinessBatch, evidence: Mapping[str, object]
) -> dict[str, object]:
    return {
        "candidate_sha256": candidate_payload_sha256(batch),
        "source_output_sha256": {
            result.dataset: dict(sorted(result.source_output_sha256.items()))
            for result in batch.dataset_results
        },
        "lanes": {
            result.dataset: {
                "hard": list(result.hard_indices),
                "standard": list(result.standard_indices),
                "priority": list(result.priority_indices),
            }
            for result in batch.dataset_results
        },
        "dataset_values_sha256": evidence["dataset_values_sha256"],
        "dataset_results": [asdict(result) for result in batch.dataset_results],
    }


def _assert_deterministic(first: object, second: object) -> None:
    if canonical_json_bytes(first) != canonical_json_bytes(second):
        raise ValueError("Exp409 determinism gate failed")


def _validate_source_batch(batch: SourceBatch) -> None:
    names = [status.source for status in batch.statuses]
    expected = {source.name for source in FEATURE_SOURCES}
    expected.add("tri_view")
    expected.update(f"synthetic_{name}" for name in MODEL_NAMES)
    if (
        len(names) != 13
        or len(set(names)) != 13
        or set(names) != expected
        or set(batch.output_sha256) != expected
    ):
        raise ValueError("Exp409 source status coverage is invalid")


def _bundle_lanes(bundle: object) -> dict[str, tuple[int, ...]]:
    return {
        "hard": tuple(
            sorted(int(value) for value in getattr(bundle, "hard_indices", ()))
        ),
        "standard": tuple(
            sorted(
                int(value) for value in getattr(bundle, "standard_indices", ())
            )
        ),
        "priority": tuple(
            sorted(
                int(value) for value in getattr(bundle, "priority_indices", ())
            )
        ),
    }


def _train_normal_counts(db_path: Path = DB_PATH) -> dict[str, int]:
    resolved = Path(db_path).resolve(strict=True)
    connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT d.name, COUNT(i.id) AS n_train
            FROM datasets AS d
            JOIN instances AS i ON i.dataset_id = d.id
              AND i.split = 'TRAIN' AND i.label = '0'
            GROUP BY d.id, d.name
            ORDER BY n_train ASC, d.name ASC
            """
        ).fetchall()
    finally:
        connection.close()
    return {str(name): int(count) for name, count in rows}


def _aggregate_source_output(batch: ReadinessBatch) -> dict[str, str]:
    result: dict[str, str] = {}
    for source_name in sorted(batch.dataset_results[0].source_output_sha256):
        values = [
            [row.dataset, row.source_output_sha256[source_name]]
            for row in batch.dataset_results
        ]
        result[source_name] = hashlib.sha256(
            canonical_json_bytes(values)
        ).hexdigest()
    return result


def _flatten_source_hashes(batch: ReadinessBatch) -> dict[str, str]:
    return {
        f"{result.dataset}/{source}": digest
        for result in batch.dataset_results
        for source, digest in sorted(result.source_output_sha256.items())
    }


def _path_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in sorted(paths.items())}


def _code_paths() -> dict[str, Path]:
    package = Path(__file__).parent
    return {
        path.name: path
        for path in sorted(package.glob("*.py"))
        if path.name != "__init__.py"
    }


def _lane_paths() -> dict[str, Path]:
    return {
        "exp291_prediction": EXP291_PREDICTION_PATH,
        "exp291_manifest": EXP291_MANIFEST_PATH,
        "exp292_prediction": EXP292_PREDICTION_PATH,
        "exp292_manifest": EXP292_MANIFEST_PATH,
    }


def _upstream_paths() -> dict[str, Path]:
    from experiments.exp403_408 import (
        consensus,
        feature_registry,
        foundation,
        synthetic_anomaly,
        tri_view_ssl,
    )

    return {
        "consensus.py": Path(consensus.__file__),
        "feature_registry.py": Path(feature_registry.__file__),
        "foundation.py": Path(foundation.__file__),
        "synthetic_anomaly.py": Path(synthetic_anomaly.__file__),
        "tri_view_ssl.py": Path(tri_view_ssl.__file__),
    }


def _resolve_git_executable() -> str:
    fallback = (
        Path.home()
        / ".cache"
        / "codex-runtimes"
        / "codex-primary-runtime"
        / "dependencies"
        / "bin"
        / "fallback"
        / "git"
    )
    if fallback.is_file():
        return str(fallback)
    executable = shutil.which("git")
    if executable:
        return executable
    raise RuntimeError("git executable is unavailable")
