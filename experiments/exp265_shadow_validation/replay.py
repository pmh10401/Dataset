from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .config import ALPHA, SOURCE_NAMES
from .contracts import SeriesPreview, ShadowRunInput, SourceEvidence, prediction_sha256
from .routing import route_shadow_run


REPLAY_COLLECTION_MODE = "retrospective_replay"
REPLAY_POLICY_VERSION = "exp265-replay-v1"
REPLAY_SCORE_THRESHOLD = 1.0 - ALPHA
REPLAY_CANDIDATE_PREVIEW_POINTS = 64
DEFAULT_DB_PATH = Path("/Users/minho/Documents/Dataset/univariate_ts.db")
DEFAULT_EXP264_PATH = Path(
    "/Users/minho/Documents/Dataset/exp263_264_full_missing_aware_integration/frozen_predictions.jsonl"
)
DEFAULT_CACHE_ROOTS = {
    "statistical": Path("/Users/minho/Documents/Dataset/exp252_sktime_statistical_features_safe_hurst/cache"),
    "structural": Path("/Users/minho/Documents/Dataset/exp254_sktime_structural_features/cache"),
    "change_point": Path("/Users/minho/Documents/Dataset/exp255_change_point_features/cache"),
    "reconstruction": Path("/Users/minho/Documents/Dataset/exp257_mask_ratio_timeseries_sweep/cache"),
}
SOURCE_VARIANTS = {
    "statistical": "combined_40",
    "structural": "structural_combined_440",
    "change_point": "change_point_combined_33",
}


def _empirical_confidence(train_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    reference = np.asarray(train_scores, dtype=float).reshape(-1)
    query = np.asarray(test_scores, dtype=float).reshape(-1)
    if not len(reference) or not np.isfinite(reference).all() or not np.isfinite(query).all():
        raise ValueError("finite TRAIN and TEST scores are required")
    pvalues = np.asarray(
        [(1.0 + np.count_nonzero(reference >= value)) / (len(reference) + 1.0) for value in query],
        dtype=float,
    )
    return 1.0 - pvalues


def load_source_confidence(
    dataset: str,
    *,
    cache_roots: Mapping[str, Path] = DEFAULT_CACHE_ROOTS,
) -> dict[str, np.ndarray | None]:
    key = hashlib.sha256(dataset.encode("utf-8")).hexdigest()[:20]
    output: dict[str, np.ndarray | None] = {}
    for source in SOURCE_NAMES:
        path = Path(cache_roots[source]) / f"{key}.npz"
        if not path.is_file():
            output[source] = None
            continue
        with np.load(path, allow_pickle=False) as archive:
            if source == "reconstruction":
                train = np.asarray(archive["mask_025__train_by_seed"], dtype=float)
                test = np.asarray(archive["mask_025__test_by_seed"], dtype=float)
                if not train.size or not test.size:
                    output[source] = None
                    continue
                output[source] = np.min(
                    np.vstack([_empirical_confidence(train[index], test[index]) for index in range(len(train))]),
                    axis=0,
                )
            else:
                variant = SOURCE_VARIANTS[source]
                train = np.asarray(archive[f"{variant}__train_oof"], dtype=float)
                test = np.asarray(archive[f"{variant}__test_scores"], dtype=float)
                if not train.size or not test.size:
                    output[source] = None
                    continue
                output[source] = _empirical_confidence(
                    train,
                    test,
                )
    return output


def _sanitize_preview(values: np.ndarray, maximum: int = 256) -> dict[str, list[float] | list[int]]:
    series = np.asarray(values, dtype=np.float32).reshape(-1).copy()
    finite = np.isfinite(series)
    if not finite.any():
        series.fill(0.0)
    elif not finite.all():
        positions = np.arange(len(series))
        series[~finite] = np.interp(positions[~finite], positions[finite], series[finite])
    if len(series) > maximum:
        indices = np.unique(np.linspace(0, len(series) - 1, maximum).round().astype(int))
    else:
        indices = np.arange(len(series), dtype=int)
    return {
        "indices": [int(index) for index in indices],
        "values": [float(series[index]) for index in indices],
    }


def _load_dataset_values(
    db_path: Path,
    dataset: str,
    candidate_indices: set[int],
) -> tuple[bool, dict[int, dict[str, list[float] | list[int]]]]:
    with sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True) as connection:
        metadata = connection.execute(
            "select id, has_missing from datasets where name = ?",
            (dataset,),
        ).fetchone()
        if metadata is None:
            raise ValueError(f"dataset not found: {dataset}")
        dataset_id, has_missing = metadata
        if not candidate_indices:
            return bool(has_missing), {}
        placeholders = ",".join("?" for _ in candidate_indices)
        rows = connection.execute(
            f"select i.instance_index, i.values_blob from instances i "
            f"where i.dataset_id = ? and i.split = 'TEST' and i.instance_index in ({placeholders}) "
            "order by i.instance_index",
            (dataset_id, *sorted(candidate_indices)),
        ).fetchall()
    previews = {
        int(index): _sanitize_preview(
            np.frombuffer(blob, dtype=np.float32),
            maximum=REPLAY_CANDIDATE_PREVIEW_POINTS,
        )
        for index, blob in rows
    }
    if set(previews) != candidate_indices:
        raise ValueError(f"candidate TEST waveform coverage mismatch: {dataset}")
    return bool(has_missing), previews


def _run_preview(n_test: int, confidence: Mapping[str, np.ndarray | None]) -> SeriesPreview:
    arrays = [np.asarray(values, dtype=float) for values in confidence.values() if values is not None]
    if arrays:
        if any(len(values) != n_test for values in arrays):
            raise ValueError("source confidence length mismatch")
        values = np.max(np.vstack(arrays), axis=0)
    else:
        values = np.zeros(n_test, dtype=float)
    preview = _sanitize_preview(values)
    return SeriesPreview(tuple(preview["indices"]), tuple(preview["values"]))


def build_replay_payload(
    row: Mapping[str, object],
    *,
    db_path: Path = DEFAULT_DB_PATH,
    confidence: Mapping[str, np.ndarray | None] | None = None,
    observed_at: str = "2026-07-21T00:00:00+00:00",
    config_sha256: str = "b" * 64,
) -> dict[str, object]:
    dataset = str(row["dataset"])
    n_test = int(row["n_test"])
    n_train = int(row["n_train"])
    hard = frozenset(map(int, row["baseline_hard_indices"]))
    standard = frozenset(map(int, row["baseline_standard_indices"]))
    priority = frozenset(map(int, row["baseline_priority_indices"]))
    candidate_indices = set(hard | standard | priority)
    source_indices = row["source_indices"]
    if not isinstance(source_indices, Mapping) or set(source_indices) != set(SOURCE_NAMES):
        raise ValueError(f"invalid source indices: {dataset}")
    confidence = dict(confidence or load_source_confidence(dataset))
    has_missing, candidate_previews = _load_dataset_values(Path(db_path), dataset, candidate_indices)
    sources: dict[str, SourceEvidence] = {}
    for source in SOURCE_NAMES:
        indices = frozenset(map(int, source_indices[source]))
        values = confidence.get(source)
        scores: dict[int, float | None] = {}
        for index in indices:
            score = None if values is None or index >= len(values) else float(values[index])
            scores[index] = score if score is not None and score + 1e-12 >= REPLAY_SCORE_THRESHOLD else None
        sources[source] = SourceEvidence(indices, MappingProxyType(scores), REPLAY_SCORE_THRESHOLD)  # type: ignore[arg-type]
    run_id = f"replay-exp264-{hashlib.sha256(dataset.encode('utf-8')).hexdigest()[:20]}"
    inputs = ShadowRunInput(
        schema_version=1,
        collection_mode=REPLAY_COLLECTION_MODE,
        policy_version=REPLAY_POLICY_VERSION,
        run_id=run_id,
        source_id="historical-experiment-replay",
        observed_at=observed_at,
        dataset_key=dataset,
        equipment_id=None,
        recipe_id=None,
        sensor_id=None,
        step_id=None,
        n_train=n_train,
        n_test=n_test,
        baseline_hard=hard,
        baseline_standard=standard,
        baseline_priority=priority,
        sources=MappingProxyType(sources),
        quality_flags=frozenset(map(int, row.get("quality_flag_indices", []))),
        has_missing=has_missing,
        missing_quality_status=str(row.get("quality_flag_source", "unknown")),
        config_sha256=config_sha256,
        series_preview=_run_preview(n_test, confidence),
    )
    payload = route_shadow_run(inputs)
    for candidate in payload["candidates"]:
        candidate["series_preview"] = candidate_previews[int(candidate["candidate_index"])]
    payload["prediction_sha256"] = prediction_sha256({key: value for key, value in payload.items() if key != "prediction_sha256"})
    return payload


def build_replay_dataset(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    prediction_path: Path = DEFAULT_EXP264_PATH,
    limit: int | None = None,
) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in Path(prediction_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda item: str(item["dataset"]))
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    config_hash = hashlib.sha256(Path(prediction_path).read_bytes()).hexdigest()
    base = datetime(2026, 7, 21, tzinfo=timezone.utc)
    return [
        build_replay_payload(
            row,
            db_path=db_path,
            observed_at=(base + timedelta(seconds=index)).isoformat(),
            config_sha256=config_hash,
        )
        for index, row in enumerate(rows)
    ]


def write_replay_dataset(path: Path, rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    data = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8") for row in rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    candidates = sum(int(row["candidate_count"]) for row in rows)
    return {
        "collection_mode": REPLAY_COLLECTION_MODE,
        "policy_version": REPLAY_POLICY_VERSION,
        "datasets": len(rows),
        "candidates": candidates,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "retrospective": True,
        "sha256": hashlib.sha256(data).hexdigest(),
    }
