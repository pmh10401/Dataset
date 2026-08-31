"""Build the read-only-derived Exp480 evidence database."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import sqlite3
from typing import Iterable, Mapping

from experiments.exp474_conservative_fallback_routing_antigravity.evaluation_antigravity import (
    load_checkpoint_records_antigravity,
)
from experiments.exp_routing_label_free_antigravity.official_router_antigravity import (
    load_train_features_antigravity,
)

from .config_antigravity import (
    CHECKPOINT_DIR_ANTIGRAVITY,
    DATABASE_PATH_ANTIGRAVITY,
    EVIDENCE_DATABASE_PATH_ANTIGRAVITY,
    EXPECTED_DATABASE_SHA256_ANTIGRAVITY,
    EXPECTED_DATASETS_ANTIGRAVITY,
    EXPECTED_EXP478_SHA256_ANTIGRAVITY,
    EXPECTED_EXP479_SHA256_ANTIGRAVITY,
    EXPECTED_FAMILIES_ANTIGRAVITY,
    EXPECTED_FEATURE_CACHE_SHA256_ANTIGRAVITY,
    EXPECTED_FEATURE_LOADER_SHA256_ANTIGRAVITY,
    EXPECTED_FROZEN_962_ANTIGRAVITY,
    EXPECTED_RECOVERED_155_ANTIGRAVITY,
    EXP478_RESULT_PATH_ANTIGRAVITY,
    EXP479_RESULT_PATH_ANTIGRAVITY,
    FEATURE_CACHE_PATH_ANTIGRAVITY,
    FEATURE_KEYS_ANTIGRAVITY,
    METHODS_ANTIGRAVITY,
    OFFICIAL_FEATURE_LOADER_PATH_ANTIGRAVITY,
)
from .contracts_antigravity import CandidateMetricAntigravity, EvidenceRowAntigravity


SCHEMA_ANTIGRAVITY = """
PRAGMA foreign_keys=ON;
CREATE TABLE dataset_evidence(
  dataset TEXT PRIMARY KEY,
  source_family TEXT NOT NULL,
  proposal TEXT NOT NULL
);
CREATE TABLE train_features(
  dataset TEXT PRIMARY KEY REFERENCES dataset_evidence(dataset),
  width REAL NOT NULL,
  n_train REAL NOT NULL,
  spectral_entropy REAL NOT NULL,
  skewness REAL NOT NULL,
  kurtosis REAL NOT NULL,
  diff_std_ratio REAL NOT NULL,
  zero_crossing_rate REAL NOT NULL
);
CREATE TABLE candidate_metrics(
  dataset TEXT NOT NULL REFERENCES dataset_evidence(dataset),
  method TEXT NOT NULL,
  auc REAL NOT NULL,
  zf INTEGER NOT NULL,
  PRIMARY KEY(dataset, method)
);
CREATE TABLE provenance(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _sha256_antigravity(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def family_from_dataset_antigravity(dataset: str) -> str:
    """Return only the exact prefix before the first `_normal_` marker."""
    if type(dataset) is not str or "_normal_" not in dataset:
        raise ValueError("dataset key must contain _normal_")
    family = dataset.split("_normal_", 1)[0]
    if not family:
        raise ValueError("dataset family must not be empty")
    return family


def _validated_source_hash_antigravity(
    path: Path, expected: str, label: str
) -> str:
    actual = _sha256_antigravity(path)
    if actual != expected:
        raise RuntimeError(f"{label} source hash changed")
    return actual


def source_provenance_antigravity() -> dict[str, str]:
    """Return deterministic provenance for every frozen result source."""
    checkpoint_hashes = [
        f"{path.name}:{_sha256_antigravity(path)}"
        for path in sorted(CHECKPOINT_DIR_ANTIGRAVITY.glob("chunk_*_antigravity.json"))
    ]
    checkpoint_digest = sha256("\n".join(checkpoint_hashes).encode("utf-8")).hexdigest()
    database_stat = DATABASE_PATH_ANTIGRAVITY.stat()
    return {
        "schema": "exp480_evidence_provenance_v1",
        "database_expected_sha256": EXPECTED_DATABASE_SHA256_ANTIGRAVITY,
        "database_inode": str(database_stat.st_ino),
        "database_size": str(database_stat.st_size),
        "exp478_result_sha256": _validated_source_hash_antigravity(
            EXP478_RESULT_PATH_ANTIGRAVITY,
            EXPECTED_EXP478_SHA256_ANTIGRAVITY,
            "Exp478 result",
        ),
        "exp479_result_sha256": _validated_source_hash_antigravity(
            EXP479_RESULT_PATH_ANTIGRAVITY,
            EXPECTED_EXP479_SHA256_ANTIGRAVITY,
            "Exp479 result",
        ),
        "feature_cache_sha256": _validated_source_hash_antigravity(
            FEATURE_CACHE_PATH_ANTIGRAVITY,
            EXPECTED_FEATURE_CACHE_SHA256_ANTIGRAVITY,
            "feature cache",
        ),
        "feature_loader_sha256": _validated_source_hash_antigravity(
            OFFICIAL_FEATURE_LOADER_PATH_ANTIGRAVITY,
            EXPECTED_FEATURE_LOADER_SHA256_ANTIGRAVITY,
            "feature loader",
        ),
        "score_checkpoint_count": str(len(checkpoint_hashes)),
        "score_checkpoints_sha256": checkpoint_digest,
    }


def _score_tuple_antigravity(
    raw_scores: Mapping[str, object], dataset: str
) -> tuple[CandidateMetricAntigravity, ...]:
    if tuple(raw_scores) != METHODS_ANTIGRAVITY:
        raise ValueError(f"candidate method order changed: {dataset}")
    metrics: list[CandidateMetricAntigravity] = []
    for method in METHODS_ANTIGRAVITY:
        raw = raw_scores[method]
        if type(raw) is not dict or set(raw) != {"auc", "zf"}:
            raise ValueError(f"candidate metrics are malformed: {dataset}/{method}")
        metrics.append(
            CandidateMetricAntigravity(
                method=method,
                auc=float(raw["auc"]),
                zf=int(raw["zf"]),
            )
        )
    return tuple(metrics)


def _load_score_map_antigravity() -> dict[str, tuple[CandidateMetricAntigravity, ...]]:
    frozen = load_checkpoint_records_antigravity(CHECKPOINT_DIR_ANTIGRAVITY)
    recovered_payload = json.loads(EXP479_RESULT_PATH_ANTIGRAVITY.read_text(encoding="utf-8"))
    recovered = recovered_payload.get("rows")
    if len(frozen) != EXPECTED_FROZEN_962_ANTIGRAVITY:
        raise ValueError("frozen checkpoint row count must be 962")
    if type(recovered) is not list or len(recovered) != EXPECTED_RECOVERED_155_ANTIGRAVITY:
        raise ValueError("Exp479 recovered row count must be 155")
    result: dict[str, tuple[CandidateMetricAntigravity, ...]] = {}
    for row in [*frozen, *recovered]:
        if type(row) is not dict or type(row.get("dataset")) is not str:
            raise ValueError("score row is malformed")
        dataset = str(row["dataset"])
        if dataset in result:
            raise ValueError(f"duplicate score row: {dataset}")
        raw_scores = row.get("scores")
        if type(raw_scores) is not dict:
            raise ValueError(f"score mapping is missing: {dataset}")
        result[dataset] = _score_tuple_antigravity(raw_scores, dataset)
    if len(result) != EXPECTED_DATASETS_ANTIGRAVITY:
        raise ValueError("combined score evidence must cover exactly 1117 datasets")
    return result


def _load_proposals_antigravity() -> dict[str, str]:
    payload = json.loads(EXP478_RESULT_PATH_ANTIGRAVITY.read_text(encoding="utf-8"))
    raw_rows = payload.get("rows")
    if type(raw_rows) is not list or len(raw_rows) != EXPECTED_DATASETS_ANTIGRAVITY:
        raise ValueError("Exp478 must contain exactly 1117 route rows")
    proposals: dict[str, str] = {}
    for row in raw_rows:
        if type(row) is not dict:
            raise ValueError("Exp478 route row is malformed")
        dataset = row.get("dataset")
        proposal = row.get("proposal")
        if type(dataset) is not str or type(proposal) is not str:
            raise ValueError("Exp478 dataset or proposal is malformed")
        if proposal not in METHODS_ANTIGRAVITY or dataset in proposals:
            raise ValueError(f"Exp478 proposal is invalid: {dataset}")
        proposals[dataset] = proposal
    return proposals


def _load_features_antigravity(datasets: Iterable[str]) -> dict[str, tuple[float, ...]]:
    cached = json.loads(FEATURE_CACHE_PATH_ANTIGRAVITY.read_text(encoding="utf-8"))
    if type(cached) is not dict:
        raise ValueError("feature cache must be an object")
    result: dict[str, tuple[float, ...]] = {}
    with sqlite3.connect(f"file:{DATABASE_PATH_ANTIGRAVITY}?mode=ro", uri=True) as connection:
        for dataset in sorted(datasets):
            raw = cached.get(dataset)
            if raw is None:
                raw = load_train_features_antigravity(dataset, connection=connection)
            if type(raw) is not dict or any(key not in raw for key in FEATURE_KEYS_ANTIGRAVITY):
                raise ValueError(f"TRAIN features are incomplete: {dataset}")
            values = tuple(float(raw[key]) for key in FEATURE_KEYS_ANTIGRAVITY)
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"TRAIN features are non-finite: {dataset}")
            result[dataset] = values
    return result


def load_evidence_rows_antigravity() -> tuple[EvidenceRowAntigravity, ...]:
    """Merge all frozen scores, proposals and TRAIN-only features in key order."""
    source_provenance_antigravity()
    scores = _load_score_map_antigravity()
    proposals = _load_proposals_antigravity()
    if set(scores) != set(proposals):
        raise ValueError("score and proposal populations differ")
    features = _load_features_antigravity(scores)
    rows = tuple(
        EvidenceRowAntigravity(
            dataset=dataset,
            source_family=family_from_dataset_antigravity(dataset),
            proposal=proposals[dataset],
            features=features[dataset],
            scores=scores[dataset],
        )
        for dataset in sorted(scores)
    )
    if len({row.source_family for row in rows}) != EXPECTED_FAMILIES_ANTIGRAVITY:
        raise ValueError("combined population must contain exactly 128 source families")
    return rows


def build_evidence_database_antigravity(
    rows: Iterable[EvidenceRowAntigravity],
    target: Path = EVIDENCE_DATABASE_PATH_ANTIGRAVITY,
    provenance: Mapping[str, str] | None = None,
) -> str:
    """Write a complete deterministic-order SQLite database by atomic replace."""
    ordered = tuple(sorted(rows, key=lambda row: row.dataset))
    if len(ordered) != EXPECTED_DATASETS_ANTIGRAVITY:
        raise ValueError("evidence database requires exactly 1117 rows")
    if len({row.dataset for row in ordered}) != len(ordered):
        raise ValueError("evidence datasets must be unique")
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("evidence_temporary_antigravity.sqlite3")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.executescript(SCHEMA_ANTIGRAVITY)
        for row in ordered:
            connection.execute(
                "INSERT INTO dataset_evidence VALUES(?,?,?)",
                (row.dataset, row.source_family, row.proposal),
            )
            connection.execute(
                "INSERT INTO train_features VALUES(?,?,?,?,?,?,?,?)",
                (row.dataset, *row.features),
            )
            connection.executemany(
                "INSERT INTO candidate_metrics VALUES(?,?,?,?)",
                (
                    (row.dataset, metric.method, metric.auc, metric.zf)
                    for metric in row.scores
                ),
            )
        connection.executemany(
            "INSERT INTO provenance VALUES(?,?)",
            sorted(dict(provenance or source_provenance_antigravity()).items()),
        )
        if connection.execute("SELECT count(*) FROM dataset_evidence").fetchone()[0] != 1117:
            raise RuntimeError("evidence dataset count changed")
        if connection.execute("SELECT count(*) FROM candidate_metrics").fetchone()[0] != 4468:
            raise RuntimeError("candidate metric count changed")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("evidence foreign-key check failed")
        connection.commit()
        connection.execute("VACUUM")
    except BaseException:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    connection.close()
    temporary.replace(destination)
    return _sha256_antigravity(destination)


def load_evidence_database_antigravity(
    path: Path = EVIDENCE_DATABASE_PATH_ANTIGRAVITY,
) -> tuple[EvidenceRowAntigravity, ...]:
    """Read back the strict evidence contracts from the derived database."""
    connection = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    try:
        metrics: dict[str, list[CandidateMetricAntigravity]] = {}
        for dataset, method, auc, zf in connection.execute(
            "SELECT dataset,method,auc,zf FROM candidate_metrics ORDER BY dataset,method"
        ):
            metrics.setdefault(str(dataset), []).append(
                CandidateMetricAntigravity(str(method), float(auc), int(zf))
            )
        rows = []
        query = (
            "SELECT d.dataset,d.source_family,d.proposal,"
            "f.width,f.n_train,f.spectral_entropy,f.skewness,f.kurtosis,"
            "f.diff_std_ratio,f.zero_crossing_rate "
            "FROM dataset_evidence d JOIN train_features f USING(dataset) ORDER BY d.dataset"
        )
        method_rank = {method: index for index, method in enumerate(METHODS_ANTIGRAVITY)}
        for raw in connection.execute(query):
            dataset = str(raw[0])
            ordered_metrics = tuple(
                sorted(metrics[dataset], key=lambda metric: method_rank[metric.method])
            )
            rows.append(
                EvidenceRowAntigravity(
                    dataset=dataset,
                    source_family=str(raw[1]),
                    proposal=str(raw[2]),
                    features=tuple(float(value) for value in raw[3:]),
                    scores=ordered_metrics,
                )
            )
    finally:
        connection.close()
    return tuple(rows)


__all__ = [
    "build_evidence_database_antigravity",
    "family_from_dataset_antigravity",
    "load_evidence_database_antigravity",
    "load_evidence_rows_antigravity",
    "source_provenance_antigravity",
]
