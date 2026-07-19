from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np

from experiments.exp226_235_hard_alert_quality.database import sha256_file
from experiments.exp239_248_diverse_score_sources.config import SEED
from experiments.exp239_248_diverse_score_sources.foundation import load_bundle, stable_json_hash
from experiments.exp239_248_diverse_score_sources.registry import build_source, registration
from experiments.exp239_248_diverse_score_sources.source_base import abstained_result


def dataset_names() -> tuple[str, ...]:
    path = Path("/Users/minho/Documents/Dataset/exp236_detection_coverage_expansion/exp236_frozen_predictions.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    names = tuple(sorted(str(row["dataset"]) for row in rows))
    if len(names) != 1117 or len(set(names)) != 1117:
        raise ValueError("frozen Exp236 population must contain 1,117 unique datasets")
    return names


def _stem(dataset: str) -> str:
    return hashlib.sha256(dataset.encode()).hexdigest()[:20]


def source_root(root: Path, experiment: int) -> Path:
    item = registration(experiment)
    return Path(root) / f"experiment_{experiment}_{item.name}"


def source_config(experiment: int, device: str) -> dict[str, object]:
    item = registration(experiment)
    source = build_source(experiment, device=device)
    return {
        "experiment": experiment,
        "source": item.name,
        "minimum_train": item.minimum_train,
        "seed": SEED,
        "device": device,
        "epochs": getattr(source, "epochs", None),
        "seeds": list(getattr(source, "seeds", ()) or (SEED, 20261720)) if item.model_based else [],
        "dependency": item.dependency,
        "contains_test_labels": False,
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name); handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    np.savez_compressed(temporary, **arrays); os.replace(temporary, path)


def cache_paths(root: Path, experiment: int, dataset: str) -> tuple[Path, Path]:
    base = source_root(root, experiment) / "cache" / _stem(dataset)
    return base.with_suffix(".npz"), base.with_suffix(".json")


def _dependency_embeddings(root: Path, dataset: str) -> tuple[np.ndarray, np.ndarray, str]:
    archive_path, metadata_path = cache_paths(root, 247, dataset)
    if not archive_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"missing Exp247 dependency: dataset={dataset}")
    metadata = json.loads(metadata_path.read_text())
    if sha256_file(archive_path) != metadata["cache_sha256"]:
        raise ValueError(f"Exp247 embedding hash mismatch: dataset={dataset}")
    with np.load(archive_path) as archive:
        return archive["artifact__train_embeddings"], archive["artifact__test_embeddings"], metadata["cache_sha256"]


def build_cache_unit(root_text: str, experiment: int, dataset: str, device: str, reset: bool = False) -> dict[str, object]:
    root = Path(root_text); archive_path, metadata_path = cache_paths(root, experiment, dataset); config = source_config(experiment, device); config_hash = stable_json_hash(config)
    if not reset and archive_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("config_hash") == config_hash and sha256_file(archive_path) == metadata.get("cache_sha256"):
            return metadata
    bundle = load_bundle(dataset); source = build_source(experiment, device=device); dependency_hash = None
    if experiment == 248:
        if len(bundle.train_normal) < source.minimum_train:
            result = abstained_result(source.name, bundle, source.minimum_train)
        else:
            train_embedding, test_embedding, dependency_hash = _dependency_embeddings(root, dataset)
            result = source.score_from_embeddings(train_embedding, test_embedding, seed=SEED)
    else:
        result = source.score(bundle, SEED, archive_path.parent)
    arrays = {"train_oof": np.asarray(result.train_oof, dtype=float), "test_scores": np.asarray(result.test_scores, dtype=float)}
    for name, values in (result.artifacts or {}).items(): arrays[f"artifact__{name}"] = np.asarray(values)
    _write_npz(archive_path, arrays)
    metadata = {
        "dataset": dataset, "experiment": experiment, "source": result.source,
        "n_train": len(bundle.train_normal), "n_test": len(bundle.test_values), "target_length": bundle.target_length,
        "abstained": result.abstained, "abstention_reason": result.abstention_reason,
        "config": config, "config_hash": config_hash, "provenance": dict(result.provenance),
        "dependency_hash": dependency_hash, "contains_test_labels": False,
        "cache_file": str(archive_path.relative_to(source_root(root, experiment))), "cache_sha256": sha256_file(archive_path),
    }
    _atomic_json(metadata_path, metadata); metadata["metadata_sha256"] = sha256_file(metadata_path)
    return metadata


def cache_source(root: Path, experiment: int, *, workers: int, limit: int | None = None, names: tuple[str, ...] | None = None, device: str = "cpu", reset: bool = False, progress: Callable[[int, int, str, int], None] | None = None) -> list[dict[str, object]]:
    item = registration(experiment)
    if item.model_based and workers != 1: raise ValueError("model-based sources require workers=1")
    selected = names if names is not None else (dataset_names()[:limit] if limit else dataset_names()); rows=[]; errors=[]
    if workers == 1:
        for done, dataset in enumerate(selected, 1):
            try: rows.append(build_cache_unit(str(root), experiment, dataset, device, reset))
            except Exception as error: errors.append({"dataset": dataset, "error": repr(error)})
            if progress: progress(done, len(selected), dataset, len(errors))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures={executor.submit(build_cache_unit,str(root),experiment,dataset,device,reset):dataset for dataset in selected}
            for done,future in enumerate(as_completed(futures),1):
                dataset=futures[future]
                try: rows.append(future.result())
                except Exception as error: errors.append({"dataset":dataset,"error":repr(error)})
                if progress: progress(done,len(selected),dataset,len(errors))
    base=source_root(root,experiment);base.mkdir(parents=True,exist_ok=True)
    fields=sorted({key for row in rows for key in row if key not in {"config","provenance"}})
    with (base/"cache_manifest.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows([{key:row.get(key,"") for key in fields} for row in sorted(rows,key=lambda value:value["dataset"])])
    error_path=base/"cache_errors.csv"
    if errors:
        with error_path.open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=("dataset","error"));writer.writeheader();writer.writerows(errors)
        raise RuntimeError(f"source cache errors: experiment={experiment} count={len(errors)}")
    error_path.unlink(missing_ok=True)
    if len(rows)!=len(selected): raise RuntimeError("source cache coverage mismatch")
    return rows
