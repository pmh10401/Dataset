from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.database import sha256_file
from experiments.exp239_248_diverse_score_sources.calibration import empirical_candidates
from experiments.exp249_250_official_mini_multirocket.cache_pipeline import (
    EXP236_PATH,
    cache_paths,
    source_root,
)
from experiments.exp249_250_official_mini_multirocket.config import ALPHAS, MODEL_CONFIGS


def build_prediction_row(
    *,
    experiment: int,
    source: str,
    variant: str,
    dataset: str,
    alpha: float,
    n_test: int,
    candidates: set[int],
    hard: set[int],
    standard: set[int],
    priority: set[int],
    abstained: bool,
    resolution_blocked: bool,
    attainable_p: float,
) -> dict[str, object]:
    frozen = hard | standard | priority
    integer_list = lambda values: sorted(map(int, values))
    return {
        "experiment_id": f"experiment_{int(experiment)}_{source}",
        "experiment": int(experiment),
        "source": str(source),
        "variant": str(variant),
        "dataset": str(dataset),
        "alpha": float(alpha),
        "n_test": int(n_test),
        "candidate_indices": integer_list(candidates),
        "hard_confirmation_indices": integer_list(candidates & hard),
        "standard_priority_indices": integer_list(candidates & standard),
        "priority_confirmation_indices": integer_list(candidates & priority),
        "outside_lane_indices": integer_list(candidates - frozen),
        "frozen_hard_indices": integer_list(hard),
        "frozen_standard_indices": integer_list(standard),
        "frozen_priority_indices": integer_list(priority),
        "calibration_abstained": bool(abstained),
        "resolution_blocked": bool(resolution_blocked),
        "attainable_p": float(attainable_p),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "hard_policy_changed": False,
    }


def _baseline() -> dict[str, dict[str, object]]:
    return {
        row["dataset"]: row
        for row in map(json.loads, EXP236_PATH.read_text().splitlines())
    }


def variants_from_archive(
    experiment: int, archive: np.lib.npyio.NpzFile
) -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    variants = {}
    for requested in MODEL_CONFIGS[int(experiment)].requested_kernels:
        train = np.asarray(archive[f"kernels_{requested}__train_by_seed"], dtype=float)
        test = np.asarray(archive[f"kernels_{requested}__test_by_seed"], dtype=float)
        variants[f"kernels_{requested}_seed_median"] = (
            np.median(train, axis=0) if train.shape[1] else np.asarray([]),
            np.median(test, axis=0) if test.shape[1] else np.asarray([]),
            "direct",
        )
        variants[f"kernels_{requested}_seed_intersection"] = (
            train,
            test,
            "intersection",
        )
    return variants


def predict_experiment(
    root: Path,
    experiment: int,
    *,
    names: tuple[str, ...],
) -> tuple[Path, dict[str, object]]:
    config = MODEL_CONFIGS[int(experiment)]
    baseline = _baseline()
    rows: list[dict[str, object]] = []
    for dataset in names:
        archive_path, metadata_path = cache_paths(root, experiment, dataset)
        metadata = json.loads(metadata_path.read_text())
        if sha256_file(archive_path) != metadata["cache_sha256"]:
            raise ValueError(f"cache hash mismatch: dataset={dataset}")
        hard = set(map(int, baseline[dataset]["hard_indices"]))
        standard = set(map(int, baseline[dataset]["standard_review_indices"]))
        priority = set(map(int, baseline[dataset]["priority_review_indices"]))
        with np.load(archive_path) as archive:
            for variant, (train_values, test_values, mode) in variants_from_archive(
                experiment, archive
            ).items():
                for alpha in ALPHAS:
                    if metadata["abstained"]:
                        candidates: set[int] = set()
                        blocked = True
                        attainable = 1.0
                    elif mode == "intersection":
                        results = [
                            empirical_candidates(train_values[index], test_values[index], alpha)
                            for index in range(len(train_values))
                        ]
                        candidates = set.intersection(
                            *(set(result.indices) for result in results)
                        )
                        blocked = any(result.resolution_blocked for result in results)
                        attainable = max(result.attainable_p for result in results)
                    else:
                        result = empirical_candidates(train_values, test_values, alpha)
                        candidates = set(result.indices)
                        blocked = result.resolution_blocked
                        attainable = result.attainable_p
                    rows.append(
                        build_prediction_row(
                            experiment=experiment,
                            source=config.source,
                            variant=variant,
                            dataset=dataset,
                            alpha=alpha,
                            n_test=int(metadata["n_test"]),
                            candidates=candidates,
                            hard=hard,
                            standard=standard,
                            priority=priority,
                            abstained=bool(metadata["abstained"]),
                            resolution_blocked=blocked,
                            attainable_p=attainable,
                        )
                    )
    rows.sort(key=lambda row: (str(row["dataset"]), str(row["variant"]), float(row["alpha"])))
    data = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    base = source_root(root, experiment)
    base.mkdir(parents=True, exist_ok=True)
    path = base / "frozen_predictions.jsonl"
    with tempfile.NamedTemporaryFile(dir=base, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    manifest = {
        "experiment": int(experiment),
        "source": config.source,
        "rows": len(rows),
        "datasets": len(names),
        "prediction_sha256": hashlib.sha256(data).hexdigest(),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "hard_policy_changed": False,
    }
    (base / "prediction_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    )
    return path, manifest
