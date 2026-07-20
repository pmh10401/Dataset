from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.database import sha256_file
from experiments.exp239_248_diverse_score_sources.calibration import empirical_candidates
from .cache_pipeline import EXP236_PATH, cache_paths
from .config import ALPHAS, EXP253_PATH, EXPERIMENT, EXPERIMENT_ID, SOURCE
from .features import FEATURE_DIMENSIONS


def build_prediction_row(
    *, variant: str, dataset: str, alpha: float, n_test: int,
    candidates: set[int], hard: set[int], standard: set[int], priority: set[int],
    exp253_hard: set[int], exp253_standard: set[int], abstained: bool,
    resolution_blocked: bool, attainable_p: float,
) -> dict[str, object]:
    frozen = hard | standard | priority
    to_list = lambda values: sorted(map(int, values))
    return {
        "experiment_id": EXPERIMENT_ID, "experiment": EXPERIMENT, "source": SOURCE,
        "variant": str(variant), "dataset": str(dataset), "alpha": float(alpha),
        "n_test": int(n_test), "candidate_indices": to_list(candidates),
        "hard_confirmation_indices": to_list(candidates & hard),
        "standard_priority_indices": to_list(candidates & standard),
        "priority_confirmation_indices": to_list(candidates & priority),
        "outside_lane_indices": to_list(candidates - frozen),
        "exp253_hard_confirmation_indices": to_list(candidates & exp253_hard),
        "exp253_remaining_standard_priority_indices": to_list(candidates & exp253_standard),
        "frozen_hard_indices": to_list(hard),
        "frozen_standard_indices": to_list(standard),
        "frozen_priority_indices": to_list(priority),
        "calibration_abstained": bool(abstained),
        "resolution_blocked": bool(resolution_blocked),
        "attainable_p": float(attainable_p),
        "contains_test_labels": False, "contains_evaluation_metrics": False,
        "hard_policy_changed": False, "retrospective": True,
    }


def _maps() -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    baseline = {str(row["dataset"]): row for row in map(json.loads, EXP236_PATH.read_text().splitlines())}
    exp253 = {str(row["dataset"]): row for row in map(json.loads, EXP253_PATH.read_text().splitlines())}
    if len(baseline) != 1117 or set(baseline) != set(exp253):
        raise ValueError("Exp236/Exp253 frozen coverage mismatch")
    return baseline, exp253


def predict_experiment(root: Path, *, names: tuple[str, ...]) -> tuple[Path, dict[str, object]]:
    baseline, exp253 = _maps()
    rows: list[dict[str, object]] = []
    for dataset in names:
        archive_path, metadata_path = cache_paths(root, dataset)
        metadata = json.loads(metadata_path.read_text())
        if sha256_file(archive_path) != metadata["cache_sha256"]:
            raise ValueError(f"cache hash mismatch: dataset={dataset}")
        base = baseline[dataset]
        new = exp253[dataset]
        hard = set(map(int, base["hard_indices"]))
        standard = set(map(int, base["standard_review_indices"]))
        priority = set(map(int, base["priority_review_indices"]))
        with np.load(archive_path) as archive:
            for variant in FEATURE_DIMENSIONS:
                train_scores = np.asarray(archive[f"{variant}__train_oof"], dtype=float)
                test_scores = np.asarray(archive[f"{variant}__test_scores"], dtype=float)
                for alpha in ALPHAS:
                    if metadata["abstained"]:
                        candidates: set[int] = set(); blocked = True; attainable = 1.0
                    else:
                        result = empirical_candidates(train_scores, test_scores, alpha)
                        candidates = set(map(int, result.indices))
                        blocked = result.resolution_blocked
                        attainable = result.attainable_p
                    rows.append(build_prediction_row(
                        variant=variant, dataset=dataset, alpha=alpha,
                        n_test=int(metadata["n_test"]), candidates=candidates,
                        hard=hard, standard=standard, priority=priority,
                        exp253_hard=set(map(int, new["final_hard_indices"])),
                        exp253_standard=set(map(int, new["final_standard_indices"])),
                        abstained=bool(metadata["abstained"]),
                        resolution_blocked=blocked, attainable_p=attainable,
                    ))
    rows.sort(key=lambda row: (str(row["dataset"]), str(row["variant"]), float(row["alpha"])))
    data = b"".join((json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "frozen_predictions.jsonl"
    with tempfile.NamedTemporaryFile(dir=root, delete=False) as handle:
        temporary = Path(handle.name); handle.write(data); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    manifest = {
        "experiment_id": EXPERIMENT_ID, "rows": len(rows), "datasets": len(names),
        "prediction_sha256": hashlib.sha256(data).hexdigest(),
        "contains_test_labels": False, "contains_evaluation_metrics": False,
        "hard_policy_changed": False, "retrospective": True,
    }
    (root / "prediction_manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return path, manifest
