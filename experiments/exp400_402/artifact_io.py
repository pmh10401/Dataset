from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.exp397_399.artifact_io import (
    EXP292_MANIFEST,
    EXP292_PREDICTION,
    MANIFEST_NAME as EXP397_MANIFEST_NAME,
    PREDICTION_NAME as EXP397_PREDICTION_NAME,
    load_exp292_rows,
    load_verified_ranking,
)

from experiments.exp400_402 import EXP400_ID
from experiments.exp400_402.contracts import (
    ALPHAS,
    EXPECTED_AUTOMATIC_HARD,
    EXPECTED_DATASETS,
    EXPECTED_PRIORITY_REVIEW,
    EXPECTED_STANDARD_CANDIDATES,
    POLICY_VERSION,
    canonical_json,
    validate_shadow_row,
)


DATA_ROOT = Path("/Users/minho/Documents/Dataset")
EXP397_ROOT = DATA_ROOT / "exp397_399_review_ranking"
EXP397_PREDICTION = EXP397_ROOT / EXP397_PREDICTION_NAME
EXP397_MANIFEST = EXP397_ROOT / EXP397_MANIFEST_NAME
PREDICTION_NAME = "exp400_shadow_ranking_replay.jsonl"
MANIFEST_NAME = "exp400_prediction_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inputs() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    exp292_rows, exp292_manifest = load_exp292_rows()
    exp397_rows, exp397_manifest = load_verified_ranking(
        EXP397_PREDICTION,
        EXP397_MANIFEST,
    )
    if exp397_manifest.get("contains_evaluation_metrics") is not False:
        raise ValueError("Exp397 manifest contains evaluation metrics")
    if exp397_manifest.get("candidate_universe_count") != EXPECTED_STANDARD_CANDIDATES:
        raise ValueError("Exp397 manifest candidate universe mismatch")
    return exp292_rows, exp397_rows, {
        "exp292_prediction_sha256": sha256_file(EXP292_PREDICTION),
        "exp292_manifest_sha256": sha256_file(EXP292_MANIFEST),
        "exp397_prediction_sha256": sha256_file(EXP397_PREDICTION),
        "exp397_manifest_sha256": sha256_file(EXP397_MANIFEST),
        "exp292_manifest_prediction_sha256": exp292_manifest["prediction_sha256"],
        "exp397_manifest_prediction_sha256": exp397_manifest["prediction_sha256"],
    }


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_prediction(
    output_root: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    full_run: bool,
    input_hashes: Mapping[str, str] | None = None,
) -> dict[str, object]:
    output_root = Path(output_root)
    ordered = [validate_shadow_row(dict(row)) for row in sorted(rows, key=lambda item: str(item["dataset"]))]
    datasets = len(ordered)
    standard = sum(int(row["standard_review_count"]) for row in ordered)
    hard = sum(int(row["automatic_hard_count"]) for row in ordered)
    priority = sum(int(row["priority_review_count"]) for row in ordered)
    if len({str(row["dataset"]) for row in ordered}) != datasets:
        raise ValueError("prediction contains duplicate datasets")
    if full_run and (
        datasets != EXPECTED_DATASETS
        or standard != EXPECTED_STANDARD_CANDIDATES
        or hard != EXPECTED_AUTOMATIC_HARD
        or priority != EXPECTED_PRIORITY_REVIEW
    ):
        raise ValueError(
            "full prediction coverage mismatch: "
            f"datasets={datasets} standard={standard} hard={hard} priority={priority}"
        )
    data = "".join(canonical_json(row) + "\n" for row in ordered).encode("utf-8")
    prediction_path = output_root / PREDICTION_NAME
    _atomic_write(prediction_path, data)
    manifest: dict[str, object] = {
        "experiment_id": EXP400_ID,
        "policy_version": POLICY_VERSION,
        "prediction_file": PREDICTION_NAME,
        "prediction_sha256": sha256_file(prediction_path),
        "datasets": datasets,
        "standard_review_candidates": standard,
        "automatic_hard_count": hard,
        "priority_review_count": priority,
        "alphas": list(ALPHAS),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "automatic_policy_changed": False,
        "test_metrics_select_policy": False,
        "retrospective": True,
        "prospective": False,
        "deployment_validated": False,
        "input_hashes": dict(input_hashes or {}),
    }
    _atomic_write(
        output_root / MANIFEST_NAME,
        (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest


def load_verified_prediction(
    prediction_path: Path,
    manifest_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    prediction_path = Path(prediction_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if sha256_file(prediction_path) != manifest.get("prediction_sha256"):
        raise ValueError("Exp400 prediction SHA256 mismatch")
    if manifest.get("contains_test_labels") is not False:
        raise ValueError("Exp400 manifest contains TEST labels")
    if manifest.get("contains_evaluation_metrics") is not False:
        raise ValueError("Exp400 manifest contains evaluation metrics")
    rows = [
        validate_shadow_row(json.loads(line))
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != int(manifest.get("datasets", -1)):
        raise ValueError("Exp400 prediction dataset count mismatch")
    if sum(int(row["standard_review_count"]) for row in rows) != int(
        manifest.get("standard_review_candidates", -1)
    ):
        raise ValueError("Exp400 prediction candidate count mismatch")
    return rows, manifest
