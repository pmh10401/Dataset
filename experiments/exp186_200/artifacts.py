from __future__ import annotations

import csv
import json
import sqlite3
import time
from pathlib import Path
from typing import Mapping

from experiments.exp186_200.core import sha256_file, validate_prediction_payload

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path("/Users/minho/Documents/Dataset")
DB_PATH = DATA_DIR / "univariate_ts.db"


class FrozenInputs:
    def __init__(
        self,
        exp137_rows: list[dict[str, str]],
        exp177_policy: dict[str, object],
        exp178_payloads: list[dict[str, object]],
        exp180_payloads: list[dict[str, object]],
        exp181_payloads: list[dict[str, object]],
        exp183_payloads: list[dict[str, object]],
        exp153_157_manifest: list[dict[str, str]],
    ):
        self.exp137_rows = exp137_rows
        self.exp177_policy = exp177_policy
        self.exp178_payloads = exp178_payloads
        self.exp180_payloads = exp180_payloads
        self.exp181_payloads = exp181_payloads
        self.exp183_payloads = exp183_payloads
        self.exp153_157_manifest = exp153_157_manifest


def load_verified_predictions(
    jsonl_path: Path,
    manifest_path: Path,
    expected_datasets: int | None = 639,
) -> list[dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sha256 = manifest["prediction_sha256"]
    if sha256_file(jsonl_path) != expected_sha256:
        raise ValueError(f"prediction hash mismatch for {jsonl_path}")
    payloads = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            validate_prediction_payload(payload)
            payloads.append(payload)

    # Reject missing/duplicated dataset rows independently for every variant.
    by_variant: dict[str, list[dict[str, object]]] = {}
    for p in payloads:
        var = str(p.get("variant", ""))
        by_variant.setdefault(var, []).append(p)

    for var, var_payloads in by_variant.items():
        datasets = [p["dataset"] for p in var_payloads]
        if expected_datasets is not None and len(datasets) != expected_datasets:
            raise ValueError(
                f"Variant {var} has {len(datasets)} datasets instead of {expected_datasets}"
            )
        if len(set(datasets)) != len(datasets):
            raise ValueError(f"Variant {var} has duplicate datasets")

    return payloads


def verify_file_hash(path: Path, expected_sha256: str, description: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{description} hash mismatch: expected={expected_sha256} actual={actual} path={path}"
        )


def load_frozen_inputs(output_root: Path = DATA_DIR / "exp177_185_unattended") -> FrozenInputs:
    exp137_path = ROOT / "results/exp137/experiment_137_operational_triage_results.csv"
    if not exp137_path.exists():
        exp137_path = DATA_DIR / "experiment_137_operational_triage_results.csv"
    with exp137_path.open(newline="", encoding="utf-8") as f:
        exp137_rows = list(csv.DictReader(f))

    exp177_policy_path = output_root / "exp177_frozen_policy.json"
    exp177_policy = json.loads(exp177_policy_path.read_text(encoding="utf-8"))

    exp178_payloads = load_verified_predictions(
        output_root / "exp178_frozen_predictions.jsonl",
        output_root / "exp178_prediction_manifest.json"
    )
    exp180_payloads = load_verified_predictions(
        output_root / "exp180_frozen_predictions.jsonl",
        output_root / "exp180_prediction_manifest.json"
    )
    exp181_payloads = load_verified_predictions(
        output_root / "exp181_frozen_predictions.jsonl",
        output_root / "exp181_prediction_manifest.json"
    )
    exp183_payloads = load_verified_predictions(
        output_root / "exp183_frozen_predictions.jsonl",
        output_root / "exp183_prediction_manifest.json"
    )

    score_manifest_path = ROOT / "outputs/exp153_157_train_only_improvements/01_score_cache_manifest.csv"
    with score_manifest_path.open(newline="", encoding="utf-8") as f:
        exp153_157_manifest = list(csv.DictReader(f))

    return FrozenInputs(
        exp137_rows=exp137_rows,
        exp177_policy=exp177_policy,
        exp178_payloads=exp178_payloads,
        exp180_payloads=exp180_payloads,
        exp181_payloads=exp181_payloads,
        exp183_payloads=exp183_payloads,
        exp153_157_manifest=exp153_157_manifest
    )


def freeze_predictions(
    stage: str,
    payloads: list[dict[str, object]],
    output_root: Path,
    experiment_id: str,
) -> Path:
    prediction_path = output_root / f"exp{stage}_frozen_predictions.jsonl"
    manifest_path = output_root / f"exp{stage}_prediction_manifest.json"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)

    with prediction_path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            validate_prediction_payload(payload)
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    manifest = {
        "experiment_id": experiment_id,
        "prediction_file": prediction_path.name,
        "prediction_sha256": sha256_file(prediction_path),
        "prediction_rows": len(payloads),
        "dataset_count": len({str(item["dataset"]) for item in payloads}),
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return prediction_path


def load_test_labels_for_evaluation(dataset: str) -> np.ndarray:
    """This function is evaluation-only and must never be called by prediction logic."""
    import numpy as np
    connection = sqlite3.connect(str(DB_PATH))
    try:
        rows = connection.execute(
            """
            SELECT CAST(i.label AS INTEGER)
            FROM instances i JOIN datasets d ON d.id = i.dataset_id
            WHERE d.name = ? AND i.split = 'TEST'
            ORDER BY i.instance_index
            """,
            (dataset,),
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        raise ValueError(f"No TEST labels found for dataset: {dataset}")
    return np.asarray([int(row[0]) for row in rows], dtype=np.int8)
