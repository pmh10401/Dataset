from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from experiments.exp293_294_train_only_moe_lite.contracts import sha256_file
from experiments.exp293_294_train_only_moe_lite.core import audit_expert

from .config import ALPHA, EXPECTED_DATASETS, EXP293_AUDIT, EXP293_AUDIT_SHA256, SOURCE_SPECS


PRIMARY = {"statistical", "structural", "change_point", "official_multirocket"}


def _boolean(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_existing_audits() -> dict[str, dict[str, dict[str, object]]]:
    if sha256_file(EXP293_AUDIT) != EXP293_AUDIT_SHA256:
        raise ValueError("Exp293 audit SHA256 mismatch")
    output: dict[str, dict[str, dict[str, object]]] = {}
    with EXP293_AUDIT.open(encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            expert = str(raw["expert"])
            if expert not in PRIMARY:
                continue
            row = dict(raw)
            for key in ("available", "provenance_verified"):
                row[key] = _boolean(row[key])
            for key in ("risk_upper_95", "stability", "attainable_p"):
                row[key] = float(row[key])
            row["stability_observable"] = int(row.get("variant_count", 0)) >= 2
            output.setdefault(str(row["dataset"]), {})[expert] = row
    if len(output) != EXPECTED_DATASETS or any(set(rows) != PRIMARY for rows in output.values()):
        raise ValueError("Exp293 primary audit coverage mismatch")
    return output


def load_new_source(expert: str) -> tuple[dict[str, dict[str, object]], dict[str, set[int]], dict[str, object]]:
    spec = SOURCE_SPECS[expert]
    root = Path(spec["root"])
    manifest_path = root / "cache_manifest.csv"
    prediction_path = root / "frozen_predictions.jsonl"
    if sha256_file(manifest_path) != spec["cache_manifest_sha256"]:
        raise ValueError(f"{expert} cache manifest SHA256 mismatch")
    if sha256_file(prediction_path) != spec["prediction_sha256"]:
        raise ValueError(f"{expert} prediction SHA256 mismatch")
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    if len(rows) != EXPECTED_DATASETS or len({row["dataset"] for row in rows}) != EXPECTED_DATASETS:
        raise ValueError(f"{expert} cache coverage mismatch")
    audits: dict[str, dict[str, object]] = {}
    config_hashes: set[str] = set()
    for row in rows:
        if row["contains_test_labels"] != "False":
            raise ValueError(f"{expert} cache contains TEST labels")
        config_hashes.add(row["config_hash"])
        path = root / row["cache_file"]
        if sha256_file(path) != row["cache_sha256"]:
            raise ValueError(f"{expert} cache archive SHA256 mismatch: {row['dataset']}")
        with np.load(path, allow_pickle=False) as archive:
            values = np.asarray(archive["train_oof"], dtype=float)
        audit = audit_expert(
            dataset=row["dataset"], expert=expert,
            train_variants={"train_oof": values} if len(values) else {},
            provenance_verified=True, alpha=ALPHA,
        )
        audit["stability"] = 0.0
        audit["stability_observable"] = False
        audit["experiment_id"] = "experiment_307_expanded_six_expert_moe"
        audit["policy_scope"] = "expanded_gate_eligible"
        audits[row["dataset"]] = audit
    candidates: dict[str, set[int]] = {}
    for line in prediction_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("variant") != spec["variant"] or float(row.get("alpha", -1.0)) != ALPHA:
            continue
        if row.get("contains_test_labels") is not False or row.get("contains_evaluation_metrics") is not False:
            raise ValueError(f"{expert} prediction contract violation")
        dataset = str(row["dataset"])
        if dataset in candidates:
            raise ValueError(f"duplicate {expert} prediction row")
        candidates[dataset] = set(map(int, row["candidate_indices"]))
    if len(candidates) != EXPECTED_DATASETS or set(candidates) != set(audits):
        raise ValueError(f"{expert} prediction coverage mismatch")
    provenance = {
        "expert": expert, "datasets": EXPECTED_DATASETS,
        "cache_manifest": str(manifest_path), "cache_manifest_sha256": spec["cache_manifest_sha256"],
        "prediction": str(prediction_path), "prediction_sha256": spec["prediction_sha256"],
        "config_hashes": sorted(config_hashes), "variant": spec["variant"],
        "stability_observable": False, "contains_test_labels": False,
    }
    return audits, candidates, provenance
