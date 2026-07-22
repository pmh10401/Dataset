from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import CACHE_SPECS, RECON_INVENTORY, RECON_INVENTORY_SHA256, RECON_KEYS, RECON_ROOT
from .contracts import sha256_file


def _expand_arrays(archive: np.lib.npyio.NpzFile, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for key in keys:
        values = np.asarray(archive[key], dtype=float)
        if values.ndim == 1:
            output[key.removesuffix("__train_oof")] = values
        elif values.ndim == 2:
            for index, row in enumerate(values):
                output[f"{key.removesuffix('__train_by_seed')}__seed_{index}"] = np.asarray(row, dtype=float)
        else:
            raise ValueError(f"unexpected TRAIN score shape: {key}={values.shape}")
    return output


def load_primary_train_scores() -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], list[dict[str, object]]]:
    scores: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    manifests: list[dict[str, object]] = []
    for spec in CACHE_SPECS:
        if sha256_file(spec.manifest) != spec.manifest_sha256:
            raise ValueError(f"cache manifest hash mismatch: {spec.expert}")
        by_dataset: dict[str, dict[str, np.ndarray]] = {}
        with spec.manifest.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 1117:
            raise ValueError(f"cache manifest coverage mismatch: {spec.expert}")
        for row in rows:
            if row["config_hash"] != spec.config_hash or row["contains_test_labels"] != "False":
                raise ValueError(f"cache contract mismatch: {spec.expert}/{row['dataset']}")
            path = spec.root / row["cache_file"]
            with np.load(path, allow_pickle=False) as archive:
                by_dataset[row["dataset"]] = _expand_arrays(archive, spec.train_keys)
        scores[spec.expert] = by_dataset
        manifests.append({"expert": spec.expert, "manifest": str(spec.manifest), "sha256": spec.manifest_sha256, "datasets": len(rows), "provenance_verified": True})
    return scores, manifests


def load_reconstruction_train_scores(datasets: list[str]) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object]]:
    if sha256_file(RECON_INVENTORY) != RECON_INVENTORY_SHA256:
        raise ValueError("reconstruction inventory hash mismatch")
    inventory = json.loads(RECON_INVENTORY.read_text(encoding="utf-8"))["archives"]
    output: dict[str, dict[str, np.ndarray]] = {}
    for dataset in datasets:
        filename = hashlib.sha256(dataset.encode("utf-8")).hexdigest()[:20] + ".npz"
        path = RECON_ROOT / filename
        if not path.exists():
            continue
        if inventory.get(filename) != sha256_file(path):
            raise ValueError(f"reconstruction archive hash mismatch: {dataset}")
        with np.load(path, allow_pickle=False) as archive:
            output[dataset] = _expand_arrays(archive, RECON_KEYS)
    return output, {"expert": "masked_reconstruction", "inventory": str(RECON_INVENTORY), "sha256": RECON_INVENTORY_SHA256, "datasets": len(output), "provenance_verified": False, "policy_scope": "research_sensitivity_only"}
