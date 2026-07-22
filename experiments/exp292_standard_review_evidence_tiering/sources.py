from __future__ import annotations

import json
from collections import defaultdict

from .config import ALPHA, SOURCE_SPECS, SourceSpec
from .contracts import sha256_file
from .routing import build_stable_family_support


def load_source_support(spec: SourceSpec) -> tuple[dict[str, set[int]], dict[str, object]]:
    observed_hash = sha256_file(spec.path)
    if observed_hash != spec.sha256:
        raise ValueError(f"source hash mismatch: {spec.name}")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    with spec.path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if float(row["alpha"]) != ALPHA:
                continue
            if row.get("contains_test_labels") is not False or row.get("contains_evaluation_metrics") is not False:
                raise ValueError(f"source contains forbidden evaluation data: {spec.name}")
            grouped[str(row["dataset"])].append(row)
    if len(grouped) != spec.expected_datasets:
        raise ValueError(f"source dataset coverage mismatch: {spec.name}={len(grouped)}/{spec.expected_datasets}")
    support = {
        dataset: build_stable_family_support(rows, expected_variants=spec.variants)
        for dataset, rows in grouped.items()
    }
    return support, {
        "name": spec.name,
        "path": str(spec.path),
        "sha256": observed_hash,
        "datasets": len(grouped),
        "variants": list(spec.variants),
        "alpha": ALPHA,
    }


def load_all_source_support() -> tuple[dict[str, dict[str, set[int]]], list[dict[str, object]]]:
    all_support: dict[str, dict[str, set[int]]] = {}
    manifests: list[dict[str, object]] = []
    for spec in SOURCE_SPECS:
        support, manifest = load_source_support(spec)
        all_support[spec.name] = support
        manifests.append(manifest)
    return all_support, manifests

