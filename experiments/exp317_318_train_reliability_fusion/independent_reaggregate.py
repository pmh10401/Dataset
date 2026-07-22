from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .contracts import sha256_file


def independent_reaggregate(root: Path) -> list[dict[str, object]]:
    root = Path(root)
    detail_path = root / "exp318_results.csv"
    with detail_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    output = []
    for variant, selected in sorted(grouped.items()):
        count = sum(int(row["candidate_count"]) for row in selected)
        tp = sum(int(row["tp"]) for row in selected)
        fp = sum(int(row["fp"]) for row in selected)
        output.append({
            "variant": variant,
            "datasets": len(selected),
            "candidate_count": count,
            "tp": tp,
            "fp": fp,
            "precision": tp / count if count else 0.0,
            "mean_candidate_f1": float(np.mean([float(row["candidate_f1"]) for row in selected])),
            "mean_combined_f1": float(np.mean([float(row["combined_f1"]) for row in selected])),
        })
    payload = {
        "source_file": str(detail_path),
        "source_sha256": sha256_file(detail_path),
        "rows": output,
        "matches_summary_checked_separately": True,
    }
    (root / "exp318_independent_reaggregate.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return output
