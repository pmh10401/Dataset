from __future__ import annotations

import csv
import json
from pathlib import Path


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def write_result_report(path: Path, experiment_id: str, rows: list[dict[str, object]]) -> None:
    lines = [
        f"# {experiment_id}", "",
        "Retrospective research Shadow evaluation. Exp291 automatic Hard is unchanged.", "",
        "| Variant | Candidates | TP | FP | Precision | Outside-all TP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['candidate_count']} | {row['tp']} | {row['fp']} | "
            f"{100*float(row['precision']):.3f}% | {row['outside_all_existing_tp']} |"
        )
    lines.extend([
        "", "- TEST labels were loaded only after prediction SHA256 verification.",
        "- No variant is selected or promoted to autonomous Hard.",
        "- The 80% precision line is descriptive and did not select a policy.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
