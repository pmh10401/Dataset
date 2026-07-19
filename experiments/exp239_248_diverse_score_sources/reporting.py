from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.exp239_248_diverse_score_sources.cache_pipeline import source_root


def read_summary(root:Path,experiment:int)->list[dict[str,str]]:
    with (source_root(root,experiment)/"summary.csv").open(newline="",encoding="utf-8") as handle:return list(csv.DictReader(handle))


def write_source_report(root:Path,experiment:int)->Path:
    rows=read_summary(root,experiment);base=source_root(root,experiment);source=rows[0]["source"] if rows else "unknown";lines=[f"# Experiment {experiment}: {source}","","Retrospective score-source qualification. Frozen Exp236 lanes were not changed.","","| Variant | Alpha | Hard confirmation | Precision | Standard priority | Precision | Outside TP | Outside FP | Precision |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:lines.append(f"| {row['variant']} | {float(row['alpha']):.3f} | {row['hard_confirmation_count']} | {100*float(row['hard_confirmation_precision']):.2f}% | {row['standard_priority_count']} | {100*float(row['standard_priority_precision']):.2f}% | {row['outside_lane_tp']} | {row['outside_lane_fp']} | {100*float(row['outside_lane_precision']):.2f}% |")
    lines.extend(["","No alpha, variant, or source is selected for deployment from these TEST results."])
    path=base/"RESULT_REPORT.md";path.write_text("\n".join(lines)+"\n");return path


def write_manifest(root:Path,experiment:int,hashes:dict[str,str])->Path:
    base=source_root(root,experiment);payload={"experiment":experiment,"files":hashes,"scope":"retrospective_score_source_qualification","autonomous_policy_changed":False,"known_limitations":["reused benchmark TEST data","no prospective validation","no operating alpha selected"]};path=base/"MANIFEST.json";path.write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n");return path

