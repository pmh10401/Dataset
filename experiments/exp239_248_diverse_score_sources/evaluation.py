from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.database import load_test_labels_for_evaluation
from experiments.exp239_248_diverse_score_sources.cache_pipeline import source_root
from experiments.exp239_248_diverse_score_sources.prediction import verify_file_hash


VIEWS = ("candidate", "hard_confirmation", "standard_priority", "priority_confirmation", "outside_lane")


def outside_lane_indices(candidates:set[int],*,hard:set[int],standard:set[int],priority:set[int])->frozenset[int]:
    return frozenset(candidates-(hard|standard|priority))


def verify_prediction_manifest(prediction_path:Path,manifest_path:Path)->dict[str,object]:
    manifest=json.loads(Path(manifest_path).read_text());verify_file_hash(Path(prediction_path),str(manifest["prediction_sha256"]));return manifest


def _metrics(indices:list[int],truth:set[int])->dict[str,float|int]:
    selected=set(map(int,indices));tp=len(selected&truth);fp=len(selected-truth);fn=len(truth-selected);precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn);f1=2*precision*recall/max(1e-12,precision+recall)
    return {"count":len(selected),"tp":tp,"fp":fp,"fn":fn,"precision":precision,"recall":recall,"f1":f1}


def _write_csv(path:Path,rows:list[dict[str,object]])->None:
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True);fields=list(rows[0])
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({key:json.dumps(value,separators=(",",":")) if isinstance(value,(list,dict,tuple)) else value for key,value in row.items()})


def evaluate_source(root:Path,experiment:int)->tuple[list[dict[str,object]],list[dict[str,object]]]:
    base=source_root(root,experiment);prediction=base/"frozen_predictions.jsonl";manifest=verify_prediction_manifest(prediction,base/"prediction_manifest.json");rows=[json.loads(line) for line in prediction.read_text().splitlines()];evaluated=[]
    for row in rows:
        labels=load_test_labels_for_evaluation(row["dataset"]);truth=set(np.flatnonzero(labels==1).astype(int));output=dict(row);output["true_anomaly_count"]=len(truth)
        for view in VIEWS:output.update({f"{view}_{key}":value for key,value in _metrics(row[f"{view}_indices"],truth).items()})
        evaluated.append(output)
    groups:dict[tuple[str,float],list[dict[str,object]]]=defaultdict(list)
    for row in evaluated:groups[(str(row["variant"]),float(row["alpha"]))].append(row)
    summaries=[]
    for (variant,alpha),group in sorted(groups.items()):
        summary={"experiment_id":group[0]["experiment_id"],"experiment":experiment,"source":group[0]["source"],"variant":variant,"alpha":alpha,"datasets":len(group),"abstained_datasets":sum(bool(row["calibration_abstained"]) for row in group),"resolution_blocked_datasets":sum(bool(row["resolution_blocked"]) for row in group),"hard_policy_changed":0,"comparison_eligible":0,"retrospective":1}
        for view in VIEWS:
            tp=sum(int(row[f"{view}_tp"]) for row in group);fp=sum(int(row[f"{view}_fp"]) for row in group);count=tp+fp
            summary.update({f"{view}_count":count,f"{view}_tp":tp,f"{view}_fp":fp,f"{view}_precision":tp/max(1,count),f"{view}_mean_f1":float(np.mean([row[f"{view}_f1"] for row in group]))})
        summary["confirmation_gate_passed"]=int(summary["hard_confirmation_count"]>=100 and summary["hard_confirmation_precision"]>=0.9 and summary["hard_confirmation_precision"]>=0.8954631379962193)
        summary["review_gate_passed"]=int(summary["standard_priority_count"]>=50 and summary["standard_priority_precision"]>=0.7)
        summary["coverage_gate_passed"]=int(summary["outside_lane_tp"]>=25 and summary["outside_lane_precision"]>=0.8)
        summary["gate_scope"]="retrospective_research_gate_only";summaries.append(summary)
    _write_csv(base/"results.csv",evaluated);_write_csv(base/"summary.csv",summaries)
    evaluation_manifest={"prediction_sha256":manifest["prediction_sha256"],"rows":len(evaluated),"summary_rows":len(summaries),"contains_test_labels":True,"labels_used_after_prediction_freeze":True};(base/"evaluation_manifest.json").write_text(json.dumps(evaluation_manifest,sort_keys=True,indent=2)+"\n")
    return evaluated,summaries

