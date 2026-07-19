from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from experiments.exp226_235_hard_alert_quality.database import load_test_labels_for_evaluation
from experiments.exp239_248_diverse_score_sources.cache_pipeline import source_root
from experiments.exp239_248_diverse_score_sources.evaluation import VIEWS, verify_prediction_manifest


def reaggregate(root:Path,experiment:int)->dict[str,object]:
    base=source_root(root,experiment);prediction=base/"frozen_predictions.jsonl";verify_prediction_manifest(prediction,base/"prediction_manifest.json");rows=[json.loads(line) for line in prediction.read_text().splitlines()]
    with (base/"summary.csv").open(newline="",encoding="utf-8") as handle:reported={(row["variant"],float(row["alpha"])):row for row in csv.DictReader(handle)}
    aggregate={};violations=[]
    for row in rows:
        key=(row["variant"],float(row["alpha"]));bucket=aggregate.setdefault(key,{view:{"count":0,"tp":0,"fp":0} for view in VIEWS});truth=set(np.flatnonzero(load_test_labels_for_evaluation(row["dataset"])==1).astype(int));n_test=int(row["n_test"])
        hard=set(row["frozen_hard_indices"]);standard=set(row["frozen_standard_indices"]);priority=set(row["frozen_priority_indices"])
        if hard&standard or hard&priority or standard&priority:violations.append([row["dataset"],row["variant"],row["alpha"],"frozen_lane_overlap"])
        for view in VIEWS:
            indices=set(row[f"{view}_indices"])
            if any(value<0 or value>=n_test for value in indices):violations.append([row["dataset"],row["variant"],row["alpha"],f"{view}_range"])
            bucket[view]["count"]+=len(indices);bucket[view]["tp"]+=len(indices&truth);bucket[view]["fp"]+=len(indices-truth)
    mismatches=[]
    for key,bucket in aggregate.items():
        report=reported[key]
        for view in VIEWS:
            for metric in ("count","tp","fp"):
                if bucket[view][metric]!=int(report[f"{view}_{metric}"]):mismatches.append([*key,view,metric,bucket[view][metric],report[f"{view}_{metric}"]])
    result={"experiment":experiment,"prediction_rows":len(rows),"violations":violations,"mismatches":mismatches,"passed":not violations and not mismatches};(base/"independent_reaggregation.json").write_text(json.dumps(result,indent=2)+"\n");return result
