from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from experiments.exp239_248_diverse_score_sources.cache_pipeline import cache_paths, dataset_names, source_root
from experiments.exp239_248_diverse_score_sources.calibration import empirical_candidates
from experiments.exp239_248_diverse_score_sources.config import ALPHAS
from experiments.exp239_248_diverse_score_sources.registry import registration


EXP236_PATH = Path("/Users/minho/Documents/Dataset/exp236_detection_coverage_expansion/exp236_frozen_predictions.jsonl")


def verify_file_hash(path: Path, expected: str | None = None) -> str:
    digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if expected is not None and digest!=expected: raise ValueError(f"hash mismatch: expected={expected} actual={digest}")
    return digest


def _baseline() -> dict[str, dict[str, object]]:
    return {row["dataset"]:row for row in map(json.loads,EXP236_PATH.read_text().splitlines())}


def build_prediction_row(*,experiment:int,source:str,variant:str,dataset:str,alpha:float,n_test:int,candidates:set[int],hard:set[int],standard:set[int],priority:set[int],abstained:bool,resolution_blocked:bool,attainable_p:float)->dict[str,object]:
    frozen=hard|standard|priority
    integer_list=lambda values:sorted(map(int,values))
    return {"experiment_id":f"experiment_{experiment}_{source}","experiment":experiment,"source":source,"variant":variant,"dataset":dataset,"alpha":float(alpha),"n_test":int(n_test),"candidate_indices":integer_list(candidates),"hard_confirmation_indices":integer_list(candidates&hard),"standard_priority_indices":integer_list(candidates&standard),"priority_confirmation_indices":integer_list(candidates&priority),"outside_lane_indices":integer_list(candidates-frozen),"frozen_hard_indices":integer_list(hard),"frozen_standard_indices":integer_list(standard),"frozen_priority_indices":integer_list(priority),"calibration_abstained":bool(abstained),"resolution_blocked":bool(resolution_blocked),"attainable_p":float(attainable_p),"contains_test_labels":False,"contains_evaluation_metrics":False,"hard_policy_changed":False}


def _variants(experiment:int,archive:np.lib.npyio.NpzFile)->dict[str,tuple[np.ndarray,np.ndarray,str]]:
    train=archive["train_oof"];test=archive["test_scores"]
    if experiment in (246,247):
        seeded_train=archive["artifact__train_scores_by_seed"] if "artifact__train_scores_by_seed" in archive else train
        seeded_test=archive["artifact__test_scores_by_seed"] if "artifact__test_scores_by_seed" in archive else test
        return {"default":(train,test,"direct"),"seed_intersection":(seeded_train,seeded_test,"intersection")}
    if experiment==248:
        typical_train=archive["artifact__train_typicality"] if "artifact__train_typicality" in archive else train
        typical_test=archive["artifact__test_typicality"] if "artifact__test_typicality" in archive else test
        return {"raw_nll":(train,test,"direct"),"typicality":(typical_train,typical_test,"direct")}
    return {"default":(train,test,"direct")}


def predict_source(root:Path,experiment:int,*,limit:int|None=None,names:tuple[str,...]|None=None)->tuple[Path,dict[str,object]]:
    item=registration(experiment);selected=names if names is not None else (dataset_names()[:limit] if limit else dataset_names());baseline=_baseline();rows=[]
    for dataset in selected:
        archive_path,metadata_path=cache_paths(root,experiment,dataset);metadata=json.loads(metadata_path.read_text());verify_file_hash(archive_path,metadata["cache_sha256"])
        hard=set(map(int,baseline[dataset]["hard_indices"]));standard=set(map(int,baseline[dataset]["standard_review_indices"]));priority=set(map(int,baseline[dataset]["priority_review_indices"]))
        with np.load(archive_path) as archive:
            for variant,(train_values,test_values,mode) in _variants(experiment,archive).items():
                for alpha in ALPHAS:
                    if metadata["abstained"]:
                        candidates=set();blocked=True;attainable=1.0
                    elif mode=="intersection":
                        results=[empirical_candidates(train_values[index],test_values[index],alpha) for index in range(len(train_values))];candidates=set.intersection(*(set(result.indices) for result in results));blocked=any(result.resolution_blocked for result in results);attainable=max(result.attainable_p for result in results)
                    else:
                        result=empirical_candidates(train_values,test_values,alpha);candidates=set(result.indices);blocked=result.resolution_blocked;attainable=result.attainable_p
                    rows.append(build_prediction_row(experiment=experiment,source=item.name,variant=variant,dataset=dataset,alpha=alpha,n_test=int(metadata["n_test"]),candidates=candidates,hard=hard,standard=standard,priority=priority,abstained=bool(metadata["abstained"]),resolution_blocked=blocked,attainable_p=attainable))
    rows.sort(key=lambda row:(row["dataset"],row["variant"],float(row["alpha"])))
    data=b"".join((json.dumps(row,sort_keys=True,separators=(",",":"))+"\n").encode() for row in rows);base=source_root(root,experiment);path=base/"frozen_predictions.jsonl";base.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=base,delete=False) as handle:temporary=Path(handle.name);handle.write(data);handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path);manifest={"experiment":experiment,"source":item.name,"rows":len(rows),"datasets":len(selected),"prediction_sha256":hashlib.sha256(data).hexdigest(),"contains_test_labels":False,"contains_evaluation_metrics":False,"hard_policy_changed":False};(base/"prediction_manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n");return path,manifest
