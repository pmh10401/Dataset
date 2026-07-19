from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp239_248_diverse_score_sources.cache_pipeline import cache_source, dataset_names, source_root
from experiments.exp239_248_diverse_score_sources.config import DB_PATH, EXTERNAL_STATUS_PATH, HEARTBEAT_PATH, OUTPUT_ROOT
from experiments.exp239_248_diverse_score_sources.evaluation import evaluate_source
from experiments.exp239_248_diverse_score_sources.foundation import database_hash
from experiments.exp239_248_diverse_score_sources.independent_reaggregate import reaggregate
from experiments.exp239_248_diverse_score_sources.prediction import predict_source, verify_file_hash
from experiments.exp239_248_diverse_score_sources.registry import registration
from experiments.exp239_248_diverse_score_sources.reporting import write_manifest, write_source_report


def _atomic_json(path:Path,payload:object)->None:
    path.parent.mkdir(parents=True,exist_ok=True);data=(json.dumps(payload,sort_keys=True,indent=2)+"\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as handle:temporary=Path(handle.name);handle.write(data);handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def write_status(root:Path,experiment:int,status:str,phase:str,done:int,total:int,current:str|None=None,errors:int=0)->dict[str,object]:
    item=registration(experiment);now=time.time();payload={"id":f"experiment_{experiment}_{item.name}","experiment_id":f"experiment_{experiment}_{item.name}","label":f"Exp{experiment} - {item.name.replace('_',' ')}","source":item.name,"status":status,"phase":phase,"pid":os.getpid(),"child_pid":os.getpid(),"done":done,"datasets_done":done,"expected":total,"expected_datasets":total,"progress_percent":100*done/max(1,total),"current_dataset":current,"errors":errors,"updated_unix":now,"updated_at":time.strftime("%Y-%m-%d %H:%M:%S"),"scope":"retrospective_score_source_qualification","comparison_eligible":False,"contains_test_labels":False}
    base=source_root(root,experiment);_atomic_json(base/"status.json",payload);_atomic_json(EXTERNAL_STATUS_PATH,payload);_atomic_json(HEARTBEAT_PATH,payload);return payload


def _selected_names(limit:int|None,list_file:Path|None)->tuple[str,...]:
    population=dataset_names()
    if list_file is None:return population[:limit] if limit else population
    requested=tuple(line.strip() for line in list_file.read_text().splitlines() if line.strip())
    unknown=sorted(set(requested)-set(population))
    if unknown:raise ValueError(f"unknown datasets in list: {unknown}")
    if len(requested)!=len(set(requested)):raise ValueError("dataset list contains duplicates")
    return requested


def audit(root:Path,experiment:int,names:tuple[str,...])->dict[str,object]:
    payload={"experiment":experiment,"source":registration(experiment).name,"database_path":str(DB_PATH),"database_sha256":database_hash(),"datasets":len(names),"full_population":len(names)==1117,"contains_test_labels":False,"retrospective":True};base=source_root(root,experiment);base.mkdir(parents=True,exist_ok=True);_atomic_json(base/"audit.json",payload);return payload


def run(args:argparse.Namespace)->None:
    experiment=registration(args.source).experiment;names=_selected_names(args.dataset_limit,args.dataset_list_file);total=len(names);root=Path(args.output_root);write_status(root,experiment,"running",args.stage,0,total)
    if args.stage in ("audit","all"):audit(root,experiment,names)
    if args.stage in ("cache","all"):
        def progress(done:int,expected:int,current:str,errors:int)->None:
            write_status(root,experiment,"running","cache",done,expected,current,errors);print(f"progress {done}/{expected} source={registration(experiment).name} last={current} errors={errors}",flush=True)
        cache_source(root,experiment,workers=args.workers,names=names,device=args.device,reset=args.reset_cache,progress=progress)
    if args.stage in ("predict","all"):write_status(root,experiment,"running","predict",0,total);predict_source(root,experiment,names=names);write_status(root,experiment,"running","predict",total,total)
    if args.stage in ("evaluate","all"):
        write_status(root,experiment,"running","evaluate",0,total);evaluate_source(root,experiment);write_source_report(root,experiment);result=reaggregate(root,experiment)
        if not result["passed"]:raise RuntimeError("independent reaggregation failed")
        base=source_root(root,experiment);hashes={name:verify_file_hash(base/name) for name in ("frozen_predictions.jsonl","prediction_manifest.json","results.csv","summary.csv","independent_reaggregation.json")};write_manifest(root,experiment,hashes);write_status(root,experiment,"completed","completed",total,total)


def parser()->argparse.ArgumentParser:
    value=argparse.ArgumentParser();value.add_argument("--stage",choices=("audit","cache","predict","evaluate","all"),default="all");value.add_argument("--source",required=True);value.add_argument("--output-root",type=Path,default=OUTPUT_ROOT);value.add_argument("--dataset-limit",type=int);value.add_argument("--dataset-list-file",type=Path);value.add_argument("--workers",type=int,default=7);value.add_argument("--device",choices=("cpu","mps"),default="cpu");value.add_argument("--reset-cache",action="store_true");return value


def main()->None:run(parser().parse_args())


if __name__=="__main__":main()
