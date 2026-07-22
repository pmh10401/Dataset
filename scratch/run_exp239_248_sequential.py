from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exp239_248_diverse_score_sources.cache_pipeline import source_root
from experiments.exp239_248_diverse_score_sources.config import OUTPUT_ROOT
from experiments.exp239_248_diverse_score_sources.registry import SOURCE_REGISTRY

CANONICAL_STATE_PATH = Path("/Users/minho/Documents/Dataset/rank_experiments_sequential_state.json")
QUEUE_LOCK_PATH = Path("/Users/minho/Documents/Dataset/exp239_248_diverse_score_sources.lock")


def _experiment_id(experiment:int)->str:
    return f"experiment_{experiment}_{SOURCE_REGISTRY[experiment].name}"


def _canonical_state()->dict[str,object]:
    if CANONICAL_STATE_PATH.exists():
        state=json.loads(CANONICAL_STATE_PATH.read_text())
    else:
        state={}
    state.setdefault("completed",[]);state.setdefault("history",[]);state.setdefault("queue",[]);state.setdefault("running",None)
    return state


def _write_canonical(state:dict[str,object])->None:
    temporary=CANONICAL_STATE_PATH.with_suffix(".exp239_248.tmp")
    temporary.write_text(json.dumps(state,sort_keys=True,indent=2)+"\n")
    os.replace(temporary,CANONICAL_STATE_PATH)


def build_queue()->list[dict[str,object]]:
    return [{"experiment":number,"source":item.name,"status":"pending","model_based":item.model_based} for number,item in SOURCE_REGISTRY.items()]


def _write(path:Path,payload:object)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n")


def _completed(root:Path,experiment:int)->bool:
    path=source_root(root,experiment)/"independent_reaggregation.json"
    return path.exists() and bool(json.loads(path.read_text()).get("passed"))


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--output-root",type=Path,default=OUTPUT_ROOT);parser.add_argument("--model-device",choices=("cpu","mps"),default="cpu");parser.add_argument("--dataset-limit",type=int);parser.add_argument("--reset-cache",action="store_true");args=parser.parse_args();queue=build_queue();state_path=args.output_root/"queue_status.json"
    QUEUE_LOCK_PATH.parent.mkdir(parents=True,exist_ok=True)
    with QUEUE_LOCK_PATH.open("w") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit("Exp239-248 sequential queue is already running")
        ids=[_experiment_id(int(item["experiment"])) for item in queue]
        canonical=_canonical_state()
        if canonical.get("running"):raise SystemExit(f"another experiment is recorded as running: {canonical['running']}")
        completed=set(canonical["completed"])
        canonical["queue"]=[item for item in canonical["queue"] if item not in ids]+[item for item in ids if item not in completed]
        _write_canonical(canonical)
        for item in queue:
            experiment=int(item["experiment"]);experiment_id=_experiment_id(experiment)
            if _completed(args.output_root,experiment):
                canonical=_canonical_state()
                if experiment_id not in canonical["completed"]:canonical["completed"].append(experiment_id)
                canonical["queue"]=[value for value in canonical["queue"] if value!=experiment_id]
                _write_canonical(canonical)
                item["status"]="completed";_write(state_path,{"updated_at":time.time(),"queue":queue});continue
            item["status"]="running";_write(state_path,{"updated_at":time.time(),"queue":queue})
            command=[sys.executable,"-m","experiments.exp239_248_diverse_score_sources.run_pipeline","--stage","all","--source",str(experiment),"--output-root",str(args.output_root),"--workers","1" if item["model_based"] else "7","--device",args.model_device if item["model_based"] else "cpu"]
            if args.dataset_limit:command.extend(["--dataset-limit",str(args.dataset_limit)])
            if args.reset_cache:command.append("--reset-cache")
            started=time.time();process=subprocess.Popen(command,cwd=REPO_ROOT)
            canonical=_canonical_state();canonical["running"]={"id":experiment_id,"label":f"Exp{experiment} - {item['source'].replace('_',' ')}","pid":os.getpid(),"child_pid":process.pid,"started_at":time.strftime("%Y-%m-%d %H:%M:%S"),"expected_datasets":args.dataset_limit or 1117};_write_canonical(canonical)
            returncode=process.wait();canonical=_canonical_state();canonical["running"]=None
            canonical.setdefault("history",[]).append({"id":experiment_id,"finished_at":time.strftime("%Y-%m-%d %H:%M:%S"),"return_code":returncode,"elapsed_minutes":round((time.time()-started)/60,2)})
            if returncode==0:
                if experiment_id not in canonical["completed"]:canonical["completed"].append(experiment_id)
                canonical["queue"]=[value for value in canonical["queue"] if value!=experiment_id]
            _write_canonical(canonical)
            if returncode:
                item["status"]="failed";item["returncode"]=returncode;_write(state_path,{"updated_at":time.time(),"queue":queue});raise SystemExit(returncode)
            item["status"]="completed";_write(state_path,{"updated_at":time.time(),"queue":queue})
        _write(state_path,{"updated_at":time.time(),"status":"completed","queue":queue})


if __name__=="__main__":main()
