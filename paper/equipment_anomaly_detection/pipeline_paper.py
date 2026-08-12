"""The sole public v2 lifecycle CLI: run, status, and context-recomputing verify."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from typing import Sequence
from .runner_paper import PipelineConfig,PipelineStateError,run_pipeline,verify_completed
def _config(args)->PipelineConfig:return PipelineConfig(args.db_path,args.run_root,args.mode,args.catalog_path,args.acceptance_path)
def run(db_path:Path,run_root:Path,mode:str,catalog_path:Path,acceptance_path:Path)->int:
    run_pipeline(PipelineConfig(db_path,run_root,mode,catalog_path,acceptance_path));return 0
def status(db_path:Path,run_root:Path,mode:str,catalog_path:Path,acceptance_path:Path)->dict[str,object]:
    return verify_completed(PipelineConfig(db_path,run_root,mode,catalog_path,acceptance_path))
def verify(db_path:Path,run_root:Path,mode:str,catalog_path:Path,acceptance_path:Path)->dict[str,object]:
    return verify_completed(PipelineConfig(db_path,run_root,mode,catalog_path,acceptance_path))
def build_parser()->argparse.ArgumentParser:
    parser=argparse.ArgumentParser(prog="pipeline_paper.py");sub=parser.add_subparsers(dest="command",required=True)
    for command in ("run","status","verify"):
        child=sub.add_parser(command);child.add_argument("--db-path",required=True,type=Path);child.add_argument("--run-root",required=True,type=Path);child.add_argument("--mode",required=True,choices=("smoke","full"));child.add_argument("--catalog-path",required=True,type=Path);child.add_argument("--acceptance-path",required=True,type=Path)
    return parser
def main(argv:Sequence[str]|None=None)->int:
    args=build_parser().parse_args(argv);value=run(args.db_path,args.run_root,args.mode,args.catalog_path,args.acceptance_path) if args.command=="run" else (status(args.db_path,args.run_root,args.mode,args.catalog_path,args.acceptance_path) if args.command=="status" else verify(args.db_path,args.run_root,args.mode,args.catalog_path,args.acceptance_path));
    if args.command!="run":print(json.dumps(value,sort_keys=True,separators=(",",":")))
    return 0
if __name__=="__main__":raise SystemExit(main())
