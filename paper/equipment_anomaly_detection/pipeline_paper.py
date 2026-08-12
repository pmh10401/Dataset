"""The public Task 8 CLI exposes only run, status, and verify."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .runner_paper import PipelineConfig, run_pipeline, verify_completed
def _config(args): return PipelineConfig(args.db_path,args.run_root,args.mode,tuple(args.smoke_dataset),args.acceptance_path)
def build_parser():
    parser=argparse.ArgumentParser(prog="pipeline_paper.py"); commands=parser.add_subparsers(dest="command",required=True)
    for command in ("run","status","verify"):
        child=commands.add_parser(command); child.add_argument("--db-path",required=True,type=Path); child.add_argument("--run-root",required=True,type=Path); child.add_argument("--mode",required=True,choices=("smoke","full")); child.add_argument("--smoke-dataset",action="append",default=[]); child.add_argument("--acceptance-path",required=True,type=Path)
    return parser
def main(argv=None):
    args=build_parser().parse_args(argv); value=run_pipeline(_config(args)) if args.command=="run" else verify_completed(_config(args))
    if args.command!="run": print(json.dumps(value,sort_keys=True,separators=(",",":")))
    return 0
if __name__=="__main__": raise SystemExit(main())
