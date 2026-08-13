"""Minimal result-focused Exp451 command adapter."""
from __future__ import annotations
import argparse, json, re
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True,slots=True)
class CliDependenciesAntigravity:
    predict:Callable; load_prediction:Callable; evaluate:Callable; predecessor:Callable; inspect:Callable; public:Callable

def build_parser_antigravity():
    parser=argparse.ArgumentParser(prog="exp451-antigravity",allow_abbrev=False); commands=parser.add_subparsers(dest="command",required=True)
    for name in ("predict","evaluate"):
        command=commands.add_parser(name,allow_abbrev=False); command.add_argument("--reviewed-commit",required=True)
    commands.add_parser("inspect",allow_abbrev=False)
    return parser

def _dependencies_antigravity():
    from .prediction_antigravity import load_frozen_prediction_authority_antigravity,run_prediction_antigravity
    from .evaluation_antigravity import evaluate_antigravity,OFFICIAL_EVALUATION_ROOT_ANTIGRAVITY
    from .predecessor_antigravity import verify_exp448_predecessor_antigravity
    from .replay_antigravity import inspect_replay_antigravity
    from .dashboard_export_antigravity import public_result_antigravity
    return CliDependenciesAntigravity(run_prediction_antigravity,load_frozen_prediction_authority_antigravity,evaluate_antigravity,verify_exp448_predecessor_antigravity,lambda authority,predecessor:inspect_replay_antigravity(OFFICIAL_EVALUATION_ROOT_ANTIGRAVITY,expected_prediction_authority=authority,expected_predecessor_authority=predecessor),public_result_antigravity)

def dispatch_antigravity(argv=None,dependencies=None):
    args=build_parser_antigravity().parse_args(argv); dependencies=dependencies or _dependencies_antigravity()
    if hasattr(args,"reviewed_commit") and not re.fullmatch(r"[0-9a-f]{40}",args.reviewed_commit): raise ValueError("reviewed commit invalid")
    if args.command=="predict":
        authority=dependencies.predict(args.reviewed_commit); value={"authority":authority.to_payload_antigravity(),"completed":48}
    elif args.command=="evaluate":
        authority=dependencies.load_prediction(); value=dependencies.public(dependencies.evaluate(authority,args.reviewed_commit))
    else:
        authority=dependencies.load_prediction(); value=dependencies.public(dependencies.inspect(authority,dependencies.predecessor()))
    print(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)); return value

def main_antigravity(): dispatch_antigravity()
if __name__=="__main__": main_antigravity()
