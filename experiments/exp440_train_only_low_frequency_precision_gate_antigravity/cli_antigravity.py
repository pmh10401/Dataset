from __future__ import annotations
import argparse,subprocess
from pathlib import Path
from .config_antigravity import FROZEN_CONFIG
def build_parser_antigravity():
 p=argparse.ArgumentParser();subs=p.add_subparsers(dest='command',required=True)
 for name in ('readiness','new','resume','inspect','evaluate'):subs.add_parser(name)
 return p
def _head():return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
def main_antigravity(argv=None):
 command=build_parser_antigravity().parse_args(argv).command
 if command=='readiness':
  from .readiness_antigravity import run_readiness_antigravity
  return run_readiness_antigravity(_head())
 if command in {'new','resume'}:
  from .readiness_antigravity import validate_readiness_for_run_antigravity
  validate_readiness_for_run_antigravity(_head())
  from .runtime_antigravity import run_exp440_new_antigravity,resume_exp440_antigravity
  fn=run_exp440_new_antigravity if command=='new' else resume_exp440_antigravity;result=fn(reviewed_commit=_head())
  from .reporting_antigravity import publish_exp440_antigravity
  publication=publish_exp440_antigravity();return {'runtime':result,'prediction_sha256':publication.prediction_sha256}
 if command=='inspect':
  from .reporting_antigravity import inspect_publication_antigravity
  return inspect_publication_antigravity(Path(FROZEN_CONFIG.publication_path))
 from .evaluation_antigravity import evaluate_exp440_antigravity
 return evaluate_exp440_antigravity()
