"""Deterministic no-clobber review-only Exp440 publication."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json,os,shutil
from pathlib import Path
from uuid import uuid4
from .config_antigravity import FROZEN_CONFIG
from .checkpoint_antigravity import Exp440CheckpointStoreAntigravity
def _json(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()
def _jsonl(values):return b''.join(_json(x) for x in values)
@dataclass(frozen=True)
class FrozenPublicationAntigravity:
 root:Path;manifest:dict[str,object]
 @property
 def prediction_sha256(self):return str(self.manifest['prediction_sha256'])
def publish_payloads_antigravity(root:Path,*,datasets:tuple[str,...],candidates:tuple[dict[str,object],...],backend:dict[str,object],reviewed_commit:str,abstentions:tuple[dict[str,object],...]=(),train_audits:tuple[dict[str,object],...]=())->FrozenPublicationAntigravity:
 root=Path(root)
 if root.exists():raise FileExistsError(root)
 if datasets!=tuple(sorted(set(datasets))) or len(candidates)>FROZEN_CONFIG.outside_hard_budget:raise ValueError('publication population or budget is invalid')
 staging=root.with_name(f'.publication_{uuid4().hex}_staging_antigravity');staging.mkdir(mode=0o700)
 try:
  prediction=[]
  for item in candidates:
   row=dict(item);row['contains_test_labels']=False;row['contains_evaluation_metrics']=False;prediction.append(row)
  files={'prediction_antigravity.jsonl':_jsonl(prediction),'dataset_summary_antigravity.jsonl':_jsonl({'dataset':d,'candidate_count':sum(x.get('dataset')==d for x in candidates)} for d in datasets),'retained_candidate_ranking_antigravity.jsonl':_jsonl(candidates),'depth_evidence_antigravity.jsonl':_jsonl({'dataset':x['dataset'],'instance_index':x['instance_index'],'depths':x.get('depth_scores',[]),'agreement_count':x.get('agreement_count')} for x in candidates),'abstention_ledger_antigravity.jsonl':_jsonl(abstentions),'train_exclusion_audit_antigravity.jsonl':_jsonl(train_audits),'backend_decision_antigravity.json':_json(backend),'status_antigravity.json':_json({'schema':'exp440.status.v1','review_only':True,'operational_alert_eligible':False,'candidate_count':len(candidates),'label_access_count':0})}
  hashes={name:sha256(raw).hexdigest() for name,raw in files.items()};manifest={'schema':'exp440.publication.v1','experiment_id':'Exp440','run_identity':FROZEN_CONFIG.run_identity,'reviewed_commit':reviewed_commit,'dataset_count':len(datasets),'candidate_count':len(candidates),'prediction_sha256':hashes['prediction_antigravity.jsonl'],'artifact_sha256':hashes,'contains_test_labels':False}
  files['manifest_antigravity.json']=_json(manifest);files['artifact_sha256_antigravity.json']=_json({'files':{**hashes,'manifest_antigravity.json':sha256(files['manifest_antigravity.json']).hexdigest()}})
  for name,raw in files.items():(staging/name).write_bytes(raw)
  if root.exists():raise FileExistsError(root)
  os.rename(staging,root)
 except BaseException:
  if staging.exists():shutil.rmtree(staging)
  raise
 return inspect_publication_antigravity(root)
def inspect_publication_antigravity(root:Path)->FrozenPublicationAntigravity:
 root=Path(root);manifest=json.loads((root/'manifest_antigravity.json').read_bytes());hashes=json.loads((root/'artifact_sha256_antigravity.json').read_bytes())['files']
 for name,digest in hashes.items():
  if sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('publication hash mismatch')
 if manifest.get('contains_test_labels') is not False or manifest.get('prediction_sha256')!=hashes['prediction_antigravity.jsonl']:raise ValueError('publication manifest is invalid')
 return FrozenPublicationAntigravity(root,manifest)

def publish_exp440_antigravity()->FrozenPublicationAntigravity:
 store=Exp440CheckpointStoreAntigravity.open_read_only(Path(FROZEN_CONFIG.checkpoint_path));output=Path(FROZEN_CONFIG.output_path)
 progress=json.loads((output/'progress_antigravity.json').read_text());pid=json.loads((output/'run_antigravity.pid').read_text())
 if store.completed!=FROZEN_CONFIG.dataset_count or progress.get('stage')!='checkpoint_complete' or progress.get('error')!=0 or pid.get('active') is not False:raise ValueError('Exp440 runtime is not publishable')
 candidates=tuple(json.loads((store.root/'retained_candidates_antigravity.json').read_text()));readiness=json.loads(Path(FROZEN_CONFIG.readiness_path).read_text());results=tuple(store.results());abstentions=tuple({'dataset':r.dataset,'instance_index':i,'reason':'nonfinite_or_unavailable_group'} for r in results for i in r.abstention_indices);audits=tuple({'dataset':r.dataset,'excluded_train_rows':r.excluded_train_rows} for r in results if r.excluded_train_rows)
 return publish_payloads_antigravity(Path(FROZEN_CONFIG.publication_path),datasets=store.datasets,candidates=candidates,backend={'backend':readiness['backend'],'workers':readiness['workers'],'acceleration_evidence':readiness['acceleration_evidence']},reviewed_commit=store.reviewed_commit,abstentions=abstentions,train_audits=audits)
