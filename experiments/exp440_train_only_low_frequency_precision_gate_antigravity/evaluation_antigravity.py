from __future__ import annotations
from hashlib import sha256
import json,math,os,shutil
from pathlib import Path
from uuid import uuid4
import numpy as np
from experiments.exp417_causal_multihorizon_shadow.inputs import verified_read_only_connection
from .config_antigravity import FROZEN_CONFIG
from .predecessor_antigravity import load_frozen_hard_coordinates_antigravity
from .reporting_antigravity import inspect_publication_antigravity
def decide_success_antigravity(facts):
 gates={'candidate_budget':facts['candidate_count']<=100,'rescued_datasets':facts['rescued_datasets']>=10,'added_false_positives':facts['added_false_positives']<=30,'candidate_precision':facts['candidate_precision']>=.35,'micro_f1_delta':facts['micro_f1_delta']>0,'macro_f1_delta':facts['macro_f1_delta']>0,'micro_ci':facts['micro_ci_lower']>0,'macro_ci':facts['macro_ci_lower']>0,'strong_dataset_guard':facts['maximum_strong_dataset_loss']<=.2}
 return {'schema':'exp440.decision.v1','gates':gates,'winner_selected':all(gates.values()),'review_only':True,'operational_alert_eligible':False}

def _metrics(c):
 tp,fp,fn,tn=(c[k] for k in ('tp','fp','fn','tn'));precision=tp/(tp+fp) if tp+fp else 0.0;recall=tp/(tp+fn) if tp+fn else 0.0;f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
 return {'precision':precision,'recall':recall,'f1':f1,'fpr':fp/(fp+tn) if fp+tn else 0.0}
def _confusion(labels,coordinates):
 selected=set(coordinates);c={'tp':0,'fp':0,'fn':0,'tn':0}
 for index,label in labels.items():c['tp' if (index in selected and label) else 'fp' if index in selected else 'fn' if label else 'tn']+=1
 return c
def _load_labels(datasets):
 result={}
 with verified_read_only_connection(Path(FROZEN_CONFIG.database_path),FROZEN_CONFIG.database_sha256) as connection:
  for dataset in datasets:
   rows=connection.execute("SELECT i.instance_index,i.label FROM instances i JOIN datasets d ON d.id=i.dataset_id WHERE d.name=? AND i.split='TEST' ORDER BY i.instance_index",(dataset,)).fetchall();values={}
   for index,label in rows:
    value=int(label)
    if value not in (0,1):raise ValueError('TEST label is not binary')
    values[int(index)]=value
   if not values:raise ValueError('TEST labels are absent')
   result[dataset]=values
 return result
def _tree_hash(root):return {p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir()) if p.is_file()}
def evaluate_exp440_antigravity():
 publication=inspect_publication_antigravity(Path(FROZEN_CONFIG.publication_path));before=_tree_hash(publication.root);datasets=tuple(json.loads(x)['dataset'] for x in (publication.root/'dataset_summary_antigravity.jsonl').read_text().splitlines() if x);hard=load_frozen_hard_coordinates_antigravity();rows=tuple(json.loads(x) for x in (publication.root/'prediction_antigravity.jsonl').read_text().splitlines() if x);candidates=frozenset((x['dataset'],int(x['instance_index'])) for x in rows)
 if candidates & hard or len(candidates)>100:raise ValueError('prediction budget or Hard exclusion differs')
 labels=_load_labels(datasets);per=[];base_total={k:0 for k in ('tp','fp','fn','tn')};union_total=dict(base_total);added_tp=added_fp=0;rescued=[];regressed=[];max_loss=0.0
 for dataset in datasets:
  base={i for d,i in hard if d==dataset};extra={i for d,i in candidates if d==dataset};b=_confusion(labels[dataset],base);u=_confusion(labels[dataset],base|extra);bm=_metrics(b);um=_metrics(u);delta=um['f1']-bm['f1'];new_tp=sum(labels[dataset].get(i)==1 for i in extra);new_fp=sum(labels[dataset].get(i)==0 for i in extra);added_tp+=new_tp;added_fp+=new_fp
  if new_tp:rescued.append(dataset)
  if delta<0:regressed.append(dataset)
  if bm['f1']>=.8:max_loss=max(max_loss,bm['f1']-um['f1'])
  for k in base_total:base_total[k]+=b[k];union_total[k]+=u[k]
  per.append({'dataset':dataset,'baseline':b,'union':u,'baseline_f1':bm['f1'],'union_f1':um['f1'],'f1_delta':delta,'added_tp':new_tp,'added_fp':new_fp})
 base_metrics=_metrics(base_total);union_metrics=_metrics(union_total);macro_delta=float(np.mean([x['f1_delta'] for x in per]));arr=np.asarray([[x['baseline'][k] for k in ('tp','fp','fn','tn')]+[x['union'][k] for k in ('tp','fp','fn','tn')]+[x['f1_delta']] for x in per],dtype=np.float64);rng=np.random.default_rng(FROZEN_CONFIG.bootstrap_seed);micro=[];macro=[]
 for _ in range(FROZEN_CONFIG.bootstrap_samples):
  sample=arr[rng.integers(0,len(arr),len(arr))];b=sample[:,:4].sum(0);u=sample[:,4:8].sum(0);micro.append(_metrics(dict(zip(('tp','fp','fn','tn'),u)))['f1']-_metrics(dict(zip(('tp','fp','fn','tn'),b)))['f1']);macro.append(float(sample[:,8].mean()))
 micro_ci=(float(np.quantile(micro,.025)),float(np.quantile(micro,.975)));macro_ci=(float(np.quantile(macro,.025)),float(np.quantile(macro,.975)));facts={'candidate_count':len(candidates),'rescued_datasets':len(set(rescued)),'added_false_positives':added_fp,'candidate_precision':added_tp/(added_tp+added_fp) if added_tp+added_fp else 0.0,'micro_f1_delta':union_metrics['f1']-base_metrics['f1'],'macro_f1_delta':macro_delta,'micro_ci_lower':micro_ci[0],'macro_ci_lower':macro_ci[0],'maximum_strong_dataset_loss':max_loss};decision=decide_success_antigravity(facts)
 payloads={'evaluation_antigravity.json':{'schema':'exp440.evaluation.v1','prediction_sha256':publication.prediction_sha256,'facts':facts},'dataset_metrics_antigravity.jsonl':per,'rescued_datasets_antigravity.jsonl':({'dataset':x} for x in sorted(set(rescued))),'regressed_datasets_antigravity.jsonl':({'dataset':x} for x in sorted(set(regressed))),'confusion_antigravity.json':{'baseline':base_total,'baseline_metrics':base_metrics,'union':union_total,'union_metrics':union_metrics},'bootstrap_antigravity.json':{'seed':FROZEN_CONFIG.bootstrap_seed,'samples':FROZEN_CONFIG.bootstrap_samples,'micro_f1_delta_ci95':micro_ci,'macro_f1_delta_ci95':macro_ci},'decision_antigravity.json':decision};root=Path(FROZEN_CONFIG.evaluation_path)
 if root.exists():raise FileExistsError(root)
 staging=root.with_name(f'.evaluation_{uuid4().hex}_staging_antigravity');staging.mkdir(mode=0o700)
 try:
  hashes={}
  for name,value in payloads.items():
   raw=b''.join((json.dumps(x,sort_keys=True,separators=(',',':'))+'\n').encode() for x in value) if name.endswith('.jsonl') else (json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode();(staging/name).write_bytes(raw);hashes[name]=sha256(raw).hexdigest()
  (staging/'artifact_sha256_antigravity.json').write_text(json.dumps({'files':hashes,'prediction_sha256':publication.prediction_sha256},sort_keys=True,separators=(',',':'))+'\n');os.rename(staging,root)
 except BaseException:
  if staging.exists():shutil.rmtree(staging)
  raise
 if _tree_hash(publication.root)!=before:raise ValueError('prediction changed during evaluation')
 return decision
