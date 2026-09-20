"""Independent rank-sum metric and frozen-routing arithmetic verification."""
from pathlib import Path
from contextlib import closing
import json,hashlib
import numpy as np
from scipy.stats import rankdata
from experiments.exp491_gain_prediction_antigravity.runtime_antigravity import readonly_sqlite_antigravity,DATABASE_PATH_ANTIGRAVITY
from experiments.exp479_missing_score_recovery_antigravity.evaluation_antigravity import load_retained_test_labels_antigravity
root=Path(__file__).resolve().parents[1];p=root/'experiments/exp493_stable_meta_routing_antigravity'
def read(f):return json.loads(f.read_text())
def digest(f):return hashlib.sha256(f.read_bytes()).hexdigest()
r=read(p/'results_antigravity.json');out=read(p/'outcomes_antigravity.json')['records'];dec=read(p/'decisions_antigravity.json')
assert r['families']==len(out)==125==len({x['source_family'] for x in out})
assert digest(p/'outcomes_antigravity.json')==r['outcomes_sha256']
assert digest(p/'decisions_antigravity.json')==r['decisions_sha256']
assert [x['dataset'] for x in out]==dec['datasets']
maximum=0.;vectors=0
with closing(readonly_sqlite_antigravity(DATABASE_PATH_ANTIGRAVITY)) as con:
 for row in out:
  dataset=row['dataset']; meta=read(p/'score_checkpoints_antigravity'/f'{dataset}_meta_antigravity.json')
  labels=load_retained_test_labels_antigravity(con,dataset,tuple(meta['test_indices']))
  pos=labels==1;npos=pos.sum();nneg=(~pos).sum()
  mini_path=root/'experiments/exp489_lof_score_head_antigravity/score_checkpoints_antigravity'/f'{dataset}_antigravity_scores_antigravity.npz'
  pair_path=root/'experiments/exp492_stable_normalization_antigravity/score_checkpoints_antigravity'/f'{dataset}_scores_antigravity.npz'
  for source in (mini_path,pair_path):
   sidecar=source.with_name(source.name.replace('_scores_antigravity.npz','_meta_antigravity.json'))
   assert digest(source)==read(sidecar)['score_sha256']
  assert digest(p/'score_checkpoints_antigravity'/f'{dataset}_scores_antigravity.npz')==meta['score_sha256']
  with np.load(mini_path) as mini,np.load(pair_path) as pair,np.load(p/'score_checkpoints_antigravity'/f'{dataset}_scores_antigravity.npz') as fresh:
   for branch in ('old','stable'):
    scores=[mini['control_test_scores'],fresh['multirocket_test'],pair[f'hydra_local_gap__{branch}__test_scores'],pair[f'hydra_multirocket_local_gap__{branch}__test_scores']]
    for v,expected in zip(scores,row[branch]):
     auc=(rankdata(v)[pos].sum()-npos*(npos+1)/2)/(npos*nneg)
     zf=int((v[pos]>v[~pos].max()).sum())
     error=abs(auc-expected['auc']);maximum=max(maximum,error)
     assert error<1e-12 and zf==expected['zf'];vectors+=1
checks=0
for name,model in dec['models'].items():
 pred=np.asarray(model['predictions'])
 for head,offset in [('auc',0),('zf_sign',3)]:
  route=np.argmax(np.column_stack((np.zeros(125),pred[:,offset:offset+3])),axis=1)
  assert route.tolist()==model['routes'][head]
  for branch in ('old','stable'):
   selected=[row[branch][i] for row,i in zip(out,route)]
   expected=r['models'][name]['evaluations'][branch][head]
   assert abs(np.mean([m['auc'] for m in selected])-expected['mean_auc'])<1e-12
   assert sum(m['zf'] for m in selected)==expected['total_zf'];checks+=1
for file,sha in {**r['binding']['inputs'],**r['binding']['upstream_meta_sha256']}.items(): assert digest(Path(file))==sha
boundary=read(p/'labels_opened_antigravity.json')
from experiments.exp491_gain_prediction_antigravity.runtime_antigravity import sha256_json_antigravity
for dataset,expected in boundary['metadata'].items():
 assert sha256_json_antigravity(read(p/'score_checkpoints_antigravity'/f'{dataset}_meta_antigravity.json'))==expected
summary=dict(state='passed',metric_vectors_checked=vectors,auc_max_abs_error=float(maximum),zf_exact=True,route_summaries_checked=checks,prior_artifacts_unchanged=True,label_boundary_metadata_unchanged=True)
(p/'verification_antigravity.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
