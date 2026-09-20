"""Independent score, metric, population and aggregation audit for Exp496."""
from pathlib import Path
from hashlib import sha256
from contextlib import closing
import json
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_curve
from experiments.exp491_gain_prediction_antigravity.runtime_antigravity import readonly_sqlite_antigravity,DATABASE_PATH_ANTIGRAVITY,sha256_json_antigravity
from experiments.exp479_missing_score_recovery_antigravity.evaluation_antigravity import load_retained_test_labels_antigravity

root=Path(__file__).resolve().parents[1];p=root/'experiments/exp496_subspace_consensus_antigravity';base=root/'experiments/exp494_all_datasets_low_fpr_antigravity'
read=lambda f:json.loads(f.read_text())
digest=lambda f:sha256(f.read_bytes()).hexdigest()
r=read(p/'full_results_antigravity.json');m=read(p/'full_manifest_antigravity.json');boundary=read(p/'full_labels_opened_antigravity.json')
assert m['state']==r['state']=='complete_frozen_full_comparison'
assert digest(p/'full_results_antigravity.json')==m['result_sha256']
assert r['binding_sha256']==m['binding_sha256']==sha256_json_antigravity(m['binding'])
assert digest(p/'full_labels_opened_antigravity.json')==r['boundary_sha256']
assert digest(DATABASE_PATH_ANTIGRAVITY)==m['binding']['database_sha256']
assert digest(base/'results_antigravity.json')==m['binding']['source494_result_sha256']
assert digest(p/'pilot_results_antigravity.json')==m['binding']['selection_sha256']
for relative,expected in m['binding']['code'].items():assert digest(root/relative)==expected
assert digest(p/'full_protocol_antigravity.html')==m['binding']['protocol_sha256']
assert digest(p/'acceleration_results_antigravity.json')==m['binding']['acceleration_sha256']
population={x['dataset']:x['source_family'] for x in read(base/'run_manifest_antigravity.json')['binding']['population']}
assert {x['dataset']:x['source_family'] for x in r['records']}==population and len(r['records'])==1117
assert len(set(population.values()))==128 and r['selected']=='half_q90'
vectors=0;max_error=0.
with closing(readonly_sqlite_antigravity(DATABASE_PATH_ANTIGRAVITY)) as con:
    for row in r['records']:
        d=row['dataset'];meta=read(p/'full_scores_antigravity'/f'{d}_meta_antigravity.json');npz=p/'full_scores_antigravity'/f'{d}_scores_antigravity.npz'
        assert digest(npz)==meta['score_sha256']==boundary['scores'][d]
        assert sha256_json_antigravity(meta)==boundary['metadata'][d] and meta['binding_sha256']==m['binding_sha256']
        source=base/'score_checkpoints_antigravity'/f'{d}_meta_antigravity.json'
        assert digest(source)==meta['source_meta_sha256'] and read(source)['test_indices']==meta['test_indices']
        y=load_retained_test_labels_antigravity(con,d,tuple(meta['test_indices']));normal=y==0;nn=int(normal.sum());np_=int((~normal).sum())
        with np.load(npz) as arrays,np.load(base/'score_checkpoints_antigravity'/f'{d}_scores_antigravity.npz') as old:
            np.testing.assert_allclose(arrays['baseline'],old['test__0'],rtol=1e-10,atol=1e-8)
            for name in ('baseline','candidate'):
                s=arrays[name];stored=row['metrics'][name];assert np.isfinite(s).all() and len(s)==len(y)
                auc=float((rankdata(s)[~normal].sum()-np_*(np_+1)/2)/(nn*np_));max_error=max(max_error,abs(auc-stored['auc']))
                assert abs(auc-stored['auc'])<1e-12
                fpr,tpr,_=roc_curve(y,s,drop_intermediate=False);fp=np.rint(fpr*nn).astype(int);budgets=nn*np.array([0,1,5,10])//1000
                target=[tpr[fp<=b].max() for b in budgets];np.testing.assert_allclose(target,stored['tpr'],rtol=0,atol=1e-15)
                thresholds=np.sort(s[normal])[::-1][budgets]
                assert stored['tp']==[int((s[~normal]>t).sum()) for t in thresholds]
                assert stored['fp']==[int((s[normal]>t).sum()) for t in thresholds]
                assert stored['zf']==stored['tp'][0] and stored['n_normal']==nn and stored['n_anomaly']==np_
                vectors+=1
summaries={}
families=sorted(set(population.values()))
for name in ('baseline','candidate'):
    ms=[x['metrics'][name] for x in r['records']];family=np.asarray([[np.mean([x['metrics'][name]['tpr'][j] for x in r['records'] if x['source_family']==g]) for j in range(4)] for g in families])
    summary=r['summaries'][name];summaries[name]=family.mean(axis=0)
    np.testing.assert_allclose(summary['family_macro_tpr'],family.mean(axis=0),rtol=0,atol=1e-12)
    np.testing.assert_allclose(summary['dataset_macro_tpr'],np.mean([x['tpr'] for x in ms],axis=0),rtol=0,atol=1e-12)
    np.testing.assert_allclose(summary['micro_tpr'],np.sum([x['tp'] for x in ms],axis=0)/sum(x['n_anomaly'] for x in ms),rtol=0,atol=1e-12)
    assert summary['total_zf']==sum(x['zf'] for x in ms)
gain=summaries['candidate']/summaries['baseline']-1
np.testing.assert_allclose(gain,r['relative_gain'],rtol=0,atol=1e-12)
assert r['observed_target_met']==bool(gain[1]>=.1 and summaries['candidate'][0]>=summaries['baseline'][0])
out=dict(state='passed',datasets=1117,families=128,score_vectors=vectors,auc_max_abs_error=max_error,primary_relative_gain=float(gain[1]),
    observed_target_met=r['observed_target_met'],result_sha256=digest(p/'full_results_antigravity.json'),verifier_sha256=digest(Path(__file__)),
    every_low_fpr_point_recomputed=True,baseline_scores_reproduced=True,frozen_code_and_inputs_verified=True)
(p/'full_verification_antigravity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
