"""Independent read-only verification of every Exp494 score and aggregate."""
from pathlib import Path
from contextlib import closing
from hashlib import sha256
import json
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_curve
from experiments.exp491_gain_prediction_antigravity.runtime_antigravity import readonly_sqlite_antigravity,DATABASE_PATH_ANTIGRAVITY,sha256_json_antigravity
from experiments.exp479_missing_score_recovery_antigravity.evaluation_antigravity import load_retained_test_labels_antigravity
root=Path(__file__).resolve().parents[1];p=root/'experiments/exp494_all_datasets_low_fpr_antigravity'
read=lambda f:json.loads(f.read_text()); digest=lambda f:sha256(f.read_bytes()).hexdigest()
r=read(p/'results_antigravity.json');manifest=read(p/'run_manifest_antigravity.json');dec=read(p/'decisions_antigravity.json')
assert digest(p/'results_antigravity.json')==manifest['result_sha256']
assert digest(p/'decisions_antigravity.json')==r['decisions_sha256']
boundary=read(p/'labels_opened_antigravity.json')
assert digest(p/'labels_opened_antigravity.json')==r['label_boundary_sha256']
assert boundary['decisions_sha256']==r['decisions_sha256']
assert digest(DATABASE_PATH_ANTIGRAVITY)==manifest['binding']['source_binding']['database']
recovery=read(p/'constant_recovery_manifest_antigravity.json')
assert recovery['state']=='scored' and recovery['nonconstant_rows_bitwise_unchanged']
assert digest(p/'constant_recovery_manifest_antigravity.json')==r['preprocessing_supplement']['manifest_sha256']==manifest['preprocessing_supplement']['manifest_sha256']
assert sha256_json_antigravity(recovery['binding'])==recovery['binding_sha256']
assert digest(p/'finalize_recovered_antigravity.py')==r['finalizer_sha256']==manifest['finalizer_sha256']
for relative,expected in recovery['binding']['files'].items():assert digest(root/relative)==expected
recovered_meta=read(p/'score_checkpoints_antigravity/UWaveGestureLibraryZ_normal_8_meta_antigravity.json')
assert recovered_meta['recovery_binding_sha256']==recovery['binding_sha256'] and recovered_meta['score_sha256']==recovery['score_sha256']
assert r['evaluated_datasets']==1117 and r['families']==128 and len(r['unavailable_42d'])==161
names=('MiniRocket','MultiRocket','Hydra stable','Hydra+MultiRocket stable','MiniRocket+LOF','Hydra old','Hydra+MultiRocket old')
grid=np.array([0,1,5,10]);lookup={row['dataset']:row for row in r['records']};vectors=0;max_auc_error=0.
with closing(readonly_sqlite_antigravity(DATABASE_PATH_ANTIGRAVITY)) as con:
    for row in r['records']:
        d=row['dataset'];meta=read(p/'score_checkpoints_antigravity'/f'{d}_meta_antigravity.json');score=p/'score_checkpoints_antigravity'/f'{d}_scores_antigravity.npz'
        assert digest(score)==meta['score_sha256']
        assert meta['score_sha256']==boundary['scores'][d]
        assert sha256_json_antigravity(meta)==boundary['metadata'][d]
        assert meta['source_family']==row['source_family'] and meta['n_train']==row['n_train']
        y=load_retained_test_labels_antigravity(con,d,tuple(meta['test_indices']));pos=y==1;np_=int(pos.sum());nn=int((~pos).sum())
        with np.load(score) as arrays:
            for j,name in enumerate(names):
                s=arrays[f'test__{j}'];stored=row['metrics'][name]
                assert np.isfinite(s).all() and len(s)==len(y)
                auc=(rankdata(s)[pos].sum()-np_*(np_+1)/2)/(np_*nn)
                max_auc_error=max(max_auc_error,abs(auc-stored['auc']));assert abs(auc-stored['auc'])<1e-12
                fpr,tpr,_=roc_curve(y,s,drop_intermediate=False);fp=np.rint(fpr*nn).astype(int)
                independent=[tpr[fp<=budget].max() for budget in nn*grid//1000]
                np.testing.assert_allclose(independent,stored['tpr'],rtol=0,atol=1e-15)
                assert np.rint(np.asarray(independent)*np_).astype(int).tolist()==stored['tp']
                thresholds=np.sort(s[~pos])[::-1][nn*grid//1000]
                assert [int((s[~pos]>threshold).sum()) for threshold in thresholds]==stored['fp']
                assert np.all(np.asarray(stored['fp'])<=nn*grid//1000)
                assert stored['zf']==int((s[pos]>max(s[~pos])).sum()) and stored['n_normal']==nn and stored['n_anomaly']==np_
                vectors+=1
methods={name:{d:row['metrics'][name] for d,row in lookup.items()} for name in names}
for name,model in dec['models'].items():
    pred=np.asarray(model['predictions']);assert len(model['datasets'])==len(pred)
    for head,offset in [('auc',0),('zf_sign',3)]:
        computed=np.argmax(np.column_stack((np.zeros(len(pred)),pred[:,offset:offset+3])),axis=1)
        assert computed.tolist()==model['routes'][head]
        methods[name+' / '+head]={d:lookup[d]['metrics'][names[j]] for d,j in zip(model['datasets'],computed)}
cohorts={'all_evaluable':list(lookup),'common_42d':[d for d in lookup if lookup[d]['n_train']>=5],'previous_representatives':dec['training_datasets']}
aggregates=0
for cohort,datasets in cohorts.items():
    for name,expected in r['summaries'][cohort].items():
        ms=[methods[name][d] for d in datasets];families=np.array([lookup[d]['source_family'] for d in datasets]);tpr=np.array([x['tpr'] for x in ms]);auc=np.array([x['auc'] for x in ms])
        groups=[families==f for f in sorted(set(families))]
        family=np.mean([np.mean(tpr[g],axis=0) for g in groups],axis=0)
        dataset=tpr.mean(axis=0); micro=np.sum([x['tp'] for x in ms],axis=0)/sum(x['n_anomaly'] for x in ms)
        for key,actual in [('family_macro_tpr',family),('dataset_macro_tpr',dataset),('micro_tpr',micro)]:np.testing.assert_allclose(actual,expected[key],rtol=0,atol=1e-12)
        assert expected['total_zf']==sum(x['zf'] for x in ms)
        assert abs(expected['family_macro_auc']-np.mean([auc[g].mean() for g in groups]))<1e-12
        assert expected['datasets']==len(datasets) and expected['families']==len(groups)
        aggregates+=1
for path,expected in manifest['binding']['source_binding']['inputs'].items():assert digest(Path(path))==expected
for relative,expected in manifest['binding']['code'].items():assert digest(root/relative)==expected
out=dict(state='passed',datasets=1117,score_vectors=vectors,aggregate_rows=aggregates,auc_max_abs_error=max_auc_error,
    all_low_fpr_points_recomputed=True,frozen_routes_verified=True,original_inputs_and_code_unchanged=True,raw_database_unchanged=True,label_boundary_metadata_unchanged=True,
    constant_recovery_provenance_verified=True,result_sha256=digest(p/'results_antigravity.json'),verifier_sha256=digest(Path(__file__)))
(p/'verification_antigravity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
