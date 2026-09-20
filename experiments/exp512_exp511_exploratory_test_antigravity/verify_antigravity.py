"""Independent stored-score ROC, mean, family-delta and bootstrap audit."""
import csv
import json
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_curve
from experiments.exp479_missing_score_recovery_antigravity.evaluation_antigravity import load_retained_test_labels_antigravity
from .scoring_antigravity import P,OUT,ROOT,DB,EXP502,read,digest,canonical,check_hashes,readonly_sqlite_antigravity,write_json_atomic_antigravity
from .scoring_antigravity import database_snapshot,assert_unbound_wal_antigravity

NAMES=('MiniRocket','oas_distance','square_joint','a_unweighted','separated_rank','crossfit_unweighted','crossfit_rank')

def main():
    assert_unbound_wal_antigravity(DB)
    result=read(P/'results_antigravity.json');manifest=read(P/'manifest_antigravity.json')
    assert manifest['state']==result['state']=='complete_exploratory_test'
    assert digest(P/'results_antigravity.json')==manifest['result_sha256']
    binding=manifest['binding'];assert canonical(binding)==manifest['binding_sha256']==result['binding_sha256']
    assert not result['train_gate_passed'] and not result['model_promotion']
    assert binding['evaluation_kind']=='user_authorized_exploratory_test' and result['candidate']=='crossfit_rank'
    boundary=read(P/'score_freeze_antigravity.json')
    assert digest(P/'score_freeze_antigravity.json')==manifest['boundary_sha256']==result['boundary_sha256']
    opened=read(P/'labels_opened_antigravity.json')
    assert opened['boundary_sha256']==result['boundary_sha256'] and opened['utc']>=boundary['utc']
    preflight=read(P/'preflight_antigravity.json');assert digest(P/'preflight_antigravity.json')==binding['preflight_sha256']
    check_hashes({**binding['code'],**preflight['artifacts']})
    check_hashes(read(P/'authorization_antigravity.json')['historical_artifacts'])
    assert digest(DB)==binding['database_sha256']
    population={row['dataset']:row['source_family'] for row in binding['population']}
    records=result['records'];assert len(records)==1117 and len(set(population.values()))==128
    assert {row['dataset']:row['source_family'] for row in records}==population
    assert set(boundary['scores'])==set(boundary['metadata'])==set(population)
    vectors=0;fallbacks=0;members=0;solver_ok=0;pg_ok=0
    with database_snapshot(DB) as con:
        for row in records:
            name=row['dataset'];path=OUT/f'{name}_scores_antigravity.npz';mp=OUT/f'{name}_meta_antigravity.json';meta=read(mp)
            assert digest(mp)==boundary['metadata'][name]
            assert digest(path)==boundary['scores'][name]==meta['score_sha256']
            assert meta['binding_sha256']==manifest['binding_sha256'] and meta['source_family']==population[name]
            assert not meta['test_labels_read']
            oldmeta=read(EXP502/'full_scores_antigravity'/f'{name}_meta_antigravity.json')
            assert digest(EXP502/'full_scores_antigravity'/f'{name}_meta_antigravity.json')==meta['source502_meta_sha256']
            assert all(meta[k]==oldmeta[k] for k in ('n_train','n_test','test_indices','source_family'))
            oldfile=EXP502/'full_scores_antigravity'/f'{name}_scores_antigravity.npz';assert digest(oldfile)==oldmeta['score_sha256']
            total=meta['n_train'];refs=np.arange(total)
            if total>1000:refs=np.sort(np.random.default_rng(20260721+7919+4096).choice(refs,1000,replace=False))
            np.testing.assert_array_equal(refs,meta['reference_indices']);fallback=total<32
            assert meta['fallback']==fallback;fallbacks+=int(fallback)
            y=load_retained_test_labels_antigravity(con,name,tuple(meta['test_indices']));nn=int((y==0).sum());np_=int((y==1).sum())
            with np.load(path,allow_pickle=False) as z,np.load(oldfile,allow_pickle=False) as old:
                expected=set(NAMES)
                if fallback:
                    assert not meta['members']
                    for key in ('crossfit_rank','crossfit_unweighted'):np.testing.assert_array_equal(z[key],old['candidate'])
                else:
                    expected|={f'member{j}__{kind}' for j in range(4) for kind in ('weights','rank','unweighted')}
                    order=np.random.default_rng(20260721+2510).permutation(len(refs));bs=[];as_=[]
                    assert len(meta['members'])==4
                    for j,m in enumerate(meta['members']):
                        local_b=np.sort(order[j*len(refs)//4:(j+1)*len(refs)//4]);b=refs[local_b]
                        a=refs[np.setdiff1d(np.arange(len(refs)),local_b)]
                        parents=b if len(b)<=128 else np.sort(np.random.default_rng(20260721+1510).choice(b,128,replace=False))
                        assert m['a']==a.tolist() and m['b']==b.tolist() and m['parents']==parents.tolist()
                        bs.extend(b);as_.extend(a);w=z[f'member{j}__weights']
                        assert w.shape==(8064,) and np.isfinite(w).all() and np.all((w>=.25)&(w<=4))
                        d=m['diagnostics'];assert d['n_a']==len(a) and d['n_b']==len(b) and d['n_parents']==len(parents)
                        solver_ok+=int(d['solver']['solver_success']);pg_ok+=int(d['projected_gradient_passed']);members+=1
                    np.testing.assert_array_equal(np.sort(bs),refs);np.testing.assert_array_equal(np.sort(as_),np.repeat(refs,3))
                    for kind,key in (('rank','crossfit_rank'),('unweighted','crossfit_unweighted')):
                        comp=[z[f'member{j}__{kind}'] for j in range(4)]
                        np.testing.assert_allclose(sum(comp)/4,z[key],rtol=1e-14,atol=1e-12)
                assert set(z.files)==expected
                for key,prior in (('MiniRocket','baseline'),('square_joint','candidate')):
                    np.testing.assert_allclose(z[key],old[prior],rtol=1e-10,atol=1e-8)
                    np.testing.assert_array_equal(np.argsort(z[key],kind='stable'),np.argsort(old[prior],kind='stable'))
                for key in NAMES:
                    s=z[key];stored=row['metrics'][key];assert s.shape==y.shape and np.isfinite(s).all()
                    auc=(rankdata(s)[y==1].sum()-np_*(np_+1)/2)/(nn*np_)
                    np.testing.assert_allclose(auc,stored['auc'],rtol=0,atol=1e-12)
                    fpr,tpr,_=roc_curve(y,s,drop_intermediate=False);fp=np.rint(fpr*nn).astype(int)
                    budgets=nn*np.array([0,1,5,10])//1000
                    np.testing.assert_allclose([tpr[fp<=b].max() for b in budgets],stored['tpr'],rtol=0,atol=1e-15)
                    thresholds=np.sort(s[y==0])[::-1][budgets]
                    assert stored['tp']==[int((s[y==1]>t).sum()) for t in thresholds]
                    assert stored['fp']==[int((s[y==0]>t).sum()) for t in thresholds]
                    assert stored['n_normal']==nn and stored['n_anomaly']==np_ and stored['zf']==stored['tp'][0]
                    vectors+=1
    families=sorted(set(population.values()));grouped={};datasets={}
    for name in NAMES:
        ms=[row['metrics'][name] for row in records];datasets[name]=np.array([m['tpr'] for m in ms])
        grouped[name]=np.array([[sum(row['metrics'][name]['tpr'][j] for row in records if row['source_family']==g)/sum(row['source_family']==g for row in records) for j in range(4)] for g in families])
        stored=result['summaries'][name]
        np.testing.assert_allclose(grouped[name].mean(axis=0),stored['family_macro_tpr'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(datasets[name].mean(axis=0),stored['dataset_macro_tpr'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(np.sum([m['tp'] for m in ms],axis=0)/sum(m['n_anomaly'] for m in ms),stored['micro_tpr'],rtol=0,atol=1e-12)
        assert stored['total_zf']==sum(m['zf'] for m in ms)
        assert stored['n_normal']==sum(m['n_normal'] for m in ms) and stored['n_anomaly']==sum(m['n_anomaly'] for m in ms)
        assert stored['datasets']==len(records) and stored['families']==len(families)
        np.testing.assert_allclose(stored['dataset_macro_auc'],sum(m['auc'] for m in ms)/len(ms),rtol=0,atol=1e-12)
        family_auc=[sum(row['metrics'][name]['auc'] for row in records if row['source_family']==g)/sum(row['source_family']==g for row in records) for g in families]
        np.testing.assert_allclose(stored['family_macro_auc'],sum(family_auc)/len(family_auc),rtol=0,atol=1e-12)
    for control in ('MiniRocket','square_joint','crossfit_unweighted'):
        c=grouped['crossfit_rank'];b=grouped[control];delta=c-b;dd=datasets['crossfit_rank']-datasets[control]
        item=result['comparisons'][control]
        for i,g in enumerate(families):np.testing.assert_allclose(delta[i],item['family_deltas'][g],rtol=0,atol=1e-12)
        np.testing.assert_allclose(c.mean(axis=0)-b.mean(axis=0),item['absolute_delta'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(c.mean(axis=0)/b.mean(axis=0)-1,item['relative_delta'],rtol=0,atol=1e-12)
        for prefix,values in (('family',delta),('dataset',dd)):
            for suffix,mask in (('improved',values>1e-12),('worsened',values < -1e-12),('unchanged',np.abs(values)<=1e-12)):
                assert item[prefix+'_'+suffix]==mask.sum(axis=0).tolist()
        rng=np.random.default_rng(20260721);absolute=[];relative=[]
        for _ in range(2000):
            ix=rng.integers(0,len(families),size=len(families));cb=c[ix].mean(axis=0);bb=b[ix].mean(axis=0)
            absolute.append(cb-bb);relative.append(cb/bb-1)
        for key,values in (('absolute_bootstrap95',absolute),('relative_bootstrap95',relative)):
            np.testing.assert_allclose(np.quantile(values,[.025,.975],axis=0).T,item[key],rtol=0,atol=1e-12)
    c=grouped['crossfit_rank'].mean(axis=0);b=grouped['MiniRocket'].mean(axis=0)
    assert result['observed_target_met']==bool(c[1]>=0.6004757235442457 and c[0]>=b[0])
    assert digest(P/'family_comparison_antigravity.csv')==result['family_csv_sha256']
    with (P/'family_comparison_antigravity.csv').open(encoding='utf-8-sig') as stream:csv_rows=list(csv.DictReader(stream))
    expected={}
    for control,item in result['comparisons'].items():
        for family,values in item['family_deltas'].items():
            for budget,delta in zip(('FP0','FPR ≤0.1%','FPR ≤0.5%','FPR ≤1%'),values):expected[(family,control,budget)]=delta
    assert len(csv_rows)==len(expected)==1536
    assert {(row['source_family'],row['comparison'],row['budget']) for row in csv_rows}==set(expected)
    for row in csv_rows:
        delta=expected[(row['source_family'],row['comparison'],row['budget'])]
        assert float(row['delta_tpr'])==delta and float(row['delta_percentage_points'])==delta*100
        assert row['status']==('improved' if delta>1e-12 else 'worsened' if delta < -1e-12 else 'unchanged')
    assert_unbound_wal_antigravity(DB);assert digest(DB)==binding['database_sha256']
    output=dict(state='passed',datasets=1117,families=128,score_vectors=vectors,baseline_vectors_replayed=2234,
                fallback_datasets=fallbacks,member_count=members,solver_successes=solver_ok,projected_gradient_passes=pg_ok,
                population_and_scores_frozen=True,all_four_budgets_recomputed=True,all_three_means_recomputed=True,
                family_deltas_and_bootstrap_recomputed=True,train_gate_passed=False,model_promotion=False,
                result_sha256=digest(P/'results_antigravity.json'),verifier_sha256=digest(P/'verify_antigravity.py'),
                scope='Stored-score independent evaluation audit; no independent refit of original representations or gradients.')
    write_json_atomic_antigravity(P/'verification_antigravity.json',output);print(json.dumps(output),flush=True)

if __name__=='__main__':main()
