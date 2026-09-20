"""Independent role, stored-vector, component-mean, replay and gate verification."""
from pathlib import Path
from hashlib import file_digest,sha256
import json
import numpy as np
from experiments.exp509_rank_signal_diagnosis_antigravity.verify_antigravity import signal,cert
from experiments.exp479_missing_score_recovery_antigravity.config_antigravity import DATABASE_PATH_ANTIGRAVITY
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import write_json_atomic_antigravity

P=Path(__file__).resolve().parent;ROOT=P.parents[1];OUT=P/'pilot_arrays_antigravity'
PRIOR=ROOT/'experiments/exp510_separated_rank_antigravity'
NAMES=('MiniRocket','oas_distance','square_joint','a_unweighted','separated_rank','crossfit_unweighted','crossfit_rank')
STANDARD=('local_level','local_scale','permuted_quarters','phase_randomized')
EXTRAS=('weak_level','weak_scale','sparse_spike','time_warp')
AUDIT=('weights','gradient','initial_gradient','normal_scores','anomaly_scores','initial_normal_scores','initial_anomaly_scores')
read=lambda p:json.loads(p.read_text())


def digest(path):
    with path.open('rb') as stream:return file_digest(stream,'sha256').hexdigest()


def main():
    result=read(P/'pilot_results_antigravity.json');manifest=read(P/'pilot_manifest_antigravity.json')
    assert result['state']==manifest['state']=='complete_train_only_pilot'
    assert digest(P/'pilot_results_antigravity.json')==manifest['result_sha256']
    binding=manifest['binding'];raw=json.dumps(binding,sort_keys=True,separators=(',',':')).encode()
    assert sha256(raw).hexdigest()==manifest['binding_sha256']==result['binding_sha256']
    preflight=read(P/'preflight_antigravity.json');assert preflight['state']=='passed'
    assert digest(P/'preflight_antigravity.json')==binding['preflight_sha256']
    for relative,expected in {**binding['code'],**preflight['artifacts']}.items():assert digest(ROOT/relative)==expected,relative
    assert digest(DATABASE_PATH_ANTIGRAVITY)==binding['database_sha256']
    assert digest(P/'protocol_antigravity.html')==binding['protocol_sha256']
    source=ROOT/'experiments/exp495_normal_geometry_antigravity/diagnostics_manifest_antigravity.json'
    assert digest(source)==binding['source_population_sha256'] and read(source)['population']==binding['population']
    assert digest(PRIOR/'pilot_results_antigravity.json')==binding['prior_result_sha256']
    previous={r['dataset']:r for r in read(PRIOR/'pilot_results_antigravity.json')['records']}
    population={r['dataset']:r for r in binding['population']};rows=result['records']
    assert len(rows)==len({r['dataset'] for r in rows})==len({r['source_family'] for r in rows})==50
    assert {r['dataset']:r['source_family'] for r in rows}=={n:r['source_family'] for n,r in population.items()}
    array_count=0;member_diagnostics=[];replayed=0
    for row in rows:
        assert row==read(OUT/f"{row['dataset']}_record_antigravity.json")
        assert row['binding_sha256']==manifest['binding_sha256']
        assert not row['test_values_read'] and not row['test_labels_read']
        size=population[row['dataset']]['n_train'];indices=np.arange(size)
        np.random.default_rng(20260721).shuffle(indices);splits=[np.sort(s) for s in np.array_split(indices,5)]
        assert len(row['folds'])==5 and row['n_train']==size
        for i,f in enumerate(row['folds']):
            held=splits[i];fit=np.sort(np.concatenate([s for j,s in enumerate(splits) if j!=i]))
            assert f['fold']==i and f['fit']==fit.tolist() and f['held']==held.tolist()
            assert not set(held)&set(fit) and len(f['members'])==4
            order=np.random.default_rng(20260721+2510).permutation(len(fit))
            oldf=previous[row['dataset']]['folds'][i];oldpath=PRIOR/'pilot_arrays_antigravity'/oldf['array_file']
            path=OUT/f['array_file'];assert digest(path)==f['array_sha256'] and digest(oldpath)==oldf['array_sha256']
            with np.load(path,allow_pickle=False) as z,np.load(oldpath,allow_pickle=False) as old:
                expected={f'member{j}__{k}' for j in range(4) for k in AUDIT}
                score_names=NAMES+tuple(f'member{j}__{k}' for j in range(4) for k in ('unweighted','rank'))
                expected|={kind+'__'+n for kind in ('normal',)+STANDARD+EXTRAS for n in score_names}
                assert set(z.files)==expected and all(z[k].ndim==1 and np.isfinite(z[k]).all() for k in z.files)
                array_count+=len(z.files);all_b=[];all_a=[]
                for j,member in enumerate(f['members']):
                    local_b=np.sort(order[j*len(fit)//4:(j+1)*len(fit)//4]);local_a=np.setdiff1d(np.arange(len(fit)),local_b)
                    a,b=fit[local_a],fit[local_b]
                    parents=b if len(b)<=128 else np.sort(np.random.default_rng(20260721+1510).choice(b,128,replace=False))
                    assert member['member']==j and member['a']==a.tolist() and member['b']==b.tolist()
                    assert member['parents']==parents.tolist() and not set(a)&set(b)
                    assert set(a)|set(b)==set(fit);all_b.extend(b);all_a.extend(a)
                    d=member['diagnostics'];assert not d['fallback'] and d['optimizer_trained']
                    assert (d['n_fit'],d['n_a'],d['n_b'],d['n_parents'])==(len(fit),len(a),len(b),len(parents))
                    assert np.isfinite(d['scale']) and d['scale']>0
                    values={key:z[f'member{j}__{key}'] for key in AUDIT};w=values['weights'];g=values['gradient']
                    assert w.shape==g.shape==values['initial_gradient'].shape==(8064,)
                    assert len(values['normal_scores'])==len(values['initial_normal_scores'])==len(parents)
                    assert len(values['anomaly_scores'])==len(values['initial_anomaly_scores'])==4*len(parents)
                    initial=signal(values['initial_normal_scores'],values['initial_anomaly_scores'],np.ones_like(w))
                    final=signal(values['normal_scores'],values['anomaly_scores'],w)
                    np.testing.assert_allclose(initial['objective'],d['initial_objective'],rtol=1e-10,atol=1e-10)
                    np.testing.assert_allclose(final['objective'],d['final_objective'],rtol=1e-10,atol=1e-10)
                    certificate=cert(w,g)
                    for key,value in certificate.items():
                        if isinstance(value,bool):assert d[key]==value
                        else:np.testing.assert_allclose(d[key],value,rtol=1e-8,atol=1e-10)
                    assert d['solver']['projected_gradient_passed']==certificate['projected_gradient_passed']
                    np.testing.assert_allclose(d['solver']['objective'],final['objective'],rtol=1e-10,atol=1e-10)
                    if j==0:
                        for key in AUDIT:np.testing.assert_allclose(values[key],old[key],rtol=1e-10,atol=1e-8)
                    member_diagnostics.append(d)
                np.testing.assert_array_equal(np.sort(all_b),fit)
                np.testing.assert_array_equal(np.sort(all_a),np.repeat(fit,3))
                for kind in ('normal',)+STANDARD+EXTRAS:
                    for suffix,key in (('unweighted','crossfit_unweighted'),('rank','crossfit_rank')):
                        components=np.stack([z[f'{kind}__member{j}__{suffix}'] for j in range(4)])
                        assert components.shape==(4,len(held))
                        np.testing.assert_allclose(components.sum(axis=0)/4,z[kind+'__'+key],rtol=1e-14,atol=1e-12)
                    for suffix,key in (('unweighted','a_unweighted'),('rank','separated_rank')):
                        np.testing.assert_array_equal(z[f'{kind}__member0__{suffix}'],z[kind+'__'+key])
                    for name in NAMES[:5]:
                        current=z[kind+'__'+name];prior=old[kind+'__'+name]
                        np.testing.assert_allclose(current,prior,rtol=1e-10,atol=1e-8)
                        np.testing.assert_array_equal(np.argsort(current,kind='stable'),np.argsort(prior,kind='stable'))
                        np.testing.assert_array_equal(current>z['normal__'+name].max(),prior>old['normal__'+name].max());replayed+=1
                for name in NAMES:
                    ns=z['normal__'+name];assert len(ns)==len(held)
                    threshold=float(max(ns));q1,median,q3=np.percentile(ns,[25,50,75])
                    tail=float((threshold-median)/(q3-q1+1e-9))
                    np.testing.assert_allclose(tail,f['normal_tail'][name],rtol=1e-12,atol=1e-12)
                    for kind in STANDARD+EXTRAS:
                        scores=z[kind+'__'+name];assert len(scores)==len(held)
                        recall=float(sum(scores>threshold)/len(scores));assert recall==f['recall'][name][kind]
                        if name in NAMES[:5]:assert recall==oldf['recall'][name][kind]
    standard={n:{k:sum(sum(f['recall'][n][k] for f in r['folds'])/5 for r in rows)/50 for k in STANDARD} for n in NAMES}
    unseen={n:{k:sum(sum(f['recall'][n][k] for f in r['folds'])/5 for r in rows)/50 for k in EXTRAS} for n in NAMES}
    tails={n:float(np.median([sum(f['normal_tail'][n] for f in r['folds'])/5 for r in rows])) for n in NAMES}
    worst={n:min(standard[n].values()) for n in NAMES};means={n:sum(standard[n].values())/4 for n in NAMES}
    extra={n:sum(unseen[n].values())/4 for n in NAMES}
    for n in NAMES:
        for k in STANDARD:np.testing.assert_allclose(standard[n][k],result['summary'][n][k],rtol=0,atol=1e-14)
        for k in EXTRAS:np.testing.assert_allclose(unseen[n][k],result['unseen_summary'][n][k],rtol=0,atol=1e-14)
        np.testing.assert_allclose(tails[n],result['normal_tail'][n],rtol=0,atol=1e-12)
        for key,values in (('worst_standard',worst),('mean_standard',means),('mean_unseen',extra)):
            np.testing.assert_allclose(values[n],result[key][n],rtol=0,atol=1e-14)
    gates=dict(minimum_margin=worst['crossfit_rank']>=max(worst[n] for n in NAMES[:-1])+.02,
               mean_nonregression=means['crossfit_rank']>=max(means[n] for n in NAMES[:-1]),
               unseen_nonregression=extra['crossfit_rank']>=max(extra[n] for n in NAMES[:-1]),
               normal_tail=tails['crossfit_rank']<=3*max(tails['MiniRocket'],1e-9))
    assert gates==result['gates'] and result['full_run_required']==all(gates.values())
    assert result['selected']==('crossfit_rank' if all(gates.values()) else 'square_joint')
    assert not any(doc[k] for doc in (result,manifest) for k in ('test_values_read','test_labels_read'))
    assert len(member_diagnostics)==1000
    output=dict(state='passed',datasets=50,families=50,folds=250,members=1000,arrays_verified=array_count,
                baseline_score_vectors_replayed=replayed,selected=result['selected'],gates=gates,
                full_run_required=result['full_run_required'],roles_and_component_means_verified=True,
                losses_stationarity_and_aggregation_recomputed=True,gradient_vectors_independently_recomputed=False,
                scope='Stored score/gradient vectors audited; geometry and gradients not regenerated from original feature matrices.',
                projected_gradient_passes=sum(d['projected_gradient_passed'] for d in member_diagnostics),
                solver_successes=sum(d['solver']['solver_success'] for d in member_diagnostics),
                result_sha256=digest(P/'pilot_results_antigravity.json'),verifier_sha256=digest(Path(__file__)),
                test_values_read=False,test_labels_read=False)
    write_json_atomic_antigravity(P/'verification_antigravity.json',output);print(json.dumps(output))


if __name__=='__main__':main()
