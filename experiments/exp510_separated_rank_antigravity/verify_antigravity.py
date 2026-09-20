"""Independent stored-vector and frozen-control audit of the A/B pilot."""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
from experiments.exp509_rank_signal_diagnosis_antigravity.verify_antigravity import signal,cert
from experiments.exp479_missing_score_recovery_antigravity.config_antigravity import DATABASE_PATH_ANTIGRAVITY
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import write_json_atomic_antigravity

P=Path(__file__).resolve().parent;ROOT=P.parents[1];OUT=P/'pilot_arrays_antigravity'
NAMES=('MiniRocket','oas_distance','square_joint','a_unweighted','separated_rank')
STANDARD=('local_level','local_scale','permuted_quarters','phase_randomized')
EXTRAS=('weak_level','weak_scale','sparse_spike','time_warp')
read=lambda p:json.loads(p.read_text());digest=lambda p:sha256(p.read_bytes()).hexdigest()


def main():
    result=read(P/'pilot_results_antigravity.json');manifest=read(P/'pilot_manifest_antigravity.json')
    assert result['state']==manifest['state']=='complete_train_only_pilot'
    assert digest(P/'pilot_results_antigravity.json')==manifest['result_sha256']
    binding=manifest['binding'];raw=json.dumps(binding,sort_keys=True,separators=(',',':')).encode()
    assert sha256(raw).hexdigest()==manifest['binding_sha256']==result['binding_sha256']
    preflight=read(P/'preflight_antigravity.json')
    assert digest(P/'preflight_antigravity.json')==binding['preflight_sha256']
    for relative,expected in {**binding['code'],**preflight['artifacts']}.items():
        assert digest(ROOT/relative)==expected,relative
    assert digest(DATABASE_PATH_ANTIGRAVITY)==binding['database_sha256']
    assert digest(P/'protocol_antigravity.html')==binding['protocol_sha256']
    source=ROOT/'experiments/exp495_normal_geometry_antigravity/diagnostics_manifest_antigravity.json'
    assert digest(source)==binding['source_population_sha256'] and read(source)['population']==binding['population']
    prior_path=ROOT/'experiments/exp508_tail_rank_antigravity/pilot_results_antigravity.json'
    assert digest(prior_path)==binding['prior_result_sha256']
    previous={r['dataset']:r for r in read(prior_path)['records']}
    population={r['dataset']:r for r in binding['population']};rows=result['records'];array_count=0;all_folds=[]
    assert len(rows)==len({r['dataset'] for r in rows})==len({r['source_family'] for r in rows})==50
    assert {r['dataset']:r['source_family'] for r in rows}=={n:r['source_family'] for n,r in population.items()}
    for row in rows:
        assert row==read(OUT/f"{row['dataset']}_record_antigravity.json")
        assert not row['test_values_read'] and not row['test_labels_read']
        size=population[row['dataset']]['n_train'];indices=np.arange(size)
        np.random.default_rng(20260721).shuffle(indices);splits=[np.sort(s) for s in np.array_split(indices,5)]
        assert len(row['folds'])==5 and row['n_train']==size
        for i,f in enumerate(row['folds']):
            held=splits[i];fit=np.sort(np.concatenate([s for j,s in enumerate(splits) if j!=i]))
            order=np.random.default_rng(20260721+2510).permutation(len(fit));cut=len(fit)//4
            a=fit[np.sort(order[cut:])];b=fit[np.sort(order[:cut])]
            assert f['fold']==i and f['fit']==fit.tolist() and f['held']==held.tolist()
            assert f['a']==a.tolist() and f['b']==b.tolist() and not set(a)&set(b)
            assert not set(held)&set(fit) and set(a)|set(b)==set(fit)
            # All pilot B sets have fewer than 128 parents; full-run cap is separately tested.
            assert len(b)<=128 and f['parents']==b.tolist()
            audit=f['diagnostics'];assert not audit['fallback'] and audit['optimizer_trained']
            assert (audit['n_fit'],audit['n_a'],audit['n_b'],audit['n_parents'])==(len(fit),len(a),len(b),len(b))
            assert np.isfinite(audit['scale']) and audit['scale']>0
            path=OUT/f['array_file'];assert digest(path)==f['array_sha256']
            with np.load(path,allow_pickle=False) as z:
                expected={'weights','gradient','initial_gradient','normal_scores','anomaly_scores','initial_normal_scores','initial_anomaly_scores'}
                expected|={kind+'__'+name for kind in ('normal',)+STANDARD+EXTRAS for name in NAMES}
                assert set(z.files)==expected and all(z[k].ndim==1 and np.isfinite(z[k]).all() for k in z.files)
                array_count+=len(z.files);w=z['weights'];gradient=z['gradient']
                assert len(w)==8064 and w.shape==gradient.shape==z['initial_gradient'].shape
                assert len(z['normal_scores'])==len(z['initial_normal_scores'])==len(b)
                assert len(z['anomaly_scores'])==len(z['initial_anomaly_scores'])==4*len(b)
                initial=signal(z['initial_normal_scores'],z['initial_anomaly_scores'],np.ones_like(w))
                final=signal(z['normal_scores'],z['anomaly_scores'],w)
                np.testing.assert_allclose(initial['objective'],audit['initial_objective'],rtol=1e-10,atol=1e-10)
                np.testing.assert_allclose(final['objective'],audit['final_objective'],rtol=1e-10,atol=1e-10)
                certificate=cert(w,gradient)
                for key,value in certificate.items():
                    if isinstance(value,bool):assert audit[key]==value
                    else:np.testing.assert_allclose(audit[key],value,rtol=1e-8,atol=1e-10)
                assert audit['solver']['projected_gradient_passed']==certificate['projected_gradient_passed']
                np.testing.assert_allclose(audit['solver']['objective'],final['objective'],rtol=1e-10,atol=1e-10)
                for name in NAMES:
                    ns=z['normal__'+name];assert len(ns)==len(held)
                    threshold=float(max(ns));q1,median,q3=np.percentile(ns,[25,50,75])
                    tail=float((threshold-median)/(q3-q1+1e-9))
                    np.testing.assert_allclose(tail,f['normal_tail'][name],rtol=1e-12,atol=1e-12)
                    for kind in STANDARD+EXTRAS:
                        scores=z[kind+'__'+name];assert len(scores)==len(held)
                        recall=float(sum(scores>threshold)/len(scores));assert recall==f['recall'][name][kind]
                        if name in NAMES[:3]:
                            key='folds' if kind in STANDARD else 'unseen_folds'
                            assert recall==previous[row['dataset']][key][i][kind][name]
            all_folds.append(f)
    standard={n:{k:sum(sum(f['recall'][n][k] for f in r['folds'])/5 for r in rows)/50 for k in STANDARD} for n in NAMES}
    unseen={n:{k:sum(sum(f['recall'][n][k] for f in r['folds'])/5 for r in rows)/50 for k in EXTRAS} for n in NAMES}
    tails={n:float(np.median([sum(f['normal_tail'][n] for f in r['folds'])/5 for r in rows])) for n in NAMES}
    for n in NAMES:
        for k in STANDARD:np.testing.assert_allclose(standard[n][k],result['summary'][n][k],rtol=0,atol=1e-14)
        for k in EXTRAS:np.testing.assert_allclose(unseen[n][k],result['unseen_summary'][n][k],rtol=0,atol=1e-14)
        np.testing.assert_allclose(tails[n],result['normal_tail'][n],rtol=0,atol=1e-12)
    worst={n:min(standard[n].values()) for n in NAMES};means={n:sum(standard[n].values())/4 for n in NAMES}
    extra={n:sum(unseen[n].values())/4 for n in NAMES}
    gates=dict(minimum_margin=worst['separated_rank']>=max(worst[n] for n in NAMES[:-1])+.02,
               mean_nonregression=means['separated_rank']>=max(means[n] for n in NAMES[:-1]),
               unseen_nonregression=extra['separated_rank']>=max(extra[n] for n in NAMES[:-1]),
               normal_tail=tails['separated_rank']<=3*max(tails['MiniRocket'],1e-9))
    assert gates==result['gates'] and result['full_run_required']==all(gates.values())
    assert result['selected']==('separated_rank' if all(gates.values()) else 'square_joint')
    assert not any(doc[k] for doc in (result,manifest) for k in ('test_values_read','test_labels_read'))
    output=dict(state='passed',datasets=50,families=50,folds=250,arrays_verified=array_count,
                selected=result['selected'],gates=gates,full_run_required=result['full_run_required'],
                baseline_fold_recalls_exact=True,role_isolation_verified=True,
                losses_and_stationarity_and_aggregation_recomputed=True,gradient_vectors_independently_recomputed=False,
                gradient_scope='Original objective finite-difference regression tests; stored gradients verified through projected conditions.',
                projected_gradient_passes=sum(f['diagnostics']['projected_gradient_passed'] for f in all_folds),
                solver_successes=sum(f['diagnostics']['solver']['solver_success'] for f in all_folds),
                result_sha256=digest(P/'pilot_results_antigravity.json'),verifier_sha256=digest(Path(__file__)),
                test_values_read=False,test_labels_read=False)
    write_json_atomic_antigravity(P/'verification_antigravity.json',output);print(json.dumps(output))


if __name__=='__main__':
    main()
