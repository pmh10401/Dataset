"""Independent stored-vector loss, KKT, recall, split and hash audit; no TEST reader."""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np
from scipy.special import expit
from experiments.exp479_missing_score_recovery_antigravity.config_antigravity import DATABASE_PATH_ANTIGRAVITY

P=Path(__file__).resolve().parent;ROOT=P.parents[1];OUT=P/'audit_arrays_antigravity'
read=lambda p:json.loads(p.read_text())
digest=lambda p:sha256(p.read_bytes()).hexdigest()
STANDARD=('local_level','local_scale','permuted_quarters','phase_randomized')
UNSEEN=('weak_level','weak_scale','sparse_spike','time_warp')
MODELS=('MiniRocket','oas_distance','square_joint','adam','lbfgsb')


def close(actual,expected):
    if isinstance(expected,dict):
        assert set(actual)==set(expected)
        for key in expected:close(actual[key],expected[key])
    elif isinstance(expected,(int,bool)):
        assert actual==expected
    else:np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-10)


def signal(ns,az,w):
    maximum=float(max(ns)); shifted=(ns-maximum)/.1
    probs=np.exp(shifted);soft=probs/probs.sum();tail=maximum+.1*np.log(probs.sum())
    delta=.25+tail-az;values=np.maximum(delta,0)+np.log1p(np.exp(-abs(delta)))
    penalty=.1*sum((w-1)**2)/len(w);sigma=expit(delta)
    return dict(objective=float(values.mean()+penalty),data_loss=float(values.mean()),penalty=float(penalty),
                n_normal=len(ns),n_anomaly=len(az),normal_median=float(np.median(ns)),normal_p95=float(np.percentile(ns,95)),
                normal_max=maximum,anomaly_median=float(np.median(az)),anomaly_p05=float(np.percentile(az,5)),
                smooth_normal_upper=float(tail),mean_sigmoid=float(sigma.mean()),
                active_001_fraction=float(sum(sigma>=.01)/len(sigma)),active_01_fraction=float(sum(sigma>=.1)/len(sigma)),
                above_normal_max_fraction=float(sum(az>maximum)/len(az)),median_margin=float(np.median(az-maximum)),
                minimum_margin=float(min(az-maximum)),effective_normal_count=float(1/sum(soft**2)))


def cert(w,g):
    assert np.all((w>=.25)&(w<=4))
    # Coordinate-wise constrained first-order conditions, independently of scipy status.
    pg=np.array([min(gi,wi-.25) if gi>=0 else max(gi,wi-4) for wi,gi in zip(w,g)])
    mu=.2/len(w);u=np.minimum(4,np.maximum(.25,w-g/mu));delta=w-u
    gap=float(sum(g*delta)-mu*sum(delta**2)/2)
    return dict(projected_gradient_inf=float(max(abs(pg))),projected_gradient_passed=bool(max(abs(pg))<=1e-8),
                objective_gap_upper_bound=max(0.,gap),rms_displacement=float(np.sqrt(sum((w-1)**2)/len(w))))


def main():
    r=read(P/'results_antigravity.json');m=read(P/'manifest_antigravity.json');pre=read(P/'preflight_antigravity.json')
    assert r['state']==m['state']=='complete_train_diagnostic'
    assert digest(P/'results_antigravity.json')==m['result_sha256']
    binding=m['binding'];canonical=json.dumps(binding,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
    assert sha256(canonical).hexdigest()==m['binding_sha256']==r['binding_sha256']
    assert digest(DATABASE_PATH_ANTIGRAVITY)==binding['database_sha256']
    for relative,h in {**binding['code'],**pre['artifacts']}.items():assert digest(ROOT/relative)==h,relative
    assert digest(P/'preflight_antigravity.json')==binding['preflight_sha256']
    assert digest(P/'protocol_antigravity.html')==binding['protocol_sha256']
    source=ROOT/'experiments/exp495_normal_geometry_antigravity/diagnostics_manifest_antigravity.json'
    assert digest(source)==binding['source_population_sha256'] and read(source)['population']==binding['population']
    prior=ROOT/'experiments/exp508_tail_rank_antigravity/pilot_results_antigravity.json'
    assert digest(prior)==binding['prior_result_sha256']
    previous={x['dataset']:x for x in read(prior)['records']};population={x['dataset']:x for x in binding['population']}
    assert len(r['records'])==len({x['dataset'] for x in r['records']})==50
    assert {x['dataset']:x['source_family'] for x in r['records']}=={k:v['source_family'] for k,v in population.items()}
    checked_folds=[];array_count=0
    for row in r['records']:
        assert row==read(OUT/f"{row['dataset']}_record_antigravity.json")
        assert row['binding_sha256']==m['binding_sha256'] and not row['test_values_read'] and not row['test_labels_read']
        size=population[row['dataset']]['n_train'];shuffled=np.arange(size);np.random.default_rng(20260721).shuffle(shuffled)
        held_parts=[np.sort(a) for a in np.array_split(shuffled,5)]
        assert len(row['folds'])==5
        for i,f in enumerate(row['folds']):
            held=held_parts[i];fit=np.sort(np.concatenate([a for j,a in enumerate(held_parts) if j!=i]))
            assert f['fit']==fit.tolist() and f['held']==held.tolist()
            expected_parent=np.sort(np.random.default_rng(20260721+508).choice(len(fit),min(128,len(fit)),replace=False))
            assert f['original_parents']==expected_parent.tolist()
            part=f['participation']['roles'];perm=np.random.default_rng(20260721+2509+i).permutation(len(fit));n=len(fit)//4
            assert part['unseen']==np.sort(perm[:n]).tolist() and part['seen']==np.sort(perm[n:2*n]).tolist()
            assert part['reference']==np.sort(perm[2*n:]).tolist()
            assert part['representation_fit']==sorted(part['reference']+part['seen'])
            assert not set(part['unseen'])&set(part['representation_fit'])
            assert f['participation']['common_reference_count']==len(fit)-2*n
            path=OUT/f['array_file'];assert digest(path)==f['array_sha256']
            with np.load(path,allow_pickle=False) as z:
                expected_keys={name+suffix for name in ('initial','adam','lbfgsb') for suffix in ('_weights','_gradient')}
                expected_keys|={'train__'+name+'__'+kind for name in ('initial','adam','lbfgsb') for kind in ('normal','anomaly')}
                expected_keys|={'held__'+kind+'__'+name for kind in ('normal',)+STANDARD+UNSEEN for name in MODELS}
                expected_keys|={name+'_'+kind for name in ('seen','unseen') for kind in ('normal','anomaly')}
                assert set(z.files)==expected_keys and all(z[k].ndim==1 and np.isfinite(z[k]).all() for k in z.files)
                array_count+=len(z.files)
                for name in ('initial','adam','lbfgsb'):
                    w=z[name+'_weights'];g=z[name+'_gradient'];ns=z['train__'+name+'__normal'];az=z['train__'+name+'__anomaly']
                    assert w.shape==g.shape and len(ns)==len(expected_parent) and len(az)==4*len(ns)
                    if name=='initial':assert np.array_equal(w,np.ones_like(w))
                    close(f['training'][name],dict(**signal(ns,az,w),**cert(w,g)))
                close(f['nonworsening_rms_bound'],float(np.sqrt(f['training']['initial']['objective']/.1)))
                close(f['solver']['objective'],f['training']['lbfgsb']['objective'])
                assert f['solver']['projected_gradient_passed']==f['training']['lbfgsb']['projected_gradient_passed']
                for name in ('seen','unseen'):
                    assert len(z[name+'_normal'])==n and len(z[name+'_anomaly'])==4*n
                    close(f['participation']['signal'][name],signal(z[name+'_normal'],z[name+'_anomaly'],z['initial_weights']))
                for name in MODELS:
                    ns=z['held__normal__'+name];assert len(ns)==len(held)
                    tail=float((max(ns)-np.median(ns))/(np.percentile(ns,75)-np.percentile(ns,25)+1e-9))
                    close(f['normal_tail'][name],tail)
                    for kind in STANDARD+UNSEEN:
                        s=z['held__'+kind+'__'+name];assert len(s)==len(held)
                        recall=float(sum(s>max(ns))/len(s));assert recall==f['recall'][name][kind]
                        if name!='lbfgsb':
                            old='tail_rank' if name=='adam' else name;key='folds' if kind in STANDARD else 'unseen_folds'
                            assert recall==previous[row['dataset']][key][i][kind][old]
                for name,key in (('initial','initial_objective'),('adam','final_objective')):
                    close(f['training'][name]['objective'],previous[row['dataset']]['diagnostics'][i][key])
            checked_folds.append(f)
    summary=r['summary'];rec=r['records']
    for name in MODELS:
        standard={k:sum(sum(f['recall'][name][k] for f in row['folds'])/5 for row in rec)/50 for k in STANDARD}
        extra={k:sum(sum(f['recall'][name][k] for f in row['folds'])/5 for row in rec)/50 for k in UNSEEN}
        close(summary['metrics'][name],dict(standard=standard,unseen=extra,worst_standard=min(standard.values()),
              mean_standard=sum(standard.values())/4,mean_unseen=sum(extra.values())/4,
              normal_tail=float(np.median([sum(f['normal_tail'][name] for f in row['folds'])/5 for row in rec]))))
    for name,values in summary['training_medians'].items():
        for key,value in values.items():close(value,float(np.median([f['training'][name][key] for f in checked_folds])))
    for name,values in summary['participation_family_medians'].items():
        for key,value in values.items():close(value,float(np.median([np.median([f['participation']['signal'][name][key] for f in row['folds']]) for row in rec])))
    ratios=[np.median([f['participation']['signal']['unseen']['normal_max']/max(f['participation']['signal']['seen']['normal_max'],1e-30) for f in row['folds']]) for row in rec]
    close(summary['family_median_normal_max_ratio'],float(np.median(ratios)))
    assert summary['families_unseen_max_higher']==sum(x>1 for x in ratios)
    assert summary['lbfgsb_projected_gradient_passes']==sum(f['solver']['projected_gradient_passed'] for f in checked_folds)
    assert summary['lbfgsb_solver_successes']==sum(f['solver']['solver_success'] for f in checked_folds)
    assert summary['lbfgsb_objective_above_initial']==sum(f['training']['lbfgsb']['objective']>f['training']['initial']['objective']+1e-12 for f in checked_folds)
    assert summary['adam_objective_increased']==sum(f['training']['adam']['objective']>f['training']['initial']['objective'] for f in checked_folds)
    assert summary['lbfgsb_max_iterations']==max(f['solver']['iterations'] for f in checked_folds)
    close(summary['median_nonworsening_rms_bound'],float(np.median([f['nonworsening_rms_bound'] for f in checked_folds])))
    assert len(checked_folds)==250 and not r['model_promotion'] and not r['exp510_started']
    assert not r['test_values_read'] and not r['test_labels_read']
    out=dict(state='passed',datasets=50,families=50,folds=250,arrays_verified=array_count,
             result_sha256=digest(P/'results_antigravity.json'),verifier_sha256=digest(Path(__file__)),
             baseline_recalls_replayed=True,roles_verified=True,losses_projected_gradients_and_aggregation_recomputed=True,
             gradients_independent_recomputation=False,gradient_scope='finite-difference unit test; full-vector gradients produced by frozen verified objective',
             test_values_read=False,test_labels_read=False,model_promotion=False)
    (P/'verification_antigravity.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))


if __name__=='__main__':main()
