"""TRAIN-only screening with exact frozen controls and locally persisted score evidence."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import closing
from pathlib import Path
from hashlib import sha256
from time import perf_counter
import json
import multiprocessing
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import ROOT,DB,SEED,readonly_sqlite_antigravity,write_json_atomic_antigravity,cpu_worker_policy_antigravity,file_sha256_antigravity
from experiments.exp495_normal_geometry_antigravity.pilot_antigravity import load_normal_train,perturbations,KINDS
from experiments.exp449_residual_fn_method_qualification_antigravity.scoring_antigravity import crossfit_plan_antigravity
from experiments.exp508_tail_rank_antigravity.ranking_antigravity import unseen_perturbations
from .separated_antigravity import SeparatedRankHeads,NAMES

P=Path(__file__).resolve().parent;OUT=P/'pilot_arrays_antigravity'
PRIOR=ROOT/'experiments/exp508_tail_rank_antigravity'
UNSEEN=('weak_level','weak_scale','sparse_spike','time_warp')
read=lambda p:json.loads(p.read_text())
digest=lambda p:sha256(p.read_bytes()).hexdigest()
json_digest=lambda x:sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def select_candidate(standard,unseen,tails):
    for name in NAMES:
        if set(standard[name])!=set(KINDS) or set(unseen[name])!=set(UNSEEN):
            raise ValueError('unexpected validation kinds')
        values=list(standard[name].values())+list(unseen[name].values())
        if not np.isfinite(values).all() or min(values)<0 or max(values)>1 or not np.isfinite(tails[name]) or tails[name]<0:
            raise ValueError('invalid selection inputs')
    worst={n:min(standard[n].values()) for n in NAMES}
    means={n:float(np.mean(list(standard[n].values()))) for n in NAMES}
    extra={n:float(np.mean(list(unseen[n].values()))) for n in NAMES}
    threshold=max(worst[n] for n in NAMES[:-1])+.02
    gates=dict(minimum_margin=bool(worst['separated_rank']>=threshold),
               mean_nonregression=bool(means['separated_rank']>=max(means[n] for n in NAMES[:-1])),
               unseen_nonregression=bool(extra['separated_rank']>=max(extra[n] for n in NAMES[:-1])),
               normal_tail=bool(tails['separated_rank']<=3*max(tails['MiniRocket'],1e-9)))
    return dict(selected='separated_rank' if all(gates.values()) else 'square_joint',gates=gates,
                worst_standard=worst,mean_standard=means,mean_unseen=extra,required_worst=threshold,
                candidate_margin=worst['separated_rank']-max(worst[n] for n in NAMES[:-1]),
                required_margin=.02,full_run_required=all(gates.values()))


def aggregate(records):
    standard={n:{k:float(np.mean([np.mean([f['recall'][n][k] for f in r['folds']]) for r in records])) for k in KINDS} for n in NAMES}
    unseen={n:{k:float(np.mean([np.mean([f['recall'][n][k] for f in r['folds']]) for r in records])) for k in UNSEEN} for n in NAMES}
    tails={n:float(np.median([np.mean([f['normal_tail'][n] for f in r['folds']]) for r in records])) for n in NAMES}
    return dict(summary=standard,unseen_summary=unseen,normal_tail=tails,**select_candidate(standard,unseen,tails))


def checked(name,binding):
    path=OUT/f'{name}_record_antigravity.json'
    if not path.exists():
        return None
    r=read(path)
    assert r['dataset']==name and r['binding_sha256']==binding and len(r['folds'])==5
    for f in r['folds']:
        assert digest(OUT/f['array_file'])==f['array_sha256']
    return r


def worker(payload):
    meta,binding=payload;name=meta['dataset'];found=checked(name,binding)
    if found is not None:
        return found
    previous=next(r for r in read(PRIOR/'pilot_results_antigravity.json')['records'] if r['dataset']==name)
    with closing(readonly_sqlite_antigravity(DB)) as con:
        x=load_normal_train(con,name)
    assert len(x)==meta['n_train']
    start=perf_counter();folds=[]
    with threadpool_limits(limits=1):
        for i,(fit,held) in enumerate(crossfit_plan_antigravity(len(x),seed=SEED)):
            h=SeparatedRankHeads(x[fit]);arrays={k:v.copy() for k,v in h.audit.items()}
            assert not h.fallback and len(h.parents)>=8
            assert not np.intersect1d(fit[h.a_indices],fit[h.b_indices]).size
            assert not np.intersect1d(fit,held).size
            queries=dict(normal=x[held],**perturbations(x[held],SEED+1709*(i+1)),
                         **unseen_perturbations(x[held],SEED+2508+1709*(i+1)))
            for kind,q in queries.items():
                arrays.update({kind+'__'+n:v for n,v in h.score(q).items()})
            recall={};tails={}
            for n in NAMES:
                normal=arrays['normal__'+n];threshold=normal.max()
                tails[n]=float((threshold-np.median(normal))/(np.quantile(normal,.75)-np.quantile(normal,.25)+1e-9))
                recall[n]={k:float(np.mean(arrays[k+'__'+n]>threshold)) for k in KINDS+UNSEEN}
                if n in NAMES[:3]:
                    np.testing.assert_allclose(tails[n],previous['normal_tail_by_fold'][i][n],rtol=1e-12,atol=1e-12)
                    for k in KINDS+UNSEEN:
                        key='folds' if k in KINDS else 'unseen_folds'
                        assert recall[n][k]==previous[key][i][k][n],(name,i,n,k,'control mismatch')
            assert all(v.ndim==1 and np.isfinite(v).all() for v in arrays.values())
            filename=f'{name}_fold{i}_antigravity.npz';temporary=OUT/f'{name}_fold{i}_temporary_antigravity.npz'
            np.savez_compressed(temporary,**arrays);temporary.replace(OUT/filename)
            folds.append(dict(fold=i,fit=fit.tolist(),held=held.tolist(),a=fit[h.a_indices].tolist(),b=fit[h.b_indices].tolist(),
                              parents=fit[h.parents].tolist(),recall=recall,normal_tail=tails,diagnostics=h.diagnostics,
                              array_file=filename,array_sha256=digest(OUT/filename)))
    record=dict(dataset=name,source_family=meta['source_family'],n_train=len(x),folds=folds,binding_sha256=binding,
                seconds=perf_counter()-start,test_values_read=False,test_labels_read=False)
    write_json_atomic_antigravity(OUT/f'{name}_record_antigravity.json',record)
    return record


def main():
    result=P/'pilot_results_antigravity.json';manifest_path=P/'pilot_manifest_antigravity.json'
    if result.exists():
        raise RuntimeError('completed pilot must be verified, not overwritten')
    preflight=read(P/'preflight_antigravity.json');assert preflight['state']=='passed'
    for relative,expected in preflight['artifacts'].items():
        assert digest(ROOT/relative)==expected,relative
    source=ROOT/'experiments/exp495_normal_geometry_antigravity/diagnostics_manifest_antigravity.json'
    population=read(source)['population'];assert len(population)==len({r['source_family'] for r in population})==50
    assert min(r['n_train']-(r['n_train']+4)//5 for r in population)>=32
    prior=read(PRIOR/'pilot_manifest_antigravity.json')
    assert file_sha256_antigravity(DB)==prior['database_sha256']
    assert digest(PRIOR/'pilot_results_antigravity.json')==prior['result_sha256']
    cm=ROOT/'docs/codemap/codemap_exp510_antigravity.json';code={n['path']:n['sha256'] for n in read(cm)['nodes']}
    for relative,expected in code.items():
        assert file_sha256_antigravity(ROOT/relative)==expected,relative
    binding=dict(code=code,database_sha256=prior['database_sha256'],population=population,
                 source_population_sha256=digest(source),prior_result_sha256=prior['result_sha256'],
                 protocol_sha256=digest(P/'protocol_antigravity.html'),preflight_sha256=digest(P/'preflight_antigravity.json'))
    identity=json_digest(binding)
    if manifest_path.exists():
        assert read(manifest_path)['binding_sha256']==identity
    policy=cpu_worker_policy_antigravity();start=perf_counter();OUT.mkdir(exist_ok=True)
    manifest=dict(state='running_train_only',binding=binding,binding_sha256=identity,cpu_policy=policy,
                  test_values_read=False,test_labels_read=False)
    write_json_atomic_antigravity(manifest_path,manifest)
    records=[];failures=[]
    with ProcessPoolExecutor(max_workers=policy['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(worker,(row,identity)):row['dataset'] for row in population}
        for future in as_completed(futures):
            try:
                records.append(future.result())
            except Exception as error:
                failures.append(dict(dataset=futures[future],error=str(error),type=type(error).__name__))
            progress=dict(completed=len(records),total=50,failures=failures,seconds=perf_counter()-start)
            write_json_atomic_antigravity(P/'pilot_progress_antigravity.json',progress);print(json.dumps(progress),flush=True)
    if failures:
        manifest.update(state='failed_train_only',failures=failures);write_json_atomic_antigravity(manifest_path,manifest)
        raise RuntimeError('pilot failure: no data exclusions or TEST evaluation')
    for relative,expected in code.items():
        assert file_sha256_antigravity(ROOT/relative)==expected,relative
    for relative,expected in preflight['artifacts'].items():
        assert digest(ROOT/relative)==expected,relative
    records=sorted(records,key=lambda r:r['dataset'])
    out=dict(state='complete_train_only_pilot',binding_sha256=identity,records=records,datasets=50,families=50,folds=250,
             seconds=perf_counter()-start,test_values_read=False,test_labels_read=False,**aggregate(records))
    write_json_atomic_antigravity(result,out)
    manifest.update(state=out['state'],result_sha256=digest(result));write_json_atomic_antigravity(manifest_path,manifest)
    print(json.dumps({k:out[k] for k in ('selected','gates','worst_standard','mean_standard','mean_unseen')}),flush=True)


if __name__=='__main__':
    main()
