"""Five-fold normal-TRAIN proxy selection; no TEST rows are queried."""
from .diagnostics_antigravity import (
    P,ROOT,DB,SEED,SQL,readonly_sqlite_antigravity,configure_inner_threads_antigravity,
    write_json_atomic_antigravity,cpu_worker_policy_antigravity,file_sha256_antigravity,
)
configure_inner_threads_antigravity()
from contextlib import closing
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from time import perf_counter
import json,multiprocessing
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.exp479_missing_score_recovery_antigravity.preprocessing_antigravity import _decode_finite_row_antigravity,_standardize_rows_antigravity,lower_median_length_antigravity
from experiments.exp449_residual_fn_method_qualification_antigravity.representations_antigravity import make_representation_antigravity
from experiments.exp449_residual_fn_method_qualification_antigravity.scoring_antigravity import crossfit_plan_antigravity
from .geometry_heads_antigravity import PairedHeads

KINDS=('local_level','local_scale','permuted_quarters','phase_randomized')

def load_normal_train(con,name):
    rows=[_decode_finite_row_antigravity(blob) for (blob,) in con.execute(SQL,(name,))]
    rows=[r for r in rows if r is not None]
    length=lower_median_length_antigravity(tuple(len(r) for r in rows))
    return _standardize_rows_antigravity(rows,length).astype(np.float32)

def perturbations(rows,seed):
    x=np.asarray(rows,float);rng=np.random.default_rng(seed);n,length=x.shape
    out={k:x.copy() for k in KINDS};width=max(1,length//5)
    for i in range(n):
        start=int(rng.integers(0,length-width+1));region=slice(start,start+width)
        out['local_level'][i,region]+=2.5*rng.choice([-1,1])
        out['local_scale'][i,region]*=rng.choice([.2,3.])
        order=rng.permutation(4)
        if np.array_equal(order,np.arange(4)):order=np.roll(order,1)
        pieces=np.array_split(x[i],4);out['permuted_quarters'][i]=np.concatenate([pieces[j] for j in order])
        spectrum=np.fft.rfft(x[i]);phase=rng.uniform(-np.pi,np.pi,len(spectrum))
        randomized=np.abs(spectrum)*np.exp(1j*phase);randomized[0]=spectrum[0]
        if length%2==0:randomized[-1]=spectrum[-1]
        out['phase_randomized'][i]=np.fft.irfft(randomized,n=length)
    for k,v in out.items():
        scale=v.std(axis=1,keepdims=True);v=(v-v.mean(axis=1,keepdims=True))/np.where(scale==0,1,scale)
        if not np.isfinite(v).all():raise ValueError('nonfinite proxy')
        out[k]=v.astype(np.float32)
    return out

def worker(meta):
    start=perf_counter();name=meta['dataset'];folds=[]
    with closing(readonly_sqlite_antigravity(DB)) as con:x=load_normal_train(con,name)
    assert len(x)==meta['n_train']
    with threadpool_limits(limits=1):
        for fold,(fit,query) in enumerate(crossfit_plan_antigravity(len(x),seed=SEED)):
            encoder=make_representation_antigravity('minirocket_local_gap',SEED)
            f=encoder.fit_transform(x[fit]);heads=PairedHeads(f)
            scores=heads.score(encoder.transform(x[query]));threshold={k:float(v.max()) for k,v in scores.items()}
            proxies=perturbations(x[query],SEED+1709*(fold+1));values={}
            for kind,proxy in proxies.items():
                s=heads.score(encoder.transform(proxy));values[kind]={k:float(np.mean(v>threshold[k])) for k,v in s.items()}
            folds.append(values)
    return dict(dataset=name,source_family=meta['source_family'],folds=folds,seconds=perf_counter()-start)

def main():
    start=perf_counter();population=json.loads((P/'diagnostics_manifest_antigravity.json').read_text())['population']
    original=json.loads((ROOT/'experiments/exp494_all_datasets_low_fpr_antigravity/run_manifest_antigravity.json').read_text())
    assert file_sha256_antigravity(DB)==original['binding']['source_binding']['database']
    result=P/'pilot_results_antigravity.json'
    if result.exists():raise RuntimeError('pilot already complete; do not overwrite')
    policy=cpu_worker_policy_antigravity();manifest=dict(state='running',population=population,cpu_policy=policy,
        code={str(path.relative_to(ROOT)):file_sha256_antigravity(path) for path in P.glob('*.py')},
        protocol_sha256=file_sha256_antigravity(P/'protocol_antigravity.html'),test_values_read=False,test_labels_read=False,
        selection_rule='max worst-corruption family-mean proxy TPR at held-normal FP0; tie mean then frozen name order',
        interpretation='TRAIN synthetic-proxy screening only, not benchmark improvement evidence',
        acceleration='Canonical float64 CPU NumPy/SciPy/sklearn for development. Full-run PyTorch/MLX/MPS/Numba parity and timing gate remains required.')
    write_json_atomic_antigravity(P/'pilot_manifest_antigravity.json',manifest);records=[]
    with ProcessPoolExecutor(max_workers=policy['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(worker,r) for r in population]):
            records.append(future.result());print(json.dumps(dict(completed=len(records),total=len(population))),flush=True)
    summary={name:{kind:float(np.mean([np.mean([fold[kind][name] for fold in r['folds']]) for r in records])) for kind in KINDS} for name in PairedHeads.names}
    selected=max(PairedHeads.names,key=lambda name:(min(summary[name].values()),np.mean(list(summary[name].values())),-PairedHeads.names.index(name)))
    out=dict(state='complete_train_only_pilot',summary=summary,selected=selected,records=sorted(records,key=lambda r:r['dataset']),
        test_values_read=False,test_labels_read=False,seconds=perf_counter()-start)
    write_json_atomic_antigravity(result,out);manifest.update(state=out['state'],result_sha256=file_sha256_antigravity(result),seconds=out['seconds'])
    write_json_atomic_antigravity(P/'pilot_manifest_antigravity.json',manifest)
    print(json.dumps(dict(selected=selected,summary=summary,seconds=out['seconds'])),flush=True)

if __name__=='__main__':main()
