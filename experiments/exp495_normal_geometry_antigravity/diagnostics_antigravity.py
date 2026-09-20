"""TRAIN-only mechanism audit before choosing a new anomaly score."""
from experiments.exp491_gain_prediction_antigravity.runtime_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import closing
from pathlib import Path
from time import perf_counter
import hashlib,json,multiprocessing,os
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.exp491_gain_prediction_antigravity.runtime_antigravity import (
    ROOT_ANTIGRAVITY as ROOT,DATABASE_PATH_ANTIGRAVITY as DB,SEED_ANTIGRAVITY as SEED,
    readonly_sqlite_antigravity,write_json_atomic_antigravity,cpu_worker_policy_antigravity,
    assess_acceleration_antigravity,file_sha256_antigravity,
)
from experiments.exp479_missing_score_recovery_antigravity.preprocessing_antigravity import (
    _decode_finite_row_antigravity,_standardize_rows_antigravity,lower_median_length_antigravity,
)
from experiments.exp449_residual_fn_method_qualification_antigravity.representations_antigravity import make_representation_antigravity
from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import LocalGapModelAntigravity

P=Path(__file__).resolve().parent
SQL="SELECT i.values_blob FROM instances i JOIN datasets d ON d.id=i.dataset_id WHERE d.name=? AND i.split='TRAIN' AND i.label='0' ORDER BY i.instance_index"

def effective_rank(x):
    x=np.asarray(x,float);x=x-x.mean(axis=0)
    gram=x@x.T;energy=float(np.sum(gram*gram))
    return float(np.trace(gram)**2/energy) if energy else 0.

def worker(meta):
    start=perf_counter();name=meta['dataset']
    with closing(readonly_sqlite_antigravity(DB)) as con:
        rows=[_decode_finite_row_antigravity(b) for (b,) in con.execute(SQL,(name,))]
    rows=[r for r in rows if r is not None]
    assert len(rows)==meta['n_train']
    target=lower_median_length_antigravity(tuple(len(r) for r in rows))
    x=_standardize_rows_antigravity(rows,target).astype(np.float32)
    order=np.random.default_rng(SEED).permutation(len(x));cut=3*len(x)//4;fit=order[:cut];query=order[cut:]
    with threadpool_limits(limits=1):
        encoder=make_representation_antigravity('minirocket_local_gap',SEED)
        f=encoder.fit_transform(x[fit]);q=encoder.transform(x[query]);model=LocalGapModelAntigravity(f)
        z=model.scaler.transform(f);v=model.scaler.transform(q)
        distances,indices=model.query_model.kneighbors(v)
        delta=(v-z[indices[:,0]])**2;total=delta.sum(axis=1)
        top=max(1,int(np.ceil(f.shape[1]*.01)))
        shares=np.sort(delta,axis=1)[:,-top:].sum(axis=1)/np.maximum(total,1e-30)
        correction=model.local_radius[indices].mean(axis=1)
        gap=(distances.mean(axis=1)-correction-1e-9)/model.scale
        np.testing.assert_allclose(gap,model.score(q),rtol=0,atol=1e-12)
        out=dict(dataset=name,source_family=meta['source_family'],n_train=len(x),fit_rows=len(f),query_rows=len(q),
            feature_dimensions=f.shape[1],effective_rank=effective_rank(z),
            zero_iqr_fraction=float(np.mean(np.ptp(np.quantile(f,[.1,.9],axis=0),axis=0)==0)),
            top_one_percent_energy_median=float(np.median(shares)),top_one_percent_energy_max=float(shares.max()),
            correction_distance_ratio_median=float(np.median(correction/np.maximum(distances.mean(axis=1),1e-30))),
            normal_gap_max=float(gap.max()),seconds=perf_counter()-start)
    return out

def main():
    start=perf_counter();source=ROOT/'experiments/exp494_all_datasets_low_fpr_antigravity'
    original=json.loads((source/'run_manifest_antigravity.json').read_text())
    assert file_sha256_antigravity(DB)==original['binding']['source_binding']['database']
    population=[]
    for path in sorted((source/'score_checkpoints_antigravity').glob('*_meta_antigravity.json')):
        r=json.loads(path.read_text())
        if r['origin']=='verified_Exp489_492_493' and 40<=r['n_train']<=400 and r['target_length']<=600:
            population.append({k:r[k] for k in ('dataset','source_family','n_train','target_length')})
    assert len(population)==50 and len({r['source_family'] for r in population})==50
    policy=cpu_worker_policy_antigravity()
    write_json_atomic_antigravity(P/'diagnostics_manifest_antigravity.json',dict(state='running',population=population,
        cpu_policy=policy,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        purpose='TRAIN-only mechanism diagnostic, not an improvement claim or model selection from TEST',
        selection='all50representatives with40<=normalTRAIN<=400 andtarget_length<=600; nooutcome-based selection',
        test_values_read=False,test_labels_read=False,acceleration=assess_acceleration_antigravity(),
        backend_reason='Frozen official CPU MiniRocket; small float64 Gram summaries and sorting use existing NumPy/BLAS. Additional JIT/device transfer not justified for this diagnostic. No new scoring algorithm.'))
    records=[]
    with ProcessPoolExecutor(max_workers=policy['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(worker,r) for r in population]):
            records.append(future.result());print(json.dumps(dict(completed=len(records),total=len(population))),flush=True)
    out=dict(state='complete_train_only_diagnostic',records=sorted(records,key=lambda r:r['dataset']),
        test_values_read=False,test_labels_read=False,seconds=perf_counter()-start)
    write_json_atomic_antigravity(P/'geometry_diagnostics_antigravity.json',out)
    manifest=json.loads((P/'diagnostics_manifest_antigravity.json').read_text())
    manifest.update(state=out['state'],result_sha256=file_sha256_antigravity(P/'geometry_diagnostics_antigravity.json'),seconds=out['seconds'])
    write_json_atomic_antigravity(P/'diagnostics_manifest_antigravity.json',manifest)
    print(json.dumps({k:float(np.median([r[k] for r in records])) for k in ('effective_rank','zero_iqr_fraction','top_one_percent_energy_median')}),flush=True)

if __name__=='__main__':main()
