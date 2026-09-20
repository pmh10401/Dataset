"""Checkpointed 50-family normal-TRAIN diagnostic. No TEST query or promotion path."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from pathlib import Path
from hashlib import sha256
from contextlib import closing
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter
import json
import multiprocessing
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import ROOT, DB, SEED, readonly_sqlite_antigravity, write_json_atomic_antigravity, cpu_worker_policy_antigravity
from experiments.exp495_normal_geometry_antigravity.pilot_antigravity import load_normal_train
from experiments.exp449_residual_fn_method_qualification_antigravity.scoring_antigravity import crossfit_plan_antigravity
from .diagnostic_antigravity import analyze_fold, aggregate, OPTIONS

P=Path(__file__).resolve().parent; OUT=P/'audit_arrays_antigravity'
PRIOR=ROOT/'experiments/exp508_tail_rank_antigravity'
read=lambda p:json.loads(p.read_text())
digest=lambda p:sha256(p.read_bytes()).hexdigest()
json_hash=lambda x:sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()


def checked(dataset, binding):
    path=OUT/f'{dataset}_record_antigravity.json'
    if not path.exists(): return None
    row=read(path)
    assert row['dataset']==dataset and row['binding_sha256']==binding and len(row['folds'])==5
    for fold in row['folds']:
        assert digest(OUT/fold['array_file'])==fold['array_sha256']
    return row


def worker(payload):
    meta,binding=payload;name=meta['dataset'];old=checked(name,binding)
    if old is not None:return old
    previous=next(r for r in read(PRIOR/'pilot_results_antigravity.json')['records'] if r['dataset']==name)
    start=perf_counter()
    with closing(readonly_sqlite_antigravity(DB)) as con:x=load_normal_train(con,name)
    assert len(x)==meta['n_train']
    folds=[]
    with threadpool_limits(limits=1):
        for index,(fit,held) in enumerate(crossfit_plan_antigravity(len(x),seed=SEED)):
            assert fit.tolist()==previous['fold_boundaries'][index]['fit']
            assert held.tolist()==previous['fold_boundaries'][index]['held']
            arrays,info=analyze_fold(x[fit],x[held],index,previous)
            assert all(np.isfinite(v).all() for v in arrays.values())
            filename=f'{name}_fold{index}_antigravity.npz';temporary=OUT/f'{name}_fold{index}_temporary_antigravity.npz'
            np.savez_compressed(temporary,**arrays);temporary.replace(OUT/filename)
            info.update(fold=index,fit=fit.tolist(),held=held.tolist(),array_file=filename,array_sha256=digest(OUT/filename))
            folds.append(info)
    result=dict(dataset=name,source_family=meta['source_family'],n_train=len(x),folds=folds,
                seconds=perf_counter()-start,binding_sha256=binding,test_values_read=False,test_labels_read=False)
    write_json_atomic_antigravity(OUT/f'{name}_record_antigravity.json',result)
    return result


def main():
    result=P/'results_antigravity.json';path=P/'manifest_antigravity.json'
    if result.exists():raise RuntimeError('completed diagnostic must be verified, not rerun')
    preflight=read(P/'preflight_antigravity.json');assert preflight['state']=='passed'
    for relative,expected in preflight['artifacts'].items():assert digest(ROOT/relative)==expected,relative
    source=ROOT/'experiments/exp495_normal_geometry_antigravity/diagnostics_manifest_antigravity.json'
    population=read(source)['population'];assert len(population)==len({r['source_family'] for r in population})==50
    # This frozen diagnostic population has at least 32 outer-fit rows. Reject a
    # different population rather than inventing arrays for Exp508's fallback.
    for meta in population:
        smallest_fit=meta['n_train']-(meta['n_train']+4)//5
        assert smallest_fit>=16 and smallest_fit//4>0 and smallest_fit-2*(smallest_fit//4)>=2
    prior_manifest=read(PRIOR/'pilot_manifest_antigravity.json')
    assert digest(DB)==prior_manifest['database_sha256']
    assert digest(PRIOR/'pilot_results_antigravity.json')==prior_manifest['result_sha256']
    nodes=read(ROOT/'docs/codemap/codemap_exp509_antigravity.json')['nodes']
    code={n['path']:n['sha256'] for n in nodes}
    for relative,expected in code.items():assert digest(ROOT/relative)==expected,relative
    binding=dict(preflight_sha256=digest(P/'preflight_antigravity.json'),code=code,options=OPTIONS,population=population,
                 source_population_sha256=digest(source),database_sha256=digest(DB),prior_result_sha256=prior_manifest['result_sha256'],
                 protocol_sha256=digest(P/'protocol_antigravity.html'))
    identity=json_hash(binding);policy=cpu_worker_policy_antigravity()
    if path.exists():assert read(path)['binding_sha256']==identity
    OUT.mkdir(exist_ok=True);start=perf_counter()
    manifest=dict(state='running',binding=binding,binding_sha256=identity,cpu_policy=policy,test_values_read=False,test_labels_read=False)
    write_json_atomic_antigravity(path,manifest)
    completed={r['dataset']:checked(r['dataset'],identity) for r in population}
    records=[r for r in completed.values() if r is not None];failures=[]
    with ProcessPoolExecutor(max_workers=policy['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        futures={pool.submit(worker,(r,identity)):r['dataset'] for r in population if completed[r['dataset']] is None}
        for future in as_completed(futures):
            try:records.append(future.result())
            except Exception as error:failures.append(dict(dataset=futures[future],error=str(error),type=type(error).__name__))
            progress=dict(completed=len(records),total=50,failures=failures,seconds=perf_counter()-start)
            write_json_atomic_antigravity(P/'progress_antigravity.json',progress)
            print(json.dumps(progress),flush=True)
    if failures:
        manifest.update(state='failed',failures=failures);write_json_atomic_antigravity(path,manifest)
        raise RuntimeError('diagnostic failed; no dataset exclusions or TEST evaluation')
    for relative,expected in code.items():assert digest(ROOT/relative)==expected,relative
    for relative,expected in preflight['artifacts'].items():assert digest(ROOT/relative)==expected,relative
    records=sorted(records,key=lambda r:r['dataset'])
    out=dict(state='complete_train_diagnostic',binding_sha256=identity,datasets=50,families=50,folds=250,
             records=records,summary=aggregate(records),test_values_read=False,test_labels_read=False,
             model_promotion=False,exp510_started=False,seconds=perf_counter()-start)
    write_json_atomic_antigravity(result,out)
    manifest.update(state=out['state'],result_sha256=digest(result),seconds=out['seconds']);write_json_atomic_antigravity(path,manifest)
    print(json.dumps(dict(state=out['state'],summary=out['summary'])),flush=True)


if __name__=='__main__':main()
