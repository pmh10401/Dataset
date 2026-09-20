"""TRAIN-only proxy screening of the preregistered spatial-pyramid grid."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from contextlib import closing
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from time import perf_counter
import json,multiprocessing
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import ROOT,DB,SEED,readonly_sqlite_antigravity,write_json_atomic_antigravity,cpu_worker_policy_antigravity,file_sha256_antigravity
from experiments.exp495_normal_geometry_antigravity.pilot_antigravity import load_normal_train,perturbations,KINDS
from experiments.exp449_residual_fn_method_qualification_antigravity.scoring_antigravity import crossfit_plan_antigravity
from .spatial_antigravity import SpatialHeads
P=Path(__file__).resolve().parent

def worker(meta):
    start=perf_counter();folds=[];tails=[];diagnostics=[]
    with closing(readonly_sqlite_antigravity(DB)) as con:x=load_normal_train(con,meta['dataset'])
    assert len(x)==meta['n_train']
    with threadpool_limits(limits=1):
        for fold,(fit,query) in enumerate(crossfit_plan_antigravity(len(x),seed=SEED)):
            assert not np.intersect1d(fit,query).size
            heads=SpatialHeads(x[fit])
            diagnostics.append(heads.diagnostics)
            scores=heads.score(x[query]);threshold={k:float(v.max()) for k,v in scores.items()}
            tails.append({k:float((v.max()-np.median(v))/(np.quantile(v,.75)-np.quantile(v,.25)+1e-9)) for k,v in scores.items()})
            values={}
            for kind,proxy in perturbations(x[query],SEED+1709*(fold+1)).items():
                s=heads.score(proxy);values[kind]={k:float(np.mean(v>threshold[k])) for k,v in s.items()}
            folds.append(values)
    return dict(dataset=meta['dataset'],source_family=meta['source_family'],folds=folds,normal_tail_by_fold=tails,diagnostics=diagnostics,seconds=perf_counter()-start)

def main():
    start=perf_counter();result=P/'pilot_results_antigravity.json'
    if result.exists():raise RuntimeError('completed pilot must not be overwritten')
    source=ROOT/'experiments/exp495_normal_geometry_antigravity/diagnostics_manifest_antigravity.json'
    population=json.loads(source.read_text())['population'];assert len(population)==len({r['source_family'] for r in population})==50
    old=json.loads((ROOT/'experiments/exp494_all_datasets_low_fpr_antigravity/run_manifest_antigravity.json').read_text())
    assert file_sha256_antigravity(DB)==old['binding']['source_binding']['database']
    cm=json.loads((ROOT/'docs/codemap/codemap_exp503_antigravity.json').read_text());code={n['path']:n['sha256'] for n in cm['nodes']}
    for path,h in code.items():assert file_sha256_antigravity(ROOT/path)==h
    policy=cpu_worker_policy_antigravity();manifest=dict(state='running',population=population,source_population_sha256=file_sha256_antigravity(source),
        code=code,protocol_sha256=file_sha256_antigravity(P/'protocol_antigravity.html'),database_sha256=file_sha256_antigravity(DB),cpu_policy=policy,
        test_values_read=False,test_labels_read=False,selection_rule='max minimum corruption mean recall, then mean, then frozen candidate order',
        backend='official Numba MiniRocket raw plus one shared quarter MiniRocket(n_kernels=1024); canonical float64 OAS; full acceleration assessment recorded separately')
    write_json_atomic_antigravity(P/'pilot_manifest_antigravity.json',manifest);records=[]
    with ProcessPoolExecutor(max_workers=policy['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(worker,r) for r in population]):
            records.append(future.result());print(json.dumps(dict(completed=len(records),total=len(population))),flush=True)
    summary={name:{kind:float(np.mean([np.mean([fold[kind][name] for fold in r['folds']]) for r in records])) for kind in KINDS} for name in SpatialHeads.names}
    tails={n:float(np.median([np.mean([v[n] for v in row['normal_tail_by_fold']]) for row in records])) for n in SpatialHeads.names}
    assert all(np.isfinite(v) for v in tails.values())
    eligible=list(SpatialHeads.names[:2])+[n for n in SpatialHeads.names[2:] if tails[n]<=3*max(tails['MiniRocket'],1e-9)]
    criterion=lambda n:(min(summary[n].values()),np.mean(list(summary[n].values())),-SpatialHeads.names.index(n))
    anchor=max(SpatialHeads.names[:2],key=criterion);unrestricted=max(eligible,key=criterion);selected=unrestricted
    if unrestricted not in SpatialHeads.names[:2] and (min(summary[unrestricted].values())<min(summary[anchor].values())+.02 or np.mean(list(summary[unrestricted].values()))<np.mean(list(summary[anchor].values()))):selected=anchor
    out=dict(state='complete_train_only_pilot',summary=summary,selected=selected,unrestricted_winner=unrestricted,best_anchor=anchor,eligible=eligible,normal_tail=tails,required_proxy_margin=.02,records=sorted(records,key=lambda r:r['dataset']),test_values_read=False,test_labels_read=False,seconds=perf_counter()-start)
    write_json_atomic_antigravity(result,out);manifest.update(state=out['state'],result_sha256=file_sha256_antigravity(result),seconds=out['seconds'])
    write_json_atomic_antigravity(P/'pilot_manifest_antigravity.json',manifest);print(json.dumps({k:out[k] for k in ('selected','summary','seconds')}),flush=True)

if __name__=='__main__':main()
