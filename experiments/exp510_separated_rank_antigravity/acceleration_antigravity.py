"""Synthetic whole-head parity and timing for the exact role-separated scorer."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from pathlib import Path
from hashlib import sha256
from importlib.metadata import version
from time import perf_counter
import json
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.exp508_tail_rank_antigravity.acceleration_antigravity import objective_numba,objective_torch,_gpu_probes
from experiments.exp508_tail_rank_antigravity.optimizer_antigravity import _objective
from experiments.exp509_rank_signal_diagnosis_antigravity.diagnostic_antigravity import solve
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import cpu_worker_policy_antigravity
from . import separated_antigravity as model
from .separated_antigravity import SeparatedRankHeads

P=Path(__file__).resolve().parent


def candidates(scores):
    return {name:{str(b):np.flatnonzero(v>np.sort(v[:1000])[::-1][b]).tolist() for b in (0,1,5,10)} for name,v in scores.items()}


def parity(actual,expected):
    close=all(np.isfinite(v).all() and np.allclose(v,expected[n],rtol=1e-10,atol=1e-8) for n,v in actual.items())
    ranking=all(np.array_equal(np.argsort(actual[n],kind='stable'),np.argsort(expected[n],kind='stable')) for n in expected)
    same=candidates(actual)==candidates(expected)
    return dict(parity=bool(close and ranking and same),allclose=bool(close),ranks_equal=bool(ranking),candidates_equal=bool(same),
                max_abs_error=max(float(np.max(abs(actual[n]-expected[n]))) for n in expected))


def stage(fit,query,objective):
    original=model.solve
    model.solve=lambda n,a:solve(n,a,objective=objective)
    try:
        start=perf_counter();head=SeparatedRankHeads(fit);scores=head.score(query);seconds=perf_counter()-start
        return head,scores,seconds
    finally:
        model.solve=original


def main():
    target=P/'acceleration_results_antigravity.json'
    if target.exists():raise RuntimeError('completed benchmark must not be overwritten')
    rng=np.random.default_rng(510);fit=rng.normal(size=(64,128));query=rng.normal(size=(1024,128))
    fit[0]=0.;query[0]=0.;reference=None;backends=[]
    with threadpool_limits(limits=1):
        for name,fn in (('numpy',_objective),('numba',objective_numba),('torch_cpu',objective_torch)):
            head,first,cold=stage(fit,query,fn);_,warm,seconds=stage(fit,query,fn)
            if reference is None:
                reference=first
                pieces=[head.score(query[:512]),head.score(query[512:])]
                batch=parity({n:np.r_[pieces[0][n],pieces[1][n]] for n in first},first)
            backends.append(dict(backend=name,cold_seconds=cold,warm_seconds=seconds,
                                 first_parity=parity(first,reference),warm_parity=parity(warm,reference)))
    for row in backends:row['speedup']=backends[0]['warm_seconds']/row['warm_seconds']
    result=dict(state='complete',success=bool(backends[0]['warm_parity']['parity'] and batch['parity']),
                backends=backends,batch_parity=batch,gpu=_gpu_probes(),seed=510,fit_shape=[64,128],query_shape=[1024,128],
                versions={n:version(n) for n in ('numpy','scipy','scikit-learn','aeon','numba','torch','mlx','mlx-metal')},
                cpu_policy=cpu_worker_policy_antigravity(),selected_backend='numpy_scipy_float64',
                timing_scope='Full F control and A-only fit, B proxies, fixed L-BFGS-B and all query scores; cold includes JIT/transfers.',
                candidate_indices=candidates(reference),source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                test_values_read=False,test_labels_read=False,official_transformer_rewritten=False)
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('success','backends','batch_parity','gpu')}),flush=True)


if __name__=='__main__':
    main()
