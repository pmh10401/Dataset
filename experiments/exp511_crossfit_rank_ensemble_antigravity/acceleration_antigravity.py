"""Synthetic whole-ensemble timing and parity, including all four optimizations."""
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
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import cpu_worker_policy_antigravity,write_json_atomic_antigravity
from experiments.exp510_separated_rank_antigravity.acceleration_antigravity import parity,candidates
from experiments.exp510_separated_rank_antigravity import separated_antigravity as prior
from . import ensemble_antigravity as model
from .ensemble_antigravity import CrossfitRankHeads

P=Path(__file__).resolve().parent


def stage(fit,query,objective):
    old_prior,old_member=prior.solve,model.solve
    prior.solve=model.solve=lambda n,a:solve(n,a,objective=objective)
    try:
        start=perf_counter();head=CrossfitRankHeads(fit);scores=head.score(query)
        return head,scores,perf_counter()-start
    finally:prior.solve,model.solve=old_prior,old_member


def main():
    target=P/'acceleration_results_antigravity.json'
    if target.exists():raise RuntimeError('completed benchmark must not be overwritten')
    rng=np.random.default_rng(511);fit=rng.normal(size=(64,128));query=rng.normal(size=(1024,128))
    fit[0]=0.;query[0]=0.;reference=None;backends=[]
    with threadpool_limits(limits=1):
        for name,fn in (('numpy',_objective),('numba',objective_numba),('torch_cpu',objective_torch)):
            head,first,cold=stage(fit,query,fn);_,warm,seconds=stage(fit,query,fn)
            if reference is None:
                reference=first;pieces=[head.score(query[:512]),head.score(query[512:])]
                batch=parity({n:np.r_[pieces[0][n],pieces[1][n]] for n in first},first)
            backends.append(dict(backend=name,cold_seconds=cold,warm_seconds=seconds,
                                 first_parity=parity(first,reference),warm_parity=parity(warm,reference)))
    for row in backends:row['speedup']=backends[0]['warm_seconds']/row['warm_seconds']
    eligible=[r['backend'] for r in backends[1:] if r['speedup']>=1.1 and r['first_parity']['parity'] and r['warm_parity']['parity']]
    out=dict(state='complete',success=bool(backends[0]['warm_parity']['parity'] and batch['parity']),
             backends=backends,batch_parity=batch,gpu=_gpu_probes(),seed=511,fit_shape=[64,128],query_shape=[1024,128],
             versions={n:version(n) for n in ('numpy','scipy','scikit-learn','aeon','numba','torch','mlx','mlx-metal')},
             cpu_policy=cpu_worker_policy_antigravity(),selected_backend='numpy_scipy_float64',eligible_alternatives=eligible,
             fallback_reason='NumPy/SciPy remains canonical pending exact whole-stage parity and 1.1x gain; eligibility must be reviewed before preflight.',
             timing_scope='Full-F controls plus four A-only members, B mutations, all four L-BFGS-B fits and query scoring; cold includes JIT/conversion.',
             candidate_indices=candidates(reference),source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
             test_values_read=False,test_labels_read=False,official_transformer_rewritten=False)
    write_json_atomic_antigravity(target,out)
    print(json.dumps({k:out[k] for k in ('success','backends','batch_parity','gpu','eligible_alternatives')}),flush=True)


if __name__=='__main__':main()
