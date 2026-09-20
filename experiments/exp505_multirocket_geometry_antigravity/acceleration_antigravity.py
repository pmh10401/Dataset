"""Complete-head CPU acceleration assessment for Exp505 MultiRocket OAS geometry."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from pathlib import Path
from time import perf_counter
import hashlib,importlib.metadata,json
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import ROOT,file_sha256_antigravity,write_json_atomic_antigravity,cpu_worker_policy_antigravity
from experiments.exp505_multirocket_geometry_antigravity.multirocket_antigravity import MultiRocketHeads
P=Path(__file__).resolve().parent

def _version(dist):
    try:return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:return None

def _digest(scores):
    return {k:hashlib.sha256(np.asarray(scores[k],dtype=np.float64).tobytes()).hexdigest() for k in MultiRocketHeads.names}

def _inherited_field(doc,name):
    if isinstance(doc,dict) and name in doc:return doc[name]
    if isinstance(doc,dict):
        for v in doc.values():
            if isinstance(v,dict) and name in v:return v[name]
    return None

def _finite(scores):
    return all(np.isfinite(np.asarray(scores[k],dtype=np.float64)).all() for k in MultiRocketHeads.names)

def _close(a,b):
    return all(np.allclose(np.asarray(a[k],dtype=np.float64),np.asarray(b[k],dtype=np.float64),rtol=1e-10,atol=1e-8) for k in MultiRocketHeads.names)

def _ranks(scores):
    return {k:np.argsort(np.asarray(scores[k],dtype=np.float64),kind='stable') for k in MultiRocketHeads.names}

def _rank_eq(a,b):
    ra,rb=_ranks(a),_ranks(b)
    return all(np.array_equal(ra[k],rb[k]) for k in MultiRocketHeads.names)

def _candidates(scores):
    result={}
    for name in MultiRocketHeads.names:
        v=np.asarray(scores[name],dtype=np.float64)
        descending=np.sort(v[:1000],kind='stable')[::-1]
        result[name]={int(b):np.flatnonzero(v>descending[b]).astype(int).tolist() for b in (0,1,5,10)}
    return result

def _run(fit,query):
    heads=MultiRocketHeads(fit);scores=heads.score(query);return heads,scores

def main():
    result=P/'acceleration_results_antigravity.json'
    policy=cpu_worker_policy_antigravity()
    inherited=ROOT/'experiments/exp495_normal_geometry_antigravity/acceleration_antigravity.json'
    inherited_doc=json.loads(inherited.read_text())
    inherited_probe_sha256=file_sha256_antigravity(inherited)
    rng=np.random.default_rng(505)
    fit=rng.normal(size=(64,128));fit[0]=0;query=rng.normal(size=(1024,128))
    with threadpool_limits(limits=1):
        t0=perf_counter();heads,first=_run(fit,query);first_seconds=perf_counter()-t0
        t0=perf_counter();_,warm=_run(fit,query);warm_seconds=perf_counter()-t0
        again=heads.score(query)
        left=heads.score(query[:512]);right=heads.score(query[512:])
        batched={k:np.concatenate((left[k],right[k])) for k in MultiRocketHeads.names}
    exact_repeated_fit=all(np.array_equal(first[k],warm[k]) for k in MultiRocketHeads.names)
    exact_rescore=all(np.array_equal(first[k],again[k]) for k in MultiRocketHeads.names)
    exact_batch_query=all(np.array_equal(first[k],batched[k]) for k in MultiRocketHeads.names)
    exact_ranks=all(np.array_equal(_ranks(first)[k],_ranks(warm)[k]) for k in MultiRocketHeads.names)
    finite_first=_finite(first);finite_warm=_finite(warm);finite_batched=_finite(batched)
    close_warm=_close(first,warm);close_batch=_close(first,batched)
    ranks_warm=_rank_eq(first,warm);ranks_batch=_rank_eq(first,batched)
    cand_first,cand_warm,cand_batch=_candidates(first),_candidates(warm),_candidates(batched)
    cand_warm_eq=cand_first==cand_warm;cand_batch_eq=cand_first==cand_batch
    success=bool(finite_first and finite_warm and finite_batched and close_warm and close_batch and ranks_warm and ranks_batch and cand_warm_eq and cand_batch_eq)
    out=dict(
        state='complete_acceleration_assessment',
        seed=505,fit_shape=[64,128],query_shape=[1024,128],constant_fit_rows=1,
        geometry_dtype='float64',
        candidates=list(MultiRocketHeads.names),
        budgets=[0,1,5,10],synthetic_normal_query_prefix=1000,
        candidate_rule='strict_greater_than_sorted_normal_descending_at_budget',
        first_seconds=first_seconds,warm_seconds=warm_seconds,
        exact_repeated_fit=bool(exact_repeated_fit),
        exact_rescore=bool(exact_rescore),
        exact_batch_query=bool(exact_batch_query),
        exact_ranks=bool(exact_ranks),
        finite_first=bool(finite_first),finite_warm=bool(finite_warm),finite_batched=bool(finite_batched),
        close_warm=bool(close_warm),close_batch=bool(close_batch),
        ranks_warm=bool(ranks_warm),ranks_batch=bool(ranks_batch),
        candidates_warm=bool(cand_warm_eq),candidates_batch=bool(cand_batch_eq),
        success=success,
        candidate_indices=cand_first,
        score_sha256=_digest(first),
        diagnostics=heads.diagnostics,
        cpu_policy=policy,
        declared_cpu_policy=dict(workers=11,inner_threads=1),
        versions={n:_version(n) for n in ('aeon','numpy','scipy','scikit-learn','numba','torch','mlx','mlx-metal')},
        inherited_probe_sha256=inherited_probe_sha256,
        inherited_mlx=_inherited_field(inherited_doc,'mlx'),
        inherited_mps=_inherited_field(inherited_doc,'mps'),
        inherited_availability=_inherited_field(inherited_doc,'availability'),
        inherited_mlx_mps_float64_oas_unsupported_probe=True,
        mlx_mps_float64_oas_supported=False,
        numba_official_transform_rewrite=False,
        numba_mean3_nearest_rewrite=False,
        numba_speedup_claimed=False,
        torch_gpu_official_multirocket_api=False,
        concurrent_gpu_jobs_benchmarked=False,
        notes='Official MiniRocket and MultiRocket transforms are already Numba. OAS uses scipy cdist and LAPACK eigh on float64 CPU. A Numba mean-3-nearest reduction was not added: no reusable small exact function is available, EnergyHeads already uses np.sort on cdist, and rewriting official transforms would change the experiment. No Numba speedup is claimed. MLX/MPS float64 OAS is unsupported (inherited Exp495 probe). Torch GPU MultiRocket is unavailable via the public official Numba API. CPU policy is 11 workers with inner 1 thread; this file benchmarks one complete head fit+score path only.',
    )
    write_json_atomic_antigravity(result,out)
    print(json.dumps({k:out[k] for k in ('first_seconds','warm_seconds','exact_repeated_fit','exact_batch_query','exact_ranks','success')}),flush=True)

if __name__=='__main__':main()
