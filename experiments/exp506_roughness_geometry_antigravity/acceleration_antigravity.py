"""Complete-head CPU acceleration assessment for Exp506 roughness OAS geometry."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from pathlib import Path
from time import perf_counter
import hashlib,importlib.metadata,json
import numpy as np,torch
from numba import njit
from threadpoolctl import threadpool_limits
torch.set_num_threads(1)
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import ROOT,file_sha256_antigravity,write_json_atomic_antigravity,cpu_worker_policy_antigravity
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import matrix
from experiments.exp506_roughness_geometry_antigravity import roughness_antigravity as R
from experiments.exp506_roughness_geometry_antigravity.roughness_antigravity import RoughnessHeads,roughness_rows
P=Path(__file__).resolve().parent
NAMES=RoughnessHeads.names

def _version(dist):
    try:return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:return None

def _top(doc,name):
    if isinstance(doc,dict) and name in doc:return doc[name]
    if isinstance(doc,dict):
        for v in doc.values():
            if isinstance(v,dict) and name in v:return v[name]
    return None

def _digest(scores):
    return {k:hashlib.sha256(np.asarray(scores[k],dtype=np.float64).tobytes()).hexdigest() for k in NAMES}

def _finite(scores):
    return all(np.isfinite(np.asarray(scores[k],dtype=np.float64)).all() for k in NAMES)

def _close(a,b):
    return all(np.allclose(np.asarray(a[k],dtype=np.float64),np.asarray(b[k],dtype=np.float64),rtol=1e-10,atol=1e-8) for k in NAMES)

def _ranks(scores):
    return {k:np.argsort(np.asarray(scores[k],dtype=np.float64),kind='stable') for k in NAMES}

def _rank_eq(a,b):
    ra,rb=_ranks(a),_ranks(b)
    return all(np.array_equal(ra[k],rb[k]) for k in NAMES)

def _candidates(scores):
    result={}
    for name in NAMES:
        v=np.asarray(scores[name],dtype=np.float64)
        descending=np.sort(v[:1000],kind='stable')[::-1]
        result[name]={int(b):np.flatnonzero(v>descending[b]).astype(int).tolist() for b in (0,1,5,10)}
    return result

def _error(a,b):
    return max(float(np.max(np.abs(np.asarray(a[k],dtype=np.float64)-np.asarray(b[k],dtype=np.float64)))) for k in NAMES)

def _run(fit,query):
    heads=RoughnessHeads(fit);scores=heads.score(query);return heads,scores

@njit
def _roughness_numba(x):
    n,L=x.shape;y=np.empty((n,L),dtype=np.float64)
    if L<2:
        y[:]=0.0
        return y
    for i in range(n):
        y[i,0]=abs(x[i,1]-x[i,0]);acc=y[i,0]
        for j in range(1,L):
            y[i,j]=abs(x[i,j]-x[i,j-1]);acc+=y[i,j]
        mean=acc/L
        for j in range(L):y[i,j]-=mean
    return y

def roughness_rows_numba(rows):
    y=_roughness_numba(matrix(rows))
    if not np.isfinite(y).all():raise ValueError('nonfinite roughness rows')
    y=y.astype(np.float32)
    if not np.isfinite(y).all():raise ValueError('nonfinite roughness rows')
    return y

def roughness_rows_torch(rows):
    t=torch.as_tensor(matrix(rows),dtype=torch.float64,device='cpu')
    if t.shape[1]<2:y=torch.zeros_like(t)
    else:
        d=torch.abs(torch.diff(t,dim=1));d=torch.cat((d[:,:1],d),dim=1);y=d-d.mean(dim=1,keepdim=True)
    out=np.asarray(y.to(torch.float32).cpu().numpy())
    if not np.isfinite(out).all():raise ValueError('nonfinite roughness rows')
    return np.asarray(out,dtype=np.float32)

def _stage(fit,query,helper):
    original=R.roughness_rows
    R.roughness_rows=helper
    try:
        t0=perf_counter();heads,first=_run(fit,query);first_seconds=perf_counter()-t0
        t0=perf_counter();_,warm=_run(fit,query);warm_seconds=perf_counter()-t0
        return heads,first,warm,first_seconds,warm_seconds
    finally:
        R.roughness_rows=original

def _backend(name,first,warm,first_seconds,warm_seconds,ref,ref_warm,helper,fit,query):
    helper_err=float(np.max(np.abs(np.asarray(helper(fit),dtype=np.float64)-np.asarray(roughness_rows(fit),dtype=np.float64))))
    helper_q=float(np.max(np.abs(np.asarray(helper(query),dtype=np.float64)-np.asarray(roughness_rows(query),dtype=np.float64))))
    parity=bool(_finite(first) and _finite(warm) and _close(first,ref) and _close(warm,ref) and _rank_eq(first,ref) and _rank_eq(warm,ref) and _candidates(first)==_candidates(ref) and _candidates(warm)==_candidates(ref))
    speedup=(None if warm_seconds<=0 else float(ref_warm/warm_seconds))
    return dict(name=name,first_seconds=first_seconds,warm_seconds=warm_seconds,speedup_vs_numpy_warm=speedup,max_abs_error_vs_numpy=_error(first,ref),helper_max_abs_error=max(helper_err,helper_q),allclose=bool(_close(first,ref)),ranks_equal=bool(_rank_eq(first,ref)),candidates_equal=_candidates(first)==_candidates(ref),finite=bool(_finite(first) and _finite(warm)),parity=parity,material_full_stage_improvement=bool(parity and speedup is not None and speedup>=1.2))

def main():
    result=P/'acceleration_results_antigravity.json'
    policy=cpu_worker_policy_antigravity()
    inherited=ROOT/'experiments/exp495_normal_geometry_antigravity/acceleration_antigravity.json'
    inherited_doc=json.loads(inherited.read_text())
    rng=np.random.default_rng(506)
    fit=rng.normal(size=(64,128));fit[0]=0;query=rng.normal(size=(1024,128));query[0]=0
    with threadpool_limits(limits=1):
        t0=perf_counter();heads,first=_run(fit,query);first_seconds=perf_counter()-t0
        t0=perf_counter();_,warm=_run(fit,query);warm_seconds=perf_counter()-t0
        again=heads.score(query)
        left=heads.score(query[:512]);right=heads.score(query[512:])
        batched={k:np.concatenate((left[k],right[k])) for k in NAMES}
        backends=[_backend('numpy',first,warm,first_seconds,warm_seconds,first,warm_seconds,roughness_rows,fit,query)]
        for name,helper in (('numba',roughness_rows_numba),('torch_cpu',roughness_rows_torch)):
            _,b_first,b_warm,b_first_s,b_warm_s=_stage(fit,query,helper)
            backends.append(_backend(name,b_first,b_warm,b_first_s,b_warm_s,first,warm_seconds,helper,fit,query))
    exact_repeated_fit=all(np.array_equal(first[k],warm[k]) for k in NAMES)
    exact_rescore=all(np.array_equal(first[k],again[k]) for k in NAMES)
    exact_batch_query=all(np.array_equal(first[k],batched[k]) for k in NAMES)
    cand_first=_candidates(first)
    success=bool(_finite(first) and _finite(warm) and _finite(batched) and _close(first,warm) and _close(first,batched) and _rank_eq(first,warm) and _rank_eq(first,batched) and cand_first==_candidates(warm)==_candidates(batched))
    out=dict(
        state='complete_acceleration_assessment',
        seed=506,fit_shape=[64,128],query_shape=[1024,128],constant_fit_rows=1,constant_query_rows=1,
        geometry_dtype='float64',candidates=list(NAMES),budgets=[0,1,5,10],synthetic_normal_query_prefix=1000,
        candidate_rule='strict_greater_than_sorted_normal_descending_at_budget',
        first_seconds=first_seconds,warm_seconds=warm_seconds,
        exact_repeated_fit=bool(exact_repeated_fit),exact_rescore=bool(exact_rescore),exact_batch_query=bool(exact_batch_query),
        exact_ranks=bool(all(np.array_equal(_ranks(first)[k],_ranks(warm)[k]) for k in NAMES)),
        finite_first=bool(_finite(first)),finite_warm=bool(_finite(warm)),finite_batched=bool(_finite(batched)),
        close_warm=bool(_close(first,warm)),close_batch=bool(_close(first,batched)),
        ranks_warm=bool(_rank_eq(first,warm)),ranks_batch=bool(_rank_eq(first,batched)),
        candidates_warm=cand_first==_candidates(warm),candidates_batch=cand_first==_candidates(batched),
        success=success,candidate_indices=cand_first,score_sha256=_digest(first),diagnostics=heads.diagnostics,
        cpu_policy=policy,declared_cpu_policy=dict(workers=11,inner_threads=1),
        versions={n:_version(n) for n in ('aeon','numpy','numba','torch','mlx','mlx-metal')},
        inherited_probe_sha256=file_sha256_antigravity(inherited),
        mlx=_top(inherited_doc,'mlx'),mps=_top(inherited_doc,'mps'),availability=_top(inherited_doc,'availability'),
        mlx_mps_float64_oas_supported=False,inherited_mlx_mps_float64_oas_unsupported_probe=True,
        numba_official_transform_rewrite=False,concurrent_gpu_jobs_benchmarked=False,
        canonical_helper='numpy',numpy_helper_retained=True,backend_promotion=None,
        backends=backends,
        notes='Canonical roughness_rows stays NumPy. A backend may be recorded as material only if full-stage parity and speedup hold; promotion needs Codex before pilot freeze. Official MiniRocket was not rewritten. Public NumPy/scipy geometry is retained. No GPU concurrent jobs. MLX/MPS float64 OAS is unsupported (inherited Exp495 probe). CPU policy is 11 workers with inner 1 thread.',
    )
    write_json_atomic_antigravity(result,out)
    print(json.dumps({k:out[k] for k in ('first_seconds','warm_seconds','success','canonical_helper')}),flush=True)

if __name__=='__main__':main()
