"""Complete QUANT fit and score benchmark; never rewrite official QUANTTransformer."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import numba,torch
from importlib.metadata import version
from threadpoolctl import threadpool_limits
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import write_json_atomic_antigravity,file_sha256_antigravity
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import matrix
from . import quant_antigravity as module
P=Path(__file__).resolve().parent

@numba.njit(cache=False)
def numba_interp_rows(x,source,target):
    n,length=x.shape;m=target.shape[0]
    out=np.empty((n,m),dtype=np.float64)
    for i in range(n):
        for j in range(m):
            t=target[j]
            if t<=source[0]:out[i,j]=x[i,0]
            elif t>=source[length-1]:out[i,j]=x[i,length-1]
            else:
                k=0
                while k<length-2 and source[k+1]<t:k+=1
                denom=source[k+1]-source[k]
                frac=0. if denom==0. else (t-source[k])/denom
                out[i,j]=(1.-frac)*x[i,k]+frac*x[i,k+1]
    return out

def numba_cap_series(rows):
    x=matrix(rows);length=x.shape[1]
    if length<=module.QUANT_CAP:capped=np.asarray(x,dtype=np.float32)
    else:
        target=np.linspace(0.0,1.0,module.QUANT_CAP);source=np.linspace(0.0,1.0,length)
        capped=np.asarray(numba_interp_rows(x,source,target),dtype=np.float32)
    if not np.isfinite(capped).all():raise ValueError('nonfinite quant series')
    return capped[:,np.newaxis,:]

def _parity(control,out):
    names=module.QuantHeads.names
    return dict(finite=all(np.isfinite(v).all() for v in out.values()),scores_close=all(np.allclose(control[k],out[k],rtol=1e-10,atol=1e-8) for k in names),ranks_equal=all(np.array_equal(np.argsort(control[k],kind='stable'),np.argsort(out[k],kind='stable')) for k in names),candidates_equal=all(np.array_equal(np.flatnonzero(control[k]>np.sort(control[k][:1000])[::-1][j]),np.flatnonzero(out[k]>np.sort(out[k][:1000])[::-1][j])) for k in names for j in (0,1,5,10)))

def main():
    rng=np.random.default_rng(504);x=rng.normal(size=(64,128));q=rng.normal(size=(1024,128));x[-1]=0;q[-1]=0
    records={};original=module.cap_series
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        start=perf_counter();head=module.QuantHeads(x);out=head.score(q);elapsed=perf_counter()-start
        repeat=head.score(q)
        record=dict(full_fit_score_seconds_including_first_compile_conversion=elapsed,**_parity(out,repeat),dtype_quant_features='float32',dtype_geometry='float64')
        start=perf_counter();warm=module.QuantHeads(x).score(q);record['full_fit_score_warm_seconds']=perf_counter()-start
        record['warm_scores_close']=all(np.allclose(out[k],warm[k],rtol=1e-10,atol=1e-8) for k in out)
        records['canonical']=record
        x2=rng.normal(size=(64,777));q2=rng.normal(size=(1024,777));x2[-1]=0;q2[-1]=0;control=None
        try:
            for name,fn in (('numpy',original),('numba',numba_cap_series)):
                module.cap_series=fn
                start=perf_counter();head=module.QuantHeads(x2);scored=head.score(q2);elapsed=perf_counter()-start
                if control is None:control=scored
                rec=dict(full_fit_score_seconds_including_first_compile_conversion=elapsed,conversion_max_abs_error=float(np.max(np.abs(fn(q2)-original(q2)))),**_parity(control,scored))
                start=perf_counter();module.QuantHeads(x2).score(q2);rec['full_fit_score_warm_seconds']=perf_counter()-start
                records[name]=rec
        finally:
            module.cap_series=original
    path=P.parent/'exp495_normal_geometry_antigravity/acceleration_antigravity.json';old=json.loads(path.read_text())
    records.update(selected='canonical NumPy cap + official Torch CPU QUANTTransformer float32 + official Numba MiniRocket + float64 OAS CPU',scope='Complete MiniRocket and official QUANT fit/transform, all OAS fits and queries, control and candidate scoring; timings include conversion and JIT warm-up; cap interpolation assessed on 64x777/1024x777 with exact same linspace grid; public Torch QUANT is not rewritten',dtype=dict(quant_features='float32',geometry='float64'),versions={n:version(n) for n in ('aeon','torch','numba','numpy','mlx','mlx-metal')},mlx=old['mlx'],mps=old['mps'],availability=old['availability'],inherited_float64_oas_probe=True,inherited_gpu_probe_sha256=file_sha256_antigravity(path),source_sha256=file_sha256_antigravity(Path(__file__)),decision='Official QUANTTransformer is public Torch CPU float32 hardcoded to CPU tensors and cannot be Numba-replaced without rewriting the official transform. GPU is unsupported because the public API is hardcoded to CPU/numpy, not because float32 cannot run on GPU. Extra cap interpolation is Numba-eligible on the same linspace grid but must pass complete-stage parity/speed; canonical NumPy is retained unless full parity and speed are proven. float64 OAS GPU unavailable (inherited Exp495 probe); CPU remains canonical. MiniRocket remains official Numba.')
    write_json_atomic_antigravity(P/'acceleration_results_antigravity.json',records);print(json.dumps({k:records[k] for k in ('canonical','numpy','numba','selected') if k in records}))

if __name__=='__main__':main()
