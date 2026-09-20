"""Complete spatial-pyramid fit and score benchmark; never rewrite official MiniRocket."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import numba,torch
from threadpoolctl import threadpool_limits
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import write_json_atomic_antigravity,file_sha256_antigravity
from . import spatial_antigravity as module
P=Path(__file__).resolve().parent

@numba.njit(cache=False)
def numba_quarters(x):
    n,length=x.shape;width=max(9,length//4);base=length//4;extra=length%4
    out=np.empty((n*4,width),dtype=np.float64)
    for i in range(n):
        start=0
        for q in range(4):
            size=base+(1 if q<extra else 0)
            dest=i*4+q
            if size==width:
                for j in range(width):out[dest,j]=x[i,start+j]
            elif size==1:
                value=x[i,start]
                for j in range(width):out[dest,j]=value
            else:
                last=size-1
                for j in range(width):
                    pos=j*last/(width-1);k=int(pos)
                    if k>=last:out[dest,j]=x[i,start+last]
                    else:
                        frac=pos-k;out[dest,j]=(1.-frac)*x[i,start+k]+frac*x[i,start+k+1]
            start+=size
    return out

def torch_quarters(x):
    y=torch.tensor(x,dtype=torch.float64);n,length=y.shape;width=max(9,length//4);base=length//4;extra=length%4
    parts=[];start=0
    for q in range(4):
        size=base+(1 if q<extra else 0);seg=y[:,start:start+size]
        if size==width:resampled=seg
        elif size==1:resampled=seg.expand(-1,width)
        else:resampled=torch.nn.functional.interpolate(seg[:,None,:],size=width,mode='linear',align_corners=True)[:,0,:]
        parts.append(resampled);start+=size
    return torch.stack(parts,dim=1).reshape(n*4,width).numpy()

def main():
    rng=np.random.default_rng(503);x=rng.normal(size=(64,128));q=rng.normal(size=(1024,128));x[-1]=0;q[-1]=0
    original=module.quarter_segments;records={};control=None
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        try:
            for name,fn in [('numpy',original),('numba',numba_quarters),('torch_cpu',torch_quarters)]:
                module.quarter_segments=fn;start=perf_counter();head=module.SpatialHeads(x);out=head.score(q);elapsed=perf_counter()-start
                if control is None:control=out
                record=dict(full_fit_score_seconds_including_first_compile_conversion=elapsed,conversion_max_abs_error=float(np.max(np.abs(fn(q)-original(q)))),score_max_abs_error=max(float(np.max(np.abs(control[k]-out[k]))) for k in out),finite=all(np.isfinite(v).all() for v in out.values()),scores_close=all(np.allclose(control[k],out[k],rtol=1e-10,atol=1e-8) for k in out),ranks_equal=all(np.array_equal(np.argsort(control[k],kind='stable'),np.argsort(out[k],kind='stable')) for k in out),candidates_equal=all(np.array_equal(np.flatnonzero(control[k]>np.sort(control[k][:1000])[::-1][j]),np.flatnonzero(out[k]>np.sort(out[k][:1000])[::-1][j])) for k in out for j in (0,1,5,10)))
                start=perf_counter();module.SpatialHeads(x).score(q);record['full_fit_score_warm_seconds']=perf_counter()-start;records[name]=record
        finally:module.quarter_segments=original
        records['supplementary_parity']={}
        for length in (11,131):
            xs=rng.normal(size=(8,length));qs=rng.normal(size=(1024,length));xs[-1]=0;qs[-1]=0
            control_conv=original(qs);control_out=module.SpatialHeads(xs).score(qs)
            block={}
            for name,fn in [('numba',numba_quarters),('torch_cpu',torch_quarters)]:
                try:
                    module.quarter_segments=fn;conv=fn(qs);out=module.SpatialHeads(xs).score(qs)
                    block[name]=dict(conversion_finite=bool(np.isfinite(conv).all() and np.isfinite(control_conv).all()),conversion_close=bool(np.allclose(conv,control_conv,rtol=1e-10,atol=1e-8)),finite=all(np.isfinite(v).all() for v in out.values()),scores_close=all(np.allclose(control_out[k],out[k],rtol=1e-10,atol=1e-8) for k in out),ranks_equal=all(np.array_equal(np.argsort(control_out[k],kind='stable'),np.argsort(out[k],kind='stable')) for k in out),candidates_equal=all(np.array_equal(np.flatnonzero(control_out[k]>np.sort(control_out[k][:1000])[::-1][j]),np.flatnonzero(out[k]>np.sort(out[k][:1000])[::-1][j])) for k in out for j in (0,1,5,10)))
                finally:module.quarter_segments=original
            records['supplementary_parity'][str(length)]=block
    path=P.parent/'exp495_normal_geometry_antigravity/acceleration_antigravity.json';old=json.loads(path.read_text())
    records.update(selected='canonical NumPy quarter resample + official Numba aeon MiniRocket + float64 OAS CPU',scope='Complete raw and shared-quarter MiniRocket fit and transform, all OAS fits and queries, control and candidate scoring; timings include conversion and JIT warm-up; interpolation parity smoke at lengths 11 and 131 is recorded separately',mlx=old['mlx'],mps=old['mps'],availability=old['availability'],inherited_gpu_probe_sha256=file_sha256_antigravity(path),source_sha256=file_sha256_antigravity(Path(__file__)),decision='Official MiniRocket is already Numba and cannot be replaced by a tensor rewrite under parity policy. Extra segment resampling is tensor-eligible but must pass complete-stage parity/speed, including interpolation smoke at lengths 11 and 131. Canonical NumPy is retained; an accelerator is not selected when any supplementary parity field is false. float64 GPU unavailable (inherited Exp495 probe); CPU remains canonical.')
    write_json_atomic_antigravity(P/'acceleration_results_antigravity.json',records);print(json.dumps({k:v for k,v in records.items() if k in ('numpy','numba','torch_cpu','selected')}))

if __name__=='__main__':main()
