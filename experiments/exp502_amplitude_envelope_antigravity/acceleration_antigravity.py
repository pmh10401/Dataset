"""Complete amplitude-view fit and score benchmark; never rewrite official MiniRocket."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import numba,torch
from threadpoolctl import threadpool_limits
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import write_json_atomic_antigravity,file_sha256_antigravity
from . import amplitude_antigravity as module
P=Path(__file__).resolve().parent

@numba.njit(cache=False)
def numba_views(x):
    a=np.empty(x.shape,dtype=np.float32);b=np.empty(x.shape,dtype=np.float32);length=x.shape[1];w=min(length,max(8,length//10));left=w//2
    for i in range(len(x)):
        square=x[i]*x[i];a[i]=square-np.mean(square);rms=np.empty(length)
        for j in range(length):
            v=0.
            for k in range(w):v+=square[min(length-1,max(0,j+k-left))]/w
            rms[j]=np.sqrt(v)
        rms-=np.mean(rms);scale=np.sqrt(np.mean(rms*rms))
        if scale==0:scale=1.
        b[i]=rms/scale
    return a,b

def torch_views(x):
    y=torch.tensor(x,dtype=torch.float64).square();square=y-y.mean(dim=1,keepdim=True);w=min(x.shape[1],max(8,x.shape[1]//10));left=w//2
    padded=torch.nn.functional.pad(y[:,None,:],(left,w-1-left),mode='replicate')
    rms=torch.nn.functional.conv1d(padded,torch.ones((1,1,w),dtype=torch.float64)/w)[:,0,:].sqrt()
    rms-=rms.mean(dim=1,keepdim=True);scale=rms.std(dim=1,keepdim=True,correction=0)
    return square.to(torch.float32).numpy(),(rms/torch.where(scale==0,1.,scale)).to(torch.float32).numpy()

def main():
    rng=np.random.default_rng(502);x=rng.normal(size=(64,128));q=rng.normal(size=(1024,128));x[-1]=0;q[-1]=0
    original=module.amplitude_views;records={};control=None
    with threadpool_limits(limits=1):
        torch.set_num_threads(1)
        try:
            for name,fn in [('numpy',original),('numba',numba_views),('torch_cpu',torch_views)]:
                module.amplitude_views=fn;start=perf_counter();head=module.AmplitudeHeads(x);out=head.score(q);elapsed=perf_counter()-start
                if control is None:control=out
                record=dict(full_fit_score_seconds_including_first_compile_conversion=elapsed,energy_max_abs_error=max(float(np.max(abs(a-b))) for a,b in zip(fn(q),original(q))),score_max_abs_error=max(float(np.max(abs(control[k]-out[k]))) for k in out),finite=all(np.isfinite(v).all() for v in out.values()),scores_close=all(np.allclose(control[k],out[k],rtol=1e-10,atol=1e-8) for k in out),ranks_equal=all(np.array_equal(np.argsort(control[k],kind='stable'),np.argsort(out[k],kind='stable')) for k in out),candidates_equal=all(np.array_equal(np.flatnonzero(control[k]>np.sort(control[k][:1000])[::-1][j]),np.flatnonzero(out[k]>np.sort(out[k][:1000])[::-1][j])) for k in out for j in (0,1,5,10)))
                start=perf_counter();module.AmplitudeHeads(x).score(q);record['full_fit_score_warm_seconds']=perf_counter()-start;records[name]=record
        finally:module.amplitude_views=original
    path=P.parent/'exp495_normal_geometry_antigravity/acceleration_antigravity.json';old=json.loads(path.read_text())
    records.update(selected='canonical NumPy amplitude conversion + official Numba aeon MiniRocket + float64 OAS CPU',scope='Complete raw/square/RMS MiniRocket fit and transform, all five OAS fits and queries, control and candidate scoring; timings include conversion and JIT warm-up',mlx=old['mlx'],mps=old['mps'],availability=old['availability'],inherited_gpu_probe_sha256=file_sha256_antigravity(path),source_sha256=file_sha256_antigravity(Path(__file__)),decision='Official MiniRocket is already Numba and cannot be replaced by a tensor rewrite under parity policy. Extra row conversion is tensor-eligible but must pass complete-stage parity/speed. float64 GPU unavailable; CPU remains canonical.')
    write_json_atomic_antigravity(P/'acceleration_results_antigravity.json',records);print(json.dumps({k:v for k,v in records.items() if k in ('numpy','numba','torch_cpu','selected')}))

if __name__=='__main__':main()
