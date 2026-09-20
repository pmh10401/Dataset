"""Assess the complete new distance head, including conversion and JIT startup."""
from .diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from time import perf_counter
from pathlib import Path
import json
import numpy as np
import numba
import torch
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from .diagnostics_antigravity import P,write_json_atomic_antigravity,assess_acceleration_antigravity

@numba.njit(cache=False)
def distances(x,y):
    out=np.empty((len(x),len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            v=0.
            for k in range(x.shape[1]):v+=(x[i,k]-y[j,k])**2
            out[i,j]=np.sqrt(v)
    return out

def head(x,q,distance):
    rr=distance(x,x);np.fill_diagonal(rr,np.inf)
    radius=np.sort(rr,axis=1)[:,:3].mean(axis=1)
    d=distance(q,x);indices=np.argsort(d,axis=1,kind='stable')[:,:3]
    base=np.take_along_axis(d,indices,axis=1).mean(axis=1)
    return base,(base-radius[indices].mean(axis=1)-1e-9)/(np.median(radius)+1e-9)

def main():
    rng=np.random.default_rng(495);x=rng.normal(size=(96,4032));q=rng.normal(size=(1024,4032))
    availability=assess_acceleration_antigravity();record={'availability':availability,'tensor_eligible':True,'selected':'NumPy/SciPy float64 CPU'}
    with threadpool_limits(limits=1):
        start=perf_counter();control=head(x,q,cdist);cpu=perf_counter()-start
        record['scipy_seconds']=cpu
        for name,distance in [('numba',distances),('torch_cpu',lambda a,b:torch.cdist(torch.tensor(a,dtype=torch.float64),torch.tensor(b,dtype=torch.float64)).numpy())]:
            torch.set_num_threads(1);start=perf_counter();actual=head(x,q,distance);seconds=perf_counter()-start
            error=max(float(np.max(abs(a-b))) for a,b in zip(actual,control))
            rank_equal=all(np.array_equal(np.argsort(a,kind='stable'),np.argsort(b,kind='stable')) for a,b in zip(actual,control))
            finite=all(np.isfinite(a).all() for a in actual)
            budget_equal=all(np.array_equal(np.flatnonzero(a>np.sort(a[:1000])[::-1][k]),np.flatnonzero(b>np.sort(b[:1000])[::-1][k])) for a,b in zip(actual,control) for k in [0,1,5,10])
            record[name]=dict(seconds_including_first_call=seconds,max_abs_error=error,rank_equal=rank_equal,candidate_indices_equal=budget_equal,finite=finite,speedup=cpu/seconds)
    for name in ('mlx','mps'):
        try:
            if name=='mlx':
                import mlx.core as mx
                value=mx.array(x[:16,:32],dtype=mx.float64)
                eigen,vectors=mx.linalg.eigh(value@value.T);mx.eval(eigen,vectors)
            else:value=torch.tensor(x,dtype=torch.float64,device='mps')
            record[name]=dict(required_float64_operations_supported=True,selected=False,reason='Full eigensystem and score parity not established; CPU remains canonical')
        except Exception as e:record[name]=dict(required_float64_operations_supported=False,reason=str(e))
    record['decision']='Retain validated SciPy CPU score pipeline. Numba compilation timing included. Torch distance-stage probe does not establish whole whitening-pipeline speed/parity; accelerator adoption requires complete pipeline smoke.'
    record['source_sha256']=__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest()
    write_json_atomic_antigravity(P/'acceleration_antigravity.json',record);print(json.dumps(record))

if __name__=='__main__':main()
