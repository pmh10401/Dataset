"""Full new rank-map fit/query timing and downstream score parity."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import numba,torch
from scipy.special import ndtri
from threadpoolctl import threadpool_limits
from experiments.exp495_normal_geometry_antigravity.geometry_heads_antigravity import OASGeometry,NeighborGeometry
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import file_sha256_antigravity,write_json_atomic_antigravity
from .copula_antigravity import GaussianRanks
P=Path(__file__).resolve().parent

@numba.njit(cache=False)
def ranks(values,queries):
    p,n=values.shape;out=np.empty_like(queries)
    for j in range(p):
        for i in range(queries.shape[1]):
            q=queries[j,i];lo=0;hi=n
            while lo<hi:
                mid=(lo+hi)//2
                if values[j,mid]<q:lo=mid+1
                else:hi=mid
            left=lo;lo=0;hi=n
            while lo<hi:
                mid=(lo+hi)//2
                if values[j,mid]<=q:lo=mid+1
                else:hi=mid
            u=(left+.5*(lo-left)+.5)/(n+1);out[j,i]=min(max(u,1/(n+1)),n/(n+1))
    return out

def stage(x,q,backend):
    mapping=GaussianRanks(x);out=[]
    for data in (x,q):
        if backend=='numpy':v=mapping.transform(data)
        elif backend=='numba':v=ndtri(ranks(np.ascontiguousarray(mapping.sorted.T),np.ascontiguousarray(data[:,mapping.columns].T))).T.copy()
        else:
            a=torch.tensor(mapping.sorted.T.copy(),dtype=torch.float64);b=torch.tensor(data[:,mapping.columns].T.copy(),dtype=torch.float64)
            l=torch.searchsorted(a,b,right=False).to(torch.float64);r=torch.searchsorted(a,b,right=True).to(torch.float64);u=(l+.5*(r-l)+.5)/(mapping.n+1)
            v=torch.special.ndtri(u.clamp(1/(mapping.n+1),mapping.n/(mapping.n+1))).numpy().T.copy()
        out.append(v)
    geometry=OASGeometry(out[0]);scores=NeighborGeometry(geometry.transform(out[0])).score(geometry.transform(out[1]))[1]
    return out,scores

def main():
    rng=np.random.default_rng(498);x=np.round(rng.uniform(size=(96,4032))*16)/16;q=np.round(rng.uniform(-.1,1.1,size=(1024,4032))*16)/16;x[:,-2:]=1;q[:,-2:]=1;record={}
    with threadpool_limits(limits=1):
        torch.set_num_threads(1);start=perf_counter();control,s=stage(x,q,'numpy');cpu=perf_counter()-start;record['canonical_seconds']=cpu
        for name in ('numba','torch_cpu'):
            start=perf_counter();features,scores=stage(x,q,name);seconds=perf_counter()-start
            record[name]=dict(seconds_including_conversion_and_first_compile=seconds,speedup=cpu/seconds,finite=np.isfinite(scores).all().item(),
                feature_max_abs_error=max(float(np.max(abs(a-b))) for a,b in zip(features,control)),score_max_abs_error=float(np.max(abs(scores-s))),
                scores_close=bool(np.allclose(scores,s,rtol=1e-10,atol=1e-8)),ranks_equal=bool(np.array_equal(np.argsort(scores,kind='stable'),np.argsort(s,kind='stable'))),
                candidates_equal=all(np.array_equal(np.flatnonzero(scores>np.sort(scores[:1000])[::-1][k]),np.flatnonzero(s>np.sort(s[:1000])[::-1][k])) for k in [0,1,5,10]))
    path=P.parent/'exp495_normal_geometry_antigravity/acceleration_antigravity.json';old=json.loads(path.read_text())
    record.update(mlx=old['mlx'],mps=old['mps'],availability=old['availability'],tensor_eligible=True,selected='canonical NumPy/SciPy float64 CPU',
        inherited_gpu_probe_sha256=file_sha256_antigravity(path),source_sha256=file_sha256_antigravity(Path(__file__)),
        scope='Complete new reference sorting+mask+TRAIN/query rank mapping and existing downstream OAS distance; official QT not rewritten',
        decision='Preserve the NumPy/SciPy path used for TRAIN selection; alternative backends require score and candidate parity plus whole-stage speed benefit. Official QuantileTransformer retained.')
    write_json_atomic_antigravity(P/'acceleration_results_antigravity.json',record);print(json.dumps(record))

if __name__=='__main__':main()
