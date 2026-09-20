"""Complete support fit/query benchmark; keep the official libsvm CPU solver."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import numba,torch
from scipy.spatial.distance import cdist
from sklearn.svm import OneClassSVM
from threadpoolctl import threadpool_limits
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import write_json_atomic_antigravity,file_sha256_antigravity
from .kernel_support_antigravity import KernelHeads,log_support
P=Path(__file__).resolve().parent

@numba.njit(cache=False)
def squared(x,y):
    out=np.empty((len(x),len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            v=0.
            for k in range(x.shape[1]):v+=(x[i,k]-y[j,k])**2
            out[i,j]=v
    return out

def stage(x,q,distance):
    d=distance(x,x);np.fill_diagonal(d,0);positive=d[np.triu_indices(len(x),1)];scale=np.median(positive[positive>1e-12]);query=distance(q,x)
    ordered=np.sort(query,axis=1);out=[np.sqrt(ordered[:,:3]).mean(axis=1),np.sqrt(ordered[:,0])]
    for factor in (.25,4.):
        gamma=factor/scale;gram=np.exp(-gamma*d)
        for nu in (.01,.1):
            model=OneClassSVM(kernel='precomputed',nu=nu,tol=1e-5,max_iter=100000,cache_size=128).fit(gram)
            assert model.fit_status_==0
            out.append(log_support(query[:,model.support_],gamma,model.dual_coef_.ravel()))
    return out

def main():
    rng=np.random.default_rng(499);raw=rng.normal(size=(96,4032));raw[-1]=raw[0];qraw=rng.normal(size=(1024,4032));record={}
    with threadpool_limits(limits=1):
        head=KernelHeads(raw);x=head.reference;q=head.geometry.transform(head.control.scaler.transform(qraw));expected=head.score(qraw)
        torch.set_num_threads(1);start=perf_counter();control=stage(x,q,lambda a,b:cdist(a,b,'sqeuclidean'));cpu=perf_counter()-start;record['canonical_seconds']=cpu
        for name,value in zip(head.names[1:],control):np.testing.assert_allclose(value,expected[name],rtol=1e-10,atol=1e-8)
        for name,fn in [('numba',squared),('torch_cpu',lambda a,b:torch.cdist(torch.tensor(a,dtype=torch.float64),torch.tensor(b,dtype=torch.float64)).square().numpy())]:
            start=perf_counter();out=stage(x,q,fn);seconds=perf_counter()-start
            record[name]=dict(seconds_including_conversion_and_first_compile=seconds,speedup=cpu/seconds,finite=all(np.isfinite(v).all() for v in out),scores_close=all(np.allclose(a,b,rtol=1e-10,atol=1e-8) for a,b in zip(out,control)),
                score_max_abs_error=max(float(np.max(abs(a-b))) for a,b in zip(out,control)),ranks_equal=all(np.array_equal(np.argsort(a,kind='stable'),np.argsort(b,kind='stable')) for a,b in zip(out,control)),
                candidates_equal=all(np.array_equal(np.flatnonzero(a>np.sort(a[:1000])[::-1][k]),np.flatnonzero(b>np.sort(b[:1000])[::-1][k])) for a,b in zip(out,control) for k in [0,1,5,10]))
    path=P.parent/'exp495_normal_geometry_antigravity/acceleration_antigravity.json';old=json.loads(path.read_text())
    record.update(mlx=old['mlx'],mps=old['mps'],availability=old['availability'],tensor_eligible=True,selected='official libsvm plus SciPy float64 CPU',
        inherited_gpu_probe_sha256=file_sha256_antigravity(path),source_sha256=file_sha256_antigravity(Path(__file__)),
        scope='Complete new pairwise distance, bandwidth, four official SVM fits and query log-support stage; inherited MiniRocket/OAS unchanged',
        decision='TRAIN-only screen selected the existing OAS anchor; no full kernel run. Keep validated official CPU path; no third-party solver rewrite.')
    write_json_atomic_antigravity(P/'acceleration_results_antigravity.json',record);print(json.dumps(record))

if __name__=='__main__':main()
