"""Benchmark the complete new OE fit, normal calibration and query stage."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import numba,torch
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import write_json_atomic_antigravity,file_sha256_antigravity
from .exposure_antigravity import bounded_exposure
P=Path(__file__).resolve().parent

@numba.njit(cache=False)
def dual_ridge(x,y,w,alpha):
    xm=np.sum(x*w[:,None],axis=0)/w.sum();ym=np.sum(y*w)/w.sum()
    a=(x-xm)*np.sqrt(w[:,None]);b=(y-ym)*np.sqrt(w)
    gram=a@a.T+alpha*np.eye(len(a));coef=a.T@np.linalg.solve(gram,b)
    return coef,ym-xm@coef

def torch_ridge(x,y,w,alpha):
    x=torch.tensor(x,dtype=torch.float64);y=torch.tensor(y,dtype=torch.float64);w=torch.tensor(w,dtype=torch.float64)
    xm=(x*w[:,None]).sum(axis=0)/w.sum();ym=(y*w).sum()/w.sum()
    a=(x-xm)*torch.sqrt(w[:,None]);b=(y-ym)*torch.sqrt(w)
    coef=a.T@torch.linalg.solve(a@a.T+alpha*torch.eye(len(a),dtype=torch.float64),b)
    return coef.numpy(),float(ym-xm@coef)

def main():
    rng=np.random.default_rng(497);n=96;normal=rng.normal(size=(n,4032));synthetic=rng.normal(size=(4*n,4032))+.3;query=rng.normal(size=(1024,4032))
    x=np.vstack((normal,synthetic));y=np.r_[np.zeros(n),np.ones(4*n)];w=np.r_[np.full(n,1/n),np.full(4*n,1/(4*n))];anchor=np.linalg.norm(query,axis=1)/np.sqrt(4032)
    def stage(solver):
        out=[]
        for alpha in (1,100):
            if solver is None:
                m=Ridge(alpha=alpha,solver='cholesky').fit(x,y,sample_weight=w);coef,intercept=m.coef_,m.intercept_
            else:coef,intercept=solver(x,y,w,alpha)
            pred=normal@coef+intercept;center=float(np.median(pred));scale=float(1.4826*np.median(abs(pred-center)))
            term=bounded_exposure(query@coef+intercept,center,scale)
            for gamma in (.25,.5):out.append(anchor+gamma*term)
        return out
    record={}
    with threadpool_limits(limits=1):
        torch.set_num_threads(1);start=perf_counter();control=stage(None);cpu=perf_counter()-start;record['canonical_seconds']=cpu
        for name,fn in [('numba',dual_ridge),('torch_cpu',torch_ridge)]:
            start=perf_counter();out=stage(fn);seconds=perf_counter()-start
            record[name]=dict(seconds_including_conversion_and_first_compile=seconds,speedup=cpu/seconds,finite=all(np.isfinite(a).all() for a in out),
                scores_close=all(np.allclose(a,b,rtol=1e-10,atol=1e-8) for a,b in zip(out,control)),max_abs_error=max(float(np.max(abs(a-b))) for a,b in zip(out,control)),
                ranks_equal=all(np.array_equal(np.argsort(a,kind='stable'),np.argsort(b,kind='stable')) for a,b in zip(out,control)),
                candidates_equal=all(np.array_equal(np.flatnonzero(a>np.sort(a[:1000])[::-1][k]),np.flatnonzero(b>np.sort(b[:1000])[::-1][k])) for a,b in zip(out,control) for k in [0,1,5,10]))
    prior=P.parent/'exp495_normal_geometry_antigravity/acceleration_antigravity.json';old=json.loads(prior.read_text())
    record.update(mlx=old['mlx'],mps=old['mps'],availability=old['availability'],tensor_eligible=True,selected='official sklearn float64 CPU',
        inherited_whitening_and_gpu_probe_sha256=file_sha256_antigravity(prior),source_sha256=file_sha256_antigravity(Path(__file__)),
        scope='Complete new OE fit+calibration+query stage for both alphas; immutable encoder/OAS assessed in Exp495',
        decision='OE rejected by preregistered TRAIN-only selection, no new full run. Preserve official Ridge CPU; accelerated alternatives not adopted.')
    write_json_atomic_antigravity(P/'acceleration_results_antigravity.json',record);print(json.dumps(record))

if __name__=='__main__':main()
