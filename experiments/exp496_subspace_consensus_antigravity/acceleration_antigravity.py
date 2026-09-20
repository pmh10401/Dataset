"""Complete subspace fit-plus-score assessment on identical deterministic arrays."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler
from threadpoolctl import threadpool_limits
from experiments.exp495_normal_geometry_antigravity.acceleration_probe_antigravity import distances,head
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import SEED,write_json_atomic_antigravity,file_sha256_antigravity
from experiments.exp449_residual_fn_method_qualification_antigravity.local_gap_antigravity import LocalGapModelAntigravity
from .subspace_antigravity import SubspaceHeads,combine
P=Path(__file__).resolve().parent

def alternative(x,q,distance):
    control=LocalGapModelAntigravity(x);base=control.score(q)
    rng=np.random.default_rng(SEED+496);bags=[]
    for _ in range(32):
        c=rng.choice(x.shape[1],min(256,x.shape[1]),replace=False)
        scaler=RobustScaler(quantile_range=(10,90));z=scaler.fit_transform(x[:,c]);v=scaler.transform(q[:,c])
        bags.append(head(z,v,distance)[1])
    return combine(base,np.asarray(bags))

def main():
    rng=np.random.default_rng(496);x=rng.normal(size=(96,4032));q=rng.normal(size=(1024,4032));record={}
    with threadpool_limits(limits=1):
        torch.set_num_threads(1);start=perf_counter();control=SubspaceHeads(x).score(q);seconds=perf_counter()-start
        record['canonical_seconds']=seconds
        for name,fn in [('numba',distances),('torch_cpu',lambda a,b:torch.cdist(torch.tensor(a,dtype=torch.float64),torch.tensor(b,dtype=torch.float64)).numpy())]:
            start=perf_counter();actual=alternative(x,q,fn);elapsed=perf_counter()-start
            finite=all(np.isfinite(a).all() for a in actual.values())
            close=all(np.allclose(actual[k],control[k],rtol=1e-10,atol=1e-8) for k in control)
            ranking=all(np.array_equal(np.argsort(actual[k],kind='stable'),np.argsort(control[k],kind='stable')) for k in control)
            candidates=all(np.array_equal(np.flatnonzero(actual[k]>np.sort(actual[k][:1000])[::-1][b]),np.flatnonzero(control[k]>np.sort(control[k][:1000])[::-1][b])) for k in control for b in [0,1,5,10])
            record[name]=dict(seconds_including_conversion_and_first_compile=elapsed,speedup=seconds/elapsed,finite=finite,scores_close=close,ranks_equal=ranking,candidates_equal=candidates,max_abs_error=max(float(np.max(abs(actual[k]-control[k]))) for k in control))
    source=P.parent/'exp495_normal_geometry_antigravity/acceleration_antigravity.json';old=json.loads(source.read_text())
    record.update(mlx=old['mlx'],mps=old['mps'],availability=old['availability'],inherited_gpu_probe_sha256=file_sha256_antigravity(source),tensor_eligible=True,
        selected='official sklearn float64 CPU LocalGap',decision='Preserve the exact official LocalGap path used for TRAIN selection. Numba includes first compilation; alternative TorchCPU has only this synthetic parity probe, not all real-data tie cases. GPU float64 unsupported.',source_sha256=file_sha256_antigravity(Path(__file__)))
    write_json_atomic_antigravity(P/'acceleration_results_antigravity.json',record);print(json.dumps(record))

if __name__=='__main__':main()
