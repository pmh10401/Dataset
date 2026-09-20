"""Compare official scalar and batch DTW over the complete new score stage."""
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import configure_inner_threads_antigravity
configure_inner_threads_antigravity()
from pathlib import Path
from time import perf_counter
import json,importlib.metadata
import numpy as np
from aeon.distances import dtw_distance,dtw_pairwise_distance
from scipy.spatial.distance import cdist
from experiments.exp495_normal_geometry_antigravity.diagnostics_antigravity import write_json_atomic_antigravity,file_sha256_antigravity
from .temporal_antigravity import temporal_rows,distance_scores
P=Path(__file__).resolve().parent

def stage(raw,query,features,qfeatures,batch):
    ref=temporal_rows(raw);q=temporal_rows(query)
    if batch:cost=dtw_pairwise_distance(q,ref,window=.1,n_jobs=1)
    else:cost=np.array([[dtw_distance(x,y,window=.1) for y in ref] for x in q])
    return distance_scores(cdist(qfeatures,features),np.sqrt(cost/128))

def main():
    rng=np.random.default_rng(500);raw=rng.normal(size=(64,256));query=rng.normal(size=(1024,256));features=rng.normal(size=(64,128));qfeatures=rng.normal(size=(1024,128));timing={};out={}
    for label,batch in [('official_scalar',False),('official_batch',True)]:
        for phase in ('first_call','warm'):
            start=perf_counter();out[label]=stage(raw,query,features,qfeatures,batch);timing[label+'_'+phase]=perf_counter()-start
    a=out['official_scalar'];b=out['official_batch'];parity=dict(finite=all(np.isfinite(v).all() for v in b.values()),scores_close=all(np.allclose(a[n],b[n],rtol=1e-10,atol=1e-8) for n in a),max_abs_error=max(float(np.max(abs(a[n]-b[n]))) for n in a),ranks_equal=all(np.array_equal(np.argsort(a[n],kind='stable'),np.argsort(b[n],kind='stable')) for n in a),candidates_equal=all(np.array_equal(np.flatnonzero(a[n]>np.sort(a[n][:1000])[::-1][k]),np.flatnonzero(b[n]>np.sort(b[n][:1000])[::-1][k])) for n in a for k in (0,1,5,10)))
    assert all(parity[k] for k in ('finite','scores_close','ranks_equal','candidates_equal'))
    inherited=P.parent/'exp495_normal_geometry_antigravity/acceleration_antigravity.json';gpu=json.loads(inherited.read_text())
    record=dict(timing_seconds=timing,parity=parity,selected='official aeon Numba batch CPU',scope='row conversion/resampling, OAS-space pairwise distance, official DTW, geometric fusion and nearest reduction; unchanged encoder/OAS fit excluded',
        warm_speedup=timing['official_scalar_warm']/timing['official_batch_warm'],first_call_note='Includes JIT cache loading/compilation and conversion; aeon shared kernels may already be cached by contract smoke',
        versions={n:importlib.metadata.version(n) for n in ('aeon','numba','numpy','mlx','mlx-metal','torch')},mlx=gpu['mlx'],mps=gpu['mps'],availability=gpu['availability'],inherited_gpu_probe_sha256=file_sha256_antigravity(inherited),
        gpu_decision='Official dynamic-programming DTW is Numba CPU. No equivalent installed MLX/PyTorch DP implementation; float64 GPU unsupported in inherited same-host probe. No third-party algorithm rewrite.',
        numba_decision='Public official pairwise API already uses Numba; retain supported implementation with exact parity. No custom replacement.',source_sha256=file_sha256_antigravity(Path(__file__)))
    write_json_atomic_antigravity(P/'acceleration_results_antigravity.json',record);print(json.dumps(record))

if __name__=='__main__':main()
