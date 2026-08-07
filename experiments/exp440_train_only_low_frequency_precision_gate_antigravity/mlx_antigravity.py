from __future__ import annotations
import time,numpy as np
from .backend_antigravity import AccelerationEvidenceAntigravity
from experiments.exp439_scale_aware_reconstruction_repair_antigravity.scoring_antigravity import score_channel_cpu_antigravity
def benchmark_mlx_antigravity(query,references,*,repeats=5):
 try:
  import mlx.core as mx,mlx
  q=np.asarray(query,dtype=np.float32);r=np.asarray(references,dtype=np.float32);t=time.perf_counter();cpu=score_channel_cpu_antigravity(q,r,3);cpu_s=(time.perf_counter()-t)/max(1,repeats)
  mq,mr=mx.array(q),mx.array(r);t=time.perf_counter();dist=mx.sqrt(mx.mean((mr-mq)**2,axis=1));out=mx.median(mx.sort(dist)[:3]);mx.eval(out);dev=time.perf_counter()-t;value=float(out.item())
  return AccelerationEvidenceAntigravity('mlx',True,True,abs(value-cpu)<1e-12,abs(value-cpu),dev and cpu_s/dev or 0.0,cpu_s,dev,getattr(mlx,'__version__','installed'),'Apple MLX',None)
 except Exception as e:return AccelerationEvidenceAntigravity.unavailable('mlx',type(e).__name__)
