from __future__ import annotations
import time,numpy as np
from .backend_antigravity import AccelerationEvidenceAntigravity
from experiments.exp439_scale_aware_reconstruction_repair_antigravity.scoring_antigravity import score_channel_cpu_antigravity
def benchmark_mlx_antigravity(query,references,*,repeats=5):
 try:
  import mlx.core as mx,mlx
  q=np.asarray(query,dtype=np.float32);r=np.asarray(references,dtype=np.float32)
  def device():
   mq,mr=mx.array(q),mx.array(r);dist=mx.sqrt(mx.mean((mr-mq)**2,axis=1));out=mx.sort(dist)[1];mx.eval(out);return float(out.item())
  device();device();cpu_times=[];device_times=[];cpu=None;value=None
  for _ in range(max(1,repeats)):
   t=time.perf_counter();cpu=score_channel_cpu_antigravity(q,r,3);cpu_times.append(time.perf_counter()-t);t=time.perf_counter();value=device();device_times.append(time.perf_counter()-t)
  cpu_s=float(np.median(cpu_times));dev=float(np.median(device_times));error=abs(float(value)-float(cpu))
  return AccelerationEvidenceAntigravity('mlx',True,True,False,error,dev and cpu_s/dev or 0.0,cpu_s,dev,getattr(mlx,'__version__','installed'),'Apple MLX',"end_to_end_candidate_parity_not_run")
 except Exception as e:return AccelerationEvidenceAntigravity.unavailable('mlx',type(e).__name__)
