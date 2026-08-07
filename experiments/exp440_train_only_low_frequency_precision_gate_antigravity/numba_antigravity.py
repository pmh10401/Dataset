from __future__ import annotations
import time,numpy as np
from .backend_antigravity import AccelerationEvidenceAntigravity
from experiments.exp439_scale_aware_reconstruction_repair_antigravity.scoring_antigravity import score_channel_cpu_antigravity
def benchmark_numba_antigravity(query,references,*,repeats=5):
 try:
  import numba
  q=np.asarray(query,dtype=np.float32);r=np.asarray(references,dtype=np.float32)
  @numba.njit(cache=False)
  def f(q,r):
   d=np.sqrt(np.mean((r-q)**2,axis=1));d.sort();return np.median(d[:3])
  f(q,r);t=time.perf_counter();cpu=score_channel_cpu_antigravity(q,r,3);cs=time.perf_counter()-t;t=time.perf_counter();v=float(f(q,r));ds=time.perf_counter()-t
  return AccelerationEvidenceAntigravity('numba',True,True,abs(v-cpu)<1e-12,abs(v-cpu),cs/ds if ds else 0.0,cs,ds,numba.__version__,'CPU JIT',None)
 except Exception as e:return AccelerationEvidenceAntigravity.unavailable('numba',type(e).__name__)
