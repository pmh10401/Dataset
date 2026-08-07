from __future__ import annotations
import time,numpy as np
from .backend_antigravity import AccelerationEvidenceAntigravity
from experiments.exp439_scale_aware_reconstruction_repair_antigravity.scoring_antigravity import score_channel_cpu_antigravity
try:
 import numba as _numba_antigravity
 @_numba_antigravity.njit(cache=True)
 def _score_numba_kernel_antigravity(q,r):
  d=np.empty(r.shape[0],dtype=np.float64)
  for i in range(r.shape[0]):
   total=0.0
   for j in range(r.shape[1]):
    delta=float(r[i,j])-float(q[j]);total+=delta*delta
   d[i]=np.sqrt(total/r.shape[1])
  d.sort();return d[1]
except Exception:_score_numba_kernel_antigravity=None
def score_numba_antigravity(query,references,k):
 q=np.ascontiguousarray(query,dtype=np.float32);r=np.ascontiguousarray(references,dtype=np.float32)
 if k!=3 or q.ndim!=1 or r.ndim!=2 or r.shape[0]<3 or r.shape[1]!=q.size or not np.isfinite(q).all() or not np.isfinite(r).all() or _score_numba_kernel_antigravity is None:raise ValueError('Numba score input is invalid')
 value=float(_score_numba_kernel_antigravity(q,r))
 if not np.isfinite(value):raise ValueError('Numba score is nonfinite')
 return value
def benchmark_numba_antigravity(query,references,*,repeats=5):
 try:
  import numba
  q=np.asarray(query,dtype=np.float32);r=np.asarray(references,dtype=np.float32)
  f=_score_numba_kernel_antigravity
  f(q,r);f(q,r);cpu_times=[];device_times=[];cpu=None;v=None
  for _ in range(max(1,repeats)):
   t=time.perf_counter();cpu=score_channel_cpu_antigravity(q,r,3);cpu_times.append(time.perf_counter()-t);t=time.perf_counter();v=float(f(q,r));device_times.append(time.perf_counter()-t)
  cs=float(np.median(cpu_times));ds=float(np.median(device_times));error=abs(float(v)-float(cpu))
  return AccelerationEvidenceAntigravity('numba',True,True,False,error,cs/ds if ds else 0.0,cs,ds,numba.__version__,'CPU JIT',"end_to_end_candidate_parity_not_run")
 except Exception as e:return AccelerationEvidenceAntigravity.unavailable('numba',type(e).__name__)
