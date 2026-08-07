from __future__ import annotations
from dataclasses import dataclass
import math
from .config_antigravity import FROZEN_CONFIG,cpu_worker_count_antigravity
@dataclass(frozen=True)
class AccelerationEvidenceAntigravity:
 backend:str;available:bool;synchronized:bool;candidate_parity:bool;max_abs_error:float;speedup:float;cpu_seconds:float;device_seconds:float;version:str|None;device:str|None;fallback_reason:str|None
 @property
 def passed(self):return self.available and self.synchronized and self.candidate_parity and math.isfinite(self.max_abs_error) and self.max_abs_error<=FROZEN_CONFIG.acceleration_tolerance and math.isfinite(self.speedup) and self.speedup>=FROZEN_CONFIG.acceleration_speedup_min and bool(self.version) and bool(self.device)
 @classmethod
 def unavailable(cls,name,reason):return cls(name,False,False,False,float('inf'),0.0,0.0,0.0,None,None,reason)
@dataclass(frozen=True)
class BackendDecisionAntigravity: backend:str;workers:int;evidence:tuple[AccelerationEvidenceAntigravity,...]
def select_backend_antigravity(evidence):
 values=tuple(evidence);passed=[x for x in values if x.passed]
 return BackendDecisionAntigravity(max(passed,key=lambda x:(x.speedup,x.backend)).backend if passed else 'cpu',1 if passed else cpu_worker_count_antigravity(),values)
