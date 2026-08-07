from __future__ import annotations
from hashlib import sha256
import json,sqlite3,subprocess,time
from pathlib import Path
import numpy as np
from .config_antigravity import FROZEN_CONFIG,cpu_worker_count_antigravity,frozen_config_payload_antigravity
from .inputs_antigravity import open_exp439_input_session
from .mlx_antigravity import benchmark_mlx_antigravity
from .numba_antigravity import benchmark_numba_antigravity
from .backend_antigravity import AccelerationEvidenceAntigravity
from .predecessor_antigravity import EXP439_PUBLICATION_ANTIGRAVITY,load_frozen_hard_coordinates_antigravity
def validate_clean_head_antigravity(head,expected,status):
 if head!=expected or len(head)!=40 or status.strip():raise ValueError('readiness requires the exact clean reviewed commit')
 return head
def _file_sha256_antigravity(path):
 digest=sha256()
 with Path(path).open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
 return digest.hexdigest()
def validate_readiness_for_run_antigravity(reviewed_commit):
 path=Path(FROZEN_CONFIG.readiness_path)
 if not path.is_file():raise ValueError('Exp440 readiness is absent')
 payload=json.loads(path.read_text());head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();status=subprocess.check_output(['git','status','--porcelain'],text=True);validate_clean_head_antigravity(head,reviewed_commit,status)
 if payload.get('reviewed_commit')!=reviewed_commit or payload.get('run_identity')!=FROZEN_CONFIG.run_identity or payload.get('dataset_count')!=FROZEN_CONFIG.dataset_count or payload.get('label_access_count')!=0 or payload.get('backend') not in {'cpu','numba'} or payload.get('workers')!=cpu_worker_count_antigravity():raise ValueError('Exp440 readiness binding differs')
 return payload

def run_readiness_antigravity(reviewed_commit:str):
 head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();status=subprocess.check_output(['git','status','--porcelain'],text=True);validate_clean_head_antigravity(head,reviewed_commit,status)
 parent=Path(FROZEN_CONFIG.run_parent)
 if parent.exists() or any(Path(x).exists() for x in (FROZEN_CONFIG.output_path,FROZEN_CONFIG.checkpoint_path,FROZEN_CONFIG.publication_path,FROZEN_CONFIG.evaluation_path)):raise ValueError('Exp440 fixed roots must be absent')
 database=Path(FROZEN_CONFIG.database_path);db_sha=_file_sha256_antigravity(database)
 if db_sha!=FROZEN_CONFIG.database_sha256:raise ValueError('database SHA256 differs')
 connection=sqlite3.connect(f'file:{database}?mode=ro',uri=True)
 try:
  if connection.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('database quick_check failed')
 finally:connection.close()
 prediction=EXP439_PUBLICATION_ANTIGRAVITY/'prediction_antigravity.jsonl'
 if sha256(prediction.read_bytes()).hexdigest()!=FROZEN_CONFIG.exp439_prediction_sha256:raise ValueError('Exp439 prediction SHA256 differs')
 hard=load_frozen_hard_coordinates_antigravity()
 with open_exp439_input_session(verified_code_commit=reviewed_commit) as session:
  datasets=tuple(session.context.datasets);label_access=session.label_access_count
  from .runtime_antigravity import CpuDatasetTaskAntigravity,score_dataset_task_antigravity
  train=session.load_train('CBF_normal_3');test=session.load_test_values('CBF_normal_3',train);task=CpuDatasetTaskAntigravity('CBF_normal_3',train,test,'cpu');start=time.perf_counter();cpu_result=score_dataset_task_antigravity(task);cpu_time=time.perf_counter()-start
  task=CpuDatasetTaskAntigravity('CBF_normal_3',train,test,'numba');start=time.perf_counter();numba_result=score_dataset_task_antigravity(task);numba_time=time.perf_counter()-start
 if len(datasets)!=FROZEN_CONFIG.dataset_count or label_access!=0:raise ValueError('label-free population gate failed')
 q=np.arange(256,dtype=np.float32);r=np.stack([q+i for i in range(16)]);mlx_evidence=benchmark_mlx_antigravity(q,r);primitive_numba=benchmark_numba_antigravity(q,r);cpu_candidates=cpu_result.provisional_candidates;numba_candidates=numba_result.provisional_candidates;parity=tuple(x.coordinate for x in cpu_candidates)==tuple(x.coordinate for x in numba_candidates);differences=[]
 if parity:
  for left,right in zip(cpu_candidates,numba_candidates):
   for a,b in zip(left.depth_scores,right.depth_scores):differences.extend((abs(a.score-b.score),abs(a.threshold-b.threshold),abs(a.excess_ratio-b.excess_ratio),abs(a.empirical_rank-b.empirical_rank)))
 maximum=max(differences,default=0.0);speedup=cpu_time/numba_time if numba_time else 0.0;numba_evidence=AccelerationEvidenceAntigravity('numba',True,True,parity,maximum,speedup,cpu_time,numba_time,primitive_numba.version,'CPU JIT',None if parity and maximum<=FROZEN_CONFIG.acceleration_tolerance and speedup>=FROZEN_CONFIG.acceleration_speedup_min else 'end_to_end_gate_failed');evidence=(mlx_evidence,numba_evidence);backend='numba' if numba_evidence.passed else 'cpu'
 payload={'schema':'exp440.readiness.v1','reviewed_commit':reviewed_commit,'run_identity':FROZEN_CONFIG.run_identity,'database_sha256':db_sha,'exp439_prediction_sha256':FROZEN_CONFIG.exp439_prediction_sha256,'hard_prediction_sha256':FROZEN_CONFIG.hard_prediction_sha256,'hard_coordinate_count':len(hard),'dataset_count':len(datasets),'label_access_count':0,'backend':backend,'workers':cpu_worker_count_antigravity(),'acceleration_evidence':[x.to_payload() for x in evidence],'acceleration_decision':backend,'acceleration_fixture':'CBF_normal_3','config':frozen_config_payload_antigravity()}
 parent.mkdir(mode=0o700);Path(FROZEN_CONFIG.readiness_path).write_text(json.dumps(payload,sort_keys=True,separators=(',',':'))+'\n');return payload
