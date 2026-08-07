from __future__ import annotations
from hashlib import sha256
import json,sqlite3,subprocess
from pathlib import Path
import numpy as np
from .config_antigravity import FROZEN_CONFIG,cpu_worker_count_antigravity,frozen_config_payload_antigravity
from .inputs_antigravity import open_exp439_input_session
from .mlx_antigravity import benchmark_mlx_antigravity
from .numba_antigravity import benchmark_numba_antigravity
from .predecessor_antigravity import EXP439_PUBLICATION_ANTIGRAVITY,load_frozen_hard_coordinates_antigravity
def validate_clean_head_antigravity(head,expected,status):
 if head!=expected or len(head)!=40 or status.strip():raise ValueError('readiness requires the exact clean reviewed commit')
 return head

def run_readiness_antigravity(reviewed_commit:str):
 head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();status=subprocess.check_output(['git','status','--porcelain'],text=True);validate_clean_head_antigravity(head,reviewed_commit,status)
 parent=Path(FROZEN_CONFIG.run_parent)
 if parent.exists() or any(Path(x).exists() for x in (FROZEN_CONFIG.output_path,FROZEN_CONFIG.checkpoint_path,FROZEN_CONFIG.publication_path,FROZEN_CONFIG.evaluation_path)):raise ValueError('Exp440 fixed roots must be absent')
 database=Path(FROZEN_CONFIG.database_path);db_sha=sha256(database.read_bytes()).hexdigest()
 if db_sha!=FROZEN_CONFIG.database_sha256:raise ValueError('database SHA256 differs')
 connection=sqlite3.connect(f'file:{database}?mode=ro',uri=True)
 try:
  if connection.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('database quick_check failed')
 finally:connection.close()
 prediction=EXP439_PUBLICATION_ANTIGRAVITY/'prediction_antigravity.jsonl'
 if sha256(prediction.read_bytes()).hexdigest()!=FROZEN_CONFIG.exp439_prediction_sha256:raise ValueError('Exp439 prediction SHA256 differs')
 hard=load_frozen_hard_coordinates_antigravity()
 with open_exp439_input_session(verified_code_commit=reviewed_commit) as session:datasets=tuple(session.context.datasets);label_access=session.label_access_count
 if len(datasets)!=FROZEN_CONFIG.dataset_count or label_access!=0:raise ValueError('label-free population gate failed')
 q=np.arange(256,dtype=np.float32);r=np.stack([q+i for i in range(16)]);evidence=(benchmark_mlx_antigravity(q,r),benchmark_numba_antigravity(q,r))
 payload={'schema':'exp440.readiness.v1','reviewed_commit':reviewed_commit,'run_identity':FROZEN_CONFIG.run_identity,'database_sha256':db_sha,'exp439_prediction_sha256':FROZEN_CONFIG.exp439_prediction_sha256,'hard_prediction_sha256':FROZEN_CONFIG.hard_prediction_sha256,'hard_coordinate_count':len(hard),'dataset_count':len(datasets),'label_access_count':0,'backend':'cpu','workers':cpu_worker_count_antigravity(),'acceleration_evidence':[x.to_payload() for x in evidence],'acceleration_decision':'cpu_until_end_to_end_candidate_parity_and_speed_gate','config':frozen_config_payload_antigravity()}
 parent.mkdir(mode=0o700);Path(FROZEN_CONFIG.readiness_path).write_text(json.dumps(payload,sort_keys=True,separators=(',',':'))+'\n');return payload
