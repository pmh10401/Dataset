from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from dataclasses import dataclass
import json,os,signal
from pathlib import Path
from .budget_antigravity import select_global_candidates_antigravity
from .checkpoint_antigravity import Exp440CheckpointStoreAntigravity
from .config_antigravity import FROZEN_CONFIG,cpu_worker_count_antigravity
from .contracts_antigravity import Exp440DatasetResultAntigravity
from .features_antigravity import fit_group_calibration_antigravity
from .inputs_antigravity import Exp439RawDataset,open_exp439_input_session,partition_train_rows_antigravity,test_row_abstention_reason_antigravity
from .predecessor_antigravity import load_frozen_hard_coordinates_antigravity
from .progress_antigravity import ProgressWriterAntigravity
from .scoring_antigravity import score_test_row_antigravity
def ordered_parallel_map_antigravity(values,function,*,workers):
 if workers!=cpu_worker_count_antigravity():raise ValueError('official CPU workers are fixed at max-minus-one')
 with ThreadPoolExecutor(max_workers=workers) as pool:return tuple(pool.map(function,values))

@dataclass(frozen=True)
class CpuDatasetTaskAntigravity:
 dataset:str;train:Exp439RawDataset;test:Exp439RawDataset

def _worker_init_antigravity():
 for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'

def score_dataset_task_antigravity(task:CpuDatasetTaskAntigravity)->Exp440DatasetResultAntigravity:
 prepared=partition_train_rows_antigravity(task.train);models={}
 for group in prepared.groups:
  try:fitted=fit_group_calibration_antigravity(group.matrix)
  except ValueError:continue
  if fitted.eligible:models[group.raw_length]=fitted
 candidates=[];abstentions=[]
 for row in task.test.series:
  if test_row_abstention_reason_antigravity(row) is not None or row.raw_length not in models:abstentions.append(row.instance_index);continue
  _,candidate=score_test_row_antigravity(task.dataset,row.instance_index,row.values,models[row.raw_length])
  if candidate is not None:candidates.append(candidate)
 return Exp440DatasetResultAntigravity(task.dataset,'scored',tuple(candidates),tuple(abstentions),len(prepared.exclusions),len(task.test.series))

def _run_antigravity(*,reviewed_commit:str,resume:bool)->tuple[int,int]:
 hard=load_frozen_hard_coordinates_antigravity();checkpoint=Path(FROZEN_CONFIG.checkpoint_path);output=Path(FROZEN_CONFIG.output_path)
 with open_exp439_input_session(verified_code_commit=reviewed_commit) as session:
  datasets=tuple(session.context.datasets)
  if len(datasets)!=FROZEN_CONFIG.dataset_count:raise ValueError('dataset population differs')
  store=Exp440CheckpointStoreAntigravity.open_read_only(checkpoint) if resume else Exp440CheckpointStoreAntigravity.create(checkpoint,datasets,reviewed_commit)
  writer=ProgressWriterAntigravity.resume(output) if resume else ProgressWriterAntigravity.create(output,total=len(datasets),checkpoint=checkpoint,backend='cpu')
  def interrupted(_signum,_frame):raise KeyboardInterrupt
  previous=signal.signal(signal.SIGTERM,interrupted)
  pending=[];capacity=cpu_worker_count_antigravity()**2
  try:
   with ProcessPoolExecutor(max_workers=cpu_worker_count_antigravity(),initializer=_worker_init_antigravity) as pool:
    for sequence,dataset in enumerate(datasets[store.completed:],start=store.completed):
     train=session.load_train(dataset);test=session.load_test_values(dataset,train);writer.start_dataset(dataset,len(train.series),len(test.series),tuple(sorted({x.raw_length for x in train.series})),tuple(sorted({x.raw_length for x in test.series})),'queued')
     pending.append((sequence,dataset,pool.submit(score_dataset_task_antigravity,CpuDatasetTaskAntigravity(dataset,train,test))))
     if len(pending)>=capacity:
      seq,name,future=pending.pop(0);result=future.result();store.commit(seq,result);writer.complete_dataset(name,result.status,provisional=len(result.provisional_candidates))
    for seq,name,future in pending:
     result=future.result();store.commit(seq,result);writer.complete_dataset(name,result.status,provisional=len(result.provisional_candidates))
   all_candidates=tuple(candidate for result in store.results() for candidate in result.provisional_candidates);retained=select_global_candidates_antigravity(all_candidates,hard);target=checkpoint/'retained_candidates_antigravity.json'
   raw=(json.dumps([x.to_payload() for x in retained],sort_keys=True,separators=(',',':'))+'\n').encode();fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
   with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
   writer.p['retained_candidates']=len(retained);writer.close('checkpoint_complete');return store.completed,len(retained)
  except KeyboardInterrupt:
   writer.close('interrupted');raise
  except BaseException as error:
   writer.fail(error);raise
  finally:signal.signal(signal.SIGTERM,previous)

def run_exp440_new_antigravity(*,reviewed_commit:str)->tuple[int,int]:return _run_antigravity(reviewed_commit=reviewed_commit,resume=False)
def resume_exp440_antigravity(*,reviewed_commit:str)->tuple[int,int]:return _run_antigravity(reviewed_commit=reviewed_commit,resume=True)
