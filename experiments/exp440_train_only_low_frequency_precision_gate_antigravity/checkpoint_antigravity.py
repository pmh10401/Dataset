"""Small ordered, hash-bound, create-only Exp440 checkpoint."""
from __future__ import annotations
from hashlib import sha256
import json,os
from pathlib import Path
from .config_antigravity import FROZEN_CONFIG
from .contracts_antigravity import Exp440DatasetResultAntigravity

def _bytes(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()
def _atomic(path,raw):
 tmp=path.with_name(path.stem+'_tmp_antigravity'+path.suffix);fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
 try:
  with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if tmp.exists():tmp.unlink()
class Exp440CheckpointStoreAntigravity:
 def __init__(self,root,datasets,commit,records):self.root=Path(root);self.datasets=tuple(datasets);self.reviewed_commit=commit;self.records=dict(records)
 @property
 def completed(self):return len(self.records)
 @classmethod
 def create(cls,root,datasets,reviewed_commit):
  root=Path(root);names=tuple(datasets)
  if root.exists():raise FileExistsError(root)
  if names!=tuple(sorted(set(names))) or not names:raise ValueError('datasets must be sorted unique')
  root.mkdir(mode=0o700);(root/'records_antigravity').mkdir(mode=0o700)
  context={'schema':'exp440.checkpoint.v1','run_identity':FROZEN_CONFIG.run_identity,'datasets':list(names),'reviewed_commit':reviewed_commit}
  (root/'context_antigravity.json').write_bytes(_bytes(context));_atomic(root/'state_antigravity.json',_bytes({'completed':0,'records':{}}));return cls(root,names,reviewed_commit,{})
 @classmethod
 def open_read_only(cls,root):
  root=Path(root)
  try:context=json.loads((root/'context_antigravity.json').read_bytes());state=json.loads((root/'state_antigravity.json').read_bytes())
  except Exception as e:raise ValueError('checkpoint is invalid') from e
  if context.get('run_identity')!=FROZEN_CONFIG.run_identity:raise ValueError('checkpoint identity differs')
  store=cls(root,tuple(context['datasets']),context['reviewed_commit'],state['records'])
  if state.get('completed')!=len(store.records) or tuple(store.records)!=store.datasets[:store.completed]:raise ValueError('checkpoint prefix is invalid')
  for dataset,digest in store.records.items():
   path=root/'records_antigravity'/f'{store.datasets.index(dataset):04d}_record_antigravity.json'
   if not path.is_file() or sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('checkpoint record differs')
   Exp440DatasetResultAntigravity.from_payload(json.loads(path.read_bytes()))
  return store
 def commit(self,sequence,result):
  if sequence!=self.completed or sequence>=len(self.datasets) or result.dataset!=self.datasets[sequence]:raise ValueError('checkpoint commit is out of order')
  raw=_bytes(result.to_payload());path=self.root/'records_antigravity'/f'{sequence:04d}_record_antigravity.json'
  fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
  with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
  self.records[result.dataset]=sha256(raw).hexdigest();_atomic(self.root/'state_antigravity.json',_bytes({'completed':self.completed,'records':self.records}))
 def results(self):
  for i,dataset in enumerate(self.datasets[:self.completed]):yield Exp440DatasetResultAntigravity.from_payload(json.loads((self.root/'records_antigravity'/f'{i:04d}_record_antigravity.json').read_bytes()))
