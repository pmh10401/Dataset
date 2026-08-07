from __future__ import annotations
from datetime import datetime,timezone
import json,os,time
from pathlib import Path
def _now():return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00','Z')
class ProgressWriterAntigravity:
 @classmethod
 def create(cls,root,*,total,checkpoint,backend):
  root=Path(root);root.mkdir(parents=True,exist_ok=False);return cls(root,total,checkpoint,backend)
 def __init__(self,root,total,checkpoint,backend):
  self.root=Path(root);self.started=time.monotonic();self.p={'pid':os.getpid(),'current_dataset':None,'train_count':None,'test_count':None,'train_lengths':[],'test_lengths':[],'stage':'created','backend':backend,'completed':0,'total':total,'imported':0,'recomputed':0,'success':0,'abstention':0,'error':0,'provisional_candidates':0,'retained_candidates':0,'throughput':0.0,'elapsed_seconds':0.0,'eta_seconds':None,'checkpoint':str(checkpoint),'heartbeat':_now(),'latest_error':None};(self.root/'run_antigravity.pid').write_text(json.dumps({'pid':os.getpid(),'active':True})+'\n');self._emit('created')
 def _emit(self,event):
  elapsed=max(0,time.monotonic()-self.started);self.p['elapsed_seconds']=elapsed;self.p['throughput']=self.p['completed']/elapsed if elapsed else 0.0;remaining=self.p['total']-self.p['completed'];self.p['eta_seconds']=remaining/self.p['throughput'] if self.p['throughput'] else None;self.p['heartbeat']=_now();raw=(json.dumps(self.p,sort_keys=True,separators=(',',':'))+'\n');tmp=self.root/'progress_tmp_antigravity.json';tmp.write_text(raw);os.replace(tmp,self.root/'progress_antigravity.json');row=dict(self.p,event=event)
  with (self.root/'events_antigravity.jsonl').open('a') as stream:stream.write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n')
  with (self.root/'parent_antigravity.log').open('a') as stream:stream.write(f"{self.p['heartbeat']} {event} dataset={self.p['current_dataset']} completed={self.p['completed']}/{self.p['total']}\n")
 def start_dataset(self,dataset,train_count,test_count,train_lengths,test_lengths,stage):self.p.update(current_dataset=dataset,train_count=train_count,test_count=test_count,train_lengths=list(train_lengths),test_lengths=list(test_lengths),stage=stage);self._emit('dataset_start')
 def complete_dataset(self,dataset,status,*,provisional=0):self.p['completed']+=1;self.p['success']+=status=='scored';self.p['abstention']+=status=='abstained';self.p['provisional_candidates']+=provisional;self.p['stage']='checkpoint';self._emit('dataset_complete')
 def close(self,state):self.p.update(stage=state,current_dataset=None);self._emit(state);(self.root/'run_antigravity.pid').write_text(json.dumps({'pid':os.getpid(),'active':False})+'\n')
