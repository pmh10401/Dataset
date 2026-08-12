"""Adversarial v3 authority tests; builders live only in this test module."""
from __future__ import annotations
from hashlib import sha256
import json,sqlite3,sys,tempfile,unittest
from pathlib import Path
import numpy as np
from paper.equipment_anomaly_detection.artifacts_paper import canonical_json_bytes,write_create_only
from paper.equipment_anomaly_detection.catalog_paper import build_catalog,write_acceptance,write_catalog

def _blob(a):return np.asarray(a,dtype='<f4').tobytes()
def _db(path:Path):
 c=sqlite3.connect(path);c.executescript('CREATE TABLE datasets(id INTEGER PRIMARY KEY,name TEXT UNIQUE,series_length INTEGER); CREATE TABLE instances(id INTEGER PRIMARY KEY,dataset_id INTEGER,split TEXT,instance_index INTEGER,label TEXT,values_blob BLOB);')
 c.executemany('INSERT INTO datasets VALUES(?,?,?)',[(1,'alpha',2),(2,'beta',2),(3,'gamma',2),(4,'delta',2)])
 rows=[];ident=1
 for d in range(1,5):
  for split,label,count in [('TRAIN','0',2),('TEST','1',2)]:
   for i in range(count):rows.append((ident,d,split,i,label,_blob((d+i,d+i+1))));ident+=1
 c.executemany('INSERT INTO instances VALUES(?,?,?,?,?,?)',rows);c.commit();c.close()

class ArtifactRunnerTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name);self.db=self.base/'data.sqlite';_db(self.db);self.catalog=self.base/'catalog_paper.json';write_catalog(self.catalog,self.db,datasets=('alpha','beta','delta','gamma'));cat=build_catalog(self.db,datasets=('alpha','beta','delta','gamma'));self.accept=self.base/'acceptance_paper.json';write_acceptance(self.accept,cat,((('alpha',0),),(('beta',0),),(('gamma',0),),(('delta',0),)));self.root=self.base/'run'
 def tearDown(self):self.tmp.cleanup()
 def config(self):
  from paper.equipment_anomaly_detection.runner_paper import PipelineConfig
  return PipelineConfig(self.db,self.root,'smoke',self.catalog,self.accept)
 def _builders(self):
  import paper.equipment_anomaly_detection.runner_paper as r
  def builder(cfg,spec,context,prior):write_create_only(r.stage_file(cfg,spec),canonical_json_bytes(r._stage_payload(spec,context,prior)))
  return {s:builder for s in r.PIPELINE_STAGE_ORDER}
 def test_red_generic_completion_is_not_an_api_and_all_stages_need_closed_builders(self):
  """Arbitrary returned values cannot complete stages or choose verifier IDs."""
  from paper.equipment_anomaly_detection.runner_paper import PIPELINE_STAGE_ORDER,PipelineStateError,run_pipeline
  with self.assertRaises(Exception):run_pipeline(self.config(),test_stage_builders={'preflight':lambda *a:'generic'})
  self.root=self.base/'full';records=run_pipeline(self.config(),test_stage_builders=self._builders());self.assertEqual([r['stage'] for r in records[1:]],list(PIPELINE_STAGE_ORDER))
 def test_red_context_recompute_rejects_modified_database_and_legacy_root(self):
  """Verify must have config and rehash current database/catalog context."""
  from paper.equipment_anomaly_detection.runner_paper import PipelineStateError,run_pipeline,verify_completed
  with self.assertRaises(Exception):run_pipeline(self.config())
  c=sqlite3.connect(self.db);c.execute("UPDATE instances SET values_blob=? WHERE id=1",(_blob((99,99)),));c.commit();c.close()
  with self.assertRaisesRegex(PipelineStateError,'context|catalog'):verify_completed(self.config())
  legacy=self.base/'legacy';legacy.mkdir();(legacy/'pipeline_state_paper.json').write_text('{}')
  self.root=legacy
  with self.assertRaisesRegex(PipelineStateError,'legacy_root_requires_new_root'):run_pipeline(self.config())
 def test_red_stage_artifact_tamper_symlink_and_namespace_pollution_are_rejected(self):
  """A hash-valid journal cannot hide altered artifact bytes or internal pollution."""
  from paper.equipment_anomaly_detection.runner_paper import PipelineStateError,run_pipeline,verify_completed
  with self.assertRaises(Exception):run_pipeline(self.config())
  artifact=self.root/'stage_artifacts_paper/preflight/preflight_paper.json';artifact.write_text('{}\n')
  with self.assertRaisesRegex(PipelineStateError,'semantic|artifact'):verify_completed(self.config())
  self.root=self.base/'polluted'
  with self.assertRaises(Exception):run_pipeline(self.config())
  (self.root/'stage_artifacts_paper/preflight/junk_paper.json').write_text('{}')
  with self.assertRaisesRegex(PipelineStateError,'pollution'):verify_completed(self.config())
  link=self.base/'link';link.symlink_to(self.root,target_is_directory=True)
  self.root=link
  with self.assertRaisesRegex(ValueError,'symlink'):verify_completed(self.config())
 def test_catalog_and_acceptance_are_nonempty_external_authority(self):
  """Smoke is catalog-derived; fake digest inventory and malformed full are rejected."""
  from paper.equipment_anomaly_detection.catalog_paper import load_catalog
  self.assertEqual(len(load_catalog(self.catalog,self.db,'smoke')['datasets']),4)
  with self.assertRaisesRegex(ValueError,'full catalog'):load_catalog(self.catalog,self.db,'full')
 def test_exact_scientific_shard_rejects_unknown_sources_hard_stages_backend_and_label_aliases(self):
  """A length-correct shard must still obey Task 4–7 names, lanes, and pending backend contract."""
  from paper.equipment_anomaly_detection.artifacts_paper import EXPECTED_HARD_STAGES,EXPECTED_SOURCES,validate_proposal_shard
  source={"score":[.1,.2],"pvalue":[.9,.8],"strength":[.1,.2],"native_pass":[False,True]}
  lanes={name:{"hard":[],"standard":[],"priority":[]} for name in EXPECTED_HARD_STAGES}
  shard={"schema":"paper-proposal-shard.v3","dataset":"alpha","train_identity":{"indices":[0,1],"digest":"a"*64},"test_identity":{"indices":[0,1],"digest":"b"*64},"sources":{name:dict(source) for name in EXPECTED_SOURCES},"hard_route":{"stages":lanes},"low_frequency":{"rows":[{"index":0,"evidence_missing":False,"strict_pass":False,"depths":[]},{"index":1,"evidence_missing":True,"strict_pass":False,"depths":[]}]},"stability":{"selected":[],"stable":[],"unstable":[],"views":[]},"policies":{name:{"alerts":[],"scores":[.1,.2]} for name in ("hard_only","hard_plus_source","hard_plus_stable","hard_plus_unstable")},"backend_decision":{"selected_backend":"cpu","reason":"task9_pending","verdict":"cpu_fallback"}}
  validate_proposal_shard(shard)
  shard["sources"]={"unknown":source}
  with self.assertRaisesRegex(ValueError,'source'):validate_proposal_shard(shard)
  shard["sources"]={name:dict(source) for name in EXPECTED_SOURCES};shard["backend_decision"]["reason"]="claimed_gpu"
  with self.assertRaisesRegex(ValueError,'backend'):validate_proposal_shard(shard)
  shard["backend_decision"]["reason"]="task9_pending";shard["low_frequency"]["true-label"]=1
  with self.assertRaisesRegex(ValueError,'forbidden'):validate_proposal_shard(shard)
 def test_no_labels_or_oracle_before_future_parity(self):
  """Production stops at prepare and does not expose label capability."""
  from paper.equipment_anomaly_detection.runner_paper import run_pipeline
  sys.modules.pop('paper.equipment_anomaly_detection.label_access_paper',None)
  with self.assertRaises(Exception):run_pipeline(self.config())
  self.assertNotIn('paper.equipment_anomaly_detection.label_access_paper',sys.modules)
if __name__=='__main__':unittest.main()
