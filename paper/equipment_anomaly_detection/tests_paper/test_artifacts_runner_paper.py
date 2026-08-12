"""Adversarial Task 8 lifecycle, typed shard, and parity tests."""
from __future__ import annotations

from hashlib import sha256
import json, sqlite3, tempfile, unittest
from pathlib import Path

import numpy as np


def _blob(values: tuple[float, ...]) -> bytes: return np.asarray(values, dtype="<f4").tobytes()
def _canonical(value: object) -> bytes: return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()+b"\n"
def _database(path: Path) -> None:
    connection=sqlite3.connect(path); connection.executescript("CREATE TABLE datasets(id INTEGER PRIMARY KEY,name TEXT UNIQUE,series_length INTEGER);CREATE TABLE instances(id INTEGER PRIMARY KEY,dataset_id INTEGER,split TEXT,instance_index INTEGER,label TEXT,values_blob BLOB);")
    connection.executemany("INSERT INTO datasets VALUES(?,?,?)",((1,"alpha",2),(2,"beta",2))); rows=[]; identity=1
    for dataset in (1,2):
        for split,label in (("TRAIN","0"),("TEST","1")):
            for index in range(2): rows.append((identity,dataset,split,index,label,_blob((dataset+index,dataset+index+.5)))); identity+=1
        rows.append((identity,dataset,"TRAIN",9,"1",_blob((9,9)))); identity+=1
    connection.executemany("INSERT INTO instances VALUES(?,?,?,?,?,?)",rows); connection.commit(); connection.close()


class Task8LifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary=tempfile.TemporaryDirectory(); self.base=Path(self.temporary.name).resolve(); self.db=self.base/"fixture.sqlite"; _database(self.db)
        from paper.equipment_anomaly_detection.catalog_paper import build_catalog
        self.catalog=build_catalog(self.db,mode="smoke",smoke_datasets=("alpha","beta")); self.oracle=self.base/"oracle_antigravity.json"; self.root=self.base/"run_antigravity"
        policies=((),(),(),()); self.oracle.write_bytes(_canonical({"schema":"paper-task8-coordinate-oracle.v1","policies":[{"policy_id":name,"coordinates":[]} for name in ("hard_only","hard_plus_source","hard_plus_stable","hard_plus_unstable")]})); self.acceptance=self.base/"acceptance_antigravity.json"; self._acceptance(policies)
    def tearDown(self): self.temporary.cleanup()
    def _acceptance(self, policies):
        from paper.equipment_anomaly_detection.catalog_paper import coordinate_digest
        records=[{"policy_id":name,"order":order,"coordinates":[[dataset,index] for dataset,index in coords],"coordinate_count":len(coords),"coordinate_sha256":coordinate_digest(coords)} for order,(name,coords) in enumerate(zip(("hard_only","hard_plus_source","hard_plus_stable","hard_plus_unstable"),policies,strict=True))]
        bare={"schema":"paper-task8-acceptance.v1","catalog_sha256":self.catalog["catalog_sha256"],"oracle":{"identifier":"fixture-oracle","version":"v1","relative_path":self.oracle.name,"sha256":sha256(self.oracle.read_bytes()).hexdigest()},"policies":records}; self.acceptance.write_bytes(_canonical({**bare,"acceptance_sha256":sha256(_canonical(bare)).hexdigest()}))
    def config(self):
        from paper.equipment_anomaly_detection.runner_paper import PipelineConfig
        return PipelineConfig(self.db,self.root,"smoke",("alpha","beta"),self.acceptance)
    def _typed(self, dataset="alpha"):
        from paper.equipment_anomaly_detection.contracts_paper import FrozenTask8BackendDecision, LowFrequencyResult, LowFrequencyRow, RawDataset
        from paper.equipment_anomaly_detection.hard_route_paper import HardRouteResult, RouteLanes, SourceExpansion
        from paper.equipment_anomaly_detection.lineage_paper import LINEAGE_NODES
        from paper.equipment_anomaly_detection.routing_paper import PolicyBundle, PolicyRoute
        from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import EXPECTED_HARD_SOURCES, HardSourceBundle, HardSourceResult
        from paper.equipment_anomaly_detection.sources_paper.stability_paper import StabilityPartition
        entry=next(item for item in self.catalog["datasets"] if item["dataset"]==dataset); dataset_id=entry["dataset_id"]
        train_blobs=(_blob((dataset_id,dataset_id+.5)),_blob((dataset_id+1,dataset_id+1.5)))
        rows=(np.array((dataset_id,dataset_id+.5),dtype="<f4"),np.array((dataset_id+1,dataset_id+1.5),dtype="<f4"))
        raw=RawDataset(dataset,2,(0,1),train_blobs,rows,(0,1),train_blobs,rows)
        sources=HardSourceBundle(EXPECTED_HARD_SOURCES,{name:HardSourceResult(name,LINEAGE_NODES[0],np.array((.1,.2)),np.array((.1,.2)),np.array((.9,.8)),.03,np.array((False,False)),True,"eligible") for name in EXPECTED_HARD_SOURCES})
        lanes=RouteLanes(frozenset(),frozenset(),frozenset()); route=HardRouteResult(lanes,frozenset(),frozenset(),frozenset(),lanes,SourceExpansion(frozenset(),frozenset()),lanes,frozenset(),lanes,frozenset(),frozenset(),2)
        low=LowFrequencyResult((dataset,),(LowFrequencyRow(dataset,0,2,(),False,None,None,False,True,"no_group"),LowFrequencyRow(dataset,1,2,(),False,None,None,False,True,"no_group")),(),(),0,{dataset:frozenset()})
        stability=StabilityPartition((),(),(),{}); policies=PolicyBundle(tuple(PolicyRoute(name,{dataset:frozenset()},{dataset:np.zeros(2)},{dataset:np.zeros(2)}) for name in ("hard_only","hard_plus_source","hard_plus_stable","hard_plus_unstable")))
        return raw,sources,route,low,stability,policies,FrozenTask8BackendDecision()
    def _freeze(self):
        from paper.equipment_anomaly_detection.artifacts_paper import freeze_score_root, write_typed_proposal_shard
        from paper.equipment_anomaly_detection.runner_paper import PipelineNotImplementedError, run_pipeline
        with self.assertRaises(PipelineNotImplementedError): run_pipeline(self.config())
        for dataset in ("alpha","beta"): write_typed_proposal_shard(self.root,self.catalog,*self._typed(dataset))
        return freeze_score_root(self.root,self.catalog,self.acceptance)
    def test_lifecycle_only_completes_real_prefix_then_fails_acceleration(self):
        from paper.equipment_anomaly_detection.runner_paper import PipelineNotImplementedError,run_pipeline,verify_completed
        with self.assertRaisesRegex(PipelineNotImplementedError,"stage_not_implemented"):run_pipeline(self.config())
        self.assertEqual(verify_completed(self.config()),{"status":"failed","completed_stages":["preflight","prepare"],"failed_stage":"acceleration","next_stage":"acceleration"})
    def test_catalog_recomputes_gapped_and_fabricated_identities(self):
        from paper.equipment_anomaly_detection.catalog_paper import verify_catalog
        forged=json.loads(_canonical(self.catalog)); forged["datasets"][0]["test"]["row_digest"]="0"*64
        with self.assertRaisesRegex(ValueError,"catalog"):verify_catalog(forged,self.db,"smoke",("alpha","beta"))
        connection=sqlite3.connect(self.db);connection.execute("UPDATE instances SET instance_index=3 WHERE dataset_id=1 AND split='TEST' AND instance_index=1");connection.commit();connection.close()
        with self.assertRaisesRegex(ValueError,"contiguous"):verify_catalog(self.catalog,self.db,"smoke",("alpha","beta"))
    def test_typed_writer_rejects_garbage_unrelated_raw_and_untyped_stability(self):
        """Replacing exact Task1--7 type checks with arbitrary mappings fails here."""
        from paper.equipment_anomaly_detection.artifacts_paper import write_typed_proposal_shard
        args=list(self._typed());
        with self.assertRaises(TypeError):write_typed_proposal_shard(self.root,self.catalog,{"garbage":"accepted"},*args[1:])
        from paper.equipment_anomaly_detection.contracts_paper import RawDataset
        raw=args[0]; bad=list(args); bad[0]=RawDataset("alpha",2,raw.train_indices,(_blob((99,99)),)+raw.train_blobs[1:],raw.train_rows,raw.test_indices,raw.test_blobs,raw.test_rows)
        with self.assertRaises(ValueError):write_typed_proposal_shard(self.root,self.catalog,*bad)
        bad=list(args); bad[4]={"selected":[]}
        with self.assertRaises(TypeError):write_typed_proposal_shard(self.root,self.catalog,*bad)
    def test_manifest_parity_and_lifecycle_recompute_canonical_files(self):
        """Trusting stored totals, acceptance dicts, or undeclared namespaces fails here."""
        from paper.equipment_anomaly_detection.artifacts_paper import MANIFEST_NAME,verify_parity,verify_score_root
        from paper.equipment_anomaly_detection.runner_paper import PipelineStateError,verify_completed
        self._freeze(); self.assertEqual(verify_parity(self.root,self.catalog)["result"],"exact"); self.assertEqual(verify_completed(self.config())["completed_stages"],["preflight","prepare"])
        manifest=self.root/"stage_artifacts_paper"/"proposal_freeze"/MANIFEST_NAME; value=json.loads(manifest.read_bytes()); value["test_row_count"]=999; manifest.write_bytes(_canonical(value))
        with self.assertRaises(ValueError):verify_score_root(self.root,self.catalog)
        with self.assertRaises(PipelineStateError):verify_completed(self.config())
    def test_parity_reloads_bound_acceptance_and_oracle(self):
        from paper.equipment_anomaly_detection.artifacts_paper import verify_parity
        self._freeze(); self.acceptance.write_bytes(b"{}\n")
        with self.assertRaises(ValueError):verify_parity(self.root,self.catalog)
        self._acceptance(((),(),(),())); self.oracle.write_bytes(_canonical({"schema":"paper-task8-coordinate-oracle.v1","policies":[]}))
        with self.assertRaises(ValueError):verify_parity(self.root,self.catalog)
    def test_freeze_namespace_rejects_missing_and_extra_files(self):
        from paper.equipment_anomaly_detection.artifacts_paper import verify_score_root
        self._freeze(); shard=self.root/"stage_artifacts_paper"/"proposal_freeze"/"shards"/"alpha_paper.json"; shard.unlink()
        with self.assertRaises(ValueError):verify_score_root(self.root,self.catalog)
        self.root=self.base/"extra_antigravity"; self._freeze(); (self.root/"stage_artifacts_paper"/"proposal_freeze"/"extra_paper.json").write_text("{}\n")
        with self.assertRaises(ValueError):verify_score_root(self.root,self.catalog)

if __name__=="__main__":unittest.main()
