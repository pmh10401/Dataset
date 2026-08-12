"""Adversarial Task 8 lifecycle, typed shard, and parity tests."""
from __future__ import annotations

from hashlib import sha256
import json, sqlite3, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
import sys

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
        for dataset in ("alpha","beta"): write_typed_proposal_shard(self.root,self.catalog,*self._typed(dataset))
        return freeze_score_root(self.root,self.catalog,self.acceptance)
    def test_production_score_stage_rejects_nonfrozen_fixture_database(self):
        from paper.equipment_anomaly_detection.runner_paper import run_pipeline
        with self.assertRaisesRegex(ValueError,"frozen authority"):
            run_pipeline(self.config())
    def test_score_freeze_without_acceptance_is_verifiable_but_cannot_claim_parity(self):
        """A score freeze binds typed shards/catalog only; an oracle remains optional post-freeze evidence."""
        from paper.equipment_anomaly_detection.artifacts_paper import freeze_score_root, verify_parity, verify_score_root, write_typed_proposal_shard
        for dataset in ("alpha", "beta"):
            write_typed_proposal_shard(self.root, self.catalog, *self._typed(dataset))
        manifest = freeze_score_root(self.root, self.catalog)
        self.assertEqual(manifest.parity_status, "not_available")
        self.assertEqual(verify_score_root(self.root, self.catalog).parity_status, "not_available")
        with self.assertRaisesRegex(ValueError, "parity"):
            verify_parity(self.root, self.catalog)
    def test_global_task6_task7_typed_values_serialize_dataset_shards_without_labels(self):
        """A mapping-wide Task6/7 result is projected into each typed Task8 shard, never recomputed locally."""
        from paper.equipment_anomaly_detection.artifacts_paper import freeze_score_root, write_typed_proposal_shard
        from paper.equipment_anomaly_detection.contracts_paper import LowFrequencyResult
        from paper.equipment_anomaly_detection.routing_paper import PolicyBundle, PolicyRoute
        first, second = self._typed("alpha"), self._typed("beta")
        raw = {"alpha": first[0], "beta": second[0]}
        low = LowFrequencyResult(("alpha", "beta"), tuple(row for values in (first[3], second[3]) for row in values.rows), (), (), 0, {"alpha": frozenset(), "beta": frozenset()})
        policies = PolicyBundle(tuple(PolicyRoute(name, {dataset: frozenset() for dataset in raw}, {dataset: np.zeros(2) for dataset in raw}, {dataset: np.zeros(2) for dataset in raw}) for name in ("hard_only", "hard_plus_source", "hard_plus_stable", "hard_plus_unstable")))
        for values in (first, second):
            write_typed_proposal_shard(self.root, self.catalog, values[0], values[1], values[2], low, first[4], policies, values[6])
        self.assertEqual(freeze_score_root(self.root, self.catalog).datasets, ("alpha", "beta"))
        self.assertNotIn("paper.equipment_anomaly_detection.label_access_paper", sys.modules)
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
    def test_typed_writer_rejects_substituted_decoded_rows_and_duck_typed_source(self):
        """Trusting RawDataset blobs without their decoded <f4 rows must fail here."""
        from paper.equipment_anomaly_detection.artifacts_paper import write_typed_proposal_shard
        from paper.equipment_anomaly_detection.contracts_paper import RawDataset
        args=list(self._typed()); raw=args[0]
        bad=list(args); bad[0]=RawDataset(raw.dataset,raw.metadata_length,raw.train_indices,raw.train_blobs,(np.array((42,43),dtype="<f4"),)+raw.train_rows[1:],raw.test_indices,raw.test_blobs,raw.test_rows)
        with self.assertRaises(ValueError):write_typed_proposal_shard(self.root,self.catalog,*bad)
        from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import HardSourceBundle
        bad=list(args); sources=bad[1]; copied=dict(sources.by_name); source=copied["rocket_exp40"]; copied["rocket_exp40"]=SimpleNamespace(name=source.name,origin=source.origin,train_oof_score=source.train_oof_score,test_score=source.test_score,test_pvalue=source.test_pvalue,threshold=source.threshold,test_native_pass=source.test_native_pass,eligible=source.eligible,eligibility_reason=source.eligibility_reason); bad[1]=HardSourceBundle(sources.source_names,copied)
        with self.assertRaises(TypeError):write_typed_proposal_shard(self.root,self.catalog,*bad)
    def test_manifest_parity_and_lifecycle_recompute_canonical_files(self):
        """Trusting stored totals, acceptance dicts, or undeclared namespaces fails here."""
        from paper.equipment_anomaly_detection.artifacts_paper import MANIFEST_NAME,verify_parity,verify_score_root
        from paper.equipment_anomaly_detection.runner_paper import PipelineStateError,verify_completed
        self._freeze(); self.assertEqual(verify_parity(self.root,self.catalog)["result"],"exact")
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


class Task9AccelerationTests(unittest.TestCase):
    def test_worker_cap_and_unavailable_exact_implementations_are_recorded_truthfully(self):
        from paper.equipment_anomaly_detection.acceleration_paper import WorkerRecord, build_acceleration_manifest
        workers = WorkerRecord.from_logical_cpu_count(12)
        self.assertEqual((workers.logical_cpu_count, workers.dataset_workers, workers.deep_method_workers), (12, 7, 1))
        manifest = build_acceleration_manifest("a" * 64, logical_cpu_count=12)
        self.assertEqual(manifest["selected_backend"], "cpu")
        self.assertEqual({entry["family"] for entry in manifest["backends"]}, {"pytorch", "mlx", "mps"})
        self.assertTrue(all(entry["decision"] == "cpu" for entry in manifest["stages"]))
        self.assertIn("implementation_unavailable", {entry["reason"] for entry in manifest["stages"]})

    def test_mlx_gate_rejects_candidate_mismatch_and_insufficient_speedup(self):
        from paper.equipment_anomaly_detection.acceleration_paper import decide_dense_mlx_gate
        mismatch = decide_dense_mlx_gate(np.asarray([.1, .2]), np.asarray([.1, .2]), (1,), (2,), cpu_seconds=1.0, mlx_seconds=.1, tolerance=1e-12)
        self.assertEqual(mismatch["selected_backend"], "cpu")
        self.assertEqual(mismatch["reason"], "candidate_indices_differ")
        slow = decide_dense_mlx_gate(np.asarray([.1, .2]), np.asarray([.1, .2]), (1,), (1,), cpu_seconds=1.0, mlx_seconds=.95, tolerance=1e-12)
        self.assertEqual(slow["selected_backend"], "cpu")
        self.assertEqual(slow["reason"], "speedup_below_1_1x")

    def test_historical_acceleration_manifest_does_not_reprobe_another_interpreter(self):
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest, verify_acceleration_manifest
        manifest = build_acceleration_manifest("b" * 64, logical_cpu_count=12)
        self.assertEqual(verify_acceleration_manifest(manifest, "b" * 64), manifest)

    def test_acceleration_resume_reports_interpreter_identity_mismatch(self):
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest, verify_acceleration_manifest
        manifest = build_acceleration_manifest("c" * 64, logical_cpu_count=12)
        runtime = dict(manifest["runtime"])
        runtime["executable"] = "/different/python"
        with self.assertRaisesRegex(ValueError, "interpreter_identity_mismatch"):
            verify_acceleration_manifest(manifest, "c" * 64, current_runtime=runtime)

if __name__=="__main__":unittest.main()
