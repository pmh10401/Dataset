"""Adversarial Task 8 lifecycle, typed shard, and parity tests."""
from __future__ import annotations

from hashlib import sha256
import contextlib, io, json, sqlite3, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import patch

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
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        return raw,sources,route,low,stability,policies,FrozenTask8BackendDecision.from_acceleration_manifest(build_acceleration_manifest("f" * 64, logical_cpu_count=3))
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

    def test_reader_rejects_a_shard_backend_that_differs_from_the_verified_assessment(self):
        """A syntactically valid CPU record is insufficient when it is not the acceleration record for this run."""
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        from paper.equipment_anomaly_detection.artifacts_paper import verify_score_root
        from paper.equipment_anomaly_detection.contracts_paper import FrozenTask8BackendDecision
        self._freeze()
        expected = FrozenTask8BackendDecision.from_acceleration_manifest(build_acceleration_manifest("c" * 64, logical_cpu_count=12))
        with self.assertRaisesRegex(ValueError, "backend"):
            verify_score_root(self.root, self.catalog, expected)


class Task4Task5SuccessRecordTests(unittest.TestCase):
    """The reusable Task4+5 cache must retain all continuous evidence, not only alerts."""

    def setUp(self) -> None:
        self.fixture = Task8LifecycleTests("run")
        self.fixture.setUp()
        self.root = self.fixture.root
        self.catalog = self.fixture.catalog

    def tearDown(self) -> None:
        self.fixture.tearDown()

    def _authority(self, backend):
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        from paper.equipment_anomaly_detection.artifacts_paper import Task4Task5SuccessAuthority
        from paper.equipment_anomaly_detection.runner_paper import context_payload
        context = context_payload(self.fixture.config())
        return Task4Task5SuccessAuthority.from_verified_context(context, self.catalog, build_acceleration_manifest(context["context_sha256"], logical_cpu_count=backend.logical_cpu_count))

    def _task4_task5_values(self):
        """Use the real frozen source threshold and origin identity for each source row."""
        from paper.equipment_anomaly_detection.artifacts_paper import TASK4_TASK5_SOURCE_ORIGIN_IDS
        from paper.equipment_anomaly_detection.lineage_paper import LINEAGE_NODES
        from paper.equipment_anomaly_detection.calibration_paper import upper_tail_pvalues
        from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import EXPECTED_HARD_SOURCES, SOURCE_ALPHA_PAPER, HardSourceBundle, HardSourceResult
        raw, _sources, route, low, stability, policies, backend = self.fixture._typed()
        origins = {node.node_id: node for node in LINEAGE_NODES}
        sources = HardSourceBundle(EXPECTED_HARD_SOURCES, {
            name: HardSourceResult(name, origins[TASK4_TASK5_SOURCE_ORIGIN_IDS[name]], np.asarray((.1, .2)), np.asarray((.1, .2)), upper_tail_pvalues(np.asarray((.1, .2)), np.asarray((.1, .2))), SOURCE_ALPHA_PAPER[name], upper_tail_pvalues(np.asarray((.1, .2)), np.asarray((.1, .2))) <= SOURCE_ALPHA_PAPER[name], True, "eligible")
            for name in EXPECTED_HARD_SOURCES
        })
        return raw, sources, route, low, stability, policies, backend

    def _nonempty_route(self):
        from paper.equipment_anomaly_detection.hard_route_paper import HardRouteResult, RouteLanes, SourceExpansion
        return HardRouteResult(
            RouteLanes(frozenset({0}), frozenset({1}), frozenset()), frozenset({0}), frozenset({1}), frozenset({0}),
            RouteLanes(frozenset(), frozenset(), frozenset({1})), SourceExpansion(frozenset({0}), frozenset({0})),
            RouteLanes(frozenset(), frozenset({1}), frozenset()), frozenset({1}), RouteLanes(frozenset({1}), frozenset(), frozenset()),
            frozenset(), frozenset({1}), 2,
        )

    def test_authority_is_derived_from_context_catalog_and_verified_acceleration(self):
        """Accepting caller-supplied hashes permits a different run's cache to masquerade as this run."""
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        from paper.equipment_anomaly_detection.artifacts_paper import Task4Task5SuccessAuthority
        from paper.equipment_anomaly_detection.runner_paper import context_payload
        _raw, _sources, _route, _low, _stability, _policies, backend = self._task4_task5_values()
        with self.assertRaises(TypeError):
            Task4Task5SuccessAuthority("a" * 64, self.catalog["catalog_sha256"], backend, "b" * 64, "c" * 64)
        context = context_payload(self.fixture.config())
        with self.assertRaises(ValueError):
            Task4Task5SuccessAuthority.from_verified_context({**context, "context_sha256": "0" * 64}, self.catalog, build_acceleration_manifest(context["context_sha256"], logical_cpu_count=3))
        with self.assertRaises(ValueError):
            Task4Task5SuccessAuthority.from_verified_context(context, self.catalog, build_acceleration_manifest("f" * 64, logical_cpu_count=3))

    def test_success_record_round_trips_continuous_sources_and_every_hard_route_stage(self):
        """Discarding OOF scores, an abstention, or any route stage would silently alter a later Task6/7 replay."""
        from paper.equipment_anomaly_detection.artifacts_paper import read_task4_task5_success_record, write_task4_task5_success_record
        from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import HardSourceBundle, HardSourceResult
        raw, sources, _route, _low, _stability, _policies, backend = self._task4_task5_values()
        changed = dict(sources.by_name)
        source = changed["rocket_exp40"]
        changed["rocket_exp40"] = HardSourceResult(source.name, source.origin, np.empty(0), np.empty(0), np.ones(2), source.threshold, np.zeros(2, dtype=bool), False, "dependency_unavailable")
        sources = HardSourceBundle(sources.source_names, changed)
        route = self._nonempty_route()
        authority = self._authority(backend)
        path = write_task4_task5_success_record(self.root, self.catalog, authority, raw, sources, route)
        with self.assertRaises(FileExistsError):
            write_task4_task5_success_record(self.root, self.catalog, authority, raw, sources, route)
        reread = read_task4_task5_success_record(self.root, self.catalog, authority, raw)
        self.assertEqual(path.name, "task4_task5_success_antigravity.json")
        self.assertEqual(reread.route, route)
        self.assertEqual(tuple(reread.sources.source_names), tuple(sources.source_names))
        self.assertFalse(reread.sources.by_name["rocket_exp40"].eligible)
        self.assertEqual(reread.sources.by_name["rocket_exp40"].eligibility_reason, "dependency_unavailable")
        for name in sources.source_names:
            expected, actual = sources.by_name[name], reread.sources.by_name[name]
            self.assertEqual(actual.origin, expected.origin)
            self.assertEqual(actual.threshold, expected.threshold)
            self.assertEqual(actual.eligible, expected.eligible)
            self.assertEqual(actual.eligibility_reason, expected.eligibility_reason)
            self.assertTrue(np.array_equal(actual.train_oof_score, expected.train_oof_score))
            self.assertTrue(np.array_equal(actual.test_score, expected.test_score))
            self.assertTrue(np.array_equal(actual.test_pvalue, expected.test_pvalue))
            self.assertTrue(np.array_equal(actual.test_native_pass, expected.test_native_pass))
            self.assertFalse(actual.test_score.flags.writeable)
            self.assertEqual(actual.test_score.dtype, np.dtype("float64"))

    def test_success_record_rejects_tampered_authorities_payload_and_namespace_attacks(self):
        """Trusting a record after any identity mutation or unsafe sibling changes cache provenance."""
        from paper.equipment_anomaly_detection.artifacts_paper import TASK4_TASK5_SUCCESS_DIRECTORY, read_task4_task5_success_record, write_task4_task5_success_record
        raw, sources, route, _low, _stability, _policies, backend = self._task4_task5_values()
        authority = self._authority(backend)
        path = write_task4_task5_success_record(self.root, self.catalog, authority, raw, sources, route)
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        from paper.equipment_anomaly_detection.contracts_paper import FrozenTask8BackendDecision, RawDataset
        original = json.loads(path.read_bytes())
        for field, value in (("context_sha256", "d" * 64), ("catalog_sha256", "e" * 64), ("scientific_config_sha256", "f" * 64), ("code_closure_sha256", "1" * 64), ("backend", FrozenTask8BackendDecision.from_acceleration_manifest(build_acceleration_manifest(authority.context_sha256, logical_cpu_count=12)).artifact_record())):
            forged = {**original, field: value}
            path.write_bytes(_canonical(forged))
            with self.assertRaises(ValueError):
                read_task4_task5_success_record(self.root, self.catalog, authority, raw)
        path.write_bytes(_canonical(original))
        bad_raw = RawDataset(raw.dataset, raw.metadata_length, raw.train_indices, raw.train_blobs, raw.train_rows, raw.test_indices, raw.test_blobs[:-1] + (_blob((99, 99)),), raw.test_rows)
        with self.assertRaises(ValueError):
            read_task4_task5_success_record(self.root, self.catalog, authority, bad_raw)
        tampered = json.loads(path.read_bytes())
        tampered["payload"]["sources"][0]["test_score"][0] = 99.0
        path.write_bytes(_canonical(tampered))
        with self.assertRaisesRegex(ValueError, "payload"):
            read_task4_task5_success_record(self.root, self.catalog, authority, raw)
        path.write_bytes(_canonical({**tampered, "payload_sha256": sha256(_canonical(tampered["payload"])).hexdigest()}))
        with self.assertRaises(ValueError):
            read_task4_task5_success_record(self.root, self.catalog, authority, raw)
        from paper.equipment_anomaly_detection.calibration_paper import upper_tail_pvalues
        tampered["payload"]["sources"][0]["test_pvalue"] = upper_tail_pvalues(np.asarray(tampered["payload"]["sources"][0]["train_oof_score"]), np.asarray(tampered["payload"]["sources"][0]["test_score"])).tolist()
        path.write_bytes(_canonical({**tampered, "payload_sha256": sha256(_canonical(tampered["payload"])).hexdigest()}))
        self.assertEqual(read_task4_task5_success_record(self.root, self.catalog, authority, raw).sources.by_name["rocket_exp40"].test_score[0], 99.0)
        path.unlink()
        write_task4_task5_success_record(self.root, self.catalog, authority, raw, sources, route)
        path.parent.joinpath("extra_antigravity.json").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "namespace"):
            read_task4_task5_success_record(self.root, self.catalog, authority, raw)
        path.parent.joinpath("extra_antigravity.json").unlink()
        work = self.root / "stage_artifacts_paper" / "proposal_work" / TASK4_TASK5_SUCCESS_DIRECTORY
        work.joinpath("unknown_antigravity").mkdir()
        with self.assertRaisesRegex(ValueError, "namespace"):
            read_task4_task5_success_record(self.root, self.catalog, authority, raw)
        work.joinpath("unknown_antigravity").rmdir()
        path.unlink()
        path.parent.rmdir()
        path.parent.symlink_to(work, target_is_directory=True)
        with self.assertRaises(ValueError):
            read_task4_task5_success_record(self.root, self.catalog, authority, raw)

    def test_reread_sources_and_route_project_to_byte_identical_existing_task8_shard(self):
        """A serialization conversion that changes the downstream Task8 bytes is not a scientifically safe reuse seam."""
        from paper.equipment_anomaly_detection.artifacts_paper import MANIFEST_NAME, freeze_score_root, read_task4_task5_success_record, verify_score_root, write_task4_task5_success_record, write_typed_proposal_shard
        from paper.equipment_anomaly_detection.contracts_paper import LowFrequencyResult
        from paper.equipment_anomaly_detection.routing_paper import PolicyBundle, PolicyRoute
        from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import HardSourceBundle, HardSourceResult
        raw, sources, _route, low, stability, policies, backend = self._task4_task5_values()
        changed = dict(sources.by_name); source = changed["rocket_exp40"]
        changed["rocket_exp40"] = HardSourceResult(source.name, source.origin, np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), np.ones(2, dtype=np.float64), source.threshold, np.zeros(2, dtype=bool), False, "dependency_unavailable")
        sources = HardSourceBundle(sources.source_names, changed)
        route = self._nonempty_route()
        low = LowFrequencyResult((raw.dataset,), low.rows, (), (), 0, {raw.dataset: route.exp253.hard})
        policies = PolicyBundle(tuple(PolicyRoute(name, {raw.dataset: route.automatic_hard}, {raw.dataset: np.asarray((0.0, 0.9))}, {raw.dataset: np.asarray((0.0, 0.6))}) for name in ("hard_only", "hard_plus_source", "hard_plus_stable", "hard_plus_unstable")))
        authority = self._authority(backend)
        write_task4_task5_success_record(self.root, self.catalog, authority, raw, sources, route)
        reread = read_task4_task5_success_record(self.root, self.catalog, authority, raw)
        memory_root = self.fixture.base / "memory_projection_antigravity"
        reread_root = self.fixture.base / "reread_projection_antigravity"
        memory_path = write_typed_proposal_shard(memory_root, self.catalog, raw, sources, route, low, stability, policies, backend)
        reread_path = write_typed_proposal_shard(reread_root, self.catalog, raw, reread.sources, reread.route, low, stability, policies, backend)
        # Complete both smoke inventories with the untouched beta fixture, then freeze/verify v3 manifests.
        beta = self.fixture._typed("beta")
        write_typed_proposal_shard(memory_root, self.catalog, *beta)
        write_typed_proposal_shard(reread_root, self.catalog, *beta)
        freeze_score_root(memory_root, self.catalog); freeze_score_root(reread_root, self.catalog)
        verify_score_root(memory_root, self.catalog); verify_score_root(reread_root, self.catalog)
        self.assertEqual(memory_path.read_bytes(), reread_path.read_bytes())
        self.assertEqual((memory_root / "stage_artifacts_paper" / "proposal_freeze" / MANIFEST_NAME).read_bytes(), (reread_root / "stage_artifacts_paper" / "proposal_freeze" / MANIFEST_NAME).read_bytes())

    def test_writer_rejects_non_float64_source_arrays(self):
        """Casting float32 or integer source evidence on write would make the cache a lossy scientific transform."""
        from paper.equipment_anomaly_detection.artifacts_paper import write_task4_task5_success_record
        from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import HardSourceBundle, HardSourceResult
        raw, sources, route, _low, _stability, _policies, backend = self._task4_task5_values()
        changed = dict(sources.by_name); source = changed["rocket_exp40"]
        changed["rocket_exp40"] = HardSourceResult(source.name, source.origin, source.train_oof_score.astype(np.float32), source.test_score.astype(np.float32), source.test_pvalue.astype(np.float32), source.threshold, source.test_native_pass, source.eligible, source.eligibility_reason)
        with self.assertRaisesRegex(ValueError, "float64"):
            write_task4_task5_success_record(self.root, self.catalog, self._authority(backend), raw, HardSourceBundle(sources.source_names, changed), route)
        changed["rocket_exp40"] = HardSourceResult(source.name, source.origin, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), np.ones(2, dtype=np.float64), source.threshold, np.zeros(2, dtype=bool), False, "dependency_unavailable")
        with self.assertRaisesRegex(ValueError, "float64"):
            write_task4_task5_success_record(self.root, self.catalog, self._authority(backend), raw, HardSourceBundle(sources.source_names, changed), route)

    def test_recursive_journal_accepts_only_verified_proposal_work_records(self):
        """A parent-published success record is valid work, while an unverified proposal-work sibling is pollution."""
        from paper.equipment_anomaly_detection.acceleration_paper import verify_acceleration_manifest
        from paper.equipment_anomaly_detection.artifacts_paper import FrozenTask8BackendDecision, canonical_json_bytes, freeze_score_root, read_canonical, write_create_only, write_task4_task5_success_record, write_typed_proposal_shard
        from paper.equipment_anomaly_detection.contracts_paper import LowFrequencyResult
        from paper.equipment_anomaly_detection.routing_paper import PolicyBundle, PolicyRoute
        from paper.equipment_anomaly_detection import runner_paper as runner
        closure_patch = patch.object(runner, "_closure", return_value={"runner_paper.py": "0" * 64})
        closure_patch.start(); self.addCleanup(closure_patch.stop)
        config = self.fixture.config(); self.root.mkdir()
        context = runner.context_payload(config)
        write_create_only(runner._journal_path(self.root, 0, "context"), canonical_json_bytes(runner._record({"schema": "paper-task8-journal-context.v1", "context": context, "previous_record_sha256": "genesis"})))
        for stage in ("preflight", "prepare", "acceleration"):
            runner._write_complete(self.root, stage, context, runner._records(self.root)[-1]["record_sha256"])
        acceleration = verify_acceleration_manifest(read_canonical(runner._artifact_path(self.root, "acceleration")), context["context_sha256"])
        raw, sources, route, _low, stability, _policies, _backend = self._task4_task5_values()
        authority = self._authority(FrozenTask8BackendDecision.from_acceleration_manifest(acceleration))
        write_task4_task5_success_record(self.root, self.catalog, authority, raw, sources, route)
        first, second = self.fixture._typed("alpha"), self.fixture._typed("beta")
        raw_patch = patch.object(runner, "load_dataset_values", side_effect=lambda _db, dataset: {"alpha": raw, "beta": second[0]}[dataset])
        raw_patch.start(); self.addCleanup(raw_patch.stop)
        low = LowFrequencyResult(("alpha", "beta"), first[3].rows + second[3].rows, (), (), 0, {"alpha": frozenset(), "beta": frozenset()})
        policies = PolicyBundle(tuple(PolicyRoute(name, {"alpha": frozenset(), "beta": frozenset()}, {"alpha": np.zeros(2), "beta": np.zeros(2)}, {"alpha": np.zeros(2), "beta": np.zeros(2)}) for name in ("hard_only", "hard_plus_source", "hard_plus_stable", "hard_plus_unstable")))
        backend = FrozenTask8BackendDecision.from_acceleration_manifest(acceleration)
        write_typed_proposal_shard(self.root, self.catalog, raw, sources, route, low, stability, policies, backend)
        write_typed_proposal_shard(self.root, self.catalog, second[0], second[1], second[2], low, second[4], policies, backend)
        freeze_score_root(self.root, self.catalog, self.fixture.acceptance)
        proposal = runner._proposal_artifact(self.root, context)
        write_create_only(runner._artifact_path(self.root, "proposal_score"), canonical_json_bytes(proposal))
        write_create_only(runner._journal_path(self.root, 4, "proposal_score"), canonical_json_bytes(runner._record({"schema": "paper-task8-journal-complete.v1", "stage": "proposal_score", "previous_record_sha256": runner._records(self.root)[-1]["record_sha256"], "artifact": runner._artifact_record(self.root, "proposal_score", context), "finished_at": runner._now()})))
        self.assertEqual(runner.verify_journal(config)[2], ["preflight", "prepare", "acceleration", "proposal_score"])
        self.root.joinpath("stage_artifacts_paper", "proposal_work", "extra_antigravity.json").write_text("{}\n")
        with self.assertRaisesRegex(runner.PipelineStateError, "namespace pollution"):
            runner.verify_journal(config)


class Task9AccelerationTests(unittest.TestCase):
    def test_context_registry_binds_exported_acceleration_schema(self):
        """A copied schema literal can leave context lineage one revision behind its artifact verifier."""
        from paper.equipment_anomaly_detection.acceleration_paper import ACCELERATION_SCHEMA
        from paper.equipment_anomaly_detection.runner_paper import context_payload
        fixture = Task8LifecycleTests("run")
        fixture.setUp()
        try:
            with patch("paper.equipment_anomaly_detection.runner_paper._closure", return_value={}):
                self.assertEqual(context_payload(fixture.config())["constants"]["registry"]["acceleration"], ACCELERATION_SCHEMA)
        finally:
            fixture.tearDown()

    def test_worker_policy_uses_all_but_one_logical_cpu_and_records_parallel_mode(self):
        """Capping a 12-core CPU stage at seven would silently violate the official policy."""
        from paper.equipment_anomaly_detection.acceleration_paper import WorkerRecord, build_acceleration_manifest
        workers = WorkerRecord.from_logical_cpu_count(12)
        self.assertEqual((workers.logical_cpu_count, workers.dataset_workers, workers.inner_numerical_threads), (12, 11, 1))
        self.assertEqual(workers.outer_execution_mode, "parallel_task4_task5")
        manifest = build_acceleration_manifest("a" * 64, logical_cpu_count=12)
        self.assertEqual(manifest["workers"], {"logical_cpu_count": 12, "dataset_workers": 11, "inner_numerical_threads": 1, "outer_execution_mode": "parallel_task4_task5"})
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
        with patch("paper.equipment_anomaly_detection.acceleration_paper._backends", side_effect=AssertionError("ambient backend reprobe")):
            self.assertEqual(verify_acceleration_manifest(manifest, "b" * 64), manifest)

    def test_acceleration_assessment_hash_rejects_any_unhashed_field_mutation(self):
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest, verify_acceleration_manifest
        manifest = build_acceleration_manifest("d" * 64, logical_cpu_count=12)
        forged = json.loads(json.dumps(manifest))
        forged["fallback_reason"] = "forged"
        with self.assertRaisesRegex(ValueError, "acceleration artifact assessment hash differs"):
            verify_acceleration_manifest(forged, "d" * 64)

    def test_acceleration_resume_reports_interpreter_identity_mismatch(self):
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest, verify_acceleration_manifest
        manifest = build_acceleration_manifest("c" * 64, logical_cpu_count=12)
        runtime = dict(manifest["runtime"])
        runtime["executable"] = "/different/python"
        with self.assertRaisesRegex(ValueError, "interpreter_identity_mismatch"):
            verify_acceleration_manifest(manifest, "c" * 64, current_runtime=runtime)

    def test_frozen_shard_backend_is_derived_from_the_verified_acceleration_assessment(self):
        """A stale task9_pending marker would discard the CPU assessment that authorized scoring."""
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        from paper.equipment_anomaly_detection.contracts_paper import FrozenTask8BackendDecision
        manifest = build_acceleration_manifest("e" * 64, logical_cpu_count=12)
        decision = FrozenTask8BackendDecision.from_acceleration_manifest(manifest)
        self.assertEqual(decision.selected_backend, "cpu")
        self.assertEqual(decision.fallback_reason, manifest["fallback_reason"])
        self.assertEqual(decision.assessment_sha256, manifest["assessment_sha256"])
        self.assertEqual((decision.logical_cpu_count, decision.dataset_workers, decision.inner_numerical_threads, decision.outer_execution_mode), (12, 11, 1, "parallel_task4_task5"))

    def test_public_cli_has_no_dataset_worker_override(self):
        """Worker selection belongs to the verified artifact, not an operator-provided score switch."""
        from paper.equipment_anomaly_detection.pipeline_paper import build_parser
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(("run", "--db-path", "fixture.sqlite", "--run-root", "new_root_antigravity", "--mode", "smoke", "--smoke-dataset", "alpha", "--dataset-workers", "2"))


class Task9ParallelCompositionTests(unittest.TestCase):
    """Bounded Task4/5 doubles keep the parallel ordering test independent of expensive transforms."""

    def setUp(self) -> None:
        self.fixture = Task8LifecycleTests("run")
        self.fixture.setUp()
        first, second = self.fixture._typed("alpha"), self.fixture._typed("beta")
        self.raw = {"alpha": first[0], "beta": second[0]}
        from paper.equipment_anomaly_detection.contracts_paper import LowFrequencyResult
        from paper.equipment_anomaly_detection.routing_paper import PolicyBundle, PolicyRoute
        self.per_dataset = {"alpha": (first[1], first[2]), "beta": (second[1], second[2])}
        self.low = LowFrequencyResult(("alpha", "beta"), first[3].rows + second[3].rows, (), (), 0, {"alpha": frozenset(), "beta": frozenset()})
        self.stability = first[4]
        self.policies = PolicyBundle(tuple(PolicyRoute(name, {dataset: frozenset() for dataset in self.raw}, {dataset: np.zeros(2) for dataset in self.raw}, {dataset: np.zeros(2) for dataset in self.raw}) for name in ("hard_only", "hard_plus_source", "hard_plus_stable", "hard_plus_unstable")))

    def tearDown(self) -> None:
        self.fixture.tearDown()

    def _acceleration(self, logical_cpu_count: int):
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        return build_acceleration_manifest("9" * 64, logical_cpu_count=logical_cpu_count)

    def _patched_composition(self, manifest, *, fail_dataset=None):
        from paper.equipment_anomaly_detection import proposal_score_paper as score
        events = []
        def hard(raw):
            events.append(("task4", raw.dataset))
            if raw.dataset == fail_dataset:
                raise RuntimeError("bounded worker failure")
            return self.per_dataset[raw.dataset][0]
        def mini(raw): return object()
        def route(raw, _sources, _mini):
            events.append(("task5", raw.dataset))
            return self.per_dataset[raw.dataset][1]
        def low(raw, lanes):
            self.assertEqual(events, [("task4", "alpha"), ("task5", "alpha"), ("task4", "beta"), ("task5", "beta")])
            self.assertEqual(lanes, {"alpha": frozenset(), "beta": frozenset()})
            events.append(("task6", "global")); return self.low
        class Future:
            def __init__(self, fn, args):
                try: self.value, self.error = fn(*args), None
                except Exception as error: self.value, self.error = None, error
            def result(self):
                if self.error is not None: raise self.error
                return self.value
            def cancel(self): return False
        class InlineExecutor:
            observed_workers = []
            def __init__(self, *, max_workers): self.observed_workers.append(max_workers)
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def submit(self, fn, *args): return Future(fn, args)
        patches = (
            patch.object(score, "score_hard_sources", side_effect=hard),
            patch.object(score, "score_minirocket_confirmations", side_effect=mini),
            patch.object(score, "replay_hard_route", side_effect=route),
            patch.object(score, "score_low_frequency", side_effect=low),
            patch.object(score, "classify_stability", return_value=self.stability),
            patch.object(score, "build_hard_evidence_surface", return_value=object()),
            patch.object(score, "build_policies", return_value=self.policies),
            patch.object(score, "ProcessPoolExecutor", InlineExecutor),
            patch.object(score, "as_completed", side_effect=lambda futures: tuple(futures)),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], patches[7], patches[8]:
            return score.compose_proposal_score(self.raw, manifest), events, InlineExecutor.observed_workers

    def test_two_worker_collection_matches_sequential_before_one_global_task6(self):
        """Moving Task6 into a dataset worker changes the population-wide low-frequency decision."""
        sequential, sequential_events, sequential_workers = self._patched_composition(self._acceleration(1))
        parallel, parallel_events, parallel_workers = self._patched_composition(self._acceleration(3))
        self.assertEqual(tuple(sequential.sources), tuple(parallel.sources))
        self.assertEqual(tuple(sequential.routes), tuple(parallel.routes))
        self.assertEqual(sequential.low_frequency, parallel.low_frequency)
        self.assertEqual(tuple(route.alerts for route in sequential.policies.policies), tuple(route.alerts for route in parallel.policies.policies))
        self.assertEqual(sequential_events[-1], ("task6", "global"))
        self.assertEqual(parallel_events[-1], ("task6", "global"))
        self.assertEqual(sequential_workers, [])
        self.assertEqual(parallel_workers, [2])

    def test_parallel_worker_failure_names_its_dataset(self):
        """Dropping a failed dataset would produce a plausible but scientifically incomplete global population."""
        with self.assertRaisesRegex(RuntimeError, "alpha.*bounded worker failure"):
            self._patched_composition(self._acceleration(3), fail_dataset="alpha")

    def test_parent_collection_publishes_worker_result_without_passing_writer_to_worker(self):
        """Letting a worker receive the writer/root would make create-only cache publication race process failures."""
        from paper.equipment_anomaly_detection import proposal_score_paper as score
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        from paper.equipment_anomaly_detection.artifacts_paper import Task4Task5SuccessAuthority, read_task4_task5_success_record
        from paper.equipment_anomaly_detection.contracts_paper import FrozenTask8BackendDecision
        from paper.equipment_anomaly_detection.runner_paper import context_payload
        record_fixture = Task4Task5SuccessRecordTests("run")
        record_fixture.setUp()
        try:
            raw, sources, route, _low, _stability, _policies, _backend = record_fixture._task4_task5_values()
            context = context_payload(record_fixture.fixture.config())
            acceleration = build_acceleration_manifest(context["context_sha256"], logical_cpu_count=3)
            authority = Task4Task5SuccessAuthority.from_verified_context(context, record_fixture.catalog, acceleration)
            publisher = score.task4_task5_success_publisher(record_fixture.root, record_fixture.catalog, authority)
            class Future:
                def result(self): return sources, route
                def cancel(self): return False
            case = self
            class ParentExecutor:
                def __init__(self, *, max_workers): self.max_workers = max_workers
                def __enter__(self): return self
                def __exit__(self, *_args): return False
                def submit(self, fn, *args):
                    case.assertEqual(len(args), 1)
                    case.assertIs(args[0], raw)
                    return Future()
            with patch.object(score, "ProcessPoolExecutor", ParentExecutor), patch.object(score, "as_completed", side_effect=lambda futures: tuple(futures)):
                observed_sources, observed_routes = score._collect_task4_task5({raw.dataset: raw}, FrozenTask8BackendDecision.from_acceleration_manifest(acceleration), publish_success=publisher)
            self.assertEqual(observed_sources[raw.dataset], sources)
            self.assertEqual(observed_routes[raw.dataset], route)
            self.assertEqual(read_task4_task5_success_record(record_fixture.root, record_fixture.catalog, authority, raw).route, route)
        finally:
            record_fixture.tearDown()

    def test_composition_with_runner_inputs_always_publishes_success(self):
        """A production compose call that skips the parent publisher leaves successful worker work unrecoverable."""
        from paper.equipment_anomaly_detection import proposal_score_paper as score
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        from paper.equipment_anomaly_detection.artifacts_paper import Task4Task5SuccessAuthority, read_task4_task5_success_record
        from paper.equipment_anomaly_detection.runner_paper import context_payload
        record_fixture = Task4Task5SuccessRecordTests("run")
        record_fixture.setUp()
        try:
            raw, sources, route, low, stability, policies, _backend = record_fixture._task4_task5_values()
            context = context_payload(record_fixture.fixture.config()); acceleration = build_acceleration_manifest(context["context_sha256"], logical_cpu_count=1)
            authority = Task4Task5SuccessAuthority.from_verified_context(context, record_fixture.catalog, acceleration)
            def collect(_raw, _backend, *, publish_success):
                publish_success(raw, sources, route)
                return {raw.dataset: sources}, {raw.dataset: route}
            with patch.object(score, "_collect_task4_task5", side_effect=collect), patch.object(score, "score_low_frequency", return_value=low), patch.object(score, "classify_stability", return_value=stability), patch.object(score, "build_hard_evidence_surface", return_value=object()), patch.object(score, "build_policies", return_value=policies):
                score.compose_proposal_score({raw.dataset: raw}, acceleration, root=record_fixture.root, context=context, catalog=record_fixture.catalog)
            self.assertEqual(read_task4_task5_success_record(record_fixture.root, record_fixture.catalog, authority, raw).route, route)
        finally:
            record_fixture.tearDown()

    def test_parallel_completion_progress_is_flush_visible_and_keeps_sorted_results(self):
        """Collecting submitted futures in dataset order hides actual completion order and leaves long runs opaque."""
        from paper.equipment_anomaly_detection import proposal_score_paper as score
        completed = []
        values = self.per_dataset
        class Future:
            def __init__(self, dataset): self.dataset = dataset
            def result(self): completed.append(self.dataset); return values[self.dataset]
        class CompletionExecutor:
            def __init__(self, *, max_workers): self.futures = {}
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def submit(self, _fn, raw): future = Future(raw.dataset); self.futures[raw.dataset] = future; return future
        def completed_first(futures):
            by_dataset = {future.dataset: future for future in futures}
            return (by_dataset["beta"], by_dataset["alpha"])
        class RecordingStderr(io.StringIO):
            def __init__(self):
                super().__init__(); self.flush_calls = 0
            def flush(self):
                self.flush_calls += 1
                super().flush()
        stderr = RecordingStderr()
        patches = (
            patch.object(score, "ProcessPoolExecutor", CompletionExecutor),
            patch.object(score, "as_completed", side_effect=completed_first),
            patch.object(score, "score_low_frequency", return_value=self.low),
            patch.object(score, "classify_stability", return_value=self.stability),
            patch.object(score, "build_hard_evidence_surface", return_value=object()),
            patch.object(score, "build_policies", return_value=self.policies),
        )
        with contextlib.redirect_stderr(stderr), patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            proposal = score.compose_proposal_score(self.raw, self._acceleration(3))
        lines = stderr.getvalue().splitlines()
        self.assertEqual(completed, ["beta", "alpha"])
        self.assertEqual(tuple(proposal.sources), ("alpha", "beta"))
        self.assertGreaterEqual(stderr.flush_calls, 7)
        self.assertTrue(any("Task4+Task5 start" in line and "0/2" in line for line in lines))
        self.assertTrue(any("Task4+Task5 1/2" in line and "50.0%" in line and "dataset=beta" in line and "elapsed=" in line and "rate=" in line and "eta=" in line for line in lines))
        self.assertTrue(any("Task4+Task5 2/2" in line and "100.0%" in line and "dataset=alpha" in line for line in lines))
        self.assertTrue(any(line == "Task6 start" for line in lines))
        self.assertTrue(any(line == "Task6 finish" for line in lines))
        self.assertTrue(any(line == "Task7 start" for line in lines))
        self.assertTrue(any(line == "Task7 finish" for line in lines))

if __name__=="__main__":unittest.main()
