"""Behavioral Task 5 tests for the Exp449-only proposal freeze boundary."""

from __future__ import annotations

import ast
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import struct
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp449_reproducible_rebaseline_antigravity.config_antigravity import (
    FROZEN_EXP449_CONFIG,
)
from experiments.exp449_reproducible_rebaseline_antigravity import contracts_antigravity as contracts
from experiments.exp449_reproducible_rebaseline_antigravity.contracts_antigravity import (
    Exp449Context,
    Exp449ProposalManifest,
    Exp449ShardDescriptor,
)
from experiments.exp449_reproducible_rebaseline_antigravity.artifacts_antigravity import (
    canonical_float64_bytes_antigravity,
    freeze_exp449_proposal_antigravity,
    validate_exp449_catalog_antigravity,
)
from experiments.exp449_reproducible_rebaseline_antigravity.verifier_antigravity import (
    verify_exp449_proposal_antigravity,
)


def _identity_antigravity(indices: tuple[int, ...], blobs: tuple[bytes, ...]) -> dict[str, object]:
    return {
        "indices": list(indices),
        "count": len(blobs),
        "values_blob_sha256": [sha256(blob).hexdigest() for blob in blobs],
    }


def _catalog_sha256_antigravity(bare: dict[str, object]) -> str:
    return sha256(json.dumps(bare, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode() + b"\n").hexdigest()


def _context_antigravity(root: Path, catalog: dict[str, object], backend: object) -> Exp449Context:
    """Build the real immutable context while redirecting only its test root authority."""
    with patch.object(contracts, "_SMOKE_ROOTS", frozenset({root.resolve()})):
        return Exp449Context.from_verified_inputs(
            run_root=root, mode="smoke", catalog=catalog, acceleration=backend,
            database_sha256="a" * 64, code_closure={"fixture_antigravity.py": "c" * 64},
            lineage_sha256="d" * 64, codemap_sha256="e" * 64,
            interpreter="typed-fixture-python", environment={"fixture": "typed"},
            seeds_and_constants={"seed": 7},
        )


class _TypedProposalFixtureAntigravity:
    """A real paper-type ProposalScore, scaled only by reusing typed values in memory."""

    def __init__(self, root: Path) -> None:
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        from paper.equipment_anomaly_detection.calibration_paper import upper_tail_pvalues
        from paper.equipment_anomaly_detection.contracts_paper import FrozenTask8BackendDecision, LowFrequencyResult, LowFrequencyRow, RawDataset
        from paper.equipment_anomaly_detection.hard_route_paper import HardRouteResult, RouteLanes, SourceExpansion
        from paper.equipment_anomaly_detection.lineage_paper import LINEAGE_NODES
        from paper.equipment_anomaly_detection.proposal_score_paper import ProposalScore
        from paper.equipment_anomaly_detection.routing_paper import PolicyBundle, PolicyRoute
        from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import EXPECTED_HARD_SOURCES, SOURCE_ALPHA_PAPER, HardSourceBundle, HardSourceResult
        from paper.equipment_anomaly_detection.sources_paper.stability_paper import StabilityPartition

        names = FROZEN_EXP449_CONFIG.smoke_datasets
        blobs = (
            np.asarray((1.0, 1.5), dtype="<f4").tobytes(),
            np.asarray((2.0, 2.5), dtype="<f4").tobytes(),
        )
        rows = tuple(np.frombuffer(blob, dtype="<f4").copy() for blob in blobs)
        raw: dict[str, object] = {}
        catalog_entries: list[dict[str, object]] = []
        for name in names:
            typed_raw = RawDataset(name, 2, (0, 1), blobs, rows, (0, 1), blobs, rows)
            raw[name] = typed_raw
            catalog_entries.append({
                "dataset": name,
                "series_length": typed_raw.metadata_length,
                "train": _identity_antigravity(typed_raw.train_indices, typed_raw.train_blobs),
                "test": _identity_antigravity(typed_raw.test_indices, typed_raw.test_blobs),
            })
        origins = {node.node_id: node for node in LINEAGE_NODES}
        origin_ids = {"rocket_exp40": "rocket_exp40", "exp55_best": "exp55_best", "exp56_best": "exp56_best", "exp84": "exp84_family_guard", "rocket_block_b": "rocket_block_b", "canonical_normal_state": "canonical_normal_state"}
        source_scores = np.asarray((0.1, 0.2))
        source_pvalues = upper_tail_pvalues(source_scores, source_scores)
        sources = HardSourceBundle(EXPECTED_HARD_SOURCES, {
            name: HardSourceResult(name, origins[origin_ids[name]], source_scores, source_scores, source_pvalues, SOURCE_ALPHA_PAPER[name], source_pvalues <= SOURCE_ALPHA_PAPER[name], True, "eligible")
            for name in EXPECTED_HARD_SOURCES
        })
        lanes = RouteLanes(frozenset(), frozenset(), frozenset())
        route = HardRouteResult(lanes, frozenset(), frozenset(), frozenset(), lanes, SourceExpansion(frozenset(), frozenset()), lanes, frozenset(), lanes, frozenset(), frozenset(), 2)
        low = LowFrequencyResult(
            names,
            tuple(LowFrequencyRow(name, index, 2, (), False, None, None, False, True, "no_group") for name in names for index in range(2)),
            (), (), 0, {name: frozenset() for name in names},
        )
        stability = StabilityPartition((), (), (), {})
        policies = PolicyBundle(tuple(
            PolicyRoute(policy, {name: frozenset() for name in names}, {name: np.asarray((0.0, 1.0 / 3.0)) for name in names}, {name: np.asarray((0.0, 0.16666666666666669)) for name in names})
            for policy in ("hard_only", "hard_plus_source", "hard_plus_stable", "hard_plus_unstable")
        ))
        backend = FrozenTask8BackendDecision.from_acceleration_manifest(build_acceleration_manifest("f" * 64, logical_cpu_count=3))
        bare = {"schema": "paper-task8-catalog.v1", "mode": "smoke", "database": {"fixture": "no-db"}, "datasets": catalog_entries}
        self.catalog = {**bare, "catalog_sha256": _catalog_sha256_antigravity(bare)}
        self.score = ProposalScore(raw, {name: sources for name in names}, {name: route for name in names}, low, stability, policies, backend, {"task4_task5": 0.0, "task6": 0.0, "task7": 0.0, "total": 0.0})
        self.context = _context_antigravity(root, self.catalog, backend)


class Exp449ManifestVerifierTests(unittest.TestCase):
    """Each test names a publication or verification break, not implementation text."""

    def setUp(self) -> None:
        self.directory = TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.directory.name) / "exp449_typed_root_antigravity"
        self.fixture = _TypedProposalFixtureAntigravity(self.root)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _freeze(self):
        return freeze_exp449_proposal_antigravity(
            self.fixture.score, root=self.root, context=self.fixture.context, catalog=self.fixture.catalog,
        )

    def _manifest_path(self) -> Path:
        return self.root / "stage_artifacts_antigravity" / "proposal_freeze_antigravity" / "proposal_manifest_antigravity.json"

    @staticmethod
    def _publish_real_task45_checkpoints(fixture: _TypedProposalFixtureAntigravity, root: Path) -> None:
        """Use only the paper test publisher in this temporary fixture root."""
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        from paper.equipment_anomaly_detection.artifacts_paper import (
            Task4Task5SuccessAuthority, append_task4_task5_attempt_record,
            write_task4_task5_success_record,
        )
        from experiments.exp449_reproducible_rebaseline_antigravity.operations_antigravity import task45_context_payload_antigravity
        bridge = task45_context_payload_antigravity(fixture.context)
        authority = Task4Task5SuccessAuthority.from_verified_context(
            bridge, fixture.catalog,
            build_acceleration_manifest(bridge["context_sha256"], logical_cpu_count=3),
        )
        first = fixture.catalog["datasets"][0]["dataset"]
        append_task4_task5_attempt_record(root, fixture.catalog, authority, fixture.score.raw[first], 1, RuntimeError("fixture interruption"))
        for entry in fixture.catalog["datasets"]:
            dataset = entry["dataset"]
            write_task4_task5_success_record(root, fixture.catalog, authority, fixture.score.raw[dataset], fixture.score.sources[dataset], fixture.score.routes[dataset])

    def _rewrite_bound_shard(self, mutate) -> None:
        """Rewrite one shard and every descriptor digest that derives from it.

        The verifier must therefore reach scientific validation; a stale file
        hash or stale coordinate digest is not allowed to make these tests
        pass accidentally.
        """
        manifest_path = self._manifest_path()
        manifest = json.loads(manifest_path.read_text())
        descriptor = manifest["entries"][0]
        path = manifest_path.parent / descriptor["relative_path"]
        shard = json.loads(path.read_text())
        mutate(shard["scientific_payload"])
        data = json.dumps(shard, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        path.write_bytes(data)
        descriptor["size"] = len(data)
        descriptor["sha256"] = sha256(data).hexdigest()
        policies = shard["scientific_payload"]["policies"]
        descriptor["candidate_c_score_sha256"] = sha256(
            struct.pack("<" + "d" * len(policies[2]["scores"]), *policies[2]["scores"])
        ).hexdigest()
        descriptor["policy_coordinate_sha256"] = {
            policy["policy_id"]: sha256(json.dumps(
                [[descriptor["dataset"], index] for index in policy["alerts"]],
                separators=(",", ":"),
            ).encode() + b"\n").hexdigest()
            for policy in policies
        }
        manifest_path.write_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n")

    def _score_with_route(self, route: object, low_frequency: object | None = None):
        """Keep the genuine Task3 typed score while replacing one valid route."""
        from paper.equipment_anomaly_detection.proposal_score_paper import ProposalScore
        return ProposalScore(
            self.fixture.score.raw, self.fixture.score.sources,
            {entry["dataset"]: route for entry in self.fixture.catalog["datasets"]},
            self.fixture.score.low_frequency if low_frequency is None else low_frequency, self.fixture.score.stability,
            self.fixture.score.policies, self.fixture.score.backend,
            self.fixture.score.stage_seconds,
        )

    def test_float_digest_bytes_are_canonical_little_endian_float64(self) -> None:
        """Host byte order or a non-float64 conversion would change Candidate C digests."""
        self.assertEqual(canonical_float64_bytes_antigravity([1.5, -2.25]), struct.pack("<dd", 1.5, -2.25))
        with self.assertRaisesRegex(ValueError, "score vector"):
            canonical_float64_bytes_antigravity([[1.0]])
        with self.assertRaisesRegex(ValueError, "score vector"):
            canonical_float64_bytes_antigravity([float("nan")])

    def test_freeze_and_scorer_free_verifier_bind_the_complete_smoke_inventory(self) -> None:
        """A partial typed-shard set must not freeze as a complete smoke proposal."""
        manifest = self._freeze()
        verified = verify_exp449_proposal_antigravity(self.root, context=self.fixture.context, catalog=self.fixture.catalog)
        self.assertEqual(tuple(entry.dataset for entry in manifest.entries), FROZEN_EXP449_CONFIG.smoke_datasets)
        self.assertEqual(tuple(entry.dataset for entry in verified.entries), FROZEN_EXP449_CONFIG.smoke_datasets)
        self.assertEqual(self._manifest_path().name, "proposal_manifest_antigravity.json")
        self.assertFalse((self.root / "stage_artifacts_paper").exists())

    def test_constructor_binds_a_build_catalog_style_self_hash_before_freeze(self) -> None:
        """Hashing the catalog including its self-hash accepts neither the paper contract nor tamper rejection."""
        self.assertEqual(self.fixture.context.catalog_sha256, self.fixture.catalog["catalog_sha256"])
        self._freeze()
        forged = dict(self.fixture.catalog)
        forged["catalog_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "catalog"):
            _context_antigravity(self.root, forged, self.fixture.score.backend)

    def test_serializer_keeps_the_frozen_paper_scientific_payload_byte_identical(self) -> None:
        """Changing a scientific field while introducing the Exp449 envelope would break compatibility."""
        from paper.equipment_anomaly_detection.artifacts_paper import _typed_payload, canonical_json_bytes

        self._freeze()
        entry = self.fixture.catalog["datasets"][0]
        dataset = entry["dataset"]
        shard = json.loads((self.root / "stage_artifacts_antigravity" / "proposal_freeze_antigravity" / "shards_antigravity" / f"{sha256(dataset.encode()).hexdigest()}_antigravity" / f"{dataset}_proposal_shard_antigravity.json").read_text())
        expected = _typed_payload(
            self.fixture.score.raw[dataset], self.fixture.score.sources[dataset], self.fixture.score.routes[dataset],
            self.fixture.score.low_frequency, self.fixture.score.stability, self.fixture.score.policies,
            self.fixture.score.backend, entry,
        )
        self.assertEqual(canonical_json_bytes(shard["scientific_payload"]), canonical_json_bytes(expected))
        self.assertEqual(shard["schema"], "exp449-proposal-shard-antigravity.v1")

    def test_verifier_rejects_altered_score_alert_and_manifest_hash(self) -> None:
        """Trusting manifest hashes without revalidating typed semantics permits score or alert tampering."""
        self._freeze()
        manifest = json.loads(self._manifest_path().read_text())
        path = self._manifest_path().parent / manifest["entries"][0]["relative_path"]
        original = path.read_bytes()
        shard = json.loads(original)
        shard["scientific_payload"]["policies"][2]["scores"][0] = 0.75
        path.write_text(json.dumps(shard, sort_keys=True, separators=(",", ":")) + "\n")
        with self.assertRaises(ValueError):
            verify_exp449_proposal_antigravity(self.root, context=self.fixture.context, catalog=self.fixture.catalog)
        path.write_bytes(original)
        self._freeze()
        shard = json.loads(path.read_text())
        shard["scientific_payload"]["policies"][2]["alerts"] = [1]
        path.write_text(json.dumps(shard, sort_keys=True, separators=(",", ":")) + "\n")
        with self.assertRaises(ValueError):
            verify_exp449_proposal_antigravity(self.root, context=self.fixture.context, catalog=self.fixture.catalog)

    def test_verifier_rejects_extra_missing_symlink_old_root_and_non_antigravity_names(self) -> None:
        """Namespace checks that accept an undeclared path can be redirected or silently supplemented."""
        self._freeze()
        freeze_root = self._manifest_path().parent
        (freeze_root / "not_antigravity.json").write_text("{}\n")
        with self.assertRaises(ValueError):
            verify_exp449_proposal_antigravity(self.root, context=self.fixture.context, catalog=self.fixture.catalog)
        (freeze_root / "not_antigravity.json").unlink()
        manifest = json.loads(self._manifest_path().read_text())
        path = freeze_root / manifest["entries"][0]["relative_path"]
        path.unlink()
        with self.assertRaises(ValueError):
            verify_exp449_proposal_antigravity(self.root, context=self.fixture.context, catalog=self.fixture.catalog)
        self._freeze()
        path.unlink()
        path.symlink_to(self._manifest_path())
        with self.assertRaises(ValueError):
            verify_exp449_proposal_antigravity(self.root, context=self.fixture.context, catalog=self.fixture.catalog)
        old_root = self.root.parent / "exp448_old_root_antigravity"
        old_context = _context_antigravity(old_root, self.fixture.catalog, self.fixture.score.backend)
        with self.assertRaises(ValueError):
            verify_exp449_proposal_antigravity(old_root, context=old_context, catalog=self.fixture.catalog)

    def test_verifier_rejects_root_siblings_outside_the_declared_exp449_namespace(self) -> None:
        """Verifying only proposal_freeze permits a second paper or Exp448 authority under the same root."""
        self._freeze()
        (self.root / "stage_artifacts_paper").mkdir(parents=True)
        with self.assertRaises(ValueError):
            verify_exp449_proposal_antigravity(self.root, context=self.fixture.context, catalog=self.fixture.catalog)

    def test_freeze_binds_each_legacy_task45_checkpoint_and_rejects_tamper(self) -> None:
        """A real-root Task4/5 namespace must be declared file-by-file, not broadly allowlisted."""
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        from paper.equipment_anomaly_detection.artifacts_paper import Task4Task5SuccessAuthority, write_task4_task5_success_record
        from experiments.exp449_reproducible_rebaseline_antigravity.operations_antigravity import task45_context_payload_antigravity

        task45_context = task45_context_payload_antigravity(self.fixture.context)
        authority = Task4Task5SuccessAuthority.from_verified_context(
            task45_context, self.fixture.catalog,
            build_acceleration_manifest(task45_context["context_sha256"], logical_cpu_count=3),
        )
        for entry in self.fixture.catalog["datasets"]:
            dataset = entry["dataset"]
            write_task4_task5_success_record(
                self.root, self.fixture.catalog, authority, self.fixture.score.raw[dataset],
                self.fixture.score.sources[dataset], self.fixture.score.routes[dataset],
            )
        manifest = self._freeze()
        self.assertEqual(len(manifest.task45_checkpoints), len(FROZEN_EXP449_CONFIG.smoke_datasets))
        verify_exp449_proposal_antigravity(self.root, context=self.fixture.context, catalog=self.fixture.catalog)
        checkpoint = self.root / manifest.task45_checkpoints[0].relative_path
        checkpoint.unlink()
        with self.assertRaisesRegex(ValueError, "(checkpoint|artifact)"):
            verify_exp449_proposal_antigravity(self.root, context=self.fixture.context, catalog=self.fixture.catalog)

    def test_exact_normal_removal_and_outside_exp93_priority_are_valid_typed_authority(self) -> None:
        """The paper route permits a priority coordinate outside Exp93 and removes exact normals after promotion."""
        from paper.equipment_anomaly_detection.hard_route_paper import HardRouteResult, RouteLanes, SourceExpansion
        route = HardRouteResult(
            RouteLanes(frozenset(), frozenset(), frozenset()), frozenset(), frozenset(), frozenset(),
            RouteLanes(frozenset(), frozenset(), frozenset({0})), SourceExpansion(frozenset({1}), frozenset()),
            RouteLanes(frozenset(), frozenset({1}), frozenset({0})), frozenset({1}),
            RouteLanes(frozenset({1}), frozenset(), frozenset({0})), frozenset({1}), frozenset(), 2,
        )
        from paper.equipment_anomaly_detection.contracts_paper import LowFrequencyResult
        low = self.fixture.score.low_frequency
        score = self._score_with_route(route, LowFrequencyResult(
            low.datasets, low.rows, low.provisional_candidates, low.selected_candidates,
            low.excluded_train_rows, {entry["dataset"]: frozenset({1}) for entry in self.fixture.catalog["datasets"]},
        ))
        # The publisher records must use the same route as the shard score.
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        from paper.equipment_anomaly_detection.artifacts_paper import Task4Task5SuccessAuthority, append_task4_task5_attempt_record, write_task4_task5_success_record
        from experiments.exp449_reproducible_rebaseline_antigravity.operations_antigravity import task45_context_payload_antigravity
        bridge = task45_context_payload_antigravity(self.fixture.context)
        authority = Task4Task5SuccessAuthority.from_verified_context(bridge, self.fixture.catalog, build_acceleration_manifest(bridge["context_sha256"], logical_cpu_count=3))
        first = self.fixture.catalog["datasets"][0]["dataset"]
        append_task4_task5_attempt_record(self.root, self.fixture.catalog, authority, score.raw[first], 1, RuntimeError("fixture interruption"))
        for entry in self.fixture.catalog["datasets"]:
            dataset = entry["dataset"]
            write_task4_task5_success_record(self.root, self.fixture.catalog, authority, score.raw[dataset], score.sources[dataset], route)
        manifest = freeze_exp449_proposal_antigravity(score, root=self.root, context=self.fixture.context, catalog=self.fixture.catalog)
        self.assertTrue(manifest.task45_checkpoints)
        verify_exp449_proposal_antigravity(self.root, context=self.fixture.context, catalog=self.fixture.catalog)
        # Refresh the success envelope's payload hash, its manifest descriptor,
        # and the outer manifest after an algebra-only checkpoint mutation.
        manifest_path = self._manifest_path()
        outer = json.loads(manifest_path.read_text())
        descriptor = next(item for item in outer["task45_checkpoints"] if item["schema"] == "paper-task4-task5-success.v1")
        checkpoint = self.root / descriptor["relative_path"]
        record = json.loads(checkpoint.read_text())
        record["payload"]["route"]["automatic_hard"] = [0]
        record["payload_sha256"] = sha256(json.dumps(record["payload"], sort_keys=True, separators=(",", ":")).encode() + b"\n").hexdigest()
        raw = json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        checkpoint.write_bytes(raw)
        descriptor["size"] = len(raw)
        descriptor["sha256"] = sha256(raw).hexdigest()
        manifest_path.write_bytes(json.dumps(outer, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        with self.assertRaisesRegex(ValueError, "(route algebra|policy alerts)"):
            verify_exp449_proposal_antigravity(self.root, context=self.fixture.context, catalog=self.fixture.catalog)

    def test_coordinated_candidate_c_tamper_still_fails_semantic_validation(self) -> None:
        """Refreshing every digest must not authorize Candidate C alerts that disagree with scores and masks."""
        self._freeze()
        manifest_path = self._manifest_path()
        manifest = json.loads(manifest_path.read_text())
        path = manifest_path.parent / manifest["entries"][0]["relative_path"]
        shard = json.loads(path.read_text())
        scientific = shard["scientific_payload"]
        scientific["hard_stages"][-1]["lanes"]["hard"] = [0]
        for policy in scientific["policies"]:
            policy["alerts"] = [0]
            policy["scores"] = [0.75, 0.0]
            policy["score_mask"] = [True, False]
        scientific["policies"][2]["score_mask"] = [False, False]
        bytes_ = json.dumps(shard, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        path.write_bytes(bytes_)
        descriptor = manifest["entries"][0]
        descriptor["size"] = len(bytes_)
        descriptor["sha256"] = sha256(bytes_).hexdigest()
        descriptor["candidate_c_score_sha256"] = sha256(struct.pack("<dd", 0.75, 0.0)).hexdigest()
        for policy in scientific["policies"]:
            descriptor["policy_coordinate_sha256"][policy["policy_id"]] = sha256(json.dumps([[descriptor["dataset"], 0]], separators=(",", ":")).encode() + b"\n").hexdigest()
        manifest_path.write_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        with self.assertRaisesRegex(ValueError, "(continuous score|score mask)"):
            verify_exp449_proposal_antigravity(self.root, context=self.fixture.context, catalog=self.fixture.catalog)

    def test_coordinated_source_low_and_route_tampers_reach_pure_semantic_checks(self) -> None:
        """Rebound descriptors cannot hide malformed typed Task4--7 evidence."""
        cases = (
            (
                "source TEST score",
                lambda payload: payload["source_rows"][0].__setitem__("test_score", [0.1]),
            ),
            (
                "source TEST score domain",
                lambda payload: payload["source_rows"][0].__setitem__("test_score", ["not-a-score", 0.2]),
            ),
            (
                "source pvalue domain/native-pass",
                lambda payload: payload["source_rows"][0].update({"pvalue": [1.25, 0.8], "native_pass": [True, False]}),
            ),
            (
                "low-frequency evidence",
                lambda payload: payload["low_frequency_rows"][0].__setitem__("depths", [{"depth": 0}]),
            ),
            (
                "hard-stage route",
                lambda payload: payload["hard_stages"][6]["lanes"].__setitem__("priority", [0]),
            ),
            (
                "selected row",
                lambda payload: payload["stability"].update({
                    "selected": [[payload["dataset"], 0]],
                    "stable": [[payload["dataset"], 0]],
                }),
            ),
        )
        for label, mutate in cases:
            with self.subTest(label=label):
                # Each create-only publication starts from a fresh owned temp
                # root, so a prior intentionally tampered shard cannot mask
                # a later semantic assertion.
                if self.root.exists():
                    shutil.rmtree(self.root)
                self.fixture = _TypedProposalFixtureAntigravity(self.root)
                self._freeze()
                self._rewrite_bound_shard(mutate)
                with self.assertRaisesRegex(ValueError, "Exp449"):
                    verify_exp449_proposal_antigravity(self.root, context=self.fixture.context, catalog=self.fixture.catalog)

    def test_coordinated_continuous_policy_score_drift_fails_pure_recomputation(self) -> None:
        """A score below .5 can retain its mask while still changing frozen evidence."""
        self._freeze()
        self._rewrite_bound_shard(
            lambda payload: payload["policies"][0]["scores"].__setitem__(0, 0.2)
        )
        with self.assertRaisesRegex(ValueError, "Exp449"):
            verify_exp449_proposal_antigravity(
                self.root, context=self.fixture.context, catalog=self.fixture.catalog,
            )

    def test_global_candidate_c_rejects_omitted_eligible_coordinate_after_rebinding(self) -> None:
        """A valid Task6 local winner omitted from Task7 must fail only at population closure."""
        self._freeze()
        manifest_path = self._manifest_path(); outer = json.loads(manifest_path.read_text())
        descriptor = outer["entries"][0]; path = manifest_path.parent / descriptor["relative_path"]
        shard = json.loads(path.read_text()); scientific = shard["scientific_payload"]
        scientific["low_frequency_rows"][0] = {
            "index": 0, "evidence_missing": False, "strict_pass": True, "provisional": True,
            "depths": [
                {"depth": 1, "score": 2.0, "threshold": 1.0, "strict_pass": True, "excess_ratio": 2.0, "empirical_rank": 0.2},
                {"depth": 2, "score": 2.0, "threshold": 1.0, "strict_pass": True, "excess_ratio": 2.0, "empirical_rank": 0.3},
            ],
        }
        # No policy alerts are added: this is a locally valid but globally
        # omitted Task6 candidate. All four continuous vectors are recomputed.
        for policy in scientific["policies"]:
            policy["scores"] = [0.35, 0.16666666666666669]
            policy["score_mask"] = [False, False]
            policy["alerts"] = []
        data = json.dumps(shard, sort_keys=True, separators=(",", ":")).encode() + b"\n"; path.write_bytes(data)
        descriptor["size"] = len(data); descriptor["sha256"] = sha256(data).hexdigest()
        descriptor["candidate_c_score_sha256"] = sha256(struct.pack("<dd", 0.35, 0.16666666666666669)).hexdigest()
        descriptor["policy_coordinate_sha256"] = {policy["policy_id"]: sha256(b"[]\n").hexdigest() for policy in scientific["policies"]}
        manifest_path.write_bytes(json.dumps(outer, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        with self.assertRaisesRegex(ValueError, "global Candidate C"):
            verify_exp449_proposal_antigravity(self.root, context=self.fixture.context, catalog=self.fixture.catalog)

    def test_candidate_c_rejects_rebound_ineligible_selected_coordinate(self) -> None:
        """An added selected coordinate without Task6 evidence cannot enter Candidate C."""
        self._freeze()
        manifest_path = self._manifest_path(); outer = json.loads(manifest_path.read_text())
        descriptor = outer["entries"][0]; path = manifest_path.parent / descriptor["relative_path"]
        shard = json.loads(path.read_text()); scientific = shard["scientific_payload"]; dataset = descriptor["dataset"]
        scientific["stability"] = {"selected": [[dataset, 0]], "stable": [[dataset, 0]], "unstable": []}
        for policy in scientific["policies"]:
            selected = policy["policy_id"] in {"hard_plus_source", "hard_plus_stable"}
            policy["alerts"] = [0] if selected else []
            policy["scores"] = [0.5 if selected else 0.0, 0.16666666666666669]
            policy["score_mask"] = [selected, False]
        data = json.dumps(shard, sort_keys=True, separators=(",", ":")).encode() + b"\n"; path.write_bytes(data)
        descriptor["size"] = len(data); descriptor["sha256"] = sha256(data).hexdigest()
        descriptor["candidate_c_score_sha256"] = sha256(struct.pack("<dd", 0.5, 0.16666666666666669)).hexdigest()
        descriptor["policy_coordinate_sha256"] = {policy["policy_id"]: sha256(json.dumps([[dataset, 0]] if policy["alerts"] else [], separators=(",", ":")).encode() + b"\n").hexdigest() for policy in scientific["policies"]}
        manifest_path.write_bytes(json.dumps(outer, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        with self.assertRaisesRegex(ValueError, "selected low-frequency"):
            verify_exp449_proposal_antigravity(self.root, context=self.fixture.context, catalog=self.fixture.catalog)

    def test_full_inventory_is_checked_structurally_without_scoring_1117_datasets(self) -> None:
        """A full catalog with a wrong count or total must fail before any score object is requested."""
        rows = [{"dataset": f"dataset_{number:04d}_antigravity", "series_length": 2, "train": {}, "test": {"count": 175}} for number in range(1117)]
        rows[-1]["test"]["count"] = 600
        bare = {"schema": "paper-task8-catalog.v1", "mode": "full", "database": {"fixture": "no-db"}, "datasets": rows}
        catalog = {**bare, "catalog_sha256": _catalog_sha256_antigravity(bare)}
        descriptors = tuple(Exp449ShardDescriptor(
            dataset=row["dataset"], relative_path=f"shards_antigravity/{sha256(row['dataset'].encode()).hexdigest()}_antigravity/{row['dataset']}_proposal_shard_antigravity.json",
            size=1, sha256="a" * 64, schema="exp449-proposal-shard-antigravity.v1",
            policy_coordinate_sha256={name: "b" * 64 for name in ("hard_only", "hard_plus_source", "hard_plus_stable", "hard_plus_unstable")}, candidate_c_score_sha256="c" * 64,
        ) for row in rows)
        manifest = Exp449ProposalManifest("exp449-proposal-manifest-antigravity.v1", "d" * 64, catalog["catalog_sha256"], self.fixture.context.backend_record, descriptors, "not_used")
        self.assertEqual(validate_exp449_catalog_antigravity(catalog, mode="official", manifest=manifest), tuple(row["dataset"] for row in rows))
        with self.assertRaises(ValueError):
            validate_exp449_catalog_antigravity({**catalog, "datasets": rows[:-1]}, mode="official", manifest=manifest)

    def test_resume_preserves_complete_shard_and_manifest_bytes_after_only_context_root_normalization(self) -> None:
        """A resumed create-only publication must preserve all science; context_sha256 is the sole root-bound exclusion."""
        uninterrupted_root = self.root.parent / "uninterrupted_antigravity"
        resumed_root = self.root.parent / "resumed_antigravity"
        uninterrupted = _TypedProposalFixtureAntigravity(uninterrupted_root)
        resumed = _TypedProposalFixtureAntigravity(resumed_root)
        self._publish_real_task45_checkpoints(uninterrupted, uninterrupted_root)
        self._publish_real_task45_checkpoints(resumed, resumed_root)
        freeze_exp449_proposal_antigravity(uninterrupted.score, root=uninterrupted_root, context=uninterrupted.context, catalog=uninterrupted.catalog)
        from experiments.exp449_reproducible_rebaseline_antigravity import artifacts_antigravity as artifacts
        real_write = artifacts._write_create_only_antigravity
        calls = {"count": 0}
        def fail_once(path: Path, data: bytes, *, reuse_identical: bool = False) -> None:
            calls["count"] += 1
            if calls["count"] == 8:
                raise OSError("injected interrupted shard publication")
            real_write(path, data, reuse_identical=reuse_identical)
        with patch.object(artifacts, "_write_create_only_antigravity", side_effect=fail_once):
            with self.assertRaisesRegex(OSError, "interrupted"):
                freeze_exp449_proposal_antigravity(resumed.score, root=resumed_root, context=resumed.context, catalog=resumed.catalog)
        freeze_exp449_proposal_antigravity(resumed.score, root=resumed_root, context=resumed.context, catalog=resumed.catalog)

        def normalized(path: Path) -> bytes:
            value = json.loads(path.read_text())
            # context_sha256 exists only on the outer manifest: it binds the
            # distinct run root.  Shards deliberately contain no root data.
            if "context_sha256" in value:
                value["context_sha256"] = "0" * 64
            # The outer manifest binds the root-bound paper checkpoints by
            # byte descriptor. Normalize each referenced envelope first, then
            # rebind its descriptor before comparing scientific manifests.
            if "task45_checkpoints" in value:
                for descriptor in value["task45_checkpoints"]:
                    checkpoint = path.parent.parent.parent / descriptor["relative_path"]
                    record = json.loads(checkpoint.read_text())
                    record["context_sha256"] = "0" * 64
                    # The paper acceleration assessment binds the same Task4/5
                    # bridge context, so its digest is operational/root-bound
                    # alongside the envelope context hash.
                    if isinstance(record.get("backend"), dict) and "assessment_sha256" in record["backend"]:
                        record["backend"]["assessment_sha256"] = "0" * 64
                    if "record_sha256" in record:
                        bare = {key: child for key, child in record.items() if key != "record_sha256"}
                        record["record_sha256"] = sha256(json.dumps(bare, sort_keys=True, separators=(",", ":")).encode() + b"\n").hexdigest()
                    raw = json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
                    descriptor["size"] = len(raw)
                    descriptor["sha256"] = sha256(raw).hexdigest()
            return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        left_root = uninterrupted_root / "stage_artifacts_antigravity" / "proposal_freeze_antigravity"
        right_root = resumed_root / "stage_artifacts_antigravity" / "proposal_freeze_antigravity"
        self.assertEqual(normalized(left_root / "proposal_manifest_antigravity.json"), normalized(right_root / "proposal_manifest_antigravity.json"))
        for left in sorted((left_root / "shards_antigravity").rglob("*_proposal_shard_antigravity.json")):
            relative = left.relative_to(left_root)
            self.assertEqual(normalized(left), normalized(right_root / relative))

    def test_verifier_fresh_import_denies_paper_scoring_sources_and_filesystem_writer(self) -> None:
        """A verifier that imports scoring or labels is not independent of proposal production."""
        verifier_path = Path("experiments/exp449_reproducible_rebaseline_antigravity/verifier_antigravity.py")
        imported = {
            alias.name
            for node in ast.walk(ast.parse(verifier_path.read_text()))
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(ast.parse(verifier_path.read_text()))
            if isinstance(node, ast.ImportFrom)
        }
        forbidden = ("proposal_score_paper", "sources_paper", "hard_route_paper", "artifacts_paper", "exp448", "label_access", "catalog_paper")
        self.assertFalse({name for name in imported if any(token in name for token in forbidden)})
        child = """
import builtins
real = builtins.__import__
def deny(name, *args, **kwargs):
    if any(token in name for token in ('artifacts_paper', 'proposal_score_paper', 'sources_paper', 'hard_route_paper', 'label_access')):
        raise AssertionError(name)
    return real(name, *args, **kwargs)
builtins.__import__ = deny
from experiments.exp449_reproducible_rebaseline_antigravity.verifier_antigravity import verify_exp449_proposal_antigravity
print('fresh-import-ok')
"""
        result = subprocess.run([sys.executable, "-c", child], text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "fresh-import-ok")


if __name__ == "__main__":
    unittest.main()
