"""Behavioral Task 5 tests for the Exp449-only proposal freeze boundary."""

from __future__ import annotations

import ast
from contextlib import ExitStack
from hashlib import sha256
import json
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp449_reproducible_rebaseline_antigravity.config_antigravity import (
    FROZEN_EXP449_CONFIG,
)
from experiments.exp449_reproducible_rebaseline_antigravity.contracts_antigravity import (
    Exp449Context,
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


def _context_antigravity(root: Path, backend_record: dict[str, object]) -> Exp449Context:
    """Use a typed in-memory context; configured roots are never touched by this test."""
    context = object.__new__(Exp449Context)
    for key, value in {
        "experiment_id": "Exp449",
        "mode": "smoke",
        "run_root": root,
        "database_sha256": "a" * 64,
        "catalog_sha256": "b" * 64,
        "code_closure": {"fixture_antigravity.py": "c" * 64},
        "lineage_sha256": "d" * 64,
        "codemap_sha256": "e" * 64,
        "interpreter": "typed-fixture-python",
        "environment": {"fixture": "typed"},
        "backend_record": backend_record,
        "seeds_and_constants": {"seed": 7},
        "historical_parity_status": "not_used",
    }.items():
        object.__setattr__(context, key, value)
    return context


class _TypedProposalFixtureAntigravity:
    """A real paper-type ProposalScore, scaled only by reusing typed values in memory."""

    def __init__(self, root: Path) -> None:
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        from paper.equipment_anomaly_detection.contracts_paper import FrozenTask8BackendDecision, LowFrequencyResult, LowFrequencyRow, RawDataset
        from paper.equipment_anomaly_detection.hard_route_paper import HardRouteResult, RouteLanes, SourceExpansion
        from paper.equipment_anomaly_detection.lineage_paper import LINEAGE_NODES
        from paper.equipment_anomaly_detection.proposal_score_paper import ProposalScore
        from paper.equipment_anomaly_detection.routing_paper import PolicyBundle, PolicyRoute
        from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import EXPECTED_HARD_SOURCES, HardSourceBundle, HardSourceResult
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
        sources = HardSourceBundle(EXPECTED_HARD_SOURCES, {
            name: HardSourceResult(name, LINEAGE_NODES[0], np.asarray((0.1, 0.2)), np.asarray((0.1, 0.2)), np.asarray((0.9, 0.8)), 0.03, np.asarray((False, False)), True, "eligible")
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
            PolicyRoute(policy, {name: frozenset() for name in names}, {name: np.zeros(2) for name in names}, {name: np.zeros(2) for name in names})
            for policy in ("hard_only", "hard_plus_source", "hard_plus_stable", "hard_plus_unstable")
        ))
        backend = FrozenTask8BackendDecision.from_acceleration_manifest(build_acceleration_manifest("f" * 64, logical_cpu_count=3))
        self.catalog = {"catalog_sha256": "b" * 64, "mode": "smoke", "datasets": catalog_entries}
        self.score = ProposalScore(raw, {name: sources for name in names}, {name: route for name in names}, low, stability, policies, backend, {"task4_task5": 0.0, "task6": 0.0, "task7": 0.0, "total": 0.0})
        self.context = _context_antigravity(root, backend.artifact_record())


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
        old_context = _context_antigravity(old_root, dict(self.fixture.context.backend_record))
        with self.assertRaises(ValueError):
            verify_exp449_proposal_antigravity(old_root, context=old_context, catalog=self.fixture.catalog)

    def test_full_inventory_is_checked_structurally_without_scoring_1117_datasets(self) -> None:
        """A full catalog with a wrong count or total must fail before any score object is requested."""
        rows = [{"dataset": f"dataset_{number:04d}_antigravity", "series_length": 2, "train": {}, "test": {"count": 175}} for number in range(1117)]
        rows[-1]["test"]["count"] = 600
        catalog = {"catalog_sha256": "f" * 64, "mode": "full", "datasets": rows}
        self.assertEqual(validate_exp449_catalog_antigravity(catalog, mode="official"), tuple(row["dataset"] for row in rows))
        with self.assertRaises(ValueError):
            validate_exp449_catalog_antigravity({**catalog, "datasets": rows[:-1]}, mode="official")

    def test_resume_preserves_complete_shard_and_manifest_bytes_after_only_context_root_normalization(self) -> None:
        """A resumed create-only publication must preserve all science; context_sha256 is the sole root-bound exclusion."""
        uninterrupted_root = self.root.parent / "uninterrupted_antigravity"
        resumed_root = self.root.parent / "resumed_antigravity"
        uninterrupted = _TypedProposalFixtureAntigravity(uninterrupted_root)
        resumed = _TypedProposalFixtureAntigravity(resumed_root)
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
            return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        left_root = uninterrupted_root / "stage_artifacts_antigravity" / "proposal_freeze_antigravity"
        right_root = resumed_root / "stage_artifacts_antigravity" / "proposal_freeze_antigravity"
        self.assertEqual(normalized(left_root / "proposal_manifest_antigravity.json"), normalized(right_root / "proposal_manifest_antigravity.json"))
        for left in sorted((left_root / "shards_antigravity").rglob("*_proposal_shard_antigravity.json")):
            relative = left.relative_to(left_root)
            self.assertEqual(normalized(left), normalized(right_root / relative))

    def test_verifier_imports_no_scoring_or_label_capability(self) -> None:
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
        self._freeze()
        original_import = __import__
        def deny_labels(name, *args, **kwargs):
            if "label_access" in name:
                raise AssertionError("label capability imported")
            return original_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=deny_labels):
            verify_exp449_proposal_antigravity(self.root, context=self.fixture.context, catalog=self.fixture.catalog)


if __name__ == "__main__":
    unittest.main()
