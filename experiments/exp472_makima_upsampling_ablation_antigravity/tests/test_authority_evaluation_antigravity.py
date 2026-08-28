"""Production-path authority and label-firewall tests for Exp472 Task 5."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np

from paper.equipment_anomaly_detection.contracts_paper import RawDataset
from experiments.exp471_exp467_evaluation_recovery_antigravity import cache_artifacts_antigravity as exp471_cache
from experiments.exp471_exp467_evaluation_recovery_antigravity import source_authority_antigravity as exp471_source
from experiments.exp472_makima_upsampling_ablation_antigravity import authority_antigravity as authority_module
from experiments.exp472_makima_upsampling_ablation_antigravity import evaluation_antigravity as evaluation_module
from experiments.exp472_makima_upsampling_ablation_antigravity.artifacts_antigravity import Exp472CatalogEntryAntigravity, catalog_sha256_exp472_antigravity
from experiments.exp472_makima_upsampling_ablation_antigravity.authority_antigravity import Exp472ReuseAuthorityAntigravity, Exp472VerifiedExp471BundleAntigravity, verify_exp471_reuse_exp472_antigravity
from experiments.exp472_makima_upsampling_ablation_antigravity.config_antigravity import EXP472_OFFICIAL_ROOT_ANTIGRAVITY
from experiments.exp472_makima_upsampling_ablation_antigravity.contracts_antigravity import Exp472BackendDecisionAntigravity, Exp472RunContextAntigravity, canonical_json_bytes_exp472_antigravity
from experiments.exp472_makima_upsampling_ablation_antigravity.evaluation_antigravity import Exp472VerifiedEBundleAntigravity, evaluate_exp472_antigravity
from experiments.exp472_makima_upsampling_ablation_antigravity.transforms_antigravity import test_identity_exp472_antigravity

_ROOT = Path("/private/tmp/exp472-task5-fixture-antigravity")
_METHODS = ("A", "B", "C", "D", "knn_distance", "lof_novelty", "ocsvm_rbf", "isolation_forest", "deep_svdd_1d", "drocc_1d")

def _digest(value: str) -> str: return sha256(value.encode()).hexdigest()

def _identity(dataset: str, index: int) -> dict[str, object]:
    bare = {"dataset": dataset, "series_length": 2, "split": "TEST", "indices": [2 * index, 2 * index + 1], "values_blob_sha256": [_digest(f"{dataset}:blob:0"), _digest(f"{dataset}:blob:1")]}
    return {**bare, "row_digest": sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest(), "count": 2}

def _bundle() -> Exp472VerifiedExp471BundleAntigravity:
    datasets = tuple(f"dataset_{index:04d}" for index in range(1117)); coordinates = {name: _identity(name, index) for index, name in enumerate(datasets)}
    scores: dict[str, dict[str, tuple[float, ...] | None]] = {}; statuses: dict[str, dict[str, str]] = {}; abstentions = 0
    for method_index, method in enumerate(_METHODS):
        scores[method], statuses[method] = {}, {}
        for index, name in enumerate(datasets):
            abstained = (index < 195 and method_index < 2) or (index < 5 and method_index == 2)
            statuses[method][name] = "abstained" if abstained else "scored"; scores[method][name] = None if abstained else (0.1, 0.9); abstentions += abstained
    assert abstentions == 395
    coordinate_sha = authority_module._coordinate_digest(datasets, coordinates)
    authority = Exp472ReuseAuthorityAntigravity(_digest("source"), _digest("cache"), _digest("evaluation"), _digest("report"), _digest("database"), coordinate_sha, _METHODS, 11170, 10775, 395, 0, 1117, 922)
    return Exp472VerifiedExp471BundleAntigravity(_ROOT, authority, datasets, coordinates, scores, statuses, _digest("path-bytes"))

def _context_and_catalog(bundle: Exp472VerifiedExp471BundleAntigravity) -> tuple[Exp472RunContextAntigravity, tuple[Exp472CatalogEntryAntigravity, ...]]:
    catalog = tuple(Exp472CatalogEntryAntigravity(dataset, bundle.coordinates[dataset]) for dataset in bundle.dataset_order)
    backend = Exp472BackendDecisionAntigravity(False, "unavailable", True, "test", False, "unavailable", False, "unavailable", False, "cpu", 2, 1)
    context = Exp472RunContextAntigravity("full", EXP472_OFFICIAL_ROOT_ANTIGRAVITY, bundle.authority.database_sha256, catalog_sha256_exp472_antigravity(catalog), {"experiments/exp472_task5_antigravity.py": _digest("closure")}, backend, _digest("config"))
    return context, catalog

def _manifest(bundle: Exp472VerifiedExp471BundleAntigravity, context: Exp472RunContextAntigravity) -> dict[str, object]:
    return {"database_sha256": bundle.authority.database_sha256, "catalog_sha256": context.catalog_sha256, "dataset_order": list(bundle.dataset_order), "descriptors": [{"dataset": dataset, "shard_sha256": _digest(f"E:{dataset}")} for dataset in bundle.dataset_order]}

def _shard(entry: Exp472CatalogEntryAntigravity, *, upsampled: bool = True) -> dict[str, object]:
    return {"dataset": entry.dataset, "shard_sha256": _digest(f"E:{entry.dataset}"), "test_identity": dict(entry.test_identity), "scores": [0.2, 0.8], "upsampled_rows": 1 if upsampled else 0}

class Exp472AuthorityEvaluationTestsAntigravity(unittest.TestCase):
    def test_catalog_projection_matches_exp472_raw_test_identity(self) -> None:
        """Adding catalog-only dataset_id to the projected digest would break E binding."""
        row = np.asarray([1.0, 2.0], dtype=np.float32)
        raw = RawDataset("fixture", 2, (0,), (row.astype("<f4").tobytes(),), (row,), (0,), (row.astype("<f4").tobytes(),), (row,))
        exp472_identity = dict(test_identity_exp472_antigravity(raw))
        projected = authority_module._project_catalog_test_identity_antigravity({**exp472_identity, "dataset_id": 13})
        self.assertEqual(canonical_json_bytes_exp472_antigravity(projected), canonical_json_bytes_exp472_antigravity(exp472_identity))

    def test_path_verifier_rejects_nonfull_terminal_inventory(self) -> None:
        """A dishonest small bundle must be rejected before any possible label access."""
        with self.assertRaises(ValueError): Exp472VerifiedExp471BundleAntigravity(_ROOT, _bundle().authority, ("small",), {}, {}, {}, _digest("path"))

    def test_exp471_path_adapter_reads_public_source_catalog(self) -> None:
        """Replacing Exp471's public source/cache readers must not bypass their path adapter."""
        bundle = _bundle(); source_authority = SimpleNamespace(authority_sha256=bundle.authority.exp471_source_authority_sha256, database_sha256=bundle.authority.database_sha256)
        catalog_rows = [{"dataset": dataset, "test": {**dict(bundle.coordinates[dataset]), "dataset_id": index}} for index, dataset in enumerate(bundle.dataset_order)]
        session = SimpleNamespace(authority=source_authority, catalog={"datasets": catalog_rows})
        manifest = SimpleNamespace(manifest_sha256=bundle.authority.exp471_cache_manifest_sha256, dataset_order=bundle.dataset_order, method_order=_METHODS, total_shards=11170, scored_shards=10775, abstained_shards=395, failed_shards=0)
        def shard(_root: Path, method: str, dataset: str):
            coordinate = bundle.coordinates[dataset]; status = bundle.statuses[method][dataset]
            return SimpleNamespace(test_count=coordinate["count"], test_indices=coordinate["indices"], status=status, raw_scores=() if status == "abstained" else bundle.scores[method][dataset])
        evaluation = {"source_authority_sha256": source_authority.authority_sha256, "cache_manifest_sha256": manifest.manifest_sha256, "evaluation_sha256": bundle.authority.exp471_evaluation_sha256}; report = {"evaluation_sha256": evaluation["evaluation_sha256"], "report_manifest_sha256": bundle.authority.exp471_report_manifest_sha256}
        with patch.object(exp471_source, "open_exp467_source_session_antigravity", return_value=session) as open_session, patch.object(exp471_source, "read_source_authority_antigravity", return_value=source_authority) as read_source, patch.object(exp471_cache, "read_cache_manifest_antigravity", return_value=manifest), patch.object(exp471_cache, "read_score_shard_antigravity", side_effect=shard), patch.object(authority_module, "_canonical_leaf", side_effect=(evaluation, report)):
            verified = verify_exp471_reuse_exp472_antigravity(_ROOT)
        self.assertEqual(verified.coordinates, bundle.coordinates); self.assertEqual(open_session.call_args.args[0], Path(authority_module.__file__).resolve().parents[2]); self.assertEqual(read_source.call_args.args[0], _ROOT)

    def test_blob_coordinate_drift_fails_before_labels(self) -> None:
        """Same count/indices with changed TEST blob evidence must not reach labels."""
        bundle = _bundle(); context, catalog = _context_and_catalog(bundle); calls: list[object] = []
        def drifted(entry: Exp472CatalogEntryAntigravity) -> dict[str, object]:
            value = _shard(entry); identity = dict(value["test_identity"]); identity["values_blob_sha256"] = [identity["values_blob_sha256"][0], _digest("changed-blob")]; value["test_identity"] = identity; return value
        with patch.object(evaluation_module, "verify_exp471_reuse_exp472_antigravity", return_value=bundle), patch.object(evaluation_module, "verify_manifest_exp472_antigravity", return_value=_manifest(bundle, context)), patch.object(evaluation_module, "verify_score_shard_exp472_antigravity", side_effect=lambda _root, _context, entry: drifted(entry) if entry.dataset == bundle.dataset_order[0] else _shard(entry)):
            with self.assertRaises(ValueError): evaluate_exp472_antigravity(_ROOT, _ROOT, lambda _order: calls.append(_order) or {}, context=context, catalog=catalog)
        self.assertEqual(calls, [])

    def test_manifest_and_shard_failures_open_no_labels(self) -> None:
        """Either public Task4 verification failure must stay on the label-free side."""
        bundle = _bundle(); context, catalog = _context_and_catalog(bundle); calls: list[object] = []; labels = lambda _order: calls.append(_order) or {}
        with patch.object(evaluation_module, "verify_exp471_reuse_exp472_antigravity", return_value=bundle), patch.object(evaluation_module, "verify_manifest_exp472_antigravity", side_effect=ValueError("bad manifest")):
            with self.assertRaisesRegex(ValueError, "bad manifest"): evaluate_exp472_antigravity(_ROOT, _ROOT, labels, context=context, catalog=catalog)
        with patch.object(evaluation_module, "verify_exp471_reuse_exp472_antigravity", return_value=bundle), patch.object(evaluation_module, "verify_manifest_exp472_antigravity", return_value=_manifest(bundle, context)), patch.object(evaluation_module, "verify_score_shard_exp472_antigravity", side_effect=ValueError("bad shard")):
            with self.assertRaisesRegex(ValueError, "bad shard"): evaluate_exp472_antigravity(_ROOT, _ROOT, labels, context=context, catalog=catalog)
        self.assertEqual(calls, [])

    def test_verified_e_bundle_freezes_scores_and_coordinates(self) -> None:
        """Mutating caller-owned E maps after verification must not alter bound evidence."""
        bundle = _bundle(); coordinates = {dataset: dict(value) for dataset, value in bundle.coordinates.items()}; scores = {dataset: (0.2, 0.8) for dataset in bundle.dataset_order}; upsampled = {dataset: 1 for dataset in bundle.dataset_order}
        evidence = Exp472VerifiedEBundleAntigravity(_ROOT, bundle.authority.database_sha256, bundle.dataset_order, coordinates, scores, upsampled, _digest("E immutable"))
        coordinates[bundle.dataset_order[0]]["indices"] = (9, 10); scores[bundle.dataset_order[0]] = (9.0, 10.0); upsampled[bundle.dataset_order[0]] = 0
        self.assertEqual(evidence.scores[bundle.dataset_order[0]], (0.2, 0.8)); self.assertEqual(evidence.upsampled_rows[bundle.dataset_order[0]], 1)
        with self.assertRaises(TypeError): evidence.scores[bundle.dataset_order[0]] = (9.0, 10.0)

    def test_real_e_adapter_calls_public_verifiers_then_opens_labels_once(self) -> None:
        """Bypassing Task4 manifest or a descriptor verifier would admit unbound E scores."""
        bundle = _bundle(); context, catalog = _context_and_catalog(bundle); calls: list[tuple[str, ...]] = []
        def labels(order: tuple[str, ...]): calls.append(order); return {dataset: (0, 1) for dataset in order}
        with patch.object(evaluation_module, "verify_exp471_reuse_exp472_antigravity", return_value=bundle), patch.object(evaluation_module, "verify_manifest_exp472_antigravity", return_value=_manifest(bundle, context)) as verify_manifest, patch.object(evaluation_module, "verify_score_shard_exp472_antigravity", side_effect=lambda _root, _context, entry: _shard(entry, upsampled=entry.dataset != bundle.dataset_order[0])) as verify_shard:
            result = evaluate_exp472_antigravity(_ROOT, _ROOT, labels, context=context, catalog=catalog)
        self.assertEqual(calls, [bundle.dataset_order]); self.assertEqual((verify_manifest.call_count, verify_shard.call_count), (1, 1117)); self.assertEqual(result.method_order, ("A", "B", "C", "D", "E", *_METHODS[4:])); self.assertEqual((result.curves["roc"]["x_name"], result.curves["pr"]["x_name"]), ("false_positive_rate", "recall"))
        self.assertEqual(len(result.subset_per_dataset), 1116 * len(result.method_order))
        self.assertEqual(result.subset_micro["E"]["dataset_denominator"], 1116)
        self.assertEqual(result.denominators["upsampled_rows_gt_zero"]["observation_count"], 2232)
        self.assertEqual({item["population"] for item in result.paired_statistics}, {"full", "upsampled_rows_gt_zero"})
        for paired in result.paired_statistics: self.assertEqual((paired["seed"], paired["draw_count"], paired["percentile"], paired["tie_tolerance"]), (47220260828, 20_000, 95.0, 1e-12))
        record = result.artifact_record(); self.assertEqual(record["schema"], "exp472-evaluation-antigravity.v1"); self.assertEqual(record["evaluation_sha256"], sha256(canonical_json_bytes_exp472_antigravity({key: value for key, value in record.items() if key != "evaluation_sha256"})).hexdigest()); self.assertNotEqual(record["evaluation_sha256"], replace(result, method_order=("changed",)).artifact_record()["evaluation_sha256"])

    def test_evaluation_digest_covers_every_scientific_section(self) -> None:
        """Omitting any scientific table from the digest would permit an unbound Task6 report."""
        bundle = _bundle(); context, catalog = _context_and_catalog(bundle)
        with patch.object(evaluation_module, "verify_exp471_reuse_exp472_antigravity", return_value=bundle), patch.object(evaluation_module, "verify_manifest_exp472_antigravity", return_value=_manifest(bundle, context)), patch.object(evaluation_module, "verify_score_shard_exp472_antigravity", side_effect=lambda _root, _context, entry: _shard(entry)):
            result = evaluate_exp472_antigravity(_ROOT, _ROOT, lambda order: {dataset: (0, 1) for dataset in order}, context=context, catalog=catalog)
        baseline = result.artifact_record()["evaluation_sha256"]
        changes = (
            replace(result, per_dataset=result.per_dataset[1:]), replace(result, macro={}), replace(result, micro={}),
            replace(result, subset_per_dataset=result.subset_per_dataset[1:]), replace(result, subset_macro={}), replace(result, subset_micro={}),
            replace(result, denominators={}), replace(result, paired_statistics=()),
            replace(result, curves={"roc": {"x_name": "false_positive_rate", "y_name": "true_positive_rate", "rows": ()}, "pr": result.curves["pr"]}),
            replace(result, curves={"roc": result.curves["roc"], "pr": {"x_name": "recall", "y_name": "precision", "rows": ()}}),
        )
        self.assertTrue(all(changed.artifact_record()["evaluation_sha256"] != baseline for changed in changes))
