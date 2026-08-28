"""Production-path authority and label-firewall tests for Exp472 Task 5."""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
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
from experiments.exp472_makima_upsampling_ablation_antigravity.config_antigravity import BOOTSTRAP_CONFIDENCE_LEVEL_ANTIGRAVITY, BOOTSTRAP_DRAWS_ANTIGRAVITY, BOOTSTRAP_SEED_ANTIGRAVITY, EXP467_VERIFICATION_REPOSITORY_ANTIGRAVITY, EXP472_OFFICIAL_ROOT_ANTIGRAVITY
from experiments.exp472_makima_upsampling_ablation_antigravity.contracts_antigravity import Exp472BackendDecisionAntigravity, Exp472RunContextAntigravity, canonical_json_bytes_exp472_antigravity
from experiments.exp472_makima_upsampling_ablation_antigravity.transforms_antigravity import Exp472AlignmentTraceAntigravity, align_rows_exp472_antigravity
from experiments.exp472_makima_upsampling_ablation_antigravity.evaluation_antigravity import Exp472VerifiedEBundleAntigravity, evaluate_exp472_antigravity
from experiments.exp472_makima_upsampling_ablation_antigravity.transforms_antigravity import test_identity_exp472_antigravity

_ROOT = Path("/private/tmp/exp472-task5-fixture-antigravity")
_METHODS = ("A", "B", "C", "D", "knn_distance", "lof_novelty", "ocsvm_rbf", "isolation_forest", "deep_svdd_1d", "drocc_1d")

def _digest(value: str) -> str: return sha256(value.encode()).hexdigest()

def _canonical_digest_document(bare: dict[str, object], digest_field: str) -> bytes:
    """Build a hand-checked canonical Exp471 document with a self digest."""
    value = {**bare, digest_field: sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest()}
    return canonical_json_bytes_exp472_antigravity(value)

_TYPED_ROOT = "exp471_typed_report_artifacts_antigravity"
_TYPED_MANIFEST = "exp471_typed_report_manifest_antigravity.json"
_TYPED_LEAVES = (
    "exp471_coverage_antigravity.csv", "exp471_coverage_antigravity.json",
    "exp471_full_population_decisions_antigravity.csv", "exp471_full_population_decisions_antigravity.json",
    "exp471_scored_support_metrics_antigravity.csv", "exp471_scored_support_metrics_antigravity.json",
    "exp471_common_support_metrics_antigravity.csv", "exp471_common_support_metrics_antigravity.json",
    "exp471_scored_support_curves_antigravity.svg", "exp471_scored_support_curves_antigravity.png",
    "exp471_common_support_curves_antigravity.svg", "exp471_common_support_curves_antigravity.png",
    "exp471_paper_report_antigravity.html",
)

def _write_production_shaped_exp471_tree(root: Path, source_authority: object, manifest: object) -> tuple[Path, Path, Path]:
    """Write the frozen Exp471 report's marker -> map -> typed-manifest shape."""
    evaluation = {
        "source_authority_sha256": source_authority.authority_sha256,
        "cache_manifest_sha256": manifest.manifest_sha256,
    }
    evaluation_path = root / "evaluation_antigravity" / "evaluation_antigravity.json"
    evaluation_path.parent.mkdir()
    evaluation_path.write_bytes(_canonical_digest_document(evaluation, "evaluation_sha256"))

    report_root = root / "report_antigravity"
    typed_root = report_root / _TYPED_ROOT
    typed_root.mkdir(parents=True)
    typed_files: dict[str, bytes] = {
        name: f"fixture typed leaf: {name}\n".encode("utf-8") for name in _TYPED_LEAVES
    }
    typed_manifest_path = typed_root / _TYPED_MANIFEST
    typed_files[_TYPED_MANIFEST] = canonical_json_bytes_exp472_antigravity({
        "schema": "exp471-typed-report-manifest-antigravity.v1",
        "files": {name: sha256(payload).hexdigest() for name, payload in sorted(typed_files.items())},
    })
    for name, payload in typed_files.items():
        (typed_root / name).write_bytes(payload)
    report_files = {
        "exp471_report_provenance_antigravity.json": sha256(b'{"fixture":"provenance"}').hexdigest(),
        **{f"{_TYPED_ROOT}/{name}": sha256(payload).hexdigest() for name, payload in typed_files.items()},
    }
    (report_root / "exp471_report_provenance_antigravity.json").write_bytes(b'{"fixture":"provenance"}')
    artifact_map_path = report_root / "artifact_sha256_antigravity.json"
    artifact_map = {
        "schema": "exp471-report-artifact-map-antigravity.v2",
        "self_excluded": "artifact_sha256_antigravity.json",
        "files": dict(sorted(report_files.items())),
    }
    artifact_map_path.write_bytes(canonical_json_bytes_exp472_antigravity(artifact_map))
    marker = {
        "schema": "exp471-report-publication-antigravity.v2",
        "source_authority_sha256": source_authority.authority_sha256,
        "cache_manifest_sha256": manifest.manifest_sha256,
        "evaluation_sha256": sha256(canonical_json_bytes_exp472_antigravity(evaluation)).hexdigest(),
        "report_file_sha256": dict(sorted(report_files.items())),
        "report_manifest_sha256": sha256(artifact_map_path.read_bytes()).hexdigest(),
        "artifact_map": "artifact_sha256_antigravity.json",
    }
    marker_path = report_root / ".exp471_report_complete_antigravity.json"
    marker_path.write_bytes(canonical_json_bytes_exp472_antigravity(marker))
    return evaluation_path, artifact_map_path, typed_manifest_path

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
    trace = Exp472AlignmentTraceAntigravity(entry.dataset)
    if upsampled:
        row = np.asarray((0.0, 1.0, 0.0), dtype="<f4")
        trace.begin_scoring((("hard_sources", "TRAIN"),))
        trace.observe((row,), 7, align_rows_exp472_antigravity((row,), 7), "hard_sources", "TRAIN")
    else:
        trace.begin_scoring(())
    trace.mark_score_completed(_digest("raw"), _digest("result"))
    diagnostic = dict(trace.seal())
    return {"dataset": entry.dataset, "shard_sha256": _digest(f"E:{entry.dataset}"), "test_identity": dict(entry.test_identity), "scores": [0.2, 0.8], "upsampled_rows": diagnostic["upsampled_rows"], "diagnostic": diagnostic}

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
        """The frozen marker -> artifact-map -> typed-manifest chain must bind reads."""
        bundle = _bundle(); source_authority = SimpleNamespace(authority_sha256=bundle.authority.exp471_source_authority_sha256, database_sha256=bundle.authority.database_sha256)
        catalog_rows = [{"dataset": dataset, "test": {**dict(bundle.coordinates[dataset]), "dataset_id": index}} for index, dataset in enumerate(bundle.dataset_order)]
        session = SimpleNamespace(authority=source_authority, catalog={"datasets": catalog_rows})
        manifest = SimpleNamespace(manifest_sha256=bundle.authority.exp471_cache_manifest_sha256, dataset_order=bundle.dataset_order, method_order=_METHODS, total_shards=11170, scored_shards=10775, abstained_shards=395, failed_shards=0)
        def shard(_root: Path, method: str, dataset: str):
            coordinate = bundle.coordinates[dataset]; status = bundle.statuses[method][dataset]
            return SimpleNamespace(test_count=coordinate["count"], test_indices=coordinate["indices"], status=status, raw_scores=() if status == "abstained" else bundle.scores[method][dataset])
        with TemporaryDirectory(prefix="exp472-task5-exp471-antigravity-") as temporary:
            root = Path(temporary)
            evaluation_path, artifact_map_path, typed_manifest_path = _write_production_shaped_exp471_tree(root, source_authority, manifest)
            expected_report_digest = sha256(artifact_map_path.read_bytes()).hexdigest()
            with patch.object(exp471_source, "open_exp467_source_session_antigravity", return_value=session) as open_session, patch.object(exp471_source, "read_source_authority_antigravity", return_value=source_authority) as read_source, patch.object(exp471_cache, "read_cache_manifest_antigravity", return_value=manifest), patch.object(exp471_cache, "read_score_shard_antigravity", side_effect=shard):
                verified = verify_exp471_reuse_exp472_antigravity(root)
        self.assertEqual(verified.coordinates, bundle.coordinates); self.assertEqual(verified.authority.exp471_evaluation_sha256, sha256(canonical_json_bytes_exp472_antigravity({"source_authority_sha256": source_authority.authority_sha256, "cache_manifest_sha256": manifest.manifest_sha256})).hexdigest()); self.assertEqual(verified.authority.exp471_report_manifest_sha256, expected_report_digest); self.assertEqual(open_session.call_args.args[0], EXP467_VERIFICATION_REPOSITORY_ANTIGRAVITY); self.assertEqual(read_source.call_args.args[:2], (root, EXP467_VERIFICATION_REPOSITORY_ANTIGRAVITY))

    def test_exp471_path_adapter_rejects_symlinked_evaluation_leaf(self) -> None:
        """A symlinked Exp471 evaluation leaf must fail before any shard or label work."""
        bundle = _bundle(); source_authority = SimpleNamespace(authority_sha256=bundle.authority.exp471_source_authority_sha256, database_sha256=bundle.authority.database_sha256)
        session = SimpleNamespace(authority=source_authority, catalog={"datasets": []})
        manifest = SimpleNamespace(manifest_sha256=bundle.authority.exp471_cache_manifest_sha256)
        with TemporaryDirectory(prefix="exp472-task5-exp471-antigravity-") as temporary:
            root = Path(temporary); evaluation_directory = root / "evaluation_antigravity"; evaluation_directory.mkdir()
            target = root / "evaluation-target-antigravity.json"; target.write_bytes(b"{}")
            (evaluation_directory / "evaluation_antigravity.json").symlink_to(target)
            with patch.object(exp471_source, "open_exp467_source_session_antigravity", return_value=session), patch.object(exp471_source, "read_source_authority_antigravity", return_value=source_authority), patch.object(exp471_cache, "read_cache_manifest_antigravity", return_value=manifest):
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    verify_exp471_reuse_exp472_antigravity(root)

    def test_exp471_path_adapter_rejects_symlinked_typed_manifest_before_shards(self) -> None:
        """A symlinked typed manifest must not cross the report authority boundary."""
        bundle = _bundle(); source_authority = SimpleNamespace(authority_sha256=bundle.authority.exp471_source_authority_sha256, database_sha256=bundle.authority.database_sha256)
        catalog_rows = [{"dataset": dataset, "test": {**dict(bundle.coordinates[dataset]), "dataset_id": index}} for index, dataset in enumerate(bundle.dataset_order)]
        session = SimpleNamespace(authority=source_authority, catalog={"datasets": catalog_rows})
        manifest = SimpleNamespace(manifest_sha256=bundle.authority.exp471_cache_manifest_sha256, dataset_order=bundle.dataset_order, method_order=_METHODS, total_shards=11170, scored_shards=10775, abstained_shards=395, failed_shards=0)
        with TemporaryDirectory(prefix="exp472-task5-exp471-antigravity-") as temporary:
            root = Path(temporary); _evaluation_path, _artifact_map_path, typed_manifest_path = _write_production_shaped_exp471_tree(root, source_authority, manifest)
            target = root / "typed-manifest-target-antigravity.json"; target.write_bytes(typed_manifest_path.read_bytes())
            typed_manifest_path.unlink(); typed_manifest_path.symlink_to(target)
            with patch.object(exp471_source, "open_exp467_source_session_antigravity", return_value=session), patch.object(exp471_source, "read_source_authority_antigravity", return_value=source_authority), patch.object(exp471_cache, "read_cache_manifest_antigravity", return_value=manifest), patch.object(exp471_cache, "read_score_shard_antigravity", side_effect=AssertionError("report authority must fail before shards")):
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    verify_exp471_reuse_exp472_antigravity(root)

    def test_exp471_path_adapter_rejects_production_report_chain_tampering_before_shards(self) -> None:
        """Marker/map/typed-manifest corruption must fail before any shard or label route."""
        bundle = _bundle(); source_authority = SimpleNamespace(authority_sha256=bundle.authority.exp471_source_authority_sha256, database_sha256=bundle.authority.database_sha256)
        catalog_rows = [{"dataset": dataset, "test": {**dict(bundle.coordinates[dataset]), "dataset_id": index}} for index, dataset in enumerate(bundle.dataset_order)]
        session = SimpleNamespace(authority=source_authority, catalog={"datasets": catalog_rows})
        manifest = SimpleNamespace(manifest_sha256=bundle.authority.exp471_cache_manifest_sha256, dataset_order=bundle.dataset_order, method_order=_METHODS, total_shards=11170, scored_shards=10775, abstained_shards=395, failed_shards=0)
        for target_name in ("evaluation", "artifact_map", "typed_manifest"):
            with self.subTest(target=target_name), TemporaryDirectory(prefix="exp472-task5-exp471-antigravity-") as temporary:
                root = Path(temporary); evaluation_path, artifact_map_path, typed_manifest_path = _write_production_shaped_exp471_tree(root, source_authority, manifest)
                target = {"evaluation": evaluation_path, "artifact_map": artifact_map_path, "typed_manifest": typed_manifest_path}[target_name]
                target.write_bytes(target.read_bytes() + b"\n")
                with patch.object(exp471_source, "open_exp467_source_session_antigravity", return_value=session), patch.object(exp471_source, "read_source_authority_antigravity", return_value=source_authority), patch.object(exp471_cache, "read_cache_manifest_antigravity", return_value=manifest), patch.object(exp471_cache, "read_score_shard_antigravity", side_effect=AssertionError("report authority must fail before shards")):
                    with self.assertRaisesRegex(ValueError, "noncanonical|digest"):
                        verify_exp471_reuse_exp472_antigravity(root)

    def test_exp471_public_reader_failure_blocks_e_and_labels(self) -> None:
        """A failed real Exp471 authority read must not invoke E verification or labels."""
        bundle = _bundle(); context, catalog = _context_and_catalog(bundle); calls: list[object] = []
        with patch.object(exp471_source, "open_exp467_source_session_antigravity", return_value=SimpleNamespace()), patch.object(exp471_source, "read_source_authority_antigravity", side_effect=ValueError("bad public Exp471 source")), patch.object(evaluation_module, "verify_manifest_exp472_antigravity", side_effect=AssertionError("E verifier must not run")) as verify_manifest:
            with self.assertRaisesRegex(ValueError, "bad public Exp471 source"):
                evaluate_exp472_antigravity(_ROOT, _ROOT, lambda order: calls.append(order) or {}, context=context, catalog=catalog)
        verify_manifest.assert_not_called()
        self.assertEqual(calls, [])

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

    def test_v3_diagnostic_tamper_with_recomputed_outer_digest_blocks_labels(self) -> None:
        """Evaluation must independently revalidate v3 trace evidence before labels."""
        bundle = _bundle(); context, catalog = _context_and_catalog(bundle); calls: list[object] = []
        def tampered(entry: Exp472CatalogEntryAntigravity) -> dict[str, object]:
            value = _shard(entry)
            changed = dict(value["diagnostic"])
            invocation = dict(changed["invocations"][0])
            invocation["makima_row_indices"] = []
            invocation["makima_rows"] = 0
            invocation["invocation_sha256"] = sha256(canonical_json_bytes_exp472_antigravity({key: item for key, item in invocation.items() if key != "invocation_sha256"})).hexdigest()
            changed["invocations"] = [invocation]
            changed["makima_row_invocation_count"] = 0
            changed["upsampled_rows"] = 0
            aggregate = {name: changed[name] for name in ("invocation_count", "row_invocation_count", "makima_row_invocation_count", "upsampled_rows")}
            changed["invocation_aggregate_sha256"] = sha256(canonical_json_bytes_exp472_antigravity(aggregate)).hexdigest()
            changed["diagnostic_sha256"] = sha256(canonical_json_bytes_exp472_antigravity({key: item for key, item in changed.items() if key != "diagnostic_sha256"})).hexdigest()
            return {**value, "upsampled_rows": 0, "diagnostic": changed}
        with patch.object(evaluation_module, "verify_exp471_reuse_exp472_antigravity", return_value=bundle), patch.object(evaluation_module, "verify_manifest_exp472_antigravity", return_value=_manifest(bundle, context)), patch.object(evaluation_module, "verify_score_shard_exp472_antigravity", side_effect=lambda _root, _context, entry: tampered(entry) if entry.dataset == bundle.dataset_order[0] else _shard(entry)):
            with self.assertRaisesRegex(ValueError, "diagnostic"):
                evaluate_exp472_antigravity(_ROOT, _ROOT, lambda _order: calls.append(_order) or {}, context=context, catalog=catalog)
        self.assertEqual(calls, [])

    def test_verified_e_bundle_freezes_scores_and_coordinates(self) -> None:
        """Mutating caller-owned E maps after verification must not alter bound evidence."""
        bundle = _bundle(); coordinates = {dataset: dict(value) for dataset, value in bundle.coordinates.items()}; scores = {dataset: (0.2, 0.8) for dataset in bundle.dataset_order}; upsampled = {dataset: 1 for dataset in bundle.dataset_order}
        makima = {dataset: 1 for dataset in bundle.dataset_order}; diagnostic_sha = {dataset: _digest(f"diagnostic:{dataset}") for dataset in bundle.dataset_order}
        evidence = Exp472VerifiedEBundleAntigravity(_ROOT, bundle.authority.database_sha256, bundle.dataset_order, coordinates, scores, upsampled, makima, diagnostic_sha, _digest("E immutable"))
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
        self.assertEqual(result.diagnostic_summary[bundle.dataset_order[0]]["upsampled_rows"], 0)
        self.assertEqual(result.diagnostic_summary[bundle.dataset_order[1]]["makima_row_invocation_count"], 1)
        first_e = next(row for row in result.per_dataset if row["method"] == "E" and row["dataset"] == bundle.dataset_order[0])
        self.assertEqual((first_e["average_precision"], first_e["auroc"]), (1.0, 1.0))
        self.assertEqual({item["population"] for item in result.paired_statistics}, {"full", "upsampled_rows_gt_zero"})
        for paired in result.paired_statistics: self.assertEqual((paired["seed"], paired["draw_count"], paired["percentile"], paired["tie_tolerance"]), (BOOTSTRAP_SEED_ANTIGRAVITY, BOOTSTRAP_DRAWS_ANTIGRAVITY, BOOTSTRAP_CONFIDENCE_LEVEL_ANTIGRAVITY * 100, 1e-12))
        record = result.artifact_record(); self.assertEqual(record["schema"], "exp472-evaluation-antigravity.v2"); self.assertEqual(record["evaluation_sha256"], sha256(canonical_json_bytes_exp472_antigravity({key: value for key, value in record.items() if key != "evaluation_sha256"})).hexdigest()); self.assertNotEqual(record["evaluation_sha256"], replace(result, method_order=("changed",)).artifact_record()["evaluation_sha256"])

    def test_evaluation_digest_covers_every_scientific_section(self) -> None:
        """Omitting any scientific table from the digest would permit an unbound Task6 report."""
        bundle = _bundle(); context, catalog = _context_and_catalog(bundle)
        with patch.object(evaluation_module, "verify_exp471_reuse_exp472_antigravity", return_value=bundle), patch.object(evaluation_module, "verify_manifest_exp472_antigravity", return_value=_manifest(bundle, context)), patch.object(evaluation_module, "verify_score_shard_exp472_antigravity", side_effect=lambda _root, _context, entry: _shard(entry)):
            result = evaluate_exp472_antigravity(_ROOT, _ROOT, lambda order: {dataset: (0, 1) for dataset in order}, context=context, catalog=catalog)
        baseline = result.artifact_record()["evaluation_sha256"]
        changes = (
            replace(result, per_dataset=result.per_dataset[1:]), replace(result, macro={}), replace(result, micro={}),
            replace(result, subset_per_dataset=result.subset_per_dataset[1:]), replace(result, subset_macro={}), replace(result, subset_micro={}),
            replace(result, denominators={}), replace(result, diagnostic_summary={}), replace(result, paired_statistics=()),
            replace(result, curves={"roc": {"x_name": "false_positive_rate", "y_name": "true_positive_rate", "rows": ()}, "pr": result.curves["pr"]}),
            replace(result, curves={"roc": result.curves["roc"], "pr": {"x_name": "recall", "y_name": "precision", "rows": ()}}),
        )
        self.assertTrue(all(changed.artifact_record()["evaluation_sha256"] != baseline for changed in changes))
