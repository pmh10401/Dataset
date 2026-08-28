"""Closed report publication and command-boundary tests for Exp472."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
import json
import numpy as np
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from experiments.exp472_makima_upsampling_ablation_antigravity.cli_antigravity import (
    main_antigravity,
)
from experiments.exp472_makima_upsampling_ablation_antigravity import cli_antigravity as cli
from experiments.exp472_makima_upsampling_ablation_antigravity.reporting_antigravity import (
    REPORT_LEAVES_ANTIGRAVITY,
    verify_exp472_report_antigravity,
    write_exp472_report_antigravity,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.contracts_antigravity import (
    canonical_json_bytes_exp472_antigravity,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.config_antigravity import (
    EXP471_REUSE_ROOT_ANTIGRAVITY,
)
from experiments.exp472_makima_upsampling_ablation_antigravity import config_antigravity as config_module
from experiments.exp472_makima_upsampling_ablation_antigravity.artifacts_antigravity import (
    score_shard_record_exp472_antigravity,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.contracts_antigravity import (
    Exp472RunContextAntigravity,
    Exp472ScoreShardAntigravity,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.transforms_antigravity import (
    dataset_shape_diagnostic_exp472_antigravity,
)
from paper.equipment_anomaly_detection.contracts_paper import RawDataset
from experiments.exp472_makima_upsampling_ablation_antigravity.tests.test_authority_evaluation_antigravity import (
    _bundle as _task5_bundle_antigravity,
    _context_and_catalog as _task5_context_catalog_antigravity,
    _manifest as _task5_manifest_antigravity,
    _shard as _task5_shard_antigravity,
)


_SHA = sha256(b"exp472-report-fixture-antigravity").hexdigest()
_METHODS = (
    "A", "B", "C", "D", "E", "knn_distance", "lof_novelty",
    "ocsvm_rbf", "isolation_forest", "deep_svdd_1d", "drocc_1d",
)


def _worker_antigravity() -> dict[str, object]:
    return {"logical_cpu_count": 8, "cpu_workers": 7, "inner_numerical_threads": 1,
            "tensor_backend": "mps", "mlx_fallback_reason": "fixed", "pytorch_available": True,
            "pytorch_version": "fixture", "scipy_available": True, "scipy_version": "fixture",
            "mlx_available": False, "mlx_version": "unavailable", "mlx_metal_available": False,
            "mlx_metal_version": "unavailable", "mps_available": True}


def _shape_diagnostic_antigravity(dataset: str) -> dict[str, object]:
    """Use Task3's actual nested aggregate, never a report-only fixture."""
    train = np.asarray([0.0, 1.0, 0.5, 1.5], dtype="<f4")
    test = np.asarray([1.0, 0.0, 1.0, 0.0], dtype="<f4")
    raw = RawDataset(
        dataset, 4, (0,), (train.tobytes(),), (train,), (1,),
        (test.tobytes(),), (test,),
    )
    return dict(dataset_shape_diagnostic_exp472_antigravity(raw))


def _shape_diagnostics_antigravity() -> tuple[dict[str, object], ...]:
    return (_shape_diagnostic_antigravity("fixture"),)


@dataclass(frozen=True)
class _AuthorityAntigravity:
    exp471_source_authority_sha256: str = _SHA
    exp471_cache_manifest_sha256: str = _SHA
    exp471_evaluation_sha256: str = _SHA
    exp471_report_manifest_sha256: str = _SHA
    database_sha256: str = _SHA
    coordinate_inventory_sha256: str = _SHA
    method_order: tuple[str, ...] = _METHODS[:-1]
    total: int = 11170
    scored: int = 10775
    abstained: int = 395
    failed: int = 0
    dataset_count: int = 1117
    common_support: int = 922


@dataclass(frozen=True)
class _EvaluationAntigravity:
    authority: _AuthorityAntigravity = _AuthorityAntigravity()
    method_order: tuple[str, ...] = _METHODS
    macro: dict[str, dict[str, object]] = None  # type: ignore[assignment]
    micro: dict[str, dict[str, object]] = None  # type: ignore[assignment]
    subset_macro: dict[str, dict[str, object]] = None  # type: ignore[assignment]
    subset_micro: dict[str, dict[str, object]] = None  # type: ignore[assignment]
    per_dataset: tuple[dict[str, object], ...] = ()
    subset_per_dataset: tuple[dict[str, object], ...] = ()
    paired_statistics: tuple[dict[str, object], ...] = ()
    denominators: dict[str, dict[str, object]] = None  # type: ignore[assignment]
    curves: dict[str, dict[str, object]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        metrics = {
            method: {
                "average_precision": 0.9 if method == "E" else 0.8,
                "auroc": 0.9 if method == "E" else 0.8,
                "average_precision_denominator": 100,
                "auroc_denominator": 100,
                "average_precision_excluded": 1017,
                "auroc_excluded": 1017,
            }
            for method in self.method_order
        }
        object.__setattr__(self, "macro", metrics)
        object.__setattr__(self, "micro", metrics)
        object.__setattr__(self, "subset_macro", {
            method: {**value, "average_precision_denominator": 3,
                     "auroc_denominator": 2,
                     "average_precision_excluded": 1,
                     "auroc_excluded": 2}
            for method, value in metrics.items()
        })
        object.__setattr__(self, "subset_micro", metrics)
        object.__setattr__(self, "per_dataset", tuple(
            {"method": method, "dataset": "fixture", "average_precision": 0.9,
             "average_precision_exclusion": None, "auroc": 0.9,
             "auroc_exclusion": None}
            for method in self.method_order
        ))
        object.__setattr__(self, "subset_per_dataset", self.per_dataset)
        object.__setattr__(self, "paired_statistics", (
            {"comparison": "E-D", "population": "full", "metric": "average_precision",
             "mean_delta": 0.1, "ci_lower": 0.01, "ci_upper": 0.2,
             "wins": 1, "ties": 0, "losses": 0, "seed": 47220260828,
             "draw_count": 20000, "percentile": 95.0, "tie_tolerance": 1e-12},
            {"comparison": "E-D", "population": "full", "metric": "auroc",
             "mean_delta": 0.0, "ci_lower": -0.1, "ci_upper": 0.1,
             "wins": 0, "ties": 1, "losses": 0, "seed": 47220260828,
             "draw_count": 20000, "percentile": 95.0, "tie_tolerance": 1e-12},
        ))
        object.__setattr__(self, "denominators", {
            "full_population": {"dataset_count": 1117, "observation_count": 195900},
            "upsampled_rows_gt_zero": {"dataset_count": 4, "observation_count": 8,
                                         "dataset_order": ("fixture",)},
        })
        object.__setattr__(self, "curves", {
            "roc": {"x_name": "false_positive_rate", "y_name": "true_positive_rate",
                    "rows": tuple({"method": method, "false_positive_rate": x,
                                    "true_positive_rate": x, "threshold": 0.5}
                                  for method in self.method_order for x in (0.0, 1.0))},
            "pr": {"x_name": "recall", "y_name": "precision",
                   "rows": tuple({"method": method, "recall": x,
                                   "precision": 1.0 - 0.2 * x}
                                 for method in self.method_order for x in (0.0, 1.0))},
        })

    def artifact_record(self) -> dict[str, object]:
        bare = {"schema": "exp472-evaluation-antigravity.v1", "fixture": "verified"}
        return {**bare, "evaluation_sha256": sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest()}


class Exp472ReportingCliTestsAntigravity(unittest.TestCase):
    def _runtime_antigravity(self, report_root: Path | None = None):
        bundle = _task5_bundle_antigravity(); context, catalog = _task5_context_catalog_antigravity(bundle)
        return bundle, cli.Exp472ControllerRuntimeAntigravity(context, catalog, EXP471_REUSE_ROOT_ANTIGRAVITY, _SHA, report_root or Path("/private/tmp/exp472-report-runtime-antigravity"))

    def test_status_is_values_only_and_reports_paths(self) -> None:
        output = io.StringIO()
        self.assertEqual(main_antigravity(("status",), stdout=output), 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value["experiment"], "Exp472")
        self.assertEqual(set(value["paths"]), {"root", "smoke", "official", "evaluation", "report", "progress"})
        self.assertNotIn("label_access", __import__("sys").modules)

    def test_status_is_read_only_and_reports_progress_eta_without_label_module(self) -> None:
        with TemporaryDirectory(prefix="exp472-status-antigravity-") as temporary:
            official = Path(temporary) / "official_antigravity"; official.mkdir()
            progress = official / "progress_antigravity.jsonl"
            progress.write_text(json.dumps({"completed": 4, "failed": 0, "remaining": 8, "elapsed_seconds": 20.0}) + "\n", encoding="utf-8")
            before = progress.stat().st_mtime_ns
            module_name = "paper.equipment_anomaly_detection.label_access_paper"; prior = sys.modules.pop(module_name, None)
            try:
                with mock.patch.object(cli, "EXP472_OFFICIAL_ROOT_ANTIGRAVITY", official):
                    value = cli._status_antigravity()
                self.assertEqual((value["scored"], value["failed"], value["remaining"], value["eta_seconds"]), (4, 0, 8, 40.0))
                self.assertEqual(progress.stat().st_mtime_ns, before)
                self.assertNotIn(module_name, sys.modules)
            finally:
                if prior is not None: sys.modules[module_name] = prior

    def test_evaluate_refuses_incomplete_runtime_before_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "typed controller"):
            main_antigravity(("evaluate",), runtime={"verify_exp471": lambda: object()})

    def test_report_writes_exact_leaves_manifest_marker_and_correct_axes(self) -> None:
        with TemporaryDirectory(prefix="exp472-report-antigravity-") as temporary:
            root = Path(temporary) / "report_antigravity"
            authority = write_exp472_report_antigravity(
                _EvaluationAntigravity(), root, _worker_antigravity(),
                e_manifest_sha256=_SHA, code_map_lock_sha256=_SHA,
                shape_diagnostics=_shape_diagnostics_antigravity(),
            )
            self.assertEqual(tuple(authority.leaf_sha256), REPORT_LEAVES_ANTIGRAVITY)
            self.assertEqual(len(authority.leaf_sha256), 15)
            self.assertTrue(verify_exp472_report_antigravity(root, authority))
            html = (root / "exp472_paper_report_antigravity.html").read_text(encoding="utf-8")
            self.assertIn("Makima", html); self.assertIn("PCHIP", html)
            self.assertIn("False-positive rate", html); self.assertIn("Recall", html)
            roc = (root / "exp472_roc_coordinates_antigravity.csv").read_text(encoding="utf-8")
            pr = (root / "exp472_pr_coordinates_antigravity.csv").read_text(encoding="utf-8")
            self.assertIn("false_positive_rate", roc); self.assertIn("true_positive_rate", roc)
            self.assertIn("recall", pr); self.assertIn("precision", pr)
            self.assertEqual(set(authority.leaf_sha256), set(REPORT_LEAVES_ANTIGRAVITY))
            self.assertTrue((root / "exp472_report_manifest_antigravity.json").is_file())
            self.assertTrue((root / ".exp472_report_complete_antigravity.json").is_file())
            audit = (root / "exp472_denominator_audit_antigravity.csv").read_text(encoding="utf-8")
            self.assertEqual(audit.count("upsampled_rows_gt_zero,"), 12)
            self.assertIn("full_population,E,,,100,100,1017,1017", audit)
            self.assertIn("upsampled_rows_gt_zero,E,,,3,2,1,2", audit)

    def test_direct_report_rejects_empty_or_duplicate_shape_diagnostics(self) -> None:
        """A paper publisher cannot silently replace verified E diagnostics."""
        with TemporaryDirectory(prefix="exp472-shape-required-antigravity-") as temporary:
            root = Path(temporary) / "report_antigravity"
            with self.assertRaisesRegex(ValueError, "shape diagnostics"):
                write_exp472_report_antigravity(_EvaluationAntigravity(), root, _worker_antigravity(), e_manifest_sha256=_SHA, code_map_lock_sha256=_SHA)
            with self.assertRaisesRegex(ValueError, "shape diagnostics"):
                write_exp472_report_antigravity(_EvaluationAntigravity(), root, _worker_antigravity(), e_manifest_sha256=_SHA, code_map_lock_sha256=_SHA, shape_diagnostics=_shape_diagnostics_antigravity() * 2)

    def test_direct_report_flattens_authentic_nested_shape_diagnostic(self) -> None:
        """All nested Task3 train/test evidence survives canonical CSV export."""
        with TemporaryDirectory(prefix="exp472-nested-shape-antigravity-") as temporary:
            root = Path(temporary) / "report_antigravity"
            write_exp472_report_antigravity(
                _EvaluationAntigravity(), root, _worker_antigravity(),
                e_manifest_sha256=_SHA, code_map_lock_sha256=_SHA,
                shape_diagnostics=_shape_diagnostics_antigravity(),
            )
            text = (root / "exp472_shape_diagnostics_antigravity.csv").read_text(encoding="utf-8")
            self.assertIn("train_rows_sha256", text)
            self.assertIn("test_rows_sha256", text)
            self.assertIn(_shape_diagnostics_antigravity()[0]["diagnostic_sha256"], text)

    def test_direct_report_rejects_nested_shape_tamper_schema_duplicate_and_missing_before_completion(self) -> None:
        """Nested E evidence is authenticated before any report completion marker."""
        diagnostic = _shape_diagnostics_antigravity()[0]
        altered_train = {**diagnostic["train"], "overshoot": 0.25}
        tampered = {**diagnostic, "train": altered_train}
        bad_schema_bare = {**diagnostic, "schema": "wrong-shape-schema-antigravity.v1"}
        bad_schema_bare.pop("diagnostic_sha256")
        bad_schema = {
            **bad_schema_bare,
            "diagnostic_sha256": sha256(canonical_json_bytes_exp472_antigravity(bad_schema_bare)).hexdigest(),
        }
        bad_nested_bare = {
            **diagnostic,
            "train": {
                key: value for key, value in diagnostic["train"].items()
                if key != "overshoot"
            },
        }
        bad_nested_bare.pop("diagnostic_sha256")
        bad_nested = {
            **bad_nested_bare,
            "diagnostic_sha256": sha256(canonical_json_bytes_exp472_antigravity(bad_nested_bare)).hexdigest(),
        }
        for name, shapes in (
            ("tamper", (tampered,)),
            ("schema", (bad_schema,)),
            ("nested-schema", (bad_nested,)),
            ("duplicate", (diagnostic, diagnostic)),
            ("missing", ()),
        ):
            with self.subTest(name=name), TemporaryDirectory(prefix=f"exp472-shape-{name}-antigravity-") as temporary:
                root = Path(temporary) / "report_antigravity"
                with self.assertRaisesRegex(ValueError, "shape diagnostics"):
                    write_exp472_report_antigravity(
                        _EvaluationAntigravity(), root, _worker_antigravity(),
                        e_manifest_sha256=_SHA, code_map_lock_sha256=_SHA,
                        shape_diagnostics=shapes,
                    )
                self.assertFalse((root / ".exp472_report_complete_antigravity.json").exists())

    def test_report_is_byte_identical_and_detects_leaf_manifest_and_marker_tampering(self) -> None:
        with TemporaryDirectory(prefix="exp472-report-antigravity-") as temporary:
            root = Path(temporary) / "report_antigravity"
            first = write_exp472_report_antigravity(_EvaluationAntigravity(), root, _worker_antigravity(), e_manifest_sha256=_SHA, code_map_lock_sha256=_SHA, shape_diagnostics=_shape_diagnostics_antigravity())
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            second = write_exp472_report_antigravity(_EvaluationAntigravity(), root, _worker_antigravity(), e_manifest_sha256=_SHA, code_map_lock_sha256=_SHA, shape_diagnostics=_shape_diagnostics_antigravity())
            self.assertEqual(first, second)
            self.assertEqual(before, {path.name: path.read_bytes() for path in root.iterdir()})
            for name in ("exp472_roc_coordinates_antigravity.csv", "exp472_report_manifest_antigravity.json", ".exp472_report_complete_antigravity.json"):
                with self.subTest(name=name):
                    original = (root / name).read_bytes()
                    (root / name).write_bytes(original + b"x")
                    with self.assertRaises(ValueError):
                        verify_exp472_report_antigravity(root, first)
                    (root / name).write_bytes(original)

    def test_report_rejects_symlink_and_closed_commands_never_start_scoring(self) -> None:
        with TemporaryDirectory(prefix="exp472-report-antigravity-") as temporary:
            root = Path(temporary) / "report_antigravity"
            authority = write_exp472_report_antigravity(_EvaluationAntigravity(), root, _worker_antigravity(), e_manifest_sha256=_SHA, code_map_lock_sha256=_SHA, shape_diagnostics=_shape_diagnostics_antigravity())
            leaf = root / "exp472_pr_coordinates_antigravity.csv"; target = root / "symlink-target-antigravity.csv"
            target.write_bytes(leaf.read_bytes()); leaf.unlink(); leaf.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                verify_exp472_report_antigravity(root, authority)
        scorer = mock.Mock(side_effect=AssertionError("report must not score"))
        with self.assertRaisesRegex(ValueError, "typed controller"):
            main_antigravity(("report",), runtime={"verify_exp471": lambda: object(), "score_one": scorer})
        scorer.assert_not_called()
        verifier = mock.Mock(return_value=object())
        with self.assertRaisesRegex(ValueError, "typed controller"):
            main_antigravity(("verify",), runtime={"verify_exp471": verifier})
        verifier.assert_not_called()

    def test_typed_runtime_reopens_task4_task5_before_evaluate_and_report(self) -> None:
        """A names-only object cannot replace typed Task4/Task5 boundaries."""
        bundle, runtime = self._runtime_antigravity()
        manifest = _task5_manifest_antigravity(bundle, runtime.context)
        with mock.patch.object(cli, "verify_exp471_reuse_exp472_antigravity", return_value=bundle) as exp471, mock.patch.object(cli, "verify_manifest_exp472_antigravity", return_value=manifest) as verify_manifest, mock.patch.object(cli, "verify_score_shard_exp472_antigravity", side_effect=lambda _root, _context, entry: _task5_shard_antigravity(entry)) as verify_shard, mock.patch.object(cli, "evaluate_exp472_antigravity", return_value=_EvaluationAntigravity()) as evaluate:
            self.assertEqual(main_antigravity(("evaluate",), runtime=runtime, stdout=io.StringIO()), 0)
        self.assertEqual((exp471.call_count, verify_manifest.call_count, verify_shard.call_count, evaluate.call_count), (1, 1, 1117, 1))
        self.assertEqual(evaluate.call_args.kwargs, {"context": runtime.context, "catalog": runtime.catalog})
        self.assertEqual(evaluate.call_args.args[:2], (runtime.context.run_root, EXP471_REUSE_ROOT_ANTIGRAVITY))

    def test_default_factory_replays_scored_backend_for_report_without_live_assessment(self) -> None:
        bundle = _task5_bundle_antigravity()
        persisted_context, persisted_catalog = _task5_context_catalog_antigravity(bundle)
        manifest = {**_task5_manifest_antigravity(bundle, persisted_context), "manifest_sha256": _SHA}
        with mock.patch.object(cli, "verify_exp471_reuse_exp472_antigravity", return_value=bundle), mock.patch.object(cli, "_read_scored_context_candidate_antigravity", return_value=persisted_context) as replay, mock.patch.object(cli, "assess_exp472_backend_antigravity", side_effect=AssertionError("post-score commands must not reassess backend")), mock.patch.object(cli, "verify_manifest_exp472_antigravity", return_value=manifest), mock.patch.object(cli, "verify_score_shard_exp472_antigravity", side_effect=lambda _root, _context, entry: {**_task5_shard_antigravity(entry), "diagnostic": _shape_diagnostic_antigravity(entry.dataset)}), mock.patch.object(cli, "evaluate_exp472_antigravity", return_value=_EvaluationAntigravity()), mock.patch.object(cli, "write_exp472_report_antigravity", return_value=mock.Mock(report_manifest_sha256=_SHA)) as publish:
            runtime = cli._build_runtime_antigravity("full")
            self.assertEqual(main_antigravity(("report",), stdout=io.StringIO()), 0)
        self.assertIsInstance(runtime, cli.Exp472ControllerRuntimeAntigravity)
        self.assertEqual((runtime.context.mode, len(runtime.catalog)), ("full", 1117))
        self.assertEqual(runtime.context.backend.artifact_record(), persisted_context.backend.artifact_record())
        replay.assert_called()
        self.assertEqual(publish.call_args.args[2], persisted_context.backend.artifact_record())
        with mock.patch.object(cli, "_build_runtime_antigravity", return_value=runtime) as build, mock.patch.object(cli.Exp472ControllerRuntimeAntigravity, "run_e_antigravity", return_value={"completed": ("E",), "failures": ()}) as run_e:
            self.assertEqual(main_antigravity(("run",), stdout=io.StringIO()), 0)
        run_e.assert_called_once_with()
        build.assert_called_once_with("full", for_scoring=True)

    def test_scored_context_candidate_reopens_the_durable_backend_record(self) -> None:
        """The replay candidate is a canonical scored shard, not a live probe."""
        bundle = _task5_bundle_antigravity()
        _old_context, catalog = _task5_context_catalog_antigravity(bundle)
        backend = _old_context.backend
        with TemporaryDirectory(prefix="exp472-context-replay-antigravity-") as temporary:
            root = Path(temporary) / "official_antigravity"; root.mkdir()
            with mock.patch.object(cli, "EXP472_OFFICIAL_ROOT_ANTIGRAVITY", root), mock.patch.object(config_module, "EXP472_OFFICIAL_ROOT_ANTIGRAVITY", root):
                context = Exp472RunContextAntigravity(
                    "full", root, cli.FROZEN_EXP472_CONFIG.database_sha256,
                    cli.catalog_sha256_exp472_antigravity(catalog),
                    {"experiments/exp472_scored_context_antigravity.py": _SHA},
                    backend, cli._config_sha256_antigravity(),
                )
                bare = {"dataset": catalog[0].dataset, "shape": "scored"}
                diagnostic = {**bare, "diagnostic_sha256": sha256(canonical_json_bytes_exp472_antigravity(bare)).hexdigest()}
                shard = Exp472ScoreShardAntigravity(
                    catalog[0].dataset, catalog[0].test_identity,
                    np.asarray([0.2, 0.8], dtype=np.float64), frozenset({1}), 1,
                    diagnostic,
                )
                record = score_shard_record_exp472_antigravity(context, shard)
                token = sha256(catalog[0].dataset.encode("utf-8")).hexdigest()
                shard_directory = root / "score_shards_antigravity" / f"{token}_antigravity"
                shard_directory.mkdir(parents=True)
                (shard_directory / "score_shard_antigravity.json").write_bytes(
                    canonical_json_bytes_exp472_antigravity(record)
                )
                replayed = cli._read_scored_context_candidate_antigravity(root, catalog)
            self.assertEqual(replayed.sha256, context.sha256)
            self.assertEqual(replayed.backend.artifact_record(), backend.artifact_record())

    def test_evaluate_and_report_reject_smoke_context_before_any_evaluation(self) -> None:
        """The smoke E inventory is never a substitute for the 1,117-run report."""
        bundle = _task5_bundle_antigravity()
        full_context, full_catalog = _task5_context_catalog_antigravity(bundle)
        smoke_context = type(full_context)(
            "smoke", cli.EXP472_SMOKE_ROOT_ANTIGRAVITY,
            full_context.database_sha256, cli.catalog_sha256_exp472_antigravity(full_catalog[:23]),
            full_context.code_closure, full_context.backend, full_context.config_sha256,
        )
        runtime = cli.Exp472ControllerRuntimeAntigravity(
            smoke_context, full_catalog[:23], EXP471_REUSE_ROOT_ANTIGRAVITY, _SHA,
        )
        with mock.patch.object(cli, "_verified_authorities_antigravity", side_effect=AssertionError("must stop before authority reopen")), mock.patch.object(cli.Exp472ControllerRuntimeAntigravity, "evaluate_antigravity", side_effect=AssertionError("must stop before labels")):
            for command in ("evaluate", "report"):
                with self.subTest(command=command):
                    with self.assertRaisesRegex(ValueError, "full E context"):
                        main_antigravity((command,), runtime=runtime, stdout=io.StringIO())

    def test_report_uses_verified_shard_diagnostics_and_never_calls_label_or_score(self) -> None:
        with TemporaryDirectory(prefix="exp472-runtime-report-antigravity-") as temporary:
            bundle, runtime = self._runtime_antigravity(Path(temporary) / "report_antigravity")
            manifest = {**_task5_manifest_antigravity(bundle, runtime.context), "manifest_sha256": _SHA}
            evaluation = _EvaluationAntigravity()
            e_rows = tuple(
                {"method": "E", "dataset": entry.dataset, "average_precision": 0.9,
                 "average_precision_exclusion": None, "auroc": 0.9,
                 "auroc_exclusion": None}
                for entry in runtime.catalog
            )
            object.__setattr__(evaluation, "per_dataset", e_rows)
            object.__setattr__(evaluation, "subset_per_dataset", e_rows)
            def shard(_root, _context, entry):
                value = _task5_shard_antigravity(entry)
                value["diagnostic"] = _shape_diagnostic_antigravity(entry.dataset)
                return value
            with mock.patch.object(cli, "verify_exp471_reuse_exp472_antigravity", return_value=bundle), mock.patch.object(cli, "verify_manifest_exp472_antigravity", return_value=manifest), mock.patch.object(cli, "verify_score_shard_exp472_antigravity", side_effect=shard), mock.patch.object(cli, "evaluate_exp472_antigravity", return_value=evaluation) as evaluate:
                self.assertEqual(main_antigravity(("report",), runtime=runtime, stdout=io.StringIO()), 0)
            self.assertEqual(evaluate.call_count, 1)
            shapes = (runtime.report_root / "exp472_shape_diagnostics_antigravity.csv").read_text(encoding="utf-8")
            self.assertEqual(len(shapes.splitlines()), 1118)
            self.assertIn("train_rows_sha256", shapes)
            self.assertIn(_shape_diagnostic_antigravity(runtime.catalog[0].dataset)["diagnostic_sha256"], shapes)

    def test_report_authority_failure_opens_no_real_label_module_or_scorer(self) -> None:
        bundle, runtime = self._runtime_antigravity()
        module_name = "paper.equipment_anomaly_detection.label_access_paper"
        prior = sys.modules.pop(module_name, None)
        try:
            with mock.patch.object(cli, "_verified_authorities_antigravity", side_effect=ValueError("incomplete E")), mock.patch.object(cli.Exp472ControllerRuntimeAntigravity, "run_e_antigravity", side_effect=AssertionError("report must not score")) as run_e:
                with self.assertRaisesRegex(ValueError, "incomplete E"):
                    main_antigravity(("report",), runtime=runtime, stdout=io.StringIO())
            run_e.assert_not_called()
            self.assertNotIn(module_name, sys.modules)
        finally:
            if prior is not None:
                sys.modules[module_name] = prior

    def test_png_is_a_real_chart_and_zero_interval_summary_does_not_claim_equivalence(self) -> None:
        fixture = _EvaluationAntigravity()
        object.__setattr__(fixture, "paired_statistics", ({"comparison": "E-D", "population": "full", "metric": "average_precision", "mean_delta": 0.0, "ci_lower": -0.1, "ci_upper": 0.1, "wins": 0, "ties": 1, "losses": 0, "seed": 47220260828, "draw_count": 20000, "percentile": 95.0, "tie_tolerance": 1e-12},))
        with TemporaryDirectory(prefix="exp472-png-antigravity-") as temporary:
            root = Path(temporary) / "report_antigravity"; write_exp472_report_antigravity(fixture, root, _worker_antigravity(), e_manifest_sha256=_SHA, code_map_lock_sha256=_SHA, shape_diagnostics=_shape_diagnostics_antigravity())
            png = (root / "exp472_curves_antigravity.png").read_bytes()
            self.assertEqual((int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")), (400, 240))
            self.assertGreater(len(png), 1000)
            summary = (root / "exp472_paper_summary_ko_antigravity.md").read_text(encoding="utf-8")
            self.assertIn("결정적", summary); self.assertIn("동등성의 증거가 아닙니다", summary)

    def test_png_bytes_follow_frozen_curve_coordinates(self) -> None:
        """A coordinate change changes the deterministic raster, not just its metadata."""
        first, second = _EvaluationAntigravity(), _EvaluationAntigravity()
        changed_roc = tuple(
            {**row, "true_positive_rate": 0.25 if row["method"] == "E" and row["false_positive_rate"] == 1.0 else row["true_positive_rate"]}
            for row in second.curves["roc"]["rows"]
        )
        object.__setattr__(second, "curves", {**second.curves, "roc": {**second.curves["roc"], "rows": changed_roc}})
        with TemporaryDirectory(prefix="exp472-png-coordinates-antigravity-") as temporary:
            root = Path(temporary)
            one, two = root / "one_antigravity", root / "two_antigravity"
            write_exp472_report_antigravity(first, one, _worker_antigravity(), e_manifest_sha256=_SHA, code_map_lock_sha256=_SHA, shape_diagnostics=_shape_diagnostics_antigravity())
            write_exp472_report_antigravity(second, two, _worker_antigravity(), e_manifest_sha256=_SHA, code_map_lock_sha256=_SHA, shape_diagnostics=_shape_diagnostics_antigravity())
            self.assertNotEqual(
                (one / "exp472_curves_antigravity.png").read_bytes(),
                (two / "exp472_curves_antigravity.png").read_bytes(),
            )

    def test_shuffled_frozen_curve_coordinates_have_identical_svg_and_png_bytes(self) -> None:
        """The canonical graph ordering is independent of Task5 row iteration order."""
        ordered, shuffled = _EvaluationAntigravity(), _EvaluationAntigravity()
        object.__setattr__(shuffled, "curves", {
            key: {**value, "rows": tuple(reversed(value["rows"]))}
            for key, value in shuffled.curves.items()
        })
        with TemporaryDirectory(prefix="exp472-canonical-curves-antigravity-") as temporary:
            root = Path(temporary)
            one, two = root / "one_antigravity", root / "two_antigravity"
            for evaluation, output in ((ordered, one), (shuffled, two)):
                write_exp472_report_antigravity(
                    evaluation, output, _worker_antigravity(),
                    e_manifest_sha256=_SHA, code_map_lock_sha256=_SHA,
                    shape_diagnostics=_shape_diagnostics_antigravity(),
                )
            for name in (
                "exp472_roc_curve_antigravity.svg",
                "exp472_pr_curve_antigravity.svg",
                "exp472_curves_antigravity.png",
            ):
                self.assertEqual((one / name).read_bytes(), (two / name).read_bytes())
