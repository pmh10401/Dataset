"""Closed report publication and command-boundary tests for Exp472."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from experiments.exp472_makima_upsampling_ablation_antigravity.cli_antigravity import (
    main_antigravity,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.reporting_antigravity import (
    REPORT_LEAVES_ANTIGRAVITY,
    verify_exp472_report_antigravity,
    write_exp472_report_antigravity,
)
from experiments.exp472_makima_upsampling_ablation_antigravity.contracts_antigravity import (
    canonical_json_bytes_exp472_antigravity,
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
                "average_precision_denominator": 1,
                "auroc_denominator": 1,
                "average_precision_excluded": 0,
                "auroc_excluded": 0,
            }
            for method in self.method_order
        }
        object.__setattr__(self, "macro", metrics)
        object.__setattr__(self, "micro", metrics)
        object.__setattr__(self, "subset_macro", metrics)
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
    def test_status_is_values_only_and_reports_paths(self) -> None:
        output = io.StringIO()
        self.assertEqual(main_antigravity(("status",), stdout=output), 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value["experiment"], "Exp472")
        self.assertEqual(set(value["paths"]), {"root", "smoke", "official", "evaluation", "report", "progress"})
        self.assertNotIn("label_access", __import__("sys").modules)

    def test_evaluate_refuses_incomplete_runtime_before_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "1,117"):
            main_antigravity(("evaluate",), runtime={"verify_exp471": lambda: object()})

    def test_report_writes_exact_leaves_manifest_marker_and_correct_axes(self) -> None:
        with TemporaryDirectory(prefix="exp472-report-antigravity-") as temporary:
            root = Path(temporary) / "report_antigravity"
            authority = write_exp472_report_antigravity(
                _EvaluationAntigravity(), root, _worker_antigravity(),
                e_manifest_sha256=_SHA, code_map_lock_sha256=_SHA,
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

    def test_report_is_byte_identical_and_detects_leaf_manifest_and_marker_tampering(self) -> None:
        with TemporaryDirectory(prefix="exp472-report-antigravity-") as temporary:
            root = Path(temporary) / "report_antigravity"
            first = write_exp472_report_antigravity(_EvaluationAntigravity(), root, _worker_antigravity(), e_manifest_sha256=_SHA, code_map_lock_sha256=_SHA)
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            second = write_exp472_report_antigravity(_EvaluationAntigravity(), root, _worker_antigravity(), e_manifest_sha256=_SHA, code_map_lock_sha256=_SHA)
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
            authority = write_exp472_report_antigravity(_EvaluationAntigravity(), root, _worker_antigravity(), e_manifest_sha256=_SHA, code_map_lock_sha256=_SHA)
            leaf = root / "exp472_pr_coordinates_antigravity.csv"; target = root / "symlink-target-antigravity.csv"
            target.write_bytes(leaf.read_bytes()); leaf.unlink(); leaf.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                verify_exp472_report_antigravity(root, authority)
        scorer = mock.Mock(side_effect=AssertionError("report must not score"))
        with self.assertRaisesRegex(ValueError, "1,117"):
            main_antigravity(("report",), runtime={"verify_exp471": lambda: object(), "score_one": scorer})
        scorer.assert_not_called()
        verifier = mock.Mock(return_value=object())
        with self.assertRaisesRegex(ValueError, "1,117"):
            main_antigravity(("verify",), runtime={"verify_exp471": verifier})
        verifier.assert_not_called()
