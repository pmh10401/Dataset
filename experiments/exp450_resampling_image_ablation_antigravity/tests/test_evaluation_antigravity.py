"""Fixture-only contracts for Exp450 continuous-score evaluation."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.exp450_resampling_image_ablation_antigravity.evaluation_antigravity import (
    METHOD_ORDER,
    curve_tables_exp450_antigravity,
    evaluate_exp450_antigravity,
    metric_tables_exp450_antigravity,
)
from experiments.paper_public_baselines.runner import FrozenScoreManifest
from experiments.exp450_resampling_image_ablation_antigravity.artifacts_antigravity import (
    write_dataset_condition_success_antigravity,
)
from experiments.exp450_resampling_image_ablation_antigravity.runner_antigravity import (
    freeze_all_conditions_antigravity,
)
from experiments.exp450_resampling_image_ablation_antigravity.verifier_antigravity import (
    verify_exp450_score_root_antigravity,
)
from experiments.exp450_resampling_image_ablation_antigravity.tests.test_runner_verifier_antigravity import (
    CONDITIONS as TASK6_CONDITIONS,
    _catalog as task6_catalog,
    _context as task6_context,
    _raw as task6_raw,
    _result as task6_result,
    _score as task6_score,
)


DB_SHA256 = "a" * 64
CATALOG_SHA256 = "b" * 64
DATASETS = ("mixed_antigravity", "normal_antigravity")
LABELS = {
    "mixed_antigravity": (0, 1, 0),
    "normal_antigravity": (0, 0),
}


def _identity_for(dataset: str, dataset_id: int, count: int) -> dict[str, object]:
    bare = {
        "dataset": dataset,
        "dataset_id": dataset_id,
        "series_length": 3,
        "split": "TEST",
        "indices": list(range(count)),
        "values_blob_sha256": [
            hashlib.sha256(f"{dataset}:{index}".encode()).hexdigest()
            for index in range(count)
        ],
    }
    encoded = __import__("json").dumps(bare, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    return {**bare, "row_digest": hashlib.sha256(encoded).hexdigest(), "count": count}


def _identity(dataset: str, count: int) -> dict[str, object]:
    return _identity_for(dataset, DATASETS.index(dataset), count)


def _condition_record(condition: str) -> dict[str, str]:
    modes = {
        "A": ("linear", "linear"), "B": ("polyphase_pchip", "linear"),
        "C": ("linear", "area"), "D": ("polyphase_pchip", "area"),
    }
    length_mode, image_mode = modes[condition]
    return {"condition_id": condition, "length_mode": length_mode, "image_mode": image_mode}


MANIFEST_FIELDS = {
    condition: {
        "database_sha256": DB_SHA256,
        "context_sha256": "c" * 64,
        "catalog_sha256": CATALOG_SHA256,
        "condition": _condition_record(condition),
    }
    for condition in ("A", "B", "C", "D")
}


def _manifest(condition: str, values: dict[str, tuple[float, ...]]) -> dict[str, object]:
    return {
        "condition": _condition_record(condition),
        "database_sha256": DB_SHA256,
        "context_sha256": MANIFEST_FIELDS[condition]["context_sha256"],
        "catalog_sha256": CATALOG_SHA256,
        "datasets": [
            {"dataset": dataset, "test": _identity(dataset, len(values[dataset]))}
            for dataset in DATASETS
        ],
        "test_identities": [_identity(dataset, len(values[dataset])) for dataset in DATASETS],
        "score_rows": values,
    }


EXP_SCORES = {
    "A": {"mixed_antigravity": (0.1, 0.8, 0.2), "normal_antigravity": (0.5, 0.4)},
    "B": {"mixed_antigravity": (0.1, 0.9, 0.2), "normal_antigravity": (0.5, 0.4)},
    "C": {"mixed_antigravity": (0.8, 0.4, 0.2), "normal_antigravity": (0.5, 0.4)},
    "D": {"mixed_antigravity": (0.2, 0.7, 0.1), "normal_antigravity": (0.5, 0.4)},
}
MANIFESTS = {condition: _manifest(condition, scores) for condition, scores in EXP_SCORES.items()}


def _baseline_manifest(methods: tuple[str, ...]) -> dict[str, object]:
    return {
        "database_sha256": DB_SHA256,
        "datasets": [
            {"dataset": dataset, "test": _identity(dataset, len(LABELS[dataset]))}
            for dataset in DATASETS
        ],
        "test_identities": [_identity(dataset, len(LABELS[dataset])) for dataset in DATASETS],
        "methods": {
            method: {
                dataset: {
                    "status": "complete",
                    "test_indices": list(range(len(LABELS[dataset]))),
                    "train_oof_scores": [0.05, 0.15, 0.25, 0.35],
                    "test_scores": list(EXP_SCORES["A"][dataset]),
                }
                for dataset in DATASETS
            }
            for method in methods
        },
    }


BASELINE_MANIFESTS = {
    "classical": _baseline_manifest(("knn_distance", "lof_novelty", "ocsvm_rbf", "isolation_forest")),
    "deep_svdd": _baseline_manifest(("deep_svdd_1d",)),
    "drocc": _baseline_manifest(("drocc_1d",)),
}
BASELINE_ROOTS = {name: Path(f"/fixture/{name}_antigravity") for name in BASELINE_MANIFESTS}
ALL_SCORES = {**EXP_SCORES, **{method: EXP_SCORES["A"] for method in METHOD_ORDER[4:]}}


def _baseline_verifier(events: list[str] | None = None, manifests=None):
    verified = BASELINE_MANIFESTS if manifests is None else manifests

    def verify(root: Path) -> dict[str, object]:
        name = root.name.removesuffix("_antigravity")
        if events is not None:
            events.append(name)
        return verified[name]

    return verify


def _rebind_test_identity(identity: dict[str, object]) -> None:
    bare = {key: value for key, value in identity.items() if key not in {"row_digest", "count"}}
    encoded = __import__("json").dumps(bare, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    identity["row_digest"] = hashlib.sha256(encoded).hexdigest()


class Exp450EvaluationTests(unittest.TestCase):
    def test_evaluator_consumes_actual_task6_verifier_output_without_context_adapter(self) -> None:
        """Task7 must load the pure Task6 v2 verifier result directly, not a forged wrapper."""
        with TemporaryDirectory(prefix="exp450_task6_output_antigravity_") as directory:
            root = Path(directory) / "run_antigravity"
            raw = task6_raw("task6_output_antigravity")
            catalog = task6_catalog((raw,))
            context = task6_context(root, catalog)
            for condition in TASK6_CONDITIONS:
                write_dataset_condition_success_antigravity(root, context, catalog, raw, task6_result(raw, condition))
            freeze_all_conditions_antigravity(
                {condition.condition_id: task6_score(raw, condition) for condition in TASK6_CONDITIONS},
                root, context, catalog,
            )
            dataset = raw.dataset
            identity = catalog["datasets"][0]["test"]
            methods_by_root = {
                "classical": ("knn_distance", "lof_novelty", "ocsvm_rbf", "isolation_forest"),
                "deep_svdd": ("deep_svdd_1d",), "drocc": ("drocc_1d",),
            }
            baselines = {
                name: {
                    "database_sha256": context.database_sha256,
                    "datasets": [{"dataset": dataset, "test": identity}], "test_identities": [identity],
                    "methods": {
                        method: {dataset: {"status": "complete", "test_indices": [0, 1], "train_oof_scores": [0.1, 0.2, 0.3, 0.4], "test_scores": [0.1, 0.9]}}
                        for method in methods
                    },
                }
                for name, methods in methods_by_root.items()
            }
            roots = {name: Path(f"/fixture/{name}_antigravity") for name in baselines}

            result = evaluate_exp450_antigravity(
                root, baseline_roots=roots,
                verifier=lambda condition: verify_exp450_score_root_antigravity(root, context=context, catalog=catalog, condition_id=condition),
                label_loader=lambda datasets: {dataset: (0, 1)},
                baseline_verifier=_baseline_verifier(manifests=baselines),
            )
            self.assertEqual(result.metric_tables.method_order, METHOD_ORDER)

    def test_frozen_score_manifest_baseline_infers_raw_identity_only_from_matching_db(self) -> None:
        """Legacy public manifests need exact DB plus TEST coordinates, never a weak identity flag."""
        with TemporaryDirectory(prefix="exp450_legacy_baseline_antigravity_") as directory:
            base = Path(directory)

            def legacy(name: str, methods: tuple[str, ...], database_sha256: str) -> FrozenScoreManifest:
                root = base / f"{name}_antigravity"
                root.mkdir()
                (root / "run_config.json").write_text(json.dumps({"datasets": list(DATASETS)}), encoding="utf-8")
                entries = []
                for method in methods:
                    for dataset in DATASETS:
                        relative = Path("shards") / method / f"{dataset}.json"
                        path = root / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps({"method": method, "dataset": dataset, "n_test": len(LABELS[dataset]), "test_indices": list(range(len(LABELS[dataset]))), "status": "complete", "train_oof_scores": [0.1, 0.2, 0.3, 0.4], "test_scores": list(EXP_SCORES["A"][dataset])}), encoding="utf-8")
                        entries.append({"method": method, "dataset": dataset, "path": relative.as_posix()})
                return FrozenScoreManifest(root, "d" * 64, database_sha256, len(entries), tuple(entries))

            legacy_manifests = {
                "classical": legacy("classical", ("knn_distance", "lof_novelty", "ocsvm_rbf", "isolation_forest"), DB_SHA256),
                "deep_svdd": legacy("deep_svdd", ("deep_svdd_1d",), DB_SHA256),
                "drocc": legacy("drocc", ("drocc_1d",), DB_SHA256),
            }
            events: list[str] = []

            def verifier(root: Path):
                name = root.name.removesuffix("_antigravity")
                events.append(name)
                return legacy_manifests[name]

            result = evaluate_exp450_antigravity(
                Path("/fixture/root_antigravity"), baseline_roots=BASELINE_ROOTS,
                verifier=lambda condition: MANIFESTS[condition], label_loader=lambda _: LABELS,
                baseline_verifier=verifier,
            )
            self.assertEqual(result.metric_tables.method_order, METHOD_ORDER)
            self.assertEqual(events, ["classical", "deep_svdd", "drocc"])

            labels_called = False
            legacy_manifests["drocc"] = legacy("wrong_db", ("drocc_1d",), "e" * 64)

            def labels(_: tuple[str, ...]):
                nonlocal labels_called
                labels_called = True
                return LABELS

            with self.assertRaisesRegex(ValueError, "baseline database identity"):
                evaluate_exp450_antigravity(
                    Path("/fixture/root_antigravity"), baseline_roots=BASELINE_ROOTS,
                    verifier=lambda condition: MANIFESTS[condition], label_loader=labels,
                    baseline_verifier=verifier,
                )
            self.assertFalse(labels_called)

    def test_labels_are_loaded_only_after_all_four_verifiers_return(self) -> None:
        """Loading labels before D verifies could leak unverified results into evaluation."""
        events: list[str] = []

        def verify(condition: str) -> dict[str, object]:
            events.append(f"verify-{condition}")
            return MANIFESTS[condition]

        def labels(datasets: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
            events.append("labels")
            self.assertEqual(datasets, DATASETS)
            return LABELS

        evaluate_exp450_antigravity(
            Path("/fixture/root_antigravity"), baseline_roots=BASELINE_ROOTS,
            verifier=verify, label_loader=labels, baseline_verifier=_baseline_verifier(events),
        )
        self.assertEqual(events, ["verify-A", "verify-B", "verify-C", "verify-D", "classical", "deep_svdd", "drocc", "labels"])

    def test_verifier_authority_needs_no_independent_expected_context(self) -> None:
        """Co-forging a verifier output and an external expected context must be impossible."""
        result = evaluate_exp450_antigravity(
            Path("/fixture/root_antigravity"), baseline_roots=BASELINE_ROOTS,
            verifier=lambda condition: MANIFESTS[condition], label_loader=lambda _: LABELS,
            baseline_verifier=_baseline_verifier(),
        )
        self.assertEqual(result.metric_tables.method_order, METHOD_ORDER)

    def test_missing_baseline_blocks_the_label_loader(self) -> None:
        """A missing frozen public root must fail before TEST labels become accessible."""
        labels_called = False

        def labels(_: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
            nonlocal labels_called
            labels_called = True
            return LABELS

        incomplete = dict(BASELINE_ROOTS)
        incomplete.pop("drocc")
        with self.assertRaisesRegex(ValueError, "baseline roots"):
            evaluate_exp450_antigravity(
                Path("/fixture/root_antigravity"), baseline_roots=incomplete,
                verifier=lambda condition: MANIFESTS[condition], label_loader=labels,
                baseline_verifier=_baseline_verifier(),
            )
        self.assertFalse(labels_called)

    def test_identity_mismatch_blocks_labels_after_the_full_verification_pass(self) -> None:
        """A mismatched D TEST population must not reach the lazy label capability."""
        labels_called = False
        mismatched = deepcopy(MANIFESTS)
        mismatched["D"]["datasets"][1]["test"]["count"] = 99

        def labels(_: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
            nonlocal labels_called
            labels_called = True
            return LABELS

        with self.assertRaisesRegex(ValueError, "TEST identity"):
            evaluate_exp450_antigravity(
                Path("/fixture/root_antigravity"), baseline_roots=BASELINE_ROOTS,
                verifier=lambda condition: mismatched[condition], label_loader=labels,
                baseline_verifier=_baseline_verifier(),
            )
        self.assertFalse(labels_called)

    def test_context_catalog_database_and_full_row_identity_mismatches_block_labels(self) -> None:
        """Rebound provenance or same-size altered TEST rows must not cross the label gate."""
        mutations = {
            "context": lambda values: values["D"].__setitem__("context_sha256", "e" * 64),
            "catalog": lambda values: values["D"].__setitem__("catalog_sha256", "f" * 64),
            "database": lambda values: values["D"].__setitem__("database_sha256", "0" * 64),
            "value_digest": lambda values: values["D"]["datasets"][0]["test"]["values_blob_sha256"].__setitem__(0, "9" * 64),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                labels_called = False
                altered = deepcopy(MANIFESTS)
                mutate(altered)

                def labels(_: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
                    nonlocal labels_called
                    labels_called = True
                    return LABELS

                with self.assertRaises(ValueError):
                    evaluate_exp450_antigravity(
                        Path("/fixture/root_antigravity"), baseline_roots=BASELINE_ROOTS,
                        verifier=lambda condition: altered[condition], label_loader=labels,
                        baseline_verifier=_baseline_verifier(),
                    )
                self.assertFalse(labels_called)

    def test_baseline_identity_mismatch_blocks_labels_after_all_baseline_verifiers(self) -> None:
        """A baseline with matching scores but a different TEST identity is not reusable."""
        events: list[str] = []
        labels_called = False
        altered = deepcopy(BASELINE_MANIFESTS)
        altered["drocc"]["datasets"][0]["test"]["values_blob_sha256"][0] = "9" * 64
        _rebind_test_identity(altered["drocc"]["datasets"][0]["test"])
        altered["drocc"]["test_identities"][0] = deepcopy(altered["drocc"]["datasets"][0]["test"])

        def labels(_: tuple[str, ...]) -> dict[str, tuple[int, ...]]:
            nonlocal labels_called
            labels_called = True
            return LABELS

        with self.assertRaisesRegex(ValueError, "baseline TEST identity"):
            evaluate_exp450_antigravity(
                Path("/fixture/root_antigravity"), baseline_roots=BASELINE_ROOTS,
                verifier=lambda condition: (events.append(condition) or MANIFESTS[condition]),
                label_loader=labels, baseline_verifier=_baseline_verifier(events, altered),
            )
        self.assertEqual(events, ["A", "B", "C", "D", "classical", "deep_svdd", "drocc"])
        self.assertFalse(labels_called)

    def test_baseline_dataset_metrics_use_raw_scores_while_micro_uses_calibration(self) -> None:
        """Using raw cross-dataset values for micro would change the frozen ranking protocol."""
        labels = {"one_antigravity": (0, 1), "two_antigravity": (0, 1)}
        datasets = tuple(labels)
        test_identity = {
            dataset: _identity_for(dataset, index, len(labels[dataset]))
            for index, dataset in enumerate(datasets)
        }
        raw_scores = {"one_antigravity": (0.1, 0.9), "two_antigravity": (100.0, 99.0)}
        manifests = {
            condition: {
                "condition": _condition_record(condition), "database_sha256": DB_SHA256,
                "context_sha256": MANIFEST_FIELDS[condition]["context_sha256"],
                "catalog_sha256": CATALOG_SHA256,
                "datasets": [{"dataset": dataset, "test": test_identity[dataset]} for dataset in datasets],
                "test_identities": [test_identity[dataset] for dataset in datasets],
                "score_rows": raw_scores,
            }
            for condition in ("A", "B", "C", "D")
        }
        baseline_methods = {
            "classical": ("knn_distance", "lof_novelty", "ocsvm_rbf", "isolation_forest"),
            "deep_svdd": ("deep_svdd_1d",), "drocc": ("drocc_1d",),
        }
        train_scores = {
            "one_antigravity": [0.2, 0.3, 0.4, 0.5],
            "two_antigravity": [99.1, 99.2, 99.3, 99.4],
        }
        baselines = {
            root: {
                "database_sha256": DB_SHA256,
                "datasets": [{"dataset": dataset, "test": test_identity[dataset]} for dataset in datasets],
                "test_identities": [test_identity[dataset] for dataset in datasets],
                "methods": {
                    method: {
                        dataset: {"status": "complete", "test_indices": [0, 1],
                                  "train_oof_scores": train_scores[dataset], "test_scores": list(raw_scores[dataset])}
                        for dataset in datasets
                    }
                    for method in methods
                },
            }
            for root, methods in baseline_methods.items()
        }
        roots = {name: Path(f"/fixture/{name}_antigravity") for name in baselines}
        result = evaluate_exp450_antigravity(
            Path("/fixture/root_antigravity"), baseline_roots=roots,
            verifier=lambda condition: manifests[condition], label_loader=lambda _: labels,
            baseline_verifier=_baseline_verifier(manifests=baselines),
        ).metric_tables
        per_dataset = next(row for row in result.per_dataset if row["method"] == "knn_distance" and row["dataset"] == "two_antigravity")
        self.assertEqual((per_dataset["average_precision"], per_dataset["auroc"]), (0.5, 0.0))
        self.assertEqual((result.micro["knn_distance"]["average_precision"], result.micro["knn_distance"]["auroc"]), (0.5, 0.5))

    def test_curve_axes_are_not_swapped(self) -> None:
        """Swapping ROC or PR coordinate columns makes the paper curves false."""
        curves = curve_tables_exp450_antigravity(LABELS, ALL_SCORES)
        self.assertEqual(curves["roc"]["x_name"], "false_positive_rate")
        self.assertEqual(curves["roc"]["y_name"], "true_positive_rate")
        self.assertEqual(curves["precision_recall"]["x_name"], "recall")
        self.assertEqual(curves["precision_recall"]["y_name"], "precision")
        first_roc = next(row for row in curves["roc"]["rows"] if row["method"] == "A")
        first_pr = next(row for row in curves["precision_recall"]["rows"] if row["method"] == "A")
        self.assertEqual((first_roc["false_positive_rate"], first_roc["true_positive_rate"]), (0.0, 0.0))
        self.assertEqual((first_pr["recall"], first_pr["precision"]), (1.0, 1.0 / 5.0))

    def test_all_four_variants_and_six_baselines_are_present(self) -> None:
        """Omitting a method changes the preregistered comparison population."""
        result = metric_tables_exp450_antigravity(LABELS, ALL_SCORES)
        self.assertEqual(result.method_order, METHOD_ORDER)
        self.assertEqual({row["method"] for row in result.per_dataset}, set(METHOD_ORDER))

    def test_single_class_denominators_are_explicit(self) -> None:
        """Silently treating normal-only data as valid AP/AUROC biases macro results."""
        result = metric_tables_exp450_antigravity(LABELS, ALL_SCORES)
        normal = next(
            row for row in result.per_dataset
            if row["method"] == "A" and row["dataset"] == "normal_antigravity"
        )
        summary = result.macro["A"]
        self.assertIsNone(normal["average_precision"])
        self.assertEqual(normal["average_precision_exclusion"], "no_positive_test_labels")
        self.assertIsNone(normal["auroc"])
        self.assertEqual(normal["auroc_exclusion"], "single_class_test_labels")
        self.assertEqual(summary["average_precision_denominator"], 1)
        self.assertEqual(summary["auroc_denominator"], 1)

    def test_nonfinite_scores_are_rejected(self) -> None:
        """NaN rankings would invalidate every AP/AUROC comparison."""
        invalid = deepcopy(ALL_SCORES)
        invalid["A"] = dict(invalid["A"])
        invalid["A"]["mixed_antigravity"] = (0.1, float("nan"), 0.2)
        with self.assertRaisesRegex(ValueError, "finite"):
            metric_tables_exp450_antigravity(LABELS, invalid)

    def test_paired_bootstrap_uses_shared_seed_and_fixed_order(self) -> None:
        """Changing seed or pairing order changes the preregistered confidence intervals."""
        labels = {"d1_antigravity": (0, 1), "d2_antigravity": (0, 1)}
        scores = {
            method: {
                "d1_antigravity": (0.1, 0.9),
                "d2_antigravity": (0.2, 0.8),
            }
            for method in METHOD_ORDER
        }
        scores["B"] = {"d1_antigravity": (0.1, 0.95), "d2_antigravity": (0.7, 0.3)}
        result = metric_tables_exp450_antigravity(labels, scores)
        statistic = next(
            row for row in result.paired_statistics
            if row["comparison"] == "B-A" and row["metric"] == "average_precision"
        )
        draws = np.random.default_rng(45020260820).integers(0, 2, size=(20_000, 2), dtype=np.int64)
        self.assertEqual(statistic["seed"], 45020260820)
        self.assertEqual(statistic["draw_count"], 20_000)
        self.assertEqual(statistic["dataset_order"], ("d1_antigravity", "d2_antigravity"))
        self.assertEqual(statistic["draw_sha256"], hashlib.sha256(draws.tobytes(order="C")).hexdigest())
        self.assertEqual((statistic["wins"], statistic["ties"], statistic["losses"]), (0, 1, 1))


if __name__ == "__main__":
    unittest.main()
