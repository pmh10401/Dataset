from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

from services.time_series_anomaly_diagnosis.anomaly_detection import FIXED_SEEDS
from services.time_series_anomaly_diagnosis import policy_qualification as qualification
from services.time_series_anomaly_diagnosis.policy_qualification import (
    BackendResult, PolicyMetrics, benchmark_backends, build_execution_contract, build_value_manifest,
    build_execution_contract_record,
    evaluate_frozen_predictions, freeze_predictions, load_frozen_predictions, load_official_population,
    load_score_checkpoint, main, render_progress_dashboard, score_with_checkpoint,
    validate_results_bundle,
    select_backend, select_qualified_policy,
    _baseline_prediction, _parser, _router_from_payload, _router_payload,
    _digest, _freeze_then_load_labels, _persist_execution_contract,
    _score_jobs, _synthetic_c_check,
)
from services.time_series_anomaly_diagnosis.dataset_repository import PreparedDatasetValues
from services.time_series_anomaly_diagnosis.score_sources import METHOD_NAMES, ScoreResult


def blob(values):
    return np.asarray(values, dtype="<f4").tobytes()


def database() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        "CREATE TABLE datasets(id INTEGER PRIMARY KEY, name TEXT UNIQUE);"
        "CREATE TABLE instances(id INTEGER PRIMARY KEY, dataset_id INTEGER, split TEXT, "
        "instance_index INTEGER, label TEXT, values_blob BLOB);"
    )
    next_row = 1
    for dataset_id, name in enumerate(("good", "bad"), 1):
        connection.execute("INSERT INTO datasets VALUES (?, ?)", (dataset_id, name))
        for index in (range(5) if name == "good" else range(4)):
            connection.execute("INSERT INTO instances VALUES (?, ?, 'TRAIN', ?, '0', ?)",
                               (next_row, dataset_id, index, blob([index, index + 1, index + 3])))
            next_row += 1
        for index, label in enumerate(("0", "1")):
            connection.execute("INSERT INTO instances VALUES (?, ?, 'TEST', ?, ?, ?)",
                               (next_row, dataset_id, index, label, blob([index, index + 2, index + 5])))
            next_row += 1
    connection.commit()
    return connection


def prediction(policy="single_minirocket", automatic=(False, False), review=(False, False)):
    return {"policy_id": policy, "dataset": "good", "test_indices": [0, 1],
            "evaluation_scores": [0.1, 0.9], "p_values": [0.9, 0.1],
            "automatic": list(automatic), "review": list(review),
            "selected_method": METHOD_NAMES[0],
            "method_p_values": [{METHOD_NAMES[0]: 0.9}, {METHOD_NAMES[0]: 0.1}],
            "method_seed_p_values": [{METHOD_NAMES[0]: [0.9, 0.9]},
                                     {METHOD_NAMES[0]: [0.1, 0.1]}]}


def process_identity(name):
    return name, os.getpid()


def metrics(policy_id, fn, family_fn, review_fp, automatic_fp, *, complete=True):
    return PolicyMetrics(policy_id, complete, None if complete else "incomplete", "x", 1, 2, 0,
                         fn, family_fn, family_fn, review_fp, automatic_fp, 0.5, 0.5, 0.0)


class ManifestAndAuthorityTest(unittest.TestCase):
    def test_manifest_is_deterministic_label_free_and_keeps_failures(self):
        connection = database()
        self.addCleanup(connection.close)
        first = build_value_manifest(connection, ["good", "bad"])
        second = build_value_manifest(connection, ["bad", "good"])
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.requested_names, ("bad", "good"))
        self.assertEqual(first.total_requested, 2)
        self.assertEqual(first.eligible_names, ("good",))
        self.assertEqual(first.eligible_count, 1)
        self.assertEqual(len(first.statuses), 2)
        self.assertEqual(first.statuses[1].test_indices, (0, 1))
        encoded = first.to_json().lower()
        for forbidden in ("label", "metric", "candidate_indices", "review_indices", "method", "route"):
            self.assertNotIn(forbidden, encoded)
        connection.execute("UPDATE instances SET values_blob=? WHERE dataset_id=2 AND split='TRAIN' AND instance_index=0", (blob([9, 8, 7]),))
        self.assertNotEqual(first.digest, build_value_manifest(connection, ["good", "bad"]).digest)
        connection.execute("UPDATE instances SET values_blob='malformed' WHERE dataset_id=2 AND split='TRAIN' AND instance_index=0")
        self.assertFalse(build_value_manifest(connection, ["bad"]).statuses[0].eligible)
        with self.assertRaises(ValueError):
            build_value_manifest(connection, ["good", "good"])

    def test_duplicate_source_test_indices_are_ineligible_and_cannot_load(self):
        connection = database()
        self.addCleanup(connection.close)
        connection.execute("UPDATE instances SET instance_index=0 WHERE dataset_id=1 AND split='TEST'")
        manifest = build_value_manifest(connection, ["good"])
        self.assertEqual(manifest.total_requested, 1)
        self.assertEqual(manifest.eligible_count, 0)
        self.assertIn("duplicate TEST indices", manifest.statuses[0].failure)

        valid_connection = database()
        self.addCleanup(valid_connection.close)
        payload = build_value_manifest(valid_connection, ["good"]).as_dict()
        payload["statuses"][0]["test_indices"] = [0, 0]
        payload["digest"] = _digest({"requested_names": payload["requested_names"],
                                     "statuses": payload["statuses"]})
        with self.assertRaises(ValueError):
            type(manifest).from_dict(payload)

    def test_loaded_manifest_requires_exact_schema_counts_and_one_status_per_name(self):
        connection = database()
        self.addCleanup(connection.close)
        manifest = build_value_manifest(connection, ["good", "bad"])
        payload = manifest.as_dict()
        for mutate in (
            lambda value: value.update({"unknown": 1}),
            lambda value: value["statuses"].append(dict(value["statuses"][0])),
            lambda value: value.update({"total_requested": 99}),
            lambda value: value.update({"requested_names": ["good"]}),
        ):
            broken = json.loads(json.dumps(payload)); mutate(broken)
            with self.assertRaises(ValueError):
                type(manifest).from_dict(broken)

    def test_test_indices_reject_boolean_and_float_types_at_manifest_boundaries(self):
        connection = database()
        self.addCleanup(connection.close)
        manifest = build_value_manifest(connection, ["good"])
        for invalid in ([False, 1], [0.0, 1]):
            payload = manifest.as_dict()
            payload["statuses"][0]["test_indices"] = invalid
            payload["digest"] = _digest({"requested_names": payload["requested_names"],
                                         "statuses": payload["statuses"]})
            with self.assertRaises(ValueError):
                type(manifest).from_dict(payload)

            prepared = PreparedDatasetValues("good", np.zeros((5, 3)), np.zeros((2, 3)),
                                             tuple(invalid), "value_only_numba",
                                             tuple(np.zeros(3) for _ in range(5)))
            with patch.object(qualification, "prepare_dataset_values", return_value=prepared):
                status = build_value_manifest(connection, ["good"]).statuses[0]
            self.assertFalse(status.eligible)
            self.assertIn("TEST indices", status.failure)

    def test_official_source_is_digest_count_and_family_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.json"
            payload = {"evaluation": {"decisions": [
                {"dataset": "b", "held_family": "f2", "selected_method": "secret"},
                {"dataset": "a", "held_family": "f1", "selected_method": "secret"}]}}
            path.write_text(json.dumps(payload), encoding="utf-8")
            digest = sha256(path.read_bytes()).hexdigest()
            names, families = load_official_population(path, digest, expected_count=2)
            self.assertEqual(names, ("a", "b"))
            self.assertEqual(families, {"a": "f1", "b": "f2"})
            with self.assertRaises(ValueError):
                load_official_population(path, "0" * 64, expected_count=2)
            with self.assertRaises(ValueError):
                load_official_population(path, digest, expected_count=3)

    def test_real_official_source_metadata_only(self):
        path = Path("experiments/exp481_strict_evidence_residual_correction_antigravity/results_antigravity.json")
        digest = sha256(path.read_bytes()).hexdigest()
        self.assertEqual(digest, "7038b994e422dc93a18927c810ce5e3bdb7601039ba01d2b388895444e09956f")
        names, families = load_official_population(path, digest)
        self.assertEqual(len(names), 1117)
        self.assertEqual(set(names), set(families))


class CheckpointAndFreezeTest(unittest.TestCase):
    def setUp(self):
        self.connection = database()
        self.addCleanup(self.connection.close)
        self.manifest = build_value_manifest(self.connection, ["good"])
        self.contract = build_execution_contract(self.manifest, "population", {"good": "family"}, {"cpu": True})

    def test_execution_contract_retains_canonical_payload_and_provenance(self):
        record = build_execution_contract_record(self.manifest, "population", {"good": "family"},
                                                 {"cpu": True}, workers=3)
        self.assertEqual(record.sha256, sha256(record.canonical_payload).hexdigest())
        self.assertEqual(record.payload["python_version"], list(__import__("sys").version_info[:3]))
        self.assertEqual(record.payload["execution"]["workers"], 3)
        self.assertIn("backend_identity", record.payload["execution"])
        self.assertIn("mlx-metal", record.payload["versions"])

    def test_contract_and_checkpoint_reuse_complete_scores(self):
        calls = []
        def scorer(train, test, method, seed):
            calls.append((method, seed))
            return ScoreResult(np.arange(len(train), dtype=float), np.arange(len(test), dtype=float), 7, seed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            first = score_with_checkpoint(self.connection, self.manifest, self.contract, path, scorer=scorer)
            self.assertEqual(len(calls), 6)
            second = score_with_checkpoint(self.connection, self.manifest, self.contract, path, scorer=scorer, resume=True)
            self.assertEqual(len(calls), 6)
            self.assertEqual(first, second)
            rows = path.read_text().splitlines()
            rows[-1] = rows[-1].replace('"feature_dimension":7', '"feature_dimension":0')
            path.write_text("\n".join(rows) + "\n")
            with self.assertRaises(ValueError):
                load_score_checkpoint(path, self.contract, self.manifest)

    def test_checkpoint_features_can_feed_the_frozen_router_after_json_reload(self):
        from services.time_series_anomaly_diagnosis.routing_policy import MethodMetric, RouterTrainingRecord

        def scorer(train, test, method, seed):
            return ScoreResult(np.arange(len(train), dtype=float), np.arange(len(test), dtype=float), 1, seed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            score_with_checkpoint(self.connection, self.manifest, self.contract, path, scorer=scorer)
            features = load_score_checkpoint(path, self.contract, self.manifest)["good"]["features"]
        metrics_by_method = tuple(MethodMetric(method, 0.5, 0) for method in METHOD_NAMES)
        RouterTrainingRecord(0, features, metrics_by_method)

    def test_checkpoint_requires_exact_nested_schemas_and_metadata_identity(self):
        def scorer(train, test, method, seed):
            return ScoreResult(np.arange(len(train), dtype=float), np.arange(len(test), dtype=float), 1, seed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            score_with_checkpoint(self.connection, self.manifest, self.contract, path, scorer=scorer)
            original = path.read_text().splitlines()
            for mutate in (
                lambda row: row.update({"unknown": 1}),
                lambda row: row["metadata"].update({"unknown": 1}),
                lambda row: row["metadata"].update({"mode": "wrong"}),
                lambda row: row["metadata"].update({"test_indices": [1, 0]}),
                lambda row: row["metadata"].update({"test_indices": [False, 1]}),
                lambda row: row["metadata"].update({"test_indices": [0.0, 1]}),
                lambda row: row["scores"][METHOD_NAMES[0]][str(FIXED_SEEDS[0])].update({"unknown": 1}),
                lambda row: row["scores"][METHOD_NAMES[0]][str(FIXED_SEEDS[0])]["test_scores"].update({"unknown": 1}),
            ):
                row = json.loads(original[1]); mutate(row)
                row["row_sha256"] = _digest({key: value for key, value in row.items() if key != "row_sha256"})
                path.write_text(original[0] + "\n" + json.dumps(row) + "\n")
                with self.assertRaises(ValueError):
                    load_score_checkpoint(path, self.contract, self.manifest)
            with self.assertRaises(ValueError):
                load_score_checkpoint(path, "different", self.manifest)
            score_with_checkpoint(self.connection, self.manifest, self.contract, path, scorer=scorer)
            rows = path.read_text().splitlines(); row = json.loads(rows[1])
            row["scores"][METHOD_NAMES[0]][str(FIXED_SEEDS[0])]["test_scores"]["bytes"] = np.asarray([8.0, 9.0]).tobytes().hex()
            path.write_text(rows[0] + "\n" + json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                load_score_checkpoint(path, self.contract, self.manifest)

    def test_scoring_failure_is_durable_and_keeps_manifest_denominator(self):
        def scorer(train, test, method, seed):
            if method == METHOD_NAMES[1]:
                raise RuntimeError("synthetic failure")
            return ScoreResult(np.arange(len(train), dtype=float), np.arange(len(test), dtype=float), 1, seed)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "score failures"):
                score_with_checkpoint(self.connection, self.manifest, self.contract,
                                      root / "scores.jsonl", scorer=scorer,
                                      failure_path=root / "failures.jsonl")
            lines = [json.loads(line) for line in (root / "failures.jsonl").read_text().splitlines()]
            self.assertEqual(lines[0]["manifest_requested"], self.manifest.total_requested)
            self.assertEqual(lines[1]["dataset"], "good")
            self.assertEqual(lines[1]["method"], METHOD_NAMES[1])
            self.assertEqual(lines[1]["seed"], FIXED_SEEDS[0])
            self.assertEqual(lines[1]["status"], "failed")

    def test_resume_rejects_an_incomplete_dataset_row(self):
        calls = []
        def scorer(train, test, method, seed):
            calls.append((method, seed))
            return ScoreResult(np.arange(len(train), dtype=float), np.arange(len(test), dtype=float), 1, seed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            score_with_checkpoint(self.connection, self.manifest, self.contract, path, scorer=scorer)
            lines = path.read_text().splitlines()
            row = json.loads(lines[1]); del row["scores"][METHOD_NAMES[0]][str(FIXED_SEEDS[0])]
            path.write_text(lines[0] + "\n" + json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                score_with_checkpoint(self.connection, self.manifest, self.contract, path, scorer=scorer, resume=True)

    def test_resume_recomputes_value_identity_before_reusing_scores(self):
        calls = []
        def scorer(train, test, method, seed):
            calls.append((method, seed))
            return ScoreResult(np.arange(len(train), dtype=float), np.arange(len(test), dtype=float), 1, seed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scores.jsonl"
            score_with_checkpoint(self.connection, self.manifest, self.contract, path, scorer=scorer)
            calls.clear()
            self.connection.execute("UPDATE instances SET values_blob=? WHERE dataset_id=1 AND split='TEST' AND instance_index=0",
                                    (blob([100, 200, 500]),))
            with self.assertRaisesRegex(ValueError, "value manifest"):
                score_with_checkpoint(self.connection, self.manifest, self.contract, path, scorer=scorer, resume=True)
            self.assertEqual(calls, [])

    def test_freeze_validates_alignment_coverage_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.json"
            frozen = freeze_predictions([prediction()], self.manifest, path, contract_sha="contract")
            self.assertEqual((frozen["dataset_count"], frozen["row_count"]), (1, 2))
            self.assertNotIn("label", path.read_text().lower())
            self.assertEqual(load_frozen_predictions(path, self.manifest, contract_sha="contract")["prediction_sha256"], frozen["prediction_sha256"])
            with self.assertRaises(ValueError):
                load_frozen_predictions(path, self.manifest, contract_sha="different")
            payload = json.loads(path.read_text()); payload["predictions"][0]["evaluation_scores"][0] = 0.2
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_frozen_predictions(path, self.manifest, contract_sha="contract")
            freeze_predictions([prediction()], self.manifest, path, contract_sha="contract")
            with self.assertRaises(ValueError):
                freeze_predictions([prediction(), prediction()], self.manifest, path, contract_sha="contract")
            broken = prediction(); broken["test_indices"] = [0, 0]
            with self.assertRaises(ValueError):
                freeze_predictions([broken], self.manifest, path, contract_sha="contract")
            duplicate_status = replace(self.manifest.statuses[0], test_indices=(0, 0))
            duplicate_manifest = replace(self.manifest, statuses=(duplicate_status,))
            with self.assertRaises(ValueError):
                freeze_predictions([broken], duplicate_manifest, path, contract_sha="contract")
            broken = prediction(); broken["p_values"] = [0.5, 1.1]
            with self.assertRaises(ValueError):
                freeze_predictions([broken], self.manifest, path, contract_sha="contract")
            broken = prediction(); broken["automatic"] = [0, 1]
            with self.assertRaises(ValueError):
                freeze_predictions([broken], self.manifest, path, contract_sha="contract")
            broken = prediction(); broken["unknown"] = 1
            with self.assertRaises(ValueError):
                freeze_predictions([broken], self.manifest, path, contract_sha="contract")
            broken = prediction(); broken["test_indices"] = [1, 0]
            with self.assertRaises(ValueError):
                freeze_predictions([broken], self.manifest, path, contract_sha="contract")
            for invalid in ([False, 1], [0.0, 1]):
                broken = prediction(); broken["test_indices"] = invalid
                with self.assertRaises(ValueError):
                    freeze_predictions([broken], self.manifest, path, contract_sha="contract")
            broken = prediction(automatic=(True, False), review=(True, False))
            with self.assertRaises(ValueError):
                freeze_predictions([broken], self.manifest, path, contract_sha="contract")
            broken = prediction(); del broken["method_seed_p_values"][0][METHOD_NAMES[0]]
            with self.assertRaises(ValueError):
                freeze_predictions([broken], self.manifest, path, contract_sha="contract")
            broken = prediction(); broken["method_p_values"][0][METHOD_NAMES[0]] = 0.8
            with self.assertRaises(ValueError):
                freeze_predictions([broken], self.manifest, path, contract_sha="contract")

    def test_phase_order_freezes_before_labels(self):
        events = ["values", "scores"]
        with tempfile.TemporaryDirectory() as directory:
            frozen, labels = _freeze_then_load_labels(
                {"single_minirocket": [prediction()]}, self.manifest, "contract",
                lambda policy: Path(directory) / f"{policy}.json",
                lambda: ({"good": {0: 0, 1: 1}} if events == ["values", "scores", "freeze"]
                         else self.fail("labels opened before freeze")), events)
        self.assertIn("single_minirocket", frozen)
        self.assertEqual(labels["good"][1], 1)
        self.assertEqual(events, ["values", "scores", "freeze", "labels"])

    def test_execution_contract_persistence_reloads_exact_payload(self):
        record = build_execution_contract_record(self.manifest, "population", {"good": "family"},
                                                 {"cpu": True}, workers=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            loaded = _persist_execution_contract(path, record, resume=False)
            self.assertEqual((loaded.payload, loaded.sha256), (record.payload, record.sha256))
            self.assertEqual(_persist_execution_contract(path, record, resume=True).payload, record.payload)
            payload = json.loads(path.read_text()); payload["payload"]["python_version"] = [0, 0, 0]
            payload["sha256"] = _digest(payload["payload"])
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                _persist_execution_contract(path, record, resume=True)

    def test_current_baseline_uses_final_floor_only_without_old_evidence(self):
        prepared = PreparedDatasetValues("good", np.zeros((5, 2)), np.zeros((2, 2)), (0, 1),
                                         "value_only_numba", tuple(np.arange(2.0) for _ in range(5)))
        def scorer(_train, _test, _method, seed):
            return ScoreResult(np.arange(5.0), np.array([9.0, 0.0]), 1, seed)
        row = _baseline_prediction(prepared, METHOD_NAMES[0], scorer)
        self.assertEqual(row["automatic"], [False, False])
        self.assertEqual(row["review"], [True, False])
        self.assertNotIn("hard", json.dumps(row).lower())


class MetricsAndSelectionTest(unittest.TestCase):
    def test_exact_metrics_and_family_macro(self):
        detected = prediction(); detected["p_values"][1] = 0.001; detected["evaluation_scores"][1] = 0.999
        detected["automatic"][1] = True; detected["method_p_values"][1][METHOD_NAMES[0]] = 0.001
        detected["method_seed_p_values"][1][METHOD_NAMES[0]] = [0.001, 0.001]
        predictions = [detected, {**prediction(), "dataset": "other", "review": [True, False]}]
        labels = {"good": {0: 0, 1: 1}, "other": {0: 0, 1: 1}}
        metrics = evaluate_frozen_predictions(predictions, labels, {"good": "f1", "other": "f2"}, elapsed_seconds=2.5)
        self.assertTrue(metrics.complete)
        self.assertEqual((metrics.residual_fn_total, metrics.residual_fn_rate), (1, 0.5))
        self.assertEqual(metrics.family_balanced_residual_fn_rate, 0.5)
        self.assertEqual((metrics.review_inclusive_fp_total, metrics.automatic_fp_total), (1, 0))
        self.assertEqual((metrics.dataset_mean_auc, metrics.family_balanced_auc), (1.0, 1.0))

    def test_evaluation_fails_closed_for_single_class(self):
        metrics = evaluate_frozen_predictions([prediction()], {"good": {0: 0, 1: 0}}, {"good": "f"})
        self.assertFalse(metrics.complete)
        self.assertTrue(metrics.failure)

    def test_strict_qualification_incomplete_and_tie_order(self):
        baseline = metrics("current_baseline", 10, 0.5, 4, 2)
        candidates = [metrics(policy, 8, 0.4, 4, 2)
                      for policy in ("train_feature_router", "three_method_consensus", "single_minirocket")]
        self.assertEqual(select_qualified_policy(baseline, candidates), "single_minirocket")
        self.assertIsNone(select_qualified_policy(baseline, [replace(candidates[0], complete=False, failure="x")]))
        self.assertIsNone(select_qualified_policy(baseline, [replace(candidates[0], residual_fn_total=10)]))


class BackendCliAndProgressTest(unittest.TestCase):
    def test_backend_gate_selected_and_fallback(self):
        accepted = BackendResult("candidate", True, True, True, 1e-12, 2.0, 1.0, 1.5, "")
        self.assertEqual(select_backend("reference", accepted), "candidate")
        for change in ({"finite": False}, {"parity": False}, {"decisions_equal": False}, {"speedup": 1.09}):
            self.assertEqual(select_backend("reference", replace(accepted, **change)), "reference")

    def test_mlx_assessment_has_complete_explicit_provenance(self):
        assessment = benchmark_backends()["router_distance"]
        for key in ("mlx_version", "mlx_metal_version", "device", "cold_seconds",
                    "warm_seconds", "parity", "route_decisions_equal", "selected",
                    "fallback_reason"):
            self.assertIn(key, assessment)
        if not assessment["installed"]:
            self.assertEqual(assessment["device"], "unavailable")
            self.assertIsNone(assessment["cold_seconds"])
            self.assertIsNone(assessment["warm_seconds"])
            self.assertTrue(assessment["fallback_reason"])

    def test_mlx_selection_uses_matched_cold_inclusive_totals(self):
        mlx = ModuleType("mlx"); core = ModuleType("mlx.core")
        core.array = lambda values: np.asarray(values)
        core.sqrt = np.sqrt; core.sum = np.sum; core.eval = lambda _values: None
        core.default_device = lambda: "fake-device"
        mlx.core = core
        resampled = np.zeros(521); pvalues = np.zeros(2000)

        def clock_for(durations):
            durations = iter(durations); elapsed = 0.0; starting = True
            def clock():
                nonlocal elapsed, starting
                if starting:
                    starting = False
                    return elapsed
                elapsed += next(durations); starting = True
                return elapsed
            return clock

        scenarios = (
            ("numpy", (2.0, 0.1, 0.1, 0.01, 9.0)),
            ("mlx", (6.0, 6.0, 9.0, 0.1, 1.0)),
        )
        for selected, route_durations in scenarios:
            with self.subTest(selected=selected), \
                    patch.dict(sys.modules, {"mlx": mlx, "mlx.core": core}), \
                    patch.object(qualification, "perf_counter", clock_for([1.0] * 8 + list(route_durations))), \
                    patch.object(qualification, "resample_series", new=lambda _row, _targets: resampled), \
                    patch.object(qualification.np, "interp", new=lambda *_args: resampled), \
                    patch.object(qualification, "empirical_upper_pvalues", new=lambda _train, _test: pvalues), \
                    patch.object(qualification, "_numpy_pvalues", new=lambda _train, _test: pvalues):
                self.assertEqual(benchmark_backends()["router_distance"]["selected"], selected)

    def test_policy_stage_records_clock_timing_and_forwards_total_to_dashboard(self):
        ticks = iter((10.0, 10.5, 12.0, 13.0))
        captured = {}
        rows, elapsed = qualification._evaluate_policy_stage(
            {"single_minirocket": [prediction()]},
            {"single_minirocket": {"good": {0: 0, 1: 1}}}, {"good": "f"},
            dashboard_path=Path("unused.html"), requested=1, eligible=1,
            clock=lambda: next(ticks), renderer=lambda _path, **values: captured.update(values))
        self.assertEqual(rows[0].elapsed_seconds, 1.5)
        self.assertEqual(elapsed, 3.0)
        self.assertEqual(captured["elapsed_seconds"], 3.0)

    def test_cli_modes_are_mutually_exclusive_and_scoped(self):
        with self.assertRaises(SystemExit):
            main(["--benchmark-backends", "--smoke", "--datasets", "x"])
        with self.assertRaises(SystemExit):
            main(["--official"])
        with self.assertRaises(SystemExit):
            main(["--smoke"])
        with self.assertRaises(SystemExit):
            main(["--benchmark-backends", "--workers", "2"])
        with self.assertRaises(SystemExit):
            main(["--smoke", "--datasets", "x", "--resume"])
        self.assertEqual(_parser().parse_args(["--official", "--resume", "--workers", "7"]).workers, 7)

    def test_official_workers_are_process_isolated_from_numba_workqueue(self):
        completed = list(_score_jobs(("a", "b", "c"), 3, process_identity))
        self.assertEqual({result[0] for result, error in completed if error is None}, {"a", "b", "c"})
        self.assertTrue(all(result[1] != os.getpid() for result, error in completed if error is None))

    def test_synthetic_c_enforces_outer_family_order_and_anonymous_groups(self):
        result = _synthetic_c_check()
        self.assertTrue(result["passed"])
        self.assertEqual(result["events"], ["global_score_freeze", "inner_labels", "router_fit",
                                            "outer_freeze_hash", "outer_labels"])
        self.assertIn("20 anonymous groups", result["scope"])
        self.assertEqual(result["first_prediction_builds"], 1)
        self.assertEqual(result["resumed_prediction_builds"], 0)
        self.assertEqual(result["first_prediction_digest"], result["resumed_prediction_digest"])
        self.assertEqual(result["resume_events"], ["global_score_freeze", "inner_labels", "router_fit",
                                                   "outer_resume_verify", "outer_labels"])

    def test_outer_resume_requires_router_training_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            connection = database()
            self.addCleanup(connection.close)
            manifest = build_value_manifest(connection, ["good"])
            path = Path(directory) / "outer.json"
            freeze_predictions([prediction(policy="train_feature_router")], manifest, path,
                               contract_sha="contract", binding_sha="router-a")
            self.assertEqual(load_frozen_predictions(path, manifest, contract_sha="contract",
                                                     binding_sha="router-a")["binding_sha256"], "router-a")
            with self.assertRaises(ValueError):
                load_frozen_predictions(path, manifest, contract_sha="contract", binding_sha="router-b")

    def test_router_artifact_is_anonymous_strict_and_route_equivalent(self):
        from services.time_series_anomaly_diagnosis.routing_policy import (
            FEATURE_KEYS, MethodMetric, RouterTrainingRecord, fit_train_feature_router,
        )
        records = []
        for group in range(20):
            features = dict(zip(FEATURE_KEYS, (group + 2, 20, group / 20, 0, 0, 1, 0.5), strict=True))
            records.append(RouterTrainingRecord(group, features, (
                MethodMetric(METHOD_NAMES[0], 0.6, 1),
                MethodMetric(METHOD_NAMES[1], 0.7, 2),
                MethodMetric(METHOD_NAMES[2], 0.5, 1),
            )))
        router = fit_train_feature_router(records)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "router.json"
            payload = _router_payload(router, records, contract_sha="contract", source_sha="source")
            path.write_text(json.dumps(payload))
            encoded = path.read_text().lower()
            for forbidden in ("dataset", "family", "label", "test_indices", "scores", '"auc"'):
                self.assertNotIn(forbidden, encoded)
            loaded = _router_from_payload(json.loads(path.read_text()), contract_sha="contract", source_sha="source")
            for record in records[::5]:
                self.assertEqual(router.route(record.features).selected_method,
                                 loaded.route(record.features).selected_method)
            payload = json.loads(path.read_text()); payload["unknown"] = 1
            with self.assertRaises(ValueError):
                _router_from_payload(payload, contract_sha="contract", source_sha="source")

    def test_results_bundle_replays_metrics_winner_and_all_digests(self):
        from services.time_series_anomaly_diagnosis.routing_policy import (
            FEATURE_KEYS, MethodMetric, RouterTrainingRecord, fit_train_feature_router,
        )
        connection = database()
        self.addCleanup(connection.close)
        manifest = build_value_manifest(connection, ["good"])
        labels = {"good": {0: 0, 1: 1}}
        families = {"good": "f"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); predictions = {}; metric_rows = []
            policies = ("current_baseline", "single_minirocket", "three_method_consensus", "train_feature_router")
            for policy in policies:
                row = prediction(policy=policy)
                if policy == "three_method_consensus":
                    row["selected_method"] = None
                    row["method_p_values"] = [{name: value for name in METHOD_NAMES} for value in (0.9, 0.1)]
                    row["method_seed_p_values"] = [{name: [value, value] for name in METHOD_NAMES} for value in (0.9, 0.1)]
                if policy == "train_feature_router":
                    floor = 1 / 6
                    row["p_values"][1] = floor; row["evaluation_scores"][1] = 1 - floor
                    row["review"][1] = True; row["method_p_values"][1][METHOD_NAMES[0]] = floor
                    row["method_seed_p_values"][1][METHOD_NAMES[0]] = [floor, floor]
                path = root / f"{policy}.json"
                payload = freeze_predictions([row], manifest, path, contract_sha="contract")
                predictions[policy] = path
                metric_rows.append(evaluate_frozen_predictions(payload["predictions"], labels, families))
            winner = select_qualified_policy(metric_rows[0], metric_rows[1:])
            self.assertEqual(winner, "train_feature_router")
            records = [RouterTrainingRecord(group,
                        dict(zip(FEATURE_KEYS, (group + 2, 20, group / 20, 0, 0, 1, 0.5), strict=True)),
                        (MethodMetric(METHOD_NAMES[0], 0.6, 1),
                         MethodMetric(METHOD_NAMES[1], 0.7, 2),
                         MethodMetric(METHOD_NAMES[2], 0.5, 1))) for group in range(20)]
            router_payload = _router_payload(fit_train_feature_router(records), records,
                                             contract_sha="contract", source_sha="source")
            population = root / "population.json"; population.write_text(json.dumps(
                {"contract_sha256": "contract", "source_sha256": "source", "count": 1, "datasets": ["good"]}))
            manifest_path = root / "manifest.json"; manifest_path.write_text(json.dumps(
                {"contract_sha256": "contract", "value_manifest": manifest.as_dict()}))
            results = root / "results.jsonl"
            result_lines = [json.dumps({"schema": "policy_qualification_results_v1", "contract_sha256": "contract",
                                        "qualified_policy": winner, "prediction_digests": {row.policy_id: row.prediction_digest for row in metric_rows}}, sort_keys=True)]
            result_lines += [json.dumps({"type": "policy_metrics", **asdict(row)}, sort_keys=True) for row in metric_rows]
            results.write_text("\n".join(result_lines) + "\n")
            policy = root / "policy.json"; policy.write_text(json.dumps(
                {"contract_sha256": "contract", "policy_id": winner, "router": router_payload}))
            kwargs = dict(results_path=results, policy_path=policy, population_path=population,
                          manifest_path=manifest_path, prediction_paths=predictions,
                          authoritative_names=("good",), source_sha="source", fresh_manifest=manifest,
                          contract_sha="contract", labels=labels, families=families,
                          authoritative_router_payload=router_payload)
            self.assertEqual(validate_results_bundle(**kwargs)["status"], "valid")
            lines = results.read_text().splitlines()
            timed = json.loads(lines[1]); timed["elapsed_seconds"] = 4.25
            results.write_text("\n".join((lines[0], json.dumps(timed), *lines[2:])) + "\n")
            self.assertEqual(validate_results_bundle(**kwargs)["status"], "valid")
            changed = list(records)
            changed[0] = RouterTrainingRecord(0, {**dict(changed[0].features), "width": 999}, changed[0].metrics)
            other_payload = _router_payload(fit_train_feature_router(changed), changed,
                                            contract_sha="contract", source_sha="source")
            policy.write_text(json.dumps({"contract_sha256": "contract", "policy_id": winner,
                                          "router": other_payload}))
            with self.assertRaises(ValueError):
                validate_results_bundle(**kwargs)
            policy.write_text(json.dumps({"contract_sha256": "contract", "policy_id": winner,
                                          "router": router_payload}))
            lines = results.read_text().splitlines(); tampered = json.loads(lines[1]); tampered["row_count"] = 99
            results.write_text(lines[0] + "\n" + json.dumps(tampered) + "\n" + "\n".join(lines[2:]) + "\n")
            with self.assertRaises(ValueError):
                validate_results_bundle(**kwargs)
            results.write_text("\n".join(lines[:-1]) + "\n")
            with self.assertRaises(ValueError):
                validate_results_bundle(**kwargs)

    def test_progress_is_aggregate_and_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.html"
            render_progress_dashboard(path, phase="scores", status="running", requested=3,
                                      eligible=2, completed=1, failures=1, backend="cpu", elapsed_seconds=4.0)
            html = path.read_text()
            self.assertIn("scores", html)
            self.assertNotIn("candidate_indices", html)


if __name__ == "__main__":
    unittest.main()
