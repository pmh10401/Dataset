"""Bounded, label-free readiness tests for the paper r4 proposal run."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import inspect
import time

from experiments.exp448_exp447_text_label_boundary_repair_antigravity.paper_r4_readiness_antigravity import (
    EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY,
    EXPECTED_AFFECTED_SHA256_ANTIGRAVITY,
    EXPECTED_UNION_DATASETS_ANTIGRAVITY,
    EXPECTED_UNION_TEST_ONLY_DATASET_ANTIGRAVITY,
    InventoryGateAntigravity,
    R4ReadinessDependenciesAntigravity,
    UnionInventoryGateAntigravity,
    _production_union_antigravity,
    _scan_pinned_union_inventory_antigravity,
    _union_gate_from_rows_antigravity,
    canonical_result_bytes_antigravity,
    main,
    run_r4_readiness_antigravity,
)
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.config_antigravity import (
    FROZEN_CONFIG as EXP443_FROZEN_CONFIG,
)
from experiments.exp443_missing_normal_train_policy_comparison_antigravity.preprocessing_antigravity import (
    POLICY_A_ANTIGRAVITY,
    prepare_dataset_antigravity,
)
from paper.equipment_anomaly_detection.sources_paper.minirocket_paper import (
    ConfigurationScoresPaper,
)


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype="<f8").copy()
    result.setflags(write=False)
    return result


def _population() -> tuple[str, ...]:
    unaffected = tuple(f"Fixture{index:04d}_normal_0" for index in range(1098))
    return tuple(sorted((*EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY, *unaffected)))


def _rows_for_dataset(
    dataset: str,
    affected_rows: int,
    *,
    one_five_missing: bool = False,
    test_label: str | None = None,
) -> SimpleNamespace:
    train = []
    for index in range(affected_rows):
        missing_count = 5 if one_five_missing and index == 0 else 6
        values = np.arange(9, dtype="<f4") + index
        values[:missing_count] = np.nan
        train.append(SimpleNamespace(dataset=dataset, split="TRAIN", instance_index=index, values=values, missing_positions=tuple(range(missing_count))))
    test = []
    for index in range(2):
        fields = {
            "dataset": dataset,
            "split": "TEST",
            "instance_index": index,
            "values": np.arange(9, dtype="<f4") + index,
            "missing_positions": (),
        }
        if test_label is not None:
            fields["label"] = test_label
        test.append(SimpleNamespace(**fields))
    return SimpleNamespace(dataset=dataset, train=tuple(train), test=tuple(test), target_length=9)


def _raw_mapping() -> dict[str, SimpleNamespace]:
    # 12 * 9 + 7 * 8 = the independently frozen 164 affected TRAIN rows.
    return {
        dataset: _rows_for_dataset(dataset, 9 if index < 12 else 8, one_five_missing=index == 0)
        for index, dataset in enumerate(EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY)
    }


def _fixture_scores(
    train: np.ndarray,
    test: np.ndarray,
    requested_kernels: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    kernel_offset = requested_kernels / 100_000.0
    seed_offset = (seed % 10_000) / 1_000_000.0
    train_score = np.asarray(train.mean(axis=1, dtype=np.float64) + kernel_offset + seed_offset, dtype="<f8")
    test_score = np.asarray(test.mean(axis=1, dtype=np.float64) + kernel_offset + seed_offset, dtype="<f8")
    return train_score, test_score


def _archive_arrays(raw: SimpleNamespace) -> dict[str, np.ndarray]:
    prepared = prepare_dataset_antigravity(raw, POLICY_A_ANTIGRAVITY)
    result: dict[str, np.ndarray] = {}
    for kernels in (1024, 4096):
        pairs = tuple(
            _fixture_scores(prepared.train_values, prepared.test_values, kernels, seed)
            for seed in (20260721, 20261721)
        )
        train = np.stack(tuple(pair[0] for pair in pairs)).astype("<f8")
        test = np.stack(tuple(pair[1] for pair in pairs)).astype("<f8")
        train.setflags(write=False)
        test.setflags(write=False)
        result[f"kernels_{kernels}__train_by_seed"] = train
        result[f"kernels_{kernels}__test_by_seed"] = test
    return result


def _snapshot(
    *,
    affected_rows: int = 164,
    all_missing_rows: int = 0,
    label_access_count: int = 0,
    database_sha256: str = EXP443_FROZEN_CONFIG.database_sha256,
) -> SimpleNamespace:
    raw = _raw_mapping()
    cache = {
        dataset: {"arrays": _archive_arrays(raw[dataset])}
        for dataset in EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY
    }
    return SimpleNamespace(
        datasets=_population(),
        affected_datasets=EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY,
        affected_rows=affected_rows,
        nonfinite_scalars=983,
        all_missing_rows=all_missing_rows,
        affected_datasets_sha256=EXPECTED_AFFECTED_SHA256_ANTIGRAVITY,
        label_access_count=label_access_count,
        cache_inventory=cache,
        source_hashes={"database": database_sha256},
    )


def _dependencies(
    *,
    snapshot: SimpleNamespace | None = None,
    delta: float = 0.0,
    candidate_mismatch: bool = False,
    raw_override: dict[str, SimpleNamespace] | None = None,
):
    authority = _snapshot() if snapshot is None else snapshot
    raw = _raw_mapping() if raw_override is None else raw_override
    opened = []

    @contextmanager
    def raw_context(_snapshot_value):
        opened.append(True)
        yield raw

    def score(dataset, train, test, *, requested_kernels, seed, expected_dimension):
        seed_index = (20260721, 20261721).index(seed)
        arrays = authority.cache_inventory[dataset]["arrays"]
        train_score, test_score = _fixture_scores(train, test, requested_kernels, seed)
        test_score[0] += delta
        if candidate_mismatch:
            train_score = np.arange(200, dtype="<f8")
            test_score = np.asarray([199.0000005], dtype="<f8")
            arrays[f"kernels_{requested_kernels}__train_by_seed"] = np.stack([np.arange(200, dtype="<f8")] * 2)
            arrays[f"kernels_{requested_kernels}__test_by_seed"] = np.stack([np.asarray([199.0], dtype="<f8")] * 2)
        return ConfigurationScoresPaper(train_score, test_score, expected_dimension)

    def verify_union(_snapshot, _raw):
        return UnionInventoryGateAntigravity(
            dataset_count=20,
            datasets=EXPECTED_UNION_DATASETS_ANTIGRAVITY,
            test_only_dataset="MelbournePedestrian_normal_10",
            test_only_rows=3,
            test_only_scalars=3,
            label_access_count=0,
        )

    return R4ReadinessDependenciesAntigravity(
        verify_sources=lambda: authority,
        affected_raw_context=raw_context,
        prepare_policy_a=lambda values: prepare_dataset_antigravity(values, POLICY_A_ANTIGRAVITY),
        score_configuration=score,
        verify_union=verify_union,
        verify_environment=lambda: {"torch": "2.13.0", "hydra": "constructed"},
    ), opened


class R4ReadinessContractTests(unittest.TestCase):
    def test_inventory_gate_rejects_any_change_to_frozen_population(self) -> None:
        gate = InventoryGateAntigravity(
            dataset_count=1117,
            affected_datasets=EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY,
            affected_rows=164,
            nonfinite_scalars=983,
            all_missing_rows=0,
            affected_datasets_sha256=EXPECTED_AFFECTED_SHA256_ANTIGRAVITY,
            label_access_count=0,
            repaired_row_count=164,
        )
        self.assertEqual(gate.affected_rows, 164)
        for forged in (
            {"dataset_count": 1116}, {"affected_rows": 163}, {"nonfinite_scalars": 982},
            {"all_missing_rows": 1}, {"label_access_count": 1}, {"repaired_row_count": 163},
        ):
            with self.subTest(forged=forged), self.assertRaises(ValueError):
                replace(gate, **forged)
        with self.assertRaises(ValueError):
            replace(gate, affected_rows=True)

    def test_inventory_failure_stops_before_raw_or_minirocket_work(self) -> None:
        dependencies, opened = _dependencies(snapshot=_snapshot(affected_rows=163))
        result = run_r4_readiness_antigravity(dependencies=dependencies)
        self.assertEqual(result.status, "not_ready")
        self.assertIn("affected_rows", result.failures[0])
        self.assertEqual(opened, [])
        self.assertEqual(result.parity, ())

    def test_database_hash_drift_stops_before_raw_or_minirocket_work(self) -> None:
        dependencies, opened = _dependencies(snapshot=_snapshot(database_sha256="8" * 64))
        result = run_r4_readiness_antigravity(dependencies=dependencies)
        self.assertEqual(result.status, "not_ready")
        self.assertIn("database", result.failures[0])
        self.assertEqual(opened, [])
        self.assertEqual(result.database_sha256, "8" * 64)

    def test_test_row_with_label_is_rejected_before_parity(self) -> None:
        raw = _raw_mapping()
        first = EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY[0]
        raw[first] = _rows_for_dataset(first, 9, one_five_missing=True, test_label="1")
        dependencies, _ = _dependencies(raw_override=raw)
        result = run_r4_readiness_antigravity(dependencies=dependencies)
        self.assertEqual(result.status, "not_ready")
        self.assertIn("TEST label", result.failures[0])
        self.assertEqual(result.parity, ())

    def test_loaded_label_access_module_fails_before_source_verification(self) -> None:
        calls = []
        dependencies, _ = _dependencies()
        dependencies = replace(dependencies, verify_sources=lambda: calls.append("source"))
        with patch.dict("sys.modules", {"paper.equipment_anomaly_detection.label_access_paper": object()}):
            result = run_r4_readiness_antigravity(dependencies=dependencies)
        self.assertEqual(result.status, "not_ready")
        self.assertIn("label-capable module", result.failures[0])
        self.assertEqual(calls, [])

    def test_missing_cache_inventory_returns_canonical_not_ready(self) -> None:
        snapshot = _snapshot()
        del snapshot.cache_inventory
        dependencies, _ = _dependencies(snapshot=snapshot)
        result = run_r4_readiness_antigravity(dependencies=dependencies)
        self.assertEqual(result.status, "not_ready")
        self.assertIn("cache_inventory", result.failures[0])


class R4ReadinessParityTests(unittest.TestCase):
    def test_exact_archive_scores_produce_76_ready_comparisons(self) -> None:
        dependencies, _ = _dependencies()
        result = run_r4_readiness_antigravity(dependencies=dependencies)
        self.assertEqual(result.status, "ready")
        self.assertEqual(len(result.parity), 76)
        self.assertTrue(all(item.candidate_equal for item in result.parity))
        self.assertLessEqual(max(item.max_abs_difference for item in result.parity), 1e-6)

    def test_score_difference_over_tolerance_fails_readiness(self) -> None:
        dependencies, _ = _dependencies(delta=1.000001e-6)
        result = run_r4_readiness_antigravity(dependencies=dependencies)
        self.assertEqual(result.status, "not_ready")
        self.assertTrue(any("score tolerance" in failure for failure in result.failures))

    def test_candidate_mismatch_fails_even_within_score_tolerance(self) -> None:
        dependencies, _ = _dependencies(candidate_mismatch=True)
        result = run_r4_readiness_antigravity(dependencies=dependencies)
        self.assertEqual(result.status, "not_ready")
        self.assertTrue(any("candidate" in failure for failure in result.failures))

    def test_canonical_result_is_deterministic_and_self_hashing(self) -> None:
        dependencies, _ = _dependencies()
        result = run_r4_readiness_antigravity(dependencies=dependencies)
        first = canonical_result_bytes_antigravity(result)
        second = canonical_result_bytes_antigravity(result)
        self.assertEqual(first, second)
        payload = json.loads(first)
        digest = payload.pop("result_sha256")
        expected = sha256(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n").hexdigest()
        self.assertEqual(digest, expected)

    def test_cli_emits_one_canonical_object_without_creating_a_run_root(self) -> None:
        dependencies, _ = _dependencies()
        expected = run_r4_readiness_antigravity(dependencies=dependencies)
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch(
            "experiments.exp448_exp447_text_label_boundary_repair_antigravity.paper_r4_readiness_antigravity.run_r4_readiness_antigravity",
            return_value=expected,
        ), patch("sys.stdout", stdout), patch("sys.stderr", stderr):
            code = main([])
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue().encode("utf-8"), canonical_result_bytes_antigravity(expected))
        self.assertEqual(json.loads(stdout.getvalue())["status"], "ready")

    def test_union_inventory_mismatch_fails_before_scoring(self) -> None:
        """Production mutation caught: readiness accepted TRAIN 19/164/983/0 without the official TRAIN-or-TEST union of 20."""
        scored = []
        dependencies, _ = _dependencies()

        def bad_union(_snapshot, _raw):
            raise ValueError("union_dataset_count differs: expected=20 actual=19")

        def score(*_args, **_kwargs):
            scored.append("score")
            raise AssertionError("union mismatch must fail before MiniRocket")

        dependencies = replace(dependencies, verify_union=bad_union, score_configuration=score)
        result = run_r4_readiness_antigravity(dependencies=dependencies)
        self.assertEqual(result.status, "not_ready")
        self.assertIn("union_dataset_count", result.failures[0])
        self.assertEqual(result.parity, ())
        self.assertEqual(scored, [])

    def test_injected_21st_nonfinite_dataset_fails_closed(self) -> None:
        """An unexpected 21st TRAIN-or-TEST nonfinite dataset must fail closed.

        If the scanner walked only the 20 expected names, this extra dataset
        would be invisible and the gate would vacuous-pass.
        """
        raw = {}
        for name in EXPECTED_AFFECTED_DATASETS_ANTIGRAVITY:
            values = np.arange(8, dtype="<f4")
            missing = values.copy()
            missing[0] = np.nan
            raw[name] = SimpleNamespace(
                train=(SimpleNamespace(values=missing), SimpleNamespace(values=values)),
                test=(SimpleNamespace(values=values + np.float32(10)),),
            )
        finite = np.arange(8, dtype="<f4")
        test_missing = []
        for index in range(3):
            row = finite.copy() + np.float32(10 + index)
            row[0] = np.nan
            test_missing.append(SimpleNamespace(values=row))
        raw[EXPECTED_UNION_TEST_ONLY_DATASET_ANTIGRAVITY] = SimpleNamespace(
            train=(SimpleNamespace(values=finite),),
            test=(*test_missing, SimpleNamespace(values=finite + np.float32(20))),
        )
        accepted = _union_gate_from_rows_antigravity(raw, label_access_count=0)
        self.assertEqual(accepted.datasets, EXPECTED_UNION_DATASETS_ANTIGRAVITY)
        self.assertEqual(accepted.test_only_rows, 3)
        self.assertEqual(accepted.test_only_scalars, 3)

        extra = finite.copy()
        extra[1] = np.nan
        raw["UnexpectedDataset_normal_0"] = SimpleNamespace(
            train=(SimpleNamespace(values=finite),),
            test=(SimpleNamespace(values=extra),),
        )
        with self.assertRaises(ValueError) as context:
            _union_gate_from_rows_antigravity(raw, label_access_count=0)
        self.assertIn("union_dataset_count differs: expected=20 actual=21", str(context.exception))
        self.assertNotIn("UnexpectedDataset_normal_0", EXPECTED_UNION_DATASETS_ANTIGRAVITY)

    def test_production_union_scans_pinned_population_not_expected_twenty(self) -> None:
        """Production must independently walk the pinned population, not the 20-name allowlist."""
        production = inspect.getsource(_production_union_antigravity)
        scanner = inspect.getsource(_scan_pinned_union_inventory_antigravity)
        self.assertNotIn("EXPECTED_UNION_DATASETS_ANTIGRAVITY", production)
        self.assertIn("_scan_pinned_union_inventory_antigravity", production)
        self.assertIn("POPULATION_SQL_ANTIGRAVITY", scanner)
        self.assertNotIn("EXPECTED_UNION_DATASETS_ANTIGRAVITY", scanner)

    def test_full_population_values_only_union_scan_antigravity(self) -> None:
        """Exact pinned 1,117-dataset values-only TRAIN-or-TEST union: 20 members, TEST-only 3/3, no labels."""
        database = Path(EXP443_FROZEN_CONFIG.database_path)
        if not database.is_file():
            self.skipTest("frozen database is unavailable")
        started = time.perf_counter()
        gate = _production_union_antigravity(SimpleNamespace(), {})
        elapsed = time.perf_counter() - started
        self.assertEqual(gate.dataset_count, 20)
        self.assertEqual(gate.datasets, EXPECTED_UNION_DATASETS_ANTIGRAVITY)
        self.assertEqual(gate.test_only_dataset, EXPECTED_UNION_TEST_ONLY_DATASET_ANTIGRAVITY)
        self.assertEqual(gate.test_only_rows, 3)
        self.assertEqual(gate.test_only_scalars, 3)
        self.assertEqual(gate.label_access_count, 0)
        self.assertLess(elapsed, 120.0)
        print(f"full_population_union_scan_seconds={elapsed:.6f}", flush=True)

    def test_bounded_db_union_identity_without_test_labels(self) -> None:
        """Official TRAIN-or-TEST union identity is 20, including MelbournePedestrian_normal_10 TEST-only 3/3."""
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.database_antigravity import (
            load_raw_dataset_antigravity,
            pin_database_snapshot_antigravity,
        )
        from experiments.exp448_exp447_text_label_boundary_repair_antigravity.paper_r4_readiness_antigravity import (
            EXPECTED_UNION_TEST_ONLY_DATASET_ANTIGRAVITY,
        )
        from tempfile import TemporaryDirectory

        database = Path(EXP443_FROZEN_CONFIG.database_path)
        if not database.is_file():
            self.skipTest("frozen database is unavailable")
        with TemporaryDirectory(prefix="paper_r4_union_db_antigravity_") as temporary:
            pinned = pin_database_snapshot_antigravity(
                str(database), Path(temporary) / "attempt_antigravity",
                expected_sha256=EXP443_FROZEN_CONFIG.database_sha256,
            )
            try:
                raw = load_raw_dataset_antigravity(pinned, EXPECTED_UNION_TEST_ONLY_DATASET_ANTIGRAVITY)
                self.assertEqual(pinned.label_access_count, 0)
                self.assertFalse(any(hasattr(row, "label") for row in raw.test))
                train_missing = sum(1 for row in raw.train if np.isnan(np.asarray(row.values, dtype="<f4")).any() or not np.isfinite(np.asarray(row.values, dtype="<f4")).all())
                test_rows = 0
                test_scalars = 0
                for row in raw.test:
                    missing = tuple(int(index) for index in np.flatnonzero(~np.isfinite(np.asarray(row.values, dtype="<f4"))))
                    if missing:
                        test_rows += 1
                        test_scalars += len(missing)
            finally:
                pinned.close(terminal_evidence_durable=True)
        self.assertEqual(train_missing, 0)
        self.assertEqual(test_rows, 3)
        self.assertEqual(test_scalars, 3)


@unittest.skipUnless(os.environ.get("PAPER_R4_READINESS_REAL_ANTIGRAVITY") == "1", "opt-in bounded real readiness")
class R4ReadinessRealTests(unittest.TestCase):
    def test_real_19_dataset_r4_readiness_antigravity(self) -> None:
        result = run_r4_readiness_antigravity()
        self.assertEqual(result.status, "ready")
        self.assertIsNotNone(result.inventory)
        self.assertEqual(result.inventory.dataset_count, 1117)
        self.assertEqual(len(result.inventory.affected_datasets), 19)
        self.assertEqual(result.inventory.affected_rows, 164)
        self.assertEqual(result.inventory.nonfinite_scalars, 983)
        self.assertEqual(result.inventory.all_missing_rows, 0)
        self.assertEqual(result.inventory.label_access_count, 0)
        self.assertEqual(len(result.parity), 76)
        self.assertLessEqual(max(item.max_abs_difference for item in result.parity), 1e-6)
        self.assertTrue(all(item.candidate_equal for item in result.parity))


if __name__ == "__main__":
    unittest.main()
