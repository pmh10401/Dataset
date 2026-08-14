"""Test suite for Paper r4 full-population hard-route parity recovery."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Mapping, Sequence
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from paper.equipment_anomaly_detection.catalog_paper import build_catalog
from paper.equipment_anomaly_detection.config_paper import FROZEN_CONFIG
from paper.equipment_anomaly_detection.contracts_paper import RawDataset
from paper.equipment_anomaly_detection.data_paper import TEST_VALUES_SQL, emit_progress_paper, load_dataset_values
from paper.equipment_anomaly_detection.hard_route_paper import (
    HardRouteResult,
    empty_hard_route_result_paper,
    replay_hard_route,
)
from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import score_hard_sources
from paper.equipment_anomaly_detection.sources_paper.legacy_route_sources_paper import build_legacy_route_evidence
from paper.equipment_anomaly_detection.sources_paper.low_frequency_paper import (
    GLOBAL_CANDIDATE_BUDGET_PAPER,
)
from paper.equipment_anomaly_detection.sources_paper.minirocket_paper import score_minirocket_confirmations
from paper.equipment_anomaly_detection.sources_paper.preprocessing_paper import DependencyUnavailablePaper
from paper.equipment_anomaly_detection.sources_paper.stability_paper import (
    _MAXIMUM_VIEWS_PAPER,
)


DB_PATH = Path("/Users/mac/Documents/Dataset/univariate_ts.db")
VERIFIED_EXP137_AUTHORITY_PATH = Path(
    "/Users/mac/Documents/Dataset/experiments/exp234_recovery_antigravity/experiment_137_operational_triage_results_antigravity.csv"
)
VERIFIED_EXP137_AUTHORITY_SHA256 = "942e2f5faac41364a0e91fd9dad7c3e45734cd765a080f31d9d149950826735d"

VERIFIED_EXP236_AUTHORITY_PATH = Path(
    "/Users/mac/Documents/Dataset/experiments/exp236_direct_run_antigravity/exp236_frozen_predictions.jsonl"
)
VERIFIED_EXP236_AUTHORITY_SHA256 = "a03ce4ea40f36cfa840f2d509c002a01f56fe99db795f60ad78d88bb767e6524"

VERIFIED_EXP291_AUTHORITY_PATH = Path(
    "/Users/mac/Documents/Dataset/exp291_final_operational_routing_freeze/exp291_frozen_operational_routing.jsonl"
)
VERIFIED_EXP291_AUTHORITY_SHA256 = "73dff56e26429158bd2672a898d87266aa38e73f0a50d15d4f7c7082e3a01848"

VERIFIED_EXP404_AUTHORITY_PATH = Path(
    "/Users/mac/Documents/Dataset/exp403_408_feature_ml_validation/exp404_hard_confidence.jsonl"
)
VERIFIED_EXP404_AUTHORITY_SHA256 = "437dc63754a29d0d65c0b30bfee54102abdd9fb56d45d2c5b1c5a511b0448e70"


def _build_test_authority():
    from paper.equipment_anomaly_detection.contracts_paper import HardRouteParityAuthority

    return HardRouteParityAuthority(
        exp137_path=VERIFIED_EXP137_AUTHORITY_PATH,
        exp137_sha256=VERIFIED_EXP137_AUTHORITY_SHA256,
        exp236_path=VERIFIED_EXP236_AUTHORITY_PATH,
        exp236_sha256=VERIFIED_EXP236_AUTHORITY_SHA256,
        exp291_path=VERIFIED_EXP291_AUTHORITY_PATH,
        exp291_sha256=VERIFIED_EXP291_AUTHORITY_SHA256,
        exp404_path=VERIFIED_EXP404_AUTHORITY_PATH,
        exp404_sha256=VERIFIED_EXP404_AUTHORITY_SHA256,
    )


class FullRouteParityRecoveryAntigravityTests(unittest.TestCase):
    """Parity recovery test matrix T1 through T14."""

    def test_pycatch22_pin_fails_closed_when_missing_or_wrong_version(self) -> None:
        """T1 (Missing-interface RED): verify_pycatch22_environment_paper raises DependencyUnavailablePaper on missing or mismatch."""
        from paper.equipment_anomaly_detection.sources_paper.canonical_paper import (
            verify_pycatch22_environment_paper,
        )

        with patch("importlib.metadata.version", side_effect=Exception("not found")):
            with self.assertRaises(DependencyUnavailablePaper):
                verify_pycatch22_environment_paper()

        with patch("importlib.metadata.version", return_value="0.4.0"):
            with self.assertRaises(DependencyUnavailablePaper):
                verify_pycatch22_environment_paper()

    def test_pycatch22_0_5_0_is_the_only_accepted_version(self) -> None:
        """T2 (Missing-interface RED): PINNED_PYCATCH22_VERSION_PAPER is '0.5.0' and returns valid metadata dict."""
        from paper.equipment_anomaly_detection.sources_paper import canonical_paper

        self.assertTrue(hasattr(canonical_paper, "PINNED_PYCATCH22_VERSION_PAPER"))
        self.assertEqual(canonical_paper.PINNED_PYCATCH22_VERSION_PAPER, "0.5.0")
        with patch("importlib.metadata.version", return_value="0.5.0"), patch.dict(
            "sys.modules", {"pycatch22": type("m", (), {"catch22_all": lambda *a: ()})()}
        ):
            meta = canonical_paper.verify_pycatch22_environment_paper()
            self.assertEqual(meta["pycatch22"], "0.5.0")

    def test_pycatch22_dependency_failure_propagates_without_swallowing(self) -> None:
        """T3: capable preflight preserves typed canonical failures before persistence."""
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        from paper.equipment_anomaly_detection.contracts_paper import FrozenTask8BackendDecision
        from paper.equipment_anomaly_detection import proposal_score_paper
        from paper.equipment_anomaly_detection.sources_paper.canonical_paper import (
            verify_pycatch22_environment_paper,
        )

        try:
            self.assertEqual(
                verify_pycatch22_environment_paper(), {"pycatch22": "0.5.0"}
            )
        except DependencyUnavailablePaper as error:
            self.skipTest(f"requires capable pycatch22==0.5.0 interpreter: {error}")

        rng = np.random.default_rng(20260814)
        train_rows = tuple(rng.normal(0, 1, size=100).astype(np.float32) for _ in range(10))
        test_rows = tuple(rng.normal(0, 1, size=100).astype(np.float32) for _ in range(20))
        raw_ds = RawDataset(
            dataset="Synthetic_eligible_dataset",
            metadata_length=100,
            train_indices=tuple(range(10)),
            train_blobs=tuple(r.tobytes() for r in train_rows),
            train_rows=train_rows,
            test_indices=tuple(range(20)),
            test_blobs=tuple(r.tobytes() for r in test_rows),
            test_rows=test_rows,
        )

        with patch(
            "paper.equipment_anomaly_detection.sources_paper.hard_sources_paper._canonical_source_paper",
            side_effect=DependencyUnavailablePaper("dependency_unavailable:pycatch22==0.5.0"),
        ):
            with self.assertRaises(DependencyUnavailablePaper):
                score_hard_sources(raw_ds)

        for logical_cpu_count, expected_workers in ((2, 1), (3, 2)):
            with self.subTest(dataset_workers=expected_workers), TemporaryDirectory() as temporary:
                manifest = build_acceleration_manifest(
                    "a" * 64, logical_cpu_count=logical_cpu_count
                )
                backend = FrozenTask8BackendDecision.from_acceleration_manifest(manifest)
                self.assertEqual(backend.dataset_workers, expected_workers)
                events: list[str] = []
                success_persist = MagicMock(name="success_persist")
                typed_error = DependencyUnavailablePaper(
                    "dependency_unavailable:pycatch22==0.5.0"
                )

                def capable_preflight() -> dict[str, str]:
                    events.append("preflight")
                    return verify_pycatch22_environment_paper()

                class RaisingFuture:
                    def result(self):
                        events.append("worker")
                        raise typed_error

                class InlineProcessPool:
                    def __init__(self, *, max_workers: int) -> None:
                        self.max_workers = max_workers
                        self.submitted: list[tuple[object, RawDataset]] = []

                    def __enter__(self):
                        return self

                    def __exit__(self, *_args) -> None:
                        return None

                    def submit(self, function, values: RawDataset) -> RaisingFuture:
                        self.submitted.append((function, values))
                        return RaisingFuture()

                executor = InlineProcessPool(max_workers=expected_workers)

                def serial_worker(_values: RawDataset):
                    events.append("worker")
                    raise typed_error

                with patch.object(
                    proposal_score_paper,
                    "verify_exp84_environment_paper",
                    return_value={"torch": "2.13.0"},
                ), patch.object(
                    proposal_score_paper,
                    "verify_pycatch22_environment_paper",
                    side_effect=capable_preflight,
                ) as pycatch_preflight, patch.object(
                    proposal_score_paper.Task4Task5SuccessAuthority,
                    "from_verified_context",
                    return_value=object(),
                ), patch.object(
                    proposal_score_paper,
                    "verify_task4_task5_success_namespace",
                    return_value=(),
                ), patch.object(
                    proposal_score_paper,
                    "verify_task4_task5_attempt_namespace",
                    return_value=(),
                ), patch.object(
                    proposal_score_paper,
                    "append_task4_task5_attempt_record",
                ) as append_attempt:
                    if expected_workers == 1:
                        with patch.object(
                            proposal_score_paper,
                            "_score_dataset_task4_task5",
                            side_effect=serial_worker,
                        ):
                            with self.assertRaises(DependencyUnavailablePaper):
                                proposal_score_paper._collect_task4_task5(
                                    {raw_ds.dataset: raw_ds},
                                    backend,
                                    root=Path(temporary),
                                    context={"mode": "smoke"},
                                    catalog={"datasets": [{"dataset": raw_ds.dataset}]},
                                    acceleration_manifest=manifest,
                                    publish_success=success_persist,
                                )
                    else:
                        with patch.object(
                            proposal_score_paper,
                            "ProcessPoolExecutor",
                            return_value=executor,
                        ), patch.object(
                            proposal_score_paper,
                            "as_completed",
                            side_effect=lambda futures: iter(futures),
                        ):
                            with self.assertRaises(DependencyUnavailablePaper):
                                proposal_score_paper._collect_task4_task5(
                                    {raw_ds.dataset: raw_ds},
                                    backend,
                                    root=Path(temporary),
                                    context={"mode": "smoke"},
                                    catalog={"datasets": [{"dataset": raw_ds.dataset}]},
                                    acceleration_manifest=manifest,
                                    publish_success=success_persist,
                                )
                        self.assertEqual(executor.max_workers, expected_workers)
                        self.assertEqual(len(executor.submitted), 1)

                self.assertEqual(events, ["preflight", "worker"])
                pycatch_preflight.assert_called_once_with()
                append_attempt.assert_not_called()
                success_persist.assert_not_called()

    def test_capable_venv_eligible_canonical_scoring_with_pycatch22(self) -> None:
        """Integration (Capable venv): Eligible dataset scores canonical_normal_state as eligible=True with pycatch22==0.5.0."""
        try:
            version = importlib.metadata.version("pycatch22")
        except Exception:
            self.skipTest("pycatch22 not installed in current interpreter; run under capable venv")
        if version != "0.5.0":
            self.skipTest(f"pycatch22 version is {version}, expected 0.5.0")

        rng = np.random.default_rng(20260814)
        train_rows = tuple(rng.normal(0, 1, size=100).astype(np.float32) for _ in range(10))
        test_rows = tuple(rng.normal(0, 1, size=100).astype(np.float32) for _ in range(20))
        raw = RawDataset(
            dataset="Synthetic_eligible_dataset",
            metadata_length=100,
            train_indices=tuple(range(10)),
            train_blobs=tuple(r.tobytes() for r in train_rows),
            train_rows=train_rows,
            test_indices=tuple(range(20)),
            test_blobs=tuple(r.tobytes() for r in test_rows),
            test_rows=test_rows,
        )
        bundle = score_hard_sources(raw)
        canonical = bundle.by_name["canonical_normal_state"]
        self.assertTrue(canonical.eligible)
        self.assertEqual(canonical.eligibility_reason, "eligible")
        self.assertEqual(len(canonical.train_oof_score), 10)
        self.assertEqual(len(canonical.test_score), 20)

    def test_replay_signature_still_rejects_historical_coordinates(self) -> None:
        """T4 (Regression Lock): replay_hard_route accepts only values, sources, and mini (no oracle kwargs)."""
        sig = inspect.signature(replay_hard_route)
        params = list(sig.parameters.keys())
        self.assertEqual(params, ["values", "sources", "mini"])
        sig_legacy = inspect.signature(build_legacy_route_evidence)
        self.assertEqual(list(sig_legacy.parameters.keys()), ["values"])

    def test_phoneme_normal_39_reproduces_historical_hard_49_without_name_branch(self) -> None:
        """T5 (Origin Replay RED): Phoneme_normal_39 (n_train=1) produces Hard {49} using values-only DB data and zero name branching."""
        from paper.equipment_anomaly_detection import hard_route_paper
        from paper.equipment_anomaly_detection.sources_paper import legacy_route_sources_paper

        hard_route_src = inspect.getsource(hard_route_paper)
        legacy_src = inspect.getsource(legacy_route_sources_paper)
        self.assertNotIn("Phoneme_normal_39", hard_route_src)
        self.assertNotIn("Phoneme_normal_39", legacy_src)

        raw = load_dataset_values(DB_PATH, "Phoneme_normal_39")
        self.assertEqual(len(raw.train_rows), 1)
        self.assertEqual(len(raw.test_rows), 50)
        sources = score_hard_sources(raw)
        mini = score_minirocket_confirmations(raw)
        route = replay_hard_route(raw, sources, mini)
        self.assertEqual(route.exp137.hard, frozenset({49}))
        self.assertEqual(route.exp236.hard, frozenset({49}))
        self.assertEqual(route.automatic_hard, frozenset({49}))

    def test_oracle_bundle_adapter_is_available_for_later_task4_diagnostic(self) -> None:
        """T6 (missing oracle-bundle adapter RED): Task 2 only requires the values-only adapter.

        Task 4 must run the independent diagnostic harness, commit its report, and
        replace or extend this test with the exact stage-level primitive RED before
        any production route-replay repair.
        """
        from paper.equipment_anomaly_detection.proposal_score_paper import load_hard_route_oracle_bundle

        self.assertTrue(callable(load_hard_route_oracle_bundle))

    def test_smoke_33_datasets_match_sha_bound_oracles(self) -> None:
        """T7 (Capable Integration): All 33 smoke datasets achieve per-dataset coordinate equality against the 4 SHA-bound oracles; direct indexing works on Exp404."""
        from paper.equipment_anomaly_detection.proposal_score_paper import (
            load_hard_route_oracle_bundle,
        )

        authority = _build_test_authority()
        smoke_datasets = (
            "ACSF1_normal_0", "ACSF1_normal_1", "ACSF1_normal_2", "ACSF1_normal_3",
            "ACSF1_normal_4", "ACSF1_normal_5", "ACSF1_normal_6", "ACSF1_normal_7",
            "ACSF1_normal_8", "ACSF1_normal_9", "ChlorineConcentration_normal_3",
            "DodgerLoopDay_normal_1", "DodgerLoopDay_normal_2", "DodgerLoopDay_normal_4",
            "DodgerLoopDay_normal_5", "DodgerLoopDay_normal_6", "DodgerLoopDay_normal_7",
            "DodgerLoopGame_normal_1", "DodgerLoopGame_normal_2", "DodgerLoopWeekend_normal_1",
            "DodgerLoopWeekend_normal_2", "MelbournePedestrian_normal_1", "MelbournePedestrian_normal_2",
            "MelbournePedestrian_normal_3", "MelbournePedestrian_normal_4", "MelbournePedestrian_normal_5",
            "MelbournePedestrian_normal_6", "MelbournePedestrian_normal_7", "MelbournePedestrian_normal_8",
            "MelbournePedestrian_normal_9", "MelbournePedestrian_normal_10", "Phoneme_normal_39",
            "PigAirwayPressure_normal_1",
        )
        self.assertEqual(len(set(smoke_datasets)), 33)

        catalog = build_catalog(DB_PATH, mode="smoke", smoke_datasets=smoke_datasets)
        n_test_map = {entry["dataset"]: int(entry["test"]["count"]) for entry in catalog["datasets"]}

        bundle = load_hard_route_oracle_bundle(
            authority=authority,
            selected_population=smoke_datasets,
            selected_n_test=n_test_map,
        )

        for dataset in smoke_datasets:
            raw = load_dataset_values(DB_PATH, dataset)
            sources = score_hard_sources(raw)
            mini = score_minirocket_confirmations(raw)
            route = replay_hard_route(raw, sources, mini)
            self.assertEqual(route.exp137.hard, bundle.exp137[dataset], f"Exp137 mismatch on {dataset}")
            self.assertEqual(route.exp236.hard, bundle.exp236[dataset], f"Exp236 mismatch on {dataset}")
            self.assertEqual(route.exp253.hard, bundle.exp253[dataset], f"Exp253 mismatch on {dataset}")
            self.assertEqual(route.automatic_hard, bundle.automatic_hard[dataset], f"Exp291 auto mismatch on {dataset}")
            self.assertEqual(route.exact_normal_removed, bundle.exact_normal_removed[dataset], f"Exp291 exact mismatch on {dataset}")
            self.assertEqual(route.automatic_hard, bundle.exp404[dataset], f"Exp404 mismatch on {dataset}")

    def test_full_1117_route_only_parity_gate_blocks_task6_and_task7(self) -> None:
        """T8 (missing verifier RED): a real route-count mismatch reaches the gate before Task6/7.

        The fake population is lexically ordered and structurally valid, so raw
        mapping validation cannot satisfy the expected parity-gate failure.
        """
        from paper.equipment_anomaly_detection.acceleration_paper import build_acceleration_manifest
        from paper.equipment_anomaly_detection.proposal_score_paper import (
            compose_proposal_score,
            verify_hard_route_parity_paper,
        )

        authority = _build_test_authority()
        dataset_names = tuple(f"d_{index:04d}" for index in range(1117))
        zero_row = np.zeros(10, dtype=np.float32)
        fake_raw = {
            dataset: RawDataset(
                dataset,
                10,
                (0,),
                (zero_row.tobytes(),),
                (zero_row,),
                tuple(range(10)),
                (zero_row.tobytes(),) * 10,
                (zero_row,) * 10,
            )
            for dataset in dataset_names
        }
        self.assertEqual(tuple(fake_raw), dataset_names)
        self.assertEqual(tuple(fake_raw), tuple(sorted(fake_raw)))
        manifest = build_acceleration_manifest("a" * 64, logical_cpu_count=2)

        # Official context without authority must fail for both full and smoke
        for mode in ("full", "smoke"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "hard_route_parity_authority"):
                    compose_proposal_score(fake_raw, manifest, context={"mode": mode})

        fake_routes = {
            dataset: empty_hard_route_result_paper(10)
            for dataset in dataset_names
        }
        empty_oracle = {dataset: frozenset() for dataset in dataset_names}
        oracle_bundle = SimpleNamespace(
            exp137=empty_oracle,
            exp236=empty_oracle,
            exp253=empty_oracle,
            exact_normal_removed=empty_oracle,
            automatic_hard=empty_oracle,
            exp404=empty_oracle,
        )

        with patch(
            "paper.equipment_anomaly_detection.proposal_score_paper._collect_task4_task5",
            return_value=({}, fake_routes),
        ) as mock_collect, patch(
            "paper.equipment_anomaly_detection.proposal_score_paper.load_hard_route_oracle_bundle",
            return_value=oracle_bundle,
        ) as mock_oracle_loader, patch(
            "paper.equipment_anomaly_detection.proposal_score_paper.score_low_frequency"
        ) as mock_task6, patch(
            "paper.equipment_anomaly_detection.proposal_score_paper.classify_stability"
        ) as mock_task7:
            with self.assertRaisesRegex(ValueError, "hard route count parity mismatch"):
                compose_proposal_score(
                    fake_raw,
                    manifest,
                    route_parity_authority=authority,
                    route_parity_mode="full",
                )

        self.assertEqual(mock_collect.call_count, 1)
        self.assertEqual(mock_oracle_loader.call_count, 1)
        self.assertEqual(
            tuple(mock_oracle_loader.call_args.kwargs["selected_population"]),
            dataset_names,
        )
        mock_task6.assert_not_called()
        mock_task7.assert_not_called()

    def test_candidate_c_rule_and_target_algebra_unchanged(self) -> None:
        """T9 (Regression Lock): Candidate C budget 30, view cap 16, survival 14/16, formula 2218 + 26 = 2244."""
        self.assertEqual(GLOBAL_CANDIDATE_BUDGET_PAPER, 30)
        self.assertEqual(_MAXIMUM_VIEWS_PAPER, 16)
        survival_threshold = 14 / 16
        self.assertAlmostEqual(survival_threshold, 0.875)
        automatic_hard_target = 2218
        stable_target = 26
        self.assertEqual(automatic_hard_target + stable_target, 2244)

    def test_clean_root_execution_and_rejection_of_live_r4_reuse(self) -> None:
        """T10 (Missing-interface RED): Runner creates fresh records in clean root and rejects live r4 root reuse."""
        forbidden_prior = Path("/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813")
        from scratch.manage_paper_r4_proposal_antigravity import validate_clean_root_destination_paper

        with self.assertRaises(ValueError):
            validate_clean_root_destination_paper(forbidden_prior)

    def test_new_smoke_and_official_root_paths_are_distinct_and_refuse_old_destinations(self) -> None:
        """T11 (missing destination validator RED): clean roots differ and every old root is refused."""
        from scratch.manage_paper_r4_proposal_antigravity import (
            AUTHORIZED_OFFICIAL_ROOT_20260814,
            AUTHORIZED_SMOKE_ROOT_20260814,
            FORBIDDEN_DESTINATION_ROOTS,
            validate_clean_root_destination_paper,
        )

        required_forbidden = frozenset(
            {
                Path("/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813"),
                Path("/Users/mac/Documents/Dataset/paper_proposal_full_r4_antigravity_20260813_ops_antigravity"),
                Path("/Users/mac/Documents/Dataset/paper_proposal_full_r4_v2_antigravity_20260814_ops_antigravity"),
                Path("/Users/mac/Documents/Dataset/paper_proposal_full_r3_antigravity_20260813"),
                Path("/Users/mac/Documents/Dataset/paper_proposal_r4_prelaunch_smoke_antigravity_20260813"),
                Path("/Users/mac/Documents/Dataset/paper_proposal_r4_prelaunch_smoke_antigravity_20260813_ops_antigravity"),
                Path("/Users/mac/Documents/Dataset/paper_proposal_r4_prelaunch_smoke_v2_antigravity_20260813"),
                Path("/Users/mac/Documents/Dataset/paper_proposal_r4_prelaunch_smoke_v2_antigravity_20260813_ops_antigravity"),
            }
        )
        self.assertTrue(required_forbidden)
        self.assertLessEqual(required_forbidden, set(FORBIDDEN_DESTINATION_ROOTS))
        self.assertNotEqual(
            AUTHORIZED_SMOKE_ROOT_20260814.resolve(),
            AUTHORIZED_OFFICIAL_ROOT_20260814.resolve(),
        )
        self.assertIn("20260814", str(AUTHORIZED_SMOKE_ROOT_20260814))
        self.assertIn("20260814", str(AUTHORIZED_OFFICIAL_ROOT_20260814))
        self.assertNotIn(AUTHORIZED_SMOKE_ROOT_20260814, FORBIDDEN_DESTINATION_ROOTS)
        self.assertNotIn(AUTHORIZED_OFFICIAL_ROOT_20260814, FORBIDDEN_DESTINATION_ROOTS)
        validate_clean_root_destination_paper(AUTHORIZED_SMOKE_ROOT_20260814)
        validate_clean_root_destination_paper(AUTHORIZED_OFFICIAL_ROOT_20260814)
        for forbidden_root in sorted(FORBIDDEN_DESTINATION_ROOTS, key=str):
            with self.subTest(forbidden_root=forbidden_root):
                with self.assertRaises(ValueError):
                    validate_clean_root_destination_paper(forbidden_root)

    def test_progress_logging_remains_optional_and_label_free(self) -> None:
        """T12 (Regression Lock): Progress logging default None is bit-identical and label-free."""
        event = {"stage": "task4_task5", "phase": "tick", "completed": 1, "total": 10, "elapsed_s": 0.5, "dataset": "Synthetic_dataset"}
        emitted: list[dict] = []
        emit_progress_paper(emitted.append, event)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["stage"], "task4_task5")
        self.assertNotIn("y_test", str(emitted[0]))
        self.assertNotIn("labels", str(emitted[0]))

    def test_zero_test_label_access_until_evaluation(self) -> None:
        """T13 (Regression Lock): TEST_VALUES_SQL selects no label column; label firewall is intact."""
        self.assertNotIn("y_test", TEST_VALUES_SQL.lower())
        self.assertNotIn("label", TEST_VALUES_SQL.lower())
        self.assertNotIn("is_anomaly", TEST_VALUES_SQL.lower())

    def test_frozen_counts_2218_remain_authoritative_target(self) -> None:
        """T14 (Regression Lock): Target counts remain 2005/2116/2253/35/2218; 2203 is not a target."""
        self.assertEqual(FROZEN_CONFIG.ablation_alert_counts[0], 2218)
        self.assertNotEqual(FROZEN_CONFIG.ablation_alert_counts[0], 2203)


if __name__ == "__main__":
    unittest.main()
