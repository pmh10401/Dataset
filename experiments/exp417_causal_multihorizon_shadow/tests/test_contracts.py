from types import MappingProxyType
import unittest

import numpy as np

from experiments.exp417_causal_multihorizon_shadow.contracts import (
    BackendAudit,
    DatasetResult,
    FrozenLaneSnapshot,
    FrozenRunContext,
    FrozenTrainContract,
    GroupQualification,
    HorizonScore,
    LengthGroup,
    ProcessedGroup,
    RawSeries,
    RawTestDataset,
    RawTrainDataset,
    RunSummary,
    SeedDecision,
    ShadowCandidate,
)
from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG


def frozen_lanes(
    *, hard: dict[str, tuple[int, ...]] | None = None
) -> FrozenLaneSnapshot:
    return FrozenLaneSnapshot(
        hard=hard or {"hard": tuple(range(2218))},
        standard={"standard": tuple(range(806))},
        priority={"priority": tuple(range(31))},
        automatic_normal={"automatic": tuple(range(35))},
    )


def frozen_context(
    *,
    database_sha256: str | None = None,
    datasets: tuple[str, ...] | None = None,
    lanes: FrozenLaneSnapshot | None = None,
) -> FrozenRunContext:
    return FrozenRunContext(
        database_sha256=database_sha256 or FROZEN_CONFIG.database_sha256,
        datasets=datasets or tuple(f"dataset-{index:04d}" for index in range(1117)),
        lanes=lanes or frozen_lanes(),
        code_commit="abc123",
        runtime_versions={"python": "3.14", "numpy": "2.3.5", "numba": "0.66.0", "torch": "2.13.0"},
    )


def horizon(horizon_value: int, *, p_value: float = 0.01, persistence: float = 0.05) -> HorizonScore:
    return HorizonScore(
        horizon=horizon_value,
        score=1.0,
        p_value=p_value,
        persistence=persistence,
    )


def candidate(dataset: str, index: int) -> ShadowCandidate:
    scores = {
        1: horizon(1),
        2: horizon(2),
        10: horizon(10, p_value=0.50, persistence=0.0),
    }
    return ShadowCandidate(
        dataset=dataset,
        instance_index=index,
        length=100,
        seed_decisions=tuple(
            SeedDecision(seed=seed, scores=scores, candidate=True)
            for seed in (41701, 41702, 41703)
        ),
    )


def group() -> GroupQualification:
    return GroupQualification(
        length=100,
        series_count=100,
        status="qualified",
        oof_candidate_rate=0.01,
        wilson_upper=0.05,
        pooled_synthetic_sensitivity=0.70,
        per_type_synthetic_sensitivity={
            "level_shift": 0.50,
            "gradual_drift": 0.50,
            "amplitude_scaling": 0.50,
            "period_change": 0.50,
        },
        seed_agreement=True,
    )


def dataset_result(
    dataset: str,
    *,
    candidates: tuple[ShadowCandidate, ...] = (),
    lanes: FrozenLaneSnapshot | None = None,
    group_count: int = 1,
    test_size: int = 20,
) -> DatasetResult:
    return DatasetResult(
        dataset=dataset,
        test_size=test_size,
        group_count=group_count,
        groups=(group(),),
        candidates=candidates,
        lanes=lanes or frozen_lanes(),
        status="qualified",
    )


def results_for_context(
    context: FrozenRunContext, *, retain_one_candidate_per_dataset: bool = False
) -> tuple[DatasetResult, ...]:
    return tuple(
        dataset_result(
            dataset,
            candidates=(candidate(dataset, 1),) if retain_one_candidate_per_dataset else (),
            lanes=context.lanes,
        )
        for dataset in context.datasets
    )


class ContractTests(unittest.TestCase):
    def test_raw_series_rejects_bool_index_and_test_dataset_rejects_nonfinite(self) -> None:
        with self.assertRaises(ValueError):
            RawSeries(instance_index=True, values=(0.0, 1.0))
        raw = RawSeries(instance_index=1, values=(0.0, float("nan")))
        train = RawTrainDataset(dataset="synthetic", series=(raw,))
        self.assertTrue(np.isnan(train.series[0].values[1]))
        with self.assertRaises(ValueError):
            RawTestDataset(dataset="synthetic", series=(raw,))

    def test_coordinates_are_sorted_unique_and_mappings_are_immutable(self) -> None:
        with self.assertRaisesRegex(ValueError, "sorted"):
            FrozenLaneSnapshot(
                hard={"A": (2, 1)}, standard={}, priority={}, automatic_normal={}
            )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            FrozenLaneSnapshot(
                hard={"A": (1, 1)}, standard={}, priority={}, automatic_normal={}
            )
        lanes = frozen_lanes(hard={"A": (1,), "hard": tuple(range(2, 2219))})
        self.assertIsInstance(lanes.hard, MappingProxyType)
        with self.assertRaises(TypeError):
            lanes.hard["B"] = (2,)  # type: ignore[index]

    def test_frozen_lanes_require_exact_totals_and_artifact_hashes(self) -> None:
        with self.assertRaisesRegex(ValueError, "total"):
            FrozenLaneSnapshot(hard={}, standard={}, priority={}, automatic_normal={})
        with self.assertRaisesRegex(ValueError, "Exp291"):
            FrozenLaneSnapshot(
                hard={"hard": tuple(range(2218))},
                standard={"standard": tuple(range(806))},
                priority={"priority": tuple(range(31))},
                automatic_normal={"automatic": tuple(range(35))},
                exp291_prediction_sha256="0" * 64,
            )

    def test_frozen_run_context_requires_frozen_database_and_population(self) -> None:
        with self.assertRaisesRegex(ValueError, "database"):
            frozen_context(database_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "population"):
            frozen_context(datasets=("A",))
        self.assertEqual(len(frozen_context().datasets), 1117)

    def test_horizon_score_requires_finite_unit_interval_evidence(self) -> None:
        with self.assertRaises(ValueError):
            HorizonScore(horizon=1, score=float("inf"), p_value=0.01, persistence=0.05)
        with self.assertRaises(ValueError):
            HorizonScore(horizon=1, score=1.0, p_value=1.01, persistence=0.05)
        with self.assertRaises(ValueError):
            HorizonScore(horizon=1, score=1.0, p_value=0.01, persistence=-0.01)

    def test_seed_decision_requires_three_exact_horizons_and_consistent_candidate(self) -> None:
        scores = {1: horizon(1), 2: horizon(2), 10: horizon(10, p_value=0.5, persistence=0.0)}
        self.assertTrue(SeedDecision(seed=41701, scores=scores, candidate=True).candidate)
        with self.assertRaisesRegex(ValueError, "three"):
            SeedDecision(seed=41701, scores={1: horizon(1), 2: horizon(2)}, candidate=True)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            SeedDecision(seed=41701, scores=scores, candidate=False)
        with self.assertRaises(ValueError):
            SeedDecision(seed=True, scores=scores, candidate=True)

    def test_group_qualification_requires_all_frozen_primary_types_and_finite_rates(self) -> None:
        with self.assertRaisesRegex(ValueError, "primary"):
            GroupQualification(
                length=100,
                series_count=100,
                status="qualified",
                oof_candidate_rate=0.01,
                wilson_upper=0.05,
                pooled_synthetic_sensitivity=0.70,
                per_type_synthetic_sensitivity={"level_shift": 0.50},
                seed_agreement=True,
            )
        with self.assertRaises(ValueError):
            GroupQualification(
                length=100,
                series_count=100,
                status="qualified",
                oof_candidate_rate=float("nan"),
                wilson_upper=0.05,
                pooled_synthetic_sensitivity=0.70,
                per_type_synthetic_sensitivity=group().per_type_synthetic_sensitivity,
                seed_agreement=True,
            )

    def test_dataset_result_rejects_lane_or_candidate_overlap(self) -> None:
        lanes = frozen_lanes(hard={"A": (1,), "hard": tuple(range(2, 2219))})
        with self.assertRaisesRegex(ValueError, "frozen lane"):
            dataset_result("A", candidates=(candidate("A", 1),), lanes=lanes)

    def test_dataset_result_requires_group_accounting_and_three_exact_seeds(self) -> None:
        with self.assertRaisesRegex(ValueError, "accounting"):
            dataset_result("A", group_count=2)
        with self.assertRaisesRegex(ValueError, "three"):
            ShadowCandidate(
                dataset="A",
                instance_index=1,
                length=100,
                seed_decisions=(
                    SeedDecision(
                        seed=41701,
                        scores={1: horizon(1), 2: horizon(2), 10: horizon(10, p_value=0.5, persistence=0.0)},
                        candidate=True,
                    ),
                ),
            )

    def test_dataset_and_global_workload_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "workload"):
            DatasetResult(
                dataset="A",
                test_size=20,
                group_count=1,
                groups=(group(),),
                candidates=(candidate("A", 1), candidate("A", 2)),
                lanes=frozen_lanes(),
                status="qualified",
                candidate_rate=0.10,
            )
        excessive = DatasetResult(
            dataset="A",
            test_size=20,
            group_count=1,
            groups=(group(),),
            candidates=(),
            lanes=frozen_lanes(),
            status="abstained",
            candidate_rate=0.10,
        )
        self.assertEqual(excessive.candidate_rate, 0.10)
        context = frozen_context()
        retained = results_for_context(context, retain_one_candidate_per_dataset=True)
        with self.assertRaisesRegex(ValueError, "global workload"):
            RunSummary(
                results=retained,
                total_test=20 * len(retained),
                publication_allowed=True,
                status="complete",
                context=context,
            )
        with self.assertRaisesRegex(ValueError, "global workload"):
            RunSummary(
                results=retained,
                total_test=20 * len(retained),
                publication_allowed=False,
                status="global_workload_gate_failed",
                context=context,
            )
        cleared = tuple(
            DatasetResult(
                dataset=result.dataset,
                test_size=result.test_size,
                group_count=result.group_count,
                groups=result.groups,
                candidates=(),
                lanes=result.lanes,
                status="abstained",
                observed_candidate_count=result.observed_candidate_count,
            )
            for result in retained
        )
        summary = RunSummary(
            results=cleared,
            total_test=20 * len(cleared),
            publication_allowed=False,
            status="global_workload_gate_failed",
            context=context,
        )
        self.assertFalse(summary.publication_allowed)
        self.assertFalse(any(result.candidates for result in summary.results))

    def test_run_summary_requires_exact_context_population_coverage(self) -> None:
        context = frozen_context()
        exact = results_for_context(context)
        complete = RunSummary(
            results=exact,
            total_test=20 * len(exact),
            publication_allowed=True,
            status="complete",
            context=context,
        )
        self.assertEqual(tuple(result.dataset for result in complete.results), context.datasets)
        with self.assertRaisesRegex(ValueError, "coverage"):
            RunSummary(
                results=exact[:-1],
                total_test=20 * (len(exact) - 1),
                publication_allowed=True,
                status="complete",
                context=context,
            )
        extra = dataset_result("dataset-1117", lanes=context.lanes)
        with self.assertRaisesRegex(ValueError, "coverage"):
            RunSummary(
                results=exact + (extra,),
                total_test=20 * (len(exact) + 1),
                publication_allowed=True,
                status="complete",
                context=context,
            )
        with self.assertRaisesRegex(ValueError, "sorted unique"):
            RunSummary(
                results=exact[:-1] + (exact[0],),
                total_test=20 * len(exact),
                publication_allowed=True,
                status="complete",
                context=context,
            )

    def test_candidate_requires_horizons_for_its_exact_length(self) -> None:
        wrong_scores = {
            1: horizon(1),
            3: horizon(3),
            10: horizon(10, p_value=0.50, persistence=0.0),
        }
        with self.assertRaisesRegex(ValueError, "horizons"):
            ShadowCandidate(
                dataset="A",
                instance_index=1,
                length=100,
                seed_decisions=tuple(
                    SeedDecision(seed=seed, scores=wrong_scores, candidate=True)
                    for seed in (41701, 41702, 41703)
                ),
            )

    def test_backend_audit_rejects_bool_and_nonfinite_measurements(self) -> None:
        with self.assertRaises(ValueError):
            BackendAudit(
                backend="cpu",
                candidate_identity=True,
                max_normalized_error=True,
                spearman=1.0,
                speedup=1.0,
                fallback=False,
            )
        with self.assertRaises(ValueError):
            BackendAudit(
                backend="cpu",
                candidate_identity=True,
                max_normalized_error=float("nan"),
                spearman=1.0,
                speedup=1.0,
                fallback=False,
            )

    def test_mps_backend_requires_every_frozen_parity_gate(self) -> None:
        with self.assertRaisesRegex(ValueError, "MPS"):
            BackendAudit(
                backend="mps",
                candidate_identity=False,
                max_normalized_error=1e-4,
                spearman=0.999,
                speedup=1.1,
                fallback=False,
            )
        audit = BackendAudit(
            backend="mps",
            candidate_identity=True,
            max_normalized_error=1e-4,
            spearman=0.999,
            speedup=1.1,
            fallback=False,
        )
        self.assertEqual(audit.backend, "mps")

    def test_processed_group_requires_legal_status_and_zero_scale_abstention(self) -> None:
        length_group = LengthGroup(
            length=3, series=(RawSeries(instance_index=0, values=(0.0, 1.0, 2.0)),)
        )
        with self.assertRaisesRegex(ValueError, "zero_scale_abstain"):
            ProcessedGroup(
                group=length_group,
                raw_center=0.0,
                raw_scale=1e-12,
                difference_center=0.0,
                difference_scale=1.0,
                status="candidate_eligible",
            )
        abstained = ProcessedGroup(
            group=length_group,
            raw_center=0.0,
            raw_scale=1e-12,
            difference_center=0.0,
            difference_scale=1.0,
            status="zero_scale_abstain",
        )
        self.assertEqual(abstained.status, "zero_scale_abstain")
        with self.assertRaisesRegex(ValueError, "processed group status"):
            ProcessedGroup(
                group=length_group,
                raw_center=0.0,
                raw_scale=1.0,
                difference_center=0.0,
                difference_scale=1.0,
                status="unknown",
            )

    def test_processed_group_retains_only_exact_immutable_float32_channels(self) -> None:
        length_group = LengthGroup(
            length=3, series=(RawSeries(instance_index=0, values=(0.0, 1.0, 2.0)),)
        )
        source = np.zeros((2, 3), dtype=np.float32)
        processed = ProcessedGroup(
            group=length_group,
            raw_center=0.0,
            raw_scale=1.0,
            difference_center=0.0,
            difference_scale=1.0,
            status="candidate_eligible",
            channels={0: source},
        )
        self.assertEqual(processed.channels[0].dtype, np.dtype("float32"))
        self.assertFalse(processed.channels[0].flags.writeable)
        source[0, 0] = 99.0
        self.assertEqual(float(processed.channels[0][0, 0]), 0.0)
        with self.assertRaises(ValueError):
            processed.channels[0].setflags(write=True)
        with self.assertRaises(ValueError):
            processed.channels[0][0, 0] = 1.0

    def test_usable_processed_groups_require_exact_channel_coverage(self) -> None:
        length_group = LengthGroup(
            length=3,
            series=(
                RawSeries(instance_index=0, values=(0.0, 1.0, 2.0)),
                RawSeries(instance_index=1, values=(1.0, 2.0, 3.0)),
            ),
        )
        channels = np.zeros((2, 3), dtype=np.float32)
        for status in ("diagnostic_only", "candidate_eligible"):
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "channels"):
                ProcessedGroup(
                    group=length_group,
                    raw_center=0.0,
                    raw_scale=1.0,
                    difference_center=0.0,
                    difference_scale=1.0,
                    status=status,
                )
            with self.subTest(status=status, coverage="missing"), self.assertRaisesRegex(ValueError, "cover"):
                ProcessedGroup(
                    group=length_group,
                    raw_center=0.0,
                    raw_scale=1.0,
                    difference_center=0.0,
                    difference_scale=1.0,
                    status=status,
                    channels={0: channels},
                )
            with self.subTest(status=status, coverage="extra"), self.assertRaisesRegex(ValueError, "cover"):
                ProcessedGroup(
                    group=length_group,
                    raw_center=0.0,
                    raw_scale=1.0,
                    difference_center=0.0,
                    difference_scale=1.0,
                    status=status,
                    channels={0: channels, 1: channels, 2: channels},
                )

        with self.assertRaisesRegex(ValueError, "abstained"):
            ProcessedGroup(
                group=length_group,
                raw_center=0.0,
                raw_scale=1.0,
                difference_center=0.0,
                difference_scale=1.0,
                status="horizon_abstain",
                channels={0: channels, 1: channels},
            )

    def test_frozen_train_contract_preserves_only_the_qualified_near_repeat_state(self) -> None:
        contract = FrozenTrainContract(
            dataset="A",
            groups=(group(),),
            backend=BackendAudit(
                backend="cpu",
                candidate_identity=True,
                max_normalized_error=0.0,
                spearman=1.0,
                speedup=1.0,
                fallback=False,
            ),
            train_identity_sha256="a" * 64,
        )
        self.assertEqual(contract.near_repeat_status, "not_available")
        with self.assertRaisesRegex(ValueError, "near-repeat"):
            FrozenTrainContract(
                dataset="A",
                groups=(group(),),
                backend=contract.backend,
                train_identity_sha256="a" * 64,
                near_repeat_status="applied",
            )


if __name__ == "__main__":
    unittest.main()
