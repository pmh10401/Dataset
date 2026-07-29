from types import MappingProxyType
import unittest

from experiments.exp417_causal_multihorizon_shadow.contracts import (
    BackendAudit,
    DatasetResult,
    FrozenLaneSnapshot,
    GroupQualification,
    HorizonScore,
    RawSeries,
    SeedDecision,
    ShadowCandidate,
)


def frozen_lanes(
    *, hard: dict[str, tuple[int, ...]] | None = None
) -> FrozenLaneSnapshot:
    return FrozenLaneSnapshot(
        hard=hard or {},
        standard={},
        priority={},
        automatic_normal={},
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
) -> DatasetResult:
    return DatasetResult(
        dataset=dataset,
        test_size=20,
        group_count=group_count,
        groups=(group(),),
        candidates=candidates,
        lanes=lanes or frozen_lanes(),
        status="qualified",
    )


class ContractTests(unittest.TestCase):
    def test_raw_series_rejects_bool_index_and_nonfinite(self) -> None:
        with self.assertRaises(ValueError):
            RawSeries(instance_index=True, values=(0.0, 1.0))
        with self.assertRaises(ValueError):
            RawSeries(instance_index=1, values=(0.0, float("nan")))

    def test_coordinates_are_sorted_unique_and_mappings_are_immutable(self) -> None:
        with self.assertRaisesRegex(ValueError, "sorted"):
            FrozenLaneSnapshot(
                hard={"A": (2, 1)}, standard={}, priority={}, automatic_normal={}
            )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            FrozenLaneSnapshot(
                hard={"A": (1, 1)}, standard={}, priority={}, automatic_normal={}
            )
        lanes = frozen_lanes(hard={"A": (1,)})
        self.assertIsInstance(lanes.hard, MappingProxyType)
        with self.assertRaises(TypeError):
            lanes.hard["B"] = (2,)  # type: ignore[index]

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
        lanes = frozen_lanes(hard={"A": (1,)})
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


if __name__ == "__main__":
    unittest.main()
