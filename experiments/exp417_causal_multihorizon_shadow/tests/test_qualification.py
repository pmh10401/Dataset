"""Synthetic boundary tests for Exp417's frozen safety and workload gates."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
import unittest

import numpy as np

from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG
from experiments.exp417_causal_multihorizon_shadow.contracts import (
    BackendAudit,
    DatasetResult,
    FrozenLaneSnapshot,
    FrozenRunContext,
    GroupQualification,
    HorizonScore,
    SeedDecision,
    ShadowCandidate,
)
from experiments.exp417_causal_multihorizon_shadow.qualification import (
    apply_dataset_workload,
    apply_global_workload,
    build_fixed_perturbations,
    exclude_frozen_lanes,
    freeze_train_contract,
    qualify_group,
    wilson_upper,
)


def lanes(*, protected: tuple[int, int, int, int] = (1, 2, 3, 4)) -> FrozenLaneSnapshot:
    """Use the four real lane types while protecting the requested TEST indices."""
    hard, standard, priority, automatic = protected
    return FrozenLaneSnapshot(
        hard={"dataset": (hard,) + tuple(range(10_000, 12_217))},
        standard={"dataset": (standard,) + tuple(range(20_000, 20_805))},
        priority={"dataset": (priority,) + tuple(range(30_000, 30_030))},
        automatic_normal={"dataset": (automatic,) + tuple(range(40_000, 40_034))},
    )


def scores(*, candidate: bool) -> dict[int, HorizonScore]:
    return {
        1: HorizonScore(1, 1.0, 0.01 if candidate else 0.5, 0.05 if candidate else 0.0),
        2: HorizonScore(2, 1.0, 0.01 if candidate else 0.5, 0.05 if candidate else 0.0),
        10: HorizonScore(10, 1.0, 0.5, 0.0),
    }


def decision_bundle(*, candidate: bool = True) -> tuple[SeedDecision, ...]:
    return tuple(
        SeedDecision(seed=seed, scores=scores(candidate=candidate), candidate=candidate)
        for seed in FROZEN_CONFIG.model_seeds
    )


def shadow_candidate(index: int, *, dataset: str = "dataset") -> ShadowCandidate:
    return ShadowCandidate(
        dataset=dataset,
        instance_index=index,
        length=100,
        seed_decisions=decision_bundle(),
    )


def qualifying_group() -> GroupQualification:
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


def synthetic_decisions(*, detected: bool = True) -> dict[str, tuple[tuple[SeedDecision, ...], ...]]:
    bundle = decision_bundle(candidate=detected)
    return {
        "level_shift": (bundle,),
        "gradual_drift": (bundle,),
        "amplitude_scaling": (bundle,),
        "period_change": (bundle,),
        "secondary_spike": (bundle,),
    }


def context_for(results: tuple[DatasetResult, ...]) -> FrozenRunContext:
    """A fully frozen, synthetic context matching a 1,117-dataset result set."""
    names = tuple(result.dataset for result in results)
    return FrozenRunContext(
        database_sha256=FROZEN_CONFIG.database_sha256,
        datasets=names,
        lanes=results[0].lanes,
        code_commit="synthetic",
        runtime_versions={"python": "synthetic"},
    )


class FixedPerturbationTests(unittest.TestCase):
    def test_fixed_perturbations_have_the_preregistered_shape_and_magnitudes(self) -> None:
        """Changing an onset, scale, or interpolation endpoint breaks fixed qualification."""
        ramp = np.arange(100, dtype=np.float64) - 49.5
        perturbations = build_fixed_perturbations(ramp, 2.0, seed=41711)

        self.assertEqual(set(perturbations), {
            "level_shift", "gradual_drift", "amplitude_scaling", "period_change", "secondary_spike"
        })
        self.assertIsInstance(perturbations, MappingProxyType)
        self.assertFalse(perturbations["level_shift"].flags.writeable)
        self.assertTrue(np.array_equal(perturbations["level_shift"][:40], ramp[:40]))
        self.assertTrue(np.array_equal(perturbations["level_shift"][40:], ramp[40:] + 6.0))
        self.assertEqual(perturbations["gradual_drift"][40], ramp[40])
        self.assertEqual(perturbations["gradual_drift"][-1], ramp[-1] + 6.0)
        self.assertTrue(np.array_equal(perturbations["amplitude_scaling"], ramp * 1.5))
        self.assertEqual(perturbations["period_change"].shape, ramp.shape)
        self.assertTrue(np.array_equal(perturbations["period_change"][:40], ramp[:40]))
        spike = perturbations["secondary_spike"] - ramp
        self.assertEqual(np.count_nonzero(spike), 1)
        self.assertEqual(float(spike[spike != 0.0][0]), 10.0)

    def test_fixed_perturbation_seed_is_deterministic(self) -> None:
        """Changing seeded placement must not silently change synthetic qualification."""
        series = np.arange(100, dtype=np.float64)
        first = build_fixed_perturbations(series, 1.0, seed=41712)
        second = build_fixed_perturbations(series, 1.0, seed=41712)
        for name in first:
            self.assertTrue(np.array_equal(first[name], second[name]))


class QualificationBoundaryTests(unittest.TestCase):
    def test_qualification_accepts_all_exact_frozen_boundaries(self) -> None:
        """Tightening a <= or >= bound must not reject an exact preregistered value."""
        normal = tuple(decision_bundle(candidate=False) for _ in range(100))
        result = qualify_group(
            length=100,
            normal_oof_decisions=normal,
            synthetic_decisions=synthetic_decisions(),
        )
        self.assertEqual(result.status, "qualified")
        self.assertEqual(result.oof_candidate_rate, 0.0)
        self.assertGreaterEqual(result.pooled_synthetic_sensitivity, 0.70)
        self.assertTrue(all(rate >= 0.50 for rate in result.per_type_synthetic_sensitivity.values()))

    def test_group_contract_rejects_next_oof_and_wilson_values_above_boundaries(self) -> None:
        """Replacing inclusive safety bounds with a permissive comparison is a bug."""
        exact = qualifying_group()
        self.assertEqual(exact.oof_candidate_rate, 0.01)
        self.assertEqual(exact.wilson_upper, 0.05)
        self.assertEqual(exact.pooled_synthetic_sensitivity, 0.70)
        self.assertTrue(all(rate == 0.50 for rate in exact.per_type_synthetic_sensitivity.values()))
        with self.assertRaises(ValueError):
            replace(exact, oof_candidate_rate=np.nextafter(0.01, np.inf))
        with self.assertRaises(ValueError):
            replace(exact, wilson_upper=np.nextafter(0.05, np.inf))

    def test_wilson_upper_uses_the_frozen_95_percent_upper_bound(self) -> None:
        """Using a lower bound or a different confidence level misstates normal OOF safety."""
        self.assertAlmostEqual(wilson_upper(0, 100), 0.03699349820698568)
        self.assertLessEqual(wilson_upper(0, 73), 0.05)
        self.assertGreater(wilson_upper(0, 72), 0.05)

    def test_one_seed_disagreement_nonfinite_or_fallback_fails_closed(self) -> None:
        """Allowing partial-seed, nonfinite, or fallback evidence would qualify unsafe TRAIN groups."""
        normal = tuple(decision_bundle(candidate=False) for _ in range(100))
        disagreeing = list(decision_bundle(candidate=True))
        disagreeing[-1] = SeedDecision(
            seed=41703, scores=scores(candidate=False), candidate=False
        )
        for kwargs in (
            {"synthetic_decisions": {**synthetic_decisions(), "level_shift": (tuple(disagreeing),)}},
            {"synthetic_decisions": synthetic_decisions(), "finite": False},
            {"synthetic_decisions": synthetic_decisions(), "backend_fallback": True},
        ):
            with self.subTest(kwargs=kwargs):
                result = qualify_group(length=100, normal_oof_decisions=normal, **kwargs)
                self.assertEqual(result.status, "failed")


class LaneAndWorkloadTests(unittest.TestCase):
    def test_excluding_all_frozen_lanes_preserves_the_original_snapshot(self) -> None:
        """Dropping a lane type or mutating its coordinates would promote frozen work."""
        snapshot = lanes()
        candidates = tuple(shadow_candidate(index) for index in range(1, 6))
        retained = exclude_frozen_lanes(candidates, snapshot)
        self.assertEqual(tuple(candidate.instance_index for candidate in retained), (5,))
        self.assertEqual(snapshot.hard["dataset"][0], 1)
        self.assertEqual(snapshot.standard["dataset"][0], 2)
        self.assertEqual(snapshot.priority["dataset"][0], 3)
        self.assertEqual(snapshot.automatic_normal["dataset"][0], 4)
        self.assertIsInstance(snapshot.hard, MappingProxyType)

    def test_dataset_workload_is_all_or_none_at_five_percent(self) -> None:
        """Retaining a top subset above the dataset cap would violate the fail-closed policy."""
        snapshot = lanes(protected=(100, 101, 102, 103))
        groups = (qualifying_group(),)
        exact = apply_dataset_workload(
            "dataset", 20, groups, (shadow_candidate(5),), snapshot
        )
        rejected = apply_dataset_workload(
            "dataset", 20, groups, (shadow_candidate(5), shadow_candidate(6)), snapshot
        )
        self.assertEqual(exact.status, "qualified")
        self.assertEqual(len(exact.candidates), 1)
        self.assertEqual(rejected.status, "abstained")
        self.assertEqual(rejected.candidates, ())
        self.assertEqual(rejected.observed_candidate_count, 2)
        self.assertEqual(rejected.candidate_rate, 0.10)

    def test_dataset_result_binds_pre_gate_count_to_its_rate(self) -> None:
        """A caller must not lower the recorded workload count after clearing a payload."""
        snapshot = lanes(protected=(100, 101, 102, 103))
        with self.assertRaises(ValueError):
            DatasetResult(
                dataset="dataset",
                test_size=20,
                group_count=1,
                groups=(qualifying_group(),),
                candidates=(),
                lanes=snapshot,
                status="abstained",
                observed_candidate_count=2,
                candidate_rate=0.05,
            )

    def test_workload_does_not_sort_or_select_top_n_candidates(self) -> None:
        """Changing candidate order must not turn workload gating into score-ranked selection."""
        snapshot = lanes(protected=(100, 101, 102, 103))
        candidates = (shadow_candidate(5), shadow_candidate(6))
        first = apply_dataset_workload("dataset", 20, (qualifying_group(),), candidates, snapshot)
        second = apply_dataset_workload("dataset", 20, (qualifying_group(),), tuple(reversed(candidates)), snapshot)
        self.assertEqual((first.status, first.candidates), ("abstained", ()))
        self.assertEqual((second.status, second.candidates), ("abstained", ()))

    def test_global_workload_is_all_or_none_at_one_percent(self) -> None:
        """Publishing a partial global candidate list above the cap would expose unofficial payloads."""
        snapshot = lanes(protected=(100, 101, 102, 103))
        groups = (qualifying_group(),)
        names = tuple(f"dataset-{index:04d}" for index in range(1117))
        exact_results = tuple(
            apply_dataset_workload(name, 100, groups, (shadow_candidate(5, dataset=name),), snapshot)
            for name in names
        )
        exact = apply_global_workload(exact_results, 111_700, context=context_for(exact_results))
        too_many = exact_results[:1] + (
            apply_dataset_workload(
                names[1], 100, groups,
                (shadow_candidate(5, dataset=names[1]), shadow_candidate(6, dataset=names[1])), snapshot,
            ),
        ) + exact_results[2:]
        rejected = apply_global_workload(too_many, 111_700, context=context_for(too_many))
        self.assertTrue(exact.publication_allowed)
        self.assertEqual(exact.status, "complete")
        self.assertFalse(rejected.publication_allowed)
        self.assertEqual(rejected.status, "global_workload_gate_failed")
        self.assertFalse(any(result.candidates for result in rejected.results))


class FreezeContractTests(unittest.TestCase):
    def test_freeze_train_contract_records_qualified_groups_and_not_available_guard(self) -> None:
        """Changing the contract builder must not invent a near-repeat capability."""
        contract = freeze_train_contract(
            "dataset",
            (qualifying_group(),),
            BackendAudit("cpu", True, 0.0, 1.0, 1.0, False),
        )
        self.assertEqual(contract.near_repeat_status, "not_available")
        self.assertEqual(contract.groups, (qualifying_group(),))
        self.assertEqual(len(contract.train_identity_sha256), 64)


if __name__ == "__main__":
    unittest.main()
