"""Synthetic boundary tests for Exp417's frozen safety and workload gates."""

from __future__ import annotations

from dataclasses import replace
from math import ceil
from types import MappingProxyType
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG
from experiments.exp417_causal_multihorizon_shadow.contracts import (
    BackendAudit,
    DatasetDecisionEvidence,
    DatasetResult,
    FrozenLaneSnapshot,
    FrozenRunContext,
    GroupQualification,
    HorizonScore,
    SeedDecision,
    ShadowCandidate,
    TestInstanceDecision,
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


def instance_decision(
    index: int, *, dataset: str = "dataset", candidate: bool = False, length: int = 100
) -> TestInstanceDecision:
    return TestInstanceDecision(
        dataset=dataset,
        instance_index=index,
        length=length,
        seed_decisions=decision_bundle(candidate=candidate),
    )


def instance_decision_for_length(index: int, length: int) -> TestInstanceDecision:
    horizons = (1, ceil(0.02 * length), ceil(0.10 * length))
    seed_scores = {
        horizons[0]: HorizonScore(horizons[0], 1.0, 0.5, 0.0),
        horizons[1]: HorizonScore(horizons[1], 1.0, 0.5, 0.0),
        horizons[2]: HorizonScore(horizons[2], 1.0, 0.5, 0.0),
    }
    return TestInstanceDecision(
        dataset="dataset",
        instance_index=index,
        length=length,
        seed_decisions=tuple(
            SeedDecision(seed=seed, scores=seed_scores, candidate=False)
            for seed in FROZEN_CONFIG.model_seeds
        ),
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


def train_contract(dataset: str = "dataset"):
    return freeze_train_contract(
        dataset,
        (qualifying_group(),),
        BackendAudit("cpu", True, 0.0, 1.0, 1.0, False),
    )


def decision_evidence(
    context: FrozenRunContext,
    *,
    dataset: str = "dataset",
    candidate_indices: tuple[int, ...] = (),
    contract=None,
    decisions: tuple[TestInstanceDecision, ...] | None = None,
) -> DatasetDecisionEvidence:
    bound_contract = train_contract(dataset) if contract is None else contract
    size = context.test_sizes[dataset]
    supplied = decisions or tuple(
        instance_decision(index, dataset=dataset, candidate=index in candidate_indices)
        for index in range(size)
    )
    return DatasetDecisionEvidence(
        dataset=dataset,
        context=context,
        frozen_train_contract=bound_contract,
        database_sha256=context.database_sha256,
        train_identity_sha256=bound_contract.train_identity_sha256,
        decisions=supplied,
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


def boundary_synthetic_decisions(
    *, per_type_detected: int = 1, spike_detected: int = 10
) -> dict[str, tuple[tuple[SeedDecision, ...], ...]]:
    """Hand-built 20-fixture pool: primary types at 1/2 and spike at 10/12."""
    return {
        **{
            name: tuple(decision_bundle(candidate=index < per_type_detected) for index in range(2))
            for name in ("level_shift", "gradual_drift", "amplitude_scaling", "period_change")
        },
        "secondary_spike": tuple(
            decision_bundle(candidate=index < spike_detected) for index in range(12)
        ),
    }


def context_for(
    names: tuple[str, ...], lanes_snapshot: FrozenLaneSnapshot, *, test_size: int
) -> FrozenRunContext:
    """A fully frozen, synthetic context matching a 1,117-dataset result set."""
    return FrozenRunContext(
        database_sha256=FROZEN_CONFIG.database_sha256,
        datasets=names,
        lanes=lanes_snapshot,
        test_identity_manifest={
            name: tuple((index, 100) for index in range(test_size)) for name in names
        },
        code_commit="synthetic",
        runtime_versions={"python": "synthetic"},
    )


def single_context(lanes_snapshot: FrozenLaneSnapshot, *, test_size: int) -> FrozenRunContext:
    names = ("dataset",) + tuple(f"synthetic-{index:04d}" for index in range(1116))
    return FrozenRunContext(
        database_sha256=FROZEN_CONFIG.database_sha256,
        datasets=names,
        lanes=lanes_snapshot,
        test_identity_manifest={
            name: tuple((index, 100) for index in range(test_size)) for name in names
        },
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
    def test_qualify_group_enforces_oof_wilson_and_synthetic_boundaries(self) -> None:
        """Relaxing any frozen boundary must not qualify the immediately unsafe synthetic fixture."""
        normal_at_rate = (decision_bundle(candidate=True),) + tuple(
            decision_bundle(candidate=False) for _ in range(99)
        )
        normal_above_rate = (decision_bundle(candidate=True),) * 2 + tuple(
            decision_bundle(candidate=False) for _ in range(98)
        )
        exact_synthetic = boundary_synthetic_decisions()
        with patch(
            "experiments.exp417_causal_multihorizon_shadow.qualification.wilson_upper",
            return_value=0.05,
        ):
            exact = qualify_group(
                length=100,
                normal_oof_decisions=normal_at_rate,
                synthetic_decisions=exact_synthetic,
            )
            above_oof = qualify_group(
                length=100,
                normal_oof_decisions=normal_above_rate,
                synthetic_decisions=exact_synthetic,
            )
            below_pooled = qualify_group(
                length=100,
                normal_oof_decisions=normal_at_rate,
                synthetic_decisions=boundary_synthetic_decisions(spike_detected=9),
            )
            below_type = qualify_group(
                length=100,
                normal_oof_decisions=normal_at_rate,
                synthetic_decisions=boundary_synthetic_decisions(per_type_detected=0),
            )
        self.assertEqual(exact.status, "qualified")
        self.assertEqual(above_oof.status, "failed")
        self.assertEqual(below_pooled.status, "failed")
        self.assertEqual(below_type.status, "failed")

    def test_qualify_group_fails_just_above_the_wilson_boundary(self) -> None:
        """A strict Wilson comparison is required even when discrete OOF counts cannot land on 0.05."""
        normal = (decision_bundle(candidate=True),) + tuple(
            decision_bundle(candidate=False) for _ in range(99)
        )
        with patch(
            "experiments.exp417_causal_multihorizon_shadow.qualification.wilson_upper",
            return_value=np.nextafter(0.05, np.inf),
        ):
            result = qualify_group(
                length=100,
                normal_oof_decisions=normal,
                synthetic_decisions=boundary_synthetic_decisions(),
            )
        self.assertEqual(result.status, "failed")

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
    def test_decision_evidence_requires_exact_noncontiguous_identity_length_manifest(self) -> None:
        """Range-only coverage or swapping two valid lengths must not be accepted as TEST evidence."""
        snapshot = lanes(protected=(100, 101, 102, 103))
        names = ("dataset",) + tuple(f"synthetic-{index:04d}" for index in range(1116))
        manifest = {name: ((0, 100),) for name in names}
        manifest["dataset"] = ((2, 64), (9, 100))
        context = FrozenRunContext(
            database_sha256=FROZEN_CONFIG.database_sha256,
            datasets=names,
            lanes=snapshot,
            test_identity_manifest=manifest,
            code_commit="synthetic",
            runtime_versions={"python": "synthetic"},
        )
        groups = (
            replace(qualifying_group(), length=64),
            qualifying_group(),
        )
        contract = freeze_train_contract(
            "dataset", groups, BackendAudit("cpu", True, 0.0, 1.0, 1.0, False)
        )
        exact = DatasetDecisionEvidence(
            dataset="dataset",
            context=context,
            frozen_train_contract=contract,
            database_sha256=context.database_sha256,
            train_identity_sha256=contract.train_identity_sha256,
            decisions=(instance_decision_for_length(2, 64), instance_decision(9)),
        )
        self.assertEqual(exact.test_size, 2)
        with self.assertRaises(ValueError):
            replace(exact, decisions=(instance_decision_for_length(2, 100), instance_decision_for_length(9, 64)))
        with self.assertRaises(ValueError):
            replace(exact, decisions=(instance_decision(0), instance_decision(1)))

    def test_complete_decision_evidence_derives_candidates_and_rejects_coverage_attacks(self) -> None:
        """Removing, duplicating, reordering, or replacing decisions must not understate workload."""
        snapshot = lanes(protected=(1, 100, 101, 102))
        context = single_context(snapshot, test_size=20)
        complete = decision_evidence(context, candidate_indices=(1, 5))
        self.assertEqual(
            tuple(candidate.instance_index for candidate in complete.derived_candidates), (5,)
        )
        malformed = (
            complete.decisions[:-1],
            complete.decisions + (complete.decisions[-1],),
            tuple(reversed(complete.decisions)),
        )
        for decisions in malformed:
            with self.subTest(decisions=decisions):
                with self.assertRaises(ValueError):
                    decision_evidence(context, decisions=decisions)
        with self.assertRaises(ValueError):
            replace(complete, decisions=())

    def test_decision_evidence_rejects_wrong_context_train_contract_length_and_seeds(self) -> None:
        """Evidence must be bound to one frozen context, TRAIN contract, length, and seed set."""
        snapshot = lanes(protected=(100, 101, 102, 103))
        context = single_context(snapshot, test_size=20)
        complete = decision_evidence(context)
        wrong_context = single_context(snapshot, test_size=21)
        with self.assertRaises(ValueError):
            replace(complete, context=wrong_context)
        with self.assertRaises(ValueError):
            replace(complete, frozen_train_contract=train_contract("other"))
        wrong_length = list(complete.decisions)
        wrong_length[0] = instance_decision_for_length(0, 64)
        with self.assertRaises(ValueError):
            replace(complete, decisions=tuple(wrong_length))
        with self.assertRaises(ValueError):
            TestInstanceDecision(
                dataset="dataset",
                instance_index=0,
                length=100,
                seed_decisions=(decision_bundle()[0], decision_bundle()[0], decision_bundle()[2]),
            )

    def test_excluding_all_frozen_lanes_preserves_the_original_snapshot(self) -> None:
        """Dropping a lane type or mutating its coordinates would promote frozen work."""
        snapshot = lanes()
        context = single_context(snapshot, test_size=20)
        retained = exclude_frozen_lanes(
            decision_evidence(context, candidate_indices=(1, 2, 3, 4, 5))
        )
        self.assertEqual(tuple(candidate.instance_index for candidate in retained), (5,))
        self.assertEqual(snapshot.hard["dataset"][0], 1)
        self.assertEqual(snapshot.standard["dataset"][0], 2)
        self.assertEqual(snapshot.priority["dataset"][0], 3)
        self.assertEqual(snapshot.automatic_normal["dataset"][0], 4)
        self.assertIsInstance(snapshot.hard, MappingProxyType)

    def test_dataset_workload_is_all_or_none_at_five_percent(self) -> None:
        """Retaining a top subset above the dataset cap would violate the fail-closed policy."""
        snapshot = lanes(protected=(100, 101, 102, 103))
        context = single_context(snapshot, test_size=20)
        exact = apply_dataset_workload(decision_evidence(context, candidate_indices=(5,)))
        rejected = apply_dataset_workload(decision_evidence(context, candidate_indices=(5, 6)))
        self.assertEqual(exact.status, "qualified")
        self.assertEqual(len(exact.candidates), 1)
        self.assertEqual(rejected.status, "abstained")
        self.assertEqual(rejected.candidates, ())
        self.assertEqual(rejected.observed_candidate_count, 2)
        self.assertEqual(rejected.candidate_rate, 0.10)

    def test_dataset_result_binds_pre_gate_count_to_its_rate(self) -> None:
        """A caller must not lower the recorded workload count after clearing a payload."""
        snapshot = lanes(protected=(100, 101, 102, 103))
        context = single_context(snapshot, test_size=20)
        with self.assertRaises(TypeError):
            DatasetResult(
                dataset="dataset",
                test_size=20,
                group_count=1,
                groups=(qualifying_group(),),
                candidates=(),
                decision_evidence=decision_evidence(context),
                lanes=snapshot,
                context=context,
                status="abstained",
                observed_candidate_count=2,
                candidate_rate=0.05,
            )

    def test_workload_does_not_sort_or_select_top_n_candidates(self) -> None:
        """Changing candidate order must not turn workload gating into score-ranked selection."""
        snapshot = lanes(protected=(100, 101, 102, 103))
        context = single_context(snapshot, test_size=20)
        first = apply_dataset_workload(decision_evidence(context, candidate_indices=(5, 6)))
        second = apply_dataset_workload(decision_evidence(context, candidate_indices=(6, 5)))
        self.assertEqual((first.status, first.candidates), ("abstained", ()))
        self.assertEqual((second.status, second.candidates), ("abstained", ()))

    def test_global_workload_is_all_or_none_at_one_percent(self) -> None:
        """Publishing a partial global candidate list above the cap would expose unofficial payloads."""
        snapshot = lanes(protected=(100, 101, 102, 103))
        names = tuple(f"dataset-{index:04d}" for index in range(1117))
        context = context_for(names, snapshot, test_size=100)
        exact_results = tuple(
            apply_dataset_workload(
                decision_evidence(context, dataset=name, candidate_indices=(5,))
            )
            for name in names
        )
        exact = apply_global_workload(exact_results, 111_700, context=context)
        too_many = exact_results[:1] + (
            apply_dataset_workload(
                decision_evidence(context, dataset=names[1], candidate_indices=(5, 6))
            ),
        ) + exact_results[2:]
        rejected = apply_global_workload(too_many, 111_700, context=context)
        self.assertTrue(exact.publication_allowed)
        self.assertEqual(exact.status, "complete")
        self.assertFalse(rejected.publication_allowed)
        self.assertEqual(rejected.status, "global_workload_gate_failed")
        self.assertFalse(any(result.candidates for result in rejected.results))

    def test_global_workload_counts_locally_abstained_observed_evidence(self) -> None:
        """Clearing a 6%-dataset payload must not erase its contribution to the 1% global gate."""
        snapshot = lanes(protected=(100, 101, 102, 103))
        names = tuple(f"dataset-{index:04d}" for index in range(1117))
        context = FrozenRunContext(
            database_sha256=FROZEN_CONFIG.database_sha256,
            datasets=names,
            lanes=snapshot,
            test_identity_manifest={
                name: tuple((index, 100) for index in range(100)) for name in names
            },
            code_commit="synthetic",
            runtime_versions={"python": "synthetic"},
        )
        local_abstention = apply_dataset_workload(
            decision_evidence(context, dataset=names[0], candidate_indices=(0, 1, 2, 3, 4, 5))
        )
        others = tuple(
            apply_dataset_workload(
                decision_evidence(context, dataset=name, candidate_indices=(0,))
            )
            for name in names[1:]
        )
        summary = apply_global_workload(
            (local_abstention,) + others, 111_700, context=context
        )
        self.assertEqual(local_abstention.candidates, ())
        self.assertEqual(len(local_abstention.observed_candidates), 6)
        self.assertEqual(summary.status, "global_workload_gate_failed")
        self.assertFalse(summary.publication_allowed)
        self.assertFalse(any(result.candidates for result in summary.results))


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
