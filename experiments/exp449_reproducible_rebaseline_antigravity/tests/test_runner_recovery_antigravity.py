"""Exp449 in-memory Task4/5 completion gate tests."""

from __future__ import annotations

from contextlib import ExitStack
from itertools import count
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp449_reproducible_rebaseline_antigravity import runner_antigravity as runner
from paper.equipment_anomaly_detection import proposal_score_paper
from paper.equipment_anomaly_detection.contracts_paper import LowFrequencyResult, RawDataset
from paper.equipment_anomaly_detection.hard_route_paper import HardRouteResult, RouteLanes
from paper.equipment_anomaly_detection.routing_paper import PolicyBundle
from paper.equipment_anomaly_detection.sources_paper.hard_sources_paper import HardSourceBundle
from paper.equipment_anomaly_detection.sources_paper.stability_paper import StabilityPartition


def _raw_dataset(dataset: str) -> RawDataset:
    train = tuple(np.asarray((float(index), float(index + 1)), dtype=np.float64) for index in range(4))
    test = (np.asarray((1.0, 2.0), dtype=np.float64),)
    return RawDataset(dataset, 2, (0, 1, 2, 3), (b"0", b"1", b"2", b"3"), train, (0,), (b"test",), test)


def _route() -> HardRouteResult:
    route = object.__new__(HardRouteResult)
    object.__setattr__(route, "exp253", RouteLanes(frozenset(), frozenset(), frozenset()))
    return route


def _typed_stage_result(cls: type[object], **attributes: object) -> object:
    result = object.__new__(cls)
    for name, value in attributes.items():
        object.__setattr__(result, name, value)
    return result


class Exp449DatasetLocalBijectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = {"Alpha": _raw_dataset("Alpha"), "Beta": _raw_dataset("Beta")}
        self.sources = {dataset: HardSourceBundle((), {}) for dataset in self.raw}
        self.routes = {dataset: _route() for dataset in self.raw}
        self.catalog = {"datasets": [{"dataset": "Alpha"}, {"dataset": "Beta"}]}

    def test_missing_dataset_is_rejected(self) -> None:
        """Removing one local Task4/5 return must close the global-scoring gate."""
        with self.assertRaisesRegex(ValueError, "catalog bijection"):
            runner.verify_dataset_local_bijection_antigravity(
                self.catalog, {"Alpha": self.raw["Alpha"]}, {"Alpha": self.sources["Alpha"]},
                {"Alpha": self.routes["Alpha"]}, mode="smoke",
            )

    def test_duplicate_catalog_dataset_is_rejected(self) -> None:
        """A duplicate catalog entry must never collapse into a mapping silently."""
        with self.assertRaisesRegex(ValueError, "catalog bijection"):
            runner.verify_dataset_local_bijection_antigravity(
                {"datasets": [{"dataset": "Alpha"}, {"dataset": "Alpha"}]},
                self.raw, self.sources, self.routes, mode="smoke",
            )

    def test_extra_dataset_is_rejected(self) -> None:
        """An unlisted local result must not enter population-global scoring."""
        raw = {**self.raw, "Gamma": _raw_dataset("Gamma")}
        sources = {**self.sources, "Gamma": HardSourceBundle((), {})}
        routes = {**self.routes, "Gamma": _route()}
        with self.assertRaisesRegex(ValueError, "catalog bijection"):
            runner.verify_dataset_local_bijection_antigravity(self.catalog, raw, sources, routes, mode="smoke")

    def test_catalog_order_drift_is_rejected(self) -> None:
        """A catalog whose order drifts from canonical lexical order is invalid."""
        with self.assertRaisesRegex(ValueError, "catalog bijection"):
            runner.verify_dataset_local_bijection_antigravity(
                {"datasets": [{"dataset": "Beta"}, {"dataset": "Alpha"}]},
                self.raw, self.sources, self.routes, mode="smoke",
            )

    def test_source_route_identity_mismatch_is_rejected(self) -> None:
        """Task4 and Task5 must each cover the exact same ordered catalog."""
        with self.assertRaisesRegex(ValueError, "catalog bijection"):
            runner.verify_dataset_local_bijection_antigravity(
                self.catalog, self.raw, {"Alpha": self.sources["Alpha"]}, self.routes, mode="smoke",
            )

    def test_full_catalog_requires_exactly_1117_datasets(self) -> None:
        """The full mode cannot accidentally run Task6 over a partial population."""
        with self.assertRaisesRegex(ValueError, "full catalog count"):
            runner.verify_dataset_local_bijection_antigravity(
                self.catalog, self.raw, self.sources, self.routes, mode="full",
            )

    def test_complete_smoke_catalog_returns_its_exact_order(self) -> None:
        """The gate returns the successful catalog order for downstream composition."""
        self.assertEqual(
            runner.verify_dataset_local_bijection_antigravity(
                self.catalog, self.raw, self.sources, self.routes, mode="smoke",
            ),
            ("Alpha", "Beta"),
        )


class Exp449CompositionGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.raw = {"Alpha": _raw_dataset("Alpha"), "Beta": _raw_dataset("Beta")}
        self.catalog = {"datasets": [{"dataset": "Alpha"}, {"dataset": "Beta"}]}
        self.acceleration = {
            "selected_backend": "cpu",
            "fallback_reason": "MLX parity benchmark not yet approved",
            "assessment_sha256": "a" * 64,
            "workers": {
                "logical_cpu_count": 2,
                "dataset_workers": 1,
                "inner_numerical_threads": 1,
                "outer_execution_mode": "parallel_task4_task5",
            },
        }
        self.sources = {dataset: HardSourceBundle((), {}) for dataset in self.raw}
        self.routes = {dataset: _route() for dataset in self.raw}
        self.low = _typed_stage_result(LowFrequencyResult, datasets=("Alpha", "Beta"))
        self.stability = _typed_stage_result(StabilityPartition)
        self.policies = _typed_stage_result(PolicyBundle)

    def _collector(self, raw: object, backend: object, **kwargs: object) -> tuple[dict[str, HardSourceBundle], dict[str, HardRouteResult]]:
        return self.sources, self.routes

    def _stage_patches(self):
        return (
            patch.object(runner, "score_low_frequency", return_value=self.low),
            patch.object(runner, "classify_stability", return_value=self.stability),
            patch.object(runner, "build_hard_evidence_surface", return_value=object()),
            patch.object(runner, "build_policies", return_value=self.policies),
        )

    def _assert_proposal_score_semantic_parity(self, actual, expected) -> None:
        self.assertEqual(tuple(actual.raw), tuple(expected.raw))
        self.assertEqual(tuple(actual.sources), tuple(expected.sources))
        self.assertEqual(tuple(actual.routes), tuple(expected.routes))
        for dataset in actual.raw:
            self.assertIs(actual.raw[dataset], expected.raw[dataset])
            self.assertIs(actual.sources[dataset], expected.sources[dataset])
            self.assertIs(actual.routes[dataset], expected.routes[dataset])
        self.assertIs(actual.low_frequency, expected.low_frequency)
        self.assertIs(actual.stability, expected.stability)
        self.assertIs(actual.policies, expected.policies)
        self.assertEqual(actual.backend, expected.backend)
        self.assertEqual(dict(actual.stage_seconds), dict(expected.stage_seconds))

    def test_missing_success_blocks_task6_and_task7(self) -> None:
        """A local omission fails before either population-global scientific task starts."""
        def missing_collector(raw: object, backend: object, **kwargs: object):
            return {"Alpha": self.sources["Alpha"]}, {"Alpha": self.routes["Alpha"]}

        with patch.object(runner, "score_low_frequency", side_effect=AssertionError("Task6 called")), \
             patch.object(runner, "classify_stability", side_effect=AssertionError("Task7 called")):
            with self.assertRaisesRegex(ValueError, "catalog bijection"):
                runner.compose_exp449_proposal_score_antigravity(
                    self.raw, self.acceleration, root=Path("/tmp/exp449_fixture_antigravity"), context=object(),
                    catalog=self.catalog, collector=missing_collector,
                )

    def test_collector_receives_the_exact_verified_recovery_inputs(self) -> None:
        """Task4 can reuse only the root, context, catalog, and assessment its caller bound."""
        received: dict[str, object] = {}

        def collector(raw: object, backend: object, **kwargs: object):
            received.update(kwargs)
            return self.sources, self.routes

        root = Path("/tmp/exp449_fixture_antigravity")
        context = object()
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, "_label_firewall"))
            for stage_patch in self._stage_patches():
                stack.enter_context(stage_patch)
            runner.compose_exp449_proposal_score_antigravity(
                self.raw, self.acceleration, root=root, context=context,
                catalog=self.catalog, collector=collector,
            )
        self.assertIs(received["root"], root)
        self.assertIs(received["context"], context)
        self.assertIs(received["catalog"], self.catalog)
        self.assertIs(received["acceleration_manifest"], self.acceleration)

    def test_complete_fixture_preserves_existing_proposal_score_semantics(self) -> None:
        """A complete catalog retains every in-memory ProposalScore semantic field."""
        paper_low = self.low
        paper_stability = self.stability
        paper_policies = self.policies
        timeline = count(100.0)
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, "_label_firewall"))
            stack.enter_context(patch.object(runner, "monotonic", side_effect=lambda: float(next(timeline))))
            for stage_patch in self._stage_patches():
                stack.enter_context(stage_patch)
            exp449 = runner.compose_exp449_proposal_score_antigravity(
                self.raw, self.acceleration, catalog=self.catalog, collector=self._collector,
            )
        timeline = count(100.0)
        with (
            patch.object(proposal_score_paper, "_label_firewall"),
            patch.object(proposal_score_paper, "monotonic", side_effect=lambda: float(next(timeline))),
            patch.object(proposal_score_paper, "_collect_task4_task5", return_value=(self.sources, self.routes)),
            patch.object(proposal_score_paper, "score_low_frequency", return_value=paper_low),
            patch.object(proposal_score_paper, "classify_stability", return_value=paper_stability),
            patch.object(proposal_score_paper, "build_hard_evidence_surface", return_value=object()),
            patch.object(proposal_score_paper, "build_policies", return_value=paper_policies),
        ):
            existing = proposal_score_paper.compose_proposal_score(self.raw, self.acceleration)

        self._assert_proposal_score_semantic_parity(exp449, existing)
        mutated_sources = dict(existing.sources)
        mutated_sources["Alpha"] = HardSourceBundle(("mutated",), {})
        mutated = proposal_score_paper.ProposalScore(
            existing.raw,
            mutated_sources,
            existing.routes,
            existing.low_frequency,
            existing.stability,
            existing.policies,
            existing.backend,
            existing.stage_seconds,
        )
        with self.assertRaises(AssertionError):
            self._assert_proposal_score_semantic_parity(exp449, mutated)


if __name__ == "__main__":
    unittest.main()
