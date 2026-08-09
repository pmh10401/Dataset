"""Behavior and opt-in real tests for the Exp443 Policy A control gate."""
from __future__ import annotations

from types import MappingProxyType, SimpleNamespace
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.exp443_missing_normal_train_policy_comparison_antigravity.contracts_antigravity import (
    KernelSeedScoresAntigravity,
    PolicyDatasetResultAntigravity,
)


POLICY_A = "linear_edge_nearest_control"
SEEDS = (20260721, 20261721)


def _fixture_snapshot() -> SimpleNamespace:
    datasets = ("A", "B")
    arrays = {
        name: np.zeros((2, 1000 if "train" in name else 6), dtype="<f8")
        for name in (
            "kernels_1024__train_by_seed",
            "kernels_1024__test_by_seed",
            "kernels_4096__train_by_seed",
            "kernels_4096__test_by_seed",
        )
    }
    inventory = {
        dataset: {
            "arrays": MappingProxyType({
                name: _readonly_copy(value) for name, value in arrays.items()
            })
        }
        for dataset in datasets
    }
    return SimpleNamespace(
        datasets=datasets,
        affected_datasets=("A",),
        cache_inventory=inventory,
        exp249_candidate_maps={"A": {1024: (), 4096: ()}, "B": {1024: (), 4096: ()}},
        exp236_lanes={
            "A": {"hard": (0,), "standard": (1,), "priority": (), "promotion": ()},
            "B": {"hard": (), "standard": (0,), "priority": (), "promotion": ()},
        },
        hard_coordinates_exp291=frozenset({("A", 0)}),
        hard_coordinates_exp404=frozenset({("A", 0)}),
        test_row_count_by_dataset={"A": 6, "B": 6},
        label_access_count=0,
    )


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    copied = np.array(value, copy=True)
    copied.setflags(write=False)
    return copied


def _result(*, delta: float = 0.0, shape_mismatch: bool = False, nonfinite: float | None = None, candidate_mismatch: bool = False) -> PolicyDatasetResultAntigravity:
    scores: list[KernelSeedScoresAntigravity] = []
    for kernel in (1024, 4096):
        for seed_index, seed in enumerate(SEEDS):
            oof = np.zeros(1000, dtype="<f8")
            test = np.zeros(7 if shape_mismatch and kernel == 1024 and seed_index == 0 else 6, dtype="<f8")
            if delta and kernel == 1024 and seed_index == 0:
                oof[0] = delta
            scores.append(KernelSeedScoresAntigravity(kernel, seed, oof, test, ()))
    candidates = {1024: (0,), 4096: ()} if candidate_mismatch else {1024: (), 4096: ()}
    result = PolicyDatasetResultAntigravity(
        POLICY_A, "A", 8, 1000, 0, 1000, 0, 6, tuple(scores), candidates, (), "scored"
    )
    if nonfinite is not None:
        forged = np.array(result.kernel_scores[0].oof_scores, copy=True)
        forged[0] = nonfinite
        object.__setattr__(result.kernel_scores[0], "oof_scores", forged)
    return result


class ControlGateAntigravityTests(unittest.TestCase):
    def _compare(self, **kwargs):
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.control_antigravity import (
            compare_policy_a_to_archive_antigravity,
        )

        return compare_policy_a_to_archive_antigravity(_fixture_snapshot(), {"A": _result(**kwargs)})

    def test_exact_equality_and_score_evidence_pass(self) -> None:
        result = self._compare()
        self.assertTrue(result.passed)
        self.assertEqual(result.status, "control_reproduction_passed")
        self.assertEqual(result.score_evidence[0]["train_shape"], (1000,))
        self.assertEqual(result.score_evidence[0]["test_shape"], (6,))
        self.assertEqual(result.score_evidence[0]["max_abs_difference"], 0.0)
        self.assertTrue(result.score_evidence[0]["candidate_equal"])

    def test_exactly_one_e_minus_six_passes(self) -> None:
        result = self._compare(delta=1e-6)
        self.assertTrue(result.passed)
        self.assertLessEqual(result.score_evidence[0]["max_abs_difference"], 1e-6)

    def test_greater_than_one_e_minus_six_fails(self) -> None:
        result = self._compare(delta=1.000001e-6)
        self.assertFalse(result.passed)
        self.assertEqual(result.status, "control_reproduction_failed")

    def test_shape_mismatch_fails(self) -> None:
        result = self._compare(shape_mismatch=True)
        self.assertFalse(result.passed)
        self.assertIn("shape", result.reason or "")

    def test_nan_and_inf_fail(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            result = self._compare(nonfinite=value)
            self.assertFalse(result.passed)
            self.assertIn("finite", result.reason or "")

    def test_candidate_mismatch_fails_even_when_scores_match(self) -> None:
        result = self._compare(candidate_mismatch=True)
        self.assertFalse(result.passed)
        self.assertIn("candidate", result.reason or "")

    def test_require_control_gate_blocks_failed_result(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.control_antigravity import (
            require_control_gate_antigravity,
        )

        require_control_gate_antigravity(self._compare())
        with self.assertRaisesRegex(ValueError, "control_reproduction_failed"):
            require_control_gate_antigravity(self._compare(delta=1.000001e-6))

    @unittest.skipUnless(os.environ.get("EXP443_REAL_CONTROL_ANTIGRAVITY") == "1", "opt-in real pinned Policy A control")
    def test_real_19_dataset_policy_a_reproduces_exp249_archive_and_routing(self) -> None:
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.config_antigravity import FROZEN_CONFIG
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.control_antigravity import (
            compare_policy_a_to_archive_antigravity,
            require_control_gate_antigravity,
        )
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.database_antigravity import (
            load_raw_dataset_antigravity,
            pin_database_snapshot_antigravity,
        )
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.minirocket_antigravity import score_prepared_dataset_antigravity
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.preprocessing_antigravity import (
            POLICY_A_ANTIGRAVITY,
            prepare_dataset_antigravity,
        )
        from experiments.exp443_missing_normal_train_policy_comparison_antigravity.sources_antigravity import verify_frozen_sources_antigravity

        for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
        ):
            os.environ[name] = "1"
        snapshot = verify_frozen_sources_antigravity()
        results: dict[str, PolicyDatasetResultAntigravity] = {}
        with tempfile.TemporaryDirectory(prefix="exp443_task6_control_antigravity_") as temporary:
            pinned = pin_database_snapshot_antigravity(
                Path(FROZEN_CONFIG.database_path), Path(temporary), expected_sha256=FROZEN_CONFIG.database_sha256
            )
            try:
                for done, dataset in enumerate(snapshot.affected_datasets, 1):
                    print(f"[Exp443 control] dataset {done}/19 {dataset}", flush=True)
                    raw = load_raw_dataset_antigravity(pinned, dataset)
                    prepared = prepare_dataset_antigravity(raw, POLICY_A_ANTIGRAVITY)
                    results[dataset] = score_prepared_dataset_antigravity(prepared)
                self.assertEqual(pinned.label_access_count, 0)
            finally:
                pinned.close(terminal_evidence_durable=True)
        self.assertEqual(snapshot.label_access_count, 0)
        self.assertEqual(len(results), 19)
        result = compare_policy_a_to_archive_antigravity(snapshot, results)
        require_control_gate_antigravity(result)
        self.assertEqual(result.routing_identity["rows"], 1117)
        self.assertEqual(result.routing_identity["promotions"], 137)
        self.assertEqual(result.routing_identity["pre_exact"], 2253)
        self.assertEqual(result.routing_identity["exact_normal_removals"], 35)
        self.assertEqual(result.routing_identity["final_hard"], 2218)
        self.assertTrue(result.routing_identity["exp291_coordinate_equal"])
        self.assertTrue(result.routing_identity["exp404_coordinate_equal"])
        self.assertEqual(result.routing_identity["recomputed_rows"], 19)
        self.assertEqual(result.routing_identity["imported_rows"], 1098)
        self.assertEqual(result.environment_identity["inner_threads"], 1)
        self.assertEqual(result.environment_identity["acceleration_decision"], "inapplicable_official_cpu_transformer")


if __name__ == "__main__":
    unittest.main()
