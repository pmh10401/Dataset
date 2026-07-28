from copy import deepcopy
import unittest

from experiments.exp414_missing_aware_near_repeat.benchmark import (
    BENCHMARK_DATASETS,
    require_acceleration_adoption,
)


_COMMIT = "a" * 40
_DATABASE_SHA256 = (
    "8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c"
)
_CONTRACTS = {
    "frozen_config_sha256": (
        "8147013af22e286f5a0bb5ca8ea111f21ebd324453970b14c98e31ab6aaaf4cb"
    ),
    "perturbation_contract_sha256": (
        "6bc750fff507189ec190a04749be2f99f4fb766cba76d70804bf7cc713f1aeff"
    ),
    "acceleration_contract_sha256": (
        "4e5a793684b576b465bfcf2606b58d99a67671a7c4fbc6a348bf4bb9f52e9d79"
    ),
    "missing_repair_contract_sha256": (
        "84b047a9d270e6714545e5853c42f5ab3a7b74f6d6d2ce688bb1110209a8e38d"
    ),
    "qualification_contract_sha256": (
        "e8044e79b3be6737a7ab3ffc8df9ff3cbbce52c9cef80ba6ba5050d36d208746"
    ),
    "execution_contract_sha256": (
        "f7e1ea03a7a160d672d91b0dc062303c04a221848f2b6365717b08dca0d36340"
    ),
}


def _passing_report() -> dict[str, object]:
    common = {
        "numpy_seconds": 4.0,
        "numba_cold_seconds": 2.5,
        "numba_warm_seconds": 1.5,
        "warm_speedup": 2.6666666666666665,
        "max_abs_error": 1e-12,
        "qualification_identity": True,
        "tag_identity": True,
        "selected_quantile_identity": True,
        "repaired_runs": 0,
        "repaired_values": 0,
        "numba_cold_backend_audit_count": 1,
        "numba_warm_backend_audit_count": 1,
        "numba_cold_backend": "numpy_cpu",
        "numba_warm_backend": "numpy_cpu",
        "numba_cold_fallback_reason": None,
        "numba_warm_fallback_reason": None,
    }
    return {
        "schema_version": 1,
        "benchmark_datasets": {
            "small": {"dataset": "ACSF1_normal_0", **common},
            "medium": {
                "dataset": "HandOutlines_normal_0",
                **common,
                "numba_cold_backend": "numba_cpu",
                "numba_warm_backend": "numba_cpu",
            },
            "large": {
                "dataset": "StarLightCurves_normal_3",
                **common,
                "numba_cold_backend": "numba_cpu",
                "numba_warm_backend": "numba_cpu",
            },
            "missing": {
                "dataset": "DodgerLoopDay_normal_1",
                **common,
                "repaired_runs": 1,
                "repaired_values": 71,
            },
        },
        "runtime": {"numba_threads": 12, "nested_threads": 1},
        "database_snapshot": {"db": _DATABASE_SHA256},
        "git_commit": _COMMIT,
        **_CONTRACTS,
    }


class BenchmarkAdoptionTests(unittest.TestCase):
    def test_fixed_train_dataset_contract_is_exact(self) -> None:
        self.assertEqual(
            BENCHMARK_DATASETS,
            {
                "small": "ACSF1_normal_0",
                "medium": "HandOutlines_normal_0",
                "large": "StarLightCurves_normal_3",
                "missing": "DodgerLoopDay_normal_1",
            },
        )

    def test_passing_missing_aware_benchmark_is_adopted(self) -> None:
        require_acceleration_adoption(_passing_report(), _COMMIT)

    def test_medium_or_large_below_two_x_is_rejected(self) -> None:
        for size in ("medium", "large"):
            report = deepcopy(_passing_report())
            report["benchmark_datasets"][size]["numpy_seconds"] = 1.99
            report["benchmark_datasets"][size]["numba_warm_seconds"] = 1.0
            report["benchmark_datasets"][size]["warm_speedup"] = 1.99
            with self.subTest(size=size), self.assertRaisesRegex(
                ValueError, "speedup"
            ):
                require_acceleration_adoption(report, _COMMIT)

    def test_small_and_missing_do_not_have_a_speed_gate(self) -> None:
        report = deepcopy(_passing_report())
        for size in ("small", "missing"):
            report["benchmark_datasets"][size]["numpy_seconds"] = 0.1
            report["benchmark_datasets"][size]["numba_warm_seconds"] = 1.0
            report["benchmark_datasets"][size]["warm_speedup"] = 0.1
        require_acceleration_adoption(report, _COMMIT)

    def test_speedup_must_equal_finite_positive_recorded_timings(self) -> None:
        mutations = (
            ("warm_speedup", 3.0),
            ("numpy_seconds", float("nan")),
            ("numba_cold_seconds", 0.0),
            ("numba_warm_seconds", -1.0),
        )
        for field, value in mutations:
            report = deepcopy(_passing_report())
            report["benchmark_datasets"]["large"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "timing"
            ):
                require_acceleration_adoption(report, _COMMIT)

    def test_every_dataset_requires_numerical_and_decision_parity(self) -> None:
        mutations = (
            ("max_abs_error", 1.000001e-6),
            ("qualification_identity", False),
            ("tag_identity", False),
            ("selected_quantile_identity", False),
        )
        for size in BENCHMARK_DATASETS:
            for field, value in mutations:
                report = deepcopy(_passing_report())
                report["benchmark_datasets"][size][field] = value
                with (
                    self.subTest(size=size, field=field),
                    self.assertRaisesRegex(ValueError, "parity"),
                ):
                    require_acceleration_adoption(report, _COMMIT)

    def test_missing_dataset_requires_exact_repair_counts(self) -> None:
        for field, value in (("repaired_runs", 0), ("repaired_values", 70)):
            report = deepcopy(_passing_report())
            report["benchmark_datasets"]["missing"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "repair"
            ):
                require_acceleration_adoption(report, _COMMIT)

    def test_backend_audit_evidence_is_complete_and_fail_closed(self) -> None:
        mutations = (
            ("numba_cold_backend_audit_count", 0),
            ("numba_warm_backend_audit_count", 2),
            ("numba_cold_fallback_reason", "pair_audit_tolerance_exceeded"),
            ("numba_warm_fallback_reason", "pair_audit_tolerance_exceeded"),
            ("numba_cold_backend", None),
            ("numba_warm_backend", None),
        )
        for field, value in mutations:
            report = deepcopy(_passing_report())
            report["benchmark_datasets"]["small"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "backend audit"
            ):
                require_acceleration_adoption(report, _COMMIT)
        for size in ("medium", "large"):
            report = deepcopy(_passing_report())
            report["benchmark_datasets"][size]["numba_warm_backend"] = (
                "numpy_cpu"
            )
            with self.subTest(size=size), self.assertRaisesRegex(
                ValueError, "Numba backend"
            ):
                require_acceleration_adoption(report, _COMMIT)

    def test_runtime_and_git_binding_are_required(self) -> None:
        mutations = (
            ("numba_threads", 11),
            ("nested_threads", 2),
        )
        for field, value in mutations:
            report = deepcopy(_passing_report())
            report["runtime"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "thread"
            ):
                require_acceleration_adoption(report, _COMMIT)
        with self.assertRaisesRegex(ValueError, "Git"):
            require_acceleration_adoption(_passing_report(), "b" * 40)

    def test_database_and_every_contract_hash_are_required(self) -> None:
        report = deepcopy(_passing_report())
        report["database_snapshot"]["db"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "database"):
            require_acceleration_adoption(report, _COMMIT)
        for field in _CONTRACTS:
            report = deepcopy(_passing_report())
            report[field] = "0" * 64
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "contract"
            ):
                require_acceleration_adoption(report, _COMMIT)


if __name__ == "__main__":
    unittest.main()
