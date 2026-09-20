from copy import deepcopy
import unittest

from experiments.exp411_near_repeat_qualification.benchmark import (
    require_acceleration_adoption,
)


def _passing_report() -> dict[str, object]:
    dataset = {
        "numpy_seconds": 4.0,
        "numba_cold_seconds": 2.5,
        "numba_warm_seconds": 1.5,
        "warm_speedup": 2.6666666666666665,
        "max_abs_error": 1e-12,
        "tag_identity": True,
        "selected_quantile_identity": True,
    }
    return {
        "schema_version": 1,
        "benchmark_datasets": {
            "small": {"dataset": "ACSF1_normal_0", **dataset},
            "medium": {"dataset": "HandOutlines_normal_0", **dataset},
            "large": {"dataset": "StarLightCurves_normal_3", **dataset},
        },
        "runtime": {
            "numba_threads": 12,
            "nested_threads": 1,
        },
    }


class BenchmarkAdoptionTests(unittest.TestCase):
    def test_passing_fixed_train_benchmark_is_adopted(self) -> None:
        require_acceleration_adoption(_passing_report())

    def test_medium_or_large_below_two_x_is_rejected(self) -> None:
        for size in ("medium", "large"):
            report = deepcopy(_passing_report())
            report["benchmark_datasets"][size]["warm_speedup"] = 1.99
            with self.subTest(size=size), self.assertRaisesRegex(
                ValueError,
                "speedup",
            ):
                require_acceleration_adoption(report)

    def test_distance_tag_or_quantile_drift_is_rejected(self) -> None:
        mutations = (
            ("max_abs_error", 1.1e-6),
            ("tag_identity", False),
            ("selected_quantile_identity", False),
        )
        for field, value in mutations:
            report = deepcopy(_passing_report())
            report["benchmark_datasets"]["large"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError,
                "parity",
            ):
                require_acceleration_adoption(report)

    def test_wrong_thread_environment_is_rejected(self) -> None:
        for field, value in (
            ("numba_threads", 11),
            ("nested_threads", 2),
        ):
            report = deepcopy(_passing_report())
            report["runtime"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError,
                "thread",
            ):
                require_acceleration_adoption(report)

    def test_missing_or_substituted_fixed_dataset_is_rejected(self) -> None:
        for mutation in ("missing", "substituted"):
            report = deepcopy(_passing_report())
            if mutation == "missing":
                del report["benchmark_datasets"]["large"]
            else:
                report["benchmark_datasets"]["large"]["dataset"] = (
                    "StarLightCurves_normal_2"
                )
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                ValueError,
                "dataset",
            ):
                require_acceleration_adoption(report)


if __name__ == "__main__":
    unittest.main()
