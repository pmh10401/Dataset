import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp411_near_repeat_qualification.acceleration import (
    full_pairwise_rms,
    minimum_rms,
)
from experiments.exp411_near_repeat_qualification.contracts import (
    DatasetBackendAudit,
    DatasetQualification,
    TrainRun,
)
from experiments.exp411_near_repeat_qualification.perturbations import (
    perturbation_provenance,
    perturbation_types,
)
from experiments.exp411_near_repeat_qualification.qualification import (
    _batched_synthetic_distances,
    aggregate_qualification,
    qualify_dataset,
    select_quantile,
)


class QualificationTests(unittest.TestCase):
    def test_selects_widest_train_only_qualified_quantile(self) -> None:
        rows = {
            0.01: dict(
                normal_acceptance=0.02,
                pooled_synthetic=0.0001,
                worst_perturbation=0.001,
                bootstrap_upper=0.002,
            ),
            0.05: dict(
                normal_acceptance=0.03,
                pooled_synthetic=0.0005,
                worst_perturbation=0.003,
                bootstrap_upper=0.004,
            ),
            0.10: dict(
                normal_acceptance=0.04,
                pooled_synthetic=0.0020,
                worst_perturbation=0.003,
                bootstrap_upper=0.004,
            ),
        }
        self.assertEqual(select_quantile(rows), 0.05)

    def test_no_safe_quantile_fails_closed(self) -> None:
        rows = {
            0.01: dict(
                normal_acceptance=0.005,
                pooled_synthetic=0.0,
                worst_perturbation=0.0,
                bootstrap_upper=0.0,
            ),
        }
        self.assertIsNone(select_quantile(rows))

    def test_exp407_perturbation_names_are_exact(self) -> None:
        self.assertEqual(
            perturbation_types(),
            (
                "spike",
                "drift",
                "level_shift",
                "flat_line",
                "amplitude_scale",
                "frequency_warp",
                "phase_order",
            ),
        )
        provenance = perturbation_provenance(target_length=4, seed=411)
        self.assertEqual(len(provenance["source_sha256"]), 64)
        self.assertEqual(len(provenance["config_sha256"]), 64)
        self.assertEqual(tuple(provenance["perturbation_types"]), perturbation_types())

    def test_dataset_qualification_uses_every_identity_as_normal_query(self) -> None:
        runs = tuple(
            TrainRun(
                dataset="Alpha_normal_0",
                instance_index=index,
                values=(
                    0.0,
                    float(index) * 0.1,
                    float(index * index) * 0.02,
                    float(index) * 0.3 + float(index * index) * 0.01,
                ),
                values_sha256=f"{index:064x}",
            )
            for index in range(6)
        )
        trace: list[dict[str, object]] = []
        result = qualify_dataset(
            dataset="Alpha_normal_0",
            family="Alpha",
            runs=runs,
            parity_trace=trace,
        )
        self.assertEqual(result.normal_total, 6)
        self.assertEqual(result.normal_abstained, 0)
        self.assertEqual(tuple(result.normal_near), ("0.01", "0.05", "0.10"))
        for key in result.synthetic_total:
            self.assertEqual(result.synthetic_total[key], 42)
            self.assertEqual(
                sum(result.perturbation_total[key].values()),
                42,
            )
        normal_trace = [row for row in trace if row["kind"] == "normal"]
        self.assertEqual(len(normal_trace), 18)
        self.assertEqual(
            {row["query_instance_index"] for row in normal_trace},
            set(range(6)),
        )
        self.assertTrue(
            all(len(str(row["query_identity_sha256"])) == 64 for row in trace)
        )

    def test_accelerated_qualification_matches_numpy_counts_and_trace(self) -> None:
        runs = tuple(
            TrainRun(
                dataset="Alpha_normal_0",
                instance_index=index,
                values=(
                    0.0,
                    float(index) * 0.1,
                    float(index * index) * 0.02,
                    float(index) * 0.3 + float(index * index) * 0.01,
                ),
                values_sha256=f"{index:064x}",
            )
            for index in range(6)
        )
        baseline_trace: list[dict[str, object]] = []
        accelerated_trace: list[dict[str, object]] = []
        audits: list[DatasetBackendAudit] = []

        baseline = qualify_dataset(
            dataset="Alpha_normal_0",
            family="Alpha",
            runs=runs,
            backend="numpy_cpu",
            parity_trace=baseline_trace,
        )
        accelerated = qualify_dataset(
            dataset="Alpha_normal_0",
            family="Alpha",
            runs=runs,
            accelerated=True,
            backend_audit=audits,
            parity_trace=accelerated_trace,
        )

        self.assertEqual(accelerated, baseline)
        self.assertEqual(accelerated_trace, baseline_trace)
        self.assertEqual(len(audits), 1)
        self.assertLessEqual(audits[0].max_abs_error, 1e-6)

    def test_five_fold_reuses_one_full_matrix_per_usable_view(self) -> None:
        runs = tuple(
            TrainRun(
                dataset="Alpha_normal_0",
                instance_index=index,
                values=tuple(
                    float(
                        np.sin((index + 1) * (column + 1) * 0.13)
                        + index * column * 0.001
                    )
                    for column in range(8)
                ),
                values_sha256=f"{index:064x}",
            )
            for index in range(20)
        )

        with patch(
            "experiments.exp411_near_repeat_qualification.qualification."
            "full_pairwise_rms",
            wraps=full_pairwise_rms,
        ) as matrix_spy:
            qualify_dataset(
                dataset="Alpha_normal_0",
                family="Alpha",
                runs=runs,
                accelerated=True,
            )

        self.assertEqual(matrix_spy.call_count, 3)

    def test_synthetic_batch_matches_unbatched_query_order(self) -> None:
        fit = np.asarray(
            [[0.0, 0.0], [3.0, 4.0], [8.0, 6.0]],
            dtype=np.float64,
        )
        query = np.asarray(
            [[0.0, 4.0], [6.0, 8.0], [2.0, 2.0], [9.0, 9.0]],
            dtype=np.float64,
        )
        expected = np.concatenate(
            [
                minimum_rms(
                    fit,
                    query[index : index + 1],
                    backend="numpy_cpu",
                )
                for index in range(len(query))
            ]
        )

        observed = _batched_synthetic_distances(
            fit,
            query,
            backend="numba_cpu",
            batch_rows=3,
        )

        np.testing.assert_allclose(
            observed,
            expected,
            rtol=0.0,
            atol=1e-6,
        )

    def test_aggregate_selects_only_train_safe_quantile(self) -> None:
        types = perturbation_types()
        totals = {name: 1000 for name in types}
        row = DatasetQualification(
            dataset="Alpha_normal_0",
            family="Alpha",
            identity_group_count=100,
            normal_total=100,
            normal_abstained=0,
            normal_near={"0.01": 2, "0.05": 3, "0.10": 4},
            synthetic_total={"0.01": 7000, "0.05": 7000, "0.10": 7000},
            synthetic_near={"0.01": 1, "0.05": 3, "0.10": 14},
            perturbation_total={
                "0.01": totals,
                "0.05": totals,
                "0.10": totals,
            },
            perturbation_near={
                "0.01": {name: int(name == "spike") for name in types},
                "0.05": {
                    name: (2 if name == "spike" else int(name == "drift"))
                    for name in types
                },
                "0.10": {name: 2 for name in types},
            },
        )
        summary = aggregate_qualification((row,), bootstrap_iterations=20)
        self.assertEqual(summary["selected_quantile"], 0.05)
        self.assertEqual(summary["status"], "qualified")


if __name__ == "__main__":
    unittest.main()
