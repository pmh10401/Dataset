import csv
from io import StringIO
import json
import math
import unittest

from experiments.exp411_near_repeat_qualification.contracts import (
    DatasetBackendAudit,
    DatasetQualification,
)
from experiments.exp414_missing_aware_near_repeat.contracts import (
    DatasetRepairAudit,
    RunRepairAudit,
)
from experiments.exp414_missing_aware_near_repeat.reporting import (
    render_payloads,
)


EXPECTED_ARTIFACTS = {
    "exp414_frozen_config.json",
    "exp414_missing_repair_audit.csv",
    "exp414_dataset_qualification.csv",
    "exp414_quantile_summary.csv",
    "exp414_backend_assessment.json",
    "exp414_selected_rule.json",
    "exp414_summary.json",
    "EXP414_RESULT_REPORT_KO.md",
}

REPAIR_FIELDS = (
    "dataset",
    "instance_index",
    "raw_values_sha256",
    "repaired_values_sha256",
    "length",
    "nan_count",
    "posinf_count",
    "neginf_count",
    "missing_positions_sha256",
    "repair_method",
)


def _qualification(dataset: str, near: int) -> DatasetQualification:
    family = dataset.split("_normal_", 1)[0]
    return DatasetQualification(
        dataset=dataset,
        family=family,
        identity_group_count=6,
        normal_total=6,
        normal_abstained=0,
        normal_near={"0.01": near, "0.05": near, "0.10": near},
        synthetic_total={"0.01": 7000, "0.05": 7000, "0.10": 7000},
        synthetic_near={"0.01": 0, "0.05": 0, "0.10": 0},
        perturbation_total={
            key: {"amplitude_scale": 1000}
            for key in ("0.01", "0.05", "0.10")
        },
        perturbation_near={
            key: {"amplitude_scale": 0}
            for key in ("0.01", "0.05", "0.10")
        },
    )


def _repair_audit(
    dataset: str,
    *,
    repaired: bool,
    first_index: int,
) -> DatasetRepairAudit:
    rows = (
        RunRepairAudit(
            dataset=dataset,
            instance_index=first_index + 1,
            raw_values_sha256="3" * 64,
            repaired_values_sha256="4" * 64,
            length=8,
            nan_count=0,
            posinf_count=0,
            neginf_count=0,
            missing_positions_sha256="5" * 64,
            repair_method="unchanged",
        ),
        RunRepairAudit(
            dataset=dataset,
            instance_index=first_index,
            raw_values_sha256="1" * 64,
            repaired_values_sha256="2" * 64,
            length=8,
            nan_count=int(repaired),
            posinf_count=0,
            neginf_count=0,
            missing_positions_sha256="6" * 64,
            repair_method=(
                "linear_interpolation" if repaired else "unchanged"
            ),
        ),
    )
    return DatasetRepairAudit(
        dataset=dataset,
        runs=rows,
        raw_train_identity_sha256="7" * 64,
        repaired_train_identity_sha256="8" * 64,
        status="ready",
    )


def _aggregate(
    *,
    dataset_count: int,
    family_count: int,
    selection_authority: bool,
) -> dict[str, object]:
    return {
        "status": "qualified",
        "selected_quantile": 0.05,
        "quantiles": {
            0.10: {
                "normal_acceptance": 0.02,
                "pooled_synthetic": 0.002,
                "worst_perturbation": 0.006,
                "bootstrap_upper": 0.006,
            },
            0.05: {
                "normal_acceptance": 0.02,
                "pooled_synthetic": 0.0005,
                "worst_perturbation": 0.004,
                "bootstrap_upper": 0.004,
            },
            0.01: {
                "normal_acceptance": 0.01,
                "pooled_synthetic": 0.0,
                "worst_perturbation": 0.0,
                "bootstrap_upper": 0.0,
            },
        },
        "dataset_count": dataset_count,
        "family_count": family_count,
        "selection_authority": selection_authority,
    }


def _fixture() -> tuple[object, ...]:
    alpha = _qualification("Alpha_normal_0", 1)
    beta = _qualification("Beta_normal_0", 2)
    records = (
        {
            "qualification": beta,
            "execution_source": "exp414_computed",
            "repair_status": "ready",
            "backend_audit": DatasetBackendAudit(
                backend="numba_cpu",
                sampled_pairs=((0, 1),),
                max_abs_error=1e-8,
                fallback_reason=None,
            ),
        },
        {
            "qualification": alpha,
            "execution_source": "migrated_exp411_verified",
            "repair_status": "ready",
            "backend_audit": DatasetBackendAudit(
                backend="numba_cpu",
                sampled_pairs=((0, 1),),
                max_abs_error=0.0,
                fallback_reason=None,
            ),
        },
    )
    audits = (
        _repair_audit("Beta_normal_0", repaired=True, first_index=4),
        _repair_audit("Alpha_normal_0", repaired=False, first_index=2),
    )
    aggregate = _aggregate(
        dataset_count=2,
        family_count=2,
        selection_authority=True,
    )
    subgroups = {
        "all": aggregate,
        "nonmissing": _aggregate(
            dataset_count=1,
            family_count=1,
            selection_authority=False,
        ),
        "missing_repaired": _aggregate(
            dataset_count=1,
            family_count=1,
            selection_authority=False,
        ),
        "DodgerLoop": {
            "status": "not_applicable",
            "selected_quantile": None,
            "quantiles": {},
            "dataset_count": 0,
            "family_count": 0,
            "selection_authority": False,
        },
    }
    return (
        records,
        audits,
        aggregate,
        subgroups,
        {
            "canonical_backend": "adaptive_numpy_numba_cpu",
            "dataset_count": 2,
            "distance_tolerance": 1e-6,
            "worst_max_abs_error": 1e-8,
            "backend_counts": {"numba_cpu": 2},
            "fallback_reason_counts": {"none": 2},
        },
        {
            "mode": "fixture",
            "test_values_used": False,
            "test_labels_used": False,
        },
    )


class ReportingTests(unittest.TestCase):
    def test_dataset_repair_status_comes_from_authoritative_audit(
        self,
    ) -> None:
        values = list(_fixture())
        values[0] = tuple(
            {
                key: child
                for key, child in record.items()
                if key != "repair_status"
            }
            for record in values[0]
        )

        payloads = render_payloads(
            *values,
            full_qualification_executed=True,
        )

        rows = tuple(
            csv.DictReader(
                StringIO(
                    payloads[
                        "exp414_dataset_qualification.csv"
                    ].decode("utf-8")
                )
            )
        )
        self.assertEqual(
            [(row["dataset"], row["repair_status"]) for row in rows],
            [
                ("Alpha_normal_0", "ready"),
                ("Beta_normal_0", "ready"),
            ],
        )

    def test_renders_exact_deterministic_artifact_coverage_and_schemas(
        self,
    ) -> None:
        values = _fixture()

        payloads = render_payloads(
            *values,
            full_qualification_executed=True,
        )
        repeated = render_payloads(
            values[0][::-1],
            values[1][::-1],
            *values[2:],
            full_qualification_executed=True,
        )

        self.assertEqual(set(payloads), EXPECTED_ARTIFACTS)
        self.assertEqual(payloads, repeated)

        repair_reader = csv.DictReader(
            StringIO(
                payloads["exp414_missing_repair_audit.csv"].decode("utf-8")
            )
        )
        repair_rows = tuple(repair_reader)
        self.assertEqual(tuple(repair_reader.fieldnames or ()), REPAIR_FIELDS)
        self.assertEqual(len(repair_rows), 4)
        self.assertEqual(
            [
                (row["dataset"], int(row["instance_index"]))
                for row in repair_rows
            ],
            [
                ("Alpha_normal_0", 2),
                ("Alpha_normal_0", 3),
                ("Beta_normal_0", 4),
                ("Beta_normal_0", 5),
            ],
        )

        dataset_reader = csv.DictReader(
            StringIO(
                payloads["exp414_dataset_qualification.csv"].decode("utf-8")
            )
        )
        dataset_rows = tuple(dataset_reader)
        self.assertIn("execution_source", dataset_reader.fieldnames or ())
        self.assertIn("repair_status", dataset_reader.fieldnames or ())
        self.assertEqual(
            [row["dataset"] for row in dataset_rows],
            ["Alpha_normal_0", "Beta_normal_0"],
        )

        quantile_rows = tuple(
            csv.DictReader(
                StringIO(
                    payloads["exp414_quantile_summary.csv"].decode("utf-8")
                )
            )
        )
        diagnostic_rows = [
            row for row in quantile_rows if row["subgroup"] != "all"
        ]
        self.assertTrue(diagnostic_rows)
        self.assertEqual(
            {row["selection_authority"] for row in diagnostic_rows},
            {"False"},
        )

        summary = json.loads(payloads["exp414_summary.json"])
        self.assertFalse(summary["exp412_prediction_created"])
        self.assertFalse(summary["exp413_audit_created"])
        self.assertFalse(summary["test_values_used"])
        self.assertFalse(summary["test_labels_used"])
        for name, row in summary["subgroups"].items():
            if name != "all":
                self.assertFalse(row["selection_authority"])

        backend = json.loads(payloads["exp414_backend_assessment.json"])
        self.assertEqual(
            set(
                key
                for key in backend
                if key
                in {
                    "dataset_count",
                    "distance_tolerance",
                    "worst_max_abs_error",
                    "backend_counts",
                    "fallback_reason_counts",
                }
            ),
            {
                "dataset_count",
                "distance_tolerance",
                "worst_max_abs_error",
                "backend_counts",
                "fallback_reason_counts",
            },
        )
        self.assertEqual(backend["dataset_count"], 2)
        self.assertEqual(backend["backend_counts"], {"numba_cpu": 2})
        self.assertEqual(backend["fallback_reason_counts"], {"none": 2})
        self.assertEqual(backend["worst_max_abs_error"], 1e-8)

        report = payloads["EXP414_RESULT_REPORT_KO.md"].decode("utf-8")
        self.assertIn("TRAIN-only", report)
        self.assertIn("결측", report)
        self.assertIn("운영 정책", report)

    def test_json_outputs_reject_nonfinite_values(self) -> None:
        values = list(_fixture())
        values[4] = {
            **values[4],
            "worst_max_abs_error": math.nan,
        }

        with self.assertRaisesRegex(ValueError, "finite|JSON"):
            render_payloads(
                *values,
                full_qualification_executed=False,
            )

    def test_ready_record_backend_gate_is_rechecked_before_rendering(
        self,
    ) -> None:
        for maximum_error, fallback, message in (
            (1.000001e-6, None, "tolerance"),
            (0.0, "pair_audit_tolerance_exceeded", "fallback"),
        ):
            with self.subTest(message=message):
                values = list(_fixture())
                records = list(values[0])
                records[0] = {
                    **records[0],
                    "backend_audit": DatasetBackendAudit(
                        backend="numba_cpu",
                        sampled_pairs=((0, 1),),
                        max_abs_error=maximum_error,
                        fallback_reason=fallback,
                    ),
                }
                values[0] = tuple(records)

                with self.assertRaisesRegex(ValueError, message):
                    render_payloads(
                        *values,
                        full_qualification_executed=True,
                    )


if __name__ == "__main__":
    unittest.main()
