"""RED/GREEN contract tests for the Exp444 acceleration gate."""
from __future__ import annotations

import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def _run(
    backend: str,
    *,
    seconds: float,
    lanes: tuple[tuple[str, int, str], ...],
    **changes: object,
) -> dict[str, object]:
    versions = {
        "cpu": {"python": "3.14.6", "numpy": "2.4.6"},
        "mlx": {"mlx": "0.32.0", "mlx-metal": "0.32.0"},
        "numba": {"numba": "0.66.0", "numpy": "2.4.6", "python": "3.14.6"},
    }
    primary_version = {"cpu": "3.14.6", "mlx": "0.32.0", "numba": "0.66.0"}
    device = "Apple M2 Max" if backend == "mlx" else "CPU"
    payload: dict[str, object] = {
        "backend": backend,
        "available": True,
        "seconds": seconds,
        "lanes": lanes,
        "max_abs_scalar_diff": 0.0,
        "finite_outputs": True,
        "synchronization_included": True,
        "device_transfer_included": True,
        "compile_warmup_policy_included": True,
        "result_materialization_included": True,
        "masks_match": True,
        "candidate_presence_match": True,
        "survival_counts_match": True,
        "lane_coordinates_match": True,
        "single_process": True,
        "no_process_pool": True,
        "process_count": 1,
        "device": device,
        "version": primary_version[backend],
        "versions": versions[backend],
    }
    payload.update(changes)
    return payload


class BackendContractAntigravityTests(unittest.TestCase):
    def test_backend_requires_identity_and_speed_gate_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.backend_antigravity import (
            choose_backend_antigravity,
        )

        decision = choose_backend_antigravity(
            cpu=_run("cpu", seconds=2.0, lanes=(("A", 1, "stable_review"),)),
            mlx=_run(
                "mlx",
                seconds=2.0 / 1.1,
                lanes=(("A", 1, "stable_review"),),
                max_abs_scalar_diff=1e-6,
            ),
        )
        self.assertEqual(decision.selected_backend, "mlx")

    def test_fast_backend_with_lane_difference_falls_back_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.backend_antigravity import (
            choose_backend_antigravity,
        )

        decision = choose_backend_antigravity(
            cpu=_run("cpu", seconds=2.0, lanes=(("A", 1, "stable_review"),)),
            mlx=_run("mlx", seconds=1.0, lanes=(("A", 1, "unstable_review"),)),
        )
        self.assertEqual(decision.selected_backend, "cpu")
        self.assertIn("lane", decision.fallback_reason or "")

    def test_exact_speed_boundary_is_inclusive_and_below_is_not_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.backend_antigravity import (
            choose_backend_antigravity,
        )

        lanes = (("A", 1, "stable_review"),)
        exact = choose_backend_antigravity(
            cpu=_run("cpu", seconds=1.1, lanes=lanes),
            numba=_run("numba", seconds=1.0, lanes=lanes),
        )
        below = choose_backend_antigravity(
            cpu=_run("cpu", seconds=1.099, lanes=lanes),
            numba=_run("numba", seconds=1.0, lanes=lanes),
        )
        self.assertEqual(exact.selected_backend, "numba")
        self.assertEqual(below.selected_backend, "cpu")

    def test_missing_required_gate_field_falls_back_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.backend_antigravity import (
            choose_backend_antigravity,
        )

        lanes = (("A", 1, "stable_review"),)
        required = (
            "available",
            "finite_outputs",
            "synchronization_included",
            "device_transfer_included",
            "compile_warmup_policy_included",
            "result_materialization_included",
            "masks_match",
            "candidate_presence_match",
            "survival_counts_match",
            "lane_coordinates_match",
            "single_process",
            "no_process_pool",
            "process_count",
            "seconds",
            "max_abs_scalar_diff",
            "lanes",
            "device",
            "version",
            "versions",
        )
        for field_name in required:
            with self.subTest(field_name=field_name):
                backend = _run("mlx", seconds=0.5, lanes=lanes)
                del backend[field_name]
                decision = choose_backend_antigravity(
                    cpu=_run("cpu", seconds=1.0, lanes=lanes),
                    mlx=backend,
                )
                self.assertEqual(decision.selected_backend, "cpu")

    def test_mlx_versions_must_be_complete_valid_and_consistent_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.backend_antigravity import (
            choose_backend_antigravity,
        )

        lanes = (("A", 1, "stable_review"),)
        invalid_versions = (
            None,
            [],
            "0.32.0",
            {},
            {"mlx": "0.32.0"},
            {"mlx-metal": "0.32.0"},
            {"mlx": "", "mlx-metal": "0.32.0"},
            {"mlx": "unknown", "mlx-metal": "0.32.0"},
            {"mlx": "0.32.0", "mlx-metal": "unavailable"},
            {"mlx": 0.32, "mlx-metal": "0.32.0"},
            {"mlx": "0.31.0", "mlx-metal": "0.32.0"},
        )
        for versions in invalid_versions:
            with self.subTest(versions=versions):
                decision = choose_backend_antigravity(
                    cpu=_run("cpu", seconds=1.0, lanes=lanes),
                    mlx=_run("mlx", seconds=0.5, lanes=lanes, versions=versions),
                )
                self.assertEqual(decision.selected_backend, "cpu")
                self.assertIn("provenance", decision.fallback_reason or "")

    def test_false_required_boolean_gate_field_falls_back_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.backend_antigravity import (
            choose_backend_antigravity,
        )

        lanes = (("A", 1, "stable_review"),)
        required_true = (
            "available",
            "finite_outputs",
            "synchronization_included",
            "device_transfer_included",
            "compile_warmup_policy_included",
            "result_materialization_included",
            "masks_match",
            "candidate_presence_match",
            "survival_counts_match",
            "lane_coordinates_match",
            "single_process",
            "no_process_pool",
        )
        for field_name in required_true:
            with self.subTest(field_name=field_name):
                decision = choose_backend_antigravity(
                    cpu=_run("cpu", seconds=1.0, lanes=lanes),
                    mlx=_run("mlx", seconds=0.5, lanes=lanes, **{field_name: False}),
                )
                self.assertEqual(decision.selected_backend, "cpu")

    def test_malformed_timing_scalar_and_process_evidence_falls_back_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.backend_antigravity import (
            choose_backend_antigravity,
        )

        lanes = (("A", 1, "stable_review"),)
        invalid_changes = (
            {"seconds": 0.0},
            {"seconds": float("nan")},
            {"seconds": float("inf")},
            {"seconds": "fast"},
            {"max_abs_scalar_diff": 1.1e-5},
            {"max_abs_scalar_diff": float("nan")},
            {"max_abs_scalar_diff": float("inf")},
            {"max_abs_scalar_diff": "small"},
            {"process_count": 2},
            {"process_count": True},
            {"lanes": (("A", 1, "unstable_review"),)},
        )
        for changes in invalid_changes:
            with self.subTest(changes=changes):
                backend = _run("mlx", seconds=0.5, lanes=lanes)
                backend.update(changes)
                try:
                    decision = choose_backend_antigravity(
                        cpu=_run("cpu", seconds=1.0, lanes=lanes),
                        mlx=backend,
                    )
                except Exception as error:
                    self.fail(f"malformed evidence raised instead of retaining CPU: {error}")
                self.assertEqual(decision.selected_backend, "cpu")

    def test_unavailable_backend_falls_back_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.backend_antigravity import (
            choose_backend_antigravity,
        )

        lanes = (("A", 1, "stable_review"),)
        decision = choose_backend_antigravity(
            cpu=_run("cpu", seconds=1.0, lanes=lanes),
            mlx=_run("mlx", seconds=0.5, lanes=lanes, available=False),
        )
        self.assertEqual(decision.selected_backend, "cpu")

    def test_distance_backends_match_float64_reference_and_enforce_k3_antigravity(self) -> None:
        import numpy as np
        from experiments.exp444_train_only_stability_precision_gate_antigravity.backend_antigravity import (
            score_distance_cpu_antigravity,
            score_distance_numba_antigravity,
        )

        query = np.asarray([0.0, 1.0, 2.0, 3.0], dtype="<f4")
        references = np.asarray(
            [[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [9.0, 9.0, 9.0, 9.0]],
            dtype="<f4",
        )
        expected = score_distance_cpu_antigravity(query, references, 3)
        self.assertTrue(math.isfinite(expected))
        self.assertAlmostEqual(score_distance_numba_antigravity(query, references, 3), expected, delta=1e-12)
        with self.assertRaises(ValueError):
            score_distance_numba_antigravity(query, references, 2)

    def test_mlx_float64_failure_records_exact_runtime_provenance_antigravity(self) -> None:
        import numpy as np
        from experiments.exp444_train_only_stability_precision_gate_antigravity.backend_antigravity import (
            BackendUnavailableAntigravity,
            score_distance_mlx_antigravity,
        )

        query = np.asarray([0.0, 1.0, 2.0], dtype="<f4")
        references = np.asarray(
            [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
            dtype="<f4",
        )
        with self.assertRaises(BackendUnavailableAntigravity) as context:
            score_distance_mlx_antigravity(query, references, 3)
        error = context.exception
        self.assertEqual(str(error), "mlx_float64_unsupported_on_device:ValueError")
        self.assertEqual(error.versions["mlx"], "0.32.0")
        self.assertEqual(error.versions["mlx-metal"], "0.32.0")
        self.assertEqual(error.versions["numpy"], "2.4.6")
        self.assertEqual(error.versions["python"], "3.14.6")
        self.assertIn("Apple M2 Max", error.device)

    def test_run_payload_records_sync_transfer_and_device_provenance_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity.backend_antigravity import (
            BackendRunAntigravity,
        )

        run = BackendRunAntigravity(
            backend="mlx",
            seconds=1.0,
            lanes=(("A", 1, "stable_review"),),
            max_diff=0.0,
            available=True,
            synchronized=True,
            device="Apple GPU",
            version="0.32.0",
            versions={"mlx": "0.32.0", "mlx-metal": "0.32.0"},
        )
        payload = run.to_payload()
        for field_name in (
            "synchronization_included",
            "device_transfer_included",
            "compile_warmup_policy_included",
            "result_materialization_included",
            "single_process",
            "no_process_pool",
        ):
            with self.subTest(field_name=field_name):
                self.assertIs(payload.get(field_name), True)
        self.assertEqual(payload["device"], "Apple GPU")
        self.assertEqual(payload["versions"], {"mlx": "0.32.0", "mlx-metal": "0.32.0"})

    def test_unexpected_mlx_stage_failure_preserves_measured_provenance_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity import backend_antigravity as backend_module

        selection = ({"dataset": "A", "instance_index": 1, "train_group_size": 5},)

        def stage(*args: object, backend: str, **kwargs: object):
            if backend == "mlx":
                raise RuntimeError("injected stage failure")
            version = "3.14.6" if backend == "cpu" else "0.66.0"
            versions = (
                {"python": "3.14.6", "numpy": "2.4.6"}
                if backend == "cpu"
                else {"python": "3.14.6", "numpy": "2.4.6", "numba": "0.66.0"}
            )
            return backend_module.BackendRunAntigravity(
                backend=backend,
                seconds=2.0 if backend == "cpu" else 1.0,
                lanes=(),
                version=version,
                versions=versions,
            ), ({},)

        measured_versions = {
            "python": "3.14.6",
            "numpy": "2.4.6",
            "mlx": "9.8.7",
            "mlx-metal": "6.5.4",
        }
        with (
            patch.object(backend_module, "_selection_inputs_antigravity", return_value=((), selection)),
            patch.object(backend_module, "_run_stage_antigravity", side_effect=stage),
            patch.object(
                backend_module,
                "_load_mlx_antigravity",
                return_value=(SimpleNamespace(__version__="9.8.7"), object(), "Measured Apple GPU", measured_versions),
            ),
            patch.object(
                backend_module,
                "_compare_stage_results_antigravity",
                return_value=(True, True, True, True, True, 0.0),
            ),
        ):
            payload = backend_module.benchmark_backends_antigravity(()).to_payload()
        mlx_run = payload["evidence"][1]["run"]
        self.assertEqual(mlx_run["version"], "9.8.7")
        self.assertEqual(mlx_run["versions"], measured_versions)
        self.assertEqual(mlx_run["device"], "Measured Apple GPU")

    def test_unexpected_mlx_probe_failure_records_unknown_not_frozen_provenance_antigravity(self) -> None:
        from experiments.exp444_train_only_stability_precision_gate_antigravity import backend_antigravity as backend_module

        selection = ({"dataset": "A", "instance_index": 1, "train_group_size": 5},)

        def stage(*args: object, backend: str, **kwargs: object):
            version = "3.14.6" if backend == "cpu" else "0.66.0"
            versions = (
                {"python": "3.14.6", "numpy": "2.4.6"}
                if backend == "cpu"
                else {"python": "3.14.6", "numpy": "2.4.6", "numba": "0.66.0"}
            )
            return backend_module.BackendRunAntigravity(
                backend=backend,
                seconds=2.0 if backend == "cpu" else 1.0,
                lanes=(),
                version=version,
                versions=versions,
            ), ({},)

        with (
            patch.object(backend_module, "_selection_inputs_antigravity", return_value=((), selection)),
            patch.object(backend_module, "_run_stage_antigravity", side_effect=stage),
            patch.object(backend_module, "_load_mlx_antigravity", side_effect=RuntimeError("probe exploded")),
            patch.object(
                backend_module,
                "_compare_stage_results_antigravity",
                return_value=(True, True, True, True, True, 0.0),
            ),
        ):
            payload = backend_module.benchmark_backends_antigravity(()).to_payload()
        mlx_run = payload["evidence"][1]["run"]
        self.assertEqual(mlx_run["version"], "unknown")
        self.assertEqual(mlx_run["versions"]["mlx"], "unknown")
        self.assertEqual(mlx_run["versions"]["mlx-metal"], "unknown")
        self.assertEqual(mlx_run["device"], "unknown")


if __name__ == "__main__":
    unittest.main()
