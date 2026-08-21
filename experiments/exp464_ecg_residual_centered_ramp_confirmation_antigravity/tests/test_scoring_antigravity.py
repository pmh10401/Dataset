from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np


METHOD_ORDER_ANTIGRAVITY = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")
CHANGE_ORDER_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)
CONFIRMATION_ORDER_ANTIGRAVITY = ("normal",) + CHANGE_ORDER_ANTIGRAVITY


def centered_basis_antigravity() -> np.ndarray:
    t = np.linspace(0.0, 1.0, 84, dtype=np.float64)
    basis = t - float(np.mean(t))
    basis /= float(np.linalg.norm(basis))
    return basis


def raw_basis_antigravity() -> np.ndarray:
    t = np.linspace(0.0, 1.0, 84, dtype=np.float64)
    basis = t / float(np.linalg.norm(t))
    return basis


def orthogonal_unit_antigravity() -> np.ndarray:
    w = np.sin(np.linspace(0.0, 3.0 * np.pi, 84, dtype=np.float64))
    w = w - float(np.mean(w))
    q = centered_basis_antigravity()
    w = w - float(np.dot(w, q)) * q
    w /= float(np.linalg.norm(w))
    return w


def build_row_antigravity(prefix: np.ndarray, suffix: np.ndarray) -> np.ndarray:
    return np.concatenate((prefix, suffix)).astype(np.float64)


def primary_rows_antigravity(values: np.ndarray) -> dict[str, np.ndarray]:
    row = np.asarray(values, dtype=np.float64)
    center = float(np.median(row))
    robust_scale = float(1.4826 * np.median(np.abs(row - center)))
    if not np.isfinite(robust_scale) or robust_scale <= 0.0:
        raise ValueError("row robust scale must be positive and finite")
    onset = 56
    duration = row.size - onset

    level = np.array(row, copy=True)
    level[onset:] += 3.0 * robust_scale

    drift = np.array(row, copy=True)
    drift[onset:] += np.linspace(0.0, 3.0 * robust_scale, duration, dtype=np.float64)

    amplitude = center + 1.5 * (row - center)

    warped = np.array(row, copy=True)
    region = row[onset:]
    source = np.arange(duration, dtype=np.float64)
    warped[onset:] = np.interp(source / 1.25, source, region)

    return {
        "level_shift": level,
        "gradual_drift": drift,
        "amplitude_scaling": amplitude,
        "period_change": warped,
    }


def centered_evidence_antigravity(
    values: np.ndarray,
    template: np.ndarray,
) -> tuple[float, float]:
    residual = np.asarray(values, dtype=np.float64) - np.asarray(template, dtype=np.float64)
    suffix = residual[56:]
    centered = suffix - float(np.mean(suffix))
    amplitude = float(np.dot(centered, centered_basis_antigravity()))
    norm = float(np.linalg.norm(centered))
    cosine = 0.0 if norm == 0.0 else amplitude / norm
    return amplitude, cosine


def raw_centered_evidence_antigravity(values: np.ndarray) -> tuple[float, float]:
    suffix = np.asarray(values, dtype=np.float64)[56:]
    centered = suffix - float(np.mean(suffix))
    amplitude = float(np.dot(centered, centered_basis_antigravity()))
    norm = float(np.linalg.norm(centered))
    cosine = 0.0 if norm == 0.0 else amplitude / norm
    return amplitude, cosine


def raw_uncentered_evidence_antigravity(
    values: np.ndarray,
    template: np.ndarray,
) -> tuple[float, float]:
    residual_suffix = np.asarray(values, dtype=np.float64)[56:] - np.asarray(
        template,
        dtype=np.float64,
    )[56:]
    amplitude = float(np.dot(residual_suffix, raw_basis_antigravity()))
    norm = float(np.linalg.norm(residual_suffix))
    cosine = 0.0 if norm == 0.0 else amplitude / norm
    return amplitude, cosine


def anomaly_rank_antigravity(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    reference_values = np.asarray(reference, dtype=np.float64)
    query_values = np.asarray(query, dtype=np.float64)
    return (1.0 + np.count_nonzero(reference_values[:, None] <= query_values[None, :], axis=0)) / (
        len(reference_values) + 1.0
    )


def confirmation_scores_antigravity(
    baseline: np.ndarray,
    amplitudes_fit: np.ndarray,
    amplitudes_query: np.ndarray,
    cosines_query: np.ndarray,
    threshold: float,
) -> np.ndarray:
    ranks = anomaly_rank_antigravity(amplitudes_fit, amplitudes_query)
    confirmed = cosines_query >= threshold
    return np.where(confirmed, np.maximum(np.asarray(baseline, dtype=np.float64), ranks), baseline)


def fake_exp463_scores_antigravity(query_len: int) -> SimpleNamespace:
    c0 = np.asarray([0.20, 0.80, 0.40], dtype=np.float64)[:query_len]
    c1 = np.asarray([0.31, 0.32, 0.33], dtype=np.float64)[:query_len]
    c2 = np.asarray([0.41, 0.42, 0.43], dtype=np.float64)[:query_len]
    c7 = np.asarray([0.51, 0.52, 0.53], dtype=np.float64)[:query_len]
    primary_offsets = {
        "level_shift": 0.01,
        "gradual_drift": 0.02,
        "amplitude_scaling": 0.03,
        "period_change": 0.04,
    }
    return SimpleNamespace(
        methods=("M0", "M1", "M2", "M3", "M4", "M5", "M6"),
        original={
            "M0": c0,
            "M1": np.zeros(query_len, dtype=np.float64),
            "M2": c1,
            "M3": np.zeros(query_len, dtype=np.float64),
            "M4": c2,
            "M5": np.zeros(query_len, dtype=np.float64),
            "M6": c7,
        },
        primary={
            method: {
                name: scores + offset
                for name, offset in primary_offsets.items()
            }
            for method, scores in {
                "M0": c0,
                "M1": np.zeros(query_len, dtype=np.float64),
                "M2": c1,
                "M3": np.zeros(query_len, dtype=np.float64),
                "M4": c2,
                "M5": np.zeros(query_len, dtype=np.float64),
                "M6": c7,
            }.items()
        },
        feature_dimensions={
            "M0": 2,
            "M1": 1,
            "M2": 2,
            "M3": 2,
            "M4": 2,
            "M5": 1,
            "M6": 24,
        },
        provenance={
            "M0": "exp463_m0",
            "M1": "exp463_m1",
            "M2": "exp463_m2",
            "M3": "exp463_m3",
            "M4": "exp463_m4",
            "M5": "exp463_m5",
            "M6": "exp463_m6",
        },
    )


class ScoringAntigravityTest(unittest.TestCase):
    def _train_fixture_antigravity(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        prefix = 0.05 * np.sin(np.linspace(0.0, np.pi, 56, dtype=np.float64))
        template_suffix = (
            0.4 * np.sin(np.linspace(0.0, 2.0 * np.pi, 84, dtype=np.float64))
            + 0.3 * np.linspace(0.0, 1.0, 84, dtype=np.float64)
            + 0.2
        )
        template = build_row_antigravity(prefix, template_suffix)
        q = centered_basis_antigravity()
        w = orthogonal_unit_antigravity()
        fit_residuals = (
            (-2.0, 0.10),
            (-1.0, -0.20),
            (0.0, 0.15),
            (1.0, -0.10),
            (2.0, 0.05),
            (3.0, -0.15),
        )
        fit_rows = [
            build_row_antigravity(prefix, template_suffix + alpha * q + beta * w)
            for alpha, beta in fit_residuals
        ]
        cosine_threshold = 0.95
        below_threshold = np.nextafter(
            np.nextafter(np.nextafter(cosine_threshold, 0.0), 0.0),
            0.0,
        )
        query_suffixes = (
            template_suffix + 3.5 * q,
            template_suffix + 2.5 * (cosine_threshold * q + np.sqrt(1.0 - cosine_threshold**2) * w),
            template_suffix + 2.0 * (below_threshold * q + np.sqrt(1.0 - below_threshold**2) * w),
        )
        query_rows = [build_row_antigravity(prefix, suffix) for suffix in query_suffixes]
        train = np.vstack((*fit_rows, *query_rows)).astype(np.float64)
        fit_indices = np.arange(len(fit_rows), dtype=np.int64)
        query_indices = np.arange(len(fit_rows), len(fit_rows) + len(query_rows), dtype=np.int64)
        return train, fit_indices, query_indices, template

    def test_scoring_preserves_method_order_and_exact_exp463_controls_antigravity(self) -> None:
        """Break caught: Exp463 control replay or method order drift."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
            capture_exp463_authority_antigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.scoring_antigravity import (
            _confirmed_scores_antigravity,
            _score_fold_methods_with_dependency_antigravity,
        )

        train, fit_indices, query_indices, _ = self._train_fixture_antigravity()
        authority = capture_exp463_authority_antigravity()
        original_train = np.array(train, copy=True)
        original_fit = np.array(fit_indices, copy=True)
        original_query = np.array(query_indices, copy=True)
        calls: list[tuple[tuple[int, ...], tuple[int, ...], int, int]] = []

        def exp463_fold_scorer(train_arg, fit_arg, query_arg, *, seed, fold_index):
            del train_arg
            calls.append((tuple(fit_arg), tuple(query_arg), seed, fold_index))
            return fake_exp463_scores_antigravity(len(query_arg))

        result = _score_fold_methods_with_dependency_antigravity(
            train,
            fit_indices,
            query_indices,
            seed=20260721,
            fold_index=3,
            authority=authority,
            exp463_fold_scorer=exp463_fold_scorer,
        )

        self.assertEqual(calls, [((0, 1, 2, 3, 4, 5), (6, 7, 8), 20260721, 3)])
        self.assertEqual(result.methods, METHOD_ORDER_ANTIGRAVITY)
        self.assertEqual(tuple(result.original), METHOD_ORDER_ANTIGRAVITY)
        self.assertEqual(tuple(result.primary), METHOD_ORDER_ANTIGRAVITY)
        self.assertEqual(tuple(result.fit_amplitudes), METHOD_ORDER_ANTIGRAVITY)
        self.assertEqual(tuple(result.query_amplitudes), METHOD_ORDER_ANTIGRAVITY)
        self.assertEqual(tuple(result.query_cosines), METHOD_ORDER_ANTIGRAVITY)
        self.assertEqual(tuple(result.provenance), METHOD_ORDER_ANTIGRAVITY)
        np.testing.assert_array_equal(result.original["C0"], np.asarray([0.20, 0.80, 0.40], dtype=np.float64))
        np.testing.assert_array_equal(result.original["C1"], np.asarray([0.31, 0.32, 0.33], dtype=np.float64))
        np.testing.assert_array_equal(result.original["C2"], np.asarray([0.41, 0.42, 0.43], dtype=np.float64))
        np.testing.assert_array_equal(result.original["C7"], np.asarray([0.51, 0.52, 0.53], dtype=np.float64))
        self.assertEqual(tuple(result.query_amplitudes["C0"]), ())
        self.assertEqual(tuple(result.query_cosines["C0"]), ())
        self.assertEqual(tuple(result.query_amplitudes["C1"]), ())
        self.assertEqual(tuple(result.query_cosines["C1"]), ())
        self.assertEqual(tuple(result.query_amplitudes["C2"]), ())
        self.assertEqual(tuple(result.query_cosines["C2"]), ())
        self.assertEqual(tuple(result.query_amplitudes["C7"]), ())
        self.assertEqual(tuple(result.query_cosines["C7"]), ())
        for method, base in (
            ("C0", np.asarray([0.20, 0.80, 0.40], dtype=np.float64)),
            ("C1", np.asarray([0.31, 0.32, 0.33], dtype=np.float64)),
            ("C2", np.asarray([0.41, 0.42, 0.43], dtype=np.float64)),
            ("C7", np.asarray([0.51, 0.52, 0.53], dtype=np.float64)),
        ):
            for change_type, offset in (
                ("level_shift", 0.01),
                ("gradual_drift", 0.02),
                ("amplitude_scaling", 0.03),
                ("period_change", 0.04),
            ):
                np.testing.assert_array_equal(result.primary[method][change_type], base + offset)
                self.assertFalse(result.primary[method][change_type].flags.writeable)
                self.assertTrue(result.primary[method][change_type].flags.owndata)
        expected_provenance = {
            "C0": "exp463_m0",
            "C1": "exp463_m2",
            "C2": "exp463_m4",
            "C7": "exp463_m6",
        }
        for method, expected in expected_provenance.items():
            self.assertEqual(result.provenance[method], expected)
        self.assertEqual(
            tuple(result.query_amplitudes["C6"]),
            CONFIRMATION_ORDER_ANTIGRAVITY,
        )
        self.assertEqual(
            tuple(result.query_cosines["C6"]),
            CONFIRMATION_ORDER_ANTIGRAVITY,
        )
        for method in ("C3", "C4", "C5", "C6", "C8"):
            self.assertEqual(tuple(result.query_amplitudes[method]), CONFIRMATION_ORDER_ANTIGRAVITY)
            self.assertEqual(tuple(result.query_cosines[method]), CONFIRMATION_ORDER_ANTIGRAVITY)
            self.assertFalse(result.fit_amplitudes[method].flags.writeable)
            self.assertTrue(result.fit_amplitudes[method].flags.owndata)
            for change_type in CONFIRMATION_ORDER_ANTIGRAVITY:
                self.assertFalse(result.query_amplitudes[method][change_type].flags.writeable)
                self.assertTrue(result.query_amplitudes[method][change_type].flags.owndata)
                self.assertFalse(result.query_cosines[method][change_type].flags.writeable)
                self.assertTrue(result.query_cosines[method][change_type].flags.owndata)
        np.testing.assert_array_equal(train, original_train)
        np.testing.assert_array_equal(fit_indices, original_fit)
        np.testing.assert_array_equal(query_indices, original_query)

    def test_scoring_matches_independent_c3_to_c8_semantics_and_threshold_boundary_antigravity(
        self,
    ) -> None:
        """Break caught: rank, centering, confirmation, or threshold semantics drift."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
            capture_exp463_authority_antigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.scoring_antigravity import (
            _confirmed_scores_antigravity,
            _score_fold_methods_with_dependency_antigravity,
        )

        train, fit_indices, query_indices, _ = self._train_fixture_antigravity()
        authority = capture_exp463_authority_antigravity()
        baseline = np.asarray([0.20, 0.80, 0.40], dtype=np.float64)

        result = _score_fold_methods_with_dependency_antigravity(
            train,
            fit_indices,
            query_indices,
            seed=20261721,
            fold_index=4,
            authority=authority,
            exp463_fold_scorer=lambda *args, **kwargs: fake_exp463_scores_antigravity(3),
        )

        fit_rows = np.asarray(train[fit_indices], dtype=np.float64)
        query_rows = np.asarray(train[query_indices], dtype=np.float64)
        template = np.median(fit_rows, axis=0)
        threshold = 0.95

        fit_centered = np.asarray(
            [centered_evidence_antigravity(row, template)[0] for row in fit_rows],
            dtype=np.float64,
        )
        query_centered_amp = np.asarray(
            [centered_evidence_antigravity(row, template)[0] for row in query_rows],
            dtype=np.float64,
        )
        query_centered_cos = np.asarray(
            [centered_evidence_antigravity(row, template)[1] for row in query_rows],
            dtype=np.float64,
        )
        query_centered_rank = anomaly_rank_antigravity(fit_centered, query_centered_amp)
        np.testing.assert_array_equal(
            result.original["C3"],
            query_centered_rank,
        )
        np.testing.assert_array_equal(
            result.original["C4"],
            np.maximum(baseline, query_centered_rank),
        )
        np.testing.assert_array_equal(
            result.original["C6"],
            confirmation_scores_antigravity(
                baseline,
                fit_centered,
                query_centered_amp,
                query_centered_cos,
                threshold,
            ),
        )
        direct_threshold = np.asarray([0.10, 0.20], dtype=np.float64)
        direct_fit = np.asarray([0.0, 0.5], dtype=np.float64)
        direct_query = np.asarray([0.6, 0.6], dtype=np.float64)
        direct_cosine = np.asarray(
            [threshold, np.nextafter(threshold, 0.0)],
            dtype=np.float64,
        )
        direct_scores, direct_count = _confirmed_scores_antigravity(
            direct_threshold,
            direct_fit,
            direct_query,
            direct_cosine,
        )
        np.testing.assert_array_equal(
            direct_scores,
            np.asarray([1.0, 0.20], dtype=np.float64),
        )
        self.assertEqual(direct_count, 1)

        fit_raw_centered = np.asarray(
            [raw_centered_evidence_antigravity(row)[0] for row in fit_rows],
            dtype=np.float64,
        )
        query_raw_centered_amp = np.asarray(
            [raw_centered_evidence_antigravity(row)[0] for row in query_rows],
            dtype=np.float64,
        )
        query_raw_centered_cos = np.asarray(
            [raw_centered_evidence_antigravity(row)[1] for row in query_rows],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(
            result.original["C5"],
            confirmation_scores_antigravity(
                baseline,
                fit_raw_centered,
                query_raw_centered_amp,
                query_raw_centered_cos,
                threshold,
            ),
        )

        fit_raw_uncentered = np.asarray(
            [raw_uncentered_evidence_antigravity(row, template)[0] for row in fit_rows],
            dtype=np.float64,
        )
        query_raw_uncentered_amp = np.asarray(
            [raw_uncentered_evidence_antigravity(row, template)[0] for row in query_rows],
            dtype=np.float64,
        )
        query_raw_uncentered_cos = np.asarray(
            [raw_uncentered_evidence_antigravity(row, template)[1] for row in query_rows],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(
            result.original["C8"],
            confirmation_scores_antigravity(
                baseline,
                fit_raw_uncentered,
                query_raw_uncentered_amp,
                query_raw_uncentered_cos,
                threshold,
            ),
        )

        query_primary = [primary_rows_antigravity(row) for row in query_rows]
        expected_c6_counts = {
            "normal": int(np.count_nonzero(query_centered_cos >= threshold)),
        }
        for method in ("C3", "C4", "C5", "C6", "C8"):
            for change_type in CHANGE_ORDER_ANTIGRAVITY:
                self.assertFalse(result.primary[method][change_type].flags.writeable)
                self.assertTrue(result.primary[method][change_type].flags.owndata)
                self.assertTrue(np.isfinite(result.primary[method][change_type]).all())

        for change_type in CHANGE_ORDER_ANTIGRAVITY:
            centered_primary_amp = np.asarray(
                [
                    centered_evidence_antigravity(rows[change_type], template)[0]
                    for rows in query_primary
                ],
                dtype=np.float64,
            )
            centered_primary_cos = np.asarray(
                [
                    centered_evidence_antigravity(rows[change_type], template)[1]
                    for rows in query_primary
                ],
                dtype=np.float64,
            )
            np.testing.assert_array_equal(
                result.primary["C3"][change_type],
                anomaly_rank_antigravity(fit_centered, centered_primary_amp),
            )
            np.testing.assert_array_equal(
                result.primary["C4"][change_type],
                np.maximum(
                    fake_exp463_scores_antigravity(3).primary["M0"][change_type],
                    anomaly_rank_antigravity(fit_centered, centered_primary_amp),
                ),
            )
            np.testing.assert_array_equal(
                result.primary["C6"][change_type],
                confirmation_scores_antigravity(
                    fake_exp463_scores_antigravity(3).primary["M0"][change_type],
                    fit_centered,
                    centered_primary_amp,
                    centered_primary_cos,
                    threshold,
                ),
            )
            expected_c6_counts[change_type] = int(
                np.count_nonzero(centered_primary_cos >= threshold)
            )

        zero_counts = {name: 0 for name in CONFIRMATION_ORDER_ANTIGRAVITY}
        for method in ("C0", "C1", "C2", "C3", "C4", "C7"):
            self.assertEqual(dict(result.confirmation_counts[method]), zero_counts)
        self.assertEqual(dict(result.confirmation_counts["C6"]), expected_c6_counts)
        self.assertEqual(
            tuple(result.confirmation_counts["C6"]),
            CONFIRMATION_ORDER_ANTIGRAVITY,
        )

    def test_scoring_result_is_immutable_and_rejects_forged_authority_antigravity(self) -> None:
        """Break caught: mutable score evidence or unvalidated predecessor authority."""
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.contracts_antigravity import (
            Exp463AuthorityAntigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.predecessor_antigravity import (
            capture_exp463_authority_antigravity,
        )
        from experiments.exp464_ecg_residual_centered_ramp_confirmation_antigravity.scoring_antigravity import (
            _score_fold_methods_with_dependency_antigravity,
        )

        train, fit_indices, query_indices, _ = self._train_fixture_antigravity()
        authority = capture_exp463_authority_antigravity()
        result = _score_fold_methods_with_dependency_antigravity(
            train,
            fit_indices,
            query_indices,
            seed=20260721,
            fold_index=0,
            authority=authority,
            exp463_fold_scorer=lambda *args, **kwargs: fake_exp463_scores_antigravity(3),
        )

        for method in METHOD_ORDER_ANTIGRAVITY:
            self.assertFalse(result.original[method].flags.writeable)
            self.assertTrue(result.original[method].flags.owndata)
        self.assertEqual(
            result.provenance["C6"],
            "exp464_template_residual_centered_ramp_confirmed_max",
        )
        self.assertEqual(
            result.provenance["C8"],
            "exp464_template_residual_uncentered_ramp_confirmed_max",
        )

        forged = Exp463AuthorityAntigravity(
            predecessor_experiment=authority.predecessor_experiment,
            terminal_closeout_commit=authority.terminal_closeout_commit,
            task5_report_sha256=authority.task5_report_sha256,
            status="forged",
            reviewed_cli_commit=authority.reviewed_cli_commit,
            private_run_count=authority.private_run_count,
            result_row_count=authority.result_row_count,
            result_failed_row_count=authority.result_failed_row_count,
            result_summary_count=authority.result_summary_count,
            result_leaf_count=authority.result_leaf_count,
            method_fold_metrics_sha256=authority.method_fold_metrics_sha256,
            artifact_map_sha256=authority.artifact_map_sha256,
            official_evaluate_count=authority.official_evaluate_count,
            official_inspect_count=authority.official_inspect_count,
            label_query_count=authority.label_query_count,
            control_summaries=authority.control_summaries,
            control_row_digests=authority.control_row_digests,
        )
        with self.assertRaises(ValueError):
            _score_fold_methods_with_dependency_antigravity(
                train,
                fit_indices,
                query_indices,
                seed=20260721,
                fold_index=0,
                authority=forged,
                exp463_fold_scorer=lambda *args, **kwargs: fake_exp463_scores_antigravity(3),
            )


if __name__ == "__main__":
    unittest.main()
