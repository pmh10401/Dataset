"""Task 5 dashboard contract tests, written before production modules."""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
import tempfile
import unittest

import numpy as np

from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.artifacts_antigravity import (
    publish_result_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.dashboard_export_antigravity import (
    export_dashboard_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.evaluation_antigravity import (
    evaluate_qualification_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.predecessor_antigravity import (
    capture_exp465_authority_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.qualification_antigravity import (
    blocked_fold_indices_antigravity,
    qualify_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.replay_antigravity import (
    inspect_result_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.scoring_antigravity import (
    FoldMethodScoresAntigravity,
    score_fold_methods_antigravity,
)


@lru_cache(maxsize=1)
def records_fixture_antigravity() -> MappingProxyType:
    x = np.arange(140, dtype=np.float64)
    base = 0.01 * x + np.sin(2.0 * np.pi * x / 21.0) + 0.3 * np.cos(2.0 * np.pi * x / 9.0)
    train = np.asarray(
        base[None, :] + np.random.default_rng(46850).normal(scale=0.01, size=(69, 140)),
        dtype=np.float64,
        order="C",
    )
    train.setflags(write=False)
    authority = capture_exp465_authority_antigravity()
    records: dict[tuple[int, int], FoldMethodScoresAntigravity] = {}
    for seed in (20260721, 20261721):
        for fold_index, (fit, query) in enumerate(blocked_fold_indices_antigravity()):
            records[(seed, fold_index)] = score_fold_methods_antigravity(
                train,
                fit,
                query,
                seed=seed,
                fold_index=fold_index,
                authority=authority,
            )
    return MappingProxyType(records)


class DashboardExportAntigravityTest(unittest.TestCase):
    def test_dashboard_is_deterministic_and_explains_period_normal_and_private_state(self) -> None:
        records = records_fixture_antigravity()
        qualification = qualify_antigravity(records)
        decision = evaluate_qualification_antigravity(qualification)
        with tempfile.TemporaryDirectory(prefix="exp468-task5-") as directory:
            root = publish_result_antigravity(Path(directory) / "root", records, qualification, decision).root
            inspected = inspect_result_antigravity(root)
            html = export_dashboard_antigravity(inspected)
            self.assertEqual(html, export_dashboard_antigravity(inspected))
            for text in (
                "period 변화",
                "C0보다 개선한 fold",
                "정상 확인 오류",
                "private 재현성 확인 대기",
                "아직 실행하지 않음",
                'type="checkbox" checked disabled',
            ):
                self.assertIn(text, html)
            self.assertIn("fold 0", html)
            self.assertIn("C10", html)
            self.assertIn(
                "qualification FAIL — private 실행 전 차단",
                html,
            )
            self.assertIn(
                "실행 1: <strong>아직 실행하지 않음</strong>",
                html,
            )
            self.assertIn(
                "실행 2: <strong>아직 실행하지 않음</strong>",
                html,
            )

            passed_decision = dataclasses.replace(
                inspected.decision,
                status="qualification_gate_passed_awaiting_private_reproducibility",
                eligible_for_private_reproducibility=True,
                primary_passed_count=10,
                failure_reasons=(),
            )
            passed_snapshot = dict(inspected.dashboard_snapshot)
            passed_snapshot["qualification_passed"] = True
            passed_snapshot["private_reproducibility"] = {
                "run_1": "not_run",
                "run_2": "not_run",
                "status": "awaiting_private_reproducibility",
            }
            passed_inspected = dataclasses.replace(
                inspected,
                decision=passed_decision,
                dashboard_snapshot=MappingProxyType(passed_snapshot),
            )
            passed_html = export_dashboard_antigravity(passed_inspected)
            self.assertIn(
                "qualification PASS — private 재현성 확인 대기",
                passed_html,
            )
            self.assertIn(
                "실행 1: <strong>아직 실행하지 않음</strong>",
                passed_html,
            )
            self.assertIn(
                "실행 2: <strong>아직 실행하지 않음</strong>",
                passed_html,
            )


if __name__ == "__main__":
    unittest.main()
