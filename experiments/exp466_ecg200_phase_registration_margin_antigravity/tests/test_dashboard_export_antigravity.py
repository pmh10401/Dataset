from __future__ import annotations

from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest

from experiments.exp466_ecg200_phase_registration_margin_antigravity.dashboard_export_antigravity import (
    export_dashboard_antigravity,
)
from experiments.exp466_ecg200_phase_registration_margin_antigravity.artifacts_antigravity import (
    publish_result_antigravity,
)
from experiments.exp466_ecg200_phase_registration_margin_antigravity.replay_antigravity import (
    inspect_result_antigravity,
)
from experiments.exp466_ecg200_phase_registration_margin_antigravity.tests.test_artifacts_antigravity import (
    LEAVES_ANTIGRAVITY,
    task5_inputs_antigravity,
)


def fold_name_antigravity(seed: int, fold_index: int) -> str:
    return f"{seed}:{fold_index}"


class TagRecorderAntigravity(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        self.tags.append(tag)


class DashboardExportAntigravityTest(unittest.TestCase):
    def test_easy_korean_snapshot_embeds_deterministic_html_without_eighth_leaf_antigravity(
        self,
    ) -> None:
        """Break caught: dashboard overclaims synthetic results or adds an HTML leaf."""
        records, qualification, decision = task5_inputs_antigravity()
        improved = tuple(
            fold_name_antigravity(row.seed, row.fold_index)
            for row in qualification.fold_results
            if row.c9_period_auc > row.c0_period_auc
        )
        failed = tuple(
            fold_name_antigravity(row.seed, row.fold_index)
            for row in qualification.fold_results
            if not row.passed
        )
        normal_confirmations = sum(qualification.normal_confirmations.values())

        first = export_dashboard_antigravity(qualification, decision)
        second = export_dashboard_antigravity(qualification, decision)
        self.assertEqual(first, second)
        self.assertEqual(first["후보"], "C9")
        self.assertEqual(first["주기_AUC_개선_fold_수"], len(improved))
        self.assertEqual(tuple(first["주기_AUC_개선_fold"]), improved)
        self.assertEqual(tuple(first["실패_fold"]), failed)
        self.assertEqual(first["정상_confirmation_수"], normal_confirmations)
        self.assertIs(first["Task5_실제_private_official_결과"], False)

        html = first["html_antigravity"]
        self.assertIs(type(html), str)
        self.assertIn(
            f"C9의 주기 AUC가 C0보다 좋아진 검증 묶음은 {len(improved)}개입니다.",
            html,
        )
        if failed:
            self.assertIn(f"실패한 검증 묶음은 {', '.join(failed)}입니다.", html)
        else:
            self.assertIn("실패한 검증 묶음은 없습니다.", html)
        self.assertIn(f"정상 오탐 확인은 {normal_confirmations}개입니다.", html)
        self.assertIn("Task 5에는 실제 비공개 또는 공식 결과가 없습니다.", html)
        self.assertIn("이 합성 값은 성능 결과가 아닙니다.", html)
        self.assertNotIn("qualification_passed", html)
        self.assertNotIn("10개 모두 통과", html)
        self.assertTrue(html.startswith('<!doctype html>\n<html lang="ko">'))
        self.assertTrue(html.endswith("</html>\n"))
        parser = TagRecorderAntigravity()
        parser.feed(html)
        self.assertEqual(parser.tags[0], "html")
        self.assertIn("body", parser.tags)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "result_antigravity"
            publish_result_antigravity(root, records, qualification, decision)
            self.assertEqual(
                tuple(sorted(path.name for path in root.iterdir())),
                LEAVES_ANTIGRAVITY,
            )
            self.assertFalse(any(path.suffix == ".html" for path in root.iterdir()))
            stored = json.loads(
                (root / "dashboard_snapshot_antigravity.json").read_text(
                    encoding="utf-8",
                ),
            )
            self.assertEqual(stored, first)
            inspected = inspect_result_antigravity(root)
            self.assertEqual(inspected.dashboard, first)
            self.assertEqual(inspected.dashboard_html, html)


if __name__ == "__main__":
    unittest.main()
