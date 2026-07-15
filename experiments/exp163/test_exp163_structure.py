from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

from experiments.exp163.exp163_io import (
    FORBIDDEN_PREDICTION_KEYS,
    freeze_prediction_file,
    load_exp162_manifest,
    verify_prediction_manifest,
    write_predictions_atomic,
)

HERE = Path(__file__).resolve().parent


class PackageStructureTests(unittest.TestCase):

    def test_expected_modules_exist(self):
        expected = {
            "__init__.py",
            "alternative_tail_core.py",
            "exp163_io.py",
            "run_experiment_163_alternative_source_tail_agreement_review.py",
            "independent_reaggregate_exp163.py",
            "test_exp163_alternative_tail_review.py",
            "test_exp163_structure.py",
        }
        self.assertTrue(expected <= {path.name for path in HERE.glob("*.py")})

    def test_prediction_functions_do_not_accept_y_test(self):
        runner_file = HERE / "run_experiment_163_alternative_source_tail_agreement_review.py"
        if runner_file.exists():
            tree = ast.parse(runner_file.read_text(encoding="utf-8"))
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and "predict" in node.name:
                    names = [arg.arg for arg in node.args.args]
                    self.assertNotIn("y_test", names, msg=f"{runner_file.name}:{node.name}")

    def test_no_test_labels_or_metrics_in_predictions_source(self):
        runner_file = HERE / "run_experiment_163_alternative_source_tail_agreement_review.py"
        if runner_file.exists():
            source = runner_file.read_text(encoding="utf-8")
            # "run_predict" 함수 안에서 y_test나 tp, fp 같은 평가 관련 인자나 변수가 쓰이지 않는지 체크
            # ast로 run_predict FunctionDef를 찾아 그 내부에서 y_test가 사용되는지 확인
            tree = ast.parse(source)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == "run_predict":
                    # Check body of run_predict
                    for subnode in ast.walk(node):
                        if isinstance(subnode, ast.Name) and subnode.id == "y_test":
                            self.fail("y_test is forbidden inside run_predict")

    def test_prediction_rows_reject_evaluation_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            with self.assertRaisesRegex(ValueError, "forbidden"):
                write_predictions_atomic(path, [{"dataset_name": "d", "hard_tp": 1}])

    def test_prediction_manifest_detects_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prediction = root / "predictions.jsonl"
            manifest = root / "manifest.json"
            rows = [{"dataset_name": "d", "alpha": 0.01, "variant": "v"}]
            write_predictions_atomic(prediction, rows)
            freeze_prediction_file(
                prediction,
                manifest,
                row_count=1,
                input_hashes={"baseline": "abc"},
                config_hash="cfg",
            )
            verify_prediction_manifest(prediction, manifest, expected_row_count=1)
            prediction.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash"):
                verify_prediction_manifest(prediction, manifest, expected_row_count=1)

    def test_manifest_duplicate_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.csv"
            path.write_text(
                "dataset,file_path,file_hash,status\n"
                "d,a.npz,h,ok\n"
                "d,b.npz,h,ok\n",
                encoding="utf-8",
            )
            from experiments.exp163.exp163_io import get_sha256
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_exp162_manifest(path, get_sha256(path))

    def test_forbidden_keys_cover_tier_metrics(self):
        self.assertIn("hard_tp", FORBIDDEN_PREDICTION_KEYS)
        self.assertIn("standard_review_fp", FORBIDDEN_PREDICTION_KEYS)
        self.assertIn("combined_f1", FORBIDDEN_PREDICTION_KEYS)

    def test_frozen_hashes_are_declared(self):
        runner_file = HERE / "run_experiment_163_alternative_source_tail_agreement_review.py"
        if runner_file.exists():
            source = runner_file.read_text(encoding="utf-8")
            self.assertIn("ce97a7df2873d4bc628bbd58e2a7b83142f9d8cf8670bdf598f2b79aa0ded73d", source)
            self.assertIn("ec8e1a1c617aa2216a81da1959bd1aec529ad7aadb908ababe3f7e05b7f288c1", source)
            self.assertIn("0a80d8a26387d81cd7b2e18ae36f3116c829d20742dd1fa79bb8a3c5bc3d6461", source)


if __name__ == "__main__":
    unittest.main()
