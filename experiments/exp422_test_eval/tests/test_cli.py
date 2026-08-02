import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scratch.run_exp422_test_eval import main


class CliTests(unittest.TestCase):
    def test_missing_prediction_writes_blocked_status_without_labels(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "attempt"
            code = main(
                [
                    "--database", str(root / "missing.db"),
                    "--prediction", str(root / "missing.jsonl"),
                    "--prediction-manifest", str(root / "missing-manifest.json"),
                    "--frozen-test-labels", str(root / "labels.csv"),
                    "--output-root", str(output),
                ]
            )
            self.assertEqual(code, 0)
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "blocked_no_prediction")
            self.assertFalse(status["test_labels_opened"])
            self.assertTrue((output / "progress.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
