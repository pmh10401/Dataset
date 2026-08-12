from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.paper_public_baselines.artifacts import (
    canonical_json_bytes,
    validate_score_payload,
    write_score_shard_create_only,
)


def valid_score_payload() -> dict[str, object]:
    return {
        "dataset": "toy",
        "method": "knn_distance",
        "n_train": 3,
        "n_test": 2,
        "status": "complete",
        "reason": None,
        "test_indices": [0, 1],
        "train_oof_scores": [0.1, 0.2, 0.3],
        "test_scores": [0.4, 0.5],
        "alerts_by_alpha": {"0.005": [], "0.01": [], "0.02": []},
        "runtime_seconds": 0.25,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }


class ArtifactTests(unittest.TestCase):
    def test_score_shard_schema_rejects_evaluation_fields(self) -> None:
        payload = valid_score_payload() | {"tp": 1}

        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_score_payload(payload)

    def test_canonical_json_is_sorted_compact_utf8_with_newline(self) -> None:
        payload = {"z": 1, "a": "한글"}

        encoded = canonical_json_bytes(payload)

        self.assertEqual(encoded, '{"a":"한글","z":1}\n'.encode())

    def test_create_only_shard_reuses_identical_bytes_and_rejects_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.json"
            payload = valid_score_payload()

            first = write_score_shard_create_only(path, payload)
            second = write_score_shard_create_only(path, payload)
            changed = json.loads(path.read_text()) | {"runtime_seconds": 9.0}

            self.assertEqual(first, second)
            with self.assertRaisesRegex(ValueError, "mismatch"):
                write_score_shard_create_only(path, changed)


if __name__ == "__main__":
    unittest.main()
