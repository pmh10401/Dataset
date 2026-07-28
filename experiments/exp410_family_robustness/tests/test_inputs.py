from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.exp410_family_robustness.config import (
    FROZEN_CONFIG,
    Exp410FrozenConfig,
)
from experiments.exp410_family_robustness.inputs import (
    load_verified_inputs,
    recompute_dataset_metrics,
)


@dataclass(frozen=True)
class FrozenFixture:
    input_dir: Path
    labels_path: Path
    exp291_path: Path
    exp292_path: Path
    config: Exp410FrozenConfig


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_frozen_fixture(root: Path) -> FrozenFixture:
    input_dir = root / "publication"
    input_dir.mkdir()
    labels_path = input_dir / "exp403_408_frozen_test_labels.csv"
    labels_path.write_text(
        "dataset,candidate_index,is_true_anomaly\n"
        "Alpha_normal_0,0,1\n"
        "Alpha_normal_0,1,0\n"
        "Beta_normal_0,0,1\n",
        encoding="utf-8",
    )

    prediction_path = input_dir / "exp404_hard_confidence.jsonl"
    _write_jsonl(
        prediction_path,
        [
            {"dataset": "Alpha_normal_0", "candidate_index": 0},
            {"dataset": "Alpha_normal_0", "candidate_index": 1},
        ],
    )
    prediction_manifest_path = input_dir / "exp404_prediction_manifest.json"
    _write_json(
        prediction_manifest_path,
        {
            "hard_confidence_file": prediction_path.name,
            "artifact_hashes": {"prediction": _digest(prediction_path)},
            "frozen_hard_lanes": {
                "Alpha_normal_0": [0, 1],
                "Beta_normal_0": [],
            },
        },
    )

    lane_rows = [
        {
            "dataset": "Alpha_normal_0",
            "automatic_hard_indices": [0, 1],
            "standard_review_indices": [],
            "priority_review_indices": [],
        },
        {
            "dataset": "Beta_normal_0",
            "automatic_hard_indices": [],
            "standard_review_indices": [],
            "priority_review_indices": [],
        },
    ]
    exp291_path = root / "exp291.jsonl"
    exp292_path = root / "exp292.jsonl"
    _write_jsonl(exp291_path, lane_rows)
    _write_jsonl(exp292_path, lane_rows)

    input_manifest_path = input_dir / "exp403_408_evaluation_input_manifest.json"
    _write_json(
        input_manifest_path,
        {
            "artifact_sha256": {
                prediction_manifest_path.name: _digest(prediction_manifest_path),
                prediction_path.name: _digest(prediction_path),
            }
        },
    )
    _write_json(
        input_dir / "exp403_408_evaluation_publication_manifest.json",
        {
            "artifact_sha256": {
                input_manifest_path.name: _digest(input_manifest_path),
                labels_path.name: _digest(labels_path),
            }
        },
    )

    return FrozenFixture(
        input_dir=input_dir,
        labels_path=labels_path,
        exp291_path=exp291_path,
        exp292_path=exp292_path,
        config=replace(
            FROZEN_CONFIG,
            expected_datasets=2,
            expected_families=2,
            expected_hard=2,
            expected_tp=1,
            expected_fp=1,
            expected_fn=1,
            expected_dataset_macro_f1=1.0 / 3.0,
            display_dataset_macro_f1="0.333333",
            prediction_sha256=_digest(prediction_path),
            frozen_labels_sha256=_digest(labels_path),
            exp291_sha256=_digest(exp291_path),
            exp292_sha256=_digest(exp292_path),
        ),
    )


class FrozenInputTests(unittest.TestCase):
    def test_loader_verifies_hashes_and_recomputes_metrics(self) -> None:
        with TemporaryDirectory() as raw:
            fixture = write_frozen_fixture(Path(raw))
            inputs = load_verified_inputs(
                fixture.input_dir,
                fixture.exp291_path,
                fixture.exp292_path,
                config=fixture.config,
            )
            metrics = recompute_dataset_metrics(inputs, config=fixture.config)
            self.assertEqual(
                [(row.tp, row.fp, row.fn) for row in metrics],
                [(1, 1, 0), (0, 0, 1)],
            )
            self.assertEqual(metrics[0].precision, 0.5)
            self.assertEqual(metrics[0].f1, 2.0 / 3.0)
            self.assertEqual(metrics[1].precision, None)
            self.assertEqual(metrics[1].f1, 0.0)

    def test_loader_fails_closed_on_rehashed_label_change(self) -> None:
        with TemporaryDirectory() as raw:
            fixture = write_frozen_fixture(Path(raw))
            fixture.labels_path.write_text(
                fixture.labels_path.read_text(encoding="utf-8").replace(",1\n", ",0\n", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_verified_inputs(
                    fixture.input_dir,
                    fixture.exp291_path,
                    fixture.exp292_path,
                    config=fixture.config,
                )


if __name__ == "__main__":
    unittest.main()
