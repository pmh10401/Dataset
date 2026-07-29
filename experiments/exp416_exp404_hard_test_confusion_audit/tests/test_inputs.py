from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from experiments.exp416_exp404_hard_test_confusion_audit.config import (
    Exp416FrozenConfig,
    FROZEN_CONFIG,
)
from experiments.exp416_exp404_hard_test_confusion_audit.contracts import (
    FrozenPrediction,
)
from experiments.exp416_exp404_hard_test_confusion_audit.inputs import (
    freeze_prediction_inputs,
    load_frozen_test_labels,
    verify_database_reference,
    verify_input_snapshots,
)


@contextmanager
def fixture_config_validation() -> Iterator[None]:
    with mock.patch.object(
        Exp416FrozenConfig,
        "validate",
        autospec=True,
        return_value=None,
    ):
        yield


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def prediction_row(dataset: str, candidate_index: object, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": dataset,
        "candidate_index": candidate_index,
        "contains_test_labels": False,
        "contains_evaluation_metrics": False,
    }
    row.update(overrides)
    return row


def valid_frozen_prediction(scope: tuple[str, ...] = ("A",)) -> FrozenPrediction:
    coordinates = tuple((dataset, 0) for dataset in scope)
    return FrozenPrediction(
        coordinates=coordinates,
        dataset_scope=scope,
        prediction_dataset_count=len(scope),
        identity_sha256="identity",
        prediction_sha256="prediction",
        manifest_sha256="manifest",
    )


class InputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write_inputs(
        self,
        rows: list[dict[str, object]],
        *,
        frozen_hard_lanes: dict[str, list[object]],
        expected_datasets: int | None = None,
        expected_prediction_datasets: int | None = None,
        expected_prediction_coordinates: int | None = None,
        manifest_overrides: dict[str, object] | None = None,
    ) -> tuple[Path, Path, Exp416FrozenConfig]:
        prediction = self.root / "prediction.jsonl"
        prediction.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        scope = list(frozen_hard_lanes)
        manifest = self.root / "manifest.json"
        payload: dict[str, object] = {
            "contains_test_labels": False,
            "contains_evaluation_metrics": False,
            "datasets": (
                len({str(row["dataset"]) for row in rows})
                if expected_prediction_datasets is None
                else expected_prediction_datasets
            ),
            "rows": (
                len(rows)
                if expected_prediction_coordinates is None
                else expected_prediction_coordinates
            ),
            "frozen_hard_lanes": frozen_hard_lanes,
        }
        payload.update(manifest_overrides or {})
        manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        config = replace(
            FROZEN_CONFIG,
            expected_datasets=len(scope) if expected_datasets is None else expected_datasets,
            expected_prediction_datasets=payload["datasets"],
            expected_prediction_coordinates=payload["rows"],
            prediction_sha256=digest(prediction),
            manifest_sha256=digest(manifest),
        )
        return prediction, manifest, config

    def write_single_prediction(
        self, *, row_overrides: dict[str, object] | None = None
    ) -> tuple[Path, Path, Exp416FrozenConfig]:
        return self.write_inputs(
            [prediction_row("A", 0, **(row_overrides or {}))],
            frozen_hard_lanes={"A": [0]},
        )

    def write_labels(
        self,
        payload: str,
        *,
        expected_rows: int,
        expected_datasets: int = 1,
    ) -> tuple[Path, Exp416FrozenConfig]:
        labels = self.root / "labels.csv"
        labels.write_text(payload, encoding="utf-8")
        return labels, replace(
            FROZEN_CONFIG,
            expected_datasets=expected_datasets,
            expected_label_rows=expected_rows,
            labels_sha256=digest(labels),
        )

    def test_database_reference_is_hash_only_and_rejects_wrong_digest(self) -> None:
        database = self.root / "reference.sqlite"
        database.write_bytes(b"synthetic sqlite bytes only")
        config = replace(FROZEN_CONFIG, database_sha256=digest(database))
        with fixture_config_validation():
            self.assertEqual(verify_database_reference(database, config=config), digest(database))
            with self.assertRaisesRegex(ValueError, "database"):
                verify_database_reference(
                    database, config=replace(config, database_sha256="0" * 64)
                )

    def test_freeze_returns_canonical_sorted_identity(self) -> None:
        prediction, manifest, config = self.write_inputs(
            [prediction_row("B", 2), prediction_row("A", 3)],
            frozen_hard_lanes={"A": [3], "B": [2]},
        )
        with fixture_config_validation():
            frozen = freeze_prediction_inputs(prediction, manifest, config=config)
        self.assertEqual(frozen.coordinates, (("A", 3), ("B", 2)))
        self.assertEqual(frozen.dataset_scope, ("A", "B"))
        self.assertEqual(
            frozen.identity_sha256,
            "d2b5bf8d97853a5cf97883965ad8a41dac0885ba87436ff710caa4ff0d6dcf20",
        )

    def test_freeze_rejects_duplicate_and_manifest_mismatch(self) -> None:
        rows = [prediction_row("A", 3), prediction_row("A", 3)]
        prediction, manifest, config = self.write_inputs(
            rows,
            frozen_hard_lanes={"A": [3], "B": []},
            expected_datasets=2,
            expected_prediction_datasets=1,
            expected_prediction_coordinates=2,
        )
        with fixture_config_validation():
            with self.assertRaisesRegex(ValueError, "duplicate"):
                freeze_prediction_inputs(prediction, manifest, config=config)

    def test_freeze_rejects_independent_prediction_and_manifest_hash_mismatches(self) -> None:
        prediction, manifest, config = self.write_single_prediction()
        with fixture_config_validation():
            with self.assertRaisesRegex(ValueError, "prediction"):
                freeze_prediction_inputs(
                    prediction,
                    manifest,
                    config=replace(config, prediction_sha256="0" * 64),
                )
            with self.assertRaisesRegex(ValueError, "manifest"):
                freeze_prediction_inputs(
                    prediction,
                    manifest,
                    config=replace(config, manifest_sha256="0" * 64),
                )

    def test_prediction_forbids_label_or_metric_fields(self) -> None:
        for field, value in (
            ("contains_test_labels", True),
            ("contains_evaluation_metrics", True),
        ):
            with self.subTest(field=field):
                prediction, manifest, config = self.write_single_prediction(
                    row_overrides={field: value}
                )
                with fixture_config_validation():
                    with self.assertRaises(ValueError):
                        freeze_prediction_inputs(prediction, manifest, config=config)

    def test_manifest_requires_exact_false_flags_and_integer_counts(self) -> None:
        for overrides in (
            {"contains_test_labels": True},
            {"contains_evaluation_metrics": True},
            {"rows": True},
        ):
            with self.subTest(overrides=overrides):
                prediction, manifest, config = self.write_inputs(
                    [prediction_row("A", 0)],
                    frozen_hard_lanes={"A": [0]},
                    manifest_overrides=overrides,
                )
                with fixture_config_validation():
                    with self.assertRaises(ValueError):
                        freeze_prediction_inputs(prediction, manifest, config=config)

    def test_prediction_rejects_bool_or_negative_candidate_index(self) -> None:
        for candidate_index in (True, -1):
            with self.subTest(candidate_index=candidate_index):
                prediction, manifest, config = self.write_inputs(
                    [prediction_row("A", candidate_index)],
                    frozen_hard_lanes={"A": [candidate_index]},
                )
                with fixture_config_validation():
                    with self.assertRaisesRegex(ValueError, "candidate_index"):
                        freeze_prediction_inputs(prediction, manifest, config=config)

    def test_freeze_rejects_duplicate_or_wrong_count_manifest_scope(self) -> None:
        prediction, manifest, config = self.write_inputs(
            [prediction_row("A", 0)], frozen_hard_lanes={"A": [0]}
        )
        manifest.write_text(
            '{"contains_test_labels":false,"contains_evaluation_metrics":false,'
            '"datasets":1,"rows":1,"frozen_hard_lanes":{"A":[0],"A":[0]}}',
            encoding="utf-8",
        )
        config = replace(config, manifest_sha256=digest(manifest))
        with fixture_config_validation():
            with self.assertRaisesRegex(ValueError, "duplicate"):
                freeze_prediction_inputs(prediction, manifest, config=config)

        prediction, manifest, config = self.write_inputs(
            [prediction_row("A", 0)],
            frozen_hard_lanes={"A": [0]},
            expected_datasets=2,
        )
        with fixture_config_validation():
            with self.assertRaisesRegex(ValueError, "population"):
                freeze_prediction_inputs(prediction, manifest, config=config)

    def test_freeze_rejects_lane_and_jsonl_coordinate_set_differences(self) -> None:
        for lanes in ({"A": [1]}, {"A": [0, 1]}):
            with self.subTest(lanes=lanes):
                prediction, manifest, config = self.write_inputs(
                    [prediction_row("A", 0)], frozen_hard_lanes=lanes
                )
                with fixture_config_validation():
                    with self.assertRaisesRegex(ValueError, "coordinate"):
                        freeze_prediction_inputs(prediction, manifest, config=config)

    def test_freeze_rehashes_source_files_after_parse(self) -> None:
        prediction, manifest, config = self.write_single_prediction()
        from experiments.exp416_exp404_hard_test_confusion_audit import inputs

        original_hash = inputs.sha256_file
        calls = 0

        def mutate_after_parse(path: Path) -> str:
            nonlocal calls
            calls += 1
            if calls == 3:
                prediction.write_text(prediction.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            return original_hash(path)

        with fixture_config_validation():
            with mock.patch.object(inputs, "sha256_file", side_effect=mutate_after_parse):
                with self.assertRaisesRegex(ValueError, "prediction"):
                    freeze_prediction_inputs(prediction, manifest, config=config)

    def test_labels_cannot_load_without_frozen_prediction(self) -> None:
        labels = self.root / "labels.csv"
        labels.write_text(
            "dataset,candidate_index,is_true_anomaly\nA,0,1\n",
            encoding="utf-8",
        )
        with mock.patch.object(Path, "open", side_effect=AssertionError("opened")):
            with self.assertRaises((TypeError, ValueError)):
                load_frozen_test_labels(
                    labels,
                    object(),  # type: ignore[arg-type]
                    config=FROZEN_CONFIG,
                )

    def test_labels_reject_bool_like_and_duplicate_coordinates(self) -> None:
        frozen = valid_frozen_prediction(("A",))
        for payload in (
            "dataset,candidate_index,is_true_anomaly\nA,0,true\n",
            "dataset,candidate_index,is_true_anomaly\nA,0,1\nA,0,0\n",
        ):
            with self.subTest(payload=payload):
                labels, config = self.write_labels(payload, expected_rows=1)
                with fixture_config_validation():
                    with self.assertRaises(ValueError):
                        load_frozen_test_labels(labels, frozen, config=config)

    def test_labels_reject_wrong_hash_before_opening_csv(self) -> None:
        labels, config = self.write_labels(
            "dataset,candidate_index,is_true_anomaly\nA,0,1\n", expected_rows=1
        )
        with fixture_config_validation():
            with self.assertRaisesRegex(ValueError, "SHA256"):
                load_frozen_test_labels(
                    labels,
                    valid_frozen_prediction(),
                    config=replace(config, labels_sha256="0" * 64),
                )

    def test_labels_require_exact_header_and_valid_coordinate_values(self) -> None:
        frozen = valid_frozen_prediction(("A",))
        for payload in (
            "candidate_index,dataset,is_true_anomaly\n0,A,1\n",
            "dataset,candidate_index,is_true_anomaly\nA,-1,1\n",
            "dataset,candidate_index,is_true_anomaly\n ,0,1\n",
        ):
            with self.subTest(payload=payload):
                labels, config = self.write_labels(payload, expected_rows=1)
                with fixture_config_validation():
                    with self.assertRaises(ValueError):
                        load_frozen_test_labels(labels, frozen, config=config)

    def test_labels_enforce_row_dataset_scope_and_prediction_coverage(self) -> None:
        cases = (
            (
                valid_frozen_prediction(("A",)),
                "dataset,candidate_index,is_true_anomaly\nA,0,1\n",
                2,
                1,
            ),
            (
                valid_frozen_prediction(("A", "B")),
                "dataset,candidate_index,is_true_anomaly\nA,0,1\n",
                1,
                2,
            ),
            (
                FrozenPrediction(
                    coordinates=(("A", 1),),
                    dataset_scope=("A",),
                    prediction_dataset_count=1,
                    identity_sha256="identity",
                    prediction_sha256="prediction",
                    manifest_sha256="manifest",
                ),
                "dataset,candidate_index,is_true_anomaly\nA,0,1\n",
                1,
                1,
            ),
        )
        for frozen, payload, expected_rows, expected_datasets in cases:
            with self.subTest(frozen=frozen):
                labels, config = self.write_labels(
                    payload,
                    expected_rows=expected_rows,
                    expected_datasets=expected_datasets,
                )
                with fixture_config_validation():
                    with self.assertRaises(ValueError):
                        load_frozen_test_labels(labels, frozen, config=config)

    def test_labels_rehash_after_streaming_and_return_sorted_rows(self) -> None:
        payload = "dataset,candidate_index,is_true_anomaly\nB,0,0\nA,0,1\n"
        labels, config = self.write_labels(payload, expected_rows=2, expected_datasets=2)
        frozen = valid_frozen_prediction(("A", "B"))
        from experiments.exp416_exp404_hard_test_confusion_audit import inputs

        original_hash = inputs.sha256_file
        calls = 0

        def mutate_after_stream(path: Path) -> str:
            nonlocal calls
            calls += 1
            if calls == 2:
                labels.write_text(payload + "\n", encoding="utf-8")
            return original_hash(path)

        with fixture_config_validation():
            with mock.patch.object(inputs, "sha256_file", side_effect=mutate_after_stream):
                with self.assertRaisesRegex(ValueError, "label"):
                    load_frozen_test_labels(labels, frozen, config=config)

        labels.write_text(payload, encoding="utf-8")
        config = replace(config, labels_sha256=digest(labels))
        with fixture_config_validation():
            result = load_frozen_test_labels(labels, frozen, config=config)
        self.assertEqual(
            tuple((row.dataset, row.candidate_index, row.is_true_anomaly) for row in result.rows),
            (("A", 0, 1), ("B", 0, 0)),
        )

    def test_snapshot_verification_recomputes_all_artifacts_without_rebuilding_objects(self) -> None:
        prediction, manifest, config = self.write_single_prediction()
        database = self.root / "reference.sqlite"
        database.write_bytes(b"synthetic database")
        labels = self.root / "labels.csv"
        labels.write_text("dataset,candidate_index,is_true_anomaly\nA,0,1\n", encoding="utf-8")
        config = replace(
            config,
            prediction_sha256=digest(prediction),
            manifest_sha256=digest(manifest),
            expected_label_rows=1,
            labels_sha256=digest(labels),
            database_sha256=digest(database),
        )
        with fixture_config_validation():
            frozen_prediction = freeze_prediction_inputs(prediction, manifest, config=config)
            frozen_labels = load_frozen_test_labels(labels, frozen_prediction, config=config)
            self.assertIsNone(
                verify_input_snapshots(
                    database,
                    prediction,
                    manifest,
                    labels,
                    config.database_sha256,
                    frozen_prediction,
                    frozen_labels,
                    config=config,
                )
            )
            with self.assertRaisesRegex(ValueError, "database reference snapshot"):
                verify_input_snapshots(
                    database,
                    prediction,
                    manifest,
                    labels,
                    "0" * 64,
                    frozen_prediction,
                    frozen_labels,
                    config=config,
                )
            for prediction_field, prediction_override, labels_override, message in (
                ("prediction_sha256", "0" * 64, None, "frozen prediction"),
                ("manifest_sha256", "0" * 64, None, "frozen manifest"),
                (None, None, "0" * 64, "frozen label"),
            ):
                with self.subTest(message=message):
                    prediction_for_check = (
                        replace(
                            frozen_prediction,
                            **{prediction_field: prediction_override},
                        )
                        if prediction_field is not None
                        else frozen_prediction
                    )
                    labels_for_check = (
                        replace(frozen_labels, labels_sha256=labels_override)
                        if labels_override is not None
                        else frozen_labels
                    )
                    with self.assertRaisesRegex(ValueError, message):
                        verify_input_snapshots(
                            database,
                            prediction,
                            manifest,
                            labels,
                            config.database_sha256,
                            prediction_for_check,
                            labels_for_check,
                            config=config,
                        )

    def test_snapshot_verification_rehashes_each_source_artifact(self) -> None:
        prediction, manifest, config = self.write_single_prediction()
        database = self.root / "reference.sqlite"
        database.write_bytes(b"synthetic database")
        labels = self.root / "labels.csv"
        labels.write_text("dataset,candidate_index,is_true_anomaly\nA,0,1\n", encoding="utf-8")
        config = replace(
            config,
            expected_label_rows=1,
            labels_sha256=digest(labels),
            database_sha256=digest(database),
        )
        with fixture_config_validation():
            frozen_prediction = freeze_prediction_inputs(prediction, manifest, config=config)
            frozen_labels = load_frozen_test_labels(labels, frozen_prediction, config=config)
            for path, suffix, message in (
                (prediction, "\n", "frozen prediction"),
                (manifest, "\n", "frozen manifest"),
                (labels, "\n", "frozen label"),
            ):
                with self.subTest(path=path.name):
                    original = path.read_text(encoding="utf-8")
                    path.write_text(original + suffix, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, message):
                        verify_input_snapshots(
                            database,
                            prediction,
                            manifest,
                            labels,
                            config.database_sha256,
                            frozen_prediction,
                            frozen_labels,
                            config=config,
                        )
                    path.write_text(original, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
