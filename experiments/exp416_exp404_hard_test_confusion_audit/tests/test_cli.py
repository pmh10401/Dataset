from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments.exp416_exp404_hard_test_confusion_audit.audit import (
    compute_confusion_audit,
)
from experiments.exp416_exp404_hard_test_confusion_audit.contracts import (
    FrozenLabels,
    FrozenPrediction,
    LabelRow,
)
from scratch import run_exp416_exp404_hard_test_confusion_audit as cli


class Exp416CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.database = self.root / "database.sqlite"
        self.prediction_path = self.root / "prediction.jsonl"
        self.manifest_path = self.root / "prediction_manifest.json"
        self.labels_path = self.root / "labels.csv"
        for path in (
            self.database,
            self.prediction_path,
            self.manifest_path,
            self.labels_path,
        ):
            path.write_bytes(b"synthetic\n")
        self.output = self.root / "new-output"
        self.argv = [
            "--database-reference",
            str(self.database),
            "--prediction",
            str(self.prediction_path),
            "--prediction-manifest",
            str(self.manifest_path),
            "--frozen-test-labels",
            str(self.labels_path),
            "--output-dir",
            str(self.output),
        ]
        self.commit = "a" * 40
        self.database_sha = "b" * 64
        self.prediction = FrozenPrediction(
            coordinates=(("A", 0),),
            dataset_scope=("A",),
            prediction_dataset_count=1,
            identity_sha256="1" * 64,
            prediction_sha256="2" * 64,
            manifest_sha256="3" * 64,
        )
        self.labels = FrozenLabels(
            rows=(LabelRow("A", 0, 1),),
            dataset_scope=("A",),
            labels_sha256="4" * 64,
        )
        self.result = compute_confusion_audit(self.prediction, self.labels)
        self.payloads = {
            "exp416_confusion_matrix.json": b"summary",
            "exp416_dataset_metrics.csv": b"csv",
            "EXP416_RESULT_REPORT_KO.md": b"report",
        }
        self.hashes = {
            "EXP416_RESULT_REPORT_KO.md": "5" * 64,
            "exp416_confusion_matrix.json": "6" * 64,
            "exp416_dataset_metrics.csv": "7" * 64,
        }

    def orchestration_patches(self):
        return (
            mock.patch.object(
                cli, "_require_clean_git_commit", return_value=self.commit
            ),
            mock.patch.object(
                cli, "verify_database_reference", return_value=self.database_sha
            ),
            mock.patch.object(
                cli, "freeze_prediction_inputs", return_value=self.prediction
            ),
            mock.patch.object(
                cli, "load_frozen_test_labels", return_value=self.labels
            ),
            mock.patch.object(
                cli, "compute_confusion_audit", return_value=self.result
            ),
            mock.patch.object(cli, "render_payloads", return_value=self.payloads),
            mock.patch.object(cli, "verify_rendered_payloads", return_value=None),
            mock.patch.object(cli, "publish_payloads", return_value=self.hashes),
        )

    def test_all_five_paths_are_required(self) -> None:
        for index in range(0, len(self.argv), 2):
            incomplete = self.argv[:index] + self.argv[index + 2 :]
            with self.subTest(missing=self.argv[index]):
                with redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        cli.main(incomplete)
                self.assertEqual(raised.exception.code, 2)

    def test_existing_output_is_rejected_before_audit(self) -> None:
        self.output.mkdir()
        with mock.patch.object(cli, "verify_database_reference") as database:
            with self.assertRaisesRegex(FileExistsError, "must not exist"):
                cli.main(self.argv)
        database.assert_not_called()

    def test_nonexistent_output_parent_is_rejected_before_audit(self) -> None:
        missing_output = self.root / "missing-parent" / "output"
        argv = self.argv[:-1] + [str(missing_output)]
        with mock.patch.object(cli, "verify_database_reference") as database:
            with self.assertRaisesRegex(FileNotFoundError, "parent"):
                cli.main(argv)
        database.assert_not_called()

    def test_nonexistent_input_is_rejected_before_audit(self) -> None:
        missing_database = self.root / "missing.sqlite"
        argv = self.argv.copy()
        argv[1] = str(missing_database)
        with mock.patch.object(cli, "verify_database_reference") as database:
            with self.assertRaisesRegex(FileNotFoundError, "input file"):
                cli.main(argv)
        database.assert_not_called()

    def test_forbidden_extra_mode_and_policy_arguments_are_rejected(self) -> None:
        for extras in (
            ["--mode", "smoke"],
            ["--alpha", "0.05"],
            ["--threshold", "0.9"],
            ["--source", "other"],
            ["--budget", "1"],
            ["--lane", "easy"],
            ["--family-filter", "A"],
            ["--dataset-subset", "A"],
        ):
            with self.subTest(extras=extras):
                with redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        cli.main([*self.argv, *extras])
                self.assertEqual(raised.exception.code, 2)

    def test_dirty_git_is_rejected_before_inputs_are_read(self) -> None:
        with (
            mock.patch.object(
                cli,
                "_require_clean_git_commit",
                side_effect=ValueError("Exp416 requires a clean Git worktree"),
            ),
            mock.patch.object(cli, "verify_database_reference") as database,
            mock.patch.object(cli, "freeze_prediction_inputs") as freeze,
            mock.patch.object(cli, "load_frozen_test_labels") as labels,
        ):
            with self.assertRaisesRegex(ValueError, "clean Git"):
                cli.main(self.argv)
        database.assert_not_called()
        freeze.assert_not_called()
        labels.assert_not_called()

    def test_cli_freezes_prediction_before_loading_labels(self) -> None:
        events: list[str] = []
        with (
            mock.patch.object(
                cli, "_require_clean_git_commit", return_value=self.commit
            ),
            mock.patch.object(
                cli,
                "verify_database_reference",
                side_effect=lambda *a, **k: events.append("database")
                or self.database_sha,
            ) as database,
            mock.patch.object(
                cli,
                "freeze_prediction_inputs",
                side_effect=lambda *a, **k: events.append("freeze")
                or self.prediction,
            ) as freeze,
            mock.patch.object(
                cli,
                "load_frozen_test_labels",
                side_effect=lambda *a, **k: events.append("labels") or self.labels,
            ) as labels,
            mock.patch.object(
                cli,
                "compute_confusion_audit",
                side_effect=lambda *a, **k: events.append("audit") or self.result,
            ) as audit,
            mock.patch.object(
                cli,
                "render_payloads",
                side_effect=lambda *a, **k: events.append("render") or self.payloads,
            ) as render,
            mock.patch.object(cli, "verify_rendered_payloads") as verify_render,
            mock.patch.object(
                cli,
                "publish_payloads",
                side_effect=lambda *a, **k: events.append("publish") or self.hashes,
            ) as publish,
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(cli.main(self.argv), 0)

        self.assertEqual(
            events,
            ["database", "freeze", "labels", "audit", "render", "publish"],
        )
        database.assert_called_once_with(self.database)
        freeze.assert_called_once_with(self.prediction_path, self.manifest_path)
        labels.assert_called_once_with(self.labels_path, self.prediction)
        audit.assert_called_once_with(self.prediction, self.labels)
        render.assert_called_once_with(
            self.result,
            self.prediction,
            self.labels,
            self.database_sha,
            self.commit,
        )
        verify_render.assert_called_once_with(self.payloads)
        self.assertEqual(publish.call_args.args[:2], (self.payloads, self.output))

    def test_prediction_failure_never_loads_labels(self) -> None:
        patches = self.orchestration_patches()
        with (
            patches[0],
            patches[1],
            mock.patch.object(
                cli,
                "freeze_prediction_inputs",
                side_effect=ValueError("prediction rejected"),
            ),
            mock.patch.object(cli, "load_frozen_test_labels") as labels,
            mock.patch.object(cli, "publish_payloads") as publish,
        ):
            with self.assertRaisesRegex(ValueError, "prediction rejected"):
                cli.main(self.argv)
        labels.assert_not_called()
        publish.assert_not_called()

    def test_label_audit_and_render_failures_never_publish(self) -> None:
        failure_cases = (
            ("load_frozen_test_labels", "label rejected"),
            ("compute_confusion_audit", "audit rejected"),
            ("render_payloads", "render rejected"),
            ("verify_rendered_payloads", "render verification rejected"),
        )
        for target, message in failure_cases:
            with self.subTest(target=target):
                patches = self.orchestration_patches()
                with (
                    patches[0],
                    patches[1],
                    patches[2],
                    patches[3],
                    patches[4],
                    patches[5],
                    patches[6],
                    patches[7] as publish,
                    mock.patch.object(cli, target, side_effect=ValueError(message)),
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        cli.main(self.argv)
                publish.assert_not_called()

    def test_publication_callback_reverifies_exact_input_snapshots(self) -> None:
        callbacks: list[object] = []

        def capture_callback(*args, **kwargs):
            callbacks.append(kwargs["pre_publish_verify"])
            kwargs["pre_publish_verify"]()
            return self.hashes

        patches = self.orchestration_patches()
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            mock.patch.object(
                cli, "publish_payloads", side_effect=capture_callback
            ),
            mock.patch.object(cli, "verify_input_snapshots") as verify_inputs,
            redirect_stdout(StringIO()),
        ):
            self.assertEqual(cli.main(self.argv), 0)

        self.assertEqual(len(callbacks), 1)
        verify_inputs.assert_called_once_with(
            self.database,
            self.prediction_path,
            self.manifest_path,
            self.labels_path,
            self.database_sha,
            self.prediction,
            self.labels,
        )

    def test_stdout_is_canonical_json_with_hashes_and_micro_counts(self) -> None:
        patches = self.orchestration_patches()
        stdout = StringIO()
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
            patches[7],
            redirect_stdout(stdout),
        ):
            self.assertEqual(cli.main(self.argv), 0)

        self.assertEqual(
            stdout.getvalue(),
            (
                '{"artifact_sha256":{'
                '"EXP416_RESULT_REPORT_KO.md":"'
                + "5" * 64
                + '","exp416_confusion_matrix.json":"'
                + "6" * 64
                + '","exp416_dataset_metrics.csv":"'
                + "7" * 64
                + '"},"micro_counts":{"fn":0,"fp":0,"tn":0,"tp":1}}\n'
            ),
        )


if __name__ == "__main__":
    unittest.main()
