"""Behavioral tests for the Exp417 official and non-official CLIs."""

from __future__ import annotations

from contextlib import redirect_stderr
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess
import unittest
from unittest.mock import patch

from experiments.exp417_causal_multihorizon_shadow.config import FROZEN_CONFIG


class OfficialCliTests(unittest.TestCase):
    def setUp(self) -> None:
        from scratch import run_exp417_causal_multihorizon_shadow as official

        self.official = official

    def test_parser_accepts_only_the_seven_frozen_path_arguments(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            for name in ("db", "p291", "m291", "p292", "m292"):
                path = root / name
                path.write_bytes(b"x")
                files.append(path)
            argv = [
                "--database", str(files[0]),
                "--exp291-prediction", str(files[1]),
                "--exp291-manifest", str(files[2]),
                "--exp292-prediction", str(files[3]),
                "--exp292-manifest", str(files[4]),
                "--checkpoint-dir", str(root / "checkpoint"),
                "--output-dir", str(root / "output"),
            ]
            args = self.official.build_parser().parse_args(argv)
            self.assertEqual(args.database, files[0].resolve())
            for forbidden in (
                "--dataset", "--max-datasets", "--family", "--alpha",
                "--horizon", "--threshold", "--seed", "--budget",
                "--backend", "--label-path", "--data",
            ):
                with self.subTest(forbidden=forbidden):
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        self.official.build_parser().parse_args(
                            argv + [forbidden, "x"]
                        )

    def test_parser_rejects_existing_output_and_missing_parents(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.official.build_parser().parse_args(
                    ["--output-dir", str(output)]
                )
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.official.build_parser().parse_args(
                    ["--checkpoint-dir", str(root / "missing" / "checkpoint")]
                )

    def test_dirty_git_is_rejected_by_the_real_cleanliness_guard(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(
                ["git", "-C", str(root), "config", "user.email", "test@example.invalid"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(root), "config", "user.name", "Exp417 Test"],
                check=True,
            )
            (root / "tracked").write_text("clean\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(root), "add", "tracked"], check=True)
            subprocess.run(
                ["git", "-C", str(root), "commit", "-qm", "fixture"], check=True
            )
            self.official.require_clean_git(root)
            (root / "tracked").write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "clean"):
                self.official.require_clean_git(root)

    def test_official_pipeline_has_strict_order_and_stops_after_failure(self) -> None:
        events: list[str] = []
        stages = self.official.OfficialStages(
            clean_git=lambda: events.append("clean_git"),
            label_guard=lambda: events.append("label_guard"),
            freeze_inputs=lambda: events.append("freeze_inputs") or "context",
            open_checkpoint=lambda context: events.append("checkpoint") or "store",
            run=lambda context, store: events.append("runner") or "summary",
            revalidate=lambda: events.append("revalidate"),
            render=lambda summary, store: events.append("render") or {"payload": b"x"},
            verify=lambda payloads, store: events.append("verify"),
            publish=lambda payloads, store: events.append("publish"),
        )
        self.assertEqual(self.official.execute_official(stages), 0)
        self.assertEqual(
            events,
            [
                "clean_git", "label_guard", "freeze_inputs", "checkpoint",
                "runner", "revalidate", "render", "verify", "publish",
            ],
        )

        for failed_stage in tuple(events):
            observed: list[str] = []

            def stage(name: str, value=None):
                def invoke(*_args):
                    observed.append(name)
                    if name == failed_stage:
                        raise RuntimeError(name)
                    return value

                return invoke

            failing = self.official.OfficialStages(
                clean_git=stage("clean_git"),
                label_guard=stage("label_guard"),
                freeze_inputs=stage("freeze_inputs", "context"),
                open_checkpoint=stage("checkpoint", "store"),
                run=stage("runner", "summary"),
                revalidate=stage("revalidate"),
                render=stage("render", {"payload": b"x"}),
                verify=stage("verify"),
                publish=stage("publish"),
            )
            with self.subTest(failed_stage=failed_stage):
                with self.assertRaisesRegex(RuntimeError, failed_stage):
                    self.official.execute_official(failing)
                self.assertEqual(observed[-1], failed_stage)
                self.assertEqual(observed, events[: events.index(failed_stage) + 1])


class SmokeCliTests(unittest.TestCase):
    def setUp(self) -> None:
        from scratch import smoke_exp417_causal_multihorizon_shadow as smoke

        self.smoke = smoke

    def test_smoke_context_is_irreversibly_nonofficial(self) -> None:
        context = self.smoke.SmokeContext(
            selected_datasets=("A", "B", "C"),
            output_dir=Path("/private/tmp/exp417-smoke"),
        )
        self.assertFalse(context.official_prediction)
        self.assertFalse(context.publication_eligible)
        self.assertFalse(context.contains_test_labels)
        self.assertFalse(context.official_checkpoint_reusable)
        self.assertEqual(context.smoke_datasets, 3)

    def test_smoke_count_is_exactly_three_and_explicit_names_are_forbidden(self) -> None:
        parser = self.smoke.build_parser()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--smoke-datasets", "2"])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            for name in ("db", "p291", "m291", "p292", "m292"):
                path = root / name
                path.write_bytes(b"x")
                files.append(path)
            argv = [
                "--database", str(files[0]),
                "--exp291-prediction", str(files[1]),
                "--exp291-manifest", str(files[2]),
                "--exp292-prediction", str(files[3]),
                "--exp292-manifest", str(files[4]),
                "--smoke-datasets", "3",
                "--output-dir", str(root / "output"),
            ]
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(argv + ["--dataset", "A"])

    def test_train_strata_selection_is_deterministic_and_uses_three_regions(self) -> None:
        rows = (
            self.smoke.TrainMetadata("Zulu-low", 10, 20, 20),
            self.smoke.TrainMetadata("Alpha-low", 10, 20, 20),
            self.smoke.TrainMetadata("Middle", 50, 100, 100),
            self.smoke.TrainMetadata("Zulu-high", 100, 500, 500),
            self.smoke.TrainMetadata("Alpha-high", 100, 500, 500),
        )
        expected = ("Alpha-high", "Alpha-low", "Middle")
        self.assertEqual(self.smoke.select_train_strata(rows, 3), expected)
        self.assertEqual(
            self.smoke.select_train_strata(tuple(reversed(rows)), 3), expected
        )

    def test_smoke_payload_cannot_be_mistaken_for_official_publication(self) -> None:
        context = self.smoke.SmokeContext(
            selected_datasets=("A", "B", "C"),
            output_dir=Path("/private/tmp/exp417-smoke"),
        )
        payload = self.smoke.render_smoke_context(
            context,
            statuses={"A": "qualified", "B": "failed", "C": "qualified"},
            backend={
                "backend": "cpu",
                "candidate_identity": True,
                "max_normalized_error": 0.0,
                "spearman": 1.0,
                "speedup": 0.67,
                "fallback": False,
                "fallback_reason": None,
            },
            elapsed_seconds=10.0,
            estimated_full_seconds=3723.3333333333335,
            estimated_full_disk_bytes=4096,
        )
        self.assertEqual(payload["experiment_id"], FROZEN_CONFIG.experiment_id)
        self.assertFalse(payload["official_prediction"])
        self.assertFalse(payload["publication_eligible"])
        self.assertFalse(payload["official_checkpoint_reusable"])
        self.assertNotIn("precision", payload)
        self.assertNotIn("recall", payload)
        self.assertNotIn("f1", payload)

    def test_smoke_write_handles_partial_writes_and_cleans_failed_output(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            real_write = __import__("os").write

            def one_byte(descriptor: int, payload: bytes) -> int:
                return real_write(descriptor, payload[:1])

            with patch.object(self.smoke.os, "write", side_effect=one_byte):
                self.smoke._write_smoke(output, {"value": 1})
            self.assertEqual(
                (output / self.smoke.SMOKE_FILE).read_text(encoding="utf-8"),
                '{"value":1}\n',
            )

        with TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            with patch.object(
                self.smoke.os, "write", side_effect=OSError("injected failure")
            ):
                with self.assertRaisesRegex(OSError, "injected"):
                    self.smoke._write_smoke(output, {"value": 1})
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
