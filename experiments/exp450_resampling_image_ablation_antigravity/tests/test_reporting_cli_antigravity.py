"""Task 8 report and closed-command contracts on fixture-only evaluation data."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zlib

from experiments.exp450_resampling_image_ablation_antigravity.config_antigravity import (
    EVALUATION_ROOT,
    OFFICIAL_ROOT,
    REPORT_ROOT,
    SMOKE_ROOT,
)
from experiments.exp450_resampling_image_ablation_antigravity.evaluation_antigravity import (
    Exp450EvaluationAntigravity,
    METHOD_ORDER,
    curve_tables_exp450_antigravity,
    metric_tables_exp450_antigravity,
)
from experiments.exp450_resampling_image_ablation_antigravity.tests.test_evaluation_antigravity import (
    ALL_SCORES,
    LABELS,
)
from experiments.exp450_resampling_image_ablation_antigravity.tests.test_runner_verifier_antigravity import (
    CONDITIONS as TASK6_CONDITIONS,
    _catalog as task6_catalog,
    _context as task6_context,
    _raw as task6_raw,
    _result as task6_result,
    _score as task6_score,
)


REPOSITORY = Path(__file__).resolve().parents[3]
SCRIPT = REPOSITORY / "scratch" / "run_exp450_resampling_image_ablation_antigravity.py"
PYTHON = sys.executable


def _evaluation() -> Exp450EvaluationAntigravity:
    return Exp450EvaluationAntigravity(
        manifests={},
        metric_tables=metric_tables_exp450_antigravity(LABELS, ALL_SCORES),
        curve_tables=curve_tables_exp450_antigravity(LABELS, ALL_SCORES),
    )


class Exp450ReportingCliTests(unittest.TestCase):
    def test_resume_eta_uses_only_current_invocation_attempts(self) -> None:
        """Historical checkpoint counts must not dilute a resumed run's throughput ETA."""
        import scratch.run_exp450_resampling_image_ablation_antigravity as cli
        import experiments.exp450_resampling_image_ablation_antigravity.runner_antigravity as runner
        from experiments.exp450_resampling_image_ablation_antigravity.artifacts_antigravity import (
            append_dataset_condition_failure_antigravity,
            write_dataset_condition_success_antigravity,
        )

        with TemporaryDirectory(prefix="exp450_resume_eta_antigravity_") as directory:
            root = Path(directory) / "run_antigravity"
            raw = task6_raw("resume_eta_antigravity")
            catalog = task6_catalog((raw,))
            context = task6_context(root, catalog)
            conditions = {condition.condition_id: condition for condition in TASK6_CONDITIONS}
            for condition_id in ("A", "B"):
                write_dataset_condition_success_antigravity(root, context, catalog, raw, task6_result(raw, conditions[condition_id]))
            append_dataset_condition_failure_antigravity(root, context, catalog, raw, conditions["C"], 1, RuntimeError("fixture wave one"))
            emitted: list[dict[str, object]] = []
            def resume_schedule(root, context, catalog, raw, backend, progress):
                progress({"dataset": raw["resume_eta_antigravity"].dataset, "condition": "D", "wave": 1, "attempt": 1, "status": "success"})
                progress({"dataset": raw["resume_eta_antigravity"].dataset, "condition": "C", "wave": 2, "attempt": 2, "status": "success"})
                return {}
            with patch.object(cli, "_science_inputs", return_value=(root, context, catalog, {raw.dataset: raw}, context.backend)), \
                 patch.object(runner, "execute_local_schedule_antigravity", side_effect=resume_schedule), \
                 patch.object(runner, "compose_all_conditions_antigravity", return_value={}), \
                 patch.object(runner, "freeze_all_conditions_antigravity"), \
                 patch.object(cli, "monotonic", side_effect=(100.0, 110.0, 130.0)), \
                 patch.object(cli, "_emit", side_effect=emitted.append):
                self.assertEqual(cli._run("smoke"), 0)
            self.assertEqual(len(emitted), 2)
            self.assertEqual((emitted[0]["completed"], emitted[0]["failed"], emitted[0]["remaining"]), (3, 1, 1))
            self.assertEqual(emitted[0]["eta_seconds"], 10.0)
            self.assertEqual((emitted[1]["completed"], emitted[1]["failed"], emitted[1]["remaining"], emitted[1]["eta_seconds"]), (4, 1, 0, 0.0))

    def test_run_resume_progress_starts_from_verified_checkpoint_inventory(self) -> None:
        """A reduced resumed run counts durable successes/failures without replaying them."""
        import scratch.run_exp450_resampling_image_ablation_antigravity as cli
        import experiments.exp450_resampling_image_ablation_antigravity.runner_antigravity as runner
        from experiments.exp450_resampling_image_ablation_antigravity.artifacts_antigravity import (
            append_dataset_condition_failure_antigravity,
            write_dataset_condition_success_antigravity,
        )

        with TemporaryDirectory(prefix="exp450_resume_progress_antigravity_") as directory:
            root = Path(directory) / "run_antigravity"
            raw = task6_raw("resume_progress_antigravity")
            catalog = task6_catalog((raw,))
            context = task6_context(root, catalog)
            conditions = {condition.condition_id: condition for condition in TASK6_CONDITIONS}
            for condition_id in ("A", "B", "D"):
                write_dataset_condition_success_antigravity(root, context, catalog, raw, task6_result(raw, conditions[condition_id]))
            append_dataset_condition_failure_antigravity(root, context, catalog, raw, conditions["C"], 1, RuntimeError("fixture wave one"))
            emitted: list[dict[str, object]] = []
            def resume_schedule(root, context, catalog, raw, backend, progress):
                self.assertEqual(len(emitted), 0)
                progress({"dataset": raw["resume_progress_antigravity"].dataset, "condition": "C", "wave": 2, "attempt": 2, "status": "success"})
                return {}
            with patch.object(cli, "_science_inputs", return_value=(root, context, catalog, {raw.dataset: raw}, context.backend)), \
                 patch.object(runner, "execute_local_schedule_antigravity", side_effect=resume_schedule), \
                 patch.object(runner, "compose_all_conditions_antigravity", return_value={}), \
                 patch.object(runner, "freeze_all_conditions_antigravity"), \
                 patch.object(cli, "_emit", side_effect=emitted.append):
                self.assertEqual(cli._run("smoke"), 0)
            self.assertEqual(len(emitted), 1)
            final = emitted[0]
            self.assertEqual((final["completed"], final["failed"], final["remaining"], final.get("total")), (4, 1, 0, 4))
            self.assertGreaterEqual(float(final["elapsed_seconds"]), 0.0)
            self.assertGreaterEqual(float(final["eta_seconds"]), 0.0)

    def test_svg_keeps_full_method_legend_when_a_curve_has_no_points(self) -> None:
        """Legend order is frozen independently from whether a method has drawable points."""
        from experiments.exp450_resampling_image_ablation_antigravity.reporting_antigravity import (
            _curve_svg,
            canonicalize_exp450_json_antigravity,
        )

        curves = canonicalize_exp450_json_antigravity(_evaluation().curve_tables)
        self.assertIsInstance(curves, dict)
        missing = "drocc_1d"
        for curve in curves.values():
            curve["rows"] = [row for row in curve["rows"] if row["method"] != missing]
        svg = _curve_svg(curves)
        self.assertEqual(svg.count(f">{missing}</text>"), 2)
        self.assertEqual(svg.count(f'data-legend-method="{missing}"'), 2)
        self.assertNotIn(f"<title>{missing}</title>", svg)

    def test_png_visibly_renders_titles_axes_and_method_legends(self) -> None:
        """The raster figure has text ink outside plotted curves for both panels."""
        from experiments.exp450_resampling_image_ablation_antigravity.reporting_antigravity import (
            write_exp450_report_antigravity,
        )

        with TemporaryDirectory(prefix="exp450_png_labels_antigravity_") as directory:
            artifact = write_exp450_report_antigravity(_evaluation(), Path(directory))
            png = artifact.png_path.read_bytes()
        offset, chunks = 8, []
        while offset < len(png):
            size = int.from_bytes(png[offset:offset + 4], "big")
            kind = png[offset + 4:offset + 8]
            chunks.append((kind, png[offset + 8:offset + 8 + size]))
            offset += 12 + size
        raw = zlib.decompress(b"".join(payload for kind, payload in chunks if kind == b"IDAT"))
        row_width = 1200 * 3 + 1
        def has_ink(left: int, top: int, right: int, bottom: int) -> bool:
            return any(
                raw[row * row_width + 1 + column * 3:row * row_width + 4 + column * 3] != b"\xff\xff\xff"
                for row in range(top, bottom) for column in range(left, right)
            )
        # Titles, x-axis labels, and the ten-row legend region all sit outside plots.
        self.assertTrue(has_ink(70, 18, 270, 55))
        self.assertTrue(has_ink(200, 520, 430, 555))
        self.assertTrue(has_ink(75, 76, 230, 230))

    def test_report_writes_canonical_tables_figures_and_exact_curve_axes(self) -> None:
        """The paper artifact retains all ten methods and unswapped coordinates."""
        from experiments.exp450_resampling_image_ablation_antigravity.reporting_antigravity import (
            write_exp450_report_antigravity,
        )

        with TemporaryDirectory(prefix="exp450_report_antigravity_") as directory:
            artifact = write_exp450_report_antigravity(_evaluation(), Path(directory))
            self.assertTrue(artifact.metric_csv_path.is_file())
            self.assertTrue(artifact.metric_json_path.is_file())
            self.assertTrue(artifact.curve_csv_path.is_file())
            self.assertTrue(artifact.curve_json_path.is_file())
            self.assertTrue(artifact.paired_csv_path.is_file())
            self.assertTrue(artifact.png_path.is_file())
            self.assertTrue(artifact.svg_path.is_file())
            html = artifact.html_path.read_text(encoding="utf-8")
            self.assertIn("False-positive rate", html)
            self.assertIn("True-positive rate", html)
            self.assertIn("Recall", html)
            self.assertIn("Precision", html)
            self.assertIn("no automatic promotion", html)
            for method in METHOD_ORDER:
                self.assertIn(method, html)
            curve_payload = json.loads(artifact.curve_json_path.read_text(encoding="utf-8"))
            self.assertEqual(curve_payload["roc"]["x_name"], "false_positive_rate")
            self.assertEqual(curve_payload["roc"]["y_name"], "true_positive_rate")
            self.assertEqual(curve_payload["precision_recall"]["x_name"], "recall")
            self.assertEqual(curve_payload["precision_recall"]["y_name"], "precision")
            self.assertIn("Macro AP / AUROC", html)
            self.assertIn("Micro AP / AUROC", html)
            self.assertIn("Paired differences", html)
            self.assertIn('id="exp450-curves"', html)
            embedded = html.split('id="exp450-curves" type="application/json">', 1)[1].split("</script>", 1)[0]
            self.assertEqual(json.loads(embedded)["roc"]["x_name"], "false_positive_rate")
            svg = artifact.svg_path.read_text(encoding="utf-8")
            for method in METHOD_ORDER:
                self.assertIn(method, svg)
            self.assertIn("False-positive rate", svg)

    def test_status_and_verify_are_scorer_and_label_free_fresh_processes(self) -> None:
        """Read-only inspection must not load scientific scoring or label capability."""
        for command in (("status",), ("readiness", "--mode", "smoke")):
            result = subprocess.run(
                [PYTHON, "-B", str(SCRIPT), *command],
                check=True,
                capture_output=True,
                text=True,
                cwd=REPOSITORY,
            )
            payload = json.loads(result.stdout)
            self.assertEqual(payload["experiment"], "Exp450")
            self.assertEqual(payload["science_modules_loaded"], [])
        self.assertTrue(all(not root.exists() for root in (SMOKE_ROOT, OFFICIAL_ROOT, EVALUATION_ROOT, REPORT_ROOT)))
    def test_progress_schema_and_evaluation_gate_do_not_start_science(self) -> None:
        """Progress is a stable dataset-condition record and evaluation is gated first."""
        from scratch.run_exp450_resampling_image_ablation_antigravity import (
            progress_record_antigravity,
        )

        record = progress_record_antigravity(
            dataset="fixture_antigravity", condition="A", status="success",
            elapsed_seconds=2.0, completed=1, total=4,
        )
        self.assertEqual(
            set(record),
            {"experiment", "dataset", "condition", "status", "elapsed_seconds", "eta_seconds", "completed", "total"},
        )
        self.assertEqual(record["eta_seconds"], 6.0)
        result = subprocess.run(
            [PYTHON, "-B", str(SCRIPT), "evaluate"],
            capture_output=True,
            text=True,
            cwd=REPOSITORY,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("all four full score manifests", result.stderr)
        self.assertTrue(all(not root.exists() for root in (SMOKE_ROOT, OFFICIAL_ROOT, EVALUATION_ROOT, REPORT_ROOT)))

    def test_evaluation_artifact_recursively_canonicalizes_real_fixture_values(self) -> None:
        """Mapping proxies, tuples, arrays, and numpy scalars must round-trip canonically."""
        from scratch.run_exp450_resampling_image_ablation_antigravity import (
            load_evaluation_artifact_antigravity,
            write_evaluation_artifact_antigravity,
        )

        with TemporaryDirectory(prefix="exp450_eval_artifact_antigravity_") as directory:
            path = write_evaluation_artifact_antigravity(_evaluation(), Path(directory))
            self.assertTrue(path.read_bytes().endswith(b"\n"))
            loaded = load_evaluation_artifact_antigravity(path)
            self.assertEqual(loaded.metric_tables.method_order, METHOD_ORDER)
            self.assertEqual(loaded.curve_tables["roc"]["x_name"], "false_positive_rate")

    def test_verify_mode_invokes_v2_verifier_and_rejects_corrupt_manifest(self) -> None:
        """Presence is insufficient: every selected-mode manifest is verifier-backed."""
        import scratch.run_exp450_resampling_image_ablation_antigravity as cli
        from experiments.exp450_resampling_image_ablation_antigravity.artifacts_antigravity import (
            write_dataset_condition_success_antigravity,
        )
        from experiments.exp450_resampling_image_ablation_antigravity.runner_antigravity import (
            freeze_all_conditions_antigravity,
        )

        with TemporaryDirectory(prefix="exp450_verify_mode_antigravity_") as directory:
            root = Path(directory) / "run_antigravity"
            raw = task6_raw("verify_mode_antigravity")
            catalog = task6_catalog((raw,))
            context = task6_context(root, catalog)
            for condition in TASK6_CONDITIONS:
                write_dataset_condition_success_antigravity(root, context, catalog, raw, task6_result(raw, condition))
            freeze_all_conditions_antigravity({condition.condition_id: task6_score(raw, condition) for condition in TASK6_CONDITIONS}, root, context, catalog)
            payload = cli._verify("smoke", verification_inputs=lambda mode: (root, context, catalog))
            self.assertEqual(tuple(payload["verified_manifests"]), ("A", "B", "C", "D"))
            manifest = root / "score_roots_antigravity" / "D_antigravity" / "condition_score_manifest_antigravity.json"
            manifest.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                cli._verify("smoke", verification_inputs=lambda mode: (root, context, catalog))

    def test_create_only_writer_completes_injected_short_writes(self) -> None:
        """A partial os.write must not truncate a large curve artifact."""
        import experiments.exp450_resampling_image_ablation_antigravity.reporting_antigravity as reporting

        original_write = reporting.os.write
        def short_write(descriptor: int, data: bytes) -> int:
            return original_write(descriptor, data[:17])

        with TemporaryDirectory(prefix="exp450_short_write_antigravity_") as directory:
            with patch.object(reporting.os, "write", side_effect=short_write):
                artifact = reporting.write_exp450_report_antigravity(_evaluation(), Path(directory))
            self.assertGreater(artifact.curve_json_path.stat().st_size, 17)
            json.loads(artifact.curve_json_path.read_text(encoding="utf-8"))

    def test_live_progress_follows_durable_checkpoint_before_next_coordinate(self) -> None:
        """The callback observes each persisted record before the next MPS score begins."""
        import experiments.exp450_resampling_image_ablation_antigravity.runner_antigravity as runner
        from experiments.exp450_resampling_image_ablation_antigravity.contracts_antigravity import (
            Exp450BackendDecision,
            Exp450RunContext,
        )

        with TemporaryDirectory(prefix="exp450_live_progress_antigravity_") as directory:
            root = Path(directory) / "run_antigravity"
            raw = task6_raw("live_progress_antigravity")
            catalog = task6_catalog((raw,))
            backend = Exp450BackendDecision("mps", "fixture", True, False, 2, 1)
            context = Exp450RunContext("smoke", root, "a" * 64, catalog["catalog_sha256"], {"runner_antigravity.py": "b" * 64}, backend, "c" * 64)
            events: list[dict[str, object]] = []
            observed_next: list[bool] = []
            def score(values, condition, active_backend):
                observed_next.append(bool(events))
                return task6_result(values, condition)
            def progress(event: dict[str, object]) -> None:
                token = __import__("hashlib").sha256(str(event["dataset"]).encode()).hexdigest()
                path = root / "checkpoints_antigravity" / f"{event['condition']}_antigravity" / f"{token}_antigravity" / "dataset_condition_success_antigravity.json"
                self.assertTrue(path.is_file())
                events.append(event)
            conditions = {condition.condition_id: condition for condition in TASK6_CONDITIONS}
            with patch.object(runner, "run_dataset_local_exp450_antigravity", side_effect=score):
                runner._run_mps_wave(((raw.dataset, "C"), (raw.dataset, "D")), root=root, context=context, catalog=catalog, raw={raw.dataset: raw}, backend=backend, by_condition=conditions, wave=1, progress=progress)
            self.assertEqual([event["condition"] for event in events], ["C", "D"])
            self.assertEqual([event["attempt"] for event in events], [1, 1])
            self.assertEqual(observed_next, [False, True])

    def test_live_failure_event_follows_durable_failure_ledger(self) -> None:
        """A terminal attempt is observable immediately after its failure record is durable."""
        import experiments.exp450_resampling_image_ablation_antigravity.runner_antigravity as runner
        from experiments.exp450_resampling_image_ablation_antigravity.contracts_antigravity import Exp450BackendDecision, Exp450RunContext

        with TemporaryDirectory(prefix="exp450_live_failure_antigravity_") as directory:
            root = Path(directory) / "run_antigravity"
            raw = task6_raw("live_failure_antigravity")
            catalog = task6_catalog((raw,))
            backend = Exp450BackendDecision("mps", "fixture", True, False, 2, 1)
            context = Exp450RunContext("smoke", root, "a" * 64, catalog["catalog_sha256"], {"runner_antigravity.py": "b" * 64}, backend, "c" * 64)
            events: list[dict[str, object]] = []
            def progress(event: dict[str, object]) -> None:
                ledger = root / "checkpoints_antigravity" / "failures_antigravity"
                self.assertTrue(ledger.is_dir())
                self.assertTrue(any(ledger.iterdir()))
                events.append(event)
            conditions = {condition.condition_id: condition for condition in TASK6_CONDITIONS}
            with patch.object(runner, "run_dataset_local_exp450_antigravity", side_effect=RuntimeError("fixture failure")):
                runner._run_mps_wave(((raw.dataset, "C"),), root=root, context=context, catalog=catalog, raw={raw.dataset: raw}, backend=backend, by_condition=conditions, wave=1, progress=progress)
                runner._run_mps_wave(((raw.dataset, "C"),), root=root, context=context, catalog=catalog, raw={raw.dataset: raw}, backend=backend, by_condition=conditions, wave=2, progress=progress)
            self.assertEqual(events[-1], {"dataset": raw.dataset, "condition": "C", "wave": 2, "attempt": 2, "status": "failure"})
