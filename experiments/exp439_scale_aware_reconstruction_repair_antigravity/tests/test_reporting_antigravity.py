from pathlib import Path
import json
import tempfile
import unittest

from experiments.exp439_scale_aware_reconstruction_repair_antigravity.contracts_antigravity import BackendDecision
from experiments.exp439_scale_aware_reconstruction_repair_antigravity.reporting_antigravity import (
    PUBLICATION_FILES_ANTIGRAVITY,
    inspect_publication_antigravity,
    publish_exp439_antigravity,
)
from experiments.exp439_scale_aware_reconstruction_repair_antigravity.tests.test_runtime_antigravity import (
    _BoundSession,
    _result,
    _run_antigravity,
    _thread_executor,
)


class Exp439ReportingAntigravityTests(unittest.TestCase):
    def _completed_roots(self, base: Path) -> tuple[Path, Path]:
        output = base / "output_antigravity"
        checkpoint = base / "checkpoint_antigravity"
        _run_antigravity(
            mode="new",
            output_root=output,
            checkpoint_root=checkpoint,
            reviewed_commit="a" * 40,
            decision=BackendDecision("cpu", None, (), "synthetic CPU fixture"),
            input_session_factory=lambda: _BoundSession(("A", "B", "C")),
            workers=2,
            executor_factory=_thread_executor,
            score_task=lambda task: _result(task.dataset),
        )
        return output, checkpoint

    def test_publish_and_replay_are_review_only_hash_bound_and_separate_channel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            output, checkpoint = self._completed_roots(base)
            frozen = publish_exp439_antigravity(
                output, checkpoint, expected_datasets_antigravity=3
            )
            self.assertEqual(
                {path.name for path in frozen.publication.iterdir()},
                set(PUBLICATION_FILES_ANTIGRAVITY),
            )
            manifest = dict(frozen.manifest)
            self.assertTrue(manifest["review_only"])
            self.assertFalse(manifest["operational_alert_eligible"])
            self.assertEqual(manifest["prediction_label_access_count"], 0)
            rows = [
                json.loads(line)
                for line in (frozen.publication / "prediction_antigravity.jsonl").read_text().splitlines()
            ]
            self.assertEqual(tuple(rows[0]["channels"]), (
                "high_frequency", "low_frequency", "amplitude_complexity"
            ))
            self.assertNotIn("combined", json.dumps(rows).casefold())
            replay = inspect_publication_antigravity(
                frozen.publication, output, checkpoint, expected_datasets_antigravity=3
            )
            self.assertEqual(replay.prediction_sha256, frozen.prediction_sha256)

    def test_publication_is_no_clobber_and_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            output, checkpoint = self._completed_roots(base)
            frozen = publish_exp439_antigravity(output, checkpoint, expected_datasets_antigravity=3)
            before = {path.name: path.read_bytes() for path in frozen.publication.iterdir()}
            with self.assertRaisesRegex(ValueError, "already exists"):
                publish_exp439_antigravity(output, checkpoint, expected_datasets_antigravity=3)
            self.assertEqual(before, {path.name: path.read_bytes() for path in frozen.publication.iterdir()})
            prediction = frozen.publication / "prediction_antigravity.jsonl"
            prediction.write_bytes(prediction.read_bytes() + b"{}\n")
            with self.assertRaisesRegex(ValueError, "hash"):
                inspect_publication_antigravity(
                    frozen.publication, output, checkpoint, expected_datasets_antigravity=3
                )


if __name__ == "__main__":
    unittest.main()
