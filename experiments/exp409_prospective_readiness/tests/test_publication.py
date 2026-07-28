from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from experiments.exp403_408.feature_registry import FEATURE_SOURCES
from experiments.exp403_408.synthetic_anomaly import MODEL_NAMES
from experiments.exp409_prospective_readiness.config import FROZEN_CONFIG
from experiments.exp409_prospective_readiness.contracts import (
    DatasetReadinessResult,
    ReadinessBatch,
    ShadowCandidate,
    SourceStatus,
)
from experiments.exp409_prospective_readiness.provenance import (
    build_canonical_provenance,
    build_runtime_envelope,
)
from experiments.exp409_prospective_readiness.publication import (
    candidate_payload_sha256,
    publish_batch,
    write_failure_record,
)


_DIGEST = "a" * 64


def _source_names() -> tuple[str, ...]:
    return tuple(
        sorted(
            {source.name for source in FEATURE_SOURCES}
            | {"tri_view"}
            | {f"synthetic_{name}" for name in MODEL_NAMES}
        )
    )


def _batch_with_count(count: int) -> ReadinessBatch:
    candidates = tuple(
        ShadowCandidate(
            dataset="sample",
            candidate_index=index,
            variant=FROZEN_CONFIG.variant,
            alpha=FROZEN_CONFIG.alpha,
            supporting_components=(("a",), ("b",), ("c",)),
            supporting_source_markers=(("a",), ("b",), ("c",)),
        )
        for index in range(count)
    )
    names = _source_names()
    return ReadinessBatch(
        (
            DatasetReadinessResult(
                dataset="sample",
                train_size=50,
                test_size=max(1, count),
                hard_indices=(),
                standard_indices=(),
                priority_indices=(),
                source_statuses=tuple(
                    SourceStatus(name, "available", "") for name in names
                ),
                candidates=candidates,
                source_output_sha256={name: _DIGEST for name in names},
                status="ready",
            ),
        )
    )


def _provenance(count: int) -> dict[str, object]:
    batch = _batch_with_count(count)
    return build_canonical_provenance(
        git_commit="b" * 40,
        code_sha256={"publication.py": _DIGEST},
        database_snapshot_sha256={"univariate_ts.db": _DIGEST},
        lane_artifact_sha256={"lanes": _DIGEST},
        source_artifact_sha256={"sources": _DIGEST},
        dataset_values_sha256={"sample": _DIGEST},
        source_output_sha256={name: _DIGEST for name in _source_names()},
        upstream_artifact_sha256={"manifest": _DIGEST},
        candidate_sha256=candidate_payload_sha256(batch),
    )


def _runtime() -> dict[str, object]:
    return build_runtime_envelope("run-1", "2026-07-28T00:00:00Z")


class PublicationTests(unittest.TestCase):
    def test_budget_one_hundred_publishes_complete_artifact_set(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "publication"

            hashes = publish_batch(
                _batch_with_count(100), output, _provenance(100), _runtime()
            )

            self.assertTrue(output.is_dir())
            self.assertEqual(set(hashes), {path.name for path in output.iterdir()})
            candidates = (output / "exp409_shadow_candidates.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(len(candidates), 100)
            summary = json.loads(
                (output / "exp409_readiness_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["performance_evaluation"], "not_evaluated")
            self.assertFalse(summary["prospective_validated"])
            self.assertFalse(summary["deployment_eligible"])
            self.assertFalse(summary["operational_policy_changed"])

    def test_budget_overflow_writes_no_candidate_artifact(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "publication"

            with self.assertRaisesRegex(ValueError, "budget"):
                publish_batch(
                    _batch_with_count(101), output, _provenance(101), _runtime()
                )

            self.assertFalse(output.exists())
            self.assertFalse(
                (output / "exp409_shadow_candidates.jsonl").exists()
            )

    def test_existing_output_is_never_overwritten(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "publication"
            output.mkdir()

            with self.assertRaisesRegex(FileExistsError, "output"):
                publish_batch(
                    _batch_with_count(1), output, _provenance(1), _runtime()
                )

    def test_staging_failure_leaves_no_publication_or_staging_directory(self) -> None:
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            output = parent / "publication"
            with patch(
                "experiments.exp409_prospective_readiness.publication._write_staged_files",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    publish_batch(
                        _batch_with_count(1),
                        output,
                        _provenance(1),
                        _runtime(),
                    )

            self.assertFalse(output.exists())
            self.assertEqual(
                list(parent.glob(".publication.*.staging")),
                [],
            )

    def test_failure_record_is_outside_publication_and_has_no_candidates(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "publication"

            failure = write_failure_record(
                output, status="blocked", reason="review budget exceeded"
            )

            self.assertEqual(failure, Path(directory) / "publication.failure.json")
            self.assertFalse(output.exists())
            payload = json.loads(failure.read_text(encoding="utf-8"))
            self.assertEqual(
                set(payload), {"experiment_id", "status", "reason"}
            )
            self.assertNotIn("candidate", failure.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
