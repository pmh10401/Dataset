from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from experiments.exp403_408.contracts import DatasetInput
from experiments.exp403_408.feature_registry import FEATURE_SOURCES
from experiments.exp403_408.synthetic_anomaly import MODEL_NAMES
from experiments.exp409_prospective_readiness.provenance import (
    build_canonical_provenance,
    build_runtime_envelope,
    dataset_values_sha256,
    provenance_closure_sha256,
    require_clean_git,
    validate_canonical_provenance,
)


_DIGEST = "a" * 64


def _source_names() -> set[str]:
    return (
        {source.name for source in FEATURE_SOURCES}
        | {"tri_view"}
        | {f"synthetic_{name}" for name in MODEL_NAMES}
    )


def _valid_provenance() -> dict[str, object]:
    return build_canonical_provenance(
        git_commit="b" * 40,
        code_sha256={"sources.py": _DIGEST},
        database_snapshot_sha256={"univariate_ts.db": _DIGEST},
        lane_artifact_sha256={"exp291": _DIGEST, "exp292": _DIGEST},
        source_artifact_sha256={"feature-cache": _DIGEST},
        dataset_values_sha256={"sample": _DIGEST},
        source_output_sha256={name: _DIGEST for name in _source_names()},
        upstream_artifact_sha256={"manifest": _DIGEST},
        candidate_sha256=_DIGEST,
    )


class ProvenanceTests(unittest.TestCase):
    def test_dirty_git_blocks_publication(self) -> None:
        with patch(
            "experiments.exp409_prospective_readiness.provenance._git_status",
            return_value=" M file.py",
        ):
            with self.assertRaisesRegex(ValueError, "dirty"):
                require_clean_git(Path("/repo"))

    def test_clean_git_returns_commit(self) -> None:
        with (
            patch(
                "experiments.exp409_prospective_readiness.provenance._git_status",
                return_value="",
            ),
            patch(
                "experiments.exp409_prospective_readiness.provenance._git_commit",
                return_value="c" * 40,
            ),
        ):
            self.assertEqual(require_clean_git(Path("/repo")), "c" * 40)

    def test_dataset_value_hash_binds_field_names_shapes_and_values(self) -> None:
        base = DatasetInput(
            dataset="sample",
            train_normal=np.arange(12, dtype=float).reshape(3, 4),
            test_values=np.arange(8, dtype=float).reshape(2, 4),
            target_length=4,
        )
        changed = DatasetInput(
            dataset="sample",
            train_normal=np.arange(12, dtype=float).reshape(3, 4),
            test_values=np.arange(8, dtype=float).reshape(4, 2),
            target_length=4,
        )

        first = dataset_values_sha256(base)

        self.assertEqual(first, dataset_values_sha256(base))
        self.assertNotEqual(first, dataset_values_sha256(changed))
        self.assertEqual(len(first), 64)

    def test_missing_source_hash_is_rejected(self) -> None:
        payload = _valid_provenance()
        del payload["source_output_sha256"]["tri_view"]  # type: ignore[index]

        with self.assertRaisesRegex(ValueError, "source provenance"):
            validate_canonical_provenance(payload)

    def test_rejects_extra_top_level_field(self) -> None:
        payload = _valid_provenance()
        payload["started_at"] = "2026-07-28T00:00:00Z"

        with self.assertRaisesRegex(ValueError, "top-level"):
            validate_canonical_provenance(payload)

    def test_rejects_invalid_digest_or_gate(self) -> None:
        bad_digest = _valid_provenance()
        bad_digest["candidate_sha256"] = "ABC"
        with self.assertRaisesRegex(ValueError, "digest"):
            validate_canonical_provenance(bad_digest)

        bad_gate = _valid_provenance()
        bad_gate["gates"]["determinism"] = "pending"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "gates"):
            validate_canonical_provenance(bad_gate)

    def test_runtime_envelope_does_not_change_closure(self) -> None:
        payload = _valid_provenance()
        before = provenance_closure_sha256(payload)

        first_runtime = build_runtime_envelope("run-1", "2026-07-28T00:00:00Z")
        second_runtime = build_runtime_envelope("run-2", "2026-07-28T00:01:00Z")

        self.assertNotEqual(first_runtime["runtime_envelope_sha256"], second_runtime["runtime_envelope_sha256"])
        self.assertEqual(before, provenance_closure_sha256(payload))

    def test_closure_changes_with_canonical_input(self) -> None:
        first = _valid_provenance()
        second = deepcopy(first)
        second["candidate_sha256"] = "c" * 64

        self.assertNotEqual(
            provenance_closure_sha256(first),
            provenance_closure_sha256(second),
        )


if __name__ == "__main__":
    unittest.main()
