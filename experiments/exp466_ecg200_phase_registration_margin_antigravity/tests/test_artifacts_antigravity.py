from __future__ import annotations

import dataclasses
from functools import lru_cache
from hashlib import sha256
import inspect
import json
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest

from experiments.exp466_ecg200_phase_registration_margin_antigravity.artifacts_antigravity import (
    canonical_json_bytes_antigravity,
    publish_result_antigravity,
)
from experiments.exp466_ecg200_phase_registration_margin_antigravity.evaluation_antigravity import (
    evaluate_qualification_antigravity,
)
from experiments.exp466_ecg200_phase_registration_margin_antigravity.qualification_antigravity import (
    qualify_antigravity,
)
from experiments.exp466_ecg200_phase_registration_margin_antigravity.tests.test_qualification_antigravity import (
    authoritative_records_antigravity,
)


LEAVES_ANTIGRAVITY = (
    "artifact_sha256_antigravity.json",
    "context_antigravity.json",
    "dashboard_snapshot_antigravity.json",
    "decision_antigravity.json",
    "failed_folds_antigravity.jsonl",
    "method_fold_metrics_antigravity.jsonl",
    "method_summary_antigravity.jsonl",
)
METHODS_ANTIGRAVITY = ("C0", "P9", "C9")
CHANGES_ANTIGRAVITY = (
    "level_shift",
    "gradual_drift",
    "amplitude_scaling",
    "period_change",
)


@lru_cache(maxsize=1)
def task5_inputs_antigravity():
    records = authoritative_records_antigravity()
    qualification = qualify_antigravity(records)
    decision = evaluate_qualification_antigravity(qualification)
    return records, qualification, decision


def canonical_test_bytes_antigravity(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def jsonl_rows_antigravity(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_jsonl_antigravity(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(canonical_test_bytes_antigravity(row) for row in rows))


def refresh_hashes_antigravity(root: Path) -> None:
    metrics = root / "method_fold_metrics_antigravity.jsonl"
    context_path = root / "context_antigravity.json"
    context = json.loads(context_path.read_text(encoding="utf-8"))
    if "method_fold_metrics_sha256" in context:
        context["method_fold_metrics_sha256"] = sha256(metrics.read_bytes()).hexdigest()
        context_path.write_bytes(canonical_test_bytes_antigravity(context))
    artifact_map = {
        name: sha256((root / name).read_bytes()).hexdigest()
        for name in LEAVES_ANTIGRAVITY
        if name != "artifact_sha256_antigravity.json"
    }
    (root / "artifact_sha256_antigravity.json").write_bytes(
        canonical_test_bytes_antigravity(artifact_map),
    )


class ArtifactsAntigravityTest(unittest.TestCase):
    def test_publish_creates_exact_canonical_self_excluding_seven_leaves_antigravity(
        self,
    ) -> None:
        """Break caught: publication adds a leaf or emits noncanonical/unhashed bytes."""
        self.assertEqual(
            tuple(inspect.signature(publish_result_antigravity).parameters),
            ("root", "records", "qualification", "decision"),
        )
        self.assertEqual(
            canonical_json_bytes_antigravity({"가": "한글", "z": 1}),
            b'{"z":1,"\xea\xb0\x80":"\xed\x95\x9c\xea\xb8\x80"}\n',
        )
        records, qualification, decision = task5_inputs_antigravity()
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first_antigravity"
            second = Path(temporary) / "second_antigravity"
            published = publish_result_antigravity(
                first,
                records,
                qualification,
                decision,
            )
            publish_result_antigravity(second, records, qualification, decision)

            self.assertEqual(
                tuple(sorted(path.name for path in first.iterdir())),
                LEAVES_ANTIGRAVITY,
            )
            self.assertTrue(all(path.is_file() for path in first.iterdir()))
            self.assertEqual(published.root, first)
            self.assertEqual(
                tuple(sorted(name for name, _digest in published.leaf_sha256)),
                LEAVES_ANTIGRAVITY,
            )
            for name in LEAVES_ANTIGRAVITY:
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

            for name in (
                "context_antigravity.json",
                "dashboard_snapshot_antigravity.json",
                "decision_antigravity.json",
                "artifact_sha256_antigravity.json",
            ):
                payload = (first / name).read_bytes()
                value = json.loads(payload.decode("utf-8"))
                self.assertEqual(payload, canonical_test_bytes_antigravity(value))
                self.assertTrue(payload.endswith(b"\n"))
                self.assertFalse(payload.endswith(b"\n\n"))

            for name in (
                "failed_folds_antigravity.jsonl",
                "method_fold_metrics_antigravity.jsonl",
                "method_summary_antigravity.jsonl",
            ):
                payload = (first / name).read_bytes()
                if payload:
                    self.assertTrue(payload.endswith(b"\n"))
                    self.assertFalse(payload.endswith(b"\n\n"))
                    for line in payload.splitlines():
                        self.assertEqual(
                            line + b"\n",
                            canonical_test_bytes_antigravity(json.loads(line)),
                        )

            metric_rows = jsonl_rows_antigravity(
                first / "method_fold_metrics_antigravity.jsonl",
            )
            self.assertEqual(
                tuple((row["seed"], row["fold_index"], row["method"]) for row in metric_rows),
                tuple(
                    (seed, fold_index, method)
                    for seed in (20260721, 20261721)
                    for fold_index in range(5)
                    for method in METHODS_ANTIGRAVITY
                ),
            )
            self.assertTrue(
                all(tuple(row["change_order"]) == CHANGES_ANTIGRAVITY for row in metric_rows),
            )
            summary_rows = jsonl_rows_antigravity(
                first / "method_summary_antigravity.jsonl",
            )
            self.assertEqual(tuple(row["method"] for row in summary_rows), METHODS_ANTIGRAVITY)

            artifact_map = json.loads(
                (first / "artifact_sha256_antigravity.json").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                tuple(sorted(artifact_map)),
                tuple(name for name in LEAVES_ANTIGRAVITY if name != "artifact_sha256_antigravity.json"),
            )
            self.assertNotIn("artifact_sha256_antigravity.json", artifact_map)
            for name, digest in artifact_map.items():
                self.assertEqual(digest, sha256((first / name).read_bytes()).hexdigest())

    def test_existing_root_and_reordered_records_are_rejected_before_write_antigravity(
        self,
    ) -> None:
        """Break caught: create-only publication overwrites a root or accepts hidden order."""
        records, qualification, decision = task5_inputs_antigravity()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "existing_antigravity"
            root.mkdir()
            sentinel = root / "sentinel_antigravity.txt"
            sentinel.write_bytes(b"keep-antigravity\n")
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            with self.assertRaises(ValueError):
                publish_result_antigravity(root, records, qualification, decision)
            self.assertEqual(
                {path.name: path.read_bytes() for path in root.iterdir()},
                before,
            )

            reordered = MappingProxyType(dict(reversed(tuple(records.items()))))
            absent = Path(temporary) / "reordered_antigravity"
            with self.assertRaises(ValueError):
                publish_result_antigravity(absent, reordered, qualification, decision)
            self.assertFalse(absent.exists())

    def test_publish_rejects_swapped_blocked_fold_indices_before_write_antigravity(
        self,
    ) -> None:
        """Break caught: publisher accepts OOF-complete but non-preregistered fold blocks."""
        records, qualification, decision = task5_inputs_antigravity()
        swapped = dict(records)
        for seed in (20260721, 20261721):
            first = records[(seed, 0)]
            second = records[(seed, 1)]
            swapped[(seed, 0)] = dataclasses.replace(
                first,
                input_fit_indices=second.input_fit_indices,
                input_query_indices=second.input_query_indices,
            )
            swapped[(seed, 1)] = dataclasses.replace(
                second,
                input_fit_indices=first.input_fit_indices,
                input_query_indices=first.input_query_indices,
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "swapped_blocks_antigravity"
            with self.assertRaisesRegex(ValueError, "blocked fold"):
                publish_result_antigravity(
                    root,
                    MappingProxyType(swapped),
                    qualification,
                    decision,
                )
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
