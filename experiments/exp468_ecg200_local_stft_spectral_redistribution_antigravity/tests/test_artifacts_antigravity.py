"""Task 5 publication contract tests, written before production modules."""

from __future__ import annotations

import dataclasses
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
import tempfile
import unittest

import numpy as np

from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.artifacts_antigravity import (
    LEAF_NAMES_ANTIGRAVITY,
    _HASHED_LEAF_NAMES_ANTIGRAVITY,
    _canonical_json_bytes_antigravity,
    _dashboard_snapshot_payload_antigravity,
    _write_jsonl_antigravity,
    PublishedResultAntigravity,
    publish_result_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.contracts_antigravity import (
    Exp468FoldResultAntigravity,
    Exp468QualificationAntigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.evaluation_antigravity import (
    evaluate_qualification_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.predecessor_antigravity import (
    capture_exp465_authority_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.qualification_antigravity import (
    blocked_fold_indices_antigravity,
    qualify_antigravity,
)
from experiments.exp468_ecg200_local_stft_spectral_redistribution_antigravity.scoring_antigravity import (
    FoldMethodScoresAntigravity,
    score_fold_methods_antigravity,
)


@lru_cache(maxsize=1)
def records_fixture_antigravity() -> MappingProxyType:
    x = np.arange(140, dtype=np.float64)
    base = 0.01 * x + np.sin(2.0 * np.pi * x / 21.0) + 0.3 * np.cos(2.0 * np.pi * x / 9.0)
    train = np.asarray(
        base[None, :] + np.random.default_rng(46850).normal(scale=0.01, size=(69, 140)),
        dtype=np.float64,
        order="C",
    )
    train.setflags(write=False)
    authority = capture_exp465_authority_antigravity()
    records: dict[tuple[int, int], FoldMethodScoresAntigravity] = {}
    for seed in (20260721, 20261721):
        for fold_index, (fit, query) in enumerate(blocked_fold_indices_antigravity()):
            records[(seed, fold_index)] = score_fold_methods_antigravity(
                train,
                fit,
                query,
                seed=seed,
                fold_index=fold_index,
                authority=authority,
            )
    return MappingProxyType(records)


@lru_cache(maxsize=1)
def qualification_fixture_antigravity():
    return qualify_antigravity(records_fixture_antigravity())


@lru_cache(maxsize=1)
def decision_fixture_antigravity():
    return evaluate_qualification_antigravity(qualification_fixture_antigravity())


def scalar_qualification_fixture_antigravity(
    *,
    passed: bool,
) -> Exp468QualificationAntigravity:
    rows: list[Exp468FoldResultAntigravity] = []
    for seed in (20260721, 20261721):
        for fold_index in range(5):
            c0 = (0.70, 0.70, 0.70, 0.85)
            c10 = (0.70, 0.70, 0.70, 0.95 if passed else 0.80)
            rows.append(
                Exp468FoldResultAntigravity(
                    seed=seed,
                    fold_index=fold_index,
                    c0_per_type_auc=c0,
                    c10_per_type_auc=c10,
                    period_confirmations=1,
                    normal_confirmations=0,
                    all_scores_finite=True,
                    passed=passed,
                    failure_reasons=() if passed else (
                        "period_auc_below_0.85",
                        "period_auc_not_strictly_above_c0",
                    ),
                ),
            )
    return Exp468QualificationAntigravity(
        fold_results=tuple(rows),
        normal_trial_counts=MappingProxyType({20260721: 69, 20261721: 69}),
        normal_confirmations=MappingProxyType({20260721: 0, 20261721: 0}),
        label_query_count=0,
        artifact_map_sha256=None,
    )


class ArtifactsAntigravityTest(unittest.TestCase):
    def test_publish_writes_exact_canonical_seven_leaf_publication(self) -> None:
        records = records_fixture_antigravity()
        qualification = qualify_antigravity(records)
        decision = evaluate_qualification_antigravity(qualification)
        with tempfile.TemporaryDirectory(prefix="exp468-task5-") as directory:
            published = publish_result_antigravity(
                Path(directory) / "root-a",
                records,
                qualification,
                decision,
            )
            self.assertIs(type(published), PublishedResultAntigravity)
            self.assertEqual(
                tuple(sorted(path.name for path in published.root.iterdir())),
                (
                    "artifact_sha256_antigravity.json",
                    "context_antigravity.json",
                    "dashboard_snapshot_antigravity.json",
                    "decision_antigravity.json",
                    "failed_folds_antigravity.jsonl",
                    "method_fold_metrics_antigravity.jsonl",
                    "method_summary_antigravity.jsonl",
                ),
            )
            self.assertEqual(published.qualification, qualification)
            self.assertEqual(published.decision, decision)
            self.assertEqual(
                len((published.root / "method_fold_metrics_antigravity.jsonl").read_text().splitlines()),
                30,
            )
            self.assertEqual(
                len((published.root / "method_summary_antigravity.jsonl").read_text().splitlines()),
                3,
            )

    def test_same_inputs_publish_byte_identical_roots_and_existing_root_is_rejected(self) -> None:
        records = records_fixture_antigravity()
        qualification = qualification_fixture_antigravity()
        decision = decision_fixture_antigravity()
        with tempfile.TemporaryDirectory(prefix="exp468-task5-") as directory:
            root_a = publish_result_antigravity(Path(directory) / "a", records, qualification, decision).root
            root_b = publish_result_antigravity(Path(directory) / "b", records, qualification, decision).root
            for path_a in root_a.iterdir():
                self.assertEqual(path_a.read_bytes(), (root_b / path_a.name).read_bytes())
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                publish_result_antigravity(root_a, records, qualification, decision)

    def test_dashboard_snapshot_branches_private_status_but_never_runs_private(self) -> None:
        for passed in (True, False):
            with self.subTest(passed=passed):
                qualification = scalar_qualification_fixture_antigravity(passed=passed)
                decision = evaluate_qualification_antigravity(qualification)
                snapshot = _dashboard_snapshot_payload_antigravity(qualification, decision)
                expected_status = (
                    "awaiting_private_reproducibility"
                    if passed
                    else "qualification_failed_blocked_before_private_run"
                )
                self.assertEqual(snapshot["private_reproducibility"]["status"], expected_status)
                self.assertEqual(
                    snapshot["private_reproducibility"]["run_1"],
                    "not_run",
                )
                self.assertEqual(
                    snapshot["private_reproducibility"]["run_2"],
                    "not_run",
                )

    def test_publication_has_canonical_bytes_hash_contract_order_and_provenance(self) -> None:
        records = records_fixture_antigravity()
        qualification = qualification_fixture_antigravity()
        decision = decision_fixture_antigravity()
        with tempfile.TemporaryDirectory(prefix="exp468-task5-") as directory:
            root = publish_result_antigravity(
                Path(directory) / "root",
                records,
                qualification,
                decision,
            ).root
            self.assertEqual(
                tuple(sorted(path.name for path in root.iterdir())),
                tuple(sorted(LEAF_NAMES_ANTIGRAVITY)),
            )
            digest_payload = json.loads(
                (root / "artifact_sha256_antigravity.json").read_text(encoding="utf-8"),
            )
            expected_digests = {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in _HASHED_LEAF_NAMES_ANTIGRAVITY
            }
            self.assertEqual(digest_payload, expected_digests)
            self.assertNotIn("artifact_sha256_antigravity.json", digest_payload)
            for name in (
                "context_antigravity.json",
                "dashboard_snapshot_antigravity.json",
                "decision_antigravity.json",
            ):
                raw = (root / name).read_bytes()
                self.assertTrue(raw.endswith(b"\n"))
                self.assertFalse(raw.endswith(b"\n\n"))
                self.assertEqual(
                    raw,
                    _canonical_json_bytes_antigravity(json.loads(raw.decode("utf-8"))),
                )
            jsonl_rows: dict[str, list[dict[str, object]]] = {}
            for name in (
                "failed_folds_antigravity.jsonl",
                "method_fold_metrics_antigravity.jsonl",
                "method_summary_antigravity.jsonl",
            ):
                raw = (root / name).read_bytes()
                rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
                self.assertEqual(
                    raw,
                    b"".join(_canonical_json_bytes_antigravity(row) for row in rows),
                )
                jsonl_rows[name] = rows
            metric_rows = jsonl_rows["method_fold_metrics_antigravity.jsonl"]
            self.assertEqual(
                [(row["seed"], row["fold_index"], row["method"]) for row in metric_rows],
                [
                    (seed, fold_index, method)
                    for seed in (20260721, 20261721)
                    for fold_index in range(5)
                    for method in ("C0", "S10", "C10")
                ],
            )
            for row in metric_rows:
                self.assertEqual(row["backend"], "numpy")
                self.assertEqual(
                    row["approval_document_sha256"],
                    "6780edb2683c992b3df361c72e15542a6e974cf3e7cbbc0df65df2739c0203a9",
                )
                self.assertEqual(
                    row["pre_approval_benchmark_sha256"],
                    "938d83b0319346270a6cc32b7b6ba3efb88122dc1bf431863d8030f4539fb63b",
                )
            summary_rows = jsonl_rows["method_summary_antigravity.jsonl"]
            self.assertEqual([row["method"] for row in summary_rows], ["C0", "S10", "C10"])
            with tempfile.TemporaryDirectory(prefix="exp468-task5-empty-") as empty_directory:
                empty_path = Path(empty_directory) / "failed_folds_antigravity.jsonl"
                _write_jsonl_antigravity(empty_path, [])
                self.assertEqual(empty_path.read_bytes(), b"")

    def test_publish_rejects_record_and_contract_boundary_tampering_before_root_creation(self) -> None:
        records = records_fixture_antigravity()
        qualification = qualification_fixture_antigravity()
        decision = decision_fixture_antigravity()
        first_key = (20260721, 0)
        second_key = (20260721, 1)
        first = records[first_key]
        second = records[second_key]
        swapped_first = dataclasses.replace(
            first,
            input_fit_indices=second.input_fit_indices,
            input_query_indices=second.input_query_indices,
        )
        exchanged = MappingProxyType(
            {
                key: swapped_first if key == first_key else value
                for key, value in records.items()
            },
        )
        missing = MappingProxyType(dict(tuple(records.items())[:-1]))
        additional_values = dict(records)
        additional_values[(20261721, 5)] = first
        additional = MappingProxyType(additional_values)
        reordered = MappingProxyType(dict(reversed(tuple(records.items()))))
        forged_fold = dataclasses.replace(
            qualification.fold_results[0],
            c10_per_type_auc=(0.70, 0.70, 0.70, 0.90),
        )
        forged_qualification = dataclasses.replace(
            qualification,
            fold_results=(forged_fold, *qualification.fold_results[1:]),
        )
        forged_decision = dataclasses.replace(
            decision,
            status="qualification_failed_blocked_before_private_run",
            eligible_for_private_reproducibility=False,
            failure_reasons=("forged",),
        )
        cases = (
            ("missing", missing, qualification, decision),
            ("additional", additional, qualification, decision),
            ("reordered", reordered, qualification, decision),
            ("fold_exchange", exchanged, qualification, decision),
            ("forged_qualification", records, forged_qualification, decision),
            ("forged_decision", records, qualification, forged_decision),
        )
        for name, candidate_records, candidate_qualification, candidate_decision in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory(
                prefix=f"exp468-task5-{name}-",
            ) as directory:
                root = Path(directory) / "root"
                with self.assertRaises(ValueError):
                    publish_result_antigravity(
                        root,
                        candidate_records,
                        candidate_qualification,
                        candidate_decision,
                    )
                self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
