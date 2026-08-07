"""Contract tests for Exp426's immutable scientific records."""

from __future__ import annotations

import unittest

from experiments.exp426_verified_baseline.contracts_antigravity import (
    Exp426Counts,
    Exp426DatasetRecord,
    Exp426Provenance,
)


class Exp426ContractTests(unittest.TestCase):
    def provenance(self, **changes: object) -> Exp426Provenance:
        values: dict[str, object] = {
            "run_identity": "exp426-run",
            "git_commit": "1" * 40,
            "database_sha256": "2" * 64,
            "source_root_sha256": "3" * 64,
            "source_progress_sha256": "4" * 64,
            "source_record_set_sha256": "5" * 64,
            "lane_artifact_sha256": {
                "hard": "6" * 64,
                "standard": "7" * 64,
                "priority": "8" * 64,
                "automatic_normal": "9" * 64,
            },
            "fold_contract_sha256": "a" * 64,
            "expected_datasets": ("A_normal_1", "B_normal_1"),
            "workers": 12,
        }
        values.update(changes)
        return Exp426Provenance(**values)

    @staticmethod
    def imported() -> Exp426DatasetRecord:
        return Exp426DatasetRecord(
            dataset="A_normal_1",
            origin="exp424_import",
            status="scored",
            candidate_coordinates=(("A_normal_1", 1),),
            source_record_sha256="b" * 64,
            source_result_sha256="c" * 64,
            train_identity_sha256=None,
            eligibility_sha256=None,
            reason="verified_exp424_nonterminal",
        )

    @staticmethod
    def abstention() -> Exp426DatasetRecord:
        return Exp426DatasetRecord(
            dataset="B_normal_1",
            origin="verified_train_abstention",
            status="abstained",
            candidate_coordinates=(),
            source_record_sha256="d" * 64,
            source_result_sha256="e" * 64,
            train_identity_sha256="f" * 64,
            eligibility_sha256="0" * 64,
            reason="oof_identity_insufficient",
        )

    def test_provenance_fixes_twelve_workers_and_sorted_population(self) -> None:
        provenance = self.provenance()
        self.assertEqual(provenance.workers, 12)
        self.assertEqual(provenance.expected_datasets, ("A_normal_1", "B_normal_1"))
        with self.assertRaisesRegex(ValueError, "workers"):
            self.provenance(workers=7)
        with self.assertRaisesRegex(ValueError, "sorted"):
            self.provenance(expected_datasets=("B_normal_1", "A_normal_1"))

    def test_imported_record_preserves_source_and_candidates(self) -> None:
        record = self.imported()
        self.assertEqual(record.scientific_payload(), {
            "schema": "exp426.dataset-record.v1",
            "dataset": "A_normal_1",
            "origin": "exp424_import",
            "status": "scored",
            "candidate_coordinates": [["A_normal_1", 1]],
            "source_record_sha256": "b" * 64,
            "source_result_sha256": "c" * 64,
            "train_identity_sha256": None,
            "eligibility_sha256": None,
            "reason": "verified_exp424_nonterminal",
        })
        self.assertEqual(len(record.sha256), 64)

    def test_imported_record_requires_scored_status_and_source_evidence(self) -> None:
        for changes, message in (
            ({"status": "abstained", "candidate_coordinates": ()}, "scored"),
            ({"source_record_sha256": None}, "source evidence"),
            ({"source_result_sha256": None}, "source evidence"),
        ):
            values = {
                "dataset": "A_normal_1", "origin": "exp424_import",
                "status": "scored",
                "candidate_coordinates": (("A_normal_1", 1),),
                "source_record_sha256": "b" * 64,
                "source_result_sha256": "c" * 64,
                "train_identity_sha256": None, "eligibility_sha256": None,
                "reason": "verified_exp424_nonterminal",
            }
            values.update(changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, message):
                Exp426DatasetRecord(**values)

    def test_contracts_reject_unknown_values_nonlowercase_hashes_and_runtime_fields(self) -> None:
        payload = self.imported().scientific_payload()
        for key in ("elapsed_seconds", "worker_pid"):
            changed = dict(payload)
            changed[key] = 1
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "payload"):
                Exp426DatasetRecord.from_payload(changed)

        for field, value in (("origin", "unknown"), ("status", "unknown")):
            changed = dict(payload)
            changed[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                Exp426DatasetRecord.from_payload(changed)

        with self.assertRaisesRegex(ValueError, "lowercase"):
            self.provenance(database_sha256="A" * 64)
        changed = dict(payload)
        changed["source_record_sha256"] = "B" * 64
        with self.assertRaisesRegex(ValueError, "lowercase"):
            Exp426DatasetRecord.from_payload(changed)

    def test_train_abstention_requires_empty_candidates_and_train_evidence(self) -> None:
        self.assertEqual(self.abstention().candidate_coordinates, ())
        with self.assertRaisesRegex(ValueError, "candidates"):
            Exp426DatasetRecord(
                dataset="B_normal_1", origin="verified_train_abstention",
                status="abstained", candidate_coordinates=(("B_normal_1", 1),),
                source_record_sha256="d" * 64, source_result_sha256="e" * 64,
                train_identity_sha256="f" * 64, eligibility_sha256="0" * 64,
                reason="oof_identity_insufficient",
            )
        with self.assertRaisesRegex(ValueError, "TRAIN evidence"):
            Exp426DatasetRecord(
                dataset="B_normal_1", origin="verified_train_abstention",
                status="abstained", candidate_coordinates=(),
                source_record_sha256="d" * 64, source_result_sha256="e" * 64,
                train_identity_sha256=None, eligibility_sha256=None,
                reason="oof_identity_insufficient",
            )

    def test_candidate_coordinates_are_sorted_unique_and_dataset_bound(self) -> None:
        for coordinates in (
            (("A_normal_1", 2), ("A_normal_1", 1)),
            (("A_normal_1", 1), ("A_normal_1", 1)),
            (("B_normal_1", 1),),
        ):
            with self.subTest(coordinates=coordinates), self.assertRaises(ValueError):
                Exp426DatasetRecord(
                    dataset="A_normal_1", origin="exp424_import", status="scored",
                    candidate_coordinates=coordinates,
                    source_record_sha256="b" * 64, source_result_sha256="c" * 64,
                    train_identity_sha256=None, eligibility_sha256=None,
                    reason="verified_exp424_nonterminal",
                )

    def test_record_rejects_bool_index_and_lane_overlap(self) -> None:
        with self.assertRaisesRegex(ValueError, "index"):
            Exp426DatasetRecord(
                dataset="A_normal_1", origin="exp424_import", status="scored",
                candidate_coordinates=(("A_normal_1", True),),
                source_record_sha256="b" * 64, source_result_sha256="c" * 64,
                train_identity_sha256=None, eligibility_sha256=None,
                reason="verified_exp424_nonterminal",
            )
        with self.assertRaisesRegex(ValueError, "lane"):
            self.imported().validate_against(
                {"A_normal_1": 3}, lambda dataset, index: (dataset, index) == ("A_normal_1", 1)
            )

    def test_counts_require_exact_population_conservation(self) -> None:
        counts = Exp426Counts(1117, 1117, 983, 134, 134, 0)
        self.assertEqual(counts.remaining, 0)
        with self.assertRaisesRegex(ValueError, "conservation"):
            Exp426Counts(1117, 1117, 983, 133, 133, 0)


if __name__ == "__main__":
    unittest.main()
