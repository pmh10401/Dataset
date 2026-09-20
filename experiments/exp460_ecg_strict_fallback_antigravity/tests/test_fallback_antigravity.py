from __future__ import annotations

from dataclasses import fields
import unittest


def _clone_antigravity(value: object) -> object:
    cloned = object.__new__(type(value))
    for field in fields(type(value)):
        object.__setattr__(cloned, field.name, getattr(value, field.name))
    return cloned


def _authority_with_passes_antigravity(passes: tuple[bool, ...]) -> object:
    from experiments.exp460_ecg_strict_fallback_antigravity.predecessor_antigravity import (
        capture_exp459_fallback_authority_antigravity,
    )

    if len(passes) != 10:
        raise AssertionError("expected ten pass flags")
    authority = _clone_antigravity(capture_exp459_fallback_authority_antigravity())
    records = []
    for original, passed in zip(authority.fold_records, passes, strict=True):
        record = _clone_antigravity(original)
        object.__setattr__(record, "passed", passed)
        records.append(record)
    record_tuple = tuple(records)
    object.__setattr__(authority, "fold_records", record_tuple)

    summaries = []
    for index, summary in enumerate(authority.seed_summaries):
        start = index * 5
        stop = start + 5
        summary_clone = _clone_antigravity(summary)
        summary_records = record_tuple[start:stop]
        object.__setattr__(summary_clone, "fold_records", summary_records)
        object.__setattr__(
            summary_clone,
            "passed",
            all(record.passed for record in summary_records),
        )
        summaries.append(summary_clone)
    object.__setattr__(authority, "seed_summaries", tuple(summaries))

    decision = _clone_antigravity(authority.decision)
    decision_passed = all(passes)
    object.__setattr__(decision, "passed", decision_passed)
    object.__setattr__(decision, "status", "stable" if decision_passed else "unstable")
    object.__setattr__(authority, "decision", decision)
    return authority


class StrictRouteAntigravityTest(unittest.TestCase):
    def test_strict_route_only_uses_candidate_for_ten_of_ten_antigravity(self) -> None:
        """Break caught: strict routing accepts any pass count below 10/10."""
        from experiments.exp460_ecg_strict_fallback_antigravity.fallback_antigravity import (
            strict_route_antigravity,
        )

        table = (
            ("ten_of_ten", (True, True, True, True, True, True, True, True, True, True), "candidate"),
            ("nine_of_ten", (False, True, True, True, True, True, True, True, True, True), "baseline"),
            ("eight_of_ten", (False, False, True, True, True, True, True, True, True, True), "baseline"),
            ("seven_of_ten", (False, False, False, True, True, True, True, True, True, True), "baseline"),
            ("zero_of_ten", (False, False, False, False, False, False, False, False, False, False), "baseline"),
        )
        for label, passes, expected in table:
            with self.subTest(case=label):
                self.assertEqual(
                    strict_route_antigravity(_authority_with_passes_antigravity(passes)),
                    expected,
                )

    def test_strict_route_rejects_malformed_authority_antigravity(self) -> None:
        """Break caught: strict routing trusts malformed authority objects."""
        from experiments.exp460_ecg_strict_fallback_antigravity.fallback_antigravity import (
            strict_route_antigravity,
        )

        malformed = _authority_with_passes_antigravity(
            (True, True, True, True, True, True, True, True, True, True)
        )
        object.__setattr__(malformed, "fold_records", malformed.fold_records[:-1])
        with self.assertRaises(ValueError):
            strict_route_antigravity(malformed)

        malformed = _authority_with_passes_antigravity(
            (True, True, True, True, True, True, True, True, True, True)
        )
        record = _clone_antigravity(malformed.fold_records[0])
        object.__setattr__(record, "passed", 1)
        object.__setattr__(
            malformed,
            "fold_records",
            (record,) + malformed.fold_records[1:],
        )
        with self.assertRaises(ValueError):
            strict_route_antigravity(malformed)

        with self.assertRaises(ValueError):
            strict_route_antigravity(object())


if __name__ == "__main__":
    unittest.main()
