from __future__ import annotations

from unittest.mock import patch
import unittest

import numpy as np

from experiments.exp417_causal_multihorizon_shadow.contracts import (
    LengthGroup,
    ProcessedGroup,
    RawSeries,
)
from experiments.exp417_causal_multihorizon_shadow.preprocessing import (
    canonical_waveform_sha256,
    identity_grouped_folds,
)
from experiments.exp425_selective_recovery.eligibility import (
    apply_group_eligibility,
    evaluate_group_eligibility,
)


def _waveform(token: int, length: int) -> tuple[float, ...]:
    positions = np.arange(length, dtype=np.float64)
    return tuple(token * 0.01 + positions * 0.125 + np.sin(positions * (token % 11 + 1)))


def _series_with_identity_classes(
    length: int, *, early_count: int, other_count: int
) -> tuple[RawSeries, ...]:
    selected: list[RawSeries] = []
    early = 0
    other = 0
    token = 0
    while early < early_count or other < other_count:
        item = RawSeries(instance_index=len(selected), values=_waveform(token, length))
        is_early = int(canonical_waveform_sha256(item), 16) % 10 == 0
        if is_early and early < early_count:
            selected.append(item)
            early += 1
        elif not is_early and other < other_count:
            selected.append(item)
            other += 1
        token += 1
    return tuple(selected)


def _processed_group(
    length: int,
    *,
    status: str,
    early_count: int = 20,
    other_count: int = 100,
    series: tuple[RawSeries, ...] | None = None,
) -> ProcessedGroup:
    group_series = series or _series_with_identity_classes(
        length, early_count=early_count, other_count=other_count
    )
    channels = {
        item.instance_index: np.zeros((2, length), dtype=np.float32) for item in group_series
    }
    return ProcessedGroup(
        group=LengthGroup(length=length, series=group_series),
        raw_center=0.0,
        raw_scale=1.0,
        difference_center=0.0,
        difference_scale=1.0,
        status=status,
        channels=channels,
    )


class GroupEligibilityTests(unittest.TestCase):
    fold_contract_sha256 = "a" * 64

    def test_eligible_fold_indices_exactly_reuse_exp417(self) -> None:
        group = _processed_group(64, status="diagnostic_only")
        captured = []

        def record_folds(series):  # type: ignore[no-untyped-def]
            folds = identity_grouped_folds(series)
            captured.append(folds)
            return folds

        with patch(
            "experiments.exp425_selective_recovery.eligibility.identity_grouped_folds",
            side_effect=record_folds,
        ):
            evidence = evaluate_group_eligibility(
                "Synthetic_normal_1", group, fold_contract_sha256=self.fold_contract_sha256
            )

        self.assertEqual(evidence.status, "eligible")
        self.assertEqual(captured, [identity_grouped_folds(group.group.series)])

    def test_fewer_than_five_early_identities_abstains_without_constructing_exp417_folds(self) -> None:
        group = _processed_group(
            64, status="diagnostic_only", early_count=4, other_count=60
        )
        with patch(
            "experiments.exp425_selective_recovery.eligibility.identity_grouped_folds"
        ) as frozen_folds:
            evidence = evaluate_group_eligibility(
                "Synthetic_normal_1", group, fold_contract_sha256=self.fold_contract_sha256
            )

        self.assertEqual(evidence.status, "oof_identity_insufficient")
        self.assertEqual(evidence.reason, "oof_identity_insufficient")
        frozen_folds.assert_not_called()

    def test_only_insufficient_group_is_replaced_with_empty_channels(self) -> None:
        eligible_group = _processed_group(64, status="diagnostic_only")
        insufficient_group = _processed_group(
            65, status="diagnostic_only", early_count=4, other_count=60
        )

        assessment = apply_group_eligibility(
            "Synthetic_normal_1",
            (eligible_group, insufficient_group),
            fold_contract_sha256=self.fold_contract_sha256,
        )

        self.assertEqual(tuple(assessment.evidence), (64, 65))
        self.assertIs(assessment.groups[0], eligible_group)
        self.assertEqual(assessment.groups[1].status, "insufficient_train_abstain")
        self.assertEqual(dict(assessment.groups[1].channels), {})

    def test_output_order_is_stable_under_input_permutation(self) -> None:
        first = _processed_group(64, status="diagnostic_only")
        second = _processed_group(
            65, status="diagnostic_only", early_count=4, other_count=60
        )

        forward = apply_group_eligibility(
            "Synthetic_normal_1", (first, second), fold_contract_sha256=self.fold_contract_sha256
        )
        reverse = apply_group_eligibility(
            "Synthetic_normal_1", (second, first), fold_contract_sha256=self.fold_contract_sha256
        )

        self.assertEqual(tuple(group.group.length for group in forward.groups), (64, 65))
        self.assertEqual(tuple(group.group.length for group in reverse.groups), (64, 65))
        self.assertEqual(forward.evidence, reverse.evidence)

    def test_duplicate_waveforms_are_one_identity(self) -> None:
        base = _series_with_identity_classes(64, early_count=20, other_count=100)
        duplicate = RawSeries(instance_index=len(base), values=base[0].values)
        group = _processed_group(
            64, status="diagnostic_only", series=base + (duplicate,)
        )

        evidence = evaluate_group_eligibility(
            "Synthetic_normal_1", group, fold_contract_sha256=self.fold_contract_sha256
        )

        self.assertEqual(evidence.unique_identity_count, len(base))

    def test_diagnostic_only_uses_oof_splits_without_a_final_fit_split(self) -> None:
        group = _processed_group(64, status="diagnostic_only")

        evidence = evaluate_group_eligibility(
            "Synthetic_normal_1", group, fold_contract_sha256=self.fold_contract_sha256
        )

        self.assertEqual(evidence.status, "eligible")
        self.assertEqual(len(evidence.folds), 5)
        self.assertIsNone(evidence.final_fit)
        self.assertIsNone(evidence.final_validation)

    def test_candidate_eligible_records_and_requires_the_final_fit_split(self) -> None:
        group = _processed_group(64, status="candidate_eligible")

        evidence = evaluate_group_eligibility(
            "Synthetic_normal_1", group, fold_contract_sha256=self.fold_contract_sha256
        )

        self.assertEqual(evidence.status, "eligible")
        self.assertGreaterEqual(evidence.final_fit or 0, 5)
        self.assertGreaterEqual(evidence.final_validation or 0, 5)

    def test_preprocessing_abstained_group_bypasses_folds_and_is_preserved(self) -> None:
        raw = RawSeries(instance_index=0, values=(1.0,) * 64)
        group = ProcessedGroup(
            group=LengthGroup(length=64, series=(raw,)),
            raw_center=0.0,
            raw_scale=0.0,
            difference_center=0.0,
            difference_scale=0.0,
            status="zero_scale_abstain",
        )
        with patch(
            "experiments.exp425_selective_recovery.eligibility.identity_grouped_folds"
        ) as frozen_folds:
            assessment = apply_group_eligibility(
                "Synthetic_normal_1", (group,), fold_contract_sha256=self.fold_contract_sha256
            )

        self.assertIs(assessment.groups[0], group)
        self.assertEqual(assessment.evidence[64].status, "preprocessing_abstained")
        self.assertEqual(assessment.evidence[64].reason, "zero_scale_abstain")
        self.assertEqual(assessment.evidence[64].folds, ())
        frozen_folds.assert_not_called()


if __name__ == "__main__":
    unittest.main()
