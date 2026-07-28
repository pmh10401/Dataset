from dataclasses import replace
import unittest

from experiments.exp411_near_repeat_qualification.config import FROZEN_CONFIG
from experiments.exp411_near_repeat_qualification.contracts import TrainRun
from experiments.exp411_near_repeat_qualification.folds import group_crossfit_plan


def run(index: int, digest: str, length: int = 4) -> TrainRun:
    return TrainRun(
        dataset="Alpha_normal_0",
        instance_index=index,
        values=tuple(float(index) for _ in range(length)),
        values_sha256=digest,
    )


class GroupFoldTests(unittest.TestCase):
    def test_duplicate_identity_never_crosses_fit_and_query(self) -> None:
        rows = tuple(
            run(index, digest)
            for index, digest in enumerate(("a", "a", "b", "c", "d", "e"))
        )
        folds = group_crossfit_plan(rows, seed=411)
        self.assertEqual(len(folds), 5)
        for fold in folds:
            fit_hashes = {rows[index].values_sha256 for index in fold.fit_indices}
            query_hashes = {
                rows[index].values_sha256 for index in fold.query_indices
            }
            self.assertTrue(fit_hashes.isdisjoint(query_hashes))

    def test_fewer_than_five_identity_groups_abstains(self) -> None:
        rows = tuple(run(index, str(index % 4)) for index in range(8))
        self.assertEqual(group_crossfit_plan(rows, seed=411), ())

    def test_twenty_groups_use_deterministic_five_fold(self) -> None:
        rows = tuple(run(index, str(index)) for index in range(20))
        first = group_crossfit_plan(rows, seed=411)
        second = group_crossfit_plan(tuple(reversed(rows)), seed=411)
        self.assertEqual(
            {tuple(sorted(fold.query_identity_sha256)) for fold in first},
            {tuple(sorted(fold.query_identity_sha256)) for fold in second},
        )
        self.assertEqual(len(first), 5)

    def test_frozen_parameters_reject_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen"):
            replace(FROZEN_CONFIG, quantiles=(0.02, 0.05, 0.10)).validate()


if __name__ == "__main__":
    unittest.main()
