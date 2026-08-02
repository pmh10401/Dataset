import unittest

from experiments.exp422_test_eval.contracts import (
    ConfusionCounts,
    LabelRow,
)


class ContractTests(unittest.TestCase):
    def test_confusion_matrix_orientation_is_tn_fp_fn_tp(self) -> None:
        counts = ConfusionCounts(tn=11, fp=2, fn=3, tp=7)
        self.assertEqual(counts.as_nested(), [[11, 2], [3, 7]])
        self.assertEqual(counts.total, 23)

    def test_counts_reject_bool_and_negative_values(self) -> None:
        with self.assertRaises(ValueError):
            ConfusionCounts(tn=True, fp=0, fn=0, tp=0)
        with self.assertRaises(ValueError):
            ConfusionCounts(tn=0, fp=-1, fn=0, tp=0)

    def test_label_requires_exact_binary_integer(self) -> None:
        self.assertEqual(LabelRow("A", 0, 1).is_true_anomaly, 1)
        for value in (True, False, 0.0, 1.0, "1", -1, 2):
            with self.subTest(value=value), self.assertRaises(ValueError):
                LabelRow("A", 0, value)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
