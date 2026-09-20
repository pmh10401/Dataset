import unittest

from experiments.exp449_residual_fn_method_qualification_antigravity.method_audit_antigravity import (
    AUDIT_SPECS_ANTIGRAVITY,
    FIXED_INTERPRETER_ANTIGRAVITY,
    audit_methods_antigravity,
)


class MethodAuditAntigravityTest(unittest.TestCase):
    def test_specs_are_exact_and_audit_is_deterministic_antigravity(self):
        self.assertEqual(tuple(item[0] for item in AUDIT_SPECS_ANTIGRAVITY), (
            "MiniRocket", "MultiRocket", "HydraTransformer", "WEASEL_V2",
            "DrCIFClassifier", "HIVECOTEV2", "InceptionTimeClassifier", "TSChiefClassifier",
        ))
        first = audit_methods_antigravity()
        second = audit_methods_antigravity()
        self.assertEqual(first, second)
        self.assertEqual(tuple(record.status for record in first), (
            "score_compatible", "score_compatible", "score_compatible", "classifier_only",
            "classifier_only", "classifier_only", "classifier_only", "unavailable",
        ))
        for record in first[:-1]:
            self.assertEqual(len(record.source_sha256 or ""), 64)
            self.assertTrue(record.methods)
        self.assertIsNone(first[-1].source_sha256)
        self.assertEqual(FIXED_INTERPRETER_ANTIGRAVITY, "/Users/mac/Documents/Dataset/venvs/detection-coverage-antigravity-aeon140-py314/bin/python")


if __name__ == "__main__":
    unittest.main()
