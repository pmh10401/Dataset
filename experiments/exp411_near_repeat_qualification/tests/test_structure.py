import ast
from pathlib import Path
import unittest

import experiments.exp411_near_repeat_qualification.inputs as input_module
import experiments.exp411_near_repeat_qualification.qualification as qualification_module
import experiments.exp411_near_repeat_qualification.runner as runner_module


class StructureTests(unittest.TestCase):
    def test_prediction_and_qualification_have_no_forbidden_identifiers(self) -> None:
        for module in (qualification_module, runner_module):
            source = Path(module.__file__).read_text(encoding="utf-8").lower()
            for forbidden in (
                "labels_blob",
                "is_true_anomaly",
                "split = 'test'",
                'split = "test"',
            ):
                self.assertNotIn(forbidden, source)

    def test_sql_select_statements_do_not_project_test_or_labels(self) -> None:
        source = Path(input_module.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        statements = [
            node.value.lower()
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and "select" in node.value.lower()
        ]
        self.assertTrue(statements)
        for statement in statements:
            self.assertNotIn("labels_blob", statement)
            self.assertNotIn("split = 'test'", statement)
            self.assertNotIn('split = "test"', statement)


if __name__ == "__main__":
    unittest.main()
