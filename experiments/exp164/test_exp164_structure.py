from __future__ import annotations
import ast
import os
import unittest

class SQLVisitor(ast.NodeVisitor):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.errors = []

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str):
            val = node.value.upper()
            # Check if it looks like SQL (e.g. SELECT ... FROM ... or other SQL patterns)
            is_sql = ("SELECT" in val and "FROM" in val) or ("INSERT" in val and "INTO" in val) or ("DELETE" in val) or ("UPDATE" in val)
            if is_sql:
                # check forbidden terms
                forbidden = ["TEST", "LABEL", "ANOMALY"]
                for term in forbidden:
                    if term in val:
                        self.errors.append(
                            f"Forbidden SQL term '{term}' found in SQL string literal in {self.filepath}: {node.value}"
                        )
        self.generic_visit(node)


class TestExp164Structure(unittest.TestCase):
    def test_forbidden_sql_in_package(self):
        exp164_dir = os.path.dirname(os.path.abspath(__file__))
        all_errors = []
        for root, _, files in os.walk(exp164_dir):
            for file in files:
                if file.endswith(".py") and not file.startswith("test_"):
                    filepath = os.path.join(root, file)
                    with open(filepath, "r", encoding="utf-8") as f:
                        source = f.read()
                    try:
                        tree = ast.parse(source, filename=filepath)
                    except SyntaxError as e:
                        all_errors.append(f"SyntaxError in {filepath}: {e}")
                        continue

                    visitor = SQLVisitor(filepath)
                    visitor.visit(tree)
                    all_errors.extend(visitor.errors)

        self.assertEqual(all_errors, [], f"Structure audit failed with SQL errors: {all_errors}")

if __name__ == "__main__":
    unittest.main()
