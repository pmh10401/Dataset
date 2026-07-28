import ast
from pathlib import Path
import re
import unittest


PACKAGE = Path(__file__).resolve().parents[1]


class ForbiddenAccessTests(unittest.TestCase):
    def test_train_only_sources_have_no_test_or_outcome_field_access(self) -> None:
        sources = [
            PACKAGE / name
            for name in (
                "inputs.py",
                "runner.py",
                "repair.py",
                "migration.py",
                "benchmark.py",
            )
            if (PACKAGE / name).exists()
        ]
        forbidden_fields = {
            "labels_blob",
            "is_true_anomaly",
            "tp",
            "fp",
            "fn",
            "f1",
        }
        failures: list[str] = []
        for path in sources:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    lowered = node.value.lower()
                    if "labels_blob" in lowered or "is_true_anomaly" in lowered:
                        failures.append(f"{path.name}:{node.lineno}:forbidden constant")
                    if re.search(
                        r"\bsplit\s*=\s*['\"]test['\"]",
                        lowered,
                    ):
                        failures.append(f"{path.name}:{node.lineno}:TEST SQL")
                if isinstance(node, ast.Attribute):
                    if node.attr.lower() in forbidden_fields:
                        failures.append(f"{path.name}:{node.lineno}:field access")
                if (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)
                    and node.slice.value.lower() in forbidden_fields
                ):
                    failures.append(f"{path.name}:{node.lineno}:field access")
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and node.args[0].value.lower() in forbidden_fields
                ):
                    failures.append(f"{path.name}:{node.lineno}:field access")
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
