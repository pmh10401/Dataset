"""Controlled-fork differential guard for the Task 2 foundation boundary."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


_ROOT_ANTIGRAVITY = Path(__file__).parents[3]
_EXP449_ANTIGRAVITY = _ROOT_ANTIGRAVITY / "experiments" / "exp449_residual_fn_method_qualification_antigravity"
_EXP452_ANTIGRAVITY = _ROOT_ANTIGRAVITY / "experiments" / "exp452_exp449_index_order_repair_antigravity"
_CONFIG_ONLY_NODES_ANTIGRAVITY = frozenset({
    "DATASET_COUNT_ANTIGRAVITY",
    "PREDICTION_COUNT_ANTIGRAVITY",
    "FAMILY_METHOD_PAIR_COUNT_ANTIGRAVITY",
    "EXP449_PREDICTION_AGGREGATE_SHA256_ANTIGRAVITY",
    "EXP449_PREDICTION_ROOT_ANTIGRAVITY",
    "EXP452_EVALUATION_ROOT_ANTIGRAVITY",
    "EXP452_RECEIPT_DIRECTORY_ANTIGRAVITY",
    "EXP452_FAILURE_AUDIT_ANTIGRAVITY",
    "INDEX_ORDER_RULE_ANTIGRAVITY",
})
_CONFIG_EXTRA_KEYS_ANTIGRAVITY = frozenset({
    "predecessor_experiment",
    "predecessor_prediction_aggregate_sha256",
    "predecessor_prediction_root",
    "evaluation_root",
    "receipt_directory",
    "failure_audit",
    "target_dataset_count",
    "target_prediction_count",
    "target_family_method_pair_count",
    "index_order_rule",
})
_CONFIG_IDENTITY_KEYS_ANTIGRAVITY = frozenset({"experiment", "schema", "kind", "run_id"})


class _IdentityNormalizerAntigravity(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if type(node.value) is str:
            node.value = (
                node.value
                .replace("Exp449", "ExpExperiment")
                .replace("Exp452", "ExpExperiment")
                .replace("exp449", "expExperiment")
                .replace("exp452", "expExperiment")
            )
        return node


def _assignment_names_antigravity(node: ast.stmt) -> set[str]:
    if not isinstance(node, ast.Assign):
        return set()
    return {target.id for target in node.targets if isinstance(target, ast.Name)}


def _normalize_config_antigravity(source: str, *, successor: bool) -> str:
    tree = ast.parse(source)
    body: list[ast.stmt] = []
    for node in tree.body:
        if successor and _assignment_names_antigravity(node) & _CONFIG_ONLY_NODES_ANTIGRAVITY:
            continue
        if isinstance(node, ast.Assign) and "FROZEN_CONFIG_ANTIGRAVITY" in _assignment_names_antigravity(node):
            call = node.value
            if not isinstance(call, ast.Call) or not call.args or not isinstance(call.args[0], ast.Dict):
                raise AssertionError("FROZEN_CONFIG_ANTIGRAVITY must remain a literal MappingProxyType dictionary")
            mapping = call.args[0]
            pairs = list(zip(mapping.keys, mapping.values, strict=True))
            kept: list[tuple[ast.expr | None, ast.expr]] = []
            for key, value in pairs:
                if not isinstance(key, ast.Constant) or type(key.value) is not str:
                    raise AssertionError("frozen config keys must remain literal exact strings")
                if successor and key.value in _CONFIG_EXTRA_KEYS_ANTIGRAVITY:
                    continue
                if key.value in _CONFIG_IDENTITY_KEYS_ANTIGRAVITY:
                    value = ast.Constant(value="<IDENTITY>")
                kept.append((key, value))
            mapping.keys = [key for key, _value in kept]
            mapping.values = [value for _key, value in kept]
        body.append(node)
    tree.body = body
    normalized = _IdentityNormalizerAntigravity().visit(tree)
    return ast.dump(ast.fix_missing_locations(normalized), include_attributes=False)


def _normalize_identity_only_antigravity(source: str) -> str:
    tree = _IdentityNormalizerAntigravity().visit(ast.parse(source))
    return ast.dump(ast.fix_missing_locations(tree), include_attributes=False)


def _function_ast_antigravity(source: str, name: str, *, allowed_return_wrapper: str | None = None) -> str:
    tree = ast.parse(source)
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name), None)
    if function is None:
        raise AssertionError(f"missing controlled function: {name}")
    if allowed_return_wrapper is not None:
        returns = [node for node in function.body if isinstance(node, ast.Return)]
        if len(returns) != 1 or not isinstance(returns[0].value, ast.Call):
            raise AssertionError("controlled function must have one wrapped final return")
        call = returns[0].value
        if not isinstance(call.func, ast.Name) or call.func.id != allowed_return_wrapper or len(call.args) != 1 or call.keywords:
            raise AssertionError("controlled function return wrapper differs")
        returns[0].value = call.args[0]
    normalized = _IdentityNormalizerAntigravity().visit(function)
    return ast.dump(ast.fix_missing_locations(normalized), include_attributes=False)


def _literal_assignment_antigravity(source: str, name: str) -> object:
    tree = ast.parse(source)
    assignment = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.Assign) and name in _assignment_names_antigravity(node)
        ),
        None,
    )
    if assignment is None:
        raise AssertionError(f"missing controlled assignment: {name}")
    return ast.literal_eval(assignment.value)


def _function_body_ast_antigravity(source: str, name: str) -> str:
    tree = ast.parse(source)
    function = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name),
        None,
    )
    if function is None:
        raise AssertionError(f"missing controlled function: {name}")
    if function.body and isinstance(function.body[0], ast.Expr) and isinstance(function.body[0].value, ast.Constant) and type(function.body[0].value.value) is str:
        function.body.pop(0)
    return ast.dump(ast.fix_missing_locations(function), include_attributes=False)


_TASK4_LEAF_NORMALIZERS_ANTIGRAVITY = (
    "context_antigravity.json", "compatibility_antigravity.json", "candidate_registry_antigravity.json",
    "frozen_predictions_antigravity.jsonl", "prediction_manifest_antigravity.json", "evaluation_antigravity.json",
    "candidate_metrics_antigravity.jsonl", "family_metrics_antigravity.jsonl", "bootstrap_antigravity.json",
    "decision_antigravity.json", "dashboard_snapshot_antigravity.json", "artifact_sha256_antigravity.json",
)


def _parse_task4_leaf_antigravity(name: str, payload: bytes) -> object:
    return [json.loads(line) for line in payload.splitlines()] if name.endswith(".jsonl") else json.loads(payload)


def _canonical_task4_leaf_antigravity(name: str, value: object) -> bytes:
    if name.endswith(".jsonl"):
        if not isinstance(value, list):
            raise AssertionError(f"{name} must remain JSONL")
        return b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n" for row in value)
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def _require_exact_keys_antigravity(value: object, expected: tuple[str, ...], name: str) -> dict[str, object]:
    if not isinstance(value, dict) or tuple(value) != expected:
        raise AssertionError(f"unallowlisted {name} shape")
    return value


def _normalize_context_task4_antigravity(value: object, successor: bool) -> object:
    copied = _require_exact_keys_antigravity(value, ("exp448_target_identity_sha256", "label_access_count", "reviewed_commit", "schema", "target_authority_sha256", "target_datasets"), "context")
    copied = copy.deepcopy(copied)
    copied["schema"] = "<schema>"
    return copied


def _normalize_compatibility_task4_antigravity(value: object, successor: bool) -> object:
    copied = _require_exact_keys_antigravity(value, ("predecessor", "prediction_authority", "schema"), "compatibility")
    return {"schema": "<schema>", "predecessor": "<predecessor>", "prediction_authority": "<prediction_authority>"}


def _normalize_manifest_task4_antigravity(value: object, successor: bool) -> object:
    allowed = (
        ("candidate_pairs", "frozen_prediction_sha256", "rows", "schema", "task5_aggregate_sha256", "task5_context_sha256", "task5_inventory", "task5_label_access_count", "task5_leaf_sha256", "task5_manifest_sha256", "task5_reviewed_commit", "task5_source_identity_sha256")
        if not successor else
        ("candidate_pairs", "exp449_prediction_aggregate_sha256", "exp449_reviewed_commit", "frozen_prediction_sha256", "label_access_count", "predecessor_attestation_sha256", "rows", "schema")
    )
    copied = _require_exact_keys_antigravity(value, allowed, "prediction manifest")
    # Explicitly discard only predecessor names/commits and their path/identity-derived hashes.
    return {
        "schema": "<schema>", "rows": copied["rows"], "candidate_pairs": copied["candidate_pairs"],
        "frozen_prediction_sha256": copied["frozen_prediction_sha256"],
        "label_access_count": copied["label_access_count"] if successor else copied["task5_label_access_count"],
    }


def _normalize_evaluation_task4_antigravity(value: object, successor: bool) -> object:
    if not isinstance(value, dict) or "schema" not in value:
        raise AssertionError("unallowlisted evaluation shape")
    copied = copy.deepcopy(value)
    copied["schema"] = "<schema>"
    return copied


def _normalize_identity_schema_task4_antigravity(value: object, successor: bool) -> object:
    if not isinstance(value, dict) or "schema" not in value:
        raise AssertionError("unallowlisted schema leaf shape")
    copied = copy.deepcopy(value)
    copied["schema"] = "<schema>"
    return copied


def _normalize_dashboard_task4_antigravity(value: object, successor: bool) -> object:
    if not isinstance(value, dict) or not {"schema", "experiment"} <= set(value):
        raise AssertionError("unallowlisted dashboard shape")
    copied = copy.deepcopy(value)
    copied["schema"] = "<schema>"
    copied["experiment"] = "<experiment>"
    return copied


_TASK4_LEAF_NORMALIZATION_TABLE_ANTIGRAVITY = {
    "context_antigravity.json": _normalize_context_task4_antigravity,
    "compatibility_antigravity.json": _normalize_compatibility_task4_antigravity,
    "candidate_registry_antigravity.json": lambda value, successor: value,
    "frozen_predictions_antigravity.jsonl": lambda value, successor: value,
    "prediction_manifest_antigravity.json": _normalize_manifest_task4_antigravity,
    "evaluation_antigravity.json": _normalize_evaluation_task4_antigravity,
    "candidate_metrics_antigravity.jsonl": lambda value, successor: value,
    "family_metrics_antigravity.jsonl": lambda value, successor: value,
    "bootstrap_antigravity.json": lambda value, successor: value,
    "decision_antigravity.json": _normalize_identity_schema_task4_antigravity,
    "dashboard_snapshot_antigravity.json": _normalize_dashboard_task4_antigravity,
}


def _assert_task4_leaf_differential_antigravity(legacy: dict[str, bytes], successor: dict[str, bytes]) -> None:
    """Compare every declared leaf with the one explicit identity/provenance allowlist."""
    if tuple(legacy) != _TASK4_LEAF_NORMALIZERS_ANTIGRAVITY or tuple(successor) != _TASK4_LEAF_NORMALIZERS_ANTIGRAVITY:
        raise AssertionError("leaf order differs")
    normalized: dict[bool, dict[str, object]] = {False: {}, True: {}}
    for name in _TASK4_LEAF_NORMALIZERS_ANTIGRAVITY[:-1]:
        normalizer = _TASK4_LEAF_NORMALIZATION_TABLE_ANTIGRAVITY[name]
        normalized[False][name] = normalizer(_parse_task4_leaf_antigravity(name, legacy[name]), False)
        normalized[True][name] = normalizer(_parse_task4_leaf_antigravity(name, successor[name]), True)
        if normalized[False][name] != normalized[True][name]:
            raise AssertionError(f"normalized leaf differs: {name}")
    common_indices = [0, 1]
    for side in (False, True):
        frozen = normalized[side]["frozen_predictions_antigravity.jsonl"]
        evaluation = normalized[side]["evaluation_antigravity.json"]
        if not isinstance(frozen, list) or not isinstance(evaluation, dict):
            raise AssertionError("frozen/evaluation canonical payload shape differs")
        evidence = [candidate["indices"] for row in evaluation["dataset_evidence"] for candidate in row["candidates"].values()]
        if [row["indices"] for row in frozen] != [common_indices] * len(frozen) or evidence != [common_indices] * len(evidence):
            raise AssertionError("evidence indices are not the common ascending canonical sequence")
    linked_maps: dict[bool, dict[str, str]] = {}
    for side, payloads in ((False, legacy), (True, successor)):
        artifact_map = _parse_task4_leaf_antigravity("artifact_sha256_antigravity.json", payloads["artifact_sha256_antigravity.json"])
        if not isinstance(artifact_map, dict) or tuple(artifact_map) != tuple(sorted(_TASK4_LEAF_NORMALIZERS_ANTIGRAVITY[:-1])):
            raise AssertionError("artifact map inventory/order differs")
        linked: dict[str, str] = {}
        for name in _TASK4_LEAF_NORMALIZERS_ANTIGRAVITY[:-1]:
            if artifact_map[name] != hashlib.sha256(payloads[name]).hexdigest():
                raise AssertionError(f"artifact map raw linkage differs: {name}")
            linked[name] = hashlib.sha256(_canonical_task4_leaf_antigravity(name, normalized[side][name])).hexdigest()
        linked_maps[side] = linked
    if linked_maps[False] != linked_maps[True]:
        raise AssertionError("artifact map normalized linkage differs")


class ControlledForkDifferentialAntigravityTest(unittest.TestCase):
    def test_config_ast_allows_only_named_identity_path_predecessor_and_index_nodes_antigravity(self) -> None:
        """Break caught: changing a scientific config node outside the explicit allowlist."""
        before = (_EXP449_ANTIGRAVITY / "config_antigravity.py").read_text(encoding="utf-8")
        after = (_EXP452_ANTIGRAVITY / "config_antigravity.py").read_text(encoding="utf-8")
        self.assertEqual(_normalize_config_antigravity(before, successor=False), _normalize_config_antigravity(after, successor=True))

    def test_contract_artifact_and_bootstrap_asts_are_scientifically_identical_antigravity(self) -> None:
        """Break caught: changing validation, canonical IO, or bootstrap syntax."""
        for module in ("contracts", "artifacts", "bootstrap"):
            before = (_EXP449_ANTIGRAVITY / f"{module}_antigravity.py").read_text(encoding="utf-8")
            after = (_EXP452_ANTIGRAVITY / f"{module}_antigravity.py").read_text(encoding="utf-8")
            with self.subTest(module=module):
                self.assertEqual(_normalize_identity_only_antigravity(before), _normalize_identity_only_antigravity(after))

    def test_dashboard_aggregate_validation_ast_changes_identity_only_antigravity(self) -> None:
        """Break caught: weakening frozen aggregate math under the dashboard sanitizer."""
        before = (_EXP449_ANTIGRAVITY / "dashboard_export_antigravity.py").read_text(encoding="utf-8")
        after = (_EXP452_ANTIGRAVITY / "dashboard_export_antigravity.py").read_text(encoding="utf-8")
        expected = _function_ast_antigravity(before, "public_result_antigravity")
        self.assertEqual(expected, _function_ast_antigravity(after, "public_result_antigravity", allowed_return_wrapper="_materialize_antigravity"))
        changed = after.replace('global_["fn"] < 1988', 'global_["fn"] < 1987', 1)
        self.assertNotEqual(expected, _function_ast_antigravity(changed, "public_result_antigravity", allowed_return_wrapper="_materialize_antigravity"))

    def test_differential_rejects_unallowlisted_scientific_change_antigravity(self) -> None:
        """Break caught: a permissive normalizer accepting a retuned cross-fit fold count."""
        before = (_EXP449_ANTIGRAVITY / "config_antigravity.py").read_text(encoding="utf-8")
        after = (_EXP452_ANTIGRAVITY / "config_antigravity.py").read_text(encoding="utf-8")
        changed = after.replace('"crossfit_folds": 5', '"crossfit_folds": 6', 1)
        self.assertNotEqual(_normalize_config_antigravity(before, successor=False), _normalize_config_antigravity(changed, successor=True))

    def test_task2_import_firewall_excludes_lifecycle_labels_network_and_accelerators_antigravity(self) -> None:
        """Break caught: foundation import opening Task 3+, network, dashboard-write, or GPU dependencies."""
        forbidden = {
            "sqlite3", "predecessor_antigravity", "sources_antigravity", "lifecycle_antigravity",
            "readiness_antigravity", "evaluation_antigravity", "replay_antigravity", "cli_antigravity",
            "requests", "urllib", "http", "mlx", "torch",
        }
        for module in ("config", "contracts", "artifacts", "bootstrap", "dashboard_export"):
            source = (_EXP452_ANTIGRAVITY / f"{module}_antigravity.py").read_text(encoding="utf-8")
            imported = {
                name
                for node in ast.walk(ast.parse(source))
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for name in (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
            }
            with self.subTest(module=module):
                self.assertFalse(any(name == blocked or name.startswith(f"{blocked}.") for name in imported for blocked in forbidden))

    def test_task3_label_database_metadata_and_conversion_are_exp449_exact_antigravity(self) -> None:
        """Break caught: the controlled fork changes TEST SQL, database identity, or label conversion."""
        before = (_EXP449_ANTIGRAVITY / "sources_antigravity.py").read_text(encoding="utf-8")
        after = (_EXP452_ANTIGRAVITY / "sources_antigravity.py").read_text(encoding="utf-8")
        for name in (
            "_DATABASE_SHA256_ANTIGRAVITY",
            "_DATABASE_INODE_ANTIGRAVITY",
            "_LABEL_SQL_ANTIGRAVITY",
            "_TARGET_DATASET_COUNT_ANTIGRAVITY",
            "_TARGET_LABEL_ROWS_ANTIGRAVITY",
        ):
            with self.subTest(name=name):
                self.assertEqual(
                    _literal_assignment_antigravity(before, name),
                    _literal_assignment_antigravity(after, name),
                )
        self.assertEqual(
            _function_body_ast_antigravity(before, "convert_test_label_row_antigravity"),
            _function_body_ast_antigravity(after, "convert_test_label_row_antigravity"),
        )

    def test_task3_predecessor_authority_excludes_qualified_metrics_and_readiness_imports_antigravity(self) -> None:
        """Break caught: stopped authority leaks qualified metrics or readiness gains label/backend imports."""
        predecessor = (_EXP452_ANTIGRAVITY / "predecessor_antigravity.py").read_text(encoding="utf-8")
        tree = ast.parse(predecessor)
        authority = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "_Exp449StoppedAuthorityAntigravity"
        )
        fields = {
            node.target.id
            for node in authority.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        self.assertFalse({"qualified_count", "candidate_metrics", "family_metrics"} & fields)

        readiness = (_EXP452_ANTIGRAVITY / "readiness_antigravity.py").read_text(encoding="utf-8")
        imported = {
            alias.name
            for node in ast.walk(ast.parse(readiness))
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(ast.parse(readiness))
            if isinstance(node, ast.ImportFrom)
        }
        forbidden = {
            "sqlite3", "numpy", "torch", "mlx", "requests", "urllib", "httpx",
            "sources_antigravity", "evaluation_antigravity", "replay_antigravity",
            "dashboard_export_antigravity",
        }
        self.assertFalse(any(name == blocked or name.startswith(f"{blocked}.") for name in imported for blocked in forbidden))

    def test_task4_allows_only_sorted_evidence_and_ordered_replay_delta_antigravity(self) -> None:
        """Break caught: evaluator preserves unordered evidence or replay normalizes its comparison."""
        evaluation = ast.parse((_EXP452_ANTIGRAVITY / "evaluation_antigravity.py").read_text(encoding="utf-8"))
        sorted_calls = [
            node for node in ast.walk(evaluation)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sorted"
            and len(node.args) == 1
        ]
        self.assertEqual(len(sorted_calls), 1)
        argument = sorted_calls[0].args[0]
        self.assertIsInstance(argument, ast.Subscript)
        replay = (_EXP452_ANTIGRAVITY / "replay_antigravity.py").read_text(encoding="utf-8")
        self.assertIn('candidate["indices"] != row["indices"]', replay)
        self.assertNotIn('sorted(candidate["indices"])', replay)
        self.assertNotIn('set(candidate["indices"])', replay)

    def test_task4_runtime_same_label_differential_preserves_scientific_payloads_antigravity(self) -> None:
        """Break caught: the ordered-evidence fork changes same-label metrics or aggregate semantics."""
        from experiments.exp449_residual_fn_method_qualification_antigravity.evaluation_antigravity import _LEAVES_ANTIGRAVITY as legacy_leaves, _evaluate_with_dependencies_antigravity as legacy_evaluate
        from experiments.exp449_residual_fn_method_qualification_antigravity.tests.test_evaluation_antigravity import _DependenciesAntigravity
        from experiments.exp452_exp449_index_order_repair_antigravity.evaluation_antigravity import _LEAVES_ANTIGRAVITY as successor_leaves, _evaluate_with_dependencies_antigravity as successor_evaluate
        from experiments.exp452_exp449_index_order_repair_antigravity.tests.test_evaluation_antigravity import EvaluationAntigravityTests
        with tempfile.TemporaryDirectory(prefix="exp452_runtime_differential_antigravity_") as temporary:
            legacy_root = Path(temporary) / "legacy_antigravity"; successor_root = Path(temporary) / "successor_antigravity"
            legacy = _DependenciesAntigravity(legacy_root)
            rows = list(legacy.frozen_predictions_antigravity())
            ordered = [0, 1]
            for row in rows: row["indices"] = ordered
            legacy.frozen_predictions_antigravity = lambda: tuple(rows)
            legacy_result = legacy_evaluate("a" * 40, legacy)
            successor = EvaluationAntigravityTests()._dependencies_antigravity(successor_root)
            successor_result = successor_evaluate("a" * 40, successor)
            self.assertEqual(legacy_leaves, successor_leaves)
            legacy_payloads = {name: (legacy_root / name).read_bytes() for name in legacy_leaves}
            successor_payloads = {name: (successor_root / name).read_bytes() for name in successor_leaves}
            _assert_task4_leaf_differential_antigravity(legacy_payloads, successor_payloads)

            mutations = (
                ("compatibility", "compatibility_antigravity.json", lambda value: value.__setitem__("unallowlisted", True)),
                ("frozen-row", "frozen_predictions_antigravity.jsonl", lambda value: value[0].__setitem__("indices", [1, 0])),
                ("manifest", "prediction_manifest_antigravity.json", lambda value: value.__setitem__("rows", 97)),
                ("evaluation", "evaluation_antigravity.json", lambda value: value.__setitem__("candidate_count", 17)),
                ("artifact-map-linkage", "artifact_sha256_antigravity.json", lambda value: value.__setitem__("context_antigravity.json", "0" * 64)),
            )
            for family, name, mutate in mutations:
                changed = dict(successor_payloads)
                value = _parse_task4_leaf_antigravity(name, changed[name])
                mutate(value)
                changed[name] = _canonical_task4_leaf_antigravity(name, value)
                with self.subTest(family=family), self.assertRaises(AssertionError):
                    _assert_task4_leaf_differential_antigravity(legacy_payloads, changed)


if __name__ == "__main__":
    unittest.main()
