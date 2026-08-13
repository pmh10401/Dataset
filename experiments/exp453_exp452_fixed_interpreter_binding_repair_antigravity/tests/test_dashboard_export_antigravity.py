from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
import unittest

from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.config_antigravity import CANDIDATE_MATRIX_ANTIGRAVITY
from experiments.exp453_exp452_fixed_interpreter_binding_repair_antigravity.dashboard_export_antigravity import public_result_antigravity


class _HybridStringMappingAntigravity(str, Mapping[str, object]):
    """Expose no mapping items while retaining a forged string identity."""

    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def __eq__(self, other: object) -> bool:
        return True

    def __ne__(self, other: object) -> bool:
        return False


class _DeceptiveDictAntigravity(dict[str, object]):
    """Hide sensitive keys from iteration while retaining direct lookup."""

    def __iter__(self) -> Iterator[str]:
        return (key for key in dict.__iter__(self) if key != "raw_labels")

    def items(self) -> Iterator[tuple[str, object]]:  # type: ignore[override]
        return ((key, dict.__getitem__(self, key)) for key in self)


class _ListSubclassAntigravity(list[object]):
    pass


class _TupleSubclassAntigravity(tuple[object, ...]):
    pass


def _valid_result_antigravity() -> dict[str, object]:
    global_f1 = 0.62664286873276
    candidates = [
        {
            "family": family,
            "method": method,
            "target_fn_delta": -1,
            "gates": {"global_fn": True, "global_fp": True, "global_f1": True, "target_fn_delta": True},
            "qualified": True,
            "global": {"tp": 1931, "fp": 314, "fn": 1987, "tn": 191668, "f1": global_f1},
        }
        for family, methods in CANDIDATE_MATRIX_ANTIGRAVITY
        for method in methods
    ]
    return {
        "schema": "exp453.public-method-qualification.v1",
        "experiment": "Exp453",
        "baseline": {"tp": 1930, "fp": 314, "fn": 1988, "tn": 191668, "f1": 0.6264199935086011},
        "candidates": candidates,
        "families": [family for family, _methods in CANDIDATE_MATRIX_ANTIGRAVITY],
        "status": "synthetic_complete",
        "repeatable": True,
        "qualified_candidate_count": 16,
        "routing_research_eligible": True,
        "runtime": {"label_sessions": 1, "label_queries": 48, "label_rows": 72300},
    }


class DashboardExportAntigravityTest(unittest.TestCase):
    def test_public_result_returns_only_the_complete_aggregate_allowlist_antigravity(self) -> None:
        """Break caught: dropping an aggregate or adding an operational/detail field."""
        payload = _valid_result_antigravity()
        self.assertEqual(public_result_antigravity(payload), payload)

    def test_public_result_rejects_forbidden_material_at_any_depth_antigravity(self) -> None:
        """Break caught: leaking labels, indices, dataset rows, paths, hashes, or logs."""
        mutations: tuple[tuple[str, object], ...] = (
            ("raw labels", {"raw_labels": [0, 1]}),
            ("indices", {"indices": [1, 3, 5]}),
            ("dataset name", {"dataset_name": "Crop_normal_1"}),
            ("per-dataset rows", {"per_dataset_rows": [{"dataset": "Crop_normal_1"}]}),
            ("local path", {"local_path": "/Users/mac/Documents/Dataset/private"}),
            ("hash", {"sha256": "a" * 64}),
            ("logs", {"logs": ["private"]}),
        )
        for label, nested in mutations:
            payload = _valid_result_antigravity()
            payload["runtime"] = {"label_sessions": 1, "label_queries": 48, "label_rows": 72300, "nested": {"deeper": nested}}
            with self.subTest(label=label), self.assertRaises(ValueError):
                public_result_antigravity(payload)

    def test_public_result_rejects_forbidden_nested_string_values_antigravity(self) -> None:
        """Break caught: hiding a path, digest, or raw dataset identity under an allowed-looking key."""
        for value in (
            "/Users/mac/Documents/Dataset/private",
            "a" * 64,
            "Crop_normal_1",
        ):
            payload = _valid_result_antigravity()
            payload["status"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                public_result_antigravity(payload)

    def test_public_result_rejects_string_subclasses_in_identity_fields_antigravity(self) -> None:
        """Break caught: a non-exact identity string bypassing checks through inherited or forged equality."""
        class StringSubclassAntigravity(str):
            pass

        class EqualityOverridingStringSubclassAntigravity(str):
            def __eq__(self, other: object) -> bool:
                return True

            def __ne__(self, other: object) -> bool:
                return False

        cases = (
            ("schema", StringSubclassAntigravity("exp453.public-method-qualification.v1")),
            ("experiment", StringSubclassAntigravity("Exp453")),
            ("schema", EqualityOverridingStringSubclassAntigravity("/Users/mac/Documents/Dataset/private")),
            ("experiment", EqualityOverridingStringSubclassAntigravity("Crop_normal_1")),
        )
        for field, value in cases:
            payload = _valid_result_antigravity()
            payload[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                public_result_antigravity(payload)

    def test_public_result_rejects_nested_string_subclasses_even_when_equality_lies_antigravity(self) -> None:
        """Break caught: a nested path or dataset identity comparing equal to an allowlisted family/method."""
        class StringSubclassAntigravity(str):
            pass

        class EqualityOverridingStringSubclassAntigravity(str):
            def __eq__(self, other: object) -> bool:
                return True

            def __ne__(self, other: object) -> bool:
                return False

        mutations = (
            ("family list subclass", lambda payload: payload["families"].__setitem__(0, StringSubclassAntigravity("Crop"))),  # type: ignore[union-attr]
            ("candidate family subclass", lambda payload: payload["candidates"][0].__setitem__("family", StringSubclassAntigravity("Crop"))),  # type: ignore[index]
            ("private path equality subclass", lambda payload: payload["candidates"][0].__setitem__("family", EqualityOverridingStringSubclassAntigravity("/Users/mac/Documents/Dataset/private"))),  # type: ignore[index]
            ("dataset equality subclass", lambda payload: payload["candidates"][0].__setitem__("method", EqualityOverridingStringSubclassAntigravity("Crop_normal_1"))),  # type: ignore[index]
        )
        for label, mutate in mutations:
            payload = _valid_result_antigravity()
            mutate(payload)
            with self.subTest(label=label), self.assertRaises(ValueError):
                public_result_antigravity(payload)

    def test_public_result_rejects_hybrid_string_mapping_identity_values_antigravity(self) -> None:
        """Break caught: Mapping dispatch hiding a forged private string identity behind empty items."""
        cases = (
            ("schema", _HybridStringMappingAntigravity("/Users/mac/Documents/Dataset/private")),
            ("experiment", _HybridStringMappingAntigravity("Crop_normal_1")),
        )
        for field, value in cases:
            payload = _valid_result_antigravity()
            payload[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                public_result_antigravity(payload)

    def test_public_result_rejects_hybrid_string_mapping_nested_values_antigravity(self) -> None:
        """Break caught: empty Mapping items hiding path/dataset strings in nested family-method fields."""
        cases = (
            ("family", _HybridStringMappingAntigravity("/Users/mac/Documents/Dataset/private")),
            ("method", _HybridStringMappingAntigravity("Crop_normal_1")),
        )
        for field, value in cases:
            payload = _valid_result_antigravity()
            payload["candidates"][0][field] = value  # type: ignore[index]
            with self.subTest(field=field), self.assertRaises(ValueError):
                public_result_antigravity(payload)

    def test_public_result_rejects_deceptive_dict_hiding_raw_labels_antigravity(self) -> None:
        """Break caught: validating a filtered dict view then returning the original raw-label carrier."""
        payload = _valid_result_antigravity()
        candidate = _DeceptiveDictAntigravity(payload["candidates"][0])  # type: ignore[index]
        candidate["raw_labels"] = [0, 1]
        payload["candidates"][0] = candidate  # type: ignore[index]
        with self.assertRaises(ValueError):
            public_result_antigravity(payload)

    def test_public_result_requires_exact_builtin_schema_containers_antigravity(self) -> None:
        """Break caught: accepting Mapping/list/tuple subclasses or unsupported tuple containers."""
        cases = (
            ("top-level dict subclass", lambda payload: _DeceptiveDictAntigravity(payload)),
            ("baseline dict subclass", lambda payload: {**payload, "baseline": _DeceptiveDictAntigravity(payload["baseline"])}),
            ("candidate list subclass", lambda payload: {**payload, "candidates": _ListSubclassAntigravity(payload["candidates"])}),
            ("family list subclass", lambda payload: {**payload, "families": _ListSubclassAntigravity(payload["families"])}),
            ("family tuple", lambda payload: {**payload, "families": tuple(payload["families"])}),
            ("family tuple subclass", lambda payload: {**payload, "families": _TupleSubclassAntigravity(payload["families"])}),
        )
        for label, build in cases:
            payload = build(_valid_result_antigravity())
            with self.subTest(label=label), self.assertRaises(ValueError):
                public_result_antigravity(payload)

    def test_public_result_materializes_a_deep_builtin_projection_antigravity(self) -> None:
        """Break caught: returning validated caller-owned nested containers that can later reveal or mutate data."""
        payload = _valid_result_antigravity()
        projected = public_result_antigravity(payload)
        self.assertIs(type(projected), dict)
        self.assertIs(type(projected["baseline"]), dict)
        self.assertIs(type(projected["candidates"]), list)
        self.assertIs(type(projected["candidates"][0]), dict)  # type: ignore[index]
        self.assertIs(type(projected["candidates"][0]["gates"]), dict)  # type: ignore[index]
        self.assertIs(type(projected["candidates"][0]["global"]), dict)  # type: ignore[index]
        self.assertIs(type(projected["families"]), list)
        self.assertIs(type(projected["runtime"]), dict)
        self.assertIsNot(projected["baseline"], payload["baseline"])
        self.assertIsNot(projected["candidates"], payload["candidates"])
        self.assertIsNot(projected["candidates"][0], payload["candidates"][0])  # type: ignore[index]
        self.assertIsNot(projected["candidates"][0]["gates"], payload["candidates"][0]["gates"])  # type: ignore[index]
        self.assertIsNot(projected["candidates"][0]["global"], payload["candidates"][0]["global"])  # type: ignore[index]
        self.assertIsNot(projected["families"], payload["families"])
        self.assertIsNot(projected["runtime"], payload["runtime"])
        original_tp = projected["baseline"]["tp"]  # type: ignore[index]
        payload["baseline"]["tp"] = 0  # type: ignore[index]
        payload["candidates"][0]["family"] = "mutated"  # type: ignore[index]
        self.assertEqual(projected["baseline"]["tp"], original_tp)  # type: ignore[index]
        self.assertEqual(projected["candidates"][0]["family"], "Crop")  # type: ignore[index]

    def test_public_result_recomputes_counts_f1_gates_and_summary_antigravity(self) -> None:
        """Break caught: accepting self-reported aggregate math without recomputation."""
        mutations = []
        wrong_count = _valid_result_antigravity()
        wrong_count["baseline"] = {"tp": True, "fp": 314, "fn": 1988, "tn": 191668, "f1": 0.6264199935086011}
        mutations.append(wrong_count)
        wrong_f1 = _valid_result_antigravity()
        wrong_f1["baseline"] = {**wrong_f1["baseline"], "f1": 0.5}  # type: ignore[arg-type]
        mutations.append(wrong_f1)
        wrong_gate = _valid_result_antigravity()
        wrong_gate["candidates"] = deepcopy(wrong_gate["candidates"])
        wrong_gate["candidates"][0]["gates"]["global_fn"] = False  # type: ignore[index]
        mutations.append(wrong_gate)
        wrong_summary = _valid_result_antigravity()
        wrong_summary["qualified_candidate_count"] = 15
        mutations.append(wrong_summary)
        for payload in mutations:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                public_result_antigravity(payload)


if __name__ == "__main__":
    unittest.main()
